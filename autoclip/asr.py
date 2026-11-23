"""本地语音识别模块：基于 faster-whisper，针对 RTX 3060 12G 自动启用 CUDA float16 加速。

v1.0.2 关键改进：
- 关闭易误吞美剧笑声/音效对白的 Silero VAD，改用 Whisper 内置静音检测，彻底解决漏句；
- 基于词级时间戳（word_timestamps）按真实句末标点与语义停顿精准拆句，避免多句粘连导致字幕慢半拍；
- 自动防止纯语气词（uh/um/oh 等）被单独切分成极短碎片。
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .config import AppConfig, get_app_dir
from .hardware import HardwareProfile

_FILLER_WORDS = {
    "uh", "uhh", "uhhh", "um", "umm", "er", "ah", "ahh", "oh", "ohh",
    "hm", "hmm", "well", "so", "hey", "yeah", "yes", "no", "wait", "okay", "ok",
}


@dataclass
class WordStamp:
    start: float
    end: float
    word: str


@dataclass
class RawSegment:
    index: int
    start: float
    end: float
    text: str
    words: list[WordStamp] = field(default_factory=list)


def _is_filler_only(words: list[WordStamp]) -> bool:
    if len(words) > 2:
        return False
    cleaned = [re.sub(r"[^a-zA-Z]", "", w.word).lower() for w in words]
    cleaned = [c for c in cleaned if c]
    return bool(cleaned) and all(c in _FILLER_WORDS for c in cleaned)


def _split_words_into_sentences(words: list[WordStamp]) -> list[tuple[float, float, str, list[WordStamp]]]:
    """按词级时间戳把 Whisper 长片段精准拆分为单句，确保每句起止时间与说话瞬间严丝合缝。"""
    if not words:
        return []

    chunks: list[list[WordStamp]] = []
    cur: list[WordStamp] = []

    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (nxt.start - w.end) if nxt is not None else 0.0
        ends_sentence = bool(re.search(r"[.?!…]+[\"'\)\]]*$", w.word))

        # 若当前只积累了 1-2 个纯语气词（如 "Uhh," / "Oh!"）且紧接着有下一词，不在此断开
        if _is_filler_only(cur) and nxt is not None and gap < 0.8:
            continue

        cur_dur = w.end - cur[0].start
        should_split = False
        if ends_sentence and len(cur) >= 2:
            should_split = True
        elif gap >= 0.45 and len(cur) >= 2:
            should_split = True
        elif cur_dur >= 4.2 and bool(re.search(r"[,;:]+$", w.word)):
            should_split = True

        if should_split and nxt is not None:
            chunks.append(cur)
            cur = []

    if cur:
        # 如果尾部剩下极短纯语气词且紧贴前一句，并入前一句
        if chunks and _is_filler_only(cur) and (cur[0].start - chunks[-1][-1].end) < 0.45:
            chunks[-1].extend(cur)
        else:
            chunks.append(cur)

    result: list[tuple[float, float, str, list[WordStamp]]] = []
    for ch in chunks:
        txt = " ".join(w.word for w in ch)
        txt = re.sub(r"\s+([,.?!;:])", r"\1", txt).strip()
        if txt:
            result.append((ch[0].start, ch[-1].end, txt, ch))
    return result


class WhisperEngine:
    """线程安全的 Whisper 模型复用管理器（批量剪辑 10-20 个视频时只加载一次显存）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._loaded_key: Optional[tuple[str, str, str]] = None

    def _ensure_hf_mirror(self) -> None:
        if not os.environ.get("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    def _get_model(
        self,
        model_name: str,
        device: str,
        compute_type: str,
        log_cb: Optional[Callable[[str], None]] = None,
    ):
        key = (model_name, device, compute_type)
        if self._model is not None and self._loaded_key == key:
            return self._model

        self._ensure_hf_mirror()
        from faster_whisper import WhisperModel

        models_dir = get_app_dir() / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        if log_cb:
            log_cb(f"正在加载语音识别模型 {model_name}（设备：{device} / {compute_type}）...")

        try:
            self._model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
                download_root=str(models_dir),
            )
            self._loaded_key = key
        except Exception as e:
            if device == "cuda":
                if log_cb:
                    log_cb(f"CUDA 加载失败（{e}），自动切换至 CPU int8 模式...")
                self._model = WhisperModel(
                    model_name,
                    device="cpu",
                    compute_type="int8",
                    download_root=str(models_dir),
                )
                self._loaded_key = (model_name, "cpu", "int8")
            else:
                raise
        return self._model

    def transcribe(
        self,
        video_path: Path,
        cfg: AppConfig,
        hw: HardwareProfile,
        log_cb: Optional[Callable[[str], None]] = None,
        progress_cb: Optional[Callable[[float], None]] = None,
    ) -> list[RawSegment]:
        with self._lock:
            model = self._get_model(
                cfg.whisper_model,
                hw.recommended_whisper_device,
                hw.recommended_whisper_compute,
                log_cb=log_cb,
            )
            if log_cb:
                log_cb(f"正在高精度识别英文语音（防漏句模式）：{video_path.name}")

            lang = cfg.source_language.strip() or None
            # 关闭 Silero VAD（防止吞掉美剧笑声/音效重叠的对白），改用 Whisper 内置静音门限
            segments_iter, info = model.transcribe(
                str(video_path),
                language=lang,
                beam_size=5,
                vad_filter=False,
                word_timestamps=True,
                condition_on_previous_text=False,
                no_speech_threshold=0.65,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4,
            )

            total_dur = max(float(getattr(info, "duration", 0.0) or 0.0), 1.0)
            results: list[RawSegment] = []
            next_idx = 1

            for seg in segments_iter:
                text = (seg.text or "").strip()
                if not text:
                    continue
                words = []
                if seg.words:
                    for w in seg.words:
                        wt = (w.word or "").strip()
                        if wt:
                            words.append(WordStamp(start=float(w.start), end=float(w.end), word=wt))

                if words:
                    sub_sentences = _split_words_into_sentences(words)
                    for st, ed, sub_txt, sub_w in sub_sentences:
                        results.append(
                            RawSegment(index=next_idx, start=st, end=ed, text=sub_txt, words=sub_w)
                        )
                        next_idx += 1
                else:
                    results.append(
                        RawSegment(
                            index=next_idx,
                            start=float(seg.start),
                            end=float(seg.end),
                            text=text,
                            words=[],
                        )
                    )
                    next_idx += 1

                if progress_cb:
                    progress_cb(min(1.0, float(seg.end) / total_dur))

            if log_cb:
                log_cb(f"语音识别完成，共提取 {len(results)} 句精准对齐的原始对白。")
            return results


_GLOBAL_ENGINE = WhisperEngine()


def get_whisper_engine() -> WhisperEngine:
    return _GLOBAL_ENGINE

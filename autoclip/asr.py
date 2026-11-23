"""本地语音识别模块：基于 faster-whisper，针对 RTX 3060 12G 自动启用 CUDA float16 加速。"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .config import AppConfig, get_app_dir
from .hardware import HardwareProfile


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


class WhisperEngine:
    """线程安全的 Whisper 模型复用管理器（批量剪辑 10-20 个视频时只加载一次显存）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._loaded_key: Optional[tuple[str, str, str]] = None

    def _ensure_hf_mirror(self) -> None:
        # 国内网络自动配置 HuggingFace 镜像，避免 Windows 上首次下载模型失败
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
                log_cb(f"正在识别英文语音：{video_path.name}")

            lang = cfg.source_language.strip() or None
            segments_iter, info = model.transcribe(
                str(video_path),
                language=lang,
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
                word_timestamps=True,
                condition_on_previous_text=True,
            )

            total_dur = max(float(getattr(info, "duration", 0.0) or 0.0), 1.0)
            results: list[RawSegment] = []
            for idx, seg in enumerate(segments_iter, start=1):
                text = (seg.text or "").strip()
                if not text:
                    continue
                words = []
                if seg.words:
                    for w in seg.words:
                        wt = (w.word or "").strip()
                        if wt:
                            words.append(WordStamp(start=float(w.start), end=float(w.end), word=wt))
                results.append(
                    RawSegment(
                        index=idx,
                        start=float(seg.start),
                        end=float(seg.end),
                        text=text,
                        words=words,
                    )
                )
                if progress_cb:
                    progress_cb(min(1.0, float(seg.end) / total_dur))

            if log_cb:
                log_cb(f"语音识别完成，共提取 {len(results)} 段原始语音。")
            return results


_GLOBAL_ENGINE = WhisperEngine()


def get_whisper_engine() -> WhisperEngine:
    return _GLOBAL_ENGINE

"""批量流水线调度模块：支持 10-20+ 视频批量导入，按硬件配置自动并行或顺序剪辑。"""

from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

from .asr import get_whisper_engine
from .config import AppConfig, assign_accounts_to_videos
from .hardware import HardwareProfile
from .ocr_sub import extract_onscreen_english_subtitles
from .renderer import render_video
from .subtitle import write_srt_subtitle
from .translator import SubtitleItem, translate_segments


@dataclass
class TaskStatus:
    video_path: Path
    state: str
    progress: float = 0.0
    message: str = ""
    output_path: Optional[Path] = None
    account_name: str = ""


def _safe_folder_name(name: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", name).strip()
    return cleaned or "默认账号"


def resolve_output_path(video_path: Path, cfg: AppConfig, account_name: str = "") -> Path:
    ext = "mov" if cfg.output_format.lower() == "mov" else "mp4"
    if cfg.output_dir.strip():
        base_dir = Path(cfg.output_dir.strip())
    else:
        base_dir = video_path.parent / "已剪辑输出"

    acc = account_name.strip()
    if acc and cfg.group_output_by_account:
        safe_acc = _safe_folder_name(acc)
        out_dir = base_dir / safe_acc
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"{safe_acc}_{video_path.stem}.{ext}"

    base_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{_safe_folder_name(acc)}_" if acc else ""
    return base_dir / f"{prefix}{video_path.stem}_已剪辑.{ext}"


class BatchPipeline:
    """自适应并发的批量剪辑引擎。

    当机器性能充足（如 RTX 3060 12G）：
    - 使用分级并发：ASR 阶段共享 GPU 显存锁（不爆显存），同时 DeepSeek 翻译可多路并发，
      FFmpeg NVENC 压制可按显卡能力 2 路并行，整体吞吐最大化；
    当性能不足或选择顺序模式时：
    - 自动退回严格单路顺序执行，确保不卡顿、不崩溃。
    """

    def __init__(
        self,
        cfg: AppConfig,
        hw: HardwareProfile,
        status_cb: Optional[Callable[[TaskStatus], None]] = None,
        log_cb: Optional[Callable[[str], None]] = None,
        review_cb: Optional[Callable[[Path, list[SubtitleItem]], Optional[list[SubtitleItem]]]] = None,
    ) -> None:
        self.cfg = cfg
        self.hw = hw
        self.status_cb = status_cb
        self.log_cb = log_cb
        self.review_cb = review_cb
        self._cancel_event = threading.Event()
        self._render_sem: Optional[threading.Semaphore] = None
        self._translate_sem: Optional[threading.Semaphore] = None

    def cancel(self) -> None:
        self._cancel_event.set()

    def _emit(
        self,
        video_path: Path,
        state: str,
        progress: float,
        message: str = "",
        output_path: Optional[Path] = None,
        account_name: str = "",
    ) -> None:
        if self.status_cb:
            self.status_cb(
                TaskStatus(
                    video_path=video_path,
                    state=state,
                    progress=progress,
                    message=message,
                    output_path=output_path,
                    account_name=account_name,
                )
            )

    def _log(self, msg: str) -> None:
        if self.log_cb:
            self.log_cb(msg)

    def _determine_concurrency(self) -> tuple[int, int, int]:
        """返回 (总任务并发数, 压制并发数, 翻译并发数)。"""
        mode = (self.cfg.concurrency_mode or "auto").strip().lower()
        if self.cfg.review_subtitles:
            # 人工校对模式下采用单任务顺序，避免同时弹出多个校对框
            return 1, 1, 1
        if mode == "auto":
            render_w = self.hw.recommended_render_workers
            trans_w = self.hw.recommended_translate_workers
            total_w = max(render_w + 1, trans_w) if render_w > 1 else 1
            return total_w, render_w, trans_w
        try:
            n = max(1, min(4, int(mode)))
        except ValueError:
            n = 1
        return n, n, max(n, 2)

    def process_single(self, video_path: Path, account_name: str = "") -> TaskStatus:
        name = video_path.name
        t0 = time.time()
        if self._cancel_event.is_set():
            st = TaskStatus(video_path, "已取消", 0.0, "任务已取消", account_name=account_name)
            self._emit(video_path, st.state, st.progress, st.message, account_name=account_name)
            return st

        try:
            out_path = resolve_output_path(video_path, self.cfg, account_name=account_name)
            if self.cfg.skip_existing and out_path.is_file() and out_path.stat().st_size > 100_000:
                self._log(f"[{name}] 已存在成品，自动跳过：{out_path.name}")
                st = TaskStatus(video_path, "已跳过", 1.0, f"已有成品：{out_path.name}", out_path, account_name)
                self._emit(video_path, st.state, st.progress, st.message, st.output_path, account_name)
                return st

            # 阶段 1：优先扫描画面英文字幕（100%对齐英文字幕出现与消失时长），若无硬字幕则自动回退 Whisper 语音识别
            raw_segments = []
            used_ocr = False
            if (self.cfg.subtitle_source or "ocr_first") == "ocr_first":
                self._emit(video_path, "识别字幕", 0.02, "正在扫描画面英文字幕...", account_name=account_name)
                try:
                    raw_segments = extract_onscreen_english_subtitles(
                        video_path,
                        log_cb=lambda m: self._log(f"[{name}] {m}"),
                        progress_cb=lambda p: self._emit(
                            video_path, "识别字幕", 0.02 + p * 0.33, "正在对齐画面英文字幕...", account_name=account_name
                        ),
                    )
                    if len(raw_segments) >= 3:
                        used_ocr = True
                    else:
                        self._log(f"[{name}] 画面未检测到足够英文字幕，自动切换至语音识别...")
                        raw_segments = []
                except Exception as ocr_err:
                    self._log(f"[{name}] 画面字幕识别跳过（{ocr_err}），自动切换至语音识别...")
                    raw_segments = []

            if not used_ocr:
                self._emit(video_path, "语音识别", 0.02, "准备识别英文语音...", account_name=account_name)
                engine = get_whisper_engine()
                raw_segments = engine.transcribe(
                    video_path,
                    self.cfg,
                    self.hw,
                    log_cb=lambda m: self._log(f"[{name}] {m}"),
                    progress_cb=lambda p: self._emit(
                        video_path, "语音识别", 0.02 + p * 0.33, "正在识别语音...", account_name=account_name
                    ),
                )
            if self._cancel_event.is_set():
                raise InterruptedError("用户取消任务")

            if not raw_segments:
                raise RuntimeError("未识别到有效英文对白或字幕，请检查原视频。")

            # 阶段 2：DeepSeek 翻译与语义断句（35% -> 60%）
            self._emit(video_path, "翻译中", 0.36, "正在翻译成通俗中文...", account_name=account_name)
            assert self._translate_sem is not None
            with self._translate_sem:
                subtitles = translate_segments(
                    raw_segments,
                    self.cfg,
                    log_cb=lambda m: self._log(f"[{name}] {m}"),
                    progress_cb=lambda p: self._emit(
                        video_path, "翻译中", 0.36 + p * 0.24, "DeepSeek 翻译中...", account_name=account_name
                    ),
                    strict_visual_sync=used_ocr,
                )
            if self._cancel_event.is_set():
                raise InterruptedError("用户取消任务")

            if self.cfg.review_subtitles and self.review_cb is not None:
                self._emit(video_path, "等待校对", 0.60, "请在弹窗中确认或微调字幕...", account_name=account_name)
                edited = self.review_cb(video_path, subtitles)
                if edited is None:
                    raise InterruptedError("校对已取消")
                subtitles = edited

            if self.cfg.keep_srt:
                srt_path = out_path.with_suffix(".srt")
                write_srt_subtitle(subtitles, srt_path)

            # 阶段 3：无痕蒙版去重 + 字幕与飘动水印压制（60% -> 100%）
            wm_tip = f"（水印：{account_name}）" if account_name else ""
            self._emit(video_path, "导出成片", 0.62, f"等待显卡导出{wm_tip}...", account_name=account_name)
            assert self._render_sem is not None
            with self._render_sem:
                if self._cancel_event.is_set():
                    raise InterruptedError("用户取消任务")
                render_video(
                    video_path,
                    out_path,
                    subtitles,
                    self.cfg,
                    self.hw,
                    log_cb=lambda m: self._log(f"[{name}] {m}"),
                    progress_cb=lambda p: self._emit(
                        video_path, "导出成片", 0.62 + p * 0.38, f"正在压制成片{wm_tip}...", account_name=account_name
                    ),
                    watermark_text=account_name,
                )

            elapsed = int(round(time.time() - t0))
            st = TaskStatus(
                video_path,
                "已完成",
                1.0,
                f"耗时 {elapsed}秒 | {out_path.name}",
                out_path,
                account_name,
            )
            self._emit(video_path, st.state, st.progress, st.message, st.output_path, account_name)
            return st

        except InterruptedError:
            st = TaskStatus(video_path, "已取消", 0.0, "已停止", account_name=account_name)
            self._emit(video_path, st.state, st.progress, st.message, account_name=account_name)
            return st
        except Exception as e:
            self._log(f"[{name}] 处理失败：{e}")
            st = TaskStatus(video_path, "失败", 0.0, str(e), account_name=account_name)
            self._emit(video_path, st.state, st.progress, st.message, account_name=account_name)
            return st

    def run_batch(
        self, video_items: Union[list[Path], list[tuple[Path, str]]]
    ) -> list[TaskStatus]:
        total_w, render_w, trans_w = self._determine_concurrency()
        self._render_sem = threading.Semaphore(render_w)
        self._translate_sem = threading.Semaphore(trans_w)

        pairs: list[tuple[Path, str]] = []
        if video_items and isinstance(video_items[0], tuple):
            pairs = list(video_items)  # type: ignore[arg-type]
        else:
            vps: list[Path] = list(video_items)  # type: ignore[arg-type]
            assigned = assign_accounts_to_videos(
                len(vps), self.cfg.watermark_accounts, enabled=self.cfg.watermark_enabled
            )
            pairs = list(zip(vps, assigned))

        mode_desc = (
            f"显卡并行加速（总流水线 {total_w} 路，同时压制 {render_w} 路）"
            if total_w > 1
            else "单路顺序剪辑"
        )
        self._log(f"开始批量剪辑，共 {len(pairs)} 条视频 | 模式：{mode_desc}")

        for vp, acc in pairs:
            self._emit(vp, "等待中", 0.0, "排队中...", account_name=acc)

        results: list[TaskStatus] = []
        if total_w <= 1 or len(pairs) == 1:
            for vp, acc in pairs:
                if self._cancel_event.is_set():
                    break
                results.append(self.process_single(vp, account_name=acc))
            return results

        with ThreadPoolExecutor(max_workers=total_w) as pool:
            fut_map = {pool.submit(self.process_single, vp, acc): vp for vp, acc in pairs}
            for fut in as_completed(fut_map):
                results.append(fut.result())
        return results

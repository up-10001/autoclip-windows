"""AutoClip 智剪 Pro v3.0 - 正式产品级图形界面。

设计理念：
- 极简三步主流程：导入素材 → 指定账号水印（支持一键按量分配 + 手动多选/双击单条指定）→ 选择导出目录并一键开剪；
- 主界面只保留高频操作，密钥与低频技术参数收纳进干净的弹窗，全部自动保存记忆；
- 支持全局总进度条、仅重剪失败项、可折叠日志、实时预览（含对应账号水印与安全区红线）。
"""

from __future__ import annotations

import os
import platform
import queue
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageTk

from . import __version__
from .config import AppConfig, assign_accounts_to_videos, load_config, save_config
from .hardware import HardwareProfile, detect_hardware
from .pipeline import BatchPipeline, TaskStatus
from .renderer import render_preview_frame
from .translator import SubtitleItem, test_deepseek_connection

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".flv", ".wmv", ".webm"}
NO_WM_LABEL = "（不加水印）"

ENHANCE_OPTIONS = [
    ("极速高清增强（推荐）", "fast"),
    ("真·AI超清（较慢）", "ai"),
    ("关闭", "off"),
]
SPEED_OPTIONS = [
    ("智能自动（3060 并行 4 路）", "auto"),
    ("一条一条顺序剪（低占用）", "1"),
    ("同时剪 2 条", "2"),
    ("同时剪 3 条", "3"),
    ("同时剪 4 条（满速）", "4"),
]
QUALITY_OPTIONS = [
    ("高清推荐（画质清晰、体积适中）", 10),
    ("超清高画质（体积稍大）", 14),
    ("省空间标准（上传最快）", 6),
]
FPS_OPTIONS = [
    ("保持原片 30 帧（推荐，省一半时间）", 30),
    ("60 帧（高帧率）", 60),
]
SUB_SOURCE_OPTIONS = [
    ("自动对齐画面英文字幕（推荐，100%同步）", "ocr_first"),
    ("仅听声音识别（无英文字幕的视频用）", "whisper"),
]

# 现代浅色产品配色
BG_APP = "#F3F5F9"
BG_CARD = "#FFFFFF"
BG_HEADER = "#1E293B"
FG_HEADER = "#FFFFFF"
ACCENT_BLUE = "#2563EB"
ACCENT_BLUE_HOVER = "#1D4ED8"
ACCENT_RED = "#DC2626"
ACCENT_GREEN = "#16A34A"
TEXT_MAIN = "#0F172A"
TEXT_MUTED = "#64748B"
BORDER_COLOR = "#E2E8F0"


def _code_to_label(options: list[tuple[str, Any]], code: Any) -> str:
    for label, c in options:
        if c == code:
            return label
    return options[0][0]


def _label_to_code(options: list[tuple[str, Any]], label: str) -> Any:
    for l, c in options:
        if l == label:
            return c
    return options[0][1]


class ApiKeyDialog(tk.Toplevel):
    """独立的 DeepSeek 翻译密钥配置弹窗。"""

    def __init__(self, parent: "AutoClipApp") -> None:
        super().__init__(parent)
        self.parent_app = parent
        self.title("DeepSeek 翻译密钥设置")
        self.geometry("540x250")
        self.resizable(False, False)
        self.configure(bg=BG_CARD)
        self.transient(parent)
        self.grab_set()

        self.var_key = tk.StringVar(value=parent.cfg.api_key)
        self.var_url = tk.StringVar(value=parent.cfg.api_base_url)
        self.var_model = tk.StringVar(value=parent.cfg.api_model)
        self.var_show = tk.BooleanVar(value=False)

        body = ttk.Frame(self, padding=18, style="Card.TFrame")
        body.pack(fill=tk.BOTH, expand=True)

        ttk.Label(body, text="DeepSeek API 密钥:", style="CardBold.TLabel").grid(row=0, column=0, sticky=tk.W, pady=8)
        self.entry_k = ttk.Entry(body, textvariable=self.var_key, width=36, show="*")
        self.entry_k.grid(row=0, column=1, sticky=tk.EW, padx=(8, 4), pady=8)
        ttk.Checkbutton(body, text="显示", variable=self.var_show, command=self._toggle, style="Card.TCheckbutton").grid(
            row=0, column=2, pady=8
        )

        ttk.Label(body, text="接口地址（默认即可）:", style="Card.TLabel").grid(row=1, column=0, sticky=tk.W, pady=6)
        ttk.Entry(body, textvariable=self.var_url).grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=(8, 0), pady=6)

        ttk.Label(body, text="模型名称（默认即可）:", style="Card.TLabel").grid(row=2, column=0, sticky=tk.W, pady=6)
        ttk.Entry(body, textvariable=self.var_model).grid(row=2, column=1, columnspan=2, sticky=tk.EW, padx=(8, 0), pady=6)

        self.lbl_status = ttk.Label(body, text="填写完成后可点“测试连接”确认密钥可用。", foreground=TEXT_MUTED, style="Card.TLabel")
        self.lbl_status.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(8, 4))

        btn_bar = ttk.Frame(body, style="Card.TFrame")
        btn_bar.grid(row=4, column=0, columnspan=3, sticky=tk.EW, pady=(12, 0))
        ttk.Button(btn_bar, text="测试连接", command=self._on_test).pack(side=tk.LEFT)
        ttk.Button(btn_bar, text="保存并关闭", style="Primary.TButton", command=self._on_save).pack(side=tk.RIGHT)
        ttk.Button(btn_bar, text="取消", command=self.destroy).pack(side=tk.RIGHT, padx=(0, 8))
        body.columnconfigure(1, weight=1)

    def _toggle(self) -> None:
        self.entry_k.configure(show="" if self.var_show.get() else "*")

    def _apply_to_cfg(self) -> None:
        self.parent_app.cfg.api_key = self.var_key.get().strip()
        self.parent_app.cfg.api_base_url = self.var_url.get().strip() or "https://api.deepseek.com"
        self.parent_app.cfg.api_model = self.var_model.get().strip() or "deepseek-chat"
        save_config(self.parent_app.cfg)
        self.parent_app._update_api_badge()

    def _on_test(self) -> None:
        self._apply_to_cfg()
        self.lbl_status.configure(text="正在测试连接，请稍候...", foreground=ACCENT_BLUE)
        self.update_idletasks()
        ok, msg = test_deepseek_connection(self.parent_app.cfg)
        self.lbl_status.configure(text=msg, foreground=ACCENT_GREEN if ok else ACCENT_RED)

    def _on_save(self) -> None:
        self._apply_to_cfg()
        self.destroy()


class AdvancedSettingsDialog(tk.Toplevel):
    """收纳低频专业参数的设置弹窗，保持主界面清爽。"""

    def __init__(self, parent: "AutoClipApp") -> None:
        super().__init__(parent)
        self.parent_app = parent
        cfg = parent.cfg
        self.title("高级参数设置")
        self.geometry("560x520")
        self.resizable(False, False)
        self.configure(bg=BG_CARD)
        self.transient(parent)
        self.grab_set()

        self.var_sub_src = tk.StringVar(value=_code_to_label(SUB_SOURCE_OPTIONS, cfg.subtitle_source))
        self.var_speed = tk.StringVar(value=_code_to_label(SPEED_OPTIONS, cfg.concurrency_mode))
        self.var_quality = tk.StringVar(value=_code_to_label(QUALITY_OPTIONS, cfg.video_bitrate_mbps))
        self.var_fps = tk.StringVar(value=_code_to_label(FPS_OPTIONS, cfg.output_fps))
        self.var_dedup = tk.BooleanVar(value=(cfg.dedup_level != "off"))
        self.var_auto_pos = tk.BooleanVar(value=cfg.auto_position_below_en)
        self.var_loudnorm = tk.BooleanVar(value=cfg.audio_loudnorm)
        self.var_skip = tk.BooleanVar(value=cfg.skip_existing)
        self.var_srt = tk.BooleanVar(value=cfg.keep_srt)

        self.var_font = tk.StringVar(value=cfg.font_name)
        self.var_fsize = tk.IntVar(value=cfg.font_size)
        self.var_fcolor = tk.StringVar(value=cfg.font_color)
        self.var_posy = tk.DoubleVar(value=cfg.position_y_pct)
        self.var_maxc = tk.IntVar(value=cfg.max_chars_per_line)
        self.var_wmsize = tk.IntVar(value=cfg.watermark_font_size)

        body = ttk.Frame(self, padding=16, style="Card.TFrame")
        body.pack(fill=tk.BOTH, expand=True)

        r = 0
        ttk.Label(body, text="字幕识别方式:", style="Card.TLabel").grid(row=r, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(body, textvariable=self.var_sub_src, values=[x[0] for x in SUB_SOURCE_OPTIONS], state="readonly", width=34).grid(row=r, column=1, columnspan=3, sticky=tk.EW, pady=5)

        r += 1
        ttk.Label(body, text="批量并行速度:", style="Card.TLabel").grid(row=r, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(body, textvariable=self.var_speed, values=[x[0] for x in SPEED_OPTIONS], state="readonly").grid(row=r, column=1, columnspan=3, sticky=tk.EW, pady=5)

        r += 1
        ttk.Label(body, text="导出清晰度:", style="Card.TLabel").grid(row=r, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(body, textvariable=self.var_quality, values=[x[0] for x in QUALITY_OPTIONS], state="readonly").grid(row=r, column=1, columnspan=3, sticky=tk.EW, pady=5)

        r += 1
        ttk.Label(body, text="导出帧率:", style="Card.TLabel").grid(row=r, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(body, textvariable=self.var_fps, values=[x[0] for x in FPS_OPTIONS], state="readonly").grid(row=r, column=1, columnspan=3, sticky=tk.EW, pady=5)

        r += 1
        ttk.Separator(body).grid(row=r, column=0, columnspan=4, sticky=tk.EW, pady=10)

        r += 1
        ttk.Label(body, text="字幕与水印字体:", style="Card.TLabel").grid(row=r, column=0, sticky=tk.W, pady=5)
        ttk.Entry(body, textvariable=self.var_font, width=18).grid(row=r, column=1, sticky=tk.W, pady=5)
        ttk.Label(body, text="字幕字号:", style="Card.TLabel").grid(row=r, column=2, sticky=tk.W, padx=(10, 4))
        ttk.Spinbox(body, from_=40, to=120, textvariable=self.var_fsize, width=7).grid(row=r, column=3, sticky=tk.W)

        r += 1
        ttk.Label(body, text="字幕颜色:", style="Card.TLabel").grid(row=r, column=0, sticky=tk.W, pady=5)
        cbox = ttk.Frame(body, style="Card.TFrame")
        cbox.grid(row=r, column=1, sticky=tk.W, pady=5)
        ttk.Entry(cbox, textvariable=self.var_fcolor, width=9).pack(side=tk.LEFT)
        ttk.Button(cbox, text="选色", width=5, command=self._pick_color).pack(side=tk.LEFT, padx=4)
        ttk.Label(body, text="水印字号:", style="Card.TLabel").grid(row=r, column=2, sticky=tk.W, padx=(10, 4))
        ttk.Spinbox(body, from_=24, to=72, textvariable=self.var_wmsize, width=7).grid(row=r, column=3, sticky=tk.W)

        r += 1
        ttk.Label(body, text="字幕垂直位置:", style="Card.TLabel").grid(row=r, column=0, sticky=tk.W, pady=5)
        prow = ttk.Frame(body, style="Card.TFrame")
        prow.grid(row=r, column=1, sticky=tk.EW, pady=5)
        ttk.Scale(prow, from_=60.0, to=92.0, variable=self.var_posy, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        lbl_p = ttk.Label(prow, text=f"{self.var_posy.get():.1f}%", width=6, style="Card.TLabel")
        lbl_p.pack(side=tk.LEFT)
        self.var_posy.trace_add("write", lambda *_: lbl_p.config(text=f"{self.var_posy.get():.1f}%"))
        ttk.Label(body, text="单行最多字数:", style="Card.TLabel").grid(row=r, column=2, sticky=tk.W, padx=(10, 4))
        ttk.Spinbox(body, from_=8, to=20, textvariable=self.var_maxc, width=7).grid(row=r, column=3, sticky=tk.W)

        r += 1
        ttk.Separator(body).grid(row=r, column=0, columnspan=4, sticky=tk.EW, pady=10)

        r += 1
        ttk.Checkbutton(body, text="中文字幕自动紧贴原英文字幕正下方（自动预留单/双行间距不重叠）", variable=self.var_auto_pos, style="Card.TCheckbutton").grid(row=r, column=0, columnspan=4, sticky=tk.W, pady=3)
        r += 1
        ttk.Checkbutton(body, text="开启隐形防搬运去重（微缩放 + 微调色 + 清除原属性）", variable=self.var_dedup, style="Card.TCheckbutton").grid(row=r, column=0, columnspan=4, sticky=tk.W, pady=3)
        r += 1
        ttk.Checkbutton(body, text="音量自动放大均衡（清晰不刺耳，适合中老年人）", variable=self.var_loudnorm, style="Card.TCheckbutton").grid(row=r, column=0, columnspan=4, sticky=tk.W, pady=3)
        r += 1
        ttk.Checkbutton(body, text="已剪过的成品自动跳过（断点续剪不重复花钱）", variable=self.var_skip, style="Card.TCheckbutton").grid(row=r, column=0, columnspan=4, sticky=tk.W, pady=3)
        r += 1
        ttk.Checkbutton(body, text="导出时顺便保存一份 SRT 字幕文件", variable=self.var_srt, style="Card.TCheckbutton").grid(row=r, column=0, columnspan=4, sticky=tk.W, pady=3)

        r += 1
        btn_bar = ttk.Frame(body, style="Card.TFrame")
        btn_bar.grid(row=r, column=0, columnspan=4, sticky=tk.EW, pady=(14, 0))
        ttk.Button(btn_bar, text="保存设置", style="Primary.TButton", command=self._on_save).pack(side=tk.RIGHT)
        ttk.Button(btn_bar, text="取消", command=self.destroy).pack(side=tk.RIGHT, padx=(0, 8))
        body.columnconfigure(1, weight=1)

    def _pick_color(self) -> None:
        _, hex_c = colorchooser.askcolor(initialcolor=self.var_fcolor.get(), title="选择字幕颜色")
        if hex_c:
            self.var_fcolor.set(hex_c.upper())

    def _on_save(self) -> None:
        cfg = self.parent_app.cfg
        cfg.subtitle_source = _label_to_code(SUB_SOURCE_OPTIONS, self.var_sub_src.get())
        cfg.concurrency_mode = _label_to_code(SPEED_OPTIONS, self.var_speed.get())
        cfg.video_bitrate_mbps = int(_label_to_code(QUALITY_OPTIONS, self.var_quality.get()))
        cfg.output_fps = int(_label_to_code(FPS_OPTIONS, self.var_fps.get()))
        cfg.dedup_level = "standard" if self.var_dedup.get() else "off"
        cfg.auto_position_below_en = self.var_auto_pos.get()
        cfg.audio_loudnorm = self.var_loudnorm.get()
        cfg.skip_existing = self.var_skip.get()
        cfg.keep_srt = self.var_srt.get()
        cfg.font_name = self.var_font.get().strip() or cfg.font_name
        cfg.font_size = int(self.var_fsize.get())
        cfg.font_color = self.var_fcolor.get().strip() or "#F4EA46"
        cfg.position_y_pct = round(float(self.var_posy.get()), 1)
        cfg.max_chars_per_line = int(self.var_maxc.get())
        cfg.watermark_font_size = int(self.var_wmsize.get())
        save_config(cfg)
        self.parent_app._append_log("高级参数已更新并自动保存。")
        self.destroy()


class AssignWatermarkDialog(tk.Toplevel):
    """双击视频行或右键时弹出的单条/多条快速指定水印弹窗。"""

    def __init__(self, parent: tk.Tk, title_text: str, account_names: list[str], current: str) -> None:
        super().__init__(parent)
        self.title("指定水印账号")
        self.geometry("380x170")
        self.resizable(False, False)
        self.configure(bg=BG_CARD)
        self.transient(parent)
        self.grab_set()
        self.chosen: Optional[str] = None

        body = ttk.Frame(self, padding=16, style="Card.TFrame")
        body.pack(fill=tk.BOTH, expand=True)
        ttk.Label(body, text=title_text, style="CardBold.TLabel").pack(anchor=tk.W, pady=(0, 8))

        vals = [NO_WM_LABEL] + account_names
        self.var_choice = tk.StringVar(value=current if current else NO_WM_LABEL)
        cb = ttk.Combobox(body, textvariable=self.var_choice, values=vals, width=32)
        cb.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(body, text="提示：可直接下拉选择已有账号，也可直接手打新水印文字。", foreground=TEXT_MUTED, style="Card.TLabel").pack(anchor=tk.W)

        btns = ttk.Frame(body, style="Card.TFrame")
        btns.pack(fill=tk.X, pady=(14, 0))
        ttk.Button(btns, text="确定", style="Primary.TButton", command=self._ok).pack(side=tk.RIGHT)
        ttk.Button(btns, text="取消", command=self.destroy).pack(side=tk.RIGHT, padx=(0, 8))
        self.bind("<Return>", lambda *_: self._ok())

    def _ok(self) -> None:
        v = self.var_choice.get().strip()
        self.chosen = "" if v == NO_WM_LABEL else v
        self.destroy()


class CopywritingDialog(tk.Toplevel):
    """查看并一键复制微信视频号发布文案的弹窗。"""

    def __init__(self, parent: tk.Tk, video_name: str, copy_text: str) -> None:
        super().__init__(parent)
        self.title(f"微信视频号发布文案 - {video_name}")
        self.geometry("620x420")
        self.configure(bg=BG_CARD)
        self.transient(parent)
        self.grab_set()

        body = ttk.Frame(self, padding=14, style="Card.TFrame")
        body.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            body,
            text="已自动保存到成品同目录的【_视频号文案.txt】及【全部视频号文案汇总.txt】，也可直接点下方一键复制：",
            style="CardMuted.TLabel",
            wraplength=580,
        ).pack(anchor=tk.W, pady=(0, 8))

        self.txt = tk.Text(body, font=("Microsoft YaHei UI", 10), bg="#F8FAFC", relief="flat", wrap=tk.WORD, padx=10, pady=10)
        self.txt.pack(fill=tk.BOTH, expand=True)
        self.txt.insert("1.0", copy_text)

        btns = ttk.Frame(body, style="Card.TFrame")
        btns.pack(fill=tk.X, pady=(10, 0))
        self.lbl_copied = ttk.Label(btns, text="", foreground=ACCENT_GREEN, style="CardBold.TLabel")
        self.lbl_copied.pack(side=tk.LEFT)
        ttk.Button(btns, text="关闭", command=self.destroy).pack(side=tk.RIGHT)
        ttk.Button(btns, text="📋 一键复制全部文案", style="Primary.TButton", command=self._copy_all).pack(side=tk.RIGHT, padx=(0, 8))

    def _copy_all(self) -> None:
        content = self.txt.get("1.0", tk.END).strip()
        self.clipboard_clear()
        self.clipboard_append(content)
        self.update()
        self.lbl_copied.configure(text="✅ 已复制到剪贴板，可直接去视频号粘贴！")


class SubtitleReviewDialog(tk.Toplevel):
    """压制前字幕校对弹窗。"""

    def __init__(self, parent: tk.Tk, video_name: str, items: list[SubtitleItem]) -> None:
        super().__init__(parent)
        self.title(f"校对中文字幕 - {video_name}")
        self.geometry("760x560")
        self.transient(parent)
        self.grab_set()
        self.result: Optional[list[SubtitleItem]] = None
        self._orig_items = items

        ttk.Label(
            self,
            text="可直接修改右侧中文内容（不要删掉竖线 | ），改完点右下角“确认并导出”。",
            foreground="#444444",
        ).pack(fill=tk.X, padx=12, pady=(10, 6))

        frame = ttk.Frame(self)
        frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
        self.text = tk.Text(frame, font=("Microsoft YaHei", 11), wrap=tk.NONE)
        y_scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.text.yview)
        self.text.configure(yscrollcommand=y_scroll.set)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        for it in items:
            self.text.insert(tk.END, f"{it.index:03d} | {it.start:7.2f}-{it.end:7.2f} | {it.text}\n")

        btn_bar = ttk.Frame(self)
        btn_bar.pack(fill=tk.X, padx=12, pady=10)
        ttk.Button(btn_bar, text="取消这条", command=self._on_cancel).pack(side=tk.RIGHT, padx=6)
        ttk.Button(btn_bar, text="确认并导出", style="Primary.TButton", command=self._on_confirm).pack(side=tk.RIGHT, padx=6)
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def _on_confirm(self) -> None:
        edited: list[SubtitleItem] = []
        for idx, line in enumerate(self.text.get("1.0", tk.END).strip().splitlines(), start=1):
            parts = line.split("|", 2)
            if len(parts) < 3:
                continue
            txt = parts[2].strip()
            if not txt:
                continue
            try:
                s_str, e_str = parts[1].strip().split("-")
                start, end = float(s_str.strip()), float(e_str.strip())
            except Exception:
                if idx <= len(self._orig_items):
                    start, end = self._orig_items[idx - 1].start, self._orig_items[idx - 1].end
                else:
                    continue
            edited.append(SubtitleItem(index=len(edited) + 1, start=start, end=end, text=txt))
        self.result = edited if edited else self._orig_items
        self.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.destroy()


class AutoClipApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"AutoClip 智剪 Pro v{__version__}")
        self.geometry("1280x840")
        self.minsize(1120, 740)
        self.configure(bg=BG_APP)

        self.cfg: AppConfig = load_config()
        self.hw: HardwareProfile = detect_hardware()

        # 核心数据：每条视频独立保存其指定的账号水印（互不覆盖！）
        self.video_list: list[Path] = []
        self.video_watermarks: dict[Path, str] = {}
        self.video_states: dict[Path, str] = {}
        self.video_copywritings: dict[Path, str] = {}
        self.video_out_paths: dict[Path, Path] = {}
        self.tree_ids: dict[Path, str] = {}

        self.accounts: list[dict[str, Any]] = [
            {"name": str(a.get("name", "")).strip(), "count": int(a.get("count", 1))}
            for a in (self.cfg.watermark_accounts or [])
            if str(a.get("name", "")).strip()
        ]

        self.pipeline: Optional[BatchPipeline] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.ui_queue: queue.Queue = queue.Queue()
        self._preview_photo: Optional[ImageTk.PhotoImage] = None
        self._log_expanded = False
        self._batch_total = 0
        self._batch_done = 0

        self._setup_styles()
        self._init_vars()
        self._build_ui()
        self._refresh_account_table()
        self._update_api_badge()
        self.protocol("WM_DELETE_WINDOW", self._on_close_window)
        self.after(120, self._poll_ui_queue)

    def _setup_styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        base_font = ("Microsoft YaHei UI", 9) if platform.system() == "Windows" else ("PingFang SC", 11)
        bold_font = ("Microsoft YaHei UI", 9, "bold") if platform.system() == "Windows" else ("PingFang SC", 11, "bold")
        title_font = ("Microsoft YaHei UI", 13, "bold") if platform.system() == "Windows" else ("PingFang SC", 14, "bold")

        style.configure(".", font=base_font, background=BG_APP, foreground=TEXT_MAIN)
        style.configure("App.TFrame", background=BG_APP)
        style.configure("Card.TFrame", background=BG_CARD)
        style.configure("Header.TFrame", background=BG_HEADER)
        style.configure("HeaderTitle.TLabel", background=BG_HEADER, foreground=FG_HEADER, font=title_font)
        style.configure("HeaderSub.TLabel", background=BG_HEADER, foreground="#94A3B8", font=base_font)
        style.configure("HeaderGreen.TLabel", background=BG_HEADER, foreground="#4ADE80", font=bold_font)

        style.configure("Card.TLabel", background=BG_CARD, foreground=TEXT_MAIN)
        style.configure("CardBold.TLabel", background=BG_CARD, foreground=TEXT_MAIN, font=bold_font)
        style.configure("CardMuted.TLabel", background=BG_CARD, foreground=TEXT_MUTED)
        style.configure("Card.TCheckbutton", background=BG_CARD, foreground=TEXT_MAIN)
        style.configure("Card.TRadiobutton", background=BG_CARD, foreground=TEXT_MAIN)

        style.configure(
            "Primary.TButton",
            background=ACCENT_BLUE,
            foreground="#FFFFFF",
            font=bold_font,
            padding=(14, 7),
            borderwidth=0,
        )
        style.map("Primary.TButton", background=[("active", ACCENT_BLUE_HOVER), ("disabled", "#93C5FD")])

        style.configure(
            "Danger.TButton",
            background=ACCENT_RED,
            foreground="#FFFFFF",
            font=bold_font,
            padding=(10, 7),
            borderwidth=0,
        )
        style.map("Danger.TButton", background=[("active", "#B91C1C"), ("disabled", "#FCA5A5")])

        style.configure("Soft.TButton", background="#E2E8F0", foreground=TEXT_MAIN, padding=(10, 5), borderwidth=0)
        style.map("Soft.TButton", background=[("active", "#CBD5E1")])

        style.configure("Treeview", background="#FFFFFF", fieldbackground="#FFFFFF", rowheight=30, borderwidth=0)
        style.configure("Treeview.Heading", background="#F1F5F9", foreground=TEXT_MAIN, font=bold_font, relief="flat")
        style.map("Treeview", background=[("selected", "#DBEAFE")], foreground=[("selected", TEXT_MAIN)])

        style.configure("Blue.Horizontal.TProgressbar", troughcolor="#E2E8F0", background=ACCENT_BLUE, thickness=10)

    def _init_vars(self) -> None:
        self.var_output_dir = tk.StringVar(value=self.cfg.output_dir)
        self.var_output_fmt = tk.StringVar(value=self.cfg.output_format.lower())
        self.var_enhance = tk.StringVar(value=_code_to_label(ENHANCE_OPTIONS, self.cfg.enhance_mode))
        self.var_feathered_mask = tk.BooleanVar(value=self.cfg.enable_feathered_mask)
        self.var_gen_copy = tk.BooleanVar(value=self.cfg.generate_copywriting)
        self.var_review = tk.BooleanVar(value=self.cfg.review_subtitles)

        self.var_wm_enabled = tk.BooleanVar(value=self.cfg.watermark_enabled)
        self.var_wm_floating = tk.BooleanVar(value=self.cfg.watermark_floating)
        self.var_wm_group_dir = tk.BooleanVar(value=self.cfg.group_output_by_account)
        self.var_wm_opacity = tk.IntVar(value=self.cfg.watermark_opacity_pct)
        self.var_new_acc_name = tk.StringVar(value="")
        self.var_new_acc_count = tk.IntVar(value=3)
        self.var_quick_assign = tk.StringVar(value=NO_WM_LABEL)

        self.var_safe_zone = tk.BooleanVar(value=True)
        self.var_global_progress = tk.DoubleVar(value=0.0)
        self.var_last_log = tk.StringVar(value="就绪：请先导入视频，选择导出文件夹，然后点“开始批量剪辑”。")

    def _make_card(self, parent: tk.Widget, title: str, pad: int = 12) -> ttk.Frame:
        outer = tk.Frame(parent, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1)
        outer.pack(fill=tk.X, pady=(0, 10))
        if title:
            hdr = tk.Frame(outer, bg=BG_CARD)
            hdr.pack(fill=tk.X, padx=pad, pady=(10, 4))
            ttk.Label(hdr, text=title, style="CardBold.TLabel").pack(side=tk.LEFT)
            tk.Frame(outer, bg="#F1F5F9", height=1).pack(fill=tk.X, padx=pad, pady=(2, 6))
        inner = ttk.Frame(outer, style="Card.TFrame", padding=(pad, 4, pad, pad))
        inner.pack(fill=tk.BOTH, expand=True)
        return inner

    def _build_ui(self) -> None:
        # 1. 顶部产品导航栏
        header = ttk.Frame(self, style="Header.TFrame", padding=(18, 12))
        header.pack(fill=tk.X)

        left_h = ttk.Frame(header, style="Header.TFrame")
        left_h.pack(side=tk.LEFT)
        ttk.Label(left_h, text=f"AutoClip 智剪 Pro  v{__version__}", style="HeaderTitle.TLabel").pack(side=tk.LEFT)
        gpu_txt = (
            f"   🟢 {self.hw.gpu_name} 显卡加速已就绪"
            if self.hw.gpu_name
            else f"   ⚪ CPU 模式（{self.hw.cpu_cores} 核）"
        )
        ttk.Label(left_h, text=gpu_txt, style="HeaderGreen.TLabel").pack(side=tk.LEFT, padx=(10, 0))

        right_h = ttk.Frame(header, style="Header.TFrame")
        right_h.pack(side=tk.RIGHT)
        self.btn_api = tk.Button(
            right_h,
            text="🔑 翻译密钥",
            bg="#334155",
            fg="#FFFFFF",
            activebackground="#475569",
            activeforeground="#FFFFFF",
            relief="flat",
            padx=12,
            pady=5,
            cursor="hand2",
            command=lambda: ApiKeyDialog(self),
        )
        self.btn_api.pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(
            right_h,
            text="⚙ 高级设置",
            bg="#334155",
            fg="#FFFFFF",
            activebackground="#475569",
            activeforeground="#FFFFFF",
            relief="flat",
            padx=12,
            pady=5,
            cursor="hand2",
            command=lambda: AdvancedSettingsDialog(self),
        ).pack(side=tk.LEFT)

        # 2. 主内容区（grid 固定右栏 410px，彻底杜绝超长日志/文件名把右侧工具栏挤出或导致疯狂闪烁！）
        content = ttk.Frame(self, style="App.TFrame", padding=12)
        content.pack(fill=tk.BOTH, expand=True)
        content.columnconfigure(0, weight=1, minsize=620)
        content.columnconfigure(1, weight=0, minsize=410)
        content.rowconfigure(0, weight=1)

        left_col = ttk.Frame(content, style="App.TFrame")
        left_col.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 10))

        right_col = ttk.Frame(content, style="App.TFrame", width=410)
        right_col.grid(row=0, column=1, sticky=tk.NSEW)
        right_col.grid_propagate(False)

        # ========== 左侧：素材列表 + 手动指定水印 + 导出位置 + 启动控制 ==========
        list_card_outer = tk.Frame(left_col, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1)
        list_card_outer.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        list_card = ttk.Frame(list_card_outer, style="Card.TFrame", padding=12)
        list_card.pack(fill=tk.BOTH, expand=True)

        # 顶部导入工具栏
        tb1 = ttk.Frame(list_card, style="Card.TFrame")
        tb1.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(tb1, text="＋ 添加视频", style="Primary.TButton", command=self._on_add_files).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(tb1, text="📁 导入文件夹", style="Soft.TButton", command=self._on_add_folder).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(tb1, text="移除选中", style="Soft.TButton", command=self._on_remove_selected).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(tb1, text="清空", style="Soft.TButton", command=self._on_clear_list).pack(side=tk.LEFT)
        self.lbl_count = ttk.Label(tb1, text="已导入 0 条视频", style="CardBold.TLabel", foreground=ACCENT_BLUE)
        self.lbl_count.pack(side=tk.RIGHT)

        # 手动指定水印快捷栏（选中任意视频即可单独指定水印！）
        tb2 = tk.Frame(list_card, bg="#F8FAFC", highlightbackground="#E2E8F0", highlightthickness=1, padx=8, pady=6)
        tb2.pack(fill=tk.X, pady=(0, 8))
        tk.Label(tb2, text="给选中视频指定水印：", bg="#F8FAFC", fg=TEXT_MAIN).pack(side=tk.LEFT)
        self.cb_quick_assign = ttk.Combobox(tb2, textvariable=self.var_quick_assign, values=[NO_WM_LABEL], width=18)
        self.cb_quick_assign.pack(side=tk.LEFT, padx=(4, 6))
        ttk.Button(tb2, text="应用到选中视频", style="Soft.TButton", command=self._on_assign_selected_wm).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(tb2, text="💡 也可按住 Ctrl 多选，或直接双击某一行单独修改水印", bg="#F8FAFC", fg=TEXT_MUTED).pack(side=tk.LEFT)

        # 素材任务表格
        tree_wrap = ttk.Frame(list_card, style="Card.TFrame")
        tree_wrap.pack(fill=tk.BOTH, expand=True)
        cols = ("idx", "name", "account", "state", "progress", "detail")
        self.tree = ttk.Treeview(tree_wrap, columns=cols, show="headings", selectmode="extended")
        self.tree.heading("idx", text="#")
        self.tree.heading("name", text="视频文件名")
        self.tree.heading("account", text="指定水印账号")
        self.tree.heading("state", text="状态")
        self.tree.heading("progress", text="进度")
        self.tree.heading("detail", text="耗时 / 导出结果")
        self.tree.column("idx", width=36, anchor=tk.CENTER)
        self.tree.column("name", width=210, anchor=tk.W)
        self.tree.column("account", width=120, anchor=tk.CENTER)
        self.tree.column("state", width=78, anchor=tk.CENTER)
        self.tree.column("progress", width=58, anchor=tk.CENTER)
        self.tree.column("detail", width=210, anchor=tk.W)

        self.tree.tag_configure("done", foreground=ACCENT_GREEN)
        self.tree.tag_configure("running", foreground=ACCENT_BLUE)
        self.tree.tag_configure("failed", foreground=ACCENT_RED)
        self.tree.tag_configure("skipped", foreground=TEXT_MUTED)

        t_scroll = ttk.Scrollbar(tree_wrap, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=t_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        t_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<Double-1>", self._on_tree_double_click)
        self.tree.bind("<Button-3>", self._on_tree_right_click)
        self.tree.bind("<<TreeviewSelect>>", lambda *_: None)

        # 底部导出目录 + 总进度 + 主操作卡片
        foot_card = self._make_card(left_col, "", pad=12)

        # 导出位置（永远放在最显眼位置，一键指定！）
        out_row = ttk.Frame(foot_card, style="Card.TFrame")
        out_row.pack(fill=tk.X, pady=(2, 10))
        ttk.Label(out_row, text="导出文件夹：", style="CardBold.TLabel").pack(side=tk.LEFT)
        self.entry_out = ttk.Entry(out_row, textvariable=self.var_output_dir)
        self.entry_out.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 6))
        ttk.Button(out_row, text="选择导出文件夹", style="Soft.TButton", command=self._on_choose_output_dir).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(out_row, text="📂 打开目录", style="Soft.TButton", command=self._on_open_output_dir).pack(side=tk.LEFT)

        # 全局总进度条
        prog_row = ttk.Frame(foot_card, style="Card.TFrame")
        prog_row.pack(fill=tk.X, pady=(0, 10))
        self.lbl_total_prog = ttk.Label(prog_row, text="总进度：0 / 0 条", style="Card.TLabel")
        self.lbl_total_prog.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Progressbar(
            prog_row,
            variable=self.var_global_progress,
            maximum=100.0,
            style="Blue.Horizontal.TProgressbar",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 主启动栏
        act_row = ttk.Frame(foot_card, style="Card.TFrame")
        act_row.pack(fill=tk.X)
        self.btn_start = ttk.Button(act_row, text="▶ 开始一键批量剪辑", style="Primary.TButton", command=self._on_start_batch)
        self.btn_start.pack(side=tk.LEFT, padx=(0, 8))
        self.btn_retry = ttk.Button(act_row, text="↻ 仅重剪失败项", style="Soft.TButton", command=self._on_retry_failed)
        self.btn_retry.pack(side=tk.LEFT, padx=(0, 8))
        self.btn_stop = ttk.Button(act_row, text="■ 停止", style="Danger.TButton", command=self._on_stop_batch, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT)

        self.btn_toggle_log = ttk.Button(act_row, text="展开详细日志 ▾", style="Soft.TButton", command=self._toggle_log)
        self.btn_toggle_log.pack(side=tk.RIGHT)

        # 单行简洁状态（固定最大字符宽，绝不撑破布局）+ 可折叠详细日志
        ttk.Label(
            foot_card, textvariable=self.var_last_log, style="CardMuted.TLabel", width=62, anchor=tk.W
        ).pack(fill=tk.X, pady=(8, 0))
        self.log_wrap = ttk.Frame(foot_card, style="Card.TFrame")
        self.log_text = tk.Text(self.log_wrap, height=7, font=("Microsoft YaHei UI", 8), bg="#F8FAFC", relief="flat", state=tk.DISABLED)
        ls = ttk.Scrollbar(self.log_wrap, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=ls.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ls.pack(side=tk.RIGHT, fill=tk.Y)

        # ========== 右侧：账号水印池 + 常用开关 + 实时预览 ==========
        # 卡片 1：多账号飘动水印池
        wm_card = self._make_card(right_col, "① 账号水印池（与字幕同字体 · 半透明白字）", pad=10)

        add_r = ttk.Frame(wm_card, style="Card.TFrame")
        add_r.pack(fill=tk.X, pady=(0, 6))
        ttk.Entry(add_r, textvariable=self.var_new_acc_name, width=15).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Spinbox(add_r, from_=1, to=200, textvariable=self.var_new_acc_count, width=5).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(add_r, text="条", style="Card.TLabel").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(add_r, text="＋ 添加", style="Soft.TButton", command=self._on_add_account).pack(side=tk.LEFT)

        acc_wrap = ttk.Frame(wm_card, style="Card.TFrame")
        acc_wrap.pack(fill=tk.X, pady=(0, 6))
        self.acc_tree = ttk.Treeview(acc_wrap, columns=("acc", "cnt"), show="headings", height=4)
        self.acc_tree.heading("acc", text="账号名（水印文字）")
        self.acc_tree.heading("cnt", text="配额")
        self.acc_tree.column("acc", width=240, anchor=tk.W)
        self.acc_tree.column("cnt", width=65, anchor=tk.CENTER)
        a_sc = ttk.Scrollbar(acc_wrap, orient=tk.VERTICAL, command=self.acc_tree.yview)
        self.acc_tree.configure(yscrollcommand=a_sc.set)
        self.acc_tree.pack(side=tk.LEFT, fill=tk.X, expand=True)
        a_sc.pack(side=tk.RIGHT, fill=tk.Y)

        acc_act = ttk.Frame(wm_card, style="Card.TFrame")
        acc_act.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(acc_act, text="按数量一键分配到左侧", style="Primary.TButton", command=self._on_auto_distribute_accounts).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(acc_act, text="删除账号", style="Soft.TButton", command=self._on_del_account).pack(side=tk.LEFT)

        ttk.Checkbutton(wm_card, text="开启账号水印（勾选生效）", variable=self.var_wm_enabled, style="Card.TCheckbutton", command=self._auto_save).pack(anchor=tk.W, pady=1)
        ttk.Checkbutton(wm_card, text="全屏柔和飘动（不挡字幕，强防搬运）", variable=self.var_wm_floating, style="Card.TCheckbutton", command=self._auto_save).pack(anchor=tk.W, pady=1)
        ttk.Checkbutton(wm_card, text="导出时按账号自动建子文件夹", variable=self.var_wm_group_dir, style="Card.TCheckbutton", command=self._auto_save).pack(anchor=tk.W, pady=1)

        op_row = ttk.Frame(wm_card, style="Card.TFrame")
        op_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(op_row, text="水印透明度:", style="Card.TLabel").pack(side=tk.LEFT)
        ttk.Scale(op_row, from_=20, to=80, variable=self.var_wm_opacity, orient=tk.HORIZONTAL, command=lambda *_: self._auto_save()).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self.lbl_op = ttk.Label(op_row, text=f"{self.var_wm_opacity.get()}%", width=4, style="Card.TLabel")
        self.lbl_op.pack(side=tk.LEFT)
        self.var_wm_opacity.trace_add("write", lambda *_: self.lbl_op.config(text=f"{int(self.var_wm_opacity.get())}%"))

        # 卡片 2：常用一键开关
        quick_card = self._make_card(right_col, "② 常用一键设置", pad=10)
        f_row = ttk.Frame(quick_card, style="Card.TFrame")
        f_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(f_row, text="导出格式:", style="Card.TLabel").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Radiobutton(f_row, text="MP4（推荐）", variable=self.var_output_fmt, value="mp4", style="Card.TRadiobutton", command=self._auto_save).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Radiobutton(f_row, text="MOV", variable=self.var_output_fmt, value="mov", style="Card.TRadiobutton", command=self._auto_save).pack(side=tk.LEFT)

        e_row = ttk.Frame(quick_card, style="Card.TFrame")
        e_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(e_row, text="画质增强:", style="Card.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        cb_e = ttk.Combobox(e_row, textvariable=self.var_enhance, values=[x[0] for x in ENHANCE_OPTIONS], state="readonly", width=22)
        cb_e.pack(side=tk.LEFT, fill=tk.X, expand=True)
        cb_e.bind("<<ComboboxSelected>>", lambda *_: self._auto_save())

        ttk.Checkbutton(quick_card, text="无痕羽化蒙版去重（顶部柔化，看不出切缝）", variable=self.var_feathered_mask, style="Card.TCheckbutton", command=self._auto_save).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(quick_card, text="自动生成微信视频号文案（同步保存 txt）", variable=self.var_gen_copy, style="Card.TCheckbutton", command=self._auto_save).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(quick_card, text="导出前弹窗校对中文字幕（半自动微调）", variable=self.var_review, style="Card.TCheckbutton", command=self._auto_save).pack(anchor=tk.W, pady=2)

        # 卡片 3：实时画面预览
        prev_outer = tk.Frame(right_col, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1)
        prev_outer.pack(fill=tk.BOTH, expand=True)
        prev_hdr = ttk.Frame(prev_outer, style="Card.TFrame", padding=(10, 8, 10, 4))
        prev_hdr.pack(fill=tk.X)
        ttk.Label(prev_hdr, text="③ 效果实时预览", style="CardBold.TLabel").pack(side=tk.LEFT)
        ttk.Button(prev_hdr, text="🔄 刷新预览", style="Soft.TButton", command=self._on_render_preview).pack(side=tk.RIGHT)
        ttk.Checkbutton(prev_hdr, text="安全区红线", variable=self.var_safe_zone, style="Card.TCheckbutton", command=self._on_render_preview).pack(side=tk.RIGHT, padx=(0, 8))

        self.preview_label = ttk.Label(
            prev_outer,
            text="选中左侧任意视频后点“刷新预览”\n查看字幕与对应账号水印效果",
            anchor=tk.CENTER,
            background="#1E293B",
            foreground="#CBD5E1",
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=(2, 10))

    # ---------- 状态与配置自动保存 ----------
    def _update_api_badge(self) -> None:
        if self.cfg.api_key.strip():
            self.btn_api.configure(text="🔑 翻译密钥（已配置）", bg="#166534")
        else:
            self.btn_api.configure(text="🔑 翻译密钥（未填写）", bg="#991B1B")

    def _collect_config(self) -> AppConfig:
        self.cfg.output_dir = self.var_output_dir.get().strip()
        self.cfg.output_format = self.var_output_fmt.get().lower()
        self.cfg.enhance_mode = _label_to_code(ENHANCE_OPTIONS, self.var_enhance.get())
        self.cfg.enable_feathered_mask = self.var_feathered_mask.get()
        self.cfg.generate_copywriting = self.var_gen_copy.get()
        self.cfg.review_subtitles = self.var_review.get()
        self.cfg.watermark_enabled = self.var_wm_enabled.get()
        self.cfg.watermark_floating = self.var_wm_floating.get()
        self.cfg.group_output_by_account = self.var_wm_group_dir.get()
        self.cfg.watermark_opacity_pct = int(self.var_wm_opacity.get())
        self.cfg.watermark_accounts = list(self.accounts)
        return self.cfg

    def _auto_save(self) -> None:
        save_config(self._collect_config())

    def _toggle_log(self) -> None:
        self._log_expanded = not self._log_expanded
        if self._log_expanded:
            self.log_wrap.pack(fill=tk.BOTH, expand=False, pady=(6, 0))
            self.btn_toggle_log.configure(text="收起详细日志 ▴")
        else:
            self.log_wrap.pack_forget()
            self.btn_toggle_log.configure(text="展开详细日志 ▾")

    # ---------- 账号池 & 独立指定水印 ----------
    def _refresh_account_table(self) -> None:
        for iid in self.acc_tree.get_children():
            self.acc_tree.delete(iid)
        names: list[str] = []
        for acc in self.accounts:
            self.acc_tree.insert("", tk.END, values=(acc["name"], f"{acc['count']} 条"))
            names.append(acc["name"])
        self.cb_quick_assign.configure(values=[NO_WM_LABEL] + names)
        if names and self.var_quick_assign.get() == NO_WM_LABEL:
            self.var_quick_assign.set(names[0])
        self._auto_save()

    def _on_add_account(self) -> None:
        name = self.var_new_acc_name.get().strip()
        if not name:
            messagebox.showinfo("提示", "请先输入账号名称（例如：老张爱看剧）。")
            return
        try:
            cnt = max(1, int(self.var_new_acc_count.get()))
        except Exception:
            cnt = 1
        for acc in self.accounts:
            if acc["name"] == name:
                acc["count"] = cnt
                self.var_new_acc_name.set("")
                self._refresh_account_table()
                return
        self.accounts.append({"name": name, "count": cnt})
        self.var_new_acc_name.set("")
        self._refresh_account_table()

    def _on_del_account(self) -> None:
        sel = self.acc_tree.selection()
        if not sel:
            return
        to_del = {self.acc_tree.item(i, "values")[0] for i in sel}
        self.accounts = [a for a in self.accounts if a["name"] not in to_del]
        self._refresh_account_table()

    def _on_auto_distribute_accounts(self) -> None:
        if not self.video_list:
            messagebox.showinfo("提示", "请先在左侧导入视频，再点击一键分配。")
            return
        if not self.accounts:
            messagebox.showinfo("提示", "请先添加至少一个账号。")
            return
        assigned = assign_accounts_to_videos(len(self.video_list), self.accounts, enabled=True)
        for vp, acc in zip(self.video_list, assigned):
            self.video_watermarks[vp] = acc
        self._refresh_video_rows()
        self._append_log("已按账号配额一键分配到左侧视频列表（您仍可手动修改任意单条）。")

    def _on_assign_selected_wm(self) -> None:
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("提示", "请先在左侧列表点选一条或多条视频（按住 Ctrl 可多选）。")
            return
        chosen = self.var_quick_assign.get().strip()
        wm = "" if chosen == NO_WM_LABEL else chosen
        for iid in sel:
            vp = next((p for p, tid in self.tree_ids.items() if tid == iid), None)
            if vp:
                self.video_watermarks[vp] = wm
        self._refresh_video_rows()
        self._append_log(f"已为选中的 {len(sel)} 条视频指定水印：{wm or '不加水印'}。")

    def _on_tree_double_click(self, _event: Any) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        vp = next((p for p, tid in self.tree_ids.items() if tid == sel[0]), None)
        if not vp:
            return
        dlg = AssignWatermarkDialog(
            self,
            f"为视频指定水印：{vp.name}",
            [a["name"] for a in self.accounts],
            self.video_watermarks.get(vp, ""),
        )
        self.wait_window(dlg)
        if dlg.chosen is not None:
            self.video_watermarks[vp] = dlg.chosen
            self._refresh_video_rows()

    def _on_tree_right_click(self, event: Any) -> None:
        iid = self.tree.identify_row(event.y)
        if iid and iid not in self.tree.selection():
            self.tree.selection_set(iid)
        if not self.tree.selection():
            return
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="📋 查看 / 复制微信视频号文案", command=self._on_view_copywriting)
        menu.add_command(label="指定水印账号...", command=self._on_tree_double_click_batch)
        menu.add_command(label="预览此视频字幕与水印", command=self._on_render_preview)
        menu.add_separator()
        menu.add_command(label="移除选中视频", command=self._on_remove_selected)
        menu.tk_popup(event.x_root, event.y_root)

    def _on_view_copywriting(self) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        vp = next((p for p, tid in self.tree_ids.items() if tid == sel[0]), None)
        if not vp:
            return
        copy_txt = self.video_copywritings.get(vp, "")
        if not copy_txt:
            out_p = self.video_out_paths.get(vp)
            if out_p:
                cfile = out_p.with_name(f"{out_p.stem}_视频号文案.txt")
                if cfile.is_file():
                    copy_txt = cfile.read_text(encoding="utf-8-sig").strip()
        if not copy_txt:
            messagebox.showinfo("暂无文案", "该视频尚未生成视频号文案（请先完成剪辑并勾选“自动生成微信视频号文案”）。")
            return
        CopywritingDialog(self, vp.name, copy_txt)

    def _on_tree_double_click_batch(self) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        dlg = AssignWatermarkDialog(
            self,
            f"为选中的 {len(sel)} 条视频统一指定水印：",
            [a["name"] for a in self.accounts],
            "",
        )
        self.wait_window(dlg)
        if dlg.chosen is not None:
            for iid in sel:
                vp = next((p for p, tid in self.tree_ids.items() if tid == iid), None)
                if vp:
                    self.video_watermarks[vp] = dlg.chosen
            self._refresh_video_rows()

    def _refresh_video_rows(self) -> None:
        for idx, vp in enumerate(self.video_list, start=1):
            iid = self.tree_ids.get(vp)
            if not iid:
                continue
            vals = list(self.tree.item(iid, "values"))
            wm = self.video_watermarks.get(vp, "")
            vals[0] = str(idx)
            vals[2] = wm if wm else NO_WM_LABEL
            self.tree.item(iid, values=vals)

    # ---------- 素材管理与导出目录 ----------
    def _add_video_paths(self, paths: list[Path]) -> None:
        added = 0
        for p in paths:
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS and p not in self.tree_ids:
                self.video_list.append(p)
                self.video_watermarks[p] = ""
                self.video_states[p] = "等待中"
                iid = self.tree.insert(
                    "",
                    tk.END,
                    values=(str(len(self.video_list)), p.name, NO_WM_LABEL, "等待中", "0%", "待剪辑"),
                )
                self.tree_ids[p] = iid
                added += 1
        self.lbl_count.config(text=f"已导入 {len(self.video_list)} 条视频")
        # 若用户已配置账号池且新导入的视频尚无水印，自动按配额预填（后续随时可手动改任意单条）
        if added and self.accounts and self.var_wm_enabled.get():
            if all(not self.video_watermarks.get(v) for v in self.video_list):
                assigned = assign_accounts_to_videos(len(self.video_list), self.accounts, enabled=True)
                for vp, acc in zip(self.video_list, assigned):
                    self.video_watermarks[vp] = acc
                self._refresh_video_rows()
        if added:
            self._append_log(f"已添加 {added} 条视频素材（当前共 {len(self.video_list)} 条）。")

    def _on_add_files(self) -> None:
        files = filedialog.askopenfilenames(
            title="选择视频素材（按住 Ctrl 或 Shift 可多选）",
            filetypes=[("视频文件", "*.mp4 *.mov *.mkv *.avi *.m4v *.flv"), ("所有文件", "*.*")],
        )
        if files:
            self._add_video_paths([Path(f) for f in files])

    def _on_add_folder(self) -> None:
        folder = filedialog.askdirectory(title="选择装有视频素材的文件夹")
        if not folder:
            return
        found = sorted([p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS])
        if not found:
            messagebox.showwarning("没有找到视频", "所选文件夹里没有视频文件。")
            return
        self._add_video_paths(found)

    def _on_remove_selected(self) -> None:
        for iid in self.tree.selection():
            target = next((p for p, tid in self.tree_ids.items() if tid == iid), None)
            if target:
                self.video_list.remove(target)
                self.video_watermarks.pop(target, None)
                self.video_states.pop(target, None)
                del self.tree_ids[target]
            self.tree.delete(iid)
        self.lbl_count.config(text=f"已导入 {len(self.video_list)} 条视频")
        self._refresh_video_rows()

    def _is_running(self) -> bool:
        return self.worker_thread is not None and self.worker_thread.is_alive()

    def _on_close_window(self) -> None:
        if self._is_running() and self.pipeline is not None:
            self.pipeline.cancel()
        self.destroy()

    def _on_clear_list(self) -> None:
        if self._is_running():
            if not messagebox.askyesno("正在剪辑中", "当前仍有任务在运行，清空列表会立即强制停止所有剪辑任务，是否继续？"):
                return
            if self.pipeline:
                self.pipeline.cancel()
        self.video_list.clear()
        self.video_watermarks.clear()
        self.video_states.clear()
        self.tree_ids.clear()
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        self.lbl_count.config(text="已导入 0 条视频")
        self.var_global_progress.set(0.0)
        self.lbl_total_prog.config(text="总进度：0 / 0 条")

    def _on_choose_output_dir(self) -> None:
        d = filedialog.askdirectory(title="选择成品保存文件夹")
        if d:
            self.var_output_dir.set(d)
            self._auto_save()
            self._append_log(f"成品导出文件夹已设为：{d}")

    def _on_open_output_dir(self) -> None:
        out = self.var_output_dir.get().strip()
        if out:
            target = Path(out)
        elif self.video_list:
            target = self.video_list[0].parent / "已剪辑输出"
        else:
            target = Path.home()
        target.mkdir(parents=True, exist_ok=True)
        if platform.system() == "Windows":
            os.startfile(str(target))  # type: ignore[attr-defined]
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])

    # ---------- 实时预览 ----------
    def _on_render_preview(self) -> None:
        sel = self.tree.selection()
        target_video: Optional[Path] = None
        if sel:
            target_video = next((p for p, tid in self.tree_ids.items() if tid == sel[0]), None)
        elif self.video_list:
            target_video = self.video_list[0]
        if not target_video:
            messagebox.showinfo("提示", "请先在左侧导入至少一条视频。")
            return

        cfg = self._collect_config()
        wm = self.video_watermarks.get(target_video, "") if cfg.watermark_enabled else ""
        try:
            img_path = render_preview_frame(target_video, cfg, watermark_text=wm)
            im = Image.open(img_path).convert("RGB")
            if self.var_safe_zone.get():
                draw = ImageDraw.Draw(im)
                w, h = im.size
                safe_y = int(h * 0.82)
                for x in range(0, w, 24):
                    draw.line([(x, safe_y), (min(x + 14, w), safe_y)], fill=(255, 70, 70), width=4)
            im.thumbnail((340, 290), Image.LANCZOS)
            self._preview_photo = ImageTk.PhotoImage(im)
            self.preview_label.configure(image=self._preview_photo, text="")
        except Exception as e:
            messagebox.showerror("预览失败", f"生成预览图出错：{e}")

    # ---------- 批量执行 ----------
    def _review_on_main_thread(
        self, video_path: Path, items: list[SubtitleItem]
    ) -> Optional[list[SubtitleItem]]:
        done_ev = threading.Event()
        holder: dict[str, Optional[list[SubtitleItem]]] = {"result": None}

        def show_dialog() -> None:
            dlg = SubtitleReviewDialog(self, video_path.name, items)
            self.wait_window(dlg)
            holder["result"] = dlg.result
            done_ev.set()

        self.ui_queue.put(("callback", show_dialog))
        done_ev.wait()
        return holder["result"]

    def _start_with_pairs(self, pairs: list[tuple[Path, str]]) -> None:
        cfg = self._collect_config()
        if not cfg.api_key:
            messagebox.showwarning("请先填写密钥", "请点击右上角「🔑 翻译密钥」填入您的 DeepSeek API Key。")
            ApiKeyDialog(self)
            return
        save_config(cfg)

        self._batch_total = len(pairs)
        self._batch_done = 0
        self.var_global_progress.set(0.0)
        self.lbl_total_prog.config(text=f"总进度：0 / {self._batch_total} 条")

        self.btn_start.configure(state=tk.DISABLED)
        self.btn_retry.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)

        self.pipeline = BatchPipeline(
            cfg=cfg,
            hw=self.hw,
            status_cb=lambda st: self.ui_queue.put(("status", st)),
            log_cb=lambda m: self.ui_queue.put(("log", m)),
            review_cb=self._review_on_main_thread if cfg.review_subtitles else None,
        )

        def run_worker() -> None:
            results = self.pipeline.run_batch(pairs)
            self.ui_queue.put(("done", results))

        self.worker_thread = threading.Thread(target=run_worker, daemon=True)
        self.worker_thread.start()

    def _on_start_batch(self) -> None:
        if not self.video_list:
            messagebox.showwarning("请导入视频", "请先点左上角「＋ 添加视频」导入素材。")
            return
        wm_on = self.var_wm_enabled.get()
        pairs = [(vp, self.video_watermarks.get(vp, "") if wm_on else "") for vp in self.video_list]
        self._start_with_pairs(pairs)

    def _on_retry_failed(self) -> None:
        wm_on = self.var_wm_enabled.get()
        failed_pairs = [
            (vp, self.video_watermarks.get(vp, "") if wm_on else "")
            for vp in self.video_list
            if self.video_states.get(vp) in ("失败", "已取消")
        ]
        if not failed_pairs:
            messagebox.showinfo("提示", "当前没有失败或取消的视频需要重剪。")
            return
        self._start_with_pairs(failed_pairs)

    def _on_stop_batch(self) -> None:
        if self.pipeline:
            self.btn_stop.configure(state=tk.DISABLED)
            self._append_log("正在立即强制停止所有正在运行的剪辑任务...")
            threading.Thread(target=self.pipeline.cancel, daemon=True).start()

    def _append_log(self, msg: str) -> None:
        short_msg = msg if len(msg) <= 68 else (msg[:65] + "...")
        self.var_last_log.set(short_msg)
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _poll_ui_queue(self) -> None:
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "status":
                    st: TaskStatus = payload
                    prev_state = self.video_states.get(st.video_path, "")
                    self.video_states[st.video_path] = st.state
                    if st.output_path is not None:
                        self.video_out_paths[st.video_path] = st.output_path
                    if st.copywriting:
                        self.video_copywritings[st.video_path] = st.copywriting
                    iid = self.tree_ids.get(st.video_path)
                    if iid:
                        idx = self.video_list.index(st.video_path) + 1 if st.video_path in self.video_list else 0
                        pct = f"{int(round(st.progress * 100))}%"
                        acc_disp = st.account_name if st.account_name else NO_WM_LABEL
                        tag = ""
                        if st.state == "已完成":
                            tag = "done"
                        elif st.state == "失败":
                            tag = "failed"
                        elif st.state == "已跳过":
                            tag = "skipped"
                        elif st.state not in ("等待中", "已取消"):
                            tag = "running"
                        self.tree.item(
                            iid,
                            values=(str(idx), st.video_path.name, acc_disp, st.state, pct, st.message),
                            tags=(tag,) if tag else (),
                        )
                    if prev_state not in ("已完成", "已跳过", "失败") and st.state in ("已完成", "已跳过", "失败"):
                        self._batch_done += 1
                        pct_all = (self._batch_done / max(1, self._batch_total)) * 100.0
                        self.var_global_progress.set(pct_all)
                        self.lbl_total_prog.config(text=f"总进度：{self._batch_done} / {self._batch_total} 条")
                elif kind == "callback":
                    payload()
                elif kind == "done":
                    results: list[TaskStatus] = payload
                    new_ok = sum(1 for r in results if r.state == "已完成")
                    skip_n = sum(1 for r in results if r.state == "已跳过")
                    fail_n = sum(1 for r in results if r.state == "失败")
                    cancel_n = sum(1 for r in results if r.state == "已取消")
                    # 统计实际落盘成品数量（按账号汇总核对）
                    acc_counts: dict[str, int] = {}
                    for r in results:
                        if r.state in ("已完成", "已跳过"):
                            k = r.account_name or "无水印"
                            acc_counts[k] = acc_counts.get(k, 0) + 1
                    acc_summary = "，".join(f"{k}:{v}条" for k, v in acc_counts.items())
                    self._append_log(
                        f"=== 批量结束：新剪成功 {new_ok} 条，跳过已有 {skip_n} 条，失败 {fail_n} 条"
                        + (f"（按账号：{acc_summary}）" if acc_summary else "")
                        + " ==="
                    )
                    self.btn_start.configure(state=tk.NORMAL)
                    self.btn_retry.configure(state=tk.NORMAL)
                    self.btn_stop.configure(state=tk.DISABLED)
                    detail_lines = [
                        f"新剪成功：{new_ok} 条",
                        f"跳过已有：{skip_n} 条",
                        f"失败：{fail_n} 条",
                    ]
                    if cancel_n:
                        detail_lines.append(f"已取消：{cancel_n} 条")
                    if acc_summary:
                        detail_lines.append(f"\n按账号产出：{acc_summary}")
                    messagebox.showinfo("批量剪辑完成", "\n".join(detail_lines))
        except queue.Empty:
            pass
        self.after(120, self._poll_ui_queue)


def launch_gui() -> None:
    app = AutoClipApp()
    app.mainloop()

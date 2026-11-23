"""人性化中文图形界面（v2.0.0）：大白话选项 + 多账号飘动水印配额分配 + 实时效果预览。"""

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

# 大白话中文映射（界面彻底去除英文专业术语）
ENHANCE_OPTIONS = [
    ("极速高清增强（推荐，不减速）", "fast"),
    ("真·AI超清（3060显卡，较慢）", "ai"),
    ("关闭", "off"),
]
SPEED_OPTIONS = [
    ("智能自动（按显卡自动并行）", "auto"),
    ("一条一条顺序剪（最稳）", "1"),
    ("同时剪 2 条（并行）", "2"),
    ("同时剪 3 条（高配并行）", "3"),
    ("同时剪 4 条（3060 满载并行）", "4"),
]
FPS_OPTIONS = [
    ("保持原片 30 帧（推荐，省一半CPU与时间）", 30),
    ("60 帧（高帧率）", 60),
]
QUALITY_OPTIONS = [
    ("高清推荐（画质清晰、体积适中）", 10),
    ("超清高画质（体积稍大）", 14),
    ("省空间标准（上传最快）", 6),
]
SUB_SOURCE_OPTIONS = [
    ("自动对齐画面英文字幕（推荐，100%同步）", "ocr_first"),
    ("仅听声音识别（无英文字幕的视频用）", "whisper"),
]


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
        ttk.Button(btn_bar, text="确认并导出", command=self._on_confirm).pack(side=tk.RIGHT, padx=6)
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
        self.title(f"自动剪辑工具 AutoClip v{__version__}（中老年字幕 · 飘动水印 · 无痕去重）")
        self.geometry("1240x820")
        self.minsize(1100, 740)

        self.cfg: AppConfig = load_config()
        self.hw: HardwareProfile = detect_hardware()
        self.video_list: list[Path] = []
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

        self._init_vars()
        self._build_ui()
        self._refresh_account_table()
        self.after(120, self._poll_ui_queue)

    def _init_vars(self) -> None:
        self.var_api_key = tk.StringVar(value=self.cfg.api_key)
        self.var_api_url = tk.StringVar(value=self.cfg.api_base_url)
        self.var_api_model = tk.StringVar(value=self.cfg.api_model)
        self.var_show_key = tk.BooleanVar(value=False)

        self.var_sub_source = tk.StringVar(value=_code_to_label(SUB_SOURCE_OPTIONS, self.cfg.subtitle_source))
        self.var_output_fmt = tk.StringVar(value=self.cfg.output_format.lower())
        self.var_speed = tk.StringVar(value=_code_to_label(SPEED_OPTIONS, self.cfg.concurrency_mode))
        self.var_enhance = tk.StringVar(value=_code_to_label(ENHANCE_OPTIONS, self.cfg.enhance_mode))
        self.var_quality = tk.StringVar(value=_code_to_label(QUALITY_OPTIONS, self.cfg.video_bitrate_mbps))
        self.var_fps = tk.StringVar(value=_code_to_label(FPS_OPTIONS, self.cfg.output_fps))
        self.var_output_dir = tk.StringVar(value=self.cfg.output_dir)

        self.var_feathered_mask = tk.BooleanVar(value=self.cfg.enable_feathered_mask)
        self.var_dedup_on = tk.BooleanVar(value=(self.cfg.dedup_level != "off"))
        self.var_loudnorm = tk.BooleanVar(value=self.cfg.audio_loudnorm)
        self.var_skip_existing = tk.BooleanVar(value=self.cfg.skip_existing)
        self.var_review = tk.BooleanVar(value=self.cfg.review_subtitles)
        self.var_keep_srt = tk.BooleanVar(value=self.cfg.keep_srt)

        # 水印变量
        self.var_wm_enabled = tk.BooleanVar(value=self.cfg.watermark_enabled)
        self.var_wm_floating = tk.BooleanVar(value=self.cfg.watermark_floating)
        self.var_wm_group_dir = tk.BooleanVar(value=self.cfg.group_output_by_account)
        self.var_wm_opacity = tk.IntVar(value=self.cfg.watermark_opacity_pct)
        self.var_wm_size = tk.IntVar(value=self.cfg.watermark_font_size)
        self.var_new_acc_name = tk.StringVar(value="")
        self.var_new_acc_count = tk.IntVar(value=3)

        # 字幕样式变量
        self.var_font_name = tk.StringVar(value=self.cfg.font_name)
        self.var_font_size = tk.IntVar(value=self.cfg.font_size)
        self.var_font_color = tk.StringVar(value=self.cfg.font_color)
        self.var_pos_y = tk.DoubleVar(value=self.cfg.position_y_pct)
        self.var_max_chars = tk.IntVar(value=self.cfg.max_chars_per_line)
        self.var_safe_zone = tk.BooleanVar(value=True)

    def _build_ui(self) -> None:
        # 1. 顶部硬件与状态条
        top_bar = ttk.Frame(self, padding=(12, 6))
        top_bar.pack(fill=tk.X)
        hw_msg = (
            f"✅ 已识别显卡：{self.hw.gpu_name}（已自动开启显卡极速加速）"
            if self.hw.gpu_name
            else f"当前模式：CPU 运行（{self.hw.cpu_cores} 核）"
        )
        ttk.Label(top_bar, text=hw_msg, foreground="#155724", font=("Microsoft YaHei", 9, "bold")).pack(side=tk.LEFT)
        ttk.Label(
            top_bar,
            text="使用步骤：① 填好密钥 → ② 左边导入视频 → ③ 右边填账号水印分配 → ④ 点开始一键剪辑",
            foreground="#555555",
        ).pack(side=tk.RIGHT)

        # 2. 翻译密钥区（简洁明了）
        api_frame = ttk.LabelFrame(self, text="第 1 步：填写 DeepSeek 翻译密钥（填一次自动记住）", padding=(10, 8))
        api_frame.pack(fill=tk.X, padx=12, pady=(2, 8))

        ttk.Label(api_frame, text="DeepSeek 密钥:").grid(row=0, column=0, sticky=tk.W, padx=(0, 6))
        self.entry_key = ttk.Entry(api_frame, textvariable=self.var_api_key, width=42, show="*")
        self.entry_key.grid(row=0, column=1, sticky=tk.EW, padx=(0, 6))
        ttk.Checkbutton(api_frame, text="显示", variable=self.var_show_key, command=self._toggle_key).grid(
            row=0, column=2, padx=(0, 10)
        )
        ttk.Button(api_frame, text="测试是否可用", command=self._on_test_api).grid(row=0, column=3, padx=4)
        ttk.Button(api_frame, text="保存全部设置", command=self._on_save_config).grid(row=0, column=4, padx=4)
        api_frame.columnconfigure(1, weight=1)

        # 3. 主区域左右分栏
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        # ---------- 左侧：视频列表 + 开始按钮 + 简要进度日志 ----------
        left_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, weight=3)

        toolbar = ttk.Frame(left_frame)
        toolbar.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(toolbar, text="＋ 导入视频（可一次选多个）", command=self._on_add_files).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="📁 导入整个文件夹", command=self._on_add_folder).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="删除选中", command=self._on_remove_selected).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(toolbar, text="清空列表", command=self._on_clear_list).pack(side=tk.LEFT, padx=(0, 6))
        self.lbl_count = ttk.Label(toolbar, text="已导入：0 条视频", font=("Microsoft YaHei", 9, "bold"))
        self.lbl_count.pack(side=tk.RIGHT)

        tree_wrap = ttk.Frame(left_frame)
        tree_wrap.pack(fill=tk.BOTH, expand=True)
        cols = ("name", "account", "state", "progress", "detail")
        self.tree = ttk.Treeview(tree_wrap, columns=cols, show="headings", height=13)
        self.tree.heading("name", text="视频文件名")
        self.tree.heading("account", text="分配水印账号")
        self.tree.heading("state", text="当前状态")
        self.tree.heading("progress", text="进度")
        self.tree.heading("detail", text="说明 / 成品名称")
        self.tree.column("name", width=185, anchor=tk.W)
        self.tree.column("account", width=110, anchor=tk.CENTER)
        self.tree.column("state", width=80, anchor=tk.CENTER)
        self.tree.column("progress", width=58, anchor=tk.CENTER)
        self.tree.column("detail", width=210, anchor=tk.W)
        t_scroll = ttk.Scrollbar(tree_wrap, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=t_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        t_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        action_bar = ttk.Frame(left_frame, padding=(0, 8))
        action_bar.pack(fill=tk.X)
        self.btn_start = ttk.Button(action_bar, text="▶ 开始一键批量剪辑", command=self._on_start_batch)
        self.btn_start.pack(side=tk.LEFT, padx=(0, 10))
        self.btn_stop = ttk.Button(action_bar, text="■ 停止", command=self._on_stop_batch, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(action_bar, text="📂 打开成品文件夹", command=self._on_open_output_dir).pack(side=tk.LEFT)

        log_frame = ttk.LabelFrame(left_frame, text="实时进度记录", padding=4)
        log_frame.pack(fill=tk.BOTH, expand=False)
        self.log_text = tk.Text(log_frame, height=8, font=("Microsoft YaHei", 9), state=tk.DISABLED)
        l_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=l_scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        l_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # ---------- 右侧：3 个超清晰的中文标签页 ----------
        right_frame = ttk.Frame(main_pane, padding=(8, 0, 0, 0))
        main_pane.add(right_frame, weight=2)

        nb = ttk.Notebook(right_frame)
        nb.pack(fill=tk.BOTH, expand=True)

        tab_wm = ttk.Frame(nb, padding=10)
        tab_basic = ttk.Frame(nb, padding=10)
        tab_sub = ttk.Frame(nb, padding=10)
        nb.add(tab_wm, text="① 账号与飘动水印")
        nb.add(tab_basic, text="② 常用剪辑设置")
        nb.add(tab_sub, text="③ 字幕样式与预览")

        # === Tab 1: 多账号与飘动水印（最常用） ===
        ttk.Checkbutton(
            tab_wm,
            text="开启半透明白色账号水印（与字幕同字体）",
            variable=self.var_wm_enabled,
            command=self._sync_video_account_column,
        ).pack(anchor=tk.W, pady=(0, 4))
        ttk.Checkbutton(
            tab_wm,
            text="水印在画面里全屏柔和飘动（不挡字幕，强防搬运）",
            variable=self.var_wm_floating,
        ).pack(anchor=tk.W, pady=(0, 4))
        ttk.Checkbutton(
            tab_wm,
            text="导出时按账号自动分文件夹保存（方便发给不同号）",
            variable=self.var_wm_group_dir,
        ).pack(anchor=tk.W, pady=(0, 8))

        add_box = ttk.LabelFrame(tab_wm, text="添加账号并分配视频条数（例如：A账号 3条、B账号 6条）", padding=8)
        add_box.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(add_box, text="账号名(水印):").grid(row=0, column=0, sticky=tk.W, padx=(0, 4))
        ttk.Entry(add_box, textvariable=self.var_new_acc_name, width=18).grid(row=0, column=1, sticky=tk.EW, padx=(0, 8))
        ttk.Label(add_box, text="分配条数:").grid(row=0, column=2, sticky=tk.W, padx=(0, 4))
        ttk.Spinbox(add_box, from_=1, to=200, textvariable=self.var_new_acc_count, width=6).grid(row=0, column=3, padx=(0, 8))
        ttk.Button(add_box, text="＋ 添加", command=self._on_add_account).grid(row=0, column=4)
        add_box.columnconfigure(1, weight=1)

        acc_list_wrap = ttk.Frame(tab_wm)
        acc_list_wrap.pack(fill=tk.BOTH, expand=True)
        self.acc_tree = ttk.Treeview(acc_list_wrap, columns=("acc", "cnt"), show="headings", height=6)
        self.acc_tree.heading("acc", text="账号名称（显示在画面里的水印）")
        self.acc_tree.heading("cnt", text="分配条数")
        self.acc_tree.column("acc", width=230, anchor=tk.W)
        self.acc_tree.column("cnt", width=80, anchor=tk.CENTER)
        a_scroll = ttk.Scrollbar(acc_list_wrap, orient=tk.VERTICAL, command=self.acc_tree.yview)
        self.acc_tree.configure(yscrollcommand=a_scroll.set)
        self.acc_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        a_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        acc_btn_bar = ttk.Frame(tab_wm, padding=(0, 6))
        acc_btn_bar.pack(fill=tk.X)
        ttk.Button(acc_btn_bar, text="删除选中账号", command=self._on_del_account).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(acc_btn_bar, text="清空账号", command=self._on_clear_accounts).pack(side=tk.LEFT)
        self.lbl_acc_summary = ttk.Label(acc_btn_bar, text="已分配：0 条", foreground="#0d6efd")
        self.lbl_acc_summary.pack(side=tk.RIGHT)

        wm_style_box = ttk.Frame(tab_wm, padding=(0, 4))
        wm_style_box.pack(fill=tk.X)
        ttk.Label(wm_style_box, text="水印透明度:").grid(row=0, column=0, sticky=tk.W, pady=3)
        ttk.Scale(wm_style_box, from_=20, to=80, variable=self.var_wm_opacity, orient=tk.HORIZONTAL).grid(
            row=0, column=1, sticky=tk.EW, padx=6
        )
        self.lbl_wm_op = ttk.Label(wm_style_box, text=f"{self.var_wm_opacity.get()}%", width=5)
        self.lbl_wm_op.grid(row=0, column=2)
        self.var_wm_opacity.trace_add("write", lambda *_: self.lbl_wm_op.config(text=f"{int(self.var_wm_opacity.get())}%"))

        ttk.Label(wm_style_box, text="水印大小:").grid(row=1, column=0, sticky=tk.W, pady=3)
        ttk.Spinbox(wm_style_box, from_=24, to=72, textvariable=self.var_wm_size, width=8).grid(row=1, column=1, sticky=tk.W, padx=6)
        wm_style_box.columnconfigure(1, weight=1)

        # === Tab 2: 常用剪辑设置（全中文大白话） ===
        r = 0
        ttk.Label(tab_basic, text="导出格式:").grid(row=r, column=0, sticky=tk.W, pady=6)
        fmt_row = ttk.Frame(tab_basic)
        fmt_row.grid(row=r, column=1, sticky=tk.W, pady=6)
        ttk.Radiobutton(fmt_row, text="MP4（推荐，平台通用）", variable=self.var_output_fmt, value="mp4").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(fmt_row, text="MOV", variable=self.var_output_fmt, value="mov").pack(side=tk.LEFT)

        r += 1
        ttk.Label(tab_basic, text="字幕识别方式:").grid(row=r, column=0, sticky=tk.W, pady=6)
        ttk.Combobox(
            tab_basic,
            textvariable=self.var_sub_source,
            values=[x[0] for x in SUB_SOURCE_OPTIONS],
            state="readonly",
            width=32,
        ).grid(row=r, column=1, sticky=tk.EW, pady=6)

        r += 1
        ttk.Label(tab_basic, text="画质超清增强:").grid(row=r, column=0, sticky=tk.W, pady=6)
        ttk.Combobox(
            tab_basic,
            textvariable=self.var_enhance,
            values=[x[0] for x in ENHANCE_OPTIONS],
            state="readonly",
            width=32,
        ).grid(row=r, column=1, sticky=tk.EW, pady=6)

        r += 1
        ttk.Label(tab_basic, text="视频清晰度:").grid(row=r, column=0, sticky=tk.W, pady=6)
        ttk.Combobox(
            tab_basic,
            textvariable=self.var_quality,
            values=[x[0] for x in QUALITY_OPTIONS],
            state="readonly",
            width=32,
        ).grid(row=r, column=1, sticky=tk.EW, pady=6)

        r += 1
        ttk.Label(tab_basic, text="导出帧率:").grid(row=r, column=0, sticky=tk.W, pady=6)
        ttk.Combobox(
            tab_basic,
            textvariable=self.var_fps,
            values=[x[0] for x in FPS_OPTIONS],
            state="readonly",
            width=32,
        ).grid(row=r, column=1, sticky=tk.EW, pady=6)

        r += 1
        ttk.Label(tab_basic, text="批量剪辑速度:").grid(row=r, column=0, sticky=tk.W, pady=6)
        ttk.Combobox(
            tab_basic,
            textvariable=self.var_speed,
            values=[x[0] for x in SPEED_OPTIONS],
            state="readonly",
            width=32,
        ).grid(row=r, column=1, sticky=tk.EW, pady=6)

        r += 1
        ttk.Label(tab_basic, text="成品保存位置:").grid(row=r, column=0, sticky=tk.W, pady=6)
        out_row = ttk.Frame(tab_basic)
        out_row.grid(row=r, column=1, sticky=tk.EW, pady=6)
        ttk.Entry(out_row, textvariable=self.var_output_dir).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(out_row, text="选择", width=6, command=self._on_choose_output_dir).pack(side=tk.LEFT, padx=(4, 0))

        r += 1
        ttk.Checkbutton(
            tab_basic, text="开启无痕羽化蒙版去重（顶部 22% 超宽柔化，看不出切缝）", variable=self.var_feathered_mask
        ).grid(row=r, column=0, columnspan=2, sticky=tk.W, pady=4)
        r += 1
        ttk.Checkbutton(
            tab_basic, text="开启隐形防搬运去重（微缩放 + 微调色 + 清除原属性）", variable=self.var_dedup_on
        ).grid(row=r, column=0, columnspan=2, sticky=tk.W, pady=4)
        r += 1
        ttk.Checkbutton(
            tab_basic, text="音量自动放大均衡（声音洪亮不刺耳，适合中老年人）", variable=self.var_loudnorm
        ).grid(row=r, column=0, columnspan=2, sticky=tk.W, pady=4)
        r += 1
        ttk.Checkbutton(
            tab_basic, text="已经剪过的视频自动跳过（断点续剪，不重复花钱）", variable=self.var_skip_existing
        ).grid(row=r, column=0, columnspan=2, sticky=tk.W, pady=4)
        r += 1
        ttk.Checkbutton(
            tab_basic, text="导出前暂停弹窗，让我手动检查修改字幕", variable=self.var_review
        ).grid(row=r, column=0, columnspan=2, sticky=tk.W, pady=4)
        r += 1
        ttk.Checkbutton(
            tab_basic, text="顺便保存一份 SRT 字幕文件", variable=self.var_keep_srt
        ).grid(row=r, column=0, columnspan=2, sticky=tk.W, pady=4)
        tab_basic.columnconfigure(1, weight=1)

        # === Tab 3: 字幕样式与实时效果预览 ===
        style_grid = ttk.Frame(tab_sub)
        style_grid.pack(fill=tk.X)

        ttk.Label(style_grid, text="字体:").grid(row=0, column=0, sticky=tk.W, pady=4)
        ttk.Entry(style_grid, textvariable=self.var_font_name, width=16).grid(row=0, column=1, sticky=tk.W, pady=4)
        ttk.Label(style_grid, text="字号:").grid(row=0, column=2, sticky=tk.W, padx=(10, 4))
        ttk.Spinbox(style_grid, from_=40, to=120, textvariable=self.var_font_size, width=6).grid(row=0, column=3, sticky=tk.W)

        ttk.Label(style_grid, text="颜色:").grid(row=1, column=0, sticky=tk.W, pady=4)
        c_row = ttk.Frame(style_grid)
        c_row.grid(row=1, column=1, sticky=tk.W, pady=4)
        ttk.Entry(c_row, textvariable=self.var_font_color, width=9).pack(side=tk.LEFT)
        ttk.Button(c_row, text="选色", width=5, command=self._on_pick_color).pack(side=tk.LEFT, padx=3)

        ttk.Label(style_grid, text="每行最多字数:").grid(row=1, column=2, sticky=tk.W, padx=(10, 4))
        ttk.Spinbox(style_grid, from_=8, to=20, textvariable=self.var_max_chars, width=6).grid(row=1, column=3, sticky=tk.W)

        ttk.Label(style_grid, text="上下位置:").grid(row=2, column=0, sticky=tk.W, pady=4)
        pos_row = ttk.Frame(style_grid)
        pos_row.grid(row=2, column=1, columnspan=3, sticky=tk.EW, pady=4)
        ttk.Scale(pos_row, from_=60.0, to=92.0, variable=self.var_pos_y, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_pos_y = ttk.Label(pos_row, text=f"{self.var_pos_y.get():.1f}%", width=6)
        self.lbl_pos_y.pack(side=tk.LEFT, padx=4)
        self.var_pos_y.trace_add("write", lambda *_: self.lbl_pos_y.config(text=f"{self.var_pos_y.get():.1f}%"))

        prev_ctrl = ttk.Frame(tab_sub, padding=(0, 6))
        prev_ctrl.pack(fill=tk.X)
        ttk.Button(prev_ctrl, text="🔄 刷新字幕与水印预览", command=self._on_render_preview).pack(side=tk.LEFT)
        ttk.Checkbutton(
            prev_ctrl, text="显示抖音/视频号防遮挡红线", variable=self.var_safe_zone, command=self._on_render_preview
        ).pack(side=tk.LEFT, padx=8)

        self.preview_label = ttk.Label(
            tab_sub,
            text="在左侧导入视频后，点上方“刷新字幕与水印预览”即可查看真实效果",
            anchor=tk.CENTER,
            background="#1e1e1e",
            foreground="#cccccc",
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True)

    # ---------- 账号水印管理 ----------
    def _refresh_account_table(self) -> None:
        for iid in self.acc_tree.get_children():
            self.acc_tree.delete(iid)
        total_alloc = 0
        for acc in self.accounts:
            self.acc_tree.insert("", tk.END, values=(acc["name"], f"{acc['count']} 条"))
            total_alloc += int(acc["count"])
        self.lbl_acc_summary.config(text=f"共 {len(self.accounts)} 个账号，已分配 {total_alloc} 条")
        self._sync_video_account_column()

    def _sync_video_account_column(self) -> None:
        assigned = assign_accounts_to_videos(
            len(self.video_list), self.accounts, enabled=self.var_wm_enabled.get()
        )
        for vp, acc in zip(self.video_list, assigned):
            iid = self.tree_ids.get(vp)
            if iid:
                cur_vals = list(self.tree.item(iid, "values"))
                if len(cur_vals) >= 5:
                    cur_vals[1] = acc if acc else "（无水印）"
                    self.tree.item(iid, values=cur_vals)

    def _on_add_account(self) -> None:
        name = self.var_new_acc_name.get().strip()
        if not name:
            messagebox.showinfo("提示", "请先输入账号名称（例如：老张爱看剧）。")
            return
        try:
            cnt = max(1, int(self.var_new_acc_count.get()))
        except Exception:
            cnt = 1
        # 若已存在同名账号则更新条数
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
        names_to_del = {self.acc_tree.item(i, "values")[0] for i in sel}
        self.accounts = [a for a in self.accounts if a["name"] not in names_to_del]
        self._refresh_account_table()

    def _on_clear_accounts(self) -> None:
        self.accounts.clear()
        self._refresh_account_table()

    # ---------- 常规操作 ----------
    def _toggle_key(self) -> None:
        self.entry_key.configure(show="" if self.var_show_key.get() else "*")

    def _collect_config(self) -> AppConfig:
        self.cfg.api_key = self.var_api_key.get().strip()
        self.cfg.api_base_url = self.var_api_url.get().strip() or "https://api.deepseek.com"
        self.cfg.api_model = self.var_api_model.get().strip() or "deepseek-chat"
        self.cfg.subtitle_source = _label_to_code(SUB_SOURCE_OPTIONS, self.var_sub_source.get())
        self.cfg.output_format = self.var_output_fmt.get().lower()
        self.cfg.concurrency_mode = _label_to_code(SPEED_OPTIONS, self.var_speed.get())
        self.cfg.enhance_mode = _label_to_code(ENHANCE_OPTIONS, self.var_enhance.get())
        self.cfg.video_bitrate_mbps = int(_label_to_code(QUALITY_OPTIONS, self.var_quality.get()))
        self.cfg.output_fps = int(_label_to_code(FPS_OPTIONS, self.var_fps.get()))
        self.cfg.output_dir = self.var_output_dir.get().strip()
        self.cfg.enable_feathered_mask = self.var_feathered_mask.get()
        self.cfg.dedup_level = "standard" if self.var_dedup_on.get() else "off"
        self.cfg.audio_loudnorm = self.var_loudnorm.get()
        self.cfg.skip_existing = self.var_skip_existing.get()
        self.cfg.review_subtitles = self.var_review.get()
        self.cfg.keep_srt = self.var_keep_srt.get()

        self.cfg.watermark_enabled = self.var_wm_enabled.get()
        self.cfg.watermark_floating = self.var_wm_floating.get()
        self.cfg.group_output_by_account = self.var_wm_group_dir.get()
        self.cfg.watermark_opacity_pct = int(self.var_wm_opacity.get())
        self.cfg.watermark_font_size = int(self.var_wm_size.get())
        self.cfg.watermark_accounts = list(self.accounts)

        self.cfg.font_name = self.var_font_name.get().strip() or self.cfg.font_name
        self.cfg.font_size = int(self.var_font_size.get())
        self.cfg.font_color = self.var_font_color.get().strip() or "#F4EA46"
        self.cfg.position_y_pct = round(float(self.var_pos_y.get()), 1)
        self.cfg.max_chars_per_line = int(self.var_max_chars.get())
        return self.cfg

    def _on_save_config(self) -> None:
        cfg = self._collect_config()
        save_config(cfg)
        self._append_log("所有设置与账号水印分配已保存。")
        messagebox.showinfo("保存成功", "设置已保存，下次打开自动记住！")

    def _on_test_api(self) -> None:
        cfg = self._collect_config()
        save_config(cfg)
        self._append_log("正在测试 DeepSeek 翻译密钥...")

        def worker() -> None:
            ok, msg = test_deepseek_connection(cfg)
            self.ui_queue.put(("api_test", (ok, msg)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_pick_color(self) -> None:
        _, hex_c = colorchooser.askcolor(initialcolor=self.var_font_color.get(), title="选择字幕颜色")
        if hex_c:
            self.var_font_color.set(hex_c.upper())

    def _on_choose_output_dir(self) -> None:
        d = filedialog.askdirectory(title="选择成品保存文件夹")
        if d:
            self.var_output_dir.set(d)

    def _add_video_paths(self, paths: list[Path]) -> None:
        added = 0
        for p in paths:
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS and p not in self.tree_ids:
                self.video_list.append(p)
                iid = self.tree.insert("", tk.END, values=(p.name, "（无水印）", "等待中", "0%", str(p.parent)))
                self.tree_ids[p] = iid
                added += 1
        self.lbl_count.config(text=f"已导入：{len(self.video_list)} 条视频")
        self._sync_video_account_column()
        if added:
            self._append_log(f"已导入 {added} 条视频（当前列表共 {len(self.video_list)} 条）。")

    def _on_add_files(self) -> None:
        files = filedialog.askopenfilenames(
            title="选择视频素材（按住 Ctrl 或 Shift 可一次多选）",
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
                del self.tree_ids[target]
            self.tree.delete(iid)
        self.lbl_count.config(text=f"已导入：{len(self.video_list)} 条视频")
        self._sync_video_account_column()

    def _on_clear_list(self) -> None:
        self.video_list.clear()
        self.tree_ids.clear()
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        self.lbl_count.config(text="已导入：0 条视频")

    def _on_render_preview(self) -> None:
        sel = self.tree.selection()
        target_video: Optional[Path] = None
        if sel:
            target_video = next((p for p, tid in self.tree_ids.items() if tid == sel[0]), None)
        elif self.video_list:
            target_video = self.video_list[0]

        if not target_video:
            messagebox.showinfo("提示", "请先在左边导入至少一条视频。")
            return

        cfg = self._collect_config()
        sample_wm = self.accounts[0]["name"] if (cfg.watermark_enabled and self.accounts) else ""
        try:
            img_path = render_preview_frame(target_video, cfg, watermark_text=sample_wm)
            im = Image.open(img_path).convert("RGB")
            if self.var_safe_zone.get():
                draw = ImageDraw.Draw(im)
                w, h = im.size
                safe_y = int(h * 0.82)
                for x in range(0, w, 24):
                    draw.line([(x, safe_y), (min(x + 14, w), safe_y)], fill=(255, 70, 70), width=4)
            im.thumbnail((340, 440), Image.LANCZOS)
            self._preview_photo = ImageTk.PhotoImage(im)
            self.preview_label.configure(image=self._preview_photo, text="")
        except Exception as e:
            messagebox.showerror("预览失败", f"生成预览图出错：{e}")

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

    def _on_start_batch(self) -> None:
        if not self.video_list:
            messagebox.showwarning("请先导入视频", "请先点击左上角“导入视频”添加至少一个视频。")
            return
        cfg = self._collect_config()
        if not cfg.api_key:
            messagebox.showwarning("缺少翻译密钥", "请先在顶部填入您的 DeepSeek 密钥。")
            return
        save_config(cfg)

        self.btn_start.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)

        videos = list(self.video_list)
        self.pipeline = BatchPipeline(
            cfg=cfg,
            hw=self.hw,
            status_cb=lambda st: self.ui_queue.put(("status", st)),
            log_cb=lambda m: self.ui_queue.put(("log", m)),
            review_cb=self._review_on_main_thread if cfg.review_subtitles else None,
        )

        def run_worker() -> None:
            results = self.pipeline.run_batch(videos)
            self.ui_queue.put(("done", results))

        self.worker_thread = threading.Thread(target=run_worker, daemon=True)
        self.worker_thread.start()

    def _on_stop_batch(self) -> None:
        if self.pipeline:
            self.pipeline.cancel()
            self._append_log("正在停止剩余任务，请稍候...")

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

    def _append_log(self, msg: str) -> None:
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
                    iid = self.tree_ids.get(st.video_path)
                    if iid:
                        pct = f"{int(round(st.progress * 100))}%"
                        acc_disp = st.account_name if st.account_name else "（无水印）"
                        self.tree.item(iid, values=(st.video_path.name, acc_disp, st.state, pct, st.message))
                elif kind == "api_test":
                    ok, msg = payload
                    self._append_log(msg)
                    if ok:
                        messagebox.showinfo("测试成功", msg)
                    else:
                        messagebox.showerror("测试失败", msg)
                elif kind == "callback":
                    payload()
                elif kind == "done":
                    results: list[TaskStatus] = payload
                    ok_n = sum(1 for r in results if r.state in ("已完成", "已跳过"))
                    fail_n = sum(1 for r in results if r.state == "失败")
                    self._append_log(f"=== 批量剪辑结束：成功/跳过 {ok_n} 条，失败 {fail_n} 条 ===")
                    self.btn_start.configure(state=tk.NORMAL)
                    self.btn_stop.configure(state=tk.DISABLED)
                    messagebox.showinfo("全部剪辑完成", f"批量剪辑已完成！\n成功：{ok_n} 条\n失败：{fail_n} 条")
        except queue.Empty:
            pass
        self.after(120, self._poll_ui_queue)


def launch_gui() -> None:
    app = AutoClipApp()
    app.mainloop()

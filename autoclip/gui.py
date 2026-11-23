"""桌面图形界面（Tkinter）：支持 DeepSeek API 配置、10-20+ 批量队列、硬件自适应并发与安全区预览。"""

from __future__ import annotations

import os
import platform
import queue
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk
from typing import Optional

from PIL import Image, ImageDraw, ImageTk

from . import __version__
from .config import AppConfig, load_config, save_config
from .hardware import HardwareProfile, detect_hardware
from .pipeline import BatchPipeline, TaskStatus
from .renderer import render_preview_frame
from .translator import SubtitleItem, test_deepseek_connection

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".flv", ".wmv", ".webm"}
ENHANCE_LABEL_TO_CODE = {
    "极速高清增强 (推荐批量)": "fast",
    "真·AI超清 (3060显卡/较慢)": "ai",
    "关闭": "off",
}
ENHANCE_CODE_TO_LABEL = {v: k for k, v in ENHANCE_LABEL_TO_CODE.items()}


class SubtitleReviewDialog(tk.Toplevel):
    """半自动校对模式弹窗：允许用户在压制前快速检查或修改中文字幕。"""

    def __init__(self, parent: tk.Tk, video_name: str, items: list[SubtitleItem]) -> None:
        super().__init__(parent)
        self.title(f"校对字幕 - {video_name}")
        self.geometry("760x560")
        self.transient(parent)
        self.grab_set()
        self.result: Optional[list[SubtitleItem]] = None
        self._orig_items = items

        tip = ttk.Label(
            self,
            text="格式：每行一条【序号 | 开始秒-结束秒 | 中文字幕】，可直接修改右侧文字后点击“确认并压制”。",
            foreground="#444444",
        )
        tip.pack(fill=tk.X, padx=12, pady=(10, 6))

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
        ttk.Button(btn_bar, text="取消本次任务", command=self._on_cancel).pack(side=tk.RIGHT, padx=6)
        ttk.Button(btn_bar, text="确认并继续压制", command=self._on_confirm).pack(side=tk.RIGHT, padx=6)
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def _on_confirm(self) -> None:
        edited: list[SubtitleItem] = []
        raw_lines = self.text.get("1.0", tk.END).strip().splitlines()
        for idx, line in enumerate(raw_lines, start=1):
            parts = line.split("|", 2)
            if len(parts) < 3:
                continue
            time_part = parts[1].strip()
            txt = parts[2].strip()
            if not txt:
                continue
            try:
                s_str, e_str = time_part.split("-")
                start, end = float(s_str.strip()), float(e_str.strip())
            except Exception:
                if idx <= len(self._orig_items):
                    start = self._orig_items[idx - 1].start
                    end = self._orig_items[idx - 1].end
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
        self.title(f"自动剪辑工具 AutoClip v{__version__}（中老年字幕 + 隐形去重）")
        self.geometry("1200x800")
        self.minsize(1060, 720)

        self.cfg: AppConfig = load_config()
        self.hw: HardwareProfile = detect_hardware()
        self.video_list: list[Path] = []
        self.tree_ids: dict[Path, str] = {}
        self.pipeline: Optional[BatchPipeline] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.ui_queue: queue.Queue = queue.Queue()
        self._preview_photo: Optional[ImageTk.PhotoImage] = None

        self._init_vars()
        self._build_ui()
        self.after(120, self._poll_ui_queue)

    def _init_vars(self) -> None:
        self.var_api_key = tk.StringVar(value=self.cfg.api_key)
        self.var_api_url = tk.StringVar(value=self.cfg.api_base_url)
        self.var_api_model = tk.StringVar(value=self.cfg.api_model)
        self.var_show_key = tk.BooleanVar(value=False)

        self.var_whisper_model = tk.StringVar(value=self.cfg.whisper_model)
        self.var_output_fmt = tk.StringVar(value=self.cfg.output_format.lower())
        self.var_concurrency = tk.StringVar(value=self.cfg.concurrency_mode)
        self.var_dedup = tk.StringVar(value=self.cfg.dedup_level)
        self.var_enhance = tk.StringVar(
            value=ENHANCE_CODE_TO_LABEL.get(self.cfg.enhance_mode, "极速高清增强 (推荐批量)")
        )
        self.var_output_dir = tk.StringVar(value=self.cfg.output_dir)

        self.var_feathered_mask = tk.BooleanVar(value=self.cfg.enable_feathered_mask)
        self.var_loudnorm = tk.BooleanVar(value=self.cfg.audio_loudnorm)
        self.var_review = tk.BooleanVar(value=self.cfg.review_subtitles)
        self.var_keep_srt = tk.BooleanVar(value=self.cfg.keep_srt)

        self.var_font_name = tk.StringVar(value=self.cfg.font_name)
        self.var_font_size = tk.IntVar(value=self.cfg.font_size)
        self.var_font_color = tk.StringVar(value=self.cfg.font_color)
        self.var_pos_y = tk.DoubleVar(value=self.cfg.position_y_pct)
        self.var_max_chars = tk.IntVar(value=self.cfg.max_chars_per_line)
        self.var_bitrate = tk.IntVar(value=self.cfg.video_bitrate_mbps)
        self.var_safe_zone = tk.BooleanVar(value=True)

    def _build_ui(self) -> None:
        # 1. 顶部硬件状态条
        hw_bar = ttk.Frame(self, padding=(12, 6))
        hw_bar.pack(fill=tk.X)
        ttk.Label(
            hw_bar,
            text=self.hw.summary_text(),
            foreground="#155724",
            font=("Microsoft YaHei", 9, "bold"),
        ).pack(side=tk.LEFT)

        # 2. DeepSeek API 配置区
        api_frame = ttk.LabelFrame(self, text="DeepSeek API 接口配置（填入后自动保存，打开即用）", padding=(10, 8))
        api_frame.pack(fill=tk.X, padx=12, pady=(2, 8))

        ttk.Label(api_frame, text="API Key:").grid(row=0, column=0, sticky=tk.W, padx=(0, 6))
        self.entry_key = ttk.Entry(api_frame, textvariable=self.var_api_key, width=38, show="*")
        self.entry_key.grid(row=0, column=1, sticky=tk.EW, padx=(0, 6))
        ttk.Checkbutton(
            api_frame,
            text="显示",
            variable=self.var_show_key,
            command=self._toggle_key_visibility,
        ).grid(row=0, column=2, padx=(0, 10))

        ttk.Label(api_frame, text="接口地址:").grid(row=0, column=3, sticky=tk.W, padx=(0, 4))
        ttk.Entry(api_frame, textvariable=self.var_api_url, width=26).grid(row=0, column=4, padx=(0, 10))

        ttk.Label(api_frame, text="模型:").grid(row=0, column=5, sticky=tk.W, padx=(0, 4))
        ttk.Entry(api_frame, textvariable=self.var_api_model, width=15).grid(row=0, column=6, padx=(0, 10))

        ttk.Button(api_frame, text="测试连接", command=self._on_test_api).grid(row=0, column=7, padx=4)
        ttk.Button(api_frame, text="保存配置", command=self._on_save_config).grid(row=0, column=8, padx=4)
        api_frame.columnconfigure(1, weight=1)

        # 3. 中部左右分栏
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        # 左侧：批量队列 + 日志
        left_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, weight=3)

        queue_toolbar = ttk.Frame(left_frame)
        queue_toolbar.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(queue_toolbar, text="＋ 批量导入视频", command=self._on_add_files).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(queue_toolbar, text="📁 导入文件夹", command=self._on_add_folder).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(queue_toolbar, text="移除选中", command=self._on_remove_selected).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(queue_toolbar, text="清空列表", command=self._on_clear_list).pack(side=tk.LEFT, padx=(0, 6))
        self.lbl_count = ttk.Label(queue_toolbar, text="已导入：0 个视频")
        self.lbl_count.pack(side=tk.RIGHT)

        tree_wrap = ttk.Frame(left_frame)
        tree_wrap.pack(fill=tk.BOTH, expand=True)
        cols = ("name", "state", "progress", "detail")
        self.tree = ttk.Treeview(tree_wrap, columns=cols, show="headings", height=13)
        self.tree.heading("name", text="视频文件名")
        self.tree.heading("state", text="当前状态")
        self.tree.heading("progress", text="进度")
        self.tree.heading("detail", text="详情 / 输出文件")
        self.tree.column("name", width=220, anchor=tk.W)
        self.tree.column("state", width=95, anchor=tk.CENTER)
        self.tree.column("progress", width=65, anchor=tk.CENTER)
        self.tree.column("detail", width=230, anchor=tk.W)
        t_scroll = ttk.Scrollbar(tree_wrap, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=t_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        t_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 运行控制按钮
        action_bar = ttk.Frame(left_frame, padding=(0, 8))
        action_bar.pack(fill=tk.X)
        self.btn_start = ttk.Button(action_bar, text="▶ 开始批量剪辑", command=self._on_start_batch)
        self.btn_start.pack(side=tk.LEFT, padx=(0, 8))
        self.btn_stop = ttk.Button(action_bar, text="■ 停止", command=self._on_stop_batch, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(action_bar, text="📂 打开输出目录", command=self._on_open_output_dir).pack(side=tk.LEFT, padx=(0, 8))

        # 日志框
        log_frame = ttk.LabelFrame(left_frame, text="运行日志", padding=4)
        log_frame.pack(fill=tk.BOTH, expand=False)
        self.log_text = tk.Text(log_frame, height=8, font=("Consolas", 9), state=tk.DISABLED)
        l_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=l_scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        l_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 右侧：剪辑参数 & 安全区预览
        right_frame = ttk.Frame(main_pane, padding=(8, 0, 0, 0))
        main_pane.add(right_frame, weight=2)

        # 参数选项卡
        nb = ttk.Notebook(right_frame)
        nb.pack(fill=tk.BOTH, expand=True)

        tab_basic = ttk.Frame(nb, padding=10)
        tab_sub = ttk.Frame(nb, padding=10)
        tab_preview = ttk.Frame(nb, padding=8)
        nb.add(tab_basic, text="输出与去重")
        nb.add(tab_sub, text="字幕样式")
        nb.add(tab_preview, text="效果预览")

        # --- Tab 1: 输出与去重 ---
        r = 0
        ttk.Label(tab_basic, text="输出视频格式:").grid(row=r, column=0, sticky=tk.W, pady=5)
        fmt_box = ttk.Frame(tab_basic)
        fmt_box.grid(row=r, column=1, sticky=tk.W, pady=5)
        ttk.Radiobutton(fmt_box, text="MP4（推荐，兼容性最好）", variable=self.var_output_fmt, value="mp4").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(fmt_box, text="MOV", variable=self.var_output_fmt, value="mov").pack(side=tk.LEFT)

        r += 1
        ttk.Label(tab_basic, text="批量并发策略:").grid(row=r, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(
            tab_basic,
            textvariable=self.var_concurrency,
            values=["auto", "1", "2", "3", "4"],
            state="readonly",
            width=18,
        ).grid(row=r, column=1, sticky=tk.W, pady=5)
        ttk.Label(tab_basic, text="(auto=按3060自动并行)", foreground="#666").grid(row=r, column=2, sticky=tk.W, padx=4)

        r += 1
        ttk.Label(tab_basic, text="语音识别模型:").grid(row=r, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(
            tab_basic,
            textvariable=self.var_whisper_model,
            values=["large-v3-turbo", "large-v3", "medium", "small"],
            state="readonly",
            width=18,
        ).grid(row=r, column=1, sticky=tk.W, pady=5)

        r += 1
        ttk.Label(tab_basic, text="隐形去重强度:").grid(row=r, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(
            tab_basic,
            textvariable=self.var_dedup,
            values=["standard", "light", "off"],
            state="readonly",
            width=18,
        ).grid(row=r, column=1, sticky=tk.W, pady=5)

        r += 1
        ttk.Label(tab_basic, text="画质超清增强:").grid(row=r, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(
            tab_basic,
            textvariable=self.var_enhance,
            values=list(ENHANCE_LABEL_TO_CODE.keys()),
            state="readonly",
            width=24,
        ).grid(row=r, column=1, columnspan=2, sticky=tk.W, pady=5)

        r += 1
        ttk.Label(tab_basic, text="输出码率(Mbps):").grid(row=r, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(tab_basic, from_=4, to=30, textvariable=self.var_bitrate, width=10).grid(row=r, column=1, sticky=tk.W, pady=5)

        r += 1
        ttk.Label(tab_basic, text="输出文件夹:").grid(row=r, column=0, sticky=tk.W, pady=5)
        out_row = ttk.Frame(tab_basic)
        out_row.grid(row=r, column=1, columnspan=2, sticky=tk.EW, pady=5)
        ttk.Entry(out_row, textvariable=self.var_output_dir, width=22).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(out_row, text="选择", width=6, command=self._on_choose_output_dir).pack(side=tk.LEFT, padx=(4, 0))

        r += 1
        ttk.Checkbutton(
            tab_basic, text="开启顶部 30% 升级版羽化隐形蒙版（消除硬切线）", variable=self.var_feathered_mask
        ).grid(row=r, column=0, columnspan=3, sticky=tk.W, pady=4)
        r += 1
        ttk.Checkbutton(
            tab_basic, text="音频响度均衡 + 防爆音（清晰适合老年人）", variable=self.var_loudnorm
        ).grid(row=r, column=0, columnspan=3, sticky=tk.W, pady=4)
        r += 1
        ttk.Checkbutton(
            tab_basic, text="压制前暂停弹窗校对字幕（半自动微调）", variable=self.var_review
        ).grid(row=r, column=0, columnspan=3, sticky=tk.W, pady=4)
        r += 1
        ttk.Checkbutton(
            tab_basic, text="额外导出 SRT 字幕文件", variable=self.var_keep_srt
        ).grid(row=r, column=0, columnspan=3, sticky=tk.W, pady=4)
        tab_basic.columnconfigure(1, weight=1)

        # --- Tab 2: 字幕样式 ---
        r = 0
        ttk.Label(tab_sub, text="字体名称:").grid(row=r, column=0, sticky=tk.W, pady=6)
        ttk.Entry(tab_sub, textvariable=self.var_font_name, width=22).grid(row=r, column=1, sticky=tk.W, pady=6)

        r += 1
        ttk.Label(tab_sub, text="字号(1080×1440基准):").grid(row=r, column=0, sticky=tk.W, pady=6)
        ttk.Spinbox(tab_sub, from_=36, to=120, textvariable=self.var_font_size, width=10).grid(row=r, column=1, sticky=tk.W, pady=6)

        r += 1
        ttk.Label(tab_sub, text="字幕颜色:").grid(row=r, column=0, sticky=tk.W, pady=6)
        color_row = ttk.Frame(tab_sub)
        color_row.grid(row=r, column=1, sticky=tk.W, pady=6)
        ttk.Entry(color_row, textvariable=self.var_font_color, width=10).pack(side=tk.LEFT)
        ttk.Button(color_row, text="选色", width=6, command=self._on_pick_color).pack(side=tk.LEFT, padx=4)

        r += 1
        ttk.Label(tab_sub, text="垂直位置(距顶部%):").grid(row=r, column=0, sticky=tk.W, pady=6)
        pos_row = ttk.Frame(tab_sub)
        pos_row.grid(row=r, column=1, sticky=tk.EW, pady=6)
        ttk.Scale(pos_row, from_=60.0, to=92.0, variable=self.var_pos_y, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lbl_pos_y = ttk.Label(pos_row, text=f"{self.var_pos_y.get():.1f}%", width=6)
        self.lbl_pos_y.pack(side=tk.LEFT, padx=4)
        self.var_pos_y.trace_add("write", lambda *_: self.lbl_pos_y.config(text=f"{self.var_pos_y.get():.1f}%"))

        r += 1
        ttk.Label(tab_sub, text="单行最多汉字数:").grid(row=r, column=0, sticky=tk.W, pady=6)
        ttk.Spinbox(tab_sub, from_=8, to=20, textvariable=self.var_max_chars, width=10).grid(row=r, column=1, sticky=tk.W, pady=6)

        ttk.Label(
            tab_sub,
            text="默认样式已精确匹配您的剪辑成品：\n亮黄 #F4EA46 + 粗黑描边阴影 + 居中单行（84%高度）",
            foreground="#555555",
        ).grid(row=r + 1, column=0, columnspan=2, sticky=tk.W, pady=(14, 4))

        # --- Tab 3: 效果与安全区预览 ---
        prev_ctrl = ttk.Frame(tab_preview)
        prev_ctrl.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(prev_ctrl, text="刷新字幕预览图", command=self._on_render_preview).pack(side=tk.LEFT)
        ttk.Checkbutton(
            prev_ctrl, text="显示抖音/视频号底部安全区红线", variable=self.var_safe_zone, command=self._on_render_preview
        ).pack(side=tk.LEFT, padx=8)

        self.preview_label = ttk.Label(
            tab_preview,
            text="请先在左侧导入视频，选中一条后点击“刷新字幕预览图”",
            anchor=tk.CENTER,
            background="#1e1e1e",
            foreground="#cccccc",
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True)

    # ---------- 事件处理 ----------
    def _toggle_key_visibility(self) -> None:
        self.entry_key.configure(show="" if self.var_show_key.get() else "*")

    def _collect_config(self) -> AppConfig:
        self.cfg.api_key = self.var_api_key.get().strip()
        self.cfg.api_base_url = self.var_api_url.get().strip() or "https://api.deepseek.com"
        self.cfg.api_model = self.var_api_model.get().strip() or "deepseek-chat"
        self.cfg.whisper_model = self.var_whisper_model.get()
        self.cfg.output_format = self.var_output_fmt.get().lower()
        self.cfg.concurrency_mode = self.var_concurrency.get()
        self.cfg.dedup_level = self.var_dedup.get()
        self.cfg.enhance_mode = ENHANCE_LABEL_TO_CODE.get(self.var_enhance.get(), "fast")
        self.cfg.output_dir = self.var_output_dir.get().strip()
        self.cfg.enable_feathered_mask = self.var_feathered_mask.get()
        self.cfg.audio_loudnorm = self.var_loudnorm.get()
        self.cfg.review_subtitles = self.var_review.get()
        self.cfg.keep_srt = self.var_keep_srt.get()
        self.cfg.font_name = self.var_font_name.get().strip() or self.cfg.font_name
        self.cfg.font_size = int(self.var_font_size.get())
        self.cfg.font_color = self.var_font_color.get().strip() or "#F4EA46"
        self.cfg.position_y_pct = round(float(self.var_pos_y.get()), 1)
        self.cfg.max_chars_per_line = int(self.var_max_chars.get())
        self.cfg.video_bitrate_mbps = int(self.var_bitrate.get())
        return self.cfg

    def _on_save_config(self) -> None:
        cfg = self._collect_config()
        p = save_config(cfg)
        self._append_log(f"配置已保存到：{p}")
        messagebox.showinfo("保存成功", "配置已保存，下次打开自动加载。")

    def _on_test_api(self) -> None:
        cfg = self._collect_config()
        save_config(cfg)
        self._append_log("正在测试 DeepSeek API 连接...")

        def worker() -> None:
            ok, msg = test_deepseek_connection(cfg)
            self.ui_queue.put(("api_test", (ok, msg)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_pick_color(self) -> None:
        _, hex_c = colorchooser.askcolor(initialcolor=self.var_font_color.get(), title="选择字幕颜色")
        if hex_c:
            self.var_font_color.set(hex_c.upper())

    def _on_choose_output_dir(self) -> None:
        d = filedialog.askdirectory(title="选择成片输出文件夹")
        if d:
            self.var_output_dir.set(d)

    def _add_video_paths(self, paths: list[Path]) -> None:
        added = 0
        for p in paths:
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS and p not in self.tree_ids:
                self.video_list.append(p)
                iid = self.tree.insert("", tk.END, values=(p.name, "等待中", "0%", str(p.parent)))
                self.tree_ids[p] = iid
                added += 1
        self.lbl_count.config(text=f"已导入：{len(self.video_list)} 个视频")
        if added:
            self._append_log(f"已添加 {added} 个视频素材（当前共 {len(self.video_list)} 个）。")

    def _on_add_files(self) -> None:
        files = filedialog.askopenfilenames(
            title="选择要剪辑的视频（可按 Ctrl/Shift 多选）",
            filetypes=[("视频文件", "*.mp4 *.mov *.mkv *.avi *.m4v *.flv"), ("所有文件", "*.*")],
        )
        if files:
            self._add_video_paths([Path(f) for f in files])

    def _on_add_folder(self) -> None:
        folder = filedialog.askdirectory(title="选择包含视频素材的文件夹")
        if not folder:
            return
        found = sorted(
            [p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
        )
        if not found:
            messagebox.showwarning("未找到视频", "所选文件夹中没有支持的视频文件。")
            return
        self._add_video_paths(found)

    def _on_remove_selected(self) -> None:
        for iid in self.tree.selection():
            target = None
            for p, tid in self.tree_ids.items():
                if tid == iid:
                    target = p
                    break
            if target:
                self.video_list.remove(target)
                del self.tree_ids[target]
            self.tree.delete(iid)
        self.lbl_count.config(text=f"已导入：{len(self.video_list)} 个视频")

    def _on_clear_list(self) -> None:
        self.video_list.clear()
        self.tree_ids.clear()
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        self.lbl_count.config(text="已导入：0 个视频")

    def _on_render_preview(self) -> None:
        sel = self.tree.selection()
        target_video: Optional[Path] = None
        if sel:
            for p, tid in self.tree_ids.items():
                if tid == sel[0]:
                    target_video = p
                    break
        elif self.video_list:
            target_video = self.video_list[0]

        if not target_video:
            messagebox.showinfo("提示", "请先在左侧导入至少一条视频。")
            return

        cfg = self._collect_config()
        try:
            img_path = render_preview_frame(target_video, cfg)
            im = Image.open(img_path).convert("RGB")
            if self.var_safe_zone.get():
                draw = ImageDraw.Draw(im)
                w, h = im.size
                # 抖音/视频号底部 18% 安全区红线提示
                safe_y = int(h * 0.82)
                for x in range(0, w, 24):
                    draw.line([(x, safe_y), (min(x + 14, w), safe_y)], fill=(255, 70, 70), width=4)
                # 顶部 30% 羽化蒙版中心线提示（淡蓝色虚线）
                if cfg.enable_feathered_mask:
                    mask_y = int(h * (cfg.mask_height_pct / 100.0))
                    for x in range(0, w, 24):
                        draw.line([(x, mask_y), (min(x + 12, w), mask_y)], fill=(80, 180, 255), width=3)

            im.thumbnail((360, 500), Image.LANCZOS)
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
            messagebox.showwarning("请导入视频", "请先批量导入至少一个原视频文件。")
            return
        cfg = self._collect_config()
        if not cfg.api_key:
            messagebox.showwarning("缺少 API Key", "请先在顶部填入您的 DeepSeek API Key。")
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
                        self.tree.item(iid, values=(st.video_path.name, st.state, pct, st.message))
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
                    ok_n = sum(1 for r in results if r.state == "已完成")
                    fail_n = sum(1 for r in results if r.state == "失败")
                    self._append_log(f"=== 批量任务结束：成功 {ok_n} 个，失败 {fail_n} 个 ===")
                    self.btn_start.configure(state=tk.NORMAL)
                    self.btn_stop.configure(state=tk.DISABLED)
                    messagebox.showinfo("批量剪辑完成", f"全部处理完成！\n成功：{ok_n} 个\n失败：{fail_n} 个")
        except queue.Empty:
            pass
        self.after(120, self._poll_ui_queue)


def launch_gui() -> None:
    app = AutoClipApp()
    app.mainloop()

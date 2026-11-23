"""配置管理模块：负责读写 config.json（包含 DeepSeek API、字幕样式、去重和输出参数）。"""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def get_app_dir() -> Path:
    """获取程序所在根目录（兼容 PyInstaller 打包后的 EXE 与源码运行）。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def get_resource_dir() -> Path:
    """获取内置资源目录（兼容 PyInstaller _MEIPASS）。"""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return get_app_dir()


def get_default_font_name() -> str:
    system = platform.system()
    if system == "Windows":
        return "Microsoft YaHei"
    if system == "Darwin":
        return "PingFang SC"
    return "Noto Sans CJK SC"


@dataclass
class AppConfig:
    # DeepSeek / OpenAI 兼容 API 配置
    api_key: str = ""
    api_base_url: str = "https://api.deepseek.com"
    api_model: str = "deepseek-chat"
    api_timeout: int = 120

    # Whisper 语音识别配置
    whisper_model: str = "large-v3-turbo"
    source_language: str = "en"

    # 字幕样式配置（默认精准复刻用户剪辑样本：亮黄大字、粗黑描边阴影、单行居中）
    font_name: str = ""
    font_size: int = 82  # 基准 1080x1440 分辨率下的字号（实测中文字高约 69px，精准匹配样本）
    font_color: str = "#F4EA46"  # 样本实测亮黄色
    outline_color: str = "#000000"
    outline_width: float = 4.2
    shadow_depth: float = 3.5
    bold: bool = True
    position_y_pct: float = 84.0  # 字幕中心距画面顶部的百分比（样本实测 84%）
    max_chars_per_line: int = 14  # 单行最大汉字数，防止折行
    min_chars_per_line: int = 4

    # 去重与蒙版配置
    enable_feathered_mask: bool = True
    mask_height_pct: float = 30.0  # 顶部 30% 羽化线性蒙版
    mask_feather_px: int = 72  # 羽化过渡带宽度（像素），消除肉眼可见横缝
    mask_shift_px: int = 4  # 蒙版区域轻微水平位移
    dedup_level: str = "standard"  # off / light / standard

    # 画质超清增强（替代剪映 SVIP 一键超清）：off / fast（极速高清增强） / ai（Real-ESRGAN 真AI超清）
    enhance_mode: str = "fast"

    # 音频处理
    audio_loudnorm: bool = True  # EBU R128 响度均衡 + 防爆音（清晰适合老年人）

    # 输出配置
    output_format: str = "mp4"  # mp4 / mov
    output_fps: int = 60
    video_bitrate_mbps: int = 10
    output_dir: str = ""  # 为空时默认输出到素材同目录的“已剪辑输出”文件夹

    # 批量与工作流配置
    concurrency_mode: str = "auto"  # auto / 1 / 2 / 3 / 4
    review_subtitles: bool = False  # 是否在压制前暂停校对字幕
    keep_srt: bool = False  # 是否顺便保留 .srt 字幕文件

    def __post_init__(self) -> None:
        if not self.font_name:
            self.font_name = get_default_font_name()


def get_config_path() -> Path:
    app_dir = get_app_dir()
    candidate = app_dir / "config.json"
    try:
        # 测试目录是否可写，若不可写则回退到用户目录
        if candidate.exists():
            if os.access(candidate, os.W_OK):
                return candidate
        elif os.access(app_dir, os.W_OK):
            return candidate
    except OSError:
        pass
    fallback_dir = Path.home() / ".autoclip"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    return fallback_dir / "config.json"


def load_config() -> AppConfig:
    cfg_path = get_config_path()
    if not cfg_path.exists():
        cfg = AppConfig()
        save_config(cfg)
        return cfg
    try:
        raw: dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8"))
        valid_fields = set(AppConfig.__dataclass_fields__.keys())
        filtered = {k: v for k, v in raw.items() if k in valid_fields}
        return AppConfig(**filtered)
    except Exception:
        return AppConfig()


def save_config(cfg: AppConfig) -> Path:
    cfg_path = get_config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return cfg_path

"""配置管理模块：负责读写 config.json（DeepSeek、多账号飘动水印、字幕样式、无痕蒙版与输出参数）。"""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def get_app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def get_resource_dir() -> Path:
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

    # 识别方式：ocr_first（优先识别画面英文字幕，100%对齐原英文字幕时长） / whisper（纯语音识别）
    subtitle_source: str = "ocr_first"
    whisper_model: str = "large-v3-turbo"
    source_language: str = "en"

    # 字幕样式配置（亮黄大字、粗黑描边阴影、单行居中）
    font_name: str = ""
    font_size: int = 82
    font_color: str = "#F4EA46"
    outline_color: str = "#000000"
    outline_width: float = 4.2
    shadow_depth: float = 3.5
    bold: bool = True
    auto_position_below_en: bool = True  # 自动紧贴画面原英文字幕正下方（预留单/双行间距不重叠）
    position_y_pct: float = 84.0
    max_chars_per_line: int = 14
    min_chars_per_line: int = 4

    # 多账号飘动水印配置
    # accounts 格式: [{"name": "老张爱看剧", "count": 3}, {"name": "李阿姨追剧", "count": 6}]
    watermark_enabled: bool = True
    watermark_floating: bool = True  # 全屏柔和飘动（避开字幕区）
    watermark_opacity_pct: int = 45  # 透明度 15~85%（默认45%半透明白字）
    watermark_font_size: int = 42  # 1080x1440 基准下的水印字号（与字幕同字体）
    watermark_accounts: list[dict[str, Any]] = field(default_factory=list)
    group_output_by_account: bool = True  # 按账号自动分文件夹导出

    # 去重与无痕羽化蒙版配置
    enable_feathered_mask: bool = True
    mask_height_pct: float = 22.0  # 移至画面上部 22%（避开人脸）+ 160px 超宽羽化，彻底消除切缝感
    mask_feather_px: int = 160
    mask_shift_px: int = 12  # 对应手动偏移效果，配合超宽羽化做到肉眼无感
    dedup_level: str = "deep"  # off / standard / deep（深度隐形去重：微变速+掐头去尾+微噪点+声纹重写+随机GOP）

    # 画质超清增强：off / fast（极速高清增强） / ai（Real-ESRGAN 真AI超清）
    enhance_mode: str = "fast"

    # 音频处理
    audio_loudnorm: bool = True

    # 输出配置
    output_format: str = "mp4"
    output_fps: int = 30  # 默认保持原片 30fps（画质一致、CPU/导出耗时直接砍半，界面可切 60fps）
    video_bitrate_mbps: int = 10
    output_dir: str = ""

    # 批量与工作流配置
    concurrency_mode: str = "auto"
    review_subtitles: bool = False
    keep_srt: bool = False
    skip_existing: bool = True  # 已导出过的视频自动跳过，防重复扣费
    generate_copywriting: bool = True  # 剪辑完成后自动调用 DeepSeek 生成微信视频号发布文案

    def __post_init__(self) -> None:
        if not self.font_name:
            self.font_name = get_default_font_name()
        if not isinstance(self.watermark_accounts, list):
            self.watermark_accounts = []


def get_config_path() -> Path:
    app_dir = get_app_dir()
    candidate = app_dir / "config.json"
    try:
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


def assign_accounts_to_videos(
    video_count: int,
    accounts: list[dict[str, Any]],
    enabled: bool = True,
) -> list[str]:
    """按用户设定的账号配额（如 A账号3条、B账号6条、C账号5条）顺序分配给每个视频。"""
    if not enabled or not accounts or video_count <= 0:
        return [""] * video_count

    assigned: list[str] = []
    for acc in accounts:
        name = str(acc.get("name", "")).strip()
        try:
            cnt = max(0, int(acc.get("count", 0)))
        except Exception:
            cnt = 0
        if name and cnt > 0:
            assigned.extend([name] * cnt)

    if len(assigned) < video_count:
        assigned.extend([""] * (video_count - len(assigned)))
    return assigned[:video_count]

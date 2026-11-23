"""字幕格式生成模块：生成精准排版（亮黄大字、粗黑描边阴影、单行居中）的 ASS 与 SRT 字幕。"""

from __future__ import annotations

from pathlib import Path

from .config import AppConfig
from .translator import SubtitleItem


def hex_to_ass_color(hex_color: str, alpha: int = 0) -> str:
    """将 #RRGGBB 转换为 ASS 颜色格式 &HAABBGGRR。"""
    s = hex_color.strip().lstrip("#")
    if len(s) != 6:
        s = "F4EA46"
    r = s[0:2].upper()
    g = s[2:4].upper()
    b = s[4:6].upper()
    a = f"{max(0, min(255, alpha)):02X}"
    return f"&H{a}{b}{g}{r}"


def _format_ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    total_cs = int(round(seconds * 100))
    cs = total_cs % 100
    total_s = total_cs // 100
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _format_srt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _escape_ass_text(text: str) -> str:
    # 严禁换行，保证一次只呈现一段单行字幕
    text = text.replace("\r", " ").replace("\n", " ").strip()
    text = text.replace("{", "｛").replace("}", "｝")
    return text


def write_ass_subtitle(
    items: list[SubtitleItem],
    out_path: Path,
    video_width: int,
    video_height: int,
    cfg: AppConfig,
) -> Path:
    """生成精准匹配原成片视觉风格的 ASS 字幕文件。"""
    scale = video_height / 1440.0
    font_size = max(24, int(round(cfg.font_size * scale)))
    outline = round(max(1.5, cfg.outline_width * scale), 2)
    shadow = round(max(1.0, cfg.shadow_depth * scale), 2)

    primary_color = hex_to_ass_color(cfg.font_color, alpha=0)
    outline_color = hex_to_ass_color(cfg.outline_color, alpha=0)
    # 柔和深黑阴影（提升浅色背景上的可读性，便于老年人观看）
    back_color = "&H64000000"
    bold_flag = -1 if cfg.bold else 0

    cx = int(round(video_width / 2.0))
    cy = int(round(video_height * (cfg.position_y_pct / 100.0)))

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Senior,{cfg.font_name},{font_size},{primary_color},&H000000FF,{outline_color},{back_color},{bold_flag},0,0,0,100,100,1,0,1,{outline},{shadow},5,24,24,24,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    for it in items:
        start_s = _format_ass_time(it.start)
        end_s = _format_ass_time(it.end)
        txt = _escape_ass_text(it.text)
        if not txt:
            continue
        # 防超宽溢出：估算该行像素宽度，若超过画面宽度的 92% 自动微调字号，保证完整居中不裁切
        vis_chars = sum(1.0 if "\u4e00" <= ch <= "\u9fff" else 0.58 for ch in txt)
        est_px = vis_chars * font_size * 0.96
        max_px = video_width * 0.92
        fs_override = ""
        if est_px > max_px and est_px > 0:
            fit_fs = max(24, int(font_size * (max_px / est_px)))
            fs_override = f"\\fs{fit_fs}"
        lines.append(
            f"Dialogue: 0,{start_s},{end_s},Senior,,0,0,0,,{{\\pos({cx},{cy})\\blur0.8{fs_override}}}{txt}\n"
        )

    out_path.write_text("".join(lines), encoding="utf-8-sig")
    return out_path


def write_srt_subtitle(items: list[SubtitleItem], out_path: Path) -> Path:
    """导出标准 SRT 字幕文件（方便留存或导入其他剪辑软件）。"""
    blocks: list[str] = []
    for idx, it in enumerate(items, start=1):
        blocks.append(
            f"{idx}\n{_format_srt_time(it.start)} --> {_format_srt_time(it.end)}\n{it.text.strip()}\n"
        )
    out_path.write_text("\n".join(blocks), encoding="utf-8-sig")
    return out_path

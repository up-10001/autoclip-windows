"""字幕与飘动水印 ASS/SRT 生成模块。

- 中文字幕：亮黄 #F4EA46 + 粗黑描边阴影 + 单行居中 + 超宽自动微缩防裁切；
- 账号水印：与字幕完全相同的字体、半透明白色 + 柔和暗影、在画面中上部柔和全屏飘动（绝不遮挡底部中英文字幕）。
"""

from __future__ import annotations

import random
from pathlib import Path

from .config import AppConfig
from .translator import SubtitleItem


def hex_to_ass_color(hex_color: str, alpha: int = 0) -> str:
    """将 #RRGGBB 转换为 ASS 颜色格式 &HAABBGGRR（alpha: 0=完全不透明, 255=完全透明）。"""
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
    text = text.replace("\r", " ").replace("\n", " ").strip()
    text = text.replace("{", "｛").replace("}", "｝")
    return text


def _build_floating_watermark_events(
    wm_text: str,
    video_width: int,
    video_height: int,
    duration: float,
    wm_font_size: int,
    floating: bool,
) -> list[str]:
    """生成半透明白色飘动水印 ASS 事件（Layer 1，位于字幕区上方柔和漂移）。"""
    txt = _escape_ass_text(wm_text)
    if not txt or duration <= 0.1:
        return []

    vis_chars = sum(1.0 if "\u4e00" <= ch <= "\u9fff" else 0.6 for ch in txt)
    half_w = int(max(60, vis_chars * wm_font_size * 0.52))
    margin_x = min(video_width // 3, half_w + int(video_width * 0.04))
    min_x = margin_x
    max_x = max(min_x + 10, video_width - margin_x)
    # 垂直飘动范围：顶部 8% ~ 中部 54%（完全避开 58% 以下的中英文字幕区）
    min_y = int(video_height * 0.08)
    max_y = int(video_height * 0.54)

    if not floating:
        rx = max_x
        ry = min_y
        return [
            f"Dialogue: 1,{_format_ass_time(0.0)},{_format_ass_time(duration)},Watermark,,0,0,0,,{{\\pos({rx},{ry})\\blur0.6}}{txt}\n"
        ]

    # 生成全屏柔和飘动轨迹（每段 7.5 秒平滑滑向画面不同区域，边缘自然折返）
    rng = random.Random(hash(txt) & 0xFFFFFFFF)
    waypoints = [
        (min_x, min_y),
        (max_x, int(min_y + (max_y - min_y) * 0.35)),
        (int((min_x + max_x) * 0.30), max_y),
        (max_x, min_y),
        (min_x, int(min_y + (max_y - min_y) * 0.75)),
        (int((min_x + max_x) * 0.72), max_y),
        (min_x, int(min_y + (max_y - min_y) * 0.25)),
        (max_x, int(min_y + (max_y - min_y) * 0.80)),
    ]
    # 随机挑选起点，让不同账号/不同视频的飘动轨迹更自然
    start_idx = rng.randrange(len(waypoints))
    ordered = waypoints[start_idx:] + waypoints[:start_idx]

    seg_dur = 7.5
    events: list[str] = []
    t = 0.0
    step = 0
    while t < duration:
        t_next = min(duration, t + seg_dur)
        x1, y1 = ordered[step % len(ordered)]
        x2, y2 = ordered[(step + 1) % len(ordered)]
        # 若最后一段不足 7.5 秒，按比例插值终点保持匀速漂移
        frac = (t_next - t) / seg_dur
        if frac < 0.99:
            x2 = int(round(x1 + (x2 - x1) * frac))
            y2 = int(round(y1 + (y2 - y1) * frac))
        events.append(
            f"Dialogue: 1,{_format_ass_time(t)},{_format_ass_time(t_next)},Watermark,,0,0,0,,"
            f"{{\\move({x1},{y1},{x2},{y2})\\blur0.6}}{txt}\n"
        )
        t = t_next
        step += 1
    return events


def write_ass_subtitle(
    items: list[SubtitleItem],
    out_path: Path,
    video_width: int,
    video_height: int,
    cfg: AppConfig,
    watermark_text: str = "",
    video_duration: float = 0.0,
) -> Path:
    """生成精准匹配原成片视觉风格的 ASS 字幕文件（含可选同字体半透明飘动水印）。"""
    scale = video_height / 1440.0
    font_size = max(24, int(round(cfg.font_size * scale)))
    outline = round(max(1.5, cfg.outline_width * scale), 2)
    shadow = round(max(1.0, cfg.shadow_depth * scale), 2)

    primary_color = hex_to_ass_color(cfg.font_color, alpha=0)
    outline_color = hex_to_ass_color(cfg.outline_color, alpha=0)
    back_color = "&H64000000"
    bold_flag = -1 if cfg.bold else 0

    # 水印样式：与字幕同字体，半透明白色 + 柔和暗影
    wm_font_size = max(20, int(round(cfg.watermark_font_size * scale)))
    opacity = max(15, min(90, int(cfg.watermark_opacity_pct)))
    wm_alpha = int(round(255 * (1.0 - opacity / 100.0)))
    wm_outline_alpha = min(245, wm_alpha + 65)
    wm_primary = hex_to_ass_color("#FFFFFF", alpha=wm_alpha)
    wm_outline = hex_to_ass_color("#000000", alpha=wm_outline_alpha)
    wm_back = hex_to_ass_color("#000000", alpha=min(250, wm_alpha + 85))

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
Style: Watermark,{cfg.font_name},{wm_font_size},{wm_primary},&H000000FF,{wm_outline},{wm_back},{bold_flag},0,0,0,100,100,1,0,1,1.6,1.2,5,20,20,20,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]

    # 1. 中文字幕层（Layer 0）
    max_end = video_duration
    for it in items:
        start_s = _format_ass_time(it.start)
        end_s = _format_ass_time(it.end)
        max_end = max(max_end, it.end)
        txt = _escape_ass_text(it.text)
        if not txt:
            continue
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

    # 2. 半透明白色飘动水印层（Layer 1）
    if cfg.watermark_enabled and watermark_text.strip():
        lines.extend(
            _build_floating_watermark_events(
                watermark_text.strip(),
                video_width,
                video_height,
                max(10.0, max_end),
                wm_font_size,
                cfg.watermark_floating,
            )
        )

    out_path.write_text("".join(lines), encoding="utf-8-sig")
    return out_path


def write_srt_subtitle(items: list[SubtitleItem], out_path: Path) -> Path:
    blocks: list[str] = []
    for idx, it in enumerate(items, start=1):
        blocks.append(
            f"{idx}\n{_format_srt_time(it.start)} --> {_format_srt_time(it.end)}\n{it.text.strip()}\n"
        )
    out_path.write_text("\n".join(blocks), encoding="utf-8-sig")
    return out_path

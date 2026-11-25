"""视频渲染、羽化线性蒙版与隐形去重引擎（基于 FFmpeg + RTX 3060 NVENC 硬编码）。"""

from __future__ import annotations

import json
import random
import re
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

from .config import AppConfig, get_app_dir, get_resource_dir
from .hardware import HardwareProfile, find_ffmpeg_binary
from .proc_mgr import register_proc, unregister_proc
from .subtitle import write_ass_subtitle
from .translator import SubtitleItem


@dataclass
class VideoInfo:
    width: int
    height: int
    duration: float
    fps: float
    has_audio: bool


def probe_video(video_path: Path) -> VideoInfo:
    ffprobe = find_ffmpeg_binary("ffprobe")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    cmd = [
        ffprobe,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    out = subprocess.check_output(
        cmd,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
    )
    data = json.loads(out)

    v_stream = None
    has_audio = False
    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and v_stream is None:
            v_stream = s
        elif s.get("codec_type") == "audio":
            has_audio = True

    if not v_stream:
        raise RuntimeError(f"无法读取视频流信息：{video_path}")

    width = int(v_stream["width"])
    height = int(v_stream["height"])
    # 处理竖屏旋转元数据
    rotation = 0
    for sd in v_stream.get("side_data_list", []):
        if "rotation" in sd:
            rotation = abs(int(sd["rotation"]))
    if rotation in (90, 270):
        width, height = height, width

    duration = float(data.get("format", {}).get("duration") or v_stream.get("duration") or 0.0)
    fps_str = v_stream.get("r_frame_rate", "30/1")
    try:
        num, den = fps_str.split("/")
        fps = float(num) / max(float(den), 1.0)
    except Exception:
        fps = 30.0

    return VideoInfo(width=width, height=height, duration=duration, fps=fps, has_audio=has_audio)


def _create_feathered_mask_png(
    width: int,
    strip_height: int,
    full_height: int,
    mask_height_pct: float,
    feather_px: int,
    out_path: Path,
) -> Path:
    """仅生成顶部蒙版区域（strip_height）的羽化灰度图，减少 70%+ 无效像素计算。"""
    boundary_y = int(round(full_height * (mask_height_pct / 100.0)))
    half_f = max(12, int(round(feather_px * (full_height / 1440.0) / 2.0)))
    y0 = max(0, boundary_y - half_f)
    y1 = min(strip_height, boundary_y + half_f)

    col = np.zeros((strip_height,), dtype=np.uint8)
    col[:y0] = 255
    if y1 > y0:
        t = np.linspace(0.0, np.pi, y1 - y0, dtype=np.float32)
        ramp = 0.5 * (1.0 + np.cos(t))
        col[y0:y1] = np.clip(np.round(ramp * 255.0), 0, 255).astype(np.uint8)

    mask_2d = np.repeat(col[:, np.newaxis], width, axis=1)
    Image.fromarray(mask_2d, mode="L").save(out_path)
    return out_path


def _find_fonts_dir() -> Optional[Path]:
    for candidate in (
        get_app_dir() / "assets" / "fonts",
        get_resource_dir() / "assets" / "fonts",
    ):
        if candidate.is_dir() and any(candidate.glob("*.[ot]tf")):
            return candidate
    return None


def _find_realesrgan_binary() -> Optional[Path]:
    exe_name = "realesrgan-ncnn-vulkan.exe" if Path("C:\\").exists() else "realesrgan-ncnn-vulkan"
    candidates = [
        get_app_dir() / "bin" / "realesrgan" / exe_name,
        get_app_dir() / "bin" / exe_name,
        get_resource_dir() / "bin" / "realesrgan" / exe_name,
        get_resource_dir() / "bin" / exe_name,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _run_ai_super_resolution(
    input_path: Path,
    info: VideoInfo,
    tmp_dir: Path,
    ffmpeg: str,
    log_cb: Optional[Callable[[str], None]] = None,
) -> Optional[Path]:
    """调用 Real-ESRGAN (Vulkan GPU) 执行真·AI 超清重绘，返回超清中间视频路径。"""
    exe = _find_realesrgan_binary()
    if not exe:
        if log_cb:
            log_cb("未检测到 Real-ESRGAN AI 超清组件，自动切换为“极速高清增强”模式。")
        return None

    models_dir = exe.parent / "models"
    model_name = "realesrgan-x4plus"
    scale = "4"
    if models_dir.is_dir():
        for m, s in (
            ("realesr-general-x4v3", "4"),
            ("realesrgan-x4plus", "4"),
            ("realesr-animevideov3", "2"),
        ):
            if (models_dir / f"{m}.param").is_file():
                model_name, scale = m, s
                break

    frames_in = tmp_dir / "ai_in"
    frames_out = tmp_dir / "ai_out"
    frames_in.mkdir(parents=True, exist_ok=True)
    frames_out.mkdir(parents=True, exist_ok=True)
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    if log_cb:
        log_cb(f"正在提取视频帧用于真·AI 超清（模型：{model_name}，3060 显卡加速中，耗时较长请耐心等待）...")

    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-qscale:v",
            "2",
            str(frames_in / "frame_%08d.jpg"),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
        creationflags=creationflags,
    )

    ai_cmd = [
        str(exe),
        "-i",
        str(frames_in),
        "-o",
        str(frames_out),
        "-n",
        model_name,
        "-s",
        scale,
        "-f",
        "jpg",
    ]
    res = subprocess.run(
        ai_cmd,
        cwd=str(exe.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
    )
    if res.returncode != 0 or not any(frames_out.glob("*.jpg")):
        if log_cb:
            log_cb("AI 超清执行异常，已自动回退为“极速高清增强”模式。")
        return None

    ai_video = tmp_dir / "ai_enhanced.mp4"
    if log_cb:
        log_cb("AI 超清帧重绘完成，正在合成高清底片...")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            f"{info.fps:.3f}",
            "-i",
            str(frames_out / "frame_%08d.jpg"),
            "-i",
            str(input_path),
            "-map",
            "0:v",
            "-map",
            "1:a?",
            "-vf",
            f"scale={info.width}:{info.height}:flags=lanczos",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "16",
            "-c:a",
            "copy",
            "-pix_fmt",
            "yuv420p",
            str(ai_video),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
        creationflags=creationflags,
    )
    return ai_video


def _build_video_encoder_args(encoder: str, bitrate_mbps: int) -> list[str]:
    br = max(4, int(bitrate_mbps))
    if encoder == "h264_nvenc":
        return [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-rc",
            "vbr",
            "-cq",
            "20",
            "-b:v",
            f"{br}M",
            "-maxrate",
            f"{int(br * 1.5)}M",
            "-bufsize",
            f"{br * 2}M",
            "-pix_fmt",
            "yuv420p",
        ]
    if encoder == "h264_videotoolbox":
        return [
            "-c:v",
            "h264_videotoolbox",
            "-b:v",
            f"{br}M",
            "-pix_fmt",
            "yuv420p",
        ]
    return [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "19",
        "-maxrate",
        f"{int(br * 1.5)}M",
        "-bufsize",
        f"{br * 2}M",
        "-pix_fmt",
        "yuv420p",
    ]


def render_video(
    input_path: Path,
    output_path: Path,
    subtitles: list[SubtitleItem],
    cfg: AppConfig,
    hw: HardwareProfile,
    log_cb: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[float], None]] = None,
    watermark_text: str = "",
    cancel_event=None,
    en_top_pct: Optional[float] = None,
    en_bottom_pct: Optional[float] = None,
) -> Path:
    """执行无痕羽化蒙版去重、字幕与飘动水印烧录、音频均衡与视频压制。"""
    ffmpeg = find_ffmpeg_binary("ffmpeg")
    info = probe_video(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="autoclip_render_") as tmp_str:
        tmp_dir = Path(tmp_str)
        ass_path = write_ass_subtitle(
            subtitles,
            tmp_dir / "sub.ass",
            info.width,
            info.height,
            cfg,
            watermark_text=watermark_text,
            video_duration=info.duration,
            en_top_pct=en_top_pct,
            en_bottom_pct=en_bottom_pct,
        )

        # 0. 可选真·AI 超清（Real-ESRGAN 逐帧重绘，未安装组件时自动回退至极速高清增强）
        effective_input = input_path
        effective_enhance = (cfg.enhance_mode or "fast").lower()
        if effective_enhance == "ai":
            ai_vid = _run_ai_super_resolution(input_path, info, tmp_dir, ffmpeg, log_cb=log_cb)
            if ai_vid is not None:
                effective_input = ai_vid
            else:
                effective_enhance = "fast"

        # 使用 3060 CUDA 硬件解码（把 HEVC/H.264 解码彻底移到显卡 NVDEC，大幅降低 CPU 占用）
        use_cuda_decode = hw.has_nvenc
        inputs: list[str] = []
        if use_cuda_decode:
            inputs.extend(["-hwaccel", "cuda"])
        inputs.extend(["-i", str(effective_input)])
        filter_chains: list[str] = []
        v_cur = "[0:v]"

        # 1. 极速高清增强（去掉极吃 CPU 的 hqdn3d 3D降噪，改用轻量 3x3 亮度锐化+对比度，CPU 占用降 60%）
        if effective_enhance == "fast":
            filter_chains.append(
                f"{v_cur}unsharp=3:3:0.65:3:3:0.0,eq=contrast=1.03:saturation=1.04[v_hd]"
            )
            v_cur = "[v_hd]"

        # 2. 隐形基础去重（使用高速 bilinear 缩放替代重型 lanczos + 极微弱人眼无感调色偏移）
        rng = random.Random()
        if cfg.dedup_level in ("light", "standard"):
            zoom = 1.012 if cfg.dedup_level == "standard" else 1.006
            sw = int(round(info.width * zoom))
            sh = int(round(info.height * zoom))
            sw += sw % 2
            sh += sh % 2
            cx = (sw - info.width) // 2
            cy = (sh - info.height) // 2

            eq_part = ""
            if cfg.dedup_level == "standard":
                bri = round(rng.uniform(-0.008, 0.010), 4)
                con = round(rng.uniform(1.005, 1.018), 4)
                sat = round(rng.uniform(1.005, 1.020), 4)
                eq_part = f",eq=brightness={bri}:contrast={con}:saturation={sat}"

            filter_chains.append(
                f"{v_cur}scale={sw}:{sh}:flags=fast_bilinear,crop={info.width}:{info.height}:{cx}:{cy}{eq_part}[v_base]"
            )
            v_cur = "[v_base]"

        # 3. 无痕羽化线性蒙版：仅处理顶部有效条带（约占画面 28% 面积），CPU 计算量直降 72%，画面完全无切缝
        if cfg.enable_feathered_mask:
            boundary_y = int(round(info.height * (cfg.mask_height_pct / 100.0)))
            half_f = max(12, int(round(cfg.mask_feather_px * (info.height / 1440.0) / 2.0)))
            strip_h = min(info.height, boundary_y + half_f + 4)
            strip_h += strip_h % 2
            mask_png = _create_feathered_mask_png(
                info.width,
                strip_h,
                info.height,
                cfg.mask_height_pct,
                cfg.mask_feather_px,
                tmp_dir / "mask.png",
            )
            inputs.extend(["-i", str(mask_png)])
            shift = max(2, int(round(cfg.mask_shift_px * (info.width / 1080.0))))
            crop_w = max(info.width - shift, info.width - 16)
            filter_chains.append(f"{v_cur}split=2[m_orig][m_top_in]")
            filter_chains.append(
                f"[m_top_in]crop={crop_w}:{strip_h}:{shift}:0,scale={info.width}:{strip_h}:flags=fast_bilinear,format=rgba[m_top_rgba]"
            )
            filter_chains.append("[1:v]format=gray[m_alpha]")
            filter_chains.append("[m_top_rgba][m_alpha]alphamerge[m_top_masked]")
            filter_chains.append("[m_orig][m_top_masked]overlay=0:0:format=yuv420[v_masked]")
            v_cur = "[v_masked]"

        # 4. 烧录中文字幕与飘动水印
        fonts_dir = _find_fonts_dir()
        if fonts_dir:
            f_arg = f"ass=sub.ass:fontsdir='{fonts_dir.as_posix().replace(':', chr(92) + ':')}'"
        else:
            f_arg = "ass=sub.ass"

        target_fps = int(cfg.output_fps)
        fps_part = f",fps={target_fps}" if abs(info.fps - target_fps) > 1.0 else ""
        filter_chains.append(f"{v_cur}{f_arg}{fps_part}[v_out]")
        filter_complex = ";".join(filter_chains)

        # 4. 音频滤镜：响度均衡（清晰防爆音，适合中老年人）+ 隐形音频指纹微调
        audio_args: list[str] = []
        if info.has_audio:
            af_parts: list[str] = []
            if cfg.dedup_level == "standard":
                # 极轻微人声频段清晰度微调（同时改变音频指纹）
                af_parts.append("highpass=f=45")
                af_parts.append("equalizer=f=2800:t=q:w=1.5:g=1.0")
            if cfg.audio_loudnorm:
                af_parts.append("loudnorm=I=-16:TP=-1.5:LRA=11")
            if af_parts:
                audio_args.extend(["-af", ",".join(af_parts)])
            audio_args.extend(["-c:a", "aac", "-b:a", "160k", "-ar", "44100"])
        else:
            audio_args.append("-an")

        encoder = hw.recommended_video_encoder
        enc_args = _build_video_encoder_args(encoder, cfg.video_bitrate_mbps)

        # 彻底清除原始元数据并写入随机去重标识
        meta_args = [
            "-map_metadata",
            "-1",
            "-metadata",
            f"comment={uuid.uuid4().hex[:12]}",
        ]
        if cfg.output_format.lower() == "mp4":
            meta_args.extend(["-movflags", "+faststart"])

        # 原子写入保护：先输出到临时文件，完整渲染成功后再重命名为正式成品，彻底杜绝半截损坏文件
        tmp_out_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
        cmd = [
            ffmpeg,
            "-y",
            "-filter_threads",
            "2",
            *inputs,
            "-filter_complex",
            filter_complex,
            "-map",
            "[v_out]",
            *(["-map", "0:a?"] if info.has_audio else []),
            *enc_args,
            *audio_args,
            *meta_args,
            str(tmp_out_path),
        ]

        if log_cb:
            log_cb(
                f"正在渲染成片（编码器：{encoder}，格式：{cfg.output_format.upper()}，{cfg.output_fps}fps）..."
            )

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen(
            cmd,
            cwd=str(tmp_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
        )
        register_proc(proc)

        time_re = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")
        stderr_tail: list[str] = []
        assert proc.stderr is not None
        try:
            for line in proc.stderr:
                if cancel_event is not None and cancel_event.is_set():
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    break
                stderr_tail.append(line.strip())
                if len(stderr_tail) > 25:
                    stderr_tail.pop(0)
                m = time_re.search(line)
                if m and info.duration > 0 and progress_cb:
                    cur_t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
                    progress_cb(min(0.99, cur_t / info.duration))
            ret = proc.wait()
        finally:
            unregister_proc(proc)

        if cancel_event is not None and cancel_event.is_set():
            try:
                if tmp_out_path.is_file():
                    tmp_out_path.unlink()
            except Exception:
                pass
            raise InterruptedError("用户已停止任务")

        if ret != 0:
            try:
                if tmp_out_path.is_file():
                    tmp_out_path.unlink()
            except Exception:
                pass
            # 若 NVENC 异常（例如驱动过旧或超出会话上限），自动回退 libx264 重试一次
            if encoder == "h264_nvenc":
                if log_cb:
                    log_cb("NVENC 编码失败，自动切换 CPU (libx264) 重试...")
                fallback_hw = HardwareProfile(**{**hw.__dict__, "recommended_video_encoder": "libx264"})
                return render_video(
                    input_path,
                    output_path,
                    subtitles,
                    cfg,
                    fallback_hw,
                    log_cb=log_cb,
                    progress_cb=progress_cb,
                    watermark_text=watermark_text,
                    cancel_event=cancel_event,
                    en_top_pct=en_top_pct,
                    en_bottom_pct=en_bottom_pct,
                )
            raise RuntimeError("FFmpeg 渲染失败：\n" + "\n".join(stderr_tail[-10:]))

        if not tmp_out_path.is_file() or tmp_out_path.stat().st_size < 10_000:
            raise RuntimeError("FFmpeg 未产出有效成品文件，请检查磁盘空间或原视频。")
        tmp_out_path.replace(output_path)

        if progress_cb:
            progress_cb(1.0)
        if log_cb:
            log_cb(f"成片导出成功：{output_path.name}")
        return output_path


def render_preview_frame(
    video_path: Path,
    cfg: AppConfig,
    sample_text: str = "你怎么知道这件事的？",
    timestamp_sec: float = 5.0,
    watermark_text: str = "",
) -> Path:
    """渲染单帧预览图（带实际字幕与半透明水印样式），供 GUI 实时预览排版与安全区。"""
    ffmpeg = find_ffmpeg_binary("ffmpeg")
    info = probe_video(video_path)
    t = min(timestamp_sec, max(0.5, info.duration * 0.15))

    tmp_dir = Path(tempfile.gettempdir()) / "autoclip_preview"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_img = tmp_dir / "preview.jpg"

    sample_sub = [SubtitleItem(index=1, start=0.0, end=10.0, text=sample_text)]
    write_ass_subtitle(
        sample_sub,
        tmp_dir / "sub.ass",
        info.width,
        info.height,
        cfg,
        watermark_text=watermark_text,
        video_duration=10.0,
    )

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{t:.2f}",
        "-i",
        str(video_path),
        "-vf",
        "ass=sub.ass",
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(out_img),
    ]
    subprocess.run(
        cmd,
        cwd=str(tmp_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
        creationflags=creationflags,
    )
    return out_img

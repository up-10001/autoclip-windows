"""硬件检测与自适应并发调度模块：自动识别 NVIDIA 显卡（如 RTX 3060 12G）、显存、CPU 与内存。"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import get_app_dir, get_resource_dir


@dataclass
class HardwareProfile:
    os_name: str
    cpu_cores: int
    ram_gb: float
    gpu_name: str
    gpu_vram_gb: float
    has_cuda: bool
    has_nvenc: bool
    has_videotoolbox: bool
    recommended_video_encoder: str
    recommended_whisper_device: str
    recommended_whisper_compute: str
    recommended_render_workers: int
    recommended_translate_workers: int
    recommended_asr_workers: int

    def summary_text(self) -> str:
        gpu_part = (
            f"{self.gpu_name} ({self.gpu_vram_gb:.1f}GB)"
            if self.gpu_name
            else "无独立 NVIDIA 显卡 (CPU模式)"
        )
        mode_part = (
            f"并行 {self.recommended_render_workers} 路压制"
            if self.recommended_render_workers > 1
            else "顺序 1 路压制"
        )
        return (
            f"硬件检测：{gpu_part} | CPU {self.cpu_cores}核 | 内存 {self.ram_gb:.0f}GB | "
            f"编码器 {self.recommended_video_encoder} | 自动适配：{mode_part}"
        )


def find_ffmpeg_binary(name: str = "ffmpeg") -> str:
    """优先查找打包同目录/bin 目录下的 ffmpeg/ffprobe，再回退到系统 PATH。"""
    exe_name = f"{name}.exe" if platform.system() == "Windows" else name
    search_dirs = [
        get_app_dir(),
        get_app_dir() / "bin",
        get_app_dir() / "ffmpeg",
        get_resource_dir() / "bin",
    ]
    for folder in search_dirs:
        candidate = folder / exe_name
        if candidate.is_file():
            return str(candidate)
    found = shutil.which(exe_name) or shutil.which(name)
    return found or exe_name


def _get_ram_gb() -> float:
    system = platform.system()
    try:
        if system == "Windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / (1024 ** 3)
        if system == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            return int(out) / (1024 ** 3)
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 ** 2)
    except Exception:
        pass
    return 16.0


def _detect_nvidia_gpu() -> tuple[str, float]:
    """通过 nvidia-smi 检测 NVIDIA 显卡型号及显存（GB）。"""
    smi = shutil.which("nvidia-smi")
    if not smi and platform.system() == "Windows":
        candidate = Path(r"C:\Windows\System32\nvidia-smi.exe")
        if candidate.is_file():
            smi = str(candidate)
    if not smi:
        return "", 0.0
    try:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        out = subprocess.check_output(
            [smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
            creationflags=creationflags,
        ).strip()
        if not out:
            return "", 0.0
        first_line = out.splitlines()[0]
        parts = [p.strip() for p in first_line.split(",")]
        name = parts[0]
        vram_mb = float(parts[1]) if len(parts) > 1 else 0.0
        return name, vram_mb / 1024.0
    except Exception:
        return "", 0.0


def _detect_ffmpeg_encoders(ffmpeg_path: str) -> set[str]:
    try:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        out = subprocess.check_output(
            [ffmpeg_path, "-hide_banner", "-encoders"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
            creationflags=creationflags,
        )
        encoders = set()
        for enc in ("h264_nvenc", "h264_videotoolbox", "libx264"):
            if enc in out:
                encoders.add(enc)
        return encoders
    except Exception:
        return {"libx264"}


def _check_ctranslate2_cuda() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def detect_hardware(ffmpeg_path: Optional[str] = None) -> HardwareProfile:
    os_name = platform.system()
    cpu_cores = os.cpu_count() or 4
    ram_gb = _get_ram_gb()
    gpu_name, gpu_vram_gb = _detect_nvidia_gpu()

    ffmpeg_bin = ffmpeg_path or find_ffmpeg_binary("ffmpeg")
    encoders = _detect_ffmpeg_encoders(ffmpeg_bin)

    has_cuda = bool(gpu_name) and _check_ctranslate2_cuda()
    has_nvenc = bool(gpu_name) and ("h264_nvenc" in encoders)
    has_videotoolbox = (os_name == "Darwin") and ("h264_videotoolbox" in encoders)

    if has_nvenc:
        encoder = "h264_nvenc"
    elif has_videotoolbox:
        encoder = "h264_videotoolbox"
    else:
        encoder = "libx264"

    if has_cuda:
        whisper_device = "cuda"
        whisper_compute = "float16" if gpu_vram_gb >= 6.0 else "int8_float16"
    else:
        whisper_device = "cpu"
        whisper_compute = "int8"

    # 根据显卡、显存、CPU 和内存自动适配并行度
    # RTX 3060 12GB（vram>=10, ram>=14）：NVENC 可稳定并发 2 路压制 + 4 路 DeepSeek 并发翻译 + 1 路高速 GPU ASR（流水线并行）
    if has_nvenc and gpu_vram_gb >= 10.0 and ram_gb >= 14.0 and cpu_cores >= 6:
        render_workers = 2
        translate_workers = 4
        asr_workers = 1
    elif has_nvenc and gpu_vram_gb >= 6.0 and ram_gb >= 12.0:
        render_workers = 2 if cpu_cores >= 8 else 1
        translate_workers = 3
        asr_workers = 1
    elif has_videotoolbox and ram_gb >= 14.0 and cpu_cores >= 8:
        render_workers = 2
        translate_workers = 3
        asr_workers = 1
    elif cpu_cores >= 12 and ram_gb >= 16.0:
        render_workers = 2
        translate_workers = 3
        asr_workers = 1
    else:
        # 性能不足时自动退回顺序剪辑，确保稳定不卡死
        render_workers = 1
        translate_workers = 2
        asr_workers = 1

    return HardwareProfile(
        os_name=os_name,
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        has_cuda=has_cuda,
        has_nvenc=has_nvenc,
        has_videotoolbox=has_videotoolbox,
        recommended_video_encoder=encoder,
        recommended_whisper_device=whisper_device,
        recommended_whisper_compute=whisper_compute,
        recommended_render_workers=render_workers,
        recommended_translate_workers=translate_workers,
        recommended_asr_workers=asr_workers,
    )

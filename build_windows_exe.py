"""Windows EXE 一键打包脚本（由 一键打包EXE.bat 调用，彻底避免 cmd 乱码与 PowerShell 语法问题）。"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
BIN_DIR = ROOT / "bin"
TSINGHUA_PYPI = "https://pypi.tuna.tsinghua.edu.cn/simple"
FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
REALESRGAN_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
    "realesrgan-ncnn-vulkan-20220424-windows.zip"
)


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def ensure_ffmpeg() -> None:
    ff = BIN_DIR / "ffmpeg.exe"
    fp = BIN_DIR / "ffprobe.exe"
    if ff.is_file() and fp.is_file():
        print("[3/4] 已检测到 bin/ffmpeg.exe，跳过下载。", flush=True)
        return

    BIN_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = BIN_DIR / "ffmpeg.zip"
    tmp_dir = BIN_DIR / "tmp_extract"
    print("[3/4] 正在下载 Windows 版 FFmpeg（约 80MB，请稍候）...", flush=True)
    urllib.request.urlretrieve(FFMPEG_URL, zip_path)

    print("      正在解压 FFmpeg...", flush=True)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)

    for target_name in ("ffmpeg.exe", "ffprobe.exe"):
        found = list(tmp_dir.rglob(target_name))
        if not found:
            raise RuntimeError(f"压缩包中未找到 {target_name}")
        shutil.copy2(found[0], BIN_DIR / target_name)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    zip_path.unlink(missing_ok=True)
    print("      FFmpeg 准备完成！", flush=True)


def ensure_realesrgan() -> None:
    """下载 Real-ESRGAN Vulkan Windows 免安装版（用于真·AI超清，失败不影响极速高清模式打包）。"""
    target_dir = BIN_DIR / "realesrgan"
    exe = target_dir / "realesrgan-ncnn-vulkan.exe"
    if exe.is_file():
        print("[3.5/4] 已检测到 bin/realesrgan，跳过下载。", flush=True)
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = BIN_DIR / "realesrgan.zip"
    print("[3.5/4] 正在下载真·AI超清组件 Real-ESRGAN（约 45MB，如网络慢可稍候）...", flush=True)
    try:
        urllib.request.urlretrieve(REALESRGAN_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
        zip_path.unlink(missing_ok=True)
        print("        Real-ESRGAN AI 超清组件准备完成！", flush=True)
    except Exception as exc:
        zip_path.unlink(missing_ok=True)
        print(f"        [提示] Real-ESRGAN 下载跳过（{exc}），不影响极速高清增强与正常打包。", flush=True)


def main() -> None:
    print("========================================================", flush=True)
    print("  AutoClip - Windows EXE 自动打包程序 (RTX 3060 12G)", flush=True)
    print("========================================================\n", flush=True)

    if not VENV_DIR.exists():
        print("[1/4] 正在创建独立运行环境 .venv ...", flush=True)
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
    else:
        print("[1/4] 检测到已有 .venv，直接使用。", flush=True)

    venv_python = VENV_DIR / "Scripts" / "python.exe"
    if not venv_python.is_file():
        raise RuntimeError(f"未找到虚拟环境 Python：{venv_python}")

    print("[2/4] 正在安装依赖与 RTX 3060 CUDA 加速库...", flush=True)
    run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "-i", TSINGHUA_PYPI])
    run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "-r",
            str(ROOT / "requirements.txt"),
            "pyinstaller",
            "-i",
            TSINGHUA_PYPI,
        ]
    )

    ensure_ffmpeg()
    ensure_realesrgan()

    print("[4/4] 正在生成 dist\\AutoClip\\AutoClip.exe ...", flush=True)
    run(
        [
            str(venv_python),
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            str(ROOT / "AutoClip.spec"),
        ]
    )

    out_dir = ROOT / "dist" / "AutoClip"
    print("\n========================================================", flush=True)
    print("  打包成功！", flush=True)
    print(f"  EXE 位置：{out_dir / 'AutoClip.exe'}", flush=True)
    print("  直接双击 AutoClip.exe 即可使用！", flush=True)
    print("========================================================", flush=True)
    if os.name == "nt" and out_dir.is_dir():
        os.startfile(str(out_dir))  # type: ignore[attr-defined]


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[打包失败] {exc}", flush=True)
        sys.exit(1)

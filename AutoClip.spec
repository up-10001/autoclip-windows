# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：生成可直接双击运行的 Windows AutoClip.exe（适配 RTX 3060 12G）。"""

from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

block_cipher = None
ROOT = Path(SPECPATH)

datas = []
binaries = []
hiddenimports = [
    "PIL._tkinter_finder",
    "httpx",
    "faster_whisper",
    "ctranslate2",
    "av",
]

for pkg in ("faster_whisper", "ctranslate2", "tokenizers", "onnxruntime", "rapidocr_onnxruntime"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

# 打包 Windows NVIDIA cuBLAS / cuDNN 动态库，确保 RTX 3060 开箱即用
for cuda_pkg in ("nvidia.cublas", "nvidia.cudnn"):
    try:
        binaries += collect_dynamic_libs(cuda_pkg)
    except Exception:
        pass

# 打包内置 ffmpeg.exe / ffprobe.exe 与 realesrgan AI 超清组件（如果放在 bin/ 目录）
bin_dir = ROOT / "bin"
if bin_dir.is_dir():
    for f in bin_dir.iterdir():
        if f.is_file():
            binaries.append((str(f), "bin"))
    realesrgan_dir = bin_dir / "realesrgan"
    if realesrgan_dir.is_dir():
        datas.append((str(realesrgan_dir), "bin/realesrgan"))

assets_dir = ROOT / "assets"
if assets_dir.is_dir():
    datas.append((str(assets_dir), "assets"))

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "scipy", "pandas", "torch", "tensorflow"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AutoClip",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # 双击不弹出黑色命令行窗口
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="AutoClip",
)

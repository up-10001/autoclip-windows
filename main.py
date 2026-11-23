"""自动剪辑工具主入口：双击 EXE 启动图形界面，也支持命令行单条/批量调用。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _register_windows_cuda_dlls() -> None:
    """将打包目录中 nvidia/cublas/bin 与 nvidia/cudnn/bin 注册到 Windows DLL 搜索路径。"""
    if os.name != "nt":
        return
    base_dirs = [Path(__file__).resolve().parent]
    if getattr(sys, "frozen", False):
        base_dirs.append(Path(sys.executable).resolve().parent)
        if hasattr(sys, "_MEIPASS"):
            base_dirs.append(Path(sys._MEIPASS))
    for base in base_dirs:
        candidates = [
            base / "nvidia" / "cublas" / "bin",
            base / "nvidia" / "cudnn" / "bin",
            base / "_internal" / "nvidia" / "cublas" / "bin",
            base / "_internal" / "nvidia" / "cudnn" / "bin",
            base / "_internal" / "shapely.libs",
            base / "shapely.libs",
        ]
        # 同时自动注册 _internal 下所有 *.libs 目录（确保 shapely/cv2/yaml 动态库秒加载）
        for parent in (base, base / "_internal"):
            if parent.is_dir():
                for item in parent.iterdir():
                    if item.is_dir() and item.name.endswith(".libs"):
                        candidates.append(item)
        for sub in candidates:
            if sub.is_dir():
                try:
                    os.add_dll_directory(str(sub))
                except Exception:
                    pass
                os.environ["PATH"] = str(sub) + os.pathsep + os.environ.get("PATH", "")


def _register_hot_patch_dir() -> None:
    """允许将热补丁目录 patch/ 放在 AutoClip.exe 同级，后续更新只需替换几 KB 的 .py 文件。"""
    app_dir = (
        Path(sys.executable).resolve().parent
        if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parent
    )
    patch_dir = app_dir / "patch"
    if patch_dir.is_dir() and str(patch_dir) not in sys.path:
        sys.path.insert(0, str(patch_dir))


def main() -> None:
    _register_windows_cuda_dlls()
    _register_hot_patch_dir()
    parser = argparse.ArgumentParser(description="自动剪辑工具 AutoClip")
    parser.add_argument("--cli", nargs="+", help="命令行模式：直接传入一个或多个视频路径进行剪辑")
    parser.add_argument("--format", choices=["mp4", "mov"], help="覆盖输出格式（mp4 或 mov）")
    parser.add_argument("--api-key", help="临时指定 DeepSeek API Key")
    parser.add_argument("--out-dir", help="指定输出文件夹")
    args = parser.parse_args()

    if args.cli:
        from autoclip.config import load_config
        from autoclip.hardware import detect_hardware
        from autoclip.pipeline import BatchPipeline

        cfg = load_config()
        if args.format:
            cfg.output_format = args.format
        if args.api_key:
            cfg.api_key = args.api_key
        if args.out_dir:
            cfg.output_dir = args.out_dir

        hw = detect_hardware()
        print(hw.summary_text())
        pipeline = BatchPipeline(cfg=cfg, hw=hw, log_cb=print)
        videos = [Path(p).resolve() for p in args.cli]
        results = pipeline.run_batch(videos)
        failed = [r for r in results if r.state != "已完成"]
        sys.exit(1 if failed else 0)

    from autoclip.gui import launch_gui

    launch_gui()


if __name__ == "__main__":
    main()

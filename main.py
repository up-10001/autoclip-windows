"""自动剪辑工具主入口：双击 EXE 启动图形界面，也支持命令行单条/批量调用。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
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

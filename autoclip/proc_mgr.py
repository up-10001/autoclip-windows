"""子进程注册表：点击“停止”或关闭窗口时，立即强制终止所有正在运行的 FFmpeg / AI 进程，杜绝后台残留。"""

from __future__ import annotations

import subprocess
import threading

_LOCK = threading.Lock()
_ACTIVE_PROCS: set[subprocess.Popen] = set()


def register_proc(proc: subprocess.Popen) -> None:
    with _LOCK:
        _ACTIVE_PROCS.add(proc)


def unregister_proc(proc: subprocess.Popen) -> None:
    with _LOCK:
        _ACTIVE_PROCS.discard(proc)


def kill_all_active_procs() -> int:
    """立即终止所有正在执行的 FFmpeg / 外部子进程，返回终止数量。"""
    with _LOCK:
        procs = list(_ACTIVE_PROCS)
    killed = 0
    for p in procs:
        try:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=1.0)
                except Exception:
                    p.kill()
                killed += 1
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    with _LOCK:
        _ACTIVE_PROCS.clear()
    return killed

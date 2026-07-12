"""Stable Streamlit launcher for Qwopus-Agent."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Launch Streamlit with native-library safety defaults."""
    project_root = Path(__file__).resolve().parents[3]
    app_path = project_root / "src" / "qwopus_agent" / "ui" / "streamlit_chat.py"

    env = os.environ.copy()
    src_path = str(project_root / "src")
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)

    # 原因：pyarrow mimalloc 在当前 macOS 环境会触发 Python segfault。
    # 作用：即使用户没有手动设置 PYTHONPATH，也让 UI 启动时使用安全内存池。
    env.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.call(command, env=env, cwd=project_root))


if __name__ == "__main__":
    main()

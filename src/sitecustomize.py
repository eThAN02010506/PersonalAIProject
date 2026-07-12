"""Process-wide runtime safety defaults for local development.

Python imports `sitecustomize` automatically when `PYTHONPATH=src` is set. We use it for native
library safety switches that must be configured before Streamlit, pandas, or pyarrow are imported.
"""

from __future__ import annotations

import os

# 原因：当前 macOS/pyarrow 环境在 mimalloc memory pool 路径中发生 SIGSEGV。
# 作用：强制 Arrow 使用系统内存池，避免 libarrow 的 mimalloc 崩溃路径。
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

# 原因：数值/图计算库可能默认开很多 native 线程，增加 Streamlit 下的不稳定性。
# 作用：降低本地 UI 进程的 native 线程压力；后续重计算可放到独立 worker。
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

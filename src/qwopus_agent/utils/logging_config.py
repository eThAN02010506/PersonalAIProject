"""Runtime logging configuration."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

DEFAULT_RUNTIME_LOG_PATH = Path("logs/qwopus_agent.log")


def configure_runtime_logging(
        log_path: Path = DEFAULT_RUNTIME_LOG_PATH,
        level: int = logging.INFO,
) -> None:
    """Configure rotating file logging for the local app runtime."""
    root_logger = logging.getLogger("qwopus_agent")
    if any(
        isinstance(handler, RotatingFileHandler)
        and Path(handler.baseFilename) == log_path.resolve()
        for handler in root_logger.handlers
    ):
        return

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            log_path,
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
    except OSError:
        # 原因：只读部署或测试环境可能禁止创建日志文件。
        # 作用：文件日志保持旁路能力，磁盘权限问题不会阻止 API 启动。
        logging.getLogger(__name__).warning(
            "runtime_file_logging_unavailable path=%s",
            log_path,
        )
        return
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s"
        )
    )
    root_logger.setLevel(level)
    root_logger.addHandler(handler)
    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a project logger."""
    return logging.getLogger(f"qwopus_agent.{name}")

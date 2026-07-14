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

    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_path,
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
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

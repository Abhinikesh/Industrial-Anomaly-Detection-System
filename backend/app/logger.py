"""
backend/app/logger.py
=====================
Application-wide logger setup.

Creates a logger that writes to BOTH:
  - stdout (console) — with colour-coded level prefix for readability
  - logs/app.log     — rotating file, max 10 MB × 5 backup files

Usage:
    from app.logger import get_logger
    log = get_logger(__name__)
    log.info("Models loaded")
    log.warning("Falling back to default model")
    log.error("MongoDB write failed: %s", exc)

Calling get_logger() multiple times with the same name returns the same
instance (standard Python logging behaviour).
"""

import logging
import logging.handlers
import os
import sys

from app.config import settings

# ── Formatting ────────────────────────────────────────────────────────────────

_FILE_FMT = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
_CONSOLE_FMT = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def _build_console_handler() -> logging.StreamHandler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_CONSOLE_FMT, datefmt=_DATE_FMT))
    return handler


def _build_file_handler() -> logging.handlers.RotatingFileHandler:
    # Resolve log file path relative to backend/ root (two levels up from this file)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = os.path.join(base_dir, settings.log_file)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_DATE_FMT))
    return handler


def _configure_root() -> None:
    """Set up handlers on the root logger exactly once."""
    root = logging.getLogger()
    if root.handlers:
        return  # already configured (e.g. uvicorn already set up)

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root.setLevel(level)
    root.addHandler(_build_console_handler())
    root.addHandler(_build_file_handler())

    # Quiet noisy third-party loggers
    logging.getLogger("pymongo").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


# Configure at import time so the first log call works immediately
_configure_root()


def get_logger(name: str) -> logging.Logger:
    """Return a named logger inheriting root handlers.

    Always call this instead of logging.getLogger() directly so the
    log file and level are already guaranteed to be configured.
    """
    logger = logging.getLogger(name)
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(level)
    return logger

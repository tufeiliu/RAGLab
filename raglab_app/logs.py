"""Logging setup shared by background jobs and the Streamlit app.

Jobs log to stdout, which the job runner redirects to the job's job.log. The app
logs interactive queries to raglab_data/logs/app.log.
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import sys

from raglab_app import LOGS

FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"
NOISY = ("httpx", "httpcore", "openai", "anthropic", "chromadb", "urllib3")


def _quiet_libraries() -> None:
    for name in NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)


def setup_job_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATEFMT, stream=sys.stdout, force=True)
    _quiet_libraries()


def setup_app_logging() -> None:
    """Idempotent: Streamlit reruns the app script on every interaction."""
    logger = logging.getLogger("raglab")
    if any(getattr(h, "raglab_app", False) for h in logger.handlers):
        return
    LOGS.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(LOGS / "app.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter(FORMAT, DATEFMT))
    handler.raglab_app = True
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    _quiet_libraries()

"""Central logging configuration."""

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once at application startup."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=_FORMAT,
        stream=sys.stdout,
        force=True,
    )
    # Silence noisy third-party loggers
    for noisy in ("ppocr", "paddle", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

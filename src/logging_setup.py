"""Shared stdout logger construction.

``src.runner`` and ``src.evaluate`` each built an identical handler; the only
thing that ever differed was the logger name, so that is the parameter.
"""

from __future__ import annotations

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return an idempotent stdout logger named ``name``.

    Re-calling with the same name returns the existing logger untouched rather
    than stacking a second handler on it.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


__all__ = ["get_logger"]

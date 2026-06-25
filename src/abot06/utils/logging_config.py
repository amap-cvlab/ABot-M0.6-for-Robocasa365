from __future__ import annotations

import logging


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name)


def setup_logging(level: int | str = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

"""Application logging: console + rotating ``logs/simulation.log``."""

from __future__ import annotations

import logging
import logging.handlers
import sys

from .config import LoggingConfig

_HANDLER_TAG = "_swarm_sim_handler"
FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure_logging(config: LoggingConfig, *, console: bool | None = None) -> None:
    """Install the platform's handlers on the root logger (idempotent)."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_TAG, False):
            root.removeHandler(handler)
            handler.close()
    level = getattr(logging, config.level.upper())
    root.setLevel(level)
    formatter = logging.Formatter(FORMAT)

    directory = config.path
    directory.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        directory / "simulation.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)
    setattr(file_handler, _HANDLER_TAG, True)
    root.addHandler(file_handler)

    if config.console if console is None else console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(formatter)
        # Per-drone chatter goes to the file; the console only shows warnings from the event bus.
        stream.addFilter(lambda r: r.name != "simulation.events" or r.levelno >= logging.WARNING)
        setattr(stream, _HANDLER_TAG, True)
        root.addHandler(stream)

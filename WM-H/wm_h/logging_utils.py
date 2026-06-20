"""Colored terminal logging for wm_h."""

from __future__ import annotations

import logging

from .terminal_ui import TqdmLoggingHandler, _C, _paint

MODULE_PREFIX = "wm_h."
CLI_LOGGER = f"{MODULE_PREFIX}cli"

_QUIET_LOGGERS = (
    "transformers",
    "diffsynth",
    f"{MODULE_PREFIX}box_editor",
    f"{MODULE_PREFIX}instruction_builder",
    f"{MODULE_PREFIX}parallel",
    f"{MODULE_PREFIX}instr_first_pipeline",
    f"{MODULE_PREFIX}instr_first",
    f"{MODULE_PREFIX}text_llm",
)


class _ColorFormatter(logging.Formatter):
    _LEVEL_STYLE = {
        logging.DEBUG: (_C.DIM, "dbg"),
        logging.INFO: (_C.DIM, "·"),
        logging.WARNING: (_C.YELLOW, "!"),
        logging.ERROR: (_C.RED, "✗"),
        logging.CRITICAL: (_C.RED + _C.BOLD, "!!"),
    }

    def __init__(self, *, verbose: bool = False):
        super().__init__(datefmt="%H:%M:%S")
        self.verbose = verbose

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        if self.verbose:
            ts = self.formatTime(record, self.datefmt)
            level = record.levelname
            name = record.name.replace(MODULE_PREFIX, "")
            return f"{ts} {level:7} {name}: {msg}"

        style = self._LEVEL_STYLE.get(record.levelno, (_C.DIM, "·"))
        prefix = _paint(style[0], f"  {style[1]} ")
        if record.levelno == logging.INFO:
            return f"{prefix}{msg}"
        return f"{prefix}{_paint(style[0], msg)}"


def get_cli_logger() -> logging.Logger:
    """Logger for run summaries; always INFO even when pipeline logs are quiet."""
    return logging.getLogger(CLI_LOGGER)


def setup_logging(*, verbose: bool = False) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    formatter = _ColorFormatter(verbose=verbose)
    handler = TqdmLoggingHandler(formatter)
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)

    level = logging.DEBUG if verbose else logging.WARNING
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(level)

    get_cli_logger().setLevel(logging.INFO)

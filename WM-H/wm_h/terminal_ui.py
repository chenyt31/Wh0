"""Colored, tqdm-safe terminal helpers for wm_h."""

from __future__ import annotations

import logging
import sys
from typing import Dict, Optional, Tuple

from tqdm import tqdm


class _C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"


def _use_color() -> bool:
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


def _paint(code: str, text: str) -> str:
    if not _use_color():
        return text
    return f"{code}{text}{_C.RESET}"


def write(msg: str = "", *, file=None) -> None:
    """Print via tqdm.write so progress bars stay intact."""
    tqdm.write(msg, file=file or sys.stderr)


def banner(title: str = "WM-H") -> None:
    line = "─" * 52
    write(_paint(_C.BOLD + _C.CYAN, f"━━ {title} " + line[:44]))


def stage(stage: str, title: str) -> None:
    write(_paint(_C.BOLD + _C.CYAN, f"── {stage}  {title}"))


def kv(key: str, value: str, indent: int = 2) -> None:
    pad = " " * indent
    write(f"{pad}{_paint(_C.DIM, key + ':')} {value}")


def note(msg: str) -> None:
    write(f"  {_paint(_C.DIM, '·')} {msg}")


def success(msg: str) -> None:
    write(f"  {_paint(_C.GREEN, '✓')} {msg}")


def warn(msg: str) -> None:
    write(f"  {_paint(_C.YELLOW, '!')} {msg}")


def error(msg: str) -> None:
    write(f"  {_paint(_C.RED, '✗')} {msg}")


def slot_counts_plain(
    counts: Dict[str, int],
    targets: Optional[Dict[str, int]] = None,
) -> str:
    parts: list[str] = []
    for name in sorted(counts):
        cur = counts[name]
        if targets and name in targets:
            parts.append(f"{name}:{cur}/{targets[name]}")
        else:
            parts.append(f"{name}:{cur}")
    return " ".join(parts)


def slot_counts(counts: Dict[str, int], targets: Optional[Dict[str, int]] = None) -> str:
    parts: list[str] = []
    for name in sorted(counts):
        cur = counts[name]
        if targets and name in targets:
            tgt = targets[name]
            parts.append(
                f"{_paint(_C.DIM, name)}:"
                f"{_paint(_C.CYAN, str(cur))}"
                f"{_paint(_C.DIM, '/' + str(tgt))}"
            )
        else:
            parts.append(
                f"{_paint(_C.DIM, name)}:{_paint(_C.CYAN, str(cur))}"
            )
    return "  ".join(parts)


def tqdm_defaults(**overrides) -> dict:
    kw: dict = {
        "dynamic_ncols": True,
        "leave": True,
        "bar_format": (
            "{desc}: {percentage:3.0f}%|{bar:22}| "
            "{n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        ),
    }
    if _use_color():
        kw["colour"] = "cyan"
    kw.update(overrides)
    return kw


class VocabExpandBar:
    """Single progress bar for LLM vocab expansion rounds."""

    def __init__(
        self,
        max_rounds: int,
        targets: Dict[str, int],
        slot_names: Tuple[str, ...],
    ):
        self._targets = targets
        self._slot_names = slot_names
        self._pbar = tqdm(
            total=max_rounds,
            desc="vocab",
            unit="round",
            **tqdm_defaults(
                leave=False,
                bar_format=(
                    "{desc} {percentage:3.0f}%|{bar:18}| "
                    "{n_fmt}/{total_fmt} [{elapsed}]"
                ),
            ),
        )

    def update_round(
        self,
        round_idx: int,
        progress: Dict[str, int],
        n_prompts: int,
    ) -> None:
        slot_txt = slot_counts_plain(progress, self._targets)
        self._pbar.set_postfix_str(
            f"r{round_idx} · {n_prompts}p · {slot_txt}",
            refresh=True,
        )
        if round_idx > 0:
            self._pbar.update(1)

    def close(self) -> None:
        self._pbar.close()


class TqdmLoggingHandler(logging.Handler):
    """Route log records through tqdm.write to avoid bar corruption."""

    def __init__(self, formatter: Optional[logging.Formatter] = None):
        super().__init__()
        if formatter is not None:
            self.setFormatter(formatter)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
        except Exception:
            self.handleError(record)

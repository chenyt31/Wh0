"""Repository path helpers for third-party dependencies."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def repo_root() -> Path:
    """Return the Wh0 repository root (parent of ``libs/``)."""
    env_root = os.environ.get("WH0_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[2]


def hawor_root() -> Path:
    return repo_root() / "WM-H" / "third_party" / "HaWoR"


def ensure_hawor_on_path() -> Path:
    root = hawor_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def default_weights_root() -> Path:
    return repo_root() / "weights"

"""Shared I/O helpers for the instr-first pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import List

import yaml

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_images(image_dir: str, recursive: bool = False) -> List[Path]:
    root = Path(image_dir)
    if recursive:
        files = sorted(
            f
            for f in root.rglob("*")
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS
        )
    else:
        files = sorted(
            f
            for f in root.iterdir()
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS
        )
    return files


def resolve_image_edit_config(full_config: dict) -> dict:
    """Merge instr_first.image_edit with optional shared image_edit defaults."""
    if_cfg = dict(full_config.get("instr_first") or {})
    shared = dict(full_config.get("image_edit") or {})
    edit_cfg = dict(shared)
    edit_cfg.update(if_cfg.get("image_edit") or {})
    return edit_cfg

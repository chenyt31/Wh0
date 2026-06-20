"""Import path helpers for Wh0 deployment scripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path


VITRA_ROOT = Path(__file__).resolve().parents[1]
WH0_ROOT = Path(os.environ.get("WH0_ROOT", VITRA_ROOT.parent)).resolve()
XR_TELEOPERATE_ROOT = Path(
    os.environ.get("XR_TELEOPERATE_ROOT", VITRA_ROOT / "thirdparty" / "xr_teleoperate")
).resolve()


def add_path(path: Path) -> None:
    path_str = str(path)
    if path.is_dir() and path_str not in sys.path:
        sys.path.insert(0, path_str)


def configure_imports(require_xr: bool = False) -> None:
    """Expose Wh0, vitra-wh0, and xr_teleoperate modules."""
    add_path(WH0_ROOT)
    add_path(VITRA_ROOT)

    xr_candidates = [
        XR_TELEOPERATE_ROOT,
        XR_TELEOPERATE_ROOT / "teleop",
        XR_TELEOPERATE_ROOT / "teleop" / "teleimager" / "src",
        XR_TELEOPERATE_ROOT / "teleimager" / "src",
    ]
    for candidate in xr_candidates:
        add_path(candidate)

    if require_xr and not XR_TELEOPERATE_ROOT.exists():
        raise FileNotFoundError(
            "xr_teleoperate was not found. Clone it with:\n"
            f"  git clone https://github.com/unitreerobotics/xr_teleoperate.git {XR_TELEOPERATE_ROOT}\n"
            "or set XR_TELEOPERATE_ROOT to an existing checkout."
        )

"""Default paths for hand reconstruction models."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from .paths import default_weights_root


@dataclass
class HandReconConfig:
    """Model and weight paths used by :class:`HandReconstructor`."""

    hawor_model_path: str = ""
    detector_path: str = ""
    moge_model_path: str = ""
    mano_path: str = ""
    droid_weights_path: str = ""

    def __post_init__(self) -> None:
        weights = default_weights_root()
        if not self.hawor_model_path:
            self.hawor_model_path = str(weights / "hawor" / "checkpoints" / "hawor.ckpt")
        if not self.detector_path:
            self.detector_path = str(weights / "hawor" / "external" / "detector.pt")
        if not self.moge_model_path:
            self.moge_model_path = os.environ.get(
                "WH0_MOGE_MODEL",
                str(weights / "models" / "Ruicheng" / "moge-2-vitl" / "model.pt"),
            )
        if not self.mano_path:
            self.mano_path = str(weights)
        if not self.droid_weights_path:
            self.droid_weights_path = os.environ.get(
                "WH0_DROID_WEIGHTS",
                str(weights / "external" / "droid.pth"),
            )

    @classmethod
    def from_args(cls, args: Optional[object] = None) -> "HandReconConfig":
        if args is None:
            return cls()

        return cls(
            hawor_model_path=getattr(args, "hawor_model_path", "") or "",
            detector_path=getattr(args, "detector_path", "") or "",
            moge_model_path=getattr(
                args,
                "moge_model_path",
                os.environ.get("WH0_MOGE_MODEL", "Ruicheng/moge-2-vitl"),
            ),
            mano_path=getattr(args, "mano_path", "") or "",
            droid_weights_path=getattr(args, "droid_weights_path", "") or "",
        )


# Backward-compatible alias used by older scripts.
Config = HandReconConfig

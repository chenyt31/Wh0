"""Hand detection for box placement and occlusion-aware image editing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

from .box_editor import rects_overlap
from .box_placement import normalized_rects_to_pixels

logger = logging.getLogger(__name__)

Rect = Tuple[int, int, int, int]


def rect_overlaps_hands(rect: Rect, hand_rects: List[Rect]) -> bool:
    return any(rects_overlap(rect, h) for h in hand_rects)


def _scale_rect(
    rect: Rect,
    *,
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> Rect:
    if src_w <= 0 or src_h <= 0:
        return rect
    sx = dst_w / src_w
    sy = dst_h / src_h
    left, top, right, bottom = rect
    return (
        int(left * sx),
        int(top * sy),
        int(right * sx),
        int(bottom * sy),
    )


def _expand_rect(
    rect: Rect,
    *,
    img_w: int,
    img_h: int,
    padding_frac: float,
) -> Rect:
    if padding_frac <= 0:
        return rect
    left, top, right, bottom = rect
    pad_x = int((right - left) * padding_frac)
    pad_y = int((bottom - top) * padding_frac)
    left = max(0, left - pad_x)
    top = max(0, top - pad_y)
    right = min(img_w, right + pad_x)
    bottom = min(img_h, bottom + pad_y)
    if right <= left or bottom <= top:
        return rect
    return (left, top, right, bottom)


class HandDetector:
    """
    Detect visible hands in a scene image.

    Uses a YOLO hand detector when available; otherwise falls back to static
    normalized hand zones from config (legacy behaviour).
    """

    def __init__(
        self,
        *,
        model_path: str = "",
        device: str = "cuda",
        conf_threshold: float = 0.35,
        enable: bool = True,
        padding_frac: float = 0.05,
        fallback_zones_norm: Optional[List[List[float]]] = None,
        use_fallback_zones: bool = False,
    ):
        self.model_path = (model_path or "").strip()
        self.device = device
        self.conf_threshold = float(conf_threshold)
        self.enable = bool(enable) and bool(self.model_path)
        self.padding_frac = max(0.0, float(padding_frac))
        self.fallback_zones_norm = list(fallback_zones_norm or [])
        self.use_fallback_zones = bool(use_fallback_zones)
        self._model = None
        self._load_failed = False

    def _load_model(self) -> bool:
        if self._model is not None:
            return True
        if self._load_failed or not self.enable:
            return False
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.warning(
                "ultralytics not installed — hand detection disabled; "
                "install with: uv pip install ultralytics"
            )
            self._load_failed = True
            return False

        path = Path(self.model_path)
        if not path.is_file():
            logger.warning("Hand detector weights not found: %s", path)
            self._load_failed = True
            return False

        try:
            self._model = YOLO(str(path))
            logger.debug("Loaded hand detector: %s", path)
            return True
        except Exception as exc:
            logger.warning("Failed to load hand detector %s: %s", path, exc)
            self._load_failed = True
            return False

    def release(self) -> None:
        self._model = None

    def _fallback_rects(self, img_w: int, img_h: int) -> List[Rect]:
        if not self.use_fallback_zones or not self.fallback_zones_norm:
            return []
        return normalized_rects_to_pixels(
            self.fallback_zones_norm, img_w, img_h
        )

    def detect(
        self,
        image_path: str,
        *,
        target_w: int,
        target_h: int,
    ) -> List[Rect]:
        """
        Return hand bounding boxes in ``target_w`` x ``target_h`` pixel coords.
        """
        if not self.enable:
            return self._fallback_rects(target_w, target_h)

        if not self._load_model():
            return self._fallback_rects(target_w, target_h)

        try:
            with Image.open(image_path) as im:
                im = im.convert("RGB")
                orig_w, orig_h = im.size
        except OSError as exc:
            logger.warning("Cannot open image for hand detection: %s", exc)
            return self._fallback_rects(target_w, target_h)

        import numpy as np

        results = self._model.predict(
            source=np.array(im),
            conf=self.conf_threshold,
            verbose=False,
            device=self.device,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        boxes = results[0].boxes.xyxy.cpu().numpy()
        out: List[Rect] = []
        for box in boxes:
            left, top, right, bottom = [int(v) for v in box[:4]]
            scaled = _scale_rect(
                (left, top, right, bottom),
                src_w=orig_w,
                src_h=orig_h,
                dst_w=target_w,
                dst_h=target_h,
            )
            expanded = _expand_rect(
                scaled,
                img_w=target_w,
                img_h=target_h,
                padding_frac=self.padding_frac,
            )
            out.append(expanded)
        return out

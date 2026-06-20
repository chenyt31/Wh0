"""
Qwen3-VL bbox coordinate conversion.

Coordinate systems
------------------
LLM output (stored in JSON / ObjectRef.bbox_2d):
  - Qwen3-VL relative grid, range [0, 1000] on the ORIGINAL image file
  - Format: [x_min, y_min, x_max, y_max]
  - Origin top-left; x → right, y → down
  - NOT pixel coordinates

Pixel space (used by PIL / box_editor):
  - Integer (left, top, right, bottom) on the original image
  - left/top inclusive; width = right - left, height = bottom - top

Conversion (same convention as wm_h/video/common.py):
  x_px = x_qwen / 1000 * image_width
  y_px = y_qwen / 1000 * image_height
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

QWEN3_VL_BBOX_COORD_MAX = 1000
BBOX_COORD_SYSTEM = "qwen3_vl_1000"

PixelRect = Tuple[int, int, int, int]


def parse_bbox_2d(raw: Any) -> Optional[List[int]]:
    """Parse bbox from LLM object field bbox_2d / bbox."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        vals = raw.get("bbox_2d") or raw.get("bbox")
    else:
        vals = raw
    if not isinstance(vals, (list, tuple)) or len(vals) < 4:
        return None
    try:
        coords = [int(round(float(v))) for v in vals[:4]]
    except (TypeError, ValueError):
        return None
    return normalize_bbox_qwen(coords)


def normalize_bbox_qwen(coords: List[int]) -> Optional[List[int]]:
    """Clamp to [0, 1000] and enforce x_min < x_max, y_min < y_max."""
    x1, y1, x2, y2 = coords
    x1 = max(0, min(QWEN3_VL_BBOX_COORD_MAX, x1))
    x2 = max(0, min(QWEN3_VL_BBOX_COORD_MAX, x2))
    y1 = max(0, min(QWEN3_VL_BBOX_COORD_MAX, y1))
    y2 = max(0, min(QWEN3_VL_BBOX_COORD_MAX, y2))
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return [x1, y1, x2, y2]


def qwen_bbox_to_pixel_rect(
    bbox_2d: List[int],
    image_width: int,
    image_height: int,
    *,
    min_size_px: int = 16,
) -> Optional[PixelRect]:
    """Convert Qwen3-VL bbox_2d → pixel (left, top, right, bottom)."""
    norm = normalize_bbox_qwen(bbox_2d)
    if not norm or image_width < 1 or image_height < 1:
        return None

    x1, y1, x2, y2 = norm
    left = _qwen_to_pixel_x(x1, image_width)
    top = _qwen_to_pixel_y(y1, image_height)
    right = _qwen_to_pixel_x(x2, image_width)
    bottom = _qwen_to_pixel_y(y2, image_height)

    if right <= left:
        right = min(image_width, left + min_size_px)
    if bottom <= top:
        bottom = min(image_height, top + min_size_px)

    left = max(0, min(left, image_width - 1))
    top = max(0, min(top, image_height - 1))
    right = max(left + 1, min(right, image_width))
    bottom = max(top + 1, min(bottom, image_height))

    if (right - left) < min_size_px or (bottom - top) < min_size_px:
        return None
    return (left, top, right, bottom)


def pixel_rect_to_qwen_bbox(
    rect: PixelRect,
    image_width: int,
    image_height: int,
) -> List[int]:
    """Convert pixel rect → Qwen3-VL bbox_2d (for logging / manifest)."""
    left, top, right, bottom = rect
    if image_width < 1 or image_height < 1:
        return [0, 0, 0, 0]
    x1 = int(round(left / image_width * QWEN3_VL_BBOX_COORD_MAX))
    y1 = int(round(top / image_height * QWEN3_VL_BBOX_COORD_MAX))
    x2 = int(round(right / image_width * QWEN3_VL_BBOX_COORD_MAX))
    y2 = int(round(bottom / image_height * QWEN3_VL_BBOX_COORD_MAX))
    out = normalize_bbox_qwen([x1, y1, x2, y2])
    return out or [0, 0, 0, 0]


def _qwen_to_pixel_x(x_qwen: int, image_width: int) -> int:
    px = x_qwen / QWEN3_VL_BBOX_COORD_MAX * image_width
    return int(round(max(0, min(image_width, px))))


def _qwen_to_pixel_y(y_qwen: int, image_height: int) -> int:
    py = y_qwen / QWEN3_VL_BBOX_COORD_MAX * image_height
    return int(round(max(0, min(image_height, py))))

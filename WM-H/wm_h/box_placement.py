"""Random non-overlapping edit boxes (instruction spatial relations are NOT staged here)."""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

from .box_editor import align_edit_dim, rects_overlap
from .schema import ObjectRef

Rect = Tuple[int, int, int, int]


def _clamp_rect(
    left: int,
    top: int,
    width: int,
    height: int,
    img_w: int,
    img_h: int,
    min_box_px: int,
) -> Optional[Rect]:
    width = max(min_box_px, int(width))
    height = max(min_box_px, int(height))
    if width > img_w or height > img_h:
        return None
    left = max(0, min(int(left), img_w - width))
    top = max(0, min(int(top), img_h - height))
    right = left + width
    bottom = top + height
    if right - left < min_box_px or bottom - top < min_box_px:
        return None
    return (left, top, right, bottom)


def normalized_rects_to_pixels(
    norm_rects: List[List[float]],
    img_w: int,
    img_h: int,
) -> List[Rect]:
    """Convert [x0,y0,x1,y1] in 0–1 image coords to pixel rects."""
    out: List[Rect] = []
    for raw in norm_rects:
        if not raw or len(raw) < 4:
            continue
        x0, y0, x1, y1 = float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])
        left = int(min(x0, x1) * img_w)
        top = int(min(y0, y1) * img_h)
        right = int(max(x0, x1) * img_w)
        bottom = int(max(y0, y1) * img_h)
        if right > left and bottom > top:
            out.append((left, top, right, bottom))
    return out


def _sample_skewed_int(
    rng: random.Random,
    lo: int,
    hi: int,
    *,
    skew: float = 1.0,
) -> int:
    """Sample with more mass at both ends when skew < 1 (more size variety)."""
    if lo >= hi:
        return lo
    if skew >= 0.99:
        return rng.randint(lo, hi)
    if rng.random() < 0.35:
        return rng.randint(lo, hi)
    u = rng.random()
    if rng.random() < 0.5:
        t = u ** skew
    else:
        t = 1.0 - (1.0 - u) ** skew
    return int(lo + t * (hi - lo))


def sample_box_size(
    rng: random.Random,
    *,
    mask_width: int,
    mask_height: int,
    width_range: Optional[Tuple[int, int]] = None,
    height_range: Optional[Tuple[int, int]] = None,
    min_box_px: int = 16,
    size_skew: float = 1.0,
) -> Tuple[int, int]:
    if width_range:
        w = _sample_skewed_int(
            rng, width_range[0], width_range[1], skew=size_skew
        )
    else:
        w = mask_width
    if height_range:
        h = _sample_skewed_int(
            rng, height_range[0], height_range[1], skew=size_skew
        )
    else:
        h = mask_height
    w = align_edit_dim(max(min_box_px, w))
    h = align_edit_dim(max(min_box_px, h))
    return w, h


def _object_center_ranges(
    rng: random.Random,
    obj_idx: int,
    n_objects: int,
    center_x_range: Tuple[float, float],
    center_y_range: Tuple[float, float],
    *,
    position_jitter: float = 0.0,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Per-object sub-zones + jitter so multi-object layouts spread across the desk."""
    cx0, cx1 = center_x_range
    cy0, cy1 = center_y_range

    if n_objects > 1:
        span = cx1 - cx0
        step = span / n_objects
        anchor = center_x_range[0] + (obj_idx + 0.5) * step
        half_width = min(step * 0.20, 0.08)
        cx0 = max(center_x_range[0], anchor - half_width)
        cx1 = min(center_x_range[1], anchor + half_width)
        position_jitter = min(position_jitter, step * 0.15)

    jx = rng.uniform(-position_jitter, position_jitter)
    jy = rng.uniform(-position_jitter, position_jitter)

    cx0 = max(0.03, min(0.94, cx0 + jx))
    cx1 = max(cx0 + 0.06, min(0.97, cx1 + jx))
    cy0 = max(0.04, min(0.86, cy0 + jy))
    cy1 = max(cy0 + 0.06, min(0.92, cy1 + jy))
    return (cx0, cx1), (cy0, cy1)


def _rect_overlaps_any(rect: Rect, zones: List[Rect]) -> bool:
    return any(rects_overlap(rect, z) for z in zones)


def random_rect_overlapping_hands(
    rng: random.Random,
    img_w: int,
    img_h: int,
    box_w: int,
    box_h: int,
    hand_zones: List[Rect],
    avoid_rects: List[Rect],
    min_box_px: int = 16,
    max_attempts: int = 200,
) -> Optional[Rect]:
    """Place box on desk so it extends into a hand zone (partial occlusion)."""
    if not hand_zones:
        return None
    for _ in range(max_attempts):
        hz = rng.choice(hand_zones)
        h_left, h_top, h_right, h_bottom = hz
        zone_w = max(min_box_px, h_right - h_left)
        cx = rng.uniform(
            h_left + box_w * 0.15,
            h_right - box_w * 0.15,
        )
        if cx < box_w / 2 or cx > img_w - box_w / 2:
            cx = rng.uniform(
                max(box_w / 2, h_left),
                min(img_w - box_w / 2, h_right),
            )
        overlap_depth = rng.uniform(0.18, 0.55)
        bottom = int(h_top + (h_bottom - h_top) * rng.uniform(0.45, 1.0))
        top = int(bottom - box_h * (1.0 - overlap_depth * 0.35))
        left = int(cx - box_w / 2)
        rect = _clamp_rect(left, top, box_w, box_h, img_w, img_h, min_box_px)
        if (
            rect
            and _rect_overlaps_any(rect, hand_zones)
            and not any(rects_overlap(rect, r) for r in avoid_rects)
        ):
            return rect
    return None


def random_rect_avoiding(
    rng: random.Random,
    img_w: int,
    img_h: int,
    box_w: int,
    box_h: int,
    center_x_range: Tuple[float, float],
    center_y_range: Tuple[float, float],
    exclude_rects: List[Rect],
    min_box_px: int = 16,
    max_attempts: int = 200,
) -> Optional[Rect]:
    for _ in range(max_attempts):
        min_cx = box_w / 2
        max_cx = img_w - box_w / 2
        min_cy = box_h / 2
        max_cy = img_h - box_h / 2
        lo_x = max(center_x_range[0] * img_w, min_cx)
        hi_x = min(center_x_range[1] * img_w, max_cx)
        lo_y = max(center_y_range[0] * img_h, min_cy)
        hi_y = min(center_y_range[1] * img_h, max_cy)
        if lo_x > hi_x:
            lo_x, hi_x = min_cx, max_cx
        if lo_y > hi_y:
            lo_y, hi_y = min_cy, max_cy
        cx = rng.uniform(lo_x, hi_x)
        cy = rng.uniform(lo_y, hi_y)
        left = int(cx - box_w / 2)
        top = int(cy - box_h / 2)
        rect = _clamp_rect(left, top, box_w, box_h, img_w, img_h, min_box_px)
        if rect and not any(rects_overlap(rect, r) for r in exclude_rects):
            return rect
    return None


def assign_edit_rects(
    objects: List[ObjectRef],
    *,
    img_w: int,
    img_h: int,
    seed: int,
    mask_width: int,
    mask_height: int,
    width_range: Optional[Tuple[int, int]],
    height_range: Optional[Tuple[int, int]],
    center_x_range: Tuple[float, float],
    center_y_range: Tuple[float, float],
    min_box_px: int,
    exclude_rects: Optional[List[Rect]] = None,
    hand_zones: Optional[List[Rect]] = None,
    hand_occlusion_probability: float = 0.0,
    size_skew: float = 1.0,
    position_jitter: float = 0.0,
    allow_rect_overlap: bool = False,
    max_attempts: int = 200,
) -> List[Tuple[ObjectRef, Rect]]:
    """
    Random independent placements for each instruction object.

    Two-object instructions (e.g. slide X beside Y) are *future* tasks — we only
    scatter objects on the desk, not pre-arrange them per the preposition.
    """
    rng = random.Random(seed)
    avoid = list(exclude_rects or [])
    placed_rects: List[Rect] = []
    out: List[Tuple[ObjectRef, Rect]] = []
    active = [o for o in objects if o.noun]

    for obj_idx, obj in enumerate(active):
        bw, bh = sample_box_size(
            rng,
            mask_width=mask_width,
            mask_height=mask_height,
            width_range=width_range,
            height_range=height_range,
            min_box_px=min_box_px,
            size_skew=size_skew,
        )
        overlap_avoid = avoid if allow_rect_overlap else avoid + placed_rects
        rect: Optional[Rect] = None
        if (
            hand_zones
            and hand_occlusion_probability > 0
            and rng.random() < hand_occlusion_probability
        ):
            rect = random_rect_overlapping_hands(
                rng,
                img_w,
                img_h,
                bw,
                bh,
                hand_zones,
                overlap_avoid,
                min_box_px=min_box_px,
                max_attempts=max_attempts,
            )
        if rect is None:
            obj_cx, obj_cy = _object_center_ranges(
                rng,
                obj_idx,
                len(active),
                center_x_range,
                center_y_range,
                position_jitter=position_jitter,
            )
            rect = random_rect_avoiding(
                rng,
                img_w,
                img_h,
                bw,
                bh,
                obj_cx,
                obj_cy,
                overlap_avoid,
                min_box_px=min_box_px,
                max_attempts=max_attempts,
            )
        if rect is None:
            return []
        placed_rects.append(rect)
        out.append((obj, rect))

    return out

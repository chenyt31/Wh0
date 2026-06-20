from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import find_peaks

from .episodic import write_episode_frame_index


EPISODE_PATTERN = re.compile(r"^(.+)_ep_(\d{6})\.npy$")


def _load_annotation(path: Path) -> dict[str, Any]:
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.ndim == 0:
        item = payload.item()
        if isinstance(item, dict):
            return item
    raise ValueError(f"Unsupported annotation payload: {path}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _right_hand_ratio(annotation: dict[str, Any]) -> float:
    kept_frames = annotation.get("right", {}).get("kept_frames", [])
    if len(kept_frames) == 0:
        return 0.0
    kept = np.asarray(kept_frames, dtype=np.float32)
    return float(kept.mean())


def _extract_speed(annotation: dict[str, Any]) -> np.ndarray:
    transl = np.asarray(annotation.get("right", {}).get("transl_worldspace", []), dtype=np.float32)
    if transl.ndim != 2 or len(transl) == 0:
        return np.asarray([], dtype=np.float32)
    deltas = np.diff(transl, axis=0, prepend=transl[[0]])
    return np.linalg.norm(deltas, axis=1)


def _detect_split_points(speed: np.ndarray) -> list[int]:
    if len(speed) < 30:
        return []
    smooth = np.convolve(speed, np.ones(5, dtype=np.float32) / 5.0, mode="same")
    peaks, props = find_peaks(-smooth, prominence=0.002, distance=8)
    valid = [int(idx) + 5 for idx in peaks if 15 <= idx < len(speed) - 15 and smooth[idx] < 0.01]
    if not valid:
        return []
    if len(valid) == 1:
        return [min(valid[0], len(speed) - 1)]
    return [min(valid[0], len(speed) - 1), min(valid[-1], len(speed) - 1)]


def _extract_instruction(annotation: dict[str, Any]) -> str:
    text_payload = annotation.get("text", {})
    candidates = text_payload.get("right", []) if isinstance(text_payload, dict) else []
    if not candidates:
        return ""
    first = candidates[0]
    if isinstance(first, (tuple, list)) and first:
        return str(first[0]).strip()
    return str(first).strip()


def _instruction_segments(text: str, minima_count: int) -> list[str]:
    cleaned = text.strip()
    parts = [part.strip() for part in cleaned.split(" and ", 1)] if " and " in cleaned.lower() else [cleaned]
    first = parts[0] if parts and parts[0] else cleaned
    second = parts[1] if len(parts) > 1 and parts[1] else "lift it"
    if minima_count == 1:
        return [first, "return home"]
    return [first, second if not second.lower().startswith("lift ") else "lift it", "return home"]


def _slice_value(value: Any, frame_slice: slice) -> Any:
    if isinstance(value, np.ndarray) and value.shape[:1]:
        return value[frame_slice]
    if isinstance(value, list) and len(value) >= frame_slice.stop:
        return value[frame_slice]
    if isinstance(value, dict):
        return {key: _slice_value(item, frame_slice) for key, item in value.items()}
    return value


def _slice_annotation(
    annotation: dict[str, Any],
    frame_start: int,
    frame_end: int,
    instruction: str,
) -> dict[str, Any]:
    frame_slice = slice(frame_start, frame_end + 1)
    sliced = {key: _slice_value(value, frame_slice) for key, value in annotation.items()}
    total_frames = frame_end - frame_start + 1
    sliced["total_frames"] = total_frames
    text_entry = (instruction, (0, total_frames - 1))
    if isinstance(sliced.get("text"), dict):
        for hand in ("left", "right"):
            if hand in sliced["text"]:
                sliced["text"][hand] = [text_entry]
    return sliced


def process_wmh_annotations(
    annotation_dir: str | Path,
    *,
    right_hand_threshold: float = 0.5,
    max_episodes: int = -1,
    output_dir_name: str = "episodic_annotations_split",
) -> dict[str, int]:
    annotation_root = Path(annotation_dir).expanduser().resolve()
    split_root = annotation_root.parent / output_dir_name
    split_root.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(path for path in annotation_root.glob("*_ep_*.npy"))
    segments_info: list[dict[str, Any]] = []
    stats = {"total": len(npy_files), "kept": 0, "filtered": 0, "segments": 0}

    processed = 0
    for path in npy_files:
        if max_episodes != -1 and processed >= max_episodes:
            break
        annotation = _load_annotation(path)
        if _right_hand_ratio(annotation) < right_hand_threshold:
            stats["filtered"] += 1
            continue
        split_points = _detect_split_points(_extract_speed(annotation))
        if len(split_points) not in {1, 2}:
            stats["filtered"] += 1
            continue

        match = EPISODE_PATTERN.match(path.name)
        if not match:
            stats["filtered"] += 1
            continue
        prefix, original_episode = match.groups()
        instruction = _extract_instruction(annotation)
        segment_instructions = _instruction_segments(instruction, len(split_points))
        boundaries = [0, *split_points, len(annotation.get("video_decode_frame", [])) - 1]

        stats["kept"] += 1
        processed += 1

        for segment_idx in range(len(boundaries) - 1):
            frame_start = boundaries[segment_idx]
            frame_end = boundaries[segment_idx + 1]
            new_episode_number = f"{int(original_episode) + segment_idx:06d}"
            episode_id = f"{prefix}_ep_{new_episode_number}"
            segment_instruction = segment_instructions[segment_idx]
            segment_payload = _slice_annotation(annotation, frame_start, frame_end, segment_instruction)
            np.save(split_root / f"{episode_id}.npy", segment_payload, allow_pickle=True)
            segments_info.append(
                {
                    "segment_ep_name": episode_id,
                    "original_instruction": instruction,
                    "instruction": segment_instruction,
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "total_frames": frame_end - frame_start + 1,
                    "segment_idx": segment_idx,
                    "minima_positions": split_points,
                    "original_ep": original_episode,
                    "npy_file": f"{episode_id}.npy",
                }
            )
            stats["segments"] += 1

    if segments_info:
        with (split_root / "segments_info.json").open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(segments_info), handle, ensure_ascii=False, indent=2)
    write_episode_frame_index(split_root, annotation_root.parent / "episode_frame_index.npz")
    return stats


def scan_wmh_annotation_roots(
    base_dir: str | Path,
    *,
    right_hand_threshold: float = 0.5,
    max_episodes: int = -1,
) -> list[dict[str, Any]]:
    base = Path(base_dir).expanduser().resolve()
    roots = sorted(path for path in base.rglob("episodic_annotations") if path.is_dir())
    return [
        {
            "annotation_dir": str(root),
            "stats": process_wmh_annotations(
                root,
                right_hand_threshold=right_hand_threshold,
                max_episodes=max_episodes,
            ),
        }
        for root in roots
    ]

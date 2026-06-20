from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def _load_annotation(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return data.item() if isinstance(data, np.ndarray) else data


def _frame_size_from_intrinsics(intrinsics: np.ndarray) -> tuple[int, int]:
    return int(round(intrinsics[0, 2] * 2)), int(round(intrinsics[1, 2] * 2))


def _compute_fov_x(intrinsics: np.ndarray, width: int) -> float:
    return float(2 * np.arctan(width / (2 * intrinsics[0, 0])) * 180 / np.pi)


def _valid_frame_indices(hand_block: dict) -> list[int]:
    kept = hand_block.get("kept_frames", [])
    return [index for index, keep in enumerate(kept) if keep == 1]


def _empty_hand() -> dict:
    return {
        "hand_pose": np.zeros((15, 3, 3), dtype=np.float32),
        "global_orient": np.eye(3, dtype=np.float32),
        "transl": np.zeros(3, dtype=np.float32),
        "beta": np.zeros(10, dtype=np.float32),
    }


def _extract_hand_frame(episode: dict, hand: str, hand_index: int) -> dict:
    block = episode.get(hand, {})
    if not isinstance(block, dict):
        return _empty_hand()
    valid = _valid_frame_indices(block)
    if hand_index >= len(valid):
        return _empty_hand()
    frame = valid[hand_index]
    return {
        "hand_pose": np.asarray(block.get("hand_pose", np.zeros((1, 15, 3, 3)))[frame], dtype=np.float32),
        "global_orient": np.asarray(block.get("global_orient_camspace", np.eye(3)[None])[frame], dtype=np.float32),
        "transl": np.asarray(block.get("transl_worldspace", np.zeros((1, 3)))[frame], dtype=np.float32),
        "beta": np.asarray(block.get("beta", np.zeros(10)), dtype=np.float32),
    }


def _extract_video_frame(video_path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
        return frame
    finally:
        cap.release()


def _extract_instruction(episode: dict) -> str:
    text = episode.get("text", "")
    if isinstance(text, dict):
        right = text.get("right", [])
        if right:
            value = right[0][0] if isinstance(right[0], tuple) else str(right[0])
            return f"Left hand: None. Right hand: {value}."
        return "Left hand: None. Right hand: None."
    if isinstance(text, str):
        return f"Left hand: None. Right hand: {text}."
    return f"Left hand: None. Right hand: {text}."


def save_annotation_frame(
    video_path: str,
    annotation_npy_path: str,
    frame_idx: int,
    output_dir: str,
    hand_index: int,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    episode = _load_annotation(annotation_npy_path)
    intrinsics = np.asarray(episode["intrinsics"], dtype=np.float32)
    width, height = _frame_size_from_intrinsics(intrinsics)
    hand_data = {
        "left": {0: _extract_hand_frame(episode, "left", hand_index)},
        "right": {0: _extract_hand_frame(episode, "right", hand_index)},
        "fov_x": _compute_fov_x(intrinsics, width),
    }
    np.save(output / "image.npy", hand_data, allow_pickle=True)

    frame = _extract_video_frame(video_path, frame_idx)
    cv2.imwrite(str(output / "image.jpg"), frame)

    info = {
        "video_path": video_path,
        "annotation_npy_path": annotation_npy_path,
        "frame_idx": frame_idx,
        "hand_index": hand_index,
        "instruction": _extract_instruction(episode),
        "text": episode.get("text", ""),
        "fov_x": hand_data["fov_x"],
        "image_width": width,
        "image_height": height,
    }
    (output / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

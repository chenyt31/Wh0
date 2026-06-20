from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np


EPISODE_PATTERN = re.compile(r"^(.+)_ep_(\d{6})\.npy$")


def iter_episode_annotations(root: str | Path) -> list[Path]:
    base = Path(root).expanduser().resolve()
    return sorted(
        path
        for path in base.rglob("*.npy")
        if "_ep_" in path.name and EPISODE_PATTERN.match(path.name)
    )


def _load_annotation_dict(path: Path) -> dict | np.ndarray:
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.ndim == 0:
        item = payload.item()
        if isinstance(item, dict):
            return item
    return payload


def frame_count(path: Path) -> int:
    payload = _load_annotation_dict(path)
    if isinstance(payload, dict):
        if "total_frames" in payload:
            return int(payload["total_frames"])
        return len(payload.get("video_decode_frame", []))
    return int(len(payload))


def build_episode_frame_index(annotation_files: Iterable[str | Path]) -> tuple[np.ndarray, np.ndarray]:
    files = [Path(path) for path in annotation_files]
    episode_ids: list[str] = []
    rows: list[tuple[int, int]] = []
    for episode_idx, path in enumerate(files):
        episode_ids.append(path.stem)
        for frame_idx in range(frame_count(path)):
            rows.append((episode_idx, frame_idx))
    index_frame_pair = np.asarray(rows, dtype=np.int64).reshape(0, 2) if not rows else np.asarray(rows, dtype=np.int64)
    return index_frame_pair, np.asarray(episode_ids, dtype=object)


def write_episode_frame_index(
    annotation_root: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    root = Path(annotation_root).expanduser().resolve()
    files = iter_episode_annotations(root)
    index_frame_pair, index_to_episode_id = build_episode_frame_index(files)
    target = Path(output_path).expanduser().resolve() if output_path else root / "episode_frame_index.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, index_frame_pair=index_frame_pair, index_to_episode_id=index_to_episode_id)
    return target


def verify_episode_frame_index(index_path: str | Path) -> bool:
    payload = np.load(Path(index_path).expanduser().resolve(), allow_pickle=True)
    index_frame_pair = payload["index_frame_pair"]
    index_to_episode_id = payload["index_to_episode_id"]
    if index_frame_pair.ndim != 2 or index_frame_pair.shape[1] != 2:
        return False
    if len(index_frame_pair) == 0:
        return len(index_to_episode_id) == 0
    return int(index_frame_pair[:, 0].max()) == len(index_to_episode_id) - 1 and np.all(index_frame_pair[:, 1] >= 0)

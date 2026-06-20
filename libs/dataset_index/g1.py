from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def find_g1_episodes(root_dir: str | Path) -> list[Path]:
    root = Path(root_dir).expanduser().resolve()
    episodes: list[Path] = []
    for episode_dir in sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("episode_")):
        if (episode_dir / "data.json").exists():
            episodes.append(episode_dir)
    for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if task_dir.name.startswith("episode_"):
            continue
        for episode_dir in sorted(path for path in task_dir.iterdir() if path.is_dir() and path.name.startswith("episode_")):
            if (episode_dir / "data.json").exists():
                episodes.append(episode_dir)
    return episodes


def _load_episode_length(episode_dir: Path) -> int:
    with (episode_dir / "data.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return len(payload.get("data", []))


def _frames_with_images(episode_dir: Path, total_frames: int) -> list[int]:
    colors_dir = episode_dir / "colors"
    return [
        frame_idx
        for frame_idx in range(total_frames)
        if (colors_dir / f"{frame_idx:06d}_color_0.jpg").exists()
    ]


def _uniform_sample(frame_indices: list[int], target_frames: int | None) -> list[int]:
    if target_frames is None or len(frame_indices) <= target_frames:
        return frame_indices
    sampled = np.linspace(0, len(frame_indices) - 1, num=target_frames, dtype=int)
    return [frame_indices[idx] for idx in sampled.tolist()]


def build_g1_training_index(
    root_dir: str | Path,
    target_frames: int | None = 81,
    use_all_frames: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    root = Path(root_dir).expanduser().resolve()
    episode_paths: list[str] = []
    sample_rows: list[tuple[int, int, int]] = []
    for episode_idx, episode_dir in enumerate(find_g1_episodes(root)):
        total_frames = _load_episode_length(episode_dir)
        valid_frames = _frames_with_images(episode_dir, total_frames)
        selected_frames = valid_frames if use_all_frames else _uniform_sample(valid_frames, target_frames)
        if not selected_frames:
            continue
        episode_paths.append(str(episode_dir.relative_to(root)))
        stored_episode_idx = len(episode_paths) - 1
        for sample_idx, frame_idx in enumerate(selected_frames):
            sample_rows.append((stored_episode_idx, sample_idx, frame_idx))
    sample_indices = np.asarray(sample_rows, dtype=np.int64).reshape(0, 3) if not sample_rows else np.asarray(sample_rows, dtype=np.int64)
    return np.asarray(episode_paths, dtype=object), sample_indices


def write_g1_training_index(
    root_dir: str | Path,
    output_path: str | Path | None = None,
    target_frames: int | None = 81,
    use_all_frames: bool = False,
) -> Path:
    root = Path(root_dir).expanduser().resolve()
    episode_paths, sample_indices = build_g1_training_index(
        root,
        target_frames=target_frames,
        use_all_frames=use_all_frames,
    )
    target = Path(output_path).expanduser().resolve() if output_path else root / "training_index.npz"
    np.savez(target, episode_paths=episode_paths, sample_indices=sample_indices)
    return target


def verify_g1_training_index(index_path: str | Path) -> bool:
    payload = np.load(Path(index_path).expanduser().resolve(), allow_pickle=True)
    episode_paths = payload["episode_paths"]
    sample_indices = payload["sample_indices"]
    if sample_indices.ndim != 2 or sample_indices.shape[1] != 3:
        return False
    if len(sample_indices) == 0:
        return True
    return int(sample_indices[:, 0].max()) < len(episode_paths) and np.all(sample_indices[:, 1:] >= 0)

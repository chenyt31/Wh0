"""RoboCasa GR-1 LeRobot reader for VITRA.

The files on disk retain the native LeRobot contract (24-D absolute GR-1
state/action). Samples exposed to VITRA are converted through
``embodiments.gr1.vitra_adapter``: state is a sparse 102-D MANO vector padded
to 212-D, while each action row is a 102-D full dual-hand MANO action padded
to VITRA's 192-D action width. Only the state is sparse; all valid action
MANO slots are supervised, with un-driven slots set to canonical zero.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset

# Training is normally launched from the nested Wh0 checkout. Make the
# workspace-level embodiment package importable without requiring callers to
# remember a second PYTHONPATH entry.
_PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from embodiments.gr1.vitra_adapter import (
    ACTION_DIM,
    MANO_DIM,
    MODEL_STATE_DIM,
    NATIVE_DIM,
    action_mask,
    model_state_from_mano,
    native24_to_mano102,
    native_target_horizon_to_action192,
    native_targets_to_action192,
    state_mask,
    validate_native24,
    validate_transform,
)
from vitra.datasets.dataset_utils import calculate_fov, compute_new_intrinsics_resize
from vitra.utils.data_utils import (
    GaussianNormalizer,
    read_dataset_statistics,
    resize_short_side_to_target,
)


HAND_WIDTH = 51
RAW_STATE_WIDTH = MANO_DIM
MODEL_STATE_WIDTH = MODEL_STATE_DIM
ACTION_WIDTH = ACTION_DIM

# Names retained for deployment callers and older experiment scripts.
ACTIVE_CHANNELS = np.flatnonzero(state_mask()).astype(np.int64)
ACTION_CHANNELS = np.arange(MANO_DIM, dtype=np.int64)


def encode_state24(state24: np.ndarray) -> np.ndarray:
    """Encode one or more native absolute states into sparse MANO states."""

    return native24_to_mano102(state24)


def encode_action24(
    action24: np.ndarray,
    current_state24: np.ndarray | None = None,
) -> np.ndarray:
    """Encode native targets as VITRA actions.

    ``current_state24`` is required because native GR-1 actions are absolute
    targets while VITRA wrist actions are relative. Passing no current state
    is kept as a compatibility path for callers that only need a native row's
    absolute sparse MANO representation.
    """

    if current_state24 is None:
        return native24_to_mano102(action24)
    return native_targets_to_action192(current_state24, action24)


def decode_action192(action192: np.ndarray) -> np.ndarray:
    """Return the first 102 MANO channels from a VITRA action tensor.

    This function intentionally does not decode a native robot action: a
    current state is needed to integrate relative wrist motion. Use
    ``decode_action_horizon_to_native24`` in the embodiment adapter for
    deployment decoding.
    """

    action = np.asarray(action192, dtype=np.float32)
    if action.ndim == 0 or action.shape[-1] < MANO_DIM:
        raise ValueError(f"Expected at least {MANO_DIM} VITRA channels, got {action.shape}")
    return action[..., :MANO_DIM]


def active_action_mask(length: int, valid: bool = True) -> np.ndarray:
    """Compatibility wrapper for a dense full-MANO action mask."""

    return action_mask(length, valid_rows=length if valid else 0)


def _episode_action_statistics(
    native_state24: np.ndarray,
    native_target24: np.ndarray,
) -> np.ndarray:
    """Return the native transitions represented by all GR-1 sample rows.

    A training window's row zero is ``state[i] -> target[i]``.  Every later
    row is ``target[i+j-1] -> target[i+j]``.  Keeping both transition classes
    gives the local normalizer the same absolute-to-relative convention as
    :meth:`GR1LeRobotEpisodeDataset.__getitem__`, while remaining independent
    of the configured window length.
    """

    states = np.asarray(native_state24, dtype=np.float32)
    targets = np.asarray(native_target24, dtype=np.float32)
    if states.shape != targets.shape or states.ndim != 2 or states.shape[1] != NATIVE_DIM:
        raise ValueError(
            "native state/target arrays must both have shape [T,24], got "
            f"{states.shape}/{targets.shape}"
        )
    first = native_targets_to_action192(states, targets)
    if len(targets) <= 1:
        return first
    adjacent = native_targets_to_action192(targets[:-1], targets[1:])
    return np.concatenate((first, adjacent), axis=0).astype(np.float32)


def _read_fixed_list_column(table: Any, name: str, width: int) -> np.ndarray:
    if name not in table.column_names:
        raise ValueError(f"LeRobot parquet is missing {name!r}")
    values = np.asarray(table[name].to_pylist(), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != width:
        raise ValueError(f"{name} must have shape [T,{width}], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return values


def discover_gr1_episode_metadata(root: Path) -> list[Path]:
    """Find all GR-1 LeRobot source manifests without duplicating views."""

    root = Path(root).expanduser()
    candidates = (root / "embodiments/gr1/samples", root)

    def unique(paths: list[Path]) -> list[Path]:
        resolved: dict[Path, Path] = {}
        for path in paths:
            resolved.setdefault(path.resolve(), path)
        return sorted(resolved.values(), key=lambda path: str(path))

    for samples_root in candidates:
        paths = unique(list(
            samples_root.glob("*/tactile_calib/lerobot_v21/meta/episode_*_source.json")
        ))
        if paths:
            return paths
    # Also accept a caller pointing directly at one LeRobot root.
    direct = unique(list((root / "meta").glob("episode_*_source.json")))
    if direct:
        return direct
    # A run collection may contain several timestamped data roots. Search
    # only for the canonical GR-1 sample suffix and de-duplicate symlinked
    # views by their resolved path.
    found: dict[Path, Path] = {}
    for path in root.rglob("embodiments/gr1/samples/*/tactile_calib/lerobot_v21/meta/episode_*_source.json"):
        found.setdefault(path.resolve(), path)
    return sorted(found.values(), key=lambda path: str(path))


def _resolve_path(value: str | Path, *, relative_to: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and relative_to is not None:
        path = relative_to / path
    return path.resolve()


def _load_episode_task(lerobot_root: Path, episode_index: int, source: dict[str, Any]) -> str:
    for name in ("episodes.jsonl", "tasks.jsonl"):
        path = lerobot_root / "meta" / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if name == "episodes.jsonl" and int(row.get("episode_index", -1)) == episode_index:
                tasks = row.get("tasks") or []
                if tasks and str(tasks[0]).strip():
                    return str(tasks[0]).strip()
            if name == "tasks.jsonl" and int(row.get("task_index", -1)) == 0:
                task = str(row.get("task", "")).strip()
                if task:
                    return task
    for key in ("instruction", "task"):
        value = str(source.get(key, "")).strip()
        if value:
            return value
    return "Execute the RoboCasa GR-1 task."


def _find_source_archive(source: dict[str, Any], metadata_path: Path) -> Path | None:
    for key in ("tactile_source_robot_action", "source_robot_action", "robot_action"):
        value = source.get(key)
        if not value:
            continue
        path = _resolve_path(value, relative_to=metadata_path.parent)
        if path.is_file():
            return path
    return None


def _load_camera_metadata(
    source: dict[str, Any], metadata_path: Path, *, width: int = 640, height: int = 480
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    """Load K and optional base/camera transforms from manifest or archive."""

    camera = source.get("camera") if isinstance(source.get("camera"), dict) else {}
    raw_k = camera.get("intrinsics", source.get("camera_intrinsics"))
    camera_from_base = camera.get("camera_from_base", source.get("camera_from_base"))
    base_from_camera = camera.get("base_from_camera", source.get("base_from_camera"))
    archive_path = _find_source_archive(source, metadata_path)
    if archive_path is not None:
        with np.load(archive_path, allow_pickle=False) as archive:
            if raw_k is None and "intrinsics" in archive:
                raw_k = np.asarray(archive["intrinsics"]).tolist()
            if camera_from_base is None and "camera_from_base" in archive:
                camera_from_base = np.asarray(archive["camera_from_base"]).tolist()
            if base_from_camera is None and "base_from_camera" in archive:
                base_from_camera = np.asarray(archive["base_from_camera"]).tolist()
    if raw_k is None:
        # Legacy one-episode bundles were generated with this calibrated
        # RoboCasa default. New exports write K explicitly in the manifest.
        raw_k = [
            [515.0, 0.0, (width - 1.0) / 2.0],
            [0.0, 515.0, (height - 1.0) / 2.0],
            [0.0, 0.0, 1.0],
        ]
        source_name = "legacy_default"
    else:
        source_name = "manifest_or_archive"
    intrinsics = np.asarray(raw_k, dtype=np.float32)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError(f"GR-1 camera intrinsics must be finite [3,3], got {intrinsics.shape}")
    camera_from_base_array = None
    base_from_camera_array = None
    if camera_from_base is not None:
        camera_from_base_array = validate_transform(np.asarray(camera_from_base), "camera_from_base")
    if base_from_camera is not None:
        base_from_camera_array = validate_transform(np.asarray(base_from_camera), "base_from_camera")
    if camera_from_base_array is not None and base_from_camera_array is not None:
        if not np.allclose(np.linalg.inv(camera_from_base_array), base_from_camera_array, atol=1e-4):
            raise ValueError("camera_from_base and base_from_camera are inconsistent")
    elif camera_from_base_array is not None:
        base_from_camera_array = np.linalg.inv(camera_from_base_array)
    elif base_from_camera_array is not None:
        camera_from_base_array = np.linalg.inv(base_from_camera_array)
    return intrinsics, camera_from_base_array, base_from_camera_array, {
        "intrinsics_source": source_name,
        "source_archive": None if archive_path is None else str(archive_path),
    }


@dataclass
class _EpisodePaths:
    metadata_path: Path
    lerobot_root: Path
    episode_index: int
    parquet_path: Path
    video_path: Path
    task: str
    fps: float
    intrinsics: np.ndarray
    camera_from_base: np.ndarray | None
    base_from_camera: np.ndarray | None
    source_metadata: dict[str, Any]
    camera_metadata: dict[str, Any]


def _episode_paths(metadata_path: Path) -> _EpisodePaths:
    metadata_path = metadata_path.expanduser().resolve()
    source = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError(f"GR-1 source metadata must be an object: {metadata_path}")
    lerobot_root = metadata_path.parent.parent
    stem = metadata_path.stem.removesuffix("_source")
    try:
        episode_index = int(stem.split("_")[-1])
    except ValueError as error:
        raise ValueError(f"Cannot parse episode index from {metadata_path.name}") from error
    parquet_path = lerobot_root / "data" / f"chunk-{episode_index // 1000:03d}" / f"{stem}.parquet"
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Missing GR-1 LeRobot parquet: {parquet_path}")
    info_path = lerobot_root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    video_key = "observation.images.camera"
    video_template = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    video_rel = video_template.format(
        episode_chunk=episode_index // 1000,
        episode_index=episode_index,
        video_key=video_key,
    )
    video_path = lerobot_root / video_rel
    if not video_path.is_file():
        # A small compatibility fallback for hand-written test fixtures.
        fallback = lerobot_root / f"episode_{episode_index:06d}.mp4"
        if fallback.is_file():
            video_path = fallback
        else:
            raise FileNotFoundError(f"Missing GR-1 LeRobot video: {video_path}")
    fps = float(info.get("fps", source.get("fps", 30.0)))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid GR-1 episode FPS: {fps}")
    intrinsics, camera_from_base, base_from_camera, camera_metadata = _load_camera_metadata(
        source, metadata_path
    )
    return _EpisodePaths(
        metadata_path=metadata_path,
        lerobot_root=lerobot_root,
        episode_index=episode_index,
        parquet_path=parquet_path,
        video_path=video_path,
        task=_load_episode_task(lerobot_root, episode_index, source),
        fps=fps,
        intrinsics=intrinsics,
        camera_from_base=camera_from_base,
        base_from_camera=base_from_camera,
        source_metadata=source,
        camera_metadata=camera_metadata,
    )


class GR1LeRobotEpisodeDataset:
    """State/action aligned reader for one LeRobot v2.1 episode."""

    def __init__(
        self,
        *,
        paths: _EpisodePaths,
        action_future_window_size: int = 16,
        load_images: bool = True,
        target_image_height: int = 224,
        statistics_path: str | None = None,
    ) -> None:
        self.paths = paths
        self.root = paths.lerobot_root
        self.metadata_path = paths.metadata_path
        self.parquet_path = paths.parquet_path
        self.video_path = paths.video_path
        self.instruction = paths.task
        self.fps = paths.fps
        self.camera_intrinsics = paths.intrinsics.copy()
        self.camera_from_base = None if paths.camera_from_base is None else paths.camera_from_base.copy()
        self.base_from_camera = None if paths.base_from_camera is None else paths.base_from_camera.copy()
        self.camera_metadata = dict(paths.camera_metadata)
        self.source_metadata = dict(paths.source_metadata)
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError("GR-1 LeRobot loading requires pyarrow") from error
        table = pq.read_table(self.parquet_path, columns=["observation.state", "action"])
        self.native_state24 = _read_fixed_list_column(table, "observation.state", NATIVE_DIM)
        self.native_target24 = _read_fixed_list_column(table, "action", NATIVE_DIM)
        if self.native_state24.shape != self.native_target24.shape:
            raise ValueError("GR-1 observation.state/action lengths differ")
        validate_native24(self.native_state24)
        validate_native24(self.native_target24)
        self.state102 = native24_to_mano102(self.native_state24)
        # Keep the diagnostic action table aligned with the labels emitted by
        # each training sample.  The first row of a sample is relative to the
        # observed state at that frame; later rows are relative to the
        # preceding absolute target.  Using one episode-level initial state
        # here would make statistics depend on the frame at which the sample
        # happened to be read.
        self.action192_absolute = _episode_action_statistics(
            self.native_state24, self.native_target24
        ).astype(np.float32)
        self.action_future_window_size = int(action_future_window_size)
        if self.action_future_window_size < 0:
            raise ValueError("action_future_window_size must be non-negative")
        self.load_images = bool(load_images)
        self.target_image_height = int(target_image_height)
        if self.target_image_height < 1:
            raise ValueError("target_image_height must be positive")
        if statistics_path is None:
            raise ValueError("GR-1 VITRA dataset requires statistics_path")
        self.statistics_path = str(Path(statistics_path).expanduser().resolve())
        self.data_statistics = read_dataset_statistics(self.statistics_path)
        self.gaussian_normalizer = GaussianNormalizer(self.data_statistics)

    def __len__(self) -> int:
        return int(len(self.native_state24))

    def _image(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        image = np.asarray(iio.imread(self.video_path, index=index), dtype=np.uint8)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected RGB frame from {self.video_path}, got {image.shape}")
        resized = np.asarray(
            resize_short_side_to_target(Image.fromarray(image), target=self.target_image_height),
            dtype=np.uint8,
        )
        intrinsics = compute_new_intrinsics_resize(self.camera_intrinsics, resized.shape[:2])
        return resized, np.asarray(intrinsics, dtype=np.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        rows = np.zeros((self.action_future_window_size, ACTION_DIM), dtype=np.float32)
        masks = action_mask(self.action_future_window_size, valid_rows=0)
        target_indices: list[int] = []
        for offset in range(self.action_future_window_size):
            target_index = index + offset
            if target_index < len(self):
                target_indices.append(target_index)
                masks[offset, :MANO_DIM] = True
        if target_indices:
            rows[: len(target_indices)] = native_target_horizon_to_action192(
                self.native_state24[index], self.native_target24[target_indices]
            )
        image, intrinsics = self._image(index)
        return {
            "instruction": self.instruction,
            "image_list": image[None, ...],
            "image_mask": np.asarray([True], dtype=bool),
            "intrinsics": intrinsics,
            "fov": calculate_fov(image.shape[0], image.shape[1], intrinsics),
            "current_state": torch.from_numpy(self.state102[index].copy()),
            "current_state_mask": torch.from_numpy(state_mask().copy()),
            "action_list": torch.from_numpy(rows),
            "action_mask": torch.from_numpy(masks),
            "native_state24": self.native_state24[index].copy(),
            "native_target24": self.native_target24[index].copy(),
            "episode_index": self.paths.episode_index,
            "frame_index": index,
        }

    def transform_trajectory(self, sample_dict: dict[str, Any], normalization: bool = True) -> dict[str, Any]:
        raw_state = np.asarray(sample_dict["current_state"], dtype=np.float32).reshape(MANO_DIM)
        raw_actions = np.asarray(sample_dict["action_list"], dtype=np.float32)
        if raw_actions.ndim != 2 or raw_actions.shape[1] != ACTION_DIM:
            raise ValueError(f"GR-1 action_list must be [T,{ACTION_DIM}], got {raw_actions.shape}")
        if normalization:
            normalized_state = self.gaussian_normalizer.normalize_state(raw_state).astype(np.float32)
            normalized_action = np.zeros_like(raw_actions, dtype=np.float32)
            normalized_action[:, :MANO_DIM] = self.gaussian_normalizer.normalize_action(
                raw_actions[:, :MANO_DIM]
            ).astype(np.float32)
        else:
            normalized_state = raw_state
            normalized_action = raw_actions.copy()
        output_state = model_state_from_mano(normalized_state)
        output_mask = np.zeros(MODEL_STATE_DIM, dtype=bool)
        output_mask[:MANO_DIM] = np.asarray(sample_dict["current_state_mask"], dtype=bool)
        sample_dict["current_state"] = torch.from_numpy(output_state)
        sample_dict["current_state_mask"] = torch.from_numpy(output_mask)
        sample_dict["action_list"] = torch.from_numpy(normalized_action)
        return sample_dict

    def set_global_data_statistics(self, _: dict[str, Any]) -> None:
        # GR-1 is a separately calibrated embodiment. Its local MANO
        # statistics must not be overwritten by a mixed human/robot average.
        return None

    def deployment_metadata(self) -> dict[str, Any]:
        return {
            "episode_index": self.paths.episode_index,
            "instruction": self.instruction,
            "fps": self.fps,
            "camera": {
                **self.camera_metadata,
                "intrinsics": self.camera_intrinsics.tolist(),
                "camera_from_base": None if self.camera_from_base is None else self.camera_from_base.tolist(),
                "base_from_camera": None if self.base_from_camera is None else self.base_from_camera.tolist(),
            },
            "source_metadata": str(self.metadata_path),
            "frames": len(self),
        }


class GR1SingleEpisodeDataset:
    """Backward-compatible reader for the legacy prepared single episode."""

    def __init__(
        self,
        root_dir: str,
        action_future_window_size: int = 16,
        load_images: bool = True,
        target_image_height: int = 224,
        statistics_path: str | None = None,
        **_: object,
    ) -> None:
        root = Path(root_dir).expanduser().resolve()
        self.root = root
        self.video_path = root / "episode_000000.mp4"
        self.states_path = root / "states24.npy"
        self.actions_path = root / "actions24.npy"
        self.metadata_path = root / "metadata.json"
        for path in (self.video_path, self.states_path, self.actions_path):
            if not path.is_file():
                raise FileNotFoundError(f"GR-1 legacy dataset is missing {path}")
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8")) if self.metadata_path.is_file() else {}
        self.instruction = str(metadata.get("instruction", metadata.get("task", "Execute the RoboCasa GR-1 task.")))
        self.fps = float(metadata.get("fps", 20.0))
        self.camera_intrinsics = np.asarray(
            metadata.get(
                "camera_intrinsics",
                [[515.0, 0.0, 320.0], [0.0, 515.0, 240.0], [0.0, 0.0, 1.0]],
            ),
            dtype=np.float32,
        )
        if self.camera_intrinsics.shape != (3, 3):
            raise ValueError(f"Invalid GR-1 camera intrinsics: {self.camera_intrinsics.shape}")
        self.camera_from_base = None
        self.base_from_camera = None
        if isinstance(metadata.get("camera_extrinsics"), dict):
            ext = metadata["camera_extrinsics"]
            if ext.get("world_to_camera") is not None:
                self.camera_from_base = validate_transform(np.asarray(ext["world_to_camera"]), "camera_from_base")
            if ext.get("camera_to_world") is not None:
                self.base_from_camera = validate_transform(np.asarray(ext["camera_to_world"]), "base_from_camera")
        self.native_state24 = np.asarray(np.load(self.states_path), dtype=np.float32)
        self.native_target24 = np.asarray(np.load(self.actions_path), dtype=np.float32)
        if self.native_state24.ndim != 2 or self.native_state24.shape[1] != NATIVE_DIM:
            raise ValueError(f"Expected states24 [N,24], got {self.native_state24.shape}")
        if self.native_target24.shape != self.native_state24.shape:
            raise ValueError(f"State/action shape mismatch: {self.native_state24.shape}/{self.native_target24.shape}")
        validate_native24(self.native_state24)
        validate_native24(self.native_target24)
        self.state102 = native24_to_mano102(self.native_state24)
        self.action192_absolute = _episode_action_statistics(
            self.native_state24, self.native_target24
        ).astype(np.float32)
        self.action_future_window_size = int(action_future_window_size)
        self.load_images = bool(load_images)
        self.target_image_height = int(target_image_height)
        if statistics_path is None:
            raise ValueError("GR-1 VITRA dataset requires statistics_path")
        self.statistics_path = str(Path(statistics_path).expanduser().resolve())
        self.data_statistics = read_dataset_statistics(self.statistics_path)
        self.gaussian_normalizer = GaussianNormalizer(self.data_statistics)

    def __len__(self) -> int:
        return int(len(self.native_state24))

    def _image(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        image = np.asarray(iio.imread(self.video_path, index=index), dtype=np.uint8)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected RGB frame from {self.video_path}, got {image.shape}")
        resized = np.asarray(resize_short_side_to_target(Image.fromarray(image), target=self.target_image_height), dtype=np.uint8)
        intrinsics = compute_new_intrinsics_resize(self.camera_intrinsics, resized.shape[:2])
        return resized, np.asarray(intrinsics, dtype=np.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        rows = np.zeros((self.action_future_window_size, ACTION_DIM), dtype=np.float32)
        masks = action_mask(self.action_future_window_size, valid_rows=0)
        target_indices: list[int] = []
        for offset in range(self.action_future_window_size):
            target = index + offset
            if target < len(self):
                target_indices.append(target)
                masks[offset, :MANO_DIM] = True
        if target_indices:
            rows[: len(target_indices)] = native_target_horizon_to_action192(
                self.native_state24[index], self.native_target24[target_indices]
            )
        image, intrinsics = self._image(index)
        return {
            "instruction": self.instruction,
            "image_list": image[None, ...],
            "image_mask": np.asarray([True], dtype=bool),
            "intrinsics": intrinsics,
            "fov": calculate_fov(image.shape[0], image.shape[1], intrinsics),
            "current_state": torch.from_numpy(self.state102[index].copy()),
            "current_state_mask": torch.from_numpy(state_mask().copy()),
            "action_list": torch.from_numpy(rows),
            "action_mask": torch.from_numpy(masks),
            "native_state24": self.native_state24[index].copy(),
            "native_target24": self.native_target24[index].copy(),
        }

    def transform_trajectory(self, sample_dict: dict[str, Any], normalization: bool = True) -> dict[str, Any]:
        raw_state = np.asarray(sample_dict["current_state"], dtype=np.float32).reshape(MANO_DIM)
        raw_actions = np.asarray(sample_dict["action_list"], dtype=np.float32)
        if normalization:
            raw_state = self.gaussian_normalizer.normalize_state(raw_state).astype(np.float32)
            output_actions = np.zeros_like(raw_actions)
            output_actions[:, :MANO_DIM] = self.gaussian_normalizer.normalize_action(raw_actions[:, :MANO_DIM])
        else:
            output_actions = raw_actions.copy()
        output_state = model_state_from_mano(raw_state)
        output_mask = np.zeros(MODEL_STATE_DIM, dtype=bool)
        output_mask[:MANO_DIM] = state_mask()
        sample_dict["current_state"] = torch.from_numpy(output_state)
        sample_dict["current_state_mask"] = torch.from_numpy(output_mask)
        sample_dict["action_list"] = torch.from_numpy(output_actions.astype(np.float32))
        return sample_dict

    def set_global_data_statistics(self, _: dict[str, Any]) -> None:
        return None


class GR1SamplesDataset(ConcatDataset):
    """Automatically discover and concatenate all GR-1 sample episodes."""

    def __init__(
        self,
        root_dir: str,
        action_future_window_size: int = 16,
        load_images: bool = True,
        target_image_height: int = 224,
        statistics_path: str | None = None,
        **_: object,
    ) -> None:
        metadata_paths = discover_gr1_episode_metadata(Path(root_dir))
        if not metadata_paths:
            raise FileNotFoundError(f"No GR-1 LeRobot episode metadata found below {root_dir}")
        if statistics_path is None:
            raise ValueError("GR-1 samples require statistics_path")
        datasets = [
            GR1LeRobotEpisodeDataset(
                paths=_episode_paths(path),
                action_future_window_size=action_future_window_size,
                load_images=load_images,
                target_image_height=target_image_height,
                statistics_path=statistics_path,
            )
            for path in metadata_paths
        ]
        self.data_statistics = datasets[0].data_statistics
        self.episode_datasets = datasets
        super().__init__(datasets)

    def transform_trajectory(self, sample_dict: dict[str, Any], normalization: bool = True) -> dict[str, Any]:
        # All episodes use the same embodiment-local statistics.
        return self.datasets[0].transform_trajectory(sample_dict, normalization)

    def set_global_data_statistics(self, _: dict[str, Any]) -> None:
        return None


__all__ = [
    "ACTION_WIDTH",
    "ACTIVE_CHANNELS",
    "GR1LeRobotEpisodeDataset",
    "GR1SamplesDataset",
    "GR1SingleEpisodeDataset",
    "MODEL_STATE_WIDTH",
    "RAW_STATE_WIDTH",
    "action_mask",
    "active_action_mask",
    "decode_action192",
    "discover_gr1_episode_metadata",
    "encode_action24",
    "encode_state24",
]

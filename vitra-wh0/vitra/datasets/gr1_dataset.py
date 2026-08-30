"""RoboCasa GR-1 Fourier-hand adapter for the existing VITRA contract.

The model-side action tensor remains VITRA's 192-D two-hand layout.  Only the
explicitly listed 24 channels are active for GR-1 and all other channels are
masked.  This module is an embodiment adapter; it does not change VITRA's
backbone, diffusion process, or training loss.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

from vitra.datasets.dataset_utils import calculate_fov, compute_new_intrinsics_resize
from vitra.utils.data_utils import GaussianNormalizer, read_dataset_statistics, resize_short_side_to_target


HAND_WIDTH = 51
RAW_STATE_WIDTH = 102
MODEL_STATE_WIDTH = 212
ACTION_WIDTH = 192

LEFT_POSE = slice(0, 6)
LEFT_HAND = slice(6, 12)
RIGHT_POSE = slice(51, 57)
RIGHT_HAND = slice(57, 63)
ACTIVE_CHANNELS = np.asarray(list(range(0, 12)) + list(range(51, 63)), dtype=np.int64)


def encode_action24(action24: np.ndarray) -> np.ndarray:
    """Encode RoboCasa action order into the sparse VITRA 192-D layout."""

    action = np.asarray(action24, dtype=np.float32).reshape(24)
    output = np.zeros(ACTION_WIDTH, dtype=np.float32)
    output[LEFT_POSE] = action[6:12]
    output[LEFT_HAND] = action[18:24]
    output[RIGHT_POSE] = action[0:6]
    output[RIGHT_HAND] = action[12:18]
    return output


def encode_state24(state24: np.ndarray) -> np.ndarray:
    """Encode [left pose/hand, right pose/hand] into the 102-D state layout."""

    state = np.asarray(state24, dtype=np.float32).reshape(24)
    output = np.zeros(RAW_STATE_WIDTH, dtype=np.float32)
    output[LEFT_POSE] = state[0:6]
    output[LEFT_HAND] = state[6:12]
    output[RIGHT_POSE] = state[12:18]
    output[RIGHT_HAND] = state[18:24]
    return output


def decode_action192(action192: np.ndarray) -> np.ndarray:
    """Decode a VITRA 192-D prediction to RoboCasa's 24-D action order."""

    action = np.asarray(action192, dtype=np.float32)
    if action.shape[-1] < RIGHT_HAND.stop:
        raise ValueError(f"Expected at least 63 VITRA channels, got {action.shape}")
    output_shape = action.shape[:-1] + (24,)
    output = np.zeros(output_shape, dtype=np.float32)
    output[..., 0:6] = action[..., RIGHT_POSE]
    output[..., 6:12] = action[..., LEFT_POSE]
    output[..., 12:18] = action[..., RIGHT_HAND]
    output[..., 18:24] = action[..., LEFT_HAND]
    return output


def active_action_mask(length: int, valid: bool = True) -> np.ndarray:
    mask = np.zeros((length, ACTION_WIDTH), dtype=np.bool_)
    if valid:
        mask[:, ACTIVE_CHANNELS] = True
    return mask


class GR1SingleEpisodeDataset:
    """Read one state-aligned RoboCasa GR-1 episode."""

    def __init__(
        self,
        root_dir: str,
        action_future_window_size: int = 16,
        load_images: bool = True,
        target_image_height: int = 224,
        statistics_path: str | None = None,
        **_: object,
    ) -> None:
        root = Path(root_dir)
        self.root = root
        self.video_path = root / "episode_000000.mp4"
        self.states_path = root / "states24.npy"
        self.actions_path = root / "actions24.npy"
        self.metadata_path = root / "metadata.json"
        if not self.video_path.is_file() or not self.states_path.is_file() or not self.actions_path.is_file():
            raise FileNotFoundError(
                "GR-1 dataset requires episode_000000.mp4, states24.npy, and actions24.npy under "
                f"{root}"
            )
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8")) if self.metadata_path.is_file() else {}
        self.instruction = str(metadata.get("instruction", "Left hand: pick up the cup, place it into the drawer and close the drawer. Right hand: None."))
        self.fps = float(metadata.get("fps", 20.0))
        self.camera_intrinsics = np.asarray(
            metadata.get("camera_intrinsics", [[515.0, 0.0, 320.0], [0.0, 515.0, 240.0], [0.0, 0.0, 1.0]]),
            dtype=np.float32,
        )
        if self.camera_intrinsics.shape != (3, 3):
            raise ValueError(f"Invalid GR-1 camera intrinsics: {self.camera_intrinsics.shape}")
        self.state24 = np.asarray(np.load(self.states_path), dtype=np.float32)
        self.action24 = np.asarray(np.load(self.actions_path), dtype=np.float32)
        if self.state24.ndim != 2 or self.state24.shape[1] != 24:
            raise ValueError(f"Expected states24 [N,24], got {self.state24.shape}")
        if self.action24.shape != self.state24.shape:
            raise ValueError(f"State/action shape mismatch: {self.state24.shape} vs {self.action24.shape}")
        self.state102 = np.stack([encode_state24(row) for row in self.state24]).astype(np.float32)
        self.action192 = np.stack([encode_action24(row) for row in self.action24]).astype(np.float32)
        self.action_future_window_size = int(action_future_window_size)
        self.load_images = bool(load_images)
        self.target_image_height = int(target_image_height)
        if statistics_path is None:
            raise ValueError("GR1SingleEpisodeDataset requires a supplied statistics_path")
        self.data_statistics = read_dataset_statistics(statistics_path)
        self.gaussian_normalizer = GaussianNormalizer(self.data_statistics)

    def __len__(self) -> int:
        return int(self.state24.shape[0])

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < len(self):
            raise IndexError(index)
        state = torch.from_numpy(self.state102[index].copy())
        state_mask = torch.zeros(RAW_STATE_WIDTH, dtype=torch.bool)
        state_mask[ACTIVE_CHANNELS] = True

        action_rows = np.zeros((self.action_future_window_size, ACTION_WIDTH), dtype=np.float32)
        action_masks = np.zeros((self.action_future_window_size, ACTION_WIDTH), dtype=np.bool_)
        for offset in range(self.action_future_window_size):
            target = index + offset
            if target < len(self):
                action_rows[offset] = self.action192[target]
                action_masks[offset, ACTIVE_CHANNELS] = True

        image, intrinsics = self._image(index)
        return {
            "instruction": self.instruction,
            "image_list": image[None, ...],
            "image_mask": np.array([True], dtype=np.bool_),
            "intrinsics": intrinsics,
            "fov": calculate_fov(image.shape[0], image.shape[1], intrinsics),
            "current_state": state,
            "current_state_mask": state_mask,
            "action_list": torch.from_numpy(action_rows),
            "action_mask": torch.from_numpy(action_masks),
        }

    def _image(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        image = np.asarray(iio.imread(self.video_path, index=index), dtype=np.uint8)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected RGB frame from {self.video_path}, got {image.shape}")
        resized = np.asarray(
            resize_short_side_to_target(Image.fromarray(image), target=self.target_image_height),
            dtype=np.uint8,
        )
        intrinsics = compute_new_intrinsics_resize(self.camera_intrinsics, resized.shape[:2])
        return resized, intrinsics

    def transform_trajectory(self, sample_dict: dict, normalization: bool = True) -> dict:
        raw_state = np.asarray(sample_dict["current_state"], dtype=np.float32).reshape(RAW_STATE_WIDTH)
        raw_actions = np.asarray(sample_dict["action_list"], dtype=np.float32)
        if normalization:
            state102 = self.gaussian_normalizer.normalize_state(raw_state)
            action102 = self.gaussian_normalizer.normalize_action(raw_actions[:, :RAW_STATE_WIDTH])
        else:
            state102 = raw_state
            action102 = raw_actions[:, :RAW_STATE_WIDTH]

        output_state = np.zeros(MODEL_STATE_WIDTH, dtype=np.float32)
        output_state[:RAW_STATE_WIDTH] = state102
        output_state_mask = np.zeros(MODEL_STATE_WIDTH, dtype=np.bool_)
        output_state_mask[:RAW_STATE_WIDTH] = np.asarray(sample_dict["current_state_mask"], dtype=np.bool_)

        output_actions = np.zeros_like(raw_actions, dtype=np.float32)
        output_actions[:, :RAW_STATE_WIDTH] = action102
        sample_dict["current_state"] = torch.from_numpy(output_state)
        sample_dict["current_state_mask"] = torch.from_numpy(output_state_mask)
        sample_dict["action_list"] = torch.from_numpy(output_actions)
        return sample_dict

    def set_global_data_statistics(self, _: dict) -> None:
        # This is a single-embodiment dataset.  Its supplied profile is
        # intentionally not replaced by a global/mixed-data profile.
        return None

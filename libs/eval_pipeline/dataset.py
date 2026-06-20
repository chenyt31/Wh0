from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

from vitra.datasets.dataset_utils import StateFeature
from vitra.datasets.robot_dataset import (
    RoboDatasetCore,
    add_passive_joints_to_human_action,
    pad_state_robot,
    transfer_inspire_to_human,
)


def _load_dataset_sample(root_dir: str, sample_idx: int, statistics_path: str | None) -> dict:
    dataset = RoboDatasetCore(
        root_dir=root_dir,
        statistics_path=statistics_path,
        action_past_window_size=0,
        action_future_window_size=16,
        image_past_window_size=0,
        image_future_window_size=0,
        load_images=True,
        augmentation=False,
        flip_augmentation=True,
        set_none_ratio=0.0,
        state_mask_prob=0.0,
        target_image_height=224,
    )
    sample = dataset[sample_idx]
    image = sample["image_list"][0]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return {
        "image": image,
        "instruction": sample["instruction"],
        "state": sample["current_state"],
        "state_mask": sample["current_state_mask"].tolist(),
    }


def _human_state_102(state_24: np.ndarray, state_mask: list[bool]) -> np.ndarray:
    unified_state, unified_mask = pad_state_robot(
        state=torch.from_numpy(state_24).float(),
        state_mask=torch.tensor(state_mask, dtype=torch.bool),
        state_dim=24,
        unified_state_dim=StateFeature.ALL_FEATURES[1],
    )
    action_mask = torch.zeros(16, 192, dtype=torch.bool)
    human_state, _, _, _ = transfer_inspire_to_human(unified_state, unified_mask, None, action_mask)
    human_state = add_passive_joints_to_human_action(human_state.unsqueeze(0)).squeeze(0)
    return human_state[:102].numpy()


def _rotations_from_euler(block: np.ndarray) -> np.ndarray:
    return np.stack([R.from_euler("xyz", joint).as_matrix() for joint in block.reshape(-1, 3)], axis=0).astype(np.float32)


def _hand_data_from_state(human_state_102: np.ndarray, state_mask: list[bool]) -> dict:
    fx = 931.20577836
    width = 1280
    payload = {"fov_x": float(2 * np.arctan(width / (2 * fx)) * 180 / np.pi)}
    for hand, enabled, start in (("left", state_mask[0], 0), ("right", state_mask[1], 51)):
        if enabled:
            block = human_state_102[start : start + 51]
            payload[hand] = {
                0: {
                    "transl": block[0:3].astype(np.float32),
                    "global_orient": R.from_euler("xyz", block[3:6]).as_matrix().astype(np.float32),
                    "hand_pose": _rotations_from_euler(block[6:51]),
                    "beta": np.zeros(10, dtype=np.float32),
                }
            }
        else:
            payload[hand] = {
                0: {
                    "transl": np.zeros(3, dtype=np.float32),
                    "global_orient": np.eye(3, dtype=np.float32),
                    "hand_pose": np.zeros((15, 3, 3), dtype=np.float32),
                    "beta": np.zeros(10, dtype=np.float32),
                }
            }
    return payload


def save_dataset_frame(root_dir: str, sample_idx: int, output_dir: str, statistics_path: str | None = None) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    sample = _load_dataset_sample(root_dir, sample_idx, statistics_path)
    human_state = _human_state_102(sample["state"], sample["state_mask"])
    hand_data = _hand_data_from_state(human_state, sample["state_mask"])

    np.save(output / "image.npy", hand_data, allow_pickle=True)
    Image.fromarray(sample["image"]).save(output / "image.jpg")

    info = {
        "root_dir": root_dir,
        "sample_idx": sample_idx,
        "instruction": sample["instruction"],
        "state_mask": sample["state_mask"],
        "fov_x": hand_data["fov_x"],
    }
    (output / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

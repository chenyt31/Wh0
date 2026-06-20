from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

from vitra.datasets.coord_trans import get_camera_pose_in_base_frame
from vitra.datasets.dataset_utils import ActionFeature, StateFeature, calculate_fov, compute_new_intrinsics_resize
from vitra.datasets.human_dataset import pad_action
from vitra.utils.data_utils import GaussianNormalizer, read_dataset_statistics, resize_short_side_to_target


CAMERA_INTRINSICS = np.array(
    [[931.20577836, 0.0, 640.0], [0.0, 937.832063295, 360.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)
DEFAULT_IMAGE_SIZE = (720, 1280)
ROBOT_STATE_DIM = 24
ROBOT_HAND_DIM = 12
HUMAN_HAND_DIM = 51

INSPIRE_QPOS_TO_URDF_JOINT_IDX = [5, 4, 3, 2, 1, 0]
INSPIRE_JOINT_LIMITS = np.array(
    [
        [-0.1, 1.3],
        [0.0, 0.5],
        [0.0, 1.7],
        [0.0, 1.7],
        [0.0, 1.7],
        [0.0, 1.7],
    ],
    dtype=np.float32,
)

# (Inspire API qpos index, MANO-axis index inside each 51-dim hand block, sign)
INSPIRE_HUMAN_MAPPING = [
    (0, 26, 1),
    (1, 35, 1),
    (2, 17, 1),
    (3, 8, 1),
    (4, 47, 1),
    (5, 42, 1),
]


def se3_from_list(pose: list | np.ndarray) -> np.ndarray:
    return np.asarray(pose, dtype=np.float32).reshape(4, 4)


def se3_to_trans_euler(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transform = np.asarray(transform, dtype=np.float32)
    return transform[:3, 3], R.from_matrix(transform[:3, :3]).as_euler("xyz", degrees=False).astype(np.float32)


def transform_pose_to_camera_frame(T_base_pose: np.ndarray, T_base_camera: np.ndarray) -> np.ndarray:
    return np.linalg.inv(T_base_camera) @ T_base_pose


def correct_gt_wrist_orientation(T_camera: np.ndarray, hand: str = "right") -> np.ndarray:
    _ = hand
    T = np.asarray(T_camera, dtype=np.float32).copy()
    r_y_pi = np.diag([-1.0, 1.0, -1.0]).astype(np.float32)
    r_x_pi = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    fix_y = np.eye(4, dtype=np.float32)
    fix_x = np.eye(4, dtype=np.float32)
    fix_y[:3, :3] = r_y_pi
    fix_x[:3, :3] = r_x_pi
    return T @ fix_y @ fix_x


def correct_gt_wrist_orientation_inverse(T_camera: np.ndarray, hand: str = "right") -> np.ndarray:
    _ = hand
    T = np.asarray(T_camera, dtype=np.float32).copy()
    r_y_pi = np.diag([-1.0, 1.0, -1.0]).astype(np.float32)
    r_x_pi = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    fix_y = np.eye(4, dtype=np.float32)
    fix_x = np.eye(4, dtype=np.float32)
    fix_y[:3, :3] = r_y_pi
    fix_x[:3, :3] = r_x_pi
    return T @ fix_x @ fix_y


def denormalize_inspire_qpos(
    qpos_normalized: np.ndarray,
    joint_limits: np.ndarray = INSPIRE_JOINT_LIMITS,
) -> np.ndarray:
    qpos = np.asarray(qpos_normalized, dtype=np.float32)
    q_rad = np.zeros(6, dtype=np.float32)
    for api_idx in range(min(len(qpos), 6)):
        urdf_idx = INSPIRE_QPOS_TO_URDF_JOINT_IDX[api_idx]
        q_min, q_max = joint_limits[urdf_idx]
        q_rad[api_idx] = q_max - qpos[api_idx] * (q_max - q_min)
    return q_rad


def _pose_to_6d(frame_state: dict, hand: str, T_base_camera: np.ndarray) -> np.ndarray:
    wrist = se3_from_list(frame_state[f"{hand}_arm"]["wrist_pose"])
    wrist_cam = transform_pose_to_camera_frame(wrist, T_base_camera)
    wrist_cam = correct_gt_wrist_orientation(wrist_cam, hand=hand)
    translation, euler = se3_to_trans_euler(wrist_cam)
    joints = denormalize_inspire_qpos(frame_state[f"{hand}_ee"]["qpos"])
    return np.concatenate([translation, euler, joints]).astype(np.float32)


def _frame_robot_state(frame_state: dict, T_base_camera: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            _pose_to_6d(frame_state, "left", T_base_camera),
            _pose_to_6d(frame_state, "right", T_base_camera),
        ]
    ).astype(np.float32)


def _relative_action(current: np.ndarray, following: np.ndarray) -> np.ndarray:
    action = np.zeros_like(current, dtype=np.float32)
    for offset in (0, ROBOT_HAND_DIM):
        curr = current[offset : offset + ROBOT_HAND_DIM]
        nxt = following[offset : offset + ROBOT_HAND_DIM]
        action[offset : offset + 3] = nxt[:3] - curr[:3]
        r_curr = R.from_euler("xyz", curr[3:6]).as_matrix()
        r_next = R.from_euler("xyz", nxt[3:6]).as_matrix()
        action[offset + 3 : offset + 6] = R.from_matrix(r_next @ r_curr.T).as_euler("xyz", degrees=False)
        action[offset + 6 : offset + ROBOT_HAND_DIM] = nxt[6:ROBOT_HAND_DIM]
    return action


def _resize_image(image: np.ndarray, intrinsics: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray]:
    resized = np.asarray(resize_short_side_to_target(Image.fromarray(image), target=target), dtype=np.uint8)
    return resized, compute_new_intrinsics_resize(intrinsics, resized.shape[:2])


def _episode_id_from_path(path: str) -> int:
    match = re.search(r"episode_(\d+)", str(path))
    return int(match.group(1)) if match else 0


def _segment_for_frame(split_info: dict | None, frame_idx: int, episode_len: int) -> tuple[str | None, int, int]:
    if not split_info:
        return None, 0, episode_len - 1
    points = [int(point) for point in split_info.get("points", [])]
    instructions = split_info.get("instructions", [])
    boundaries = [0, *points, episode_len - 1]
    segment_idx = 0
    for boundary in points:
        if int(frame_idx) > boundary:
            segment_idx += 1
        else:
            break
    segment_idx = min(segment_idx, max(0, len(boundaries) - 2))
    instruction = instructions[min(segment_idx, len(instructions) - 1)] if instructions else None
    return instruction, boundaries[segment_idx], boundaries[segment_idx + 1]


def _load_episode_json(episode_dir: Path) -> dict:
    with (episode_dir / "data.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_robot_image(episode_dir: Path, frame_idx: int) -> np.ndarray:
    colors_dir = episode_dir / "colors"
    left_path = colors_dir / f"{frame_idx:06d}_color_0.jpg"
    right_path = colors_dir / f"{frame_idx:06d}_color_1.jpg"
    if not left_path.exists():
        raise FileNotFoundError(f"Missing robot image: {left_path}")
    left = np.asarray(Image.open(left_path).convert("RGB"), dtype=np.uint8)
    if not right_path.exists():
        return left
    right = np.asarray(Image.open(right_path).convert("RGB"), dtype=np.uint8)
    return np.concatenate([left, right], axis=1)


class RoboDatasetCore:
    """Robot dataset reader for G1/Inspire episodes indexed by training_index.npz."""

    def __init__(
        self,
        root_dir: str,
        statistics_path: Optional[str] = None,
        action_past_window_size: int = 0,
        action_future_window_size: int = 16,
        image_past_window_size: int = 0,
        image_future_window_size: int = 0,
        load_images: bool = True,
        augmentation: bool = True,
        flip_augmentation: bool = True,
        set_none_ratio: float = 0.0,
        state_mask_prob: float = 0.1,
        target_image_height: int = 224,
    ):
        self.root = Path(root_dir)
        self.action_future_window_size = action_future_window_size
        self.load_images = load_images
        self.augmentation = augmentation
        self.state_mask_prob = state_mask_prob
        self.target_image_height = target_image_height
        self.data_statistics = read_dataset_statistics(statistics_path) if statistics_path else None
        self.gaussian_normalizer = GaussianNormalizer(self.data_statistics) if self.data_statistics else None
        self._T_base_camera: np.ndarray | None = None

        index_path = self.root / "training_index.npz"
        index = np.load(index_path, allow_pickle=True)
        self.episode_paths = index["episode_paths"]
        self.sample_indices = index["sample_indices"]

        split_path = self.root / "split_points.json"
        self.split_points = json.loads(split_path.read_text(encoding="utf-8")) if split_path.exists() else {}

        _ = action_past_window_size, image_past_window_size, image_future_window_size, flip_augmentation, set_none_ratio

    def __len__(self) -> int:
        return int(self.sample_indices.shape[0])

    @property
    def T_base_camera(self) -> np.ndarray:
        if self._T_base_camera is None:
            self._T_base_camera = get_camera_pose_in_base_frame()
        return self._T_base_camera

    def _episode_context(self, idx: int) -> tuple[int, int, int, str, Path, dict]:
        episode_idx, sample_idx, frame_idx = [int(value) for value in self.sample_indices[idx]]
        episode_path = str(self.episode_paths[episode_idx])
        episode_dir = self.root / episode_path
        return episode_idx, sample_idx, frame_idx, episode_path, episode_dir, _load_episode_json(episode_dir)

    def _instruction(self, episode_path: str, episode_data: dict, frame_idx: int) -> tuple[str, int, int]:
        episode_id = _episode_id_from_path(episode_path)
        split_key = f"episode_{episode_id:04d}"
        goal, segment_start, segment_end = _segment_for_frame(
            self.split_points.get(split_key),
            frame_idx,
            len(episode_data["data"]),
        )
        if goal is None:
            goal = episode_data.get("text", {}).get("goal", "")
        return f"Left hand: None. Right hand: {goal}.", segment_start, segment_end

    def _action_sequence(
        self,
        idx: int,
        episode_idx: int,
        episode_data: dict,
        segment_start: int,
        segment_end: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        poses = np.zeros((self.action_future_window_size + 1, ROBOT_STATE_DIM), dtype=np.float32)
        frame_cursor = int(self.sample_indices[idx][2])
        for local_t in range(self.action_future_window_size + 1):
            index_t = min(idx + local_t, len(self.sample_indices) - 1)
            next_episode_idx, _, next_frame_idx = [int(value) for value in self.sample_indices[index_t]]
            if next_episode_idx == episode_idx:
                frame_cursor = int(np.clip(next_frame_idx, segment_start, segment_end))
            frame_state = episode_data["data"][max(0, frame_cursor)]["states"]
            poses[local_t] = _frame_robot_state(frame_state, self.T_base_camera)

        actions = np.stack([_relative_action(poses[t], poses[t + 1]) for t in range(self.action_future_window_size)])
        action_mask = np.zeros((self.action_future_window_size, 2), dtype=np.bool_)
        action_mask[:, 1] = True
        return poses[0], actions, action_mask

    def __getitem__(self, idx: int) -> dict:
        episode_idx, _, frame_idx, episode_path, episode_dir, episode_data = self._episode_context(idx)
        instruction, segment_start, segment_end = self._instruction(episode_path, episode_data, frame_idx)
        current_state, action_list, action_mask = self._action_sequence(
            idx,
            episode_idx,
            episode_data,
            segment_start,
            segment_end,
        )

        sample = {
            "instruction": instruction,
            "intrinsics": CAMERA_INTRINSICS.copy(),
            "current_state": current_state,
            "current_state_mask": np.array([False, True], dtype=np.bool_),
            "action_list": action_list,
            "action_mask": action_mask,
        }

        if self.load_images:
            image = _load_robot_image(episode_dir, frame_idx)
            image, sample["intrinsics"] = _resize_image(image, sample["intrinsics"], self.target_image_height)
            sample["image_list"] = image[None, ...]
            sample["image_mask"] = np.array([True], dtype=np.bool_)

        if self.augmentation and random.random() < self.state_mask_prob:
            sample["current_state"] = np.zeros_like(sample["current_state"])
            sample["current_state_mask"] = np.array([False, False], dtype=np.bool_)

        image_shape = sample["image_list"][0].shape[:2] if self.load_images else DEFAULT_IMAGE_SIZE
        sample["fov"] = calculate_fov(image_shape[0], image_shape[1], sample["intrinsics"])
        return sample

    def set_global_data_statistics(self, global_data_statistics: dict) -> None:
        self.data_statistics = {
            key: np.asarray(value).copy() if isinstance(value, (list, np.ndarray)) else value
            for key, value in global_data_statistics.items()
        }
        self.gaussian_normalizer = GaussianNormalizer(self.data_statistics)

    def transform_trajectory(self, sample_dict: dict | None = None, normalization: bool = True) -> dict:
        action_np = np.asarray(sample_dict["action_list"], dtype=np.float32)
        state_np = np.asarray(sample_dict["current_state"], dtype=np.float32)

        unified_action, unified_action_mask = pad_action(
            action_np,
            sample_dict["action_mask"],
            action_np.shape[1],
            ActionFeature.ALL_FEATURES[1],
        )
        unified_state, unified_state_mask = pad_state_robot(
            state_np,
            sample_dict["current_state_mask"],
            state_np.shape[0],
            StateFeature.ALL_FEATURES[1],
        )

        human_state, human_state_mask, human_action, human_action_mask = transfer_inspire_to_human(
            unified_state,
            unified_state_mask,
            unified_action,
            unified_action_mask,
        )

        if human_action is not None:
            human_action = add_passive_joints_to_human_action(human_action)
            if normalization and self.gaussian_normalizer is not None:
                normalized = self.gaussian_normalizer.normalize_action(human_action[:, :102].numpy())
                human_action[:, :102] = torch.from_numpy(normalized).to(human_action.dtype)

        human_state = add_passive_joints_to_human_action(human_state.unsqueeze(0)).squeeze(0)
        if normalization and self.gaussian_normalizer is not None:
            human_state[:102] = _normalize_human_state_102(human_state[:102], self.gaussian_normalizer)

        sample_dict["action_list"] = human_action
        sample_dict["action_mask"] = human_action_mask
        sample_dict["current_state"] = human_state
        sample_dict["current_state_mask"] = human_state_mask
        return sample_dict


def _normalize_human_state_102(human_state_102: torch.Tensor, normalizer: GaussianNormalizer) -> torch.Tensor:
    padded = np.zeros(122, dtype=np.float32)
    state_np = human_state_102.detach().cpu().numpy().astype(np.float32)
    padded[:51] = state_np[:51]
    padded[61:112] = state_np[51:102]
    normalized = normalizer.normalize_state(padded)
    compact = np.concatenate([normalized[:51], normalized[61:112]]).astype(np.float32)
    return torch.from_numpy(compact).to(human_state_102.dtype)


def pad_state_robot(
    state: torch.Tensor | np.ndarray,
    state_mask: torch.Tensor | np.ndarray,
    state_dim: int,
    unified_state_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    current_state = torch.as_tensor(state, dtype=torch.float32)
    hand_mask = torch.as_tensor(state_mask, dtype=torch.bool)
    expanded_mask = hand_mask.repeat_interleave(state_dim // 2)
    masked_state = current_state * expanded_mask.to(current_state.dtype)

    padded_state = torch.zeros(unified_state_dim, dtype=current_state.dtype)
    padded_mask = torch.zeros(unified_state_dim, dtype=torch.bool)
    padded_state[:state_dim] = masked_state[:state_dim]
    padded_mask[:state_dim] = expanded_mask[:state_dim]
    return padded_state, padded_mask


def add_passive_joints_to_human_action(human_action: torch.Tensor) -> torch.Tensor:
    is_single = human_action.ndim == 1
    if is_single:
        human_action = human_action.unsqueeze(0)
    output = human_action.clone()
    for hand_offset in (0, HUMAN_HAND_DIM):
        for inspire_idx, human_idx, _ in INSPIRE_HUMAN_MAPPING:
            passive_idx = hand_offset + human_idx + 3
            active_idx = hand_offset + human_idx
            if hand_offset + 6 <= passive_idx < hand_offset + HUMAN_HAND_DIM:
                output[:, passive_idx] = output[:, active_idx]
            _ = inspire_idx
    return output.squeeze(0) if is_single else output


def transfer_inspire_to_human(
    unified_state: torch.Tensor,
    unified_state_mask: torch.Tensor,
    unified_action: Optional[torch.Tensor],
    unified_action_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    human_state = torch.zeros(unified_state.shape[0], dtype=torch.float32)
    human_state_mask = torch.zeros(unified_state_mask.shape[0], dtype=torch.bool)

    human_state[0:6] = unified_state[0:6]
    human_state[51:57] = unified_state[12:18]
    for src, dst, sign in INSPIRE_HUMAN_MAPPING:
        human_state[dst] = sign * unified_state[src + 6]
        human_state[dst + 51] = sign * unified_state[src + 18]

    human_state_mask[0:51] = bool(unified_state_mask[0])
    human_state_mask[51:102] = bool(unified_state_mask[12])

    human_action_mask = torch.zeros((unified_action_mask.shape[0], ActionFeature.ALL_FEATURES[1]), dtype=torch.bool)
    human_action_mask[:, 0:51] = unified_action_mask[:, 0].unsqueeze(1).expand(-1, 51)
    human_action_mask[:, 51:102] = unified_action_mask[:, 12].unsqueeze(1).expand(-1, 51)

    if unified_action is None:
        return human_state, human_state_mask, None, human_action_mask

    human_action = torch.zeros((unified_action.shape[0], ActionFeature.ALL_FEATURES[1]), dtype=torch.float32)
    human_action[:, 0:6] = unified_action[:, 0:6]
    human_action[:, 51:57] = unified_action[:, 12:18]
    for src, dst, sign in INSPIRE_HUMAN_MAPPING:
        human_action[:, dst] = sign * unified_action[:, src + 6]
        human_action[:, dst + 51] = sign * unified_action[:, src + 18]
    return human_state, human_state_mask, human_action, human_action_mask


def transfer_human_to_inspire(human_action: torch.Tensor) -> torch.Tensor:
    output = torch.zeros((human_action.shape[0], ROBOT_STATE_DIM), dtype=torch.float32)
    output[:, 0:6] = human_action[:, 0:6]
    output[:, 12:18] = human_action[:, 51:57]
    for src, dst, sign in INSPIRE_HUMAN_MAPPING:
        output[:, src + 6] = sign * human_action[:, dst]
        output[:, src + 18] = sign * human_action[:, dst + 51]
    return output


def transform_camera_pose_to_base_frame(
    camera_pose: np.ndarray,
    T_base_camera: np.ndarray | None = None,
) -> np.ndarray:
    T_base_camera = get_camera_pose_in_base_frame() if T_base_camera is None else T_base_camera
    if camera_pose.shape == (6,):
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = R.from_euler("xyz", camera_pose[3:6]).as_matrix()
        transform[:3, 3] = camera_pose[:3]
    else:
        transform = camera_pose
    return T_base_camera @ transform


def _pose6d_to_matrix(pose6d: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = R.from_euler("xyz", pose6d[3:6]).as_matrix()
    transform[:3, 3] = pose6d[:3]
    return transform

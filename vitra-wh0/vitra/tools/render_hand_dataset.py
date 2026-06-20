from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
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
from vitra.utils.data_utils import recon_traj

try:
    from visualization.visualize_core import HandVisualizer, Renderer, process_single_hand_labels
    from visualization.visualize_core import Config as HandConfig

    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False


INTRINSICS = np.array(
    [
        [931.20577836, 0.0, 640.0],
        [0.0, 937.832063295, 360.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


@dataclass
class RenderConfig:
    root_dir: str
    sample_idx: int
    output_dir: str
    statistics_path: str | None = None
    mano_path: str = "./weights/mano"
    render_hand: bool = True
    num_frames: int = 81
    fps: int = 15
    compare_action: bool = True
    num_samples: int = 1
    sample_stride: int = 16
    left_color: tuple[float, float, float] = (0.10, 0.70, 1.00)
    right_color: tuple[float, float, float] = (1.00, 0.10, 0.05)
    whole_episode: bool = False


def _apply_render_colors(hand_config: HandConfig, config: RenderConfig) -> HandConfig:
    hand_config.LEFT_COLOR = np.array(config.left_color, dtype=np.float32)
    hand_config.RIGHT_COLOR = np.array(config.right_color, dtype=np.float32)
    return hand_config


def load_dataset_sample(root_dir: str, sample_idx: int, statistics_path: str | None) -> dict:
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
    training_index = np.load(Path(root_dir) / "training_index.npz", allow_pickle=True)
    episode_paths = training_index["episode_paths"]
    episode_idx, _, _ = training_index["sample_indices"][sample_idx]
    return {
        "image": image,
        "instruction": sample["instruction"],
        "state": sample["current_state"],
        "state_mask": sample["current_state_mask"].tolist(),
        "intrinsics": np.asarray(sample["intrinsics"], dtype=np.float32),
        "episode_dir": str(Path(root_dir) / episode_paths[episode_idx]),
    }


def convert_inspire_to_human_state_102(state_24: np.ndarray, state_mask: list[bool]) -> np.ndarray:
    unified_state_dim = StateFeature.ALL_FEATURES[1]
    unified_state, unified_state_mask = pad_state_robot(
        state=torch.from_numpy(state_24).float(),
        state_mask=torch.tensor(state_mask, dtype=torch.bool),
        state_dim=24,
        unified_state_dim=unified_state_dim,
    )
    unified_action_mask = torch.zeros(16, 192, dtype=torch.bool)
    human_state, _, _, _ = transfer_inspire_to_human(unified_state, unified_state_mask, None, unified_action_mask)
    human_state = add_passive_joints_to_human_action(human_state.unsqueeze(0)).squeeze(0)
    return human_state[:102].numpy()


def _euler_block_to_rotations(block: np.ndarray) -> np.ndarray:
    return np.stack([R.from_euler("xyz", joint).as_matrix() for joint in block.reshape(-1, 3)], axis=0).astype(np.float32)


def convert_human_state_to_hand_data(
    human_state_102: np.ndarray,
    state_mask: list[bool],
    intrinsics: np.ndarray = INTRINSICS,
) -> dict:
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    width = intrinsics[0, 2] * 2
    fx = intrinsics[0, 0]
    hand_data = {
        "fov_x": float(2 * np.arctan(width / (2 * fx)) * 180 / np.pi),
        "intrinsics": intrinsics,
    }
    for hand, enabled, start in (("left", state_mask[0], 0), ("right", state_mask[1], 51)):
        if enabled:
            block = human_state_102[start : start + 51]
            hand_data[hand] = {
                0: {
                    "transl": block[0:3].astype(np.float32),
                    "global_orient": R.from_euler("xyz", block[3:6]).as_matrix().astype(np.float32),
                    "hand_pose": _euler_block_to_rotations(block[6:51]),
                    "beta": np.zeros(10, dtype=np.float32),
                }
            }
        else:
            hand_data[hand] = {
                0: {
                    "transl": np.zeros(3, dtype=np.float32),
                    "global_orient": np.eye(3, dtype=np.float32),
                    "hand_pose": np.zeros((15, 3, 3), dtype=np.float32),
                    "beta": np.zeros(10, dtype=np.float32),
                }
            }
    return hand_data


def _labels_from_hand_data(hand_data: dict, hand: str) -> dict:
    data = hand_data[hand][0]
    return {
        "transl_worldspace": np.asarray(data["transl"], dtype=np.float32).reshape(1, 3),
        "global_orient_worldspace": np.asarray(data["global_orient"], dtype=np.float32).reshape(1, 3, 3),
        "hand_pose": np.asarray(data["hand_pose"], dtype=np.float32).reshape(1, 15, 3, 3),
        "beta": np.asarray(data["beta"], dtype=np.float32),
    }


def _labels_from_human_traj(traj: np.ndarray, beta: np.ndarray) -> dict:
    traj = np.asarray(traj, dtype=np.float32)
    return {
        "transl_worldspace": traj[:, 0:3],
        "global_orient_worldspace": R.from_euler("xyz", traj[:, 3:6]).as_matrix().astype(np.float32),
        "hand_pose": np.stack(
            [_euler_block_to_rotations(block) for block in traj[:, 6:51]],
            axis=0,
        ),
        "beta": np.asarray(beta, dtype=np.float32),
    }


def render_hand_on_image(hand_data: dict, image: np.ndarray, state_mask: list[bool], config: RenderConfig) -> np.ndarray:
    if not HAS_VISUALIZATION:
        raise RuntimeError("visualization package is unavailable")
    intrinsics = np.asarray(hand_data.get("intrinsics", INTRINSICS), dtype=np.float32)
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    image_render = image[:, : int(round(2 * cx)), :].copy() if image.shape[1] >= int(round(4 * cx)) else image.copy()
    width, height = image_render.shape[1], image_render.shape[0]

    class Args:
        mano_model_path = config.mano_path

    visualizer = HandVisualizer(_apply_render_colors(HandConfig(Args()), config), render_gradual_traj=False)
    renderer = Renderer(width, height, (float(fx), float(fy)), "cuda")
    video_frames = [image_render[..., ::-1].astype(np.uint8)]
    masks = (np.array([bool(state_mask[0])]), np.array([bool(state_mask[1])]))
    left = process_single_hand_labels(_labels_from_hand_data(hand_data, "left"), masks[0], visualizer.mano, is_left=True)[0]
    right = process_single_hand_labels(_labels_from_hand_data(hand_data, "right"), masks[1], visualizer.mano, is_left=False)[0]
    extrinsics = (np.broadcast_to(np.eye(3), (1, 3, 3)).copy(), np.zeros((1, 3, 1), dtype=np.float32))
    frames = visualizer._render_hand_trajectory(video_frames, (left, right), masks, extrinsics, renderer, mode="first")
    return frames[0]


def render_human_traj_on_images(
    human_traj_102: np.ndarray,
    images: list[np.ndarray],
    state_mask: list[bool],
    mano_path: str,
    config: RenderConfig,
    intrinsics: np.ndarray,
) -> list[np.ndarray]:
    if not HAS_VISUALIZATION:
        raise RuntimeError("visualization package is unavailable")
    if len(images) == 0:
        raise ValueError("images must not be empty")

    width, height = images[0].shape[1], images[0].shape[0]

    class Args:
        mano_model_path = mano_path

    visualizer = HandVisualizer(_apply_render_colors(HandConfig(Args()), config), render_gradual_traj=False)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    renderer = Renderer(width, height, (float(intrinsics[0, 0]), float(intrinsics[1, 1])), "cuda")
    masks = (
        np.full(len(images), bool(state_mask[0]), dtype=np.bool_),
        np.full(len(images), bool(state_mask[1]), dtype=np.bool_),
    )
    beta = np.zeros(10, dtype=np.float32)
    left = process_single_hand_labels(
        _labels_from_human_traj(human_traj_102[:, 0:51], beta),
        masks[0],
        visualizer.mano,
        is_left=True,
    )[0]
    right = process_single_hand_labels(
        _labels_from_human_traj(human_traj_102[:, 51:102], beta),
        masks[1],
        visualizer.mano,
        is_left=False,
    )[0]
    extrinsics = (
        np.broadcast_to(np.eye(3), (len(images), 3, 3)).copy(),
        np.zeros((len(images), 3, 1), dtype=np.float32),
    )
    video_frames = [image[..., ::-1].astype(np.uint8) for image in images]
    return visualizer._render_hand_trajectory(video_frames, (left, right), masks, extrinsics, renderer, mode="first")


def collect_sequence_frames(config: RenderConfig) -> tuple[list[np.ndarray], dict]:
    training_index = np.load(Path(config.root_dir) / "training_index.npz", allow_pickle=True)
    episode_paths = training_index["episode_paths"]
    sample_indices = training_index["sample_indices"]
    start_episode_idx, _, start_frame_idx = sample_indices[config.sample_idx]

    split_path = Path(config.root_dir) / "split_points.json"
    split_points = json.loads(split_path.read_text()) if split_path.exists() else {}
    episode_key = f"episode_{start_episode_idx:04d}"
    split_info = split_points.get(episode_key)
    seg_start, seg_end = None, None
    if split_info is not None:
        episode_data = json.loads((Path(config.root_dir) / episode_paths[start_episode_idx] / "data.json").read_text())
        boundaries = [0] + split_info["points"] + [len(episode_data["data"]) - 1]
        seg_idx = 0
        for boundary in split_info["points"]:
            if int(start_frame_idx) > boundary:
                seg_idx += 1
            else:
                break
        seg_start, seg_end = boundaries[seg_idx], boundaries[seg_idx + 1]

    if config.whole_episode:
        sample_positions = [
            idx for idx, row in enumerate(sample_indices)
            if int(row[0]) == int(start_episode_idx)
        ]
    else:
        sample_positions = []
        for offset in range(config.num_frames):
            current_idx = config.sample_idx + offset
            if current_idx >= len(sample_indices):
                break
            episode_idx, _, frame_idx = sample_indices[current_idx]
            if episode_idx != start_episode_idx:
                break
            if seg_start is not None and (frame_idx < seg_start or frame_idx > seg_end):
                break
            sample_positions.append(current_idx)

    frames: list[np.ndarray] = []
    payload: dict | None = None
    for current_idx in sample_positions:
        sample = load_dataset_sample(config.root_dir, current_idx, config.statistics_path)
        human_state = convert_inspire_to_human_state_102(sample["state"], sample["state_mask"])
        hand_data = convert_human_state_to_hand_data(human_state, sample["state_mask"], sample["intrinsics"])
        image = sample["image"]
        if config.render_hand and HAS_VISUALIZATION:
            rendered = render_hand_on_image(hand_data, image, sample["state_mask"], config)
            if rendered.shape[:2] != image.shape[:2]:
                canvas = image.copy()
                canvas[: rendered.shape[0], : rendered.shape[1], :] = rendered
                rendered = canvas
        else:
            rendered = image
        frames.append(rendered.astype(np.uint8))
        if payload is None:
            payload = {
                "sample_idx": current_idx,
                "instruction": sample["instruction"],
                "state_mask": sample["state_mask"],
                "human_state_102": human_state,
                "hand_data": hand_data,
                "original_image": image,
                "episode_path": episode_paths[start_episode_idx],
            }
    if payload is None:
        raise RuntimeError("No frames were collected for rendering.")
    return frames, payload


def collect_action_rollout_frames(config: RenderConfig, background_images: list[np.ndarray]) -> tuple[list[np.ndarray], dict]:
    dataset = RoboDatasetCore(
        root_dir=config.root_dir,
        statistics_path=config.statistics_path,
        action_past_window_size=0,
        action_future_window_size=max(config.num_frames - 1, 1),
        image_past_window_size=0,
        image_future_window_size=0,
        load_images=True,
        augmentation=False,
        flip_augmentation=True,
        set_none_ratio=0.0,
        state_mask_prob=0.0,
        target_image_height=224,
    )
    sample = dataset[config.sample_idx]
    state_mask = sample["current_state_mask"].tolist()
    transformed = dataset.transform_trajectory(sample, normalization=False)
    human_state = transformed["current_state"][:102].detach().cpu().numpy().astype(np.float32)
    human_action = transformed["action_list"][:, :102].detach().cpu().numpy().astype(np.float32)

    traj_left = np.zeros((human_action.shape[0] + 1, 51), dtype=np.float32)
    traj_right = np.zeros((human_action.shape[0] + 1, 51), dtype=np.float32)
    if state_mask[0]:
        traj_left = recon_traj(human_state[0:51], human_action[:, 0:51])
    if state_mask[1]:
        traj_right = recon_traj(human_state[51:102], human_action[:, 51:102])
    human_traj = np.concatenate([traj_left, traj_right], axis=1)

    images = background_images[: len(human_traj)]
    if len(images) < len(human_traj):
        images = images + [background_images[-1]] * (len(human_traj) - len(images))
    if config.render_hand and HAS_VISUALIZATION:
        frames = render_human_traj_on_images(human_traj, images, state_mask, config.mano_path, config, sample["intrinsics"])
    else:
        frames = images
    return frames, {
        "state_mask": state_mask,
        "human_state_102": human_state,
        "human_action_102": human_action,
        "human_action_rollout_102": human_traj,
    }


def _concat_frame_pairs(left_frames: list[np.ndarray], right_frames: list[np.ndarray]) -> list[np.ndarray]:
    pair_count = min(len(left_frames), len(right_frames))
    frames = []
    for idx in range(pair_count):
        left = left_frames[idx]
        right = right_frames[idx]
        if left.shape[0] != right.shape[0]:
            target_h = min(left.shape[0], right.shape[0])
            left = left[:target_h]
            right = right[:target_h]
        frames.append(np.concatenate([left, right], axis=1).astype(np.uint8))
    return frames


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    with imageio.get_writer(path, fps=float(fps)) as writer:
        for frame in frames:
            writer.append_data(frame.astype(np.uint8))


def save_one_render_output(config: RenderConfig) -> None:
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    frames, payload = collect_sequence_frames(config)
    np.save(Path(config.output_dir) / "image.npy", payload["hand_data"], allow_pickle=True)
    Image.fromarray(payload["original_image"]).save(Path(config.output_dir) / "image.jpg")
    Image.fromarray(frames[0]).save(Path(config.output_dir) / "image_with_hand.jpg")

    state = payload["human_state_102"]
    info = {
        "sample_idx": payload["sample_idx"],
        "num_frames_requested": config.num_frames,
        "num_frames_rendered": len(frames),
        "whole_episode": config.whole_episode,
        "episode_path": payload["episode_path"],
        "state_mask": payload["state_mask"],
        "instruction": payload["instruction"],
        "fov_x": payload["hand_data"]["fov_x"],
        "human_state_102_summary": {
            "left_transl": state[0:3].tolist(),
            "left_global_orient": state[3:6].tolist(),
            "left_hand_pose": state[6:51].tolist(),
            "right_transl": state[51:54].tolist(),
            "right_global_orient": state[54:57].tolist(),
            "right_hand_pose": state[57:102].tolist(),
            "mask": payload["state_mask"],
        },
    }
    (Path(config.output_dir) / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    _write_video(Path(config.output_dir) / "hand_rendered_video.mp4", frames, config.fps)

    if config.compare_action:
        action_frames, action_payload = collect_action_rollout_frames(config, frames)
        np.save(
            Path(config.output_dir) / "action_rollout.npy",
            {
                "human_state_102": action_payload["human_state_102"],
                "human_action_102": action_payload["human_action_102"],
                "human_action_rollout_102": action_payload["human_action_rollout_102"],
                "state_mask": action_payload["state_mask"],
            },
            allow_pickle=True,
        )
        _write_video(Path(config.output_dir) / "action_rollout_video.mp4", action_frames, config.fps)
        _write_video(
            Path(config.output_dir) / "state_vs_action_video.mp4",
            _concat_frame_pairs(frames, action_frames),
            config.fps,
        )


def save_render_outputs(config: RenderConfig) -> None:
    if config.num_samples <= 1:
        save_one_render_output(config)
        return

    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    sample_count = len(np.load(Path(config.root_dir) / "training_index.npz", allow_pickle=True)["sample_indices"])
    summary = []
    for sample_offset in range(config.num_samples):
        sample_idx = config.sample_idx + sample_offset * config.sample_stride
        if sample_idx >= sample_count:
            break
        sample_dir = root / f"sample_{sample_idx:06d}"
        sample_config = RenderConfig(
            root_dir=config.root_dir,
            sample_idx=sample_idx,
            output_dir=str(sample_dir),
            statistics_path=config.statistics_path,
            mano_path=config.mano_path,
            render_hand=config.render_hand,
            num_frames=config.num_frames,
            fps=config.fps,
            compare_action=config.compare_action,
            num_samples=1,
            sample_stride=config.sample_stride,
            left_color=config.left_color,
            right_color=config.right_color,
            whole_episode=config.whole_episode,
        )
        save_one_render_output(sample_config)
        summary.append({"sample_idx": sample_idx, "output_dir": str(sample_dir)})
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _parse_rgb(value: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("RGB color must have three comma-separated values")
    if any(part < 0.0 or part > 1.0 for part in parts):
        raise argparse.ArgumentTypeError("RGB values must be in [0, 1]")
    return (parts[0], parts[1], parts[2])


def parse_args() -> RenderConfig:
    parser = argparse.ArgumentParser(description="Render dataset hand trajectories for visualization")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--output-dir", default="./extracted_frame")
    parser.add_argument("--statistics-path")
    parser.add_argument("--mano-path", default="./weights/mano")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-action-compare", action="store_true")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--sample-stride", type=int, default=16)
    parser.add_argument("--left-color", type=_parse_rgb, default=(0.10, 0.70, 1.00))
    parser.add_argument("--right-color", type=_parse_rgb, default=(1.00, 0.10, 0.05))
    parser.add_argument("--whole-episode", action="store_true")
    args = parser.parse_args()
    return RenderConfig(
        root_dir=args.root_dir,
        sample_idx=args.sample_idx,
        output_dir=args.output_dir,
        statistics_path=args.statistics_path,
        mano_path=args.mano_path,
        render_hand=not args.no_render,
        num_frames=args.num_frames,
        fps=args.fps,
        compare_action=not args.no_action_compare,
        num_samples=args.num_samples,
        sample_stride=args.sample_stride,
        left_color=args.left_color,
        right_color=args.right_color,
        whole_episode=args.whole_episode,
    )


def main() -> None:
    save_render_outputs(parse_args())


if __name__ == "__main__":
    main()

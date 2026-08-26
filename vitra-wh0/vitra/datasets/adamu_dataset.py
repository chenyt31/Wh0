"""AdamU reader using only the robot-observable EEF and 11-D hand state.

The 51-D VITRA hand state contains the measured AdamU EEF pose and a sparse,
semantic placement of the 11 AdamU joint angles in Wh0's MANO Euler layout.
It deliberately does *not* regress unavailable human MANO joints.
"""

from __future__ import annotations

import json
from pathlib import Path
from torch.utils.data import ConcatDataset

import imageio.v3 as iio
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

from vitra.datasets.dataset_utils import calculate_fov
from vitra.utils.data_utils import GaussianNormalizer, read_dataset_statistics


HAND_WIDTH = 51  # translation(3), wrist rotation(3), MANO joint Euler(45)
RAW_STATE_WIDTH = 122  # left [MANO(51), beta(10)], right [MANO(51), beta(10)]
MODEL_STATE_WIDTH = 212  # Wh0 unified state input (MANO values + reserved padding)
ACTION_WIDTH = 192
# AdamU qpos order is thumb MCP1 (opposition), MCP2 (flexion), PIP
# (flexion), then MCP/DIP flexion for index, middle, ring and pinky.  Wh0's
# MANO pose order is index, middle, pinky, ring, thumb (three joints each).
#
# Every entry is ``(AdamU qpos index, absolute 51-D Wh0 slot,
# left sign, right sign)``.  These signs preserve the established AdamU qpos
# semantics.  In particular, thumb MCP1 is the thumb-root x axis, while
# MCP2/PIP are the y axes of thumb root/next joint.
ADAMU_QPOS_TO_MANO = (
    (0, 42, 1.0, 1.0),   # thumb MCP1: root x / opposition
    (1, 43, -1.0, -1.0), # thumb MCP2: root y / flexion
    (2, 46, -1.0, -1.0), # thumb PIP: next-joint y / flexion
    # Wh0 reflects the left mesh after MANO FK.  Flexion axes are axial
    # vectors, so this reflection cancels the apparent left/right sign flip:
    # positive AdamU MCP/DIP closes all four non-thumb digits on both sides.
    (3, 8, 1.0, 1.0), (4, 11, 1.0, 1.0),
    (5, 17, 1.0, 1.0), (6, 20, 1.0, 1.0),
    (7, 35, 1.0, 1.0), (8, 38, 1.0, 1.0),
    (9, 26, 1.0, 1.0), (10, 29, 1.0, 1.0),
)
# qpos slots 2, 4, 6, 8, and 10 are passive.  Historic exports populated them
# with an approximate 0.7 coupling.  The AdamU hand contract is now exact:
# each passive PIP/DIP follows its source active joint 1:1.
ADAMU_QPOS_SOURCE_INDEX = (0, 1, 1, 3, 3, 5, 5, 7, 7, 9, 9)
# Calibrated AdamU wristRoll -> anatomical robot-palm axes. These are fixed
# embodiment mounts, not fitted from paired MANO trajectories at load time.
ADAMU_EEF_TO_MANO_ROTATION = {
    "left": np.array([[0.06198339, -0.08275302, 0.99464064],
                      [0.11234489, 0.99080197, 0.07543261],
                      [-0.99173418, 0.10706722, 0.07071014]], dtype=np.float32),
    "right": np.array([[0.06198339, -0.08275302, -0.99464064],
                       [-0.11234489, -0.99080197, 0.07543261],
                       [-0.99173418, 0.10706722, -0.07071014]], dtype=np.float32),
}
# Wh0 evaluates MANO_RIGHT and mirrors local x coordinates for left. Its root
# frame is not the anatomical palm frame. These neutral-MANO constants use the
# same joint_palm_frames convention as RobotAlign, never trajectory labels.
MANO_RENDERER_ROOT_TO_PALM = {
    "right": np.array([[-0.99924976, -0.01281482, 0.03654730],
                       [-0.01561217, 0.99688466, -0.07731255],
                       [-0.03544269, -0.07782513, -0.99633682]], dtype=np.float32),
    "left": np.array([[0.99924976, 0.01281482, 0.03654730],
                      [-0.01561217, 0.99688466, 0.07731255],
                      [-0.03544269, -0.07782513, 0.99633682]], dtype=np.float32),
}


def discover_adamu_episode_metadata(root: Path) -> list[Path]:
    """Find AdamU episodes from either a published root or a samples root.

    Production EgoS2 outputs use ``embodiments/adamu/samples/<capture>``.
    Older training configs point directly at the samples directory.  Prefer
    the canonical published layout when both views are present so ``final``
    symlinks cannot duplicate episodes.
    """

    candidates = (
        root / "embodiments/adamu/samples",
        root,
    )
    for samples_root in candidates:
        metadata_paths = sorted(
            samples_root.glob(
                "*/tactile_calib/lerobot_v21/meta/episode_*_source.json"
            )
        )
        if metadata_paths:
            return metadata_paths
    return []


class AdamUSingleEpisodeDataset:
    """Read images from AdamU while supervising with its paired MANO motion."""

    def __init__(
        self,
        root_dir: str,
        action_future_window_size: int = 16,
        load_images: bool = True,
        target_image_height: int = 224,
        statistics_path: str | None = None,
        source_episode_metadata_path: str | None = None,
        **_: object,
    ) -> None:
        root = Path(root_dir)
        self.parquet_path = root / "episode_000000.parquet"
        self.video_path = root / "episode_000000.mp4"
        if not self.parquet_path.is_file() or not self.video_path.is_file():
            raise FileNotFoundError(f"Expected episode_000000.parquet and episode_000000.mp4 under {root}")
        # This is the only runtime state available on a real AdamU robot.
        table = pq.read_table(self.parquet_path, columns=["observation.state", "action"])
        states36 = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        actions36 = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if states36.shape != actions36.shape or states36.ndim != 2 or states36.shape[1] != 36:
            raise ValueError(f"AdamU labels must be matching [N, 36], got {states36.shape}/{actions36.shape}")

        if source_episode_metadata_path is None:
            raise ValueError("AdamU requires source_episode_metadata_path for instruction, K, and paired MANO labels")
        self.source_episode_metadata_path = Path(source_episode_metadata_path)
        if not self.source_episode_metadata_path.is_file():
            raise FileNotFoundError(f"Missing AdamU source episode metadata: {self.source_episode_metadata_path}")
        self.instruction, self.camera_intrinsics, trajectory_metadata = self._load_source_metadata(
            self.source_episode_metadata_path
        )
        mano_path = Path(trajectory_metadata["trajectory"]["mano_source"])
        if not mano_path.is_file():
            raise FileNotFoundError(f"Missing paired canonical MANO archive: {mano_path}")
        self.source_frame_offset = int(trajectory_metadata["trajectory"]["interpolation_frames"])
        self.robot_states36 = states36
        self.robot_actions36 = actions36
        self._load_mano_labels(mano_path)
        # Direct AdamU supervision includes the exporter preroll as well.
        self.source_frame_offset = 0
        self.action_future_window_size = int(action_future_window_size)
        self.load_images = load_images
        self.target_image_height = int(target_image_height)
        if statistics_path is None:
            raise ValueError("AdamU strict-overfit requires the supplied VITRA statistics_path")
        self.data_statistics = read_dataset_statistics(statistics_path)
        self.gaussian_normalizer = GaussianNormalizer(self.data_statistics)

    @staticmethod
    def _load_source_metadata(source_metadata_path: Path) -> tuple[str, np.ndarray, dict]:
        source = json.loads(source_metadata_path.read_text(encoding="utf-8"))
        episode_index = int(source["episode_index"])
        trajectory_metadata_path = Path(source["trajectory"]) / "metadata.json"
        # Exported sample bundles deliberately contain only robot_align and
        # LeRobot data.  Their source JSON preserves the original absolute
        # workspace path, so resolve it to the bundled capture when needed.
        if not trajectory_metadata_path.is_file():
            bundled_trajectory = source_metadata_path.parents[3] / "robot_align" / Path(source["trajectory"]).name
            trajectory_metadata_path = bundled_trajectory / "metadata.json"
        if not trajectory_metadata_path.is_file():
            raise FileNotFoundError(f"Missing AdamU trajectory metadata: {trajectory_metadata_path}")
        trajectory_metadata = json.loads(trajectory_metadata_path.read_text(encoding="utf-8"))
        intrinsics = np.asarray(trajectory_metadata["camera_calibration"]["intrinsics"], dtype=np.float32)
        if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
            raise ValueError(f"Invalid calibrated intrinsics in {trajectory_metadata_path}: {intrinsics}")
        episodes_path = source_metadata_path.parent / "episodes.jsonl"
        if not episodes_path.is_file():
            raise FileNotFoundError(f"Missing LeRobot episodes metadata: {episodes_path}")
        episode = next((json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines()
                        if line.strip() and int(json.loads(line)["episode_index"]) == episode_index), None)
        if episode is None or not episode.get("tasks") or not episode["tasks"][0].strip():
            raise ValueError(f"No task instruction for episode {episode_index} in {episodes_path}")
        task = str(episode["tasks"][0]).strip()
        return f"Left hand:{task} Right hand:{task}", intrinsics, trajectory_metadata

    @staticmethod
    def _mano_pose_euler(axis_angle: np.ndarray, is_left: bool) -> np.ndarray:
        """Convert 15 MANO axis-angle joints to VITRA's xyz Euler layout."""
        matrices = R.from_rotvec(axis_angle.reshape(-1, 3)).as_matrix()
        if is_left:
            # Wh0 renders a left hand as a mirrored MANO_RIGHT mesh.  The
            # HumanSyn archive uses MANO_LEFT coordinates, so conjugate every
            # local joint rotation before handing it to Wh0's left renderer.
            reflection = np.diag((-1.0, 1.0, 1.0))
            matrices = reflection @ matrices @ reflection
        return R.from_matrix(matrices).as_euler("xyz").reshape(-1, 45).astype(np.float32)

    @staticmethod
    def _mano_pose_rotvec(axis_angle: np.ndarray, is_left: bool) -> np.ndarray:
        """Return the same pose in Wh0's MANO_RIGHT coordinate convention."""
        matrices = R.from_rotvec(axis_angle.reshape(-1, 3)).as_matrix()
        if is_left:
            reflection = np.diag((-1.0, 1.0, 1.0))
            matrices = reflection @ matrices @ reflection
        return R.from_matrix(matrices).as_rotvec().reshape(-1, 45).astype(np.float32)

    @staticmethod
    def _eef_matrices(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decode LeRobot [xyz, qw qx qy qz] EEF state in camera coordinates."""
        return values[:, :3], R.from_quat(values[:, 3:7][:, [1, 2, 3, 0]]).as_matrix()

    @staticmethod
    def _fit_linear_adapter(qpos: np.ndarray, target: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
        """Fit y = [1, qpos] @ weights; retained for identical server inference."""
        design = np.concatenate((np.ones((len(qpos), 1), dtype=np.float32), qpos), axis=1)
        penalty = np.eye(design.shape[1], dtype=np.float32) * ridge
        penalty[0, 0] = 0.0
        return np.linalg.solve(design.T @ design + penalty, design.T @ target).astype(np.float32)

    @staticmethod
    def _apply_linear_adapter(qpos: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return (np.concatenate((np.ones((len(qpos), 1), dtype=np.float32), qpos), axis=1) @ weights).astype(np.float32)

    def _load_mano_labels(self, mano_path: Path) -> None:
        archive = np.load(mano_path)
        # MANO beta is render-only.  It does not create action labels.
        def padded_beta(side: str) -> np.ndarray:
            beta = np.asarray(archive[f"{side}_betas"], dtype=np.float32)
            if beta.ndim != 2 or beta.shape[1] != 10 or len(beta) == 0:
                raise ValueError(f"Invalid {side} MANO beta shape {beta.shape} in {mano_path}")
            padded = np.concatenate(
                (np.repeat(beta[:1], self.source_frame_offset, axis=0), beta),
                axis=0,
            )
            expected_shape = (len(self.robot_states36), 10)
            if padded.shape != expected_shape:
                raise ValueError(
                    f"AdamU {side} MANO beta/robot length mismatch in {mano_path}: "
                    f"beta={len(beta)}, interpolation_frames={self.source_frame_offset}, "
                    f"robot={len(self.robot_states36)}"
                )
            return padded

        self.left_betas, self.right_betas = padded_beta("left"), padded_beta("right")
        self.left_state, self.left_state_mask = self._direct_hand_state(0, "left")
        self.right_state, self.right_state_mask = self._direct_hand_state(18, "right")
        self.left_action = self._direct_hand_action(0, "left")
        self.right_action = self._direct_hand_action(18, "right")
        self.left_pose_adapter = self.right_pose_adapter = np.zeros((0, 0), dtype=np.float32)
        self.left_wrist_mount = self.right_wrist_mount = np.zeros(6, dtype=np.float32)
        return

        # Retained below as calibration reference code; direct AdamU mode above
        # deliberately does not infer unavailable MANO joints.
        def calibrate_hand(side: str, robot_offset: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            trans = np.asarray(archive[f"{side}_trans"], dtype=np.float32)
            root_rot = R.from_rotvec(np.asarray(archive[f"{side}_rot"], dtype=np.float32)).as_matrix()
            pose_rotvec = self._mano_pose_rotvec(
                np.asarray(archive[f"{side}_hand_pose"], dtype=np.float32), is_left=(side == "left")
            )
            beta = np.asarray(archive[f"{side}_betas"], dtype=np.float32)
            robot = self.robot_states36[self.source_frame_offset:]
            eef_t, eef_r = self._eef_matrices(robot[:, robot_offset:robot_offset + 7])
            qpos = robot[:, robot_offset + 7:robot_offset + 18]
            if trans.shape != eef_t.shape or pose_rotvec.shape != (len(trans), 45) or beta.shape != (len(trans), 10):
                raise ValueError(f"Invalid paired {side} MANO shapes in {mano_path}")
            # Fixed EEF -> MANO transform: this is what server-side inference
            # applies when it receives only a robot EEF state.
            mount_r = R.from_matrix(np.swapaxes(eef_r, -1, -2) @ root_rot).mean().as_matrix()
            # The paired MANO translation is a human-hand placement, not the
            # AdamU wrist origin in the robot-composite image.  Preserving EEF
            # xyz is the deployable robot overlay contract.  A future measured
            # AdamU palm-origin offset can be inserted here after calibration.
            mount_t = np.zeros(3, dtype=np.float32)
            reconstructed_t = eef_t
            reconstructed_r = eef_r @ mount_r
            weights = self._fit_linear_adapter(qpos, pose_rotvec)
            reconstructed_pose = self._mano_pose_euler(self._apply_linear_adapter(qpos, weights), is_left=False)
            state = np.concatenate((reconstructed_t, R.from_matrix(reconstructed_r).as_euler("xyz"), reconstructed_pose), axis=1).astype(np.float32)
            return state, beta, weights, np.concatenate((mount_t, R.from_matrix(mount_r).as_rotvec())).astype(np.float32)

        self.left_state, self.left_betas, self.left_pose_adapter, self.left_wrist_mount = calibrate_hand("left", 0)
        self.right_state, self.right_betas, self.right_pose_adapter, self.right_wrist_mount = calibrate_hand("right", 18)
        if self.left_state.shape != self.right_state.shape:
            raise ValueError("Paired left/right MANO trajectories must have equal lengths")
        # VITRA action: position delta, camera-frame LEFT rotational delta,
        # then the next absolute MANO joint pose.
        self.left_action = self._make_actions(self.left_state)
        self.right_action = self._make_actions(self.right_state)

    def _direct_hand_state(self, offset: int, side: str) -> tuple[np.ndarray, np.ndarray]:
        values = self.robot_states36[:, offset:offset + 18]
        state = np.zeros((len(values), HAND_WIDTH), dtype=np.float32)
        state[:, :3] = values[:, :3]
        eef_rotation = R.from_quat(values[:, 3:7][:, [1, 2, 3, 0]]).as_matrix()
        palm_rotation = eef_rotation @ ADAMU_EEF_TO_MANO_ROTATION[side]
        root_rotation = palm_rotation @ MANO_RENDERER_ROOT_TO_PALM[side].T
        state[:, 3:6] = R.from_matrix(root_rotation).as_euler("xyz").astype(np.float32)
        for qpos_index, mano_index, left_sign, right_sign in ADAMU_QPOS_TO_MANO:
            sign = left_sign if side == "left" else right_sign
            source_index = ADAMU_QPOS_SOURCE_INDEX[qpos_index]
            state[:, mano_index] = sign * values[:, 7 + source_index]
        mask = np.zeros(HAND_WIDTH, dtype=np.bool_)
        mask[:6] = True
        mask[[destination for _, destination, _, _ in ADAMU_QPOS_TO_MANO]] = True
        return state, mask

    def _direct_hand_action(self, offset: int, side: str) -> np.ndarray:
        values = self.robot_actions36[:, offset:offset + 18]
        action = np.zeros((len(values), HAND_WIDTH), dtype=np.float32)
        action[:, :3] = values[:, :3]
        local_delta = R.from_quat(values[:, 3:7][:, [1, 2, 3, 0]]).as_matrix()
        current_eef = R.from_quat(self.robot_states36[:, offset + 3:offset + 7][:, [1, 2, 3, 0]]).as_matrix()
        # The fixed EEF->palm mount cancels in R_next @ R_current.T.
        action[:, 3:6] = R.from_matrix(current_eef @ local_delta @ np.swapaxes(current_eef, -1, -2)).as_euler("xyz").astype(np.float32)
        for qpos_index, mano_index, left_sign, right_sign in ADAMU_QPOS_TO_MANO:
            sign = left_sign if side == "left" else right_sign
            source_index = ADAMU_QPOS_SOURCE_INDEX[qpos_index]
            action[:, mano_index] = sign * values[:, 7 + source_index]
        return action

    @staticmethod
    def _make_actions(states: np.ndarray) -> np.ndarray:
        actions = np.zeros_like(states)
        actions[:-1, :3] = states[1:, :3] - states[:-1, :3]
        current = R.from_euler("xyz", states[:-1, 3:6]).as_matrix()
        target = R.from_euler("xyz", states[1:, 3:6]).as_matrix()
        actions[:-1, 3:6] = R.from_matrix(target @ np.swapaxes(current, -1, -2)).as_euler("xyz")
        actions[:-1, 6:] = states[1:, 6:]
        # Terminal is zero translation, identity rotation, and the final pose.
        actions[-1, 6:] = states[-1, 6:]
        return actions.astype(np.float32)

    def __len__(self) -> int:
        return len(self.left_state)

    def _image(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        image = np.asarray(iio.imread(self.video_path, index=index + self.source_frame_offset), dtype=np.uint8)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected RGB frame from {self.video_path}, got {image.shape}")
        h, w = image.shape[:2]
        resized = np.asarray(Image.fromarray(image).resize((round(w * self.target_image_height / h), self.target_image_height)), dtype=np.uint8)
        intrinsics = self.camera_intrinsics.copy()
        intrinsics[0, :] *= resized.shape[1] / w
        intrinsics[1, :] *= resized.shape[0] / h
        intrinsics[2, :] = (0.0, 0.0, 1.0)
        return resized, intrinsics

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < len(self):
            raise IndexError(index)
        # Keep the raw state in the same 122-D layout as the Wh0 statistics.
        # It is normalized in that layout below, then placed into the 212-D
        # unified model input with beta slots intentionally omitted.
        state = np.zeros(RAW_STATE_WIDTH, dtype=np.float32)
        state[:HAND_WIDTH] = self.left_state[index]
        state[51:61] = self.left_betas[index]
        state[61:112] = self.right_state[index]
        state[112:122] = self.right_betas[index]
        state_mask = np.zeros(RAW_STATE_WIDTH, dtype=np.bool_)
        state_mask[:HAND_WIDTH] = self.left_state_mask
        state_mask[61:112] = self.right_state_mask
        action_rows, action_masks = [], []
        for offset in range(self.action_future_window_size):
            target = index + offset
            row = np.zeros(ACTION_WIDTH, dtype=np.float32)
            row_mask = np.zeros(ACTION_WIDTH, dtype=np.bool_)
            if target < len(self):
                row[:HAND_WIDTH] = self.left_action[target]
                row[HAND_WIDTH:2 * HAND_WIDTH] = self.right_action[target]
                # Keep the complete 51-D MANO action space supervised for
                # each hand.  Slots not driven by AdamU are canonical zero in
                # raw MANO coordinates and become their normalizer-consistent
                # values after ``transform_trajectory``.  This preserves the
                # dense Wh0 pretraining output contract while state remains
                # sparse (only robot-observable quantities are conditioned).
                row_mask[:HAND_WIDTH] = True
                row_mask[HAND_WIDTH:2 * HAND_WIDTH] = True
            action_rows.append(row)
            action_masks.append(row_mask)
        image, intrinsics = self._image(index)
        return {
            "instruction": self.instruction,
            "image_list": image[None, ...],
            "image_mask": np.array([True], dtype=np.bool_),
            "intrinsics": intrinsics,
            "fov": calculate_fov(image.shape[0], image.shape[1], intrinsics),
            "current_state": torch.from_numpy(state),
            "current_state_mask": torch.from_numpy(state_mask),
            "action_list": torch.from_numpy(np.stack(action_rows)),
            "action_mask": torch.from_numpy(np.stack(action_masks)),
            # Raw beta is deliberately not inserted in the 212-D VITRA state:
            # pad_state_human drops it.  It is retained for faithful mesh replay.
            "beta_left": self.left_betas[index].copy(),
            "beta_right": self.right_betas[index].copy(),
        }

    def set_global_data_statistics(self, _: dict) -> None:
        return None

    def deployment_adapter(self) -> dict[str, object]:
        """Serializable calibration required to reconstruct 51-D inputs from AdamU telemetry."""
        return {
            "version": 1,
            "input_contract": "per hand [eef_xyz(3), eef_quat_wxyz(4), qpos_11]",
            "output_contract": "per hand VITRA [mano_trans(3), mano_euler_xyz(3), mano_joint_euler_xyz(45)]",
            "pose_adapter": {
                "left": self.left_pose_adapter.tolist(),
                "right": self.right_pose_adapter.tolist(),
            },
            "wrist_mount": {
                "left": self.left_wrist_mount.tolist(),
                "right": self.right_wrist_mount.tolist(),
                "encoding": "[translation_in_eef_m(3), rotation_axis_angle(3)]",
            },
        }

    def transform_trajectory(self, sample_dict: dict, normalization: bool = True) -> dict:
        if not normalization:
            return sample_dict
        raw_state = np.asarray(sample_dict["current_state"], dtype=np.float32)
        if raw_state.shape != (RAW_STATE_WIDTH,):
            raise ValueError(f"Expected raw AdamU state [{RAW_STATE_WIDTH}], got {raw_state.shape}")
        normalized_state = self.gaussian_normalizer.normalize_state(raw_state)
        output_state = np.zeros(MODEL_STATE_WIDTH, dtype=np.float32)
        output_state[:51] = normalized_state[:51]
        output_state[51:102] = normalized_state[61:112]
        output_mask = np.zeros(MODEL_STATE_WIDTH, dtype=np.bool_)
        raw_mask = np.asarray(sample_dict["current_state_mask"], dtype=np.bool_)
        output_mask[:51] = raw_mask[:51]
        output_mask[51:102] = raw_mask[61:112]
        sample_dict["current_state"] = torch.from_numpy(output_state)
        sample_dict["current_state_mask"] = torch.from_numpy(output_mask)
        raw_actions = np.asarray(sample_dict["action_list"], dtype=np.float32)
        output_actions = np.zeros_like(raw_actions)
        output_actions[:, :102] = self.gaussian_normalizer.normalize_action(raw_actions[:, :102])
        sample_dict["action_list"] = torch.from_numpy(output_actions)
        return sample_dict


class AdamUSamplesDataset(ConcatDataset):
    """Read every exported AdamU LeRobot episode below a samples directory.

    The source tree remains read-only: each child dataset refers directly to
    its parquet, video, and source metadata rather than materializing a copy.
    """

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
        metadata_paths = discover_adamu_episode_metadata(root)
        if not metadata_paths:
            raise FileNotFoundError(f"No AdamU LeRobot episode metadata found below {root}")
        datasets = []
        for metadata_path in metadata_paths:
            lerobot_root = metadata_path.parent.parent
            episode_stem = metadata_path.stem.removesuffix("_source")
            parquet_path = lerobot_root / "data/chunk-000" / f"{episode_stem}.parquet"
            video_path = lerobot_root / "videos/chunk-000/observation.images.camera" / f"{episode_stem}.mp4"
            if not parquet_path.is_file() or not video_path.is_file():
                raise FileNotFoundError(f"Missing paired LeRobot files for {metadata_path}")
            dataset = AdamUSingleEpisodeDataset.__new__(AdamUSingleEpisodeDataset)
            dataset.parquet_path = parquet_path
            dataset.video_path = video_path
            table = pq.read_table(parquet_path, columns=["observation.state", "action"])
            states36 = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            actions36 = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            if states36.shape != actions36.shape or states36.ndim != 2 or states36.shape[1] != 36:
                raise ValueError(f"AdamU labels must be matching [N, 36], got {states36.shape}/{actions36.shape}")
            dataset.source_episode_metadata_path = metadata_path
            dataset.instruction, dataset.camera_intrinsics, trajectory_metadata = dataset._load_source_metadata(metadata_path)
            # RobotAlign prepends a variable home-to-first-pose interpolation
            # segment. Use its recorded length while aligning MANO beta, then
            # reset the image offset because exported LeRobot video already
            # contains those prepended frames.
            dataset.source_frame_offset = int(trajectory_metadata["trajectory"]["interpolation_frames"])
            dataset.robot_states36 = states36
            dataset.robot_actions36 = actions36
            mano_path = Path(trajectory_metadata["trajectory"]["mano_source"])
            if mano_path.is_file():
                dataset._load_mano_labels(mano_path)
            else:
                # Sample bundles omit HumanSyn's render-only MANO archive.
                # AdamU's deployable state/action labels are fully defined by
                # the 36-D robot telemetry, so retain zero beta placeholders.
                dataset.left_betas = np.zeros((len(states36), 10), dtype=np.float32)
                dataset.right_betas = np.zeros((len(states36), 10), dtype=np.float32)
                dataset.left_state, dataset.left_state_mask = dataset._direct_hand_state(0, "left")
                dataset.right_state, dataset.right_state_mask = dataset._direct_hand_state(18, "right")
                dataset.left_action = dataset._direct_hand_action(0, "left")
                dataset.right_action = dataset._direct_hand_action(18, "right")
                dataset.left_pose_adapter = dataset.right_pose_adapter = np.zeros((0, 0), dtype=np.float32)
                dataset.left_wrist_mount = dataset.right_wrist_mount = np.zeros(6, dtype=np.float32)
            dataset.source_frame_offset = 0
            dataset.action_future_window_size = int(action_future_window_size)
            dataset.load_images = load_images
            dataset.target_image_height = int(target_image_height)
            if statistics_path is None:
                raise ValueError("AdamU requires statistics_path")
            dataset.data_statistics = read_dataset_statistics(statistics_path)
            dataset.gaussian_normalizer = GaussianNormalizer(dataset.data_statistics)
            datasets.append(dataset)
        self.data_statistics = datasets[0].data_statistics
        super().__init__(datasets)

    def transform_trajectory(self, sample_dict: dict, normalization: bool = True) -> dict:
        # All children share a normalizer; their transformation is identical.
        return self.datasets[0].transform_trajectory(sample_dict, normalization)

    def set_global_data_statistics(self, _: dict) -> None:
        return None

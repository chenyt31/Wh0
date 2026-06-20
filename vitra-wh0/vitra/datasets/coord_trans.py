"""G1 camera/base coordinate transforms."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


G1_PELVIS_TO_TORSO = np.array([-0.0039635, 0.0, 0.044], dtype=np.float32)
G1_TORSO_TO_D435 = np.array([0.0576235, 0.01753, 0.42987], dtype=np.float32)
G1_D435_MOUNT_PITCH = 0.8307767239493009


def transform_target_to_base(
    t_base_camera: np.ndarray,
    t_camera_object: np.ndarray,
    t_object_grasp: np.ndarray | None = None,
) -> np.ndarray:
    """Return a target pose in the robot base frame."""
    t_base_object = t_base_camera @ t_camera_object
    if t_object_grasp is None:
        return t_base_object
    return t_base_object @ t_object_grasp


def get_target_pose_in_base_frame(
    T_base_camera: np.ndarray,
    T_camera_object: np.ndarray,
    T_grasp_offset: np.ndarray | None = None,
) -> np.ndarray:
    """Backward-compatible wrapper for older callers."""
    return transform_target_to_base(T_base_camera, T_camera_object, T_grasp_offset)


def _make_transform(rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    if rotation is not None:
        transform[:3, :3] = rotation
    if translation is not None:
        transform[:3, 3] = translation
    return transform


def get_camera_pose_in_base_frame(
    waist_yaw: float = 0.0,
    waist_roll: float = 0.0,
    waist_pitch: float = 0.0,
) -> np.ndarray:
    """Return the G1 D435 optical-frame pose in the pelvis/base frame."""
    t_pelvis_yaw = _make_transform(R.from_euler("z", waist_yaw).as_matrix())
    t_yaw_roll = _make_transform(
        R.from_euler("x", waist_roll).as_matrix(),
        G1_PELVIS_TO_TORSO,
    )
    t_roll_torso = _make_transform(R.from_euler("y", waist_pitch).as_matrix())
    t_torso_camera_link = _make_transform(
        R.from_euler("y", G1_D435_MOUNT_PITCH).as_matrix(),
        G1_TORSO_TO_D435,
    )

    # Camera link: x forward, y left, z up.
    # Optical frame: z forward, x right, y down.
    r_link_optical = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    t_link_optical = _make_transform(r_link_optical)

    return t_pelvis_yaw @ t_yaw_roll @ t_roll_torso @ t_torso_camera_link @ t_link_optical


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    t_base_camera = get_camera_pose_in_base_frame()
    optical_z_base = t_base_camera[:3, 2]
    pitch_deg = np.degrees(np.arctan2(-optical_z_base[2], optical_z_base[0]))
    print("T_base_camera_optical:\n", t_base_camera)
    print("Optical z-axis in base frame:", optical_z_base)
    print(f"Optical pitch: {pitch_deg:.2f} deg")

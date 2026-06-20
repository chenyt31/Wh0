"""WebSocket inference server for WM-H to Wh0 robot actions."""

import os
import json
import asyncio
import tempfile
import websockets
import argparse
import numpy as np
import torch
import base64
import cv2
import imageio
import threading
import atexit
import signal
from io import BytesIO
from PIL import Image
from pathlib import Path
import multiprocessing as mp
from types import SimpleNamespace
from scipy.spatial.transform import Rotation as R
from datetime import datetime

from paths import WH0_ROOT, configure_imports

configure_imports(require_xr=False)

from visualization.visualize_core import Config as HandConfig
from visualization.visualize_core import HandVisualizer
from visualization.render_utils import Renderer

from vitra.utils.data_utils import resize_short_side_to_target
from vitra.utils.config_utils import load_config
from vitra.datasets.robot_dataset import (
    transfer_inspire_to_human,
    pad_state_robot,
    add_passive_joints_to_human_action,
    get_camera_pose_in_base_frame,
    transform_pose_to_camera_frame,
    transform_camera_pose_to_base_frame,
    correct_gt_wrist_orientation,
    correct_gt_wrist_orientation_inverse,
    se3_to_trans_euler,
    denormalize_inspire_qpos,
    mano_hand_pose_to_inspire_qpos,
)
from vitra.datasets.dataset_utils import ActionFeature, StateFeature
from vitra.utils.data_utils import recon_traj


def _apply_wrist_correction(pose_base: np.ndarray) -> np.ndarray:
    """Apply the final wrist-frame correction used by the G1 controller."""
    t_offset = np.array([0, 0.0, 0], dtype=np.float32)
    Rx = R.from_euler('x', 0, degrees=True).as_matrix()
    Ry = R.from_euler('y', 0, degrees=True).as_matrix()
    Rz = R.from_euler('z', 0, degrees=True).as_matrix()
    pose_corrected = pose_base.copy()
    pose_corrected[:3, 3] = pose_base[:3, 3] + t_offset
    pose_corrected[:3, :3] = pose_base[:3, :3] @ Rx @ Ry @ Rz
    return pose_corrected


CAMERA_FRAME_ACTION_OFFSET = np.array([0.0, 0.00, 0.0], dtype=np.float32)

def _apply_predicted_position_offset(
    traj_transl: np.ndarray,
    offset: np.ndarray = CAMERA_FRAME_ACTION_OFFSET
) -> np.ndarray:
    """Apply a camera-frame translation offset to predicted positions."""
    return traj_transl + offset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


_video_save_path = str(WH0_ROOT / "validation_outputs" / "deployment" / "model_server_visualization.mp4")
_visualization_enabled = True
_visualization_initialized = False
_visualizer = None
_hand_config = None
_mano_model = None
_visualization_threads = []
_visualization_lock = threading.Lock()


def _wait_for_visualization_threads(timeout=None):
    """Wait for all pending visualization threads to complete.

    Args:
        timeout: Maximum seconds to wait per thread. None means wait indefinitely.
    """
    global _visualization_threads
    with _visualization_lock:
        threads = _visualization_threads.copy()

    if not threads:
        return

    print(f"[Viz] Waiting for {len(threads)} visualization thread(s) to complete...", flush=True)
    for t in threads:
        t.join(timeout=timeout)

    with _visualization_lock:
        _visualization_threads = [t for t in _visualization_threads if t.is_alive()]
        if _visualization_threads:
            print(f"[Viz] {len(_visualization_threads)} thread(s) still running", flush=True)
            _visualization_threads = []
        else:
            print("[Viz] All visualization threads completed", flush=True)


_video_segment_dir = None
_video_segment_index = 0
_visualization_video_path = None


def _save_video_frames(frames, save_path, fps=8, append=True):
    """Write frames as a segment; _close_video_writer merges all segments."""
    global _video_segment_dir, _video_segment_index, _visualization_video_path

    frames = np.asarray(frames)
    if frames.ndim == 4 and frames.shape[0] == 1:
        frames = frames[0]

    if not append:
        if _video_segment_dir is not None and os.path.isdir(_video_segment_dir):
            for f in os.listdir(_video_segment_dir):
                os.remove(os.path.join(_video_segment_dir, f))
        _video_segment_index = 0
        _visualization_video_path = None

    if _video_segment_dir is None:
        _visualization_video_path = os.path.abspath(save_path)
        _video_segment_dir = tempfile.mkdtemp(
            prefix='wh0_visualization_',
            dir=os.path.dirname(_visualization_video_path),
        )
        print(f"[Viz] Created temp segment directory: {_video_segment_dir}")

    seg_path = os.path.join(_video_segment_dir, f'seg_{_video_segment_index:04d}.mp4')
    _video_segment_index += 1

    print(f"[Viz] Saving segment {_video_segment_index}: {frames.shape}, path: {seg_path}")

    try:
        imageio.mimsave(
            seg_path,
            frames,
            fps=fps,
            codec='libx264',
            macro_block_size=1,
        )
        print(f"[Viz] Segment {_video_segment_index} saved successfully")
    except Exception as e:
        print(f"[Viz] Failed to write segment {seg_path}: {e}")
        import traceback
        traceback.print_exc()
        return

    _visualization_video_path = os.path.abspath(save_path)


def _close_video_writer():
    """Merge all segment files into the final output using FFmpeg concat, then clean up.

    Returns:
        The final saved path, or None if nothing was saved.
    """
    global _video_segment_dir, _video_segment_index, _visualization_video_path

    if _video_segment_dir is None or _video_segment_index == 0:
        print("[Viz] No video segments to save")
        return None

    if _visualization_video_path is None:
        print("[Viz] No video final path set")
        return None

    final_path = os.path.abspath(_visualization_video_path)
    if os.path.exists(final_path):
        base, ext = os.path.splitext(final_path)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_path = f"{base}_{timestamp}{ext}"
        print(f"[Viz] Target file exists, saving with timestamp: {final_path}")

    print(f"[Viz] Merging {_video_segment_index} segments into {final_path}...")

    concat_list_path = os.path.join(_video_segment_dir, 'concat.txt')
    with open(concat_list_path, 'w') as f:
        for i in range(_video_segment_index):
            seg = os.path.join(_video_segment_dir, f'seg_{i:04d}.mp4')
            f.write(f"file '{seg}'\n")

    try:
        import subprocess
        result = subprocess.run(
            [
                'ffmpeg', '-y',
                '-f', 'concat', '-safe', '0',
                '-i', concat_list_path,
                '-c:v', 'copy',
                final_path
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            print(f"[Viz] Video saved ({_video_segment_index} segments): {final_path}")
        else:
            print(f"[Viz] FFmpeg concat failed: {result.stderr[-500:]}")
            final_path = None
    except subprocess.TimeoutExpired:
        print("[Viz] FFmpeg concat timed out")
        final_path = None
    except Exception as e:
        print(f"[Viz] Failed to merge segments: {e}")
        final_path = None
    finally:
        if _video_segment_dir is not None and os.path.isdir(_video_segment_dir):
            try:
                if final_path and os.path.exists(final_path):
                    for fname in os.listdir(_video_segment_dir):
                        os.remove(os.path.join(_video_segment_dir, fname))
                    os.rmdir(_video_segment_dir)
                    print("[Viz] Cleaned up temp segment files")
                else:
                    print(f"[Viz] Kept segment files in {_video_segment_dir} for debugging")
            except Exception as cleanup_err:
                print(f"[Viz] Cleanup error: {cleanup_err}")

        _video_segment_dir = None
        _video_segment_index = 0
        _visualization_video_path = None

    return final_path




def _init_visualization(mano_path=None, fps=8):
    """Initialize MANO rendering utilities."""
    global _visualizer, _hand_config, _mano_model, _visualization_initialized

    if _visualization_initialized:
        return
    if mano_path is None:
        mano_path = str(WH0_ROOT / "weights" / "mano")

    args_obj = SimpleNamespace(mano_model_path=mano_path)
    _hand_config = HandConfig(args_obj)
    _hand_config.FPS = fps
    _visualizer = HandVisualizer(_hand_config, render_gradual_traj=False)
    _mano_model = _visualizer.mano

    _visualization_initialized = True
    print("[Viz] Visualization components initialized.")


def _render_and_save_visualization(image_np, unnorm_action, beta_left, beta_right,
                                    fov, intrinsics, use_left, use_right,
                                    viz_traj_left=None, viz_traj_right=None):
    """Render hand mesh overlays and append them to the visualization video."""
    global _visualizer, _hand_config

    if not _visualization_enabled:
        return

    try:
        if _visualizer is None:
            _init_visualization()
            if _visualizer is None:
                print("[Viz] Failed to initialize visualizer, skipping visualization")
                return

        if unnorm_action.ndim == 3:
            unnorm_action = unnorm_action[0]

        T = unnorm_action.shape[0]
        H, W = image_np.shape[:2]

        traj_left = viz_traj_left
        traj_right = viz_traj_right

        if use_left and viz_traj_left is not None:
            T = viz_traj_left.shape[0]
        elif use_right and viz_traj_right is not None:
            T = viz_traj_right.shape[0]
        else:
            T = unnorm_action.shape[0]

        traj_mask = np.tile(np.array([[use_left, use_right]], dtype=bool), (T, 1))
        left_hand_mask = traj_mask[:, 0]
        right_hand_mask = traj_mask[:, 1]
        hand_mask = (left_hand_mask, right_hand_mask)

        left_hand_labels = None
        right_hand_labels = None

        if use_left and viz_traj_left is not None:
            left_hand_labels = {
                'transl_worldspace': viz_traj_left[:, 0:3],
                'global_orient_worldspace': R.from_euler('xyz', viz_traj_left[:, 3:6]).as_matrix(),
                'hand_pose': euler_traj_to_rotmat_traj(viz_traj_left[:, 6:51]),
                'beta': beta_left,
            }

        if use_right and viz_traj_right is not None:
            right_hand_labels = {
                'transl_worldspace': viz_traj_right[:, 0:3],
                'global_orient_worldspace': R.from_euler('xyz', viz_traj_right[:, 3:6]).as_matrix(),
                'hand_pose': euler_traj_to_rotmat_traj(viz_traj_right[:, 6:51]),
                'beta': beta_right,
            }

        verts_left_worldspace = None
        verts_right_worldspace = None

        if use_left and left_hand_labels is not None:
            func = _get_process_single_hand_labels()
            verts_left_worldspace, _ = func(
                left_hand_labels, left_hand_mask, _mano_model, is_left=True
            )

        if use_right and right_hand_labels is not None:
            func = _get_process_single_hand_labels()
            verts_right_worldspace, _ = func(
                right_hand_labels, right_hand_mask, _mano_model, is_left=False
            )

        hand_traj_wordspace = (verts_left_worldspace, verts_right_worldspace)

        R_w2c = np.broadcast_to(np.eye(3), (T, 3, 3)).copy()
        t_w2c = np.zeros((T, 3, 1), dtype=np.float32)
        extrinsics = (R_w2c, t_w2c)

        # visualize_core expects BGR input and returns RGB frames.
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        resize_video_frames = [image_bgr] * T

        fx_exo = intrinsics[0, 0]
        fy_exo = intrinsics[1, 1]
        renderer = Renderer(W, H, (fx_exo, fy_exo), 'cuda')

        save_frames = _visualizer._render_hand_trajectory(
            resize_video_frames,
            hand_traj_wordspace,
            hand_mask,
            extrinsics,
            renderer,
            mode='first'
        )

        if save_frames:
            all_frames = [image_np] + save_frames
            frames_np = np.stack(all_frames)
            _save_video_frames(frames_np, _video_save_path, fps=8, append=True)
        else:
            print("[Viz] No frames returned from renderer")

    except Exception as e:
        import traceback
        print(f"[Viz] Rendering error: {e}")
        print(f"[Viz] Traceback: {traceback.format_exc()}")


def euler_traj_to_rotmat_traj(euler_traj):
    """Convert Euler angle trajectory to rotation matrix trajectory.

    Args:
        euler_traj: numpy array of shape [T, 45] (T timesteps, 15 joints * 3 euler angles)

    Returns:
        numpy array of shape [T, 15, 3, 3]
    """
    T = euler_traj.shape[0]
    hand_pose = euler_traj.reshape(-1, 3)  # [T*15, 3]
    pose_matrices = R.from_euler('xyz', hand_pose).as_matrix()  # [T*15, 3, 3]
    pose_matrices = pose_matrices.reshape(T, 15, 3, 3)  # [T, 15, 3, 3]
    return pose_matrices


_process_single_hand_labels_func = None

def _get_process_single_hand_labels():
    """Lazy import to avoid circular dependency."""
    global _process_single_hand_labels_func
    if _process_single_hand_labels_func is None:
        from visualization.visualize_core import process_single_hand_labels as _func
        _process_single_hand_labels_func = _func
    return _process_single_hand_labels_func


def process_single_hand_labels(hand_labels, hand_mask, mano, is_left=False):
    """Wrapper for process_single_hand_labels from visualize_core."""
    func = _get_process_single_hand_labels()
    return func(hand_labels, hand_mask, mano, is_left)


def convert_inspire_state_to_human(state_24, state_mask):
    """
    Convert 24-dim inspire state to 102-dim human state.

    Args:
        state_24: (24,) numpy array in inspire format
            [left_trans(3), left_euler(3), left_joints(6), right_trans(3), right_euler(3), right_joints(6)]
        state_mask: [use_left, use_right] boolean list

    Returns:
        human_state_102: (102,) numpy array in human format
    """
    state_tensor = torch.from_numpy(state_24).float()
    state_mask_tensor = torch.tensor(state_mask, dtype=torch.bool)

    state_dim = 24
    unified_state_dim = StateFeature.ALL_FEATURES[1]  # 132

    # Pad state
    unified_state, unified_state_mask = pad_state_robot(
        state=state_tensor,
        state_mask=state_mask_tensor,
        state_dim=state_dim,
        unified_state_dim=unified_state_dim,
    )

    unified_action_mask = torch.zeros(16, 192, dtype=torch.bool)
    human_state, human_state_mask, _, _ = transfer_inspire_to_human(
        unified_state, unified_state_mask,
        None, unified_action_mask
    )

    # Add passive joints
    human_state = add_passive_joints_to_human_action(human_state.unsqueeze(0)).squeeze(0)

    # Extract 102-dim
    human_state_102 = human_state[:102].numpy()

    return human_state_102

def process_raw_robot_state(state_24, state_mask):
    """Convert base-frame G1/Inspire state to camera-frame Inspire state."""
    use_left, use_right = state_mask

    if not hasattr(process_raw_robot_state, '_T_base_camera'):
        process_raw_robot_state._T_base_camera = get_camera_pose_in_base_frame()

    T_base_camera = process_raw_robot_state._T_base_camera

    left_trans_base = state_24[0:3]
    left_euler_base = state_24[3:6]
    left_joints_norm = state_24[6:12]

    right_trans_base = state_24[12:15]
    right_euler_base = state_24[15:18]
    right_joints_norm = state_24[18:24]

    if use_left:
        R_left_base = R.from_euler('xyz', left_euler_base).as_matrix()
        left_wrist_T_base = np.eye(4, dtype=np.float32)
        left_wrist_T_base[:3, :3] = R_left_base
        left_wrist_T_base[:3, 3] = left_trans_base

        left_wrist_T_cam = transform_pose_to_camera_frame(left_wrist_T_base, T_base_camera)
        left_wrist_T_cam = correct_gt_wrist_orientation(left_wrist_T_cam, hand='left')
        left_trans_cam, left_euler_cam = se3_to_trans_euler(left_wrist_T_cam)
        left_joints = denormalize_inspire_qpos(left_joints_norm)
    else:
        left_trans_cam = np.zeros(3, dtype=np.float32)
        left_euler_cam = np.zeros(3, dtype=np.float32)
        left_joints = np.zeros(6, dtype=np.float32)

    if use_right:
        R_right_base = R.from_euler('xyz', right_euler_base).as_matrix()
        right_wrist_T_base = np.eye(4, dtype=np.float32)
        right_wrist_T_base[:3, :3] = R_right_base
        right_wrist_T_base[:3, 3] = right_trans_base

        right_wrist_T_cam = transform_pose_to_camera_frame(right_wrist_T_base, T_base_camera)
        right_wrist_T_cam = correct_gt_wrist_orientation(right_wrist_T_cam, hand='right')
        right_trans_cam, right_euler_cam = se3_to_trans_euler(right_wrist_T_cam)
        right_joints = denormalize_inspire_qpos(right_joints_norm)
    else:
        right_trans_cam = np.zeros(3, dtype=np.float32)
        right_euler_cam = np.zeros(3, dtype=np.float32)
        right_joints = np.zeros(6, dtype=np.float32)

    state_24_processed = np.concatenate([
        left_trans_cam, left_euler_cam, left_joints,      # left hand: 12
        right_trans_cam, right_euler_cam, right_joints    # right hand: 12
    ], dtype=np.float32)

    return state_24_processed

def process_inference_request(data, vla_service, configs):
    """
    Process a single inference request.

    Args:
        data: dict containing image, instruction, state, state_mask, action_mask
        vla_service: VLAInferenceService instance
        configs: model configuration

    Returns:
        result: dict with inference results
    """
    try:
        # Parse request data
        image_b64 = data['image']
        instruction = data['instruction']
        state_raw = np.array(data['state'], dtype=np.float32)  # (24,) raw robot state in base frame
        state_mask = np.array(data['state_mask'], dtype=bool)  # (2,)
        action_mask = np.array(data['action_mask'], dtype=bool)  # (T, 2)

        # Process raw robot state: convert from base frame to camera frame, denormalize joints
        state_24 = process_raw_robot_state(state_raw, state_mask.tolist())

        # Decode image
        img_str = image_b64.split(',')[1] if ',' in image_b64 else image_b64
        img_data = base64.b64decode(img_str)
        img = Image.open(BytesIO(img_data))
        image = np.array(img)

        # Note: PIL/Image.open returns RGB format for JPEG images.
        # If the input image is already RGB (common for JPEG), no conversion needed.
        # If it's BGR (e.g., from some camera APIs), you would need: image = image[..., ::-1]
        # Currently assuming input is RGB format, so no conversion.

        # Resize image
        image_resized = resize_short_side_to_target(Image.fromarray(image), target=224)
        image_np = np.array(image_resized)

        w, h = image_np.shape[1], image_np.shape[0]

        # Compute intrinsics (assume standard camera)
        # Use a default focal length based on common D435 parameters
        fx, fy = 931.20577836, 937.832063295
        cx, cy = 640.0, 360.0

        # Scale intrinsics for resized image
        scale = w / 1280.0
        fx_scaled = fx * scale
        fy_scaled = fy * scale
        cx_scaled = cx * (w / 1280.0)
        cy_scaled = cy * (h / 720.0)

        intrinsics = np.array([
            [fx_scaled, 0, cx_scaled],
            [0, fy_scaled, cy_scaled],
            [0, 0, 1]
        ], dtype=np.float32)

        # Compute FOV
        fov_x = 2 * np.arctan(w / (2 * fx_scaled))
        fov_y = 2 * np.arctan(h / (2 * fy_scaled))
        fov = np.array([fov_x, fov_y], dtype=np.float32)


        # Convert inspire state to human state
        human_state = convert_inspire_state_to_human(state_24, state_mask.tolist())

        # Determine which hands are used
        use_left = state_mask[0]
        use_right = state_mask[1]

        # Prepare state for VLA
        # Concatenate left and right hand states
        if use_left and use_right:
            state = human_state
        elif use_left:
            state = np.concatenate([human_state[:51], np.zeros(51)])
        else:
            state = np.concatenate([np.zeros(51), human_state[51:102]])

        beta_left = np.zeros(10)
        beta_right = np.zeros(10)
        state = np.concatenate([state[:51], beta_left, state[51:102], beta_right], axis=0)

        # Run VLA inference
        print(f"Running VLA inference for instruction: {instruction}")
        predict_result = vla_service.predict(
            image=image_np,
            instruction=instruction,
            state=state,
            state_mask=state_mask,
            action_mask=action_mask,
            fov=fov,
            num_ddim_steps=10,
            cfg_scale=5.0,
            sample_times=1,  # Return single sample
        )

        # Unpack inference result
        unnorm_action = predict_result['unnorm_action']  # (1, T, 102) or (T, 102)
        inspire_hand_left = predict_result['inspire_hand_left']  # (T, 6)
        inspire_hand_right = predict_result['inspire_hand_right']  # (T, 6)

        # unnorm_action shape: (1, T, 102) or (T, 102)
        if unnorm_action.ndim == 3:
            unnorm_action = unnorm_action[0]  # Take first sample

        # unnorm_action is relative action in human format
        # Shape: (T, 102) = (T, 51 left + 51 right)
        # Format: [left_trans(3), left_global_orient(3), left_hand_pose(45),
        #          right_trans(3), right_global_orient(3), right_hand_pose(45)]

        # Get current state in human format
        current_state_left = human_state[:51] if use_left else None
        current_state_right = human_state[51:102] if use_right else None

        # Get T from action_mask
        T = unnorm_action.shape[0]

        traj_left = None
        traj_right = None

        # Reconstruct trajectories using recon_traj
        # This converts relative actions to absolute trajectories
        if use_left:
            traj_left = recon_traj(
                state=current_state_left,
                rel_action=unnorm_action[:, 0:51],
                abs_joint=True,
                rel_mode='step'
            )
            traj_left = traj_left[1:]  # Shape: [T, 51]

        if use_right:
            traj_right = recon_traj(
                state=current_state_right,
                rel_action=unnorm_action[:, 51:102],
                abs_joint=True,
                rel_mode='step'
            )
            traj_right = traj_right[1:]  # Shape: [T, 51]

        # Apply the optional camera-frame position offset after trajectory reconstruction.
        if use_left:
            traj_left[:, 0:3] = _apply_predicted_position_offset(traj_left[:, 0:3], CAMERA_FRAME_ACTION_OFFSET)
        if use_right:
            traj_right[:, 0:3] = _apply_predicted_position_offset(traj_right[:, 0:3], CAMERA_FRAME_ACTION_OFFSET)

        # Get camera pose in base frame
        T_base_camera = get_camera_pose_in_base_frame()

        # Wh0 robot action: [left_pose(16), left_inspire(6), right_pose(16), right_inspire(6)].
        wh0_robot_action = np.zeros((T, 44), dtype=np.float32)

        translation_scale = 1

        for t in range(T):
            if use_left:
                pose_cam = np.eye(4)
                pose_cam[:3, :3] = R.from_euler('xyz', traj_left[t, 3:6]).as_matrix()
                pose_cam[:3, 3] = traj_left[t, 0:3] * translation_scale
                pose_waist = transform_camera_pose_to_base_frame(
                    correct_gt_wrist_orientation_inverse(pose_cam, hand='left'), T_base_camera)
                pose_waist = _apply_wrist_correction(pose_waist)
                wh0_robot_action[t, 0:16] = pose_waist.flatten()
                wh0_robot_action[t, 16:22] = inspire_hand_left[t]
            if use_right:
                pose_cam = np.eye(4)
                pose_cam[:3, :3] = R.from_euler('xyz', traj_right[t, 3:6]).as_matrix()
                pose_cam[:3, 3] = traj_right[t, 0:3] * translation_scale
                pose_waist = transform_camera_pose_to_base_frame(
                    correct_gt_wrist_orientation_inverse(pose_cam, hand='right'), T_base_camera)
                pose_waist = _apply_wrist_correction(pose_waist)
                wh0_robot_action[t, 22:38] = pose_waist.flatten()
                wh0_robot_action[t, 38:44] = inspire_hand_right[t]

        viz_traj_left = None
        viz_traj_right = None

        if use_left:
            viz_traj_left = recon_traj(
                state=current_state_left,
                rel_action=unnorm_action[:, 0:51],
                abs_joint=True,
                rel_mode='step'
            )

        if use_right:
            viz_traj_right = recon_traj(
                state=current_state_right,
                rel_action=unnorm_action[:, 51:102],
                abs_joint=True,
                rel_mode='step'
            )

        viz_action = unnorm_action.copy()

        if _visualization_enabled:
            try:
                viz_thread = threading.Thread(
                    target=_render_and_save_visualization,
                    args=(image_np, viz_action, beta_left, beta_right,
                          fov, intrinsics, use_left, use_right,
                          viz_traj_left, viz_traj_right)
                )
                with _visualization_lock:
                    _visualization_threads.append(viz_thread)
                viz_thread.start()
            except Exception as viz_err:
                print(f"[Viz] Visualization error (non-fatal): {viz_err}")

        return {
            'success': True,
            'data': {
                'wh0_robot_action': wh0_robot_action.tolist(),
            }
        }

    except Exception as e:
        import traceback
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


async def handle_client(websocket, vla_service, configs):
    """Handle WebSocket client connection."""
    try:
        async for message in websocket:
            try:
                # Parse request
                if isinstance(message, str):
                    request = json.loads(message)
                else:
                    request = message

                if request.get('type') == 'inference_request':
                    data = request.get('data', {})
                    result = process_inference_request(data, vla_service, configs)

                    # Send response
                    response = json.dumps(result)
                    await websocket.send(response)

                    if _visualization_enabled:
                        _wait_for_visualization_threads(timeout=5.0)

                elif request.get('type') == 'ping':
                    # Ping-pong for connection keep-alive
                    await websocket.send(json.dumps({'type': 'pong'}))

                elif request.get('type') == 'shutdown':
                    # Shutdown signal
                    await websocket.send(json.dumps({'type': 'shutdown_ack'}))
                    break
                else:
                    await websocket.send(json.dumps({
                        'success': False,
                        'error': f'Unknown request type: {request.get("type")}'
                    }))

            except json.JSONDecodeError as e:
                await websocket.send(json.dumps({
                    'success': False,
                    'error': f'Invalid JSON: {e}'
                }))
            except Exception as e:
                await websocket.send(json.dumps({
                    'success': False,
                    'error': str(e)
                }))

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    except Exception as e:
        print(f"Error handling client: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if _visualization_enabled:
            _wait_for_visualization_threads(timeout=10.0)
            saved = _close_video_writer()
            if saved:
                print(f"[Viz] Video saved: {saved}")
            else:
                print("[Viz] No video to save or save failed")


async def start_server(vla_service, configs, host, port):
    """Start WebSocket server."""
    # Create wrapper to handle client and check shutdown
    async def handle_client_with_shutdown(websocket):
        await handle_client(websocket, vla_service, configs)

    # Create server
    server = await websockets.serve(handle_client_with_shutdown, host, port)
    print(f"Server started on ws://{host}:{port}", flush=True)

    # Wait for shutdown signal using threading.Event (poll periodically)
    while not _shutdown_event.is_set():
        await asyncio.sleep(0.1)

    print("[Server] Server stopping...", flush=True)

    # Close server gracefully
    server.close()
    await server.wait_closed()


_shutdown_requested = False
_force_exit_requested = False
_shutdown_event = None


def _handle_shutdown(signum, frame):
    """Signal handler that sets shutdown flag."""
    print(f"\n[Server] Received signal {signum}, initiating shutdown...", flush=True)
    global _shutdown_requested, _force_exit_requested
    _shutdown_requested = True
    _force_exit_requested = True

    if _shutdown_event is not None:
        _shutdown_event.set()


def default_video_path() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(WH0_ROOT / "validation_outputs" / "deployment" / f"model_server_visualization_{timestamp}.mp4")


def main():
    parser = argparse.ArgumentParser(description="WebSocket VLA Inference Server")
    parser.add_argument('--config_path', type=str, default="config.json", help='Path to model configuration JSON file')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint (overrides config)')
    parser.add_argument('--statistics_path', type=str, default=None, help='Path to normalization statistics JSON (overrides config)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=8765, help='Server port')
    parser.add_argument('--video_path', type=str, default=None, help='Path to save visualization video')
    parser.add_argument('--mano_path', type=str, default=str(WH0_ROOT / "weights" / "mano"), help='MANO model directory')
    parser.add_argument('--no_viz', action='store_true', help='Disable visualization')

    args = parser.parse_args()

    configs = load_config(args.config_path)

    if args.model_path is not None:
        configs['model_load_path'] = args.model_path
    if args.statistics_path is not None:
        configs['statistics_path'] = args.statistics_path

    global _video_save_path, _visualization_enabled
    _video_save_path = args.video_path or default_video_path()
    Path(_video_save_path).parent.mkdir(parents=True, exist_ok=True)
    _visualization_enabled = not args.no_viz

    global _shutdown_event
    _shutdown_event = threading.Event()

    if not hasattr(_handle_shutdown, '_registered'):
        signal.signal(signal.SIGINT, _handle_shutdown)
        signal.signal(signal.SIGTERM, _handle_shutdown)
        _handle_shutdown._registered = True

    if _visualization_enabled:
        print("Initializing visualization components...")
        _init_visualization(mano_path=args.mano_path, fps=8)
        atexit.register(_close_video_writer)

    print("Initializing VLA service...")
    vla_service = VLAInferenceService(configs)
    print("VLA service ready")

    try:
        asyncio.run(start_server(vla_service, configs, args.host, args.port))
    except KeyboardInterrupt:
        print("Server shutdown requested")
    finally:
        print("Shutting down VLA service...")
        vla_service.shutdown()
        if _visualization_enabled:
            _wait_for_visualization_threads(timeout=10.0)
            saved = _close_video_writer()
            if saved:
                print(f"[Viz] Video saved to {saved}")
            else:
                print("[Viz] No video to save")
        print("Server stopped")

        # Force exit if signal was received (bypass any hanging cleanup)
        if _force_exit_requested:
            import os
            print("Force exiting...", flush=True)
            os._exit(0)


# VLA Inference Service (from original inference_server.py)

def _vla_inference_worker(configs_dict, task_queue, result_queue):
    """Persistent worker for VLA model inference."""
    import traceback as tb
    from vitra.models import load_model
    from vitra.utils.data_utils import load_normalizer
    from vitra.datasets.human_dataset import pad_state_human, pad_action
    from vitra.datasets.dataset_utils import ActionFeature, StateFeature
    from vitra.datasets.robot_dataset import mano_hand_pose_to_inspire_qpos

    model = None
    normalizer = None

    try:
        print("[VLA Process] Loading VLA model...")
        model = load_model(configs_dict).cuda()
        model.eval()
        normalizer = load_normalizer(configs_dict)
        print("[VLA Process] VLA model ready.")

        # Signal ready
        print("[VLA Process] Sending ready signal...")
        result_queue.put({'type': 'ready'})
        print("[VLA Process] Ready signal sent.")

        # Process tasks in loop
        print("[VLA Process] Entering task processing loop...")
        while True:
            try:
                # Use timeout to allow checking for shutdown
                task = task_queue.get(timeout=1)
            except:
                # Timeout - continue loop to check for shutdown or more tasks
                continue

            if task['type'] == 'shutdown':
                print("[VLA Process] Received shutdown signal")
                break

            elif task['type'] == 'predict':
                try:
                    image = task['image']
                    instruction = task['instruction']
                    state = task['state']
                    state_mask = task['state_mask']
                    action_mask = task['action_mask']
                    fov = task['fov']
                    num_ddim_steps = task.get('num_ddim_steps', 10)
                    cfg_scale = task.get('cfg_scale', 5.0)
                    sample_times = task.get('sample_times', 1)

                    # Normalize state
                    norm_state = normalizer.normalize_state(state.copy())

                    # Pad state and action
                    unified_action_dim = ActionFeature.ALL_FEATURES[1]  # 192
                    unified_state_dim = StateFeature.ALL_FEATURES[1]    # 212

                    unified_state, unified_state_mask = pad_state_human(
                        state=norm_state,
                        state_mask=state_mask,
                        action_dim=normalizer.action_mean.shape[0],
                        state_dim=normalizer.state_mean.shape[0],
                        unified_state_dim=unified_state_dim,
                    )

                    _, unified_action_mask = pad_action(
                        actions=None,
                        action_mask=action_mask.copy(),
                        action_dim=normalizer.action_mean.shape[0],
                        unified_action_dim=unified_action_dim
                    )

                    # Convert to torch and move to GPU
                    fov = torch.from_numpy(fov).unsqueeze(0)
                    unified_state = unified_state.unsqueeze(0)
                    unified_state_mask = unified_state_mask.unsqueeze(0)
                    unified_action_mask = unified_action_mask.unsqueeze(0)

                    # Run inference
                    norm_action = model.predict_action(
                        image=image,
                        instruction=instruction,
                        current_state=unified_state,
                        current_state_mask=unified_state_mask,
                        action_mask_torch=unified_action_mask,
                        num_ddim_steps=num_ddim_steps,
                        cfg_scale=cfg_scale,
                        fov=fov,
                        sample_times=sample_times,
                    )

                    # Extract and denormalize action (use original normalizer for action)
                    norm_action = norm_action[:, :, :102]
                    unnorm_action = normalizer.unnormalize_action(norm_action)

                    # Convert to numpy
                    if isinstance(unnorm_action, torch.Tensor):
                        unnorm_action_np = unnorm_action.cpu().numpy()
                    else:
                        unnorm_action_np = np.array(unnorm_action)

                    # Use mano_hand_pose_to_inspire_qpos to convert hand pose MANO parameters to Inspire hand joints
                    # unnorm_action_np shape: (sample_times, T, 102)
                    # Take first sample: (T, 102)
                    action_for_retarget = unnorm_action_np[0]
                    T_action = action_for_retarget.shape[0]

                    inspire_hand_left = np.zeros((T_action, 6), dtype=np.float32)
                    inspire_hand_right = np.zeros((T_action, 6), dtype=np.float32)

                    for t in range(T_action):
                        # Left hand: indices 6:51 (6-DoF wrist + 45 MANO pose)
                        left_hand_pose_45 = action_for_retarget[t, 6:51]
                        inspire_hand_left[t] = mano_hand_pose_to_inspire_qpos(left_hand_pose_45)

                        # Right hand: indices 57:102
                        right_hand_pose_45 = action_for_retarget[t, 57:102]
                        inspire_hand_right[t] = mano_hand_pose_to_inspire_qpos(right_hand_pose_45)

                    result_queue.put({
                        'type': 'result',
                        'success': True,
                        'data': unnorm_action_np,
                        'inspire_hand_left': inspire_hand_left,
                        'inspire_hand_right': inspire_hand_right,
                    })

                except Exception as e:
                    result_queue.put({
                        'type': 'result',
                        'success': False,
                        'error': str(e),
                        'traceback': tb.format_exc()
                    })

    except Exception as e:
        print(f"[VLA Process] Fatal error: {e}")
        print(f"[VLA Process] Traceback: {tb.format_exc()}")
        try:
            result_queue.put({
                'type': 'error',
                'error': str(e),
                'traceback': tb.format_exc()
            })
        except:
            pass

    finally:
        if model is not None:
            del model
        if normalizer is not None:
            del normalizer
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("[VLA Process] Cleaned up and exiting")


class VLAInferenceService:
    """Service wrapper for persistent VLA inference process"""

    def __init__(self, configs):
        self.ctx = mp.get_context('spawn')
        self.task_queue = self.ctx.Queue()
        self.result_queue = self.ctx.Queue()

        # Start persistent process with daemon=False (so it can be properly terminated)
        self.process = self.ctx.Process(
            target=_vla_inference_worker,
            args=(configs, self.task_queue, self.result_queue),
            daemon=False  # Allow proper cleanup on shutdown
        )
        self.process.start()

        # Wait for ready signal with timeout
        import time
        start_time = time.time()
        timeout = 120  # 2 minutes for model loading
        while time.time() - start_time < timeout:
            try:
                ready_msg = self.result_queue.get(timeout=1)
                if ready_msg['type'] == 'ready':
                    print("VLA inference service initialized")
                    return
                elif ready_msg['type'] == 'error':
                    raise RuntimeError(f"Failed to initialize VLA model: {ready_msg['error']}")
            except:
                # Check if process died unexpectedly
                if not self.process.is_alive():
                    raise RuntimeError("VLA process died during initialization")
                continue

        raise RuntimeError("VLA initialization timed out")

    def predict(self, image, instruction, state, state_mask, action_mask,
                fov, num_ddim_steps=10, cfg_scale=5.0, sample_times=1):
        """Request action prediction with state normalization and padding"""

        self.task_queue.put({
            'type': 'predict',
            'image': image,
            'instruction': instruction,
            'state': state,
            'state_mask': state_mask,
            'action_mask': action_mask,
            'fov': fov,
            'num_ddim_steps': num_ddim_steps,
            'cfg_scale': cfg_scale,
            'sample_times': sample_times,
        })

        result = self.result_queue.get()
        if result['type'] == 'result' and result['success']:
            return {
                'unnorm_action': result['data'],
                'inspire_hand_left': result.get('inspire_hand_left'),
                'inspire_hand_right': result.get('inspire_hand_right'),
            }
        else:
            raise RuntimeError(f"VLA inference failed: {result.get('error', 'Unknown error')}")

    def shutdown(self):
        """Shutdown the persistent process"""
        self.task_queue.put({'type': 'shutdown'})
        # Give process a moment to respond to shutdown signal
        self.process.join(timeout=3)
        if self.process.is_alive():
            print("[VLA Service] Process did not exit cleanly, terminating...")
            self.process.terminate()
            self.process.join(timeout=5)
        if self.process.is_alive():
            print("[VLA Service] Process still alive, killing...")
            self.process.kill()
            self.process.join()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()

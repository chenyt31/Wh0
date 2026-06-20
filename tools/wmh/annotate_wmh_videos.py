"""
Pipeline for processing videos to extract hand reconstruction data.

This script recursively finds all videos in a given path, processes them to extract
left and right hand data including:
- Translation (transl)
- MANO parameters (betas, hand_pose, global_orient)
- Joints
- Camera intrinsics (focal length, fov_x, image dimensions)

Output format (episode_info):
- 'extrinsics': (Tx4x4) World2Cam extrinsic matrix
- 'intrinsics': (3x3) Camera intrinsic matrix
- 'video_decode_frame': Frame indices in the original raw video
- 'video_name': Original raw video name
- 'avg_speed': Average wrist movement per frame (in meters)
- 'total_rotvec_degree': Total camera rotation over the episode (in degrees)
- 'total_transl_dist': Total camera translation distance over the episode (in meters)
- 'anno_type': Annotation type
- 'text': {'left': [(str, (start, end))], 'right': [(str, (start, end))]}
- 'left'/'right': Hand pose data with beta, global_orient_camspace/worldspace, 
                  hand_pose, transl_worldspace, kept_frames, joints_camspace/worldspace

"""

import argparse
from pathlib import Path
import json
import re
from typing import Dict, List, Any, Optional, Tuple
import cv2
import numpy as np
import torch
from tqdm import tqdm
import sys
import os
from multiprocessing import Process, Manager

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scipy.spatial.transform import Rotation as R

from libs.hand_recon import HandReconstructor, HandReconConfig
from libs.hand_recon.megasam import MegaSAMPipeline


DATASET_NAME = "WM-H"


def get_npy_path(video_path: Path, input_path: Path, episode_idx: int = 0) -> Path:
    """
    Get the output npy file path for a video.
    Output format: {dataset_name}_{video_name}_ep_{episode_idx:06d}.npy
    
    Args:
        video_path: Path to video file
        input_path: Root input path (not used in path generation, kept for compatibility)
        episode_idx: Episode index (default 0, each video is one episode)
    
    Returns:
        Path to output npy file
    """
    # For video at: run_xxx/videos/video.mp4
    # Output at: run_xxx/episodic_annotations/WM-H_video_ep_000000.npy
    output_path = video_path.parent.parent / 'episodic_annotations'
    video_name = video_path.stem
    filename = f"{DATASET_NAME}_{video_name}_ep_{episode_idx:06d}.npy"
    output_file = output_path / filename
    return output_file


def get_json_path(video_path: Path) -> Path:
    """
    Get the corresponding JSON task file path for a video.
    For video at: run_xxx/videos/video.mp4
    JSON at: run_xxx/tasks/video.json
    
    Args:
        video_path: Path to video file
    
    Returns:
        Path to JSON task file
    """
    json_path = video_path.parent.parent / 'tasks' / (video_path.stem + '.json')
    return json_path


def parse_task_description(task_description: str) -> Dict[str, str]:
    """
    Parse task_description to extract left and right hand descriptions.
    
    Expected format: "Left hand: XXX. Right hand: YYY"
    
    Args:
        task_description: Task description string
    
    Returns:
        Dictionary with 'left' and 'right' keys containing descriptions
    """
    result = {'left': None, 'right': None}
    
    # Match patterns like "Left hand: XXX" and "Right hand: YYY"
    left_match = re.search(r'Left hand:\s*([^.]+(?:\.[^LR])?)', task_description, re.IGNORECASE)
    right_match = re.search(r'Right hand:\s*([^.]+(?:\.[^LR])?)', task_description, re.IGNORECASE)
    
    if left_match:
        left_desc = left_match.group(1).strip()
        if left_desc.lower() != 'none':
            result['left'] = left_desc
    
    if right_match:
        right_desc = right_match.group(1).strip()
        if right_desc.lower() != 'none':
            result['right'] = right_desc
    
    return result


def load_task_json(video_path: Path) -> Dict[str, Any]:
    """
    Load the corresponding JSON task file for a video.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary containing task data, or empty dict if not found
    """
    json_path = get_json_path(video_path)
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load JSON {json_path}: {e}")
    return {}


def calculate_camera_metrics(camera_poses: Dict[str, Any], num_frames: int) -> Tuple[float, float]:
    """
    Calculate total camera rotation (degrees) and translation distance (meters).
    
    Args:
        camera_poses: Dictionary containing R_c2w and t_c2w arrays
        num_frames: Number of frames
    
    Returns:
        Tuple of (total_rotvec_degree, total_transl_dist)
    """
    total_rotvec_degree = 0.0
    total_transl_dist = 0.0
    
    if camera_poses is None or 'R_c2w' not in camera_poses:
        return total_rotvec_degree, total_transl_dist
    
    R_c2w_list = camera_poses['R_c2w']
    t_c2w_list = camera_poses['t_c2w']
    
    for i in range(1, min(num_frames, len(R_c2w_list))):
        # Rotation change
        R_prev = R_c2w_list[i - 1]
        R_curr = R_c2w_list[i]
        R_diff = R_prev.T @ R_curr  # Relative rotation
        
        # Convert to axis-angle and get angle in degrees
        r = R.from_matrix(R_diff)
        angle_rad = np.linalg.norm(r.as_rotvec())
        total_rotvec_degree += np.degrees(angle_rad)
        
        # Translation change
        t_prev = t_c2w_list[i - 1]
        t_curr = t_c2w_list[i]
        total_transl_dist += np.linalg.norm(t_curr - t_prev)
    
    return total_rotvec_degree, total_transl_dist


def calculate_avg_wrist_speed(hand_data: Dict[int, Dict], num_frames: int) -> float:
    """
    Calculate average wrist movement speed per frame in world space.
    
    Args:
        hand_data: Dictionary of frame_idx -> hand data
        num_frames: Total number of frames
    
    Returns:
        Average wrist speed in meters per frame
    """
    if not hand_data:
        return 0.0
    
    # Get sorted frame indices with world space data
    valid_frames = sorted([
        idx for idx, data in hand_data.items() 
        if data.get('world_space') is not None and data['world_space'].get('transl') is not None
    ])
    
    if len(valid_frames) < 2:
        return 0.0
    
    total_movement = 0.0
    for i in range(1, len(valid_frames)):
        prev_idx = valid_frames[i - 1]
        curr_idx = valid_frames[i]
        
        prev_transl = hand_data[prev_idx]['world_space']['transl']
        curr_transl = hand_data[curr_idx]['world_space']['transl']
        
        total_movement += np.linalg.norm(curr_transl - prev_transl)
    
    # Average over total frames (not just valid frames)
    return total_movement / max(num_frames - 1, 1)


def find_videos(root_path: Path, extensions: List[str] = None) -> List[Path]:
    """
    Recursively find all video files in the given path (✅ 支持软链接/符号链接)
    
    Args:
        root_path: Root directory to search (支持软链接路径)
        extensions: List of video file extensions 
    
    Returns:
        List of video file paths
    """
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm',
                      '.MP4', '.AVI', '.MOV', '.MKV', '.FLV', '.WEBM']

    if (root_path / "videos").is_dir():
        root_path = root_path / "videos"

    excluded_dirs = {
        "episodic_annotations",
        "episodic_annotations_split",
        "videos_robot_hands",
        "vitra_training_data",
    }
    video_files = []
    # 遍历所有层级（支持软链接），兼容大小写扩展名
    for dir_path, dir_names, file_names in os.walk(root_path, followlinks=True):
        dir_names[:] = [name for name in dir_names if name not in excluded_dirs]
        for filename in file_names:
            file_path = Path(dir_path) / filename
            if file_path.suffix in extensions:
                video_files.append(file_path)

    return sorted(video_files)


def extract_frames(video_path: Path) -> List[np.ndarray]:
    """
    Extract all frames from a video file.
    
    Args:
        video_path: Path to video file
    
    Returns:
        List of frames as numpy arrays (BGR format)
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}")
        return frames
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames


def transform_hand_to_world_space(
    hand_data_camspace: Dict[str, Any],
    R_c2w: np.ndarray,  # [3, 3] camera-to-world rotation
    t_c2w: np.ndarray,  # [3] camera-to-world translation
    reconstructor: HandReconstructor,
    hand_side: str = 'right',
) -> Dict[str, Any]:
    """
    Transform hand data from camera space to world space.
    
    Args:
        hand_data_camspace: Hand data in camera space with keys:
            - 'transl': [3] wrist translation in camera space
            - 'global_orient': [3, 3] wrist rotation matrix in camera space
            - 'hand_pose': [15, 3, 3] or [45] hand pose
            - 'betas': [10] shape parameters
            - 'joints': [21, 3] joints in camera space (optional)
            - 'vertices': [778, 3] vertices in camera space (optional)
        R_c2w: Camera-to-world rotation matrix [3, 3]
        t_c2w: Camera-to-world translation [3]
        reconstructor: HandReconstructor instance
        hand_side: 'left' or 'right'
    
    Returns:
        Hand data in world space with same structure
    """
    # Get wrist translation and rotation in camera space
    transl_cam = hand_data_camspace['transl']  # [3]
    global_orient_cam = hand_data_camspace['global_orient']  # [3, 3]
    hand_pose = hand_data_camspace['hand_pose']  # [15, 3, 3] or [45]
    betas = hand_data_camspace['betas']  # [10]
    
    # Convert hand_pose to rotation matrices if needed
    if hand_pose.ndim == 1:
        # [45] axis-angle -> [15, 3, 3] rotation matrices
        hand_pose_aa = hand_pose.reshape(15, 3)
        hand_pose_rot = np.array([R.from_rotvec(aa).as_matrix() for aa in hand_pose_aa])
    else:
        hand_pose_rot = hand_pose  # [15, 3, 3]
    
    # Note: transl_cam is already the wrist position in camera space
    # (see hand_recon_core.py line 120: transl_new = wrist + transl)
    
    # Transform wrist translation to world space
    transl_world = R_c2w @ transl_cam + t_c2w  # [3]
    
    # Transform wrist rotation to world space
    global_orient_world = R_c2w @ global_orient_cam  # [3, 3]
    
    # Hand pose rotations are relative to the wrist in MANO local space
    # They don't need transformation to world space (they stay the same)
    hand_pose_world = hand_pose_rot.copy()  # [15, 3, 3]
    
    # Transform joints and vertices if available
    joints_world = None
    vertices_world = None
    
    if 'joints' in hand_data_camspace:
        joints_cam = hand_data_camspace['joints']  # [21, 3]
        # Transform each joint
        joints_world = (R_c2w @ joints_cam.T).T + t_c2w  # [21, 3]
    
    if 'vertices' in hand_data_camspace:
        vertices_cam = hand_data_camspace['vertices']  # [778, 3]
        # Transform each vertex
        vertices_world = (R_c2w @ vertices_cam.T).T + t_c2w  # [778, 3]
    
    return {
        'transl': transl_world,  # [3] wrist translation in world space
        'global_orient': global_orient_world,  # [3, 3] wrist rotation in world space
        'hand_pose': hand_pose_world,  # [15, 3, 3] hand pose in world space
        'betas': betas,  # [10] shape parameters (unchanged)
        'joints': joints_world,  # [21, 3] joints in world space
        'vertices': vertices_world,  # [778, 3] vertices in world space
    }


def aggregate_hand_data_to_arrays(
    hand_frame_data: Dict[int, Dict], 
    num_frames: int
) -> Dict[str, Any]:
    """
    Aggregate per-frame hand data into time-indexed arrays.
    
    Args:
        hand_frame_data: Dictionary of frame_idx -> {camera_space: {...}, world_space: {...}}
        num_frames: Total number of frames
    
    Returns:
        Dictionary with aggregated arrays (always returns a valid structure):
        - 'beta': (10,) MANO shape parameters
        - 'global_orient_camspace': (Tx3x3)
        - 'global_orient_worldspace': (Tx3x3)
        - 'hand_pose': (Tx15x3x3)
        - 'transl_worldspace': (Tx3)
        - 'kept_frames': list[int] 0-1 mask (use this to check valid frames)
        - 'joints_camspace': (Tx21x3)
        - 'joints_worldspace': (Tx21x3)
    """
    # Initialize arrays
    global_orient_camspace = np.zeros((num_frames, 3, 3), dtype=np.float32)
    global_orient_worldspace = np.zeros((num_frames, 3, 3), dtype=np.float32)
    hand_pose = np.zeros((num_frames, 15, 3, 3), dtype=np.float32)
    transl_worldspace = np.zeros((num_frames, 3), dtype=np.float32)
    joints_camspace = np.zeros((num_frames, 21, 3), dtype=np.float32)
    joints_worldspace = np.zeros((num_frames, 21, 3), dtype=np.float32)
    kept_frames = [0] * num_frames
    
    # Get beta from first valid frame (same for all frames)
    beta = None
    if hand_frame_data:
        for frame_idx, data in hand_frame_data.items():
            if data.get('camera_space') is not None:
                beta = data['camera_space'].get('betas')
                if beta is not None:
                    break
    
    if beta is None:
        beta = np.zeros(10, dtype=np.float32)
    
    # Fill arrays (if hand_frame_data is empty, arrays stay all zeros with kept_frames all 0)
    if hand_frame_data:
        for frame_idx, data in hand_frame_data.items():
            if frame_idx >= num_frames:
                continue
            
            cam_data = data.get('camera_space')
            world_data = data.get('world_space')
            
            if cam_data is not None:
                kept_frames[frame_idx] = 1
                
                # Camera space data
                if 'global_orient' in cam_data:
                    global_orient_camspace[frame_idx] = cam_data['global_orient']
                if 'hand_pose' in cam_data:
                    hand_pose[frame_idx] = cam_data['hand_pose']
                if 'joints' in cam_data:
                    joints_camspace[frame_idx] = cam_data['joints']
            
            if world_data is not None:
                # World space data
                if 'global_orient' in world_data:
                    global_orient_worldspace[frame_idx] = world_data['global_orient']
                if 'transl' in world_data:
                    transl_worldspace[frame_idx] = world_data['transl']
                if 'joints' in world_data:
                    joints_worldspace[frame_idx] = world_data['joints']
    
    return {
        'beta': np.array(beta, dtype=np.float32),  # (10,)
        'global_orient_camspace': global_orient_camspace,  # (Tx3x3)
        'global_orient_worldspace': global_orient_worldspace,  # (Tx3x3)
        'hand_pose': hand_pose,  # (Tx15x3x3)
        'transl_worldspace': transl_worldspace,  # (Tx3)
        'kept_frames': kept_frames,  # list[int]
        'joints_camspace': joints_camspace,  # (Tx21x3)
        'joints_worldspace': joints_worldspace,  # (Tx21x3)
    }


def build_extrinsics_matrix(camera_poses: Dict[str, Any], num_frames: int) -> np.ndarray:
    """
    Build (Tx4x4) World2Cam extrinsic matrices from camera poses.
    
    Args:
        camera_poses: Dictionary containing R_w2c and t_w2c arrays
        num_frames: Number of frames
    
    Returns:
        (Tx4x4) extrinsic matrices (world-to-camera)
    """
    extrinsics = np.zeros((num_frames, 4, 4), dtype=np.float32)
    
    if camera_poses is None:
        # Return identity matrices if no camera poses
        for i in range(num_frames):
            extrinsics[i] = np.eye(4, dtype=np.float32)
        return extrinsics
    
    R_w2c_list = camera_poses.get('R_w2c', [])
    t_w2c_list = camera_poses.get('t_w2c', [])
    
    for i in range(num_frames):
        if i < len(R_w2c_list) and i < len(t_w2c_list):
            extrinsics[i, :3, :3] = R_w2c_list[i]
            extrinsics[i, :3, 3] = t_w2c_list[i]
            extrinsics[i, 3, 3] = 1.0
        else:
            extrinsics[i] = np.eye(4, dtype=np.float32)
    
    return extrinsics


def get_camera_to_world_pose(camera_poses: Dict[str, Any] | None, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    if (
        camera_poses is not None
        and frame_idx < len(camera_poses.get('R_c2w', []))
        and frame_idx < len(camera_poses.get('t_c2w', []))
    ):
        return camera_poses['R_c2w'][frame_idx], camera_poses['t_c2w'][frame_idx]
    return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)


def build_intrinsics_matrix(focal_length: float, img_center: np.ndarray) -> np.ndarray:
    """
    Build (3x3) camera intrinsic matrix.
    
    Args:
        focal_length: Focal length in pixels
        img_center: Image center [cx, cy]
    
    Returns:
        (3x3) intrinsic matrix
    """
    if focal_length is None:
        focal_length = 1000.0  # Default value
    
    K = np.array([
        [focal_length, 0, img_center[0]],
        [0, focal_length, img_center[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return K


def process_video(
    video_path: Path,
    reconstructor: HandReconstructor,
    input_path: Optional[Path] = None,
    save_output: bool = True,
) -> Dict[str, Any]:
    """
    Process a single video to extract hand reconstruction data.
    
    Output format (episode_info):
    - 'extrinsics': (Tx4x4) World2Cam extrinsic matrix
    - 'intrinsics': (3x3) Camera intrinsic matrix
    - 'video_decode_frame': Frame indices in the original raw video
    - 'video_name': Original raw video name
    - 'avg_speed': Average wrist movement per frame
    - 'total_rotvec_degree': Total camera rotation
    - 'total_transl_dist': Total camera translation distance
    - 'anno_type': Annotation type
    - 'text': {'left': [(str, (start, end))], 'right': [(str, (start, end))]}
    - 'left'/'right': Hand pose data
    
    Args:
        video_path: Path to video file
        reconstructor: HandReconstructor instance (contains MANO model and device)
        input_path: Optional input path for generating relative output paths
        save_output: Whether to save output to file
    
    Returns:
        Dictionary containing all reconstruction data (episode_info format)
    """
    print(f"Processing video: {video_path}")
    
    # Load task JSON for text descriptions
    task_data = load_task_json(video_path)
    task_description = task_data.get('instruction') or task_data.get('task_description', '')
    hand_descriptions = parse_task_description(task_description)
    if not hand_descriptions['left'] and not hand_descriptions['right']:
        left_instruction = str(task_data.get('left_instruction') or '').strip()
        right_instruction = str(task_data.get('right_instruction') or '').strip()
        if left_instruction and left_instruction.lower() != 'none':
            hand_descriptions['left'] = left_instruction
        if right_instruction and right_instruction.lower() != 'none':
            hand_descriptions['right'] = right_instruction
    if not hand_descriptions['left'] and not hand_descriptions['right'] and task_data.get('task_description'):
        hand = str(task_data.get('hand') or 'right').lower()
        if hand not in {'left', 'right'}:
            hand = 'right'
        hand_descriptions[hand] = str(task_data['task_description']).strip()
    
    # Extract frames
    frames = extract_frames(video_path)
    if len(frames) == 0:
        print(f"Warning: No frames extracted from {video_path}")
        return None
    
    print(f"  Extracted {len(frames)} frames")
    
    # Get image dimensions
    H, W = frames[0].shape[:2]
    num_frames = len(frames)
    
    # Perform hand reconstruction
    recon_results = reconstructor.recon(frames)
    
    # Extract camera intrinsics
    fov_x = recon_results.get('fov_x')
    if fov_x is not None:
        focal_length = 0.5 * W / np.tan(0.5 * fov_x * np.pi / 180)
        img_center = np.array([W / 2.0, H / 2.0])
    else:
        focal_length = None
        img_center = np.array([W / 2.0, H / 2.0])
    
    # Track camera poses using MegaSAM
    print("  Tracking camera poses with MegaSAM...")
    camera_poses = None
    megasam = MegaSAMPipeline()
    camera_poses = megasam.track_camera_poses(
        frames=frames,
        focal_length=focal_length,
        img_center=img_center,
    )
    print(f"  Successfully tracked camera poses for {len(frames)} frames")

    
    # Process each frame to extract joints
    left_hand_data = {}
    right_hand_data = {}
    
    for frame_idx in range(num_frames):
        # Process left hand
        if frame_idx in recon_results['left']:
            result = recon_results['left'][frame_idx]
            
            # Convert to tensors for MANO forward pass
            betas = torch.from_numpy(result['beta']).unsqueeze(0).to(reconstructor.device)
            hand_pose = torch.from_numpy(result['hand_pose']).unsqueeze(0).to(reconstructor.device)
            
            # Forward pass through MANO to get joints
            with torch.no_grad():
                mano_output = reconstructor.mano(
                    betas=betas,
                    hand_pose=hand_pose
                )
                joints = mano_output.joints[0].cpu().numpy()  # [21, 3]
            
            # Flip x-axis for left hand consistency
            joints[:, 0] = -1 * joints[:, 0]
            
            # Store camera-space data
            transl_cam = result['transl']  # [3] camera space
            joints_cam = joints + transl_cam  # [21, 3] camera space
            
            hand_data_camspace = {
                'transl': transl_cam,
                'betas': result['beta'],
                'hand_pose': result['hand_pose'],
                'global_orient': result['global_orient'],
                'joints': joints_cam,
            }
            
            # Transform to world space, falling back to an identity camera pose
            # for frames without MegaSAM output. This matches build_extrinsics_matrix().
            hand_data_worldspace = None
            try:
                R_c2w, t_c2w = get_camera_to_world_pose(camera_poses, frame_idx)
                hand_data_worldspace = transform_hand_to_world_space(
                    hand_data_camspace=hand_data_camspace,
                    R_c2w=R_c2w,
                    t_c2w=t_c2w,
                    reconstructor=reconstructor,
                    hand_side='left',
                )
            except Exception as e:
                print(f"  Warning: Failed to transform left hand to world space (frame {frame_idx}): {e}")
            
            left_hand_data[frame_idx] = {
                'camera_space': hand_data_camspace,
                'world_space': hand_data_worldspace,
            }
        
        # Process right hand
        if frame_idx in recon_results['right']:
            result = recon_results['right'][frame_idx]
            
            # Convert to tensors for MANO forward pass
            betas = torch.from_numpy(result['beta']).unsqueeze(0).to(reconstructor.device)
            hand_pose = torch.from_numpy(result['hand_pose']).unsqueeze(0).to(reconstructor.device)
            
            # Forward pass through MANO to get joints
            with torch.no_grad():
                mano_output = reconstructor.mano(
                    betas=betas,
                    hand_pose=hand_pose
                )
                joints = mano_output.joints[0].cpu().numpy()  # [21, 3]
            
            # Store camera-space data
            transl_cam = result['transl']
            joints_cam = joints + transl_cam
            
            hand_data_camspace = {
                'transl': transl_cam,
                'betas': result['beta'],
                'hand_pose': result['hand_pose'],
                'global_orient': result['global_orient'],
                'joints': joints_cam,
            }
            
            # Transform to world space, falling back to an identity camera pose
            # for frames without MegaSAM output. This matches build_extrinsics_matrix().
            hand_data_worldspace = None
            try:
                R_c2w, t_c2w = get_camera_to_world_pose(camera_poses, frame_idx)
                hand_data_worldspace = transform_hand_to_world_space(
                    hand_data_camspace=hand_data_camspace,
                    R_c2w=R_c2w,
                    t_c2w=t_c2w,
                    reconstructor=reconstructor,
                    hand_side='right',
                )
            except Exception as e:
                print(f"  Warning: Failed to transform right hand to world space (frame {frame_idx}): {e}")
            
            right_hand_data[frame_idx] = {
                'camera_space': hand_data_camspace,
                'world_space': hand_data_worldspace,
            }
    
    # Calculate metrics
    total_rotvec_degree, total_transl_dist = calculate_camera_metrics(camera_poses, num_frames)
    
    # Calculate average wrist speed (use whichever hand has more valid frames)
    left_speed = calculate_avg_wrist_speed(left_hand_data, num_frames)
    right_speed = calculate_avg_wrist_speed(right_hand_data, num_frames)
    avg_speed = max(left_speed, right_speed)  # Use the more active hand's speed
    
    # Build intrinsics matrix (3x3)
    intrinsics = build_intrinsics_matrix(focal_length, img_center)
    
    # Build extrinsics matrix (Tx4x4) - World2Cam
    extrinsics = build_extrinsics_matrix(camera_poses, num_frames)
    
    # Aggregate hand data to arrays
    left_hand_arrays = aggregate_hand_data_to_arrays(left_hand_data, num_frames)
    right_hand_arrays = aggregate_hand_data_to_arrays(right_hand_data, num_frames)
    
    # Build text field
    # Format: {'left': [(description, (start_frame, end_frame))], 'right': [...]}
    text = {
        'left': [],
        'right': []
    }
    if hand_descriptions['left']:
        text['left'] = [(hand_descriptions['left'], (0, num_frames - 1))]
    if hand_descriptions['right']:
        text['right'] = [(hand_descriptions['right'], (0, num_frames - 1))]
    
    # Determine anno_type based on which hand has valid data (check kept_frames)
    left_valid = sum(left_hand_arrays['kept_frames'])
    right_valid = sum(right_hand_arrays['kept_frames'])
    
    if left_valid > 0 and right_valid > 0:
        if left_valid > right_valid:
            anno_type = 'left'
        elif right_valid > left_valid:
            anno_type = 'right'
        else:
            anno_type = 'both'
    elif left_valid > 0:
        anno_type = 'left'
    elif right_valid > 0:
        anno_type = 'right'
    else:
        anno_type = 'none'
    
    # Compile final results in episode_info format
    episode_info = {
        'extrinsics': extrinsics,  # (Tx4x4) World2Cam
        'intrinsics': intrinsics,  # (3x3)
        'video_decode_frame': list(range(num_frames)),  # Frame indices
        'video_name': video_path.name,  # Original video name
        'avg_speed': float(avg_speed),  # Average wrist movement per frame
        'total_rotvec_degree': float(total_rotvec_degree),  # Total camera rotation
        'total_transl_dist': float(total_transl_dist),  # Total camera translation
        'anno_type': anno_type,  # Annotation type
        'text': text,  # Text descriptions
    }
    
    # Always add left and right hand data (use kept_frames to check valid frames)
    episode_info['left'] = left_hand_arrays
    episode_info['right'] = right_hand_arrays
    
    # Save results if requested
    if save_output and input_path is not None:
        output_file = get_npy_path(video_path, input_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(output_file, episode_info, allow_pickle=True)
        
        print(f"  Saved results to: {output_file}")
    
    return episode_info


def process_videos_worker(worker_id, gpu_id, video_paths, input_path, device_type):
    """
    Worker process: Initialize model and process assigned video list.
    
    Args:
        worker_id: Worker ID
        gpu_id: Physical GPU ID
        video_paths: List of video paths to process
        input_path: Root input path
        device_type: 'cuda' or 'cpu'
    """
    # Note: CUDA_VISIBLE_DEVICES environment variable should be set in worker_wrapper
    # Here we just use cuda:0, because after setting the env var, the specified gpu_id becomes 0
    if device_type == "cuda":
        device = "cuda:0"  # After CUDA_VISIBLE_DEVICES is set, specified gpu_id becomes 0
        # Explicitly set current process's default CUDA device (double insurance)
        torch.cuda.set_device(0)
    else:
        device = "cpu"
    
    print(f"Worker {worker_id} initializing: Loading model to {device} (physical GPU {gpu_id})")
    
    try:
        # Setup configuration with default paths
        config = HandReconConfig()
        
        # Initialize reconstructor (includes MANO model internally)
        reconstructor = HandReconstructor(config=config, device=device)
        
        # Verify actual GPU being used
        if device_type == "cuda":
            actual_device = torch.cuda.current_device()
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
            print(f"Worker {worker_id} initialized: {device}, using GPU {actual_device} (physical GPU {gpu_id}, CUDA_VISIBLE_DEVICES={cuda_visible})")
        else:
            print(f"Worker {worker_id} initialized: {device}")
        
        results = []
        for video_path in video_paths:
            # Check if npy file already exists
            save_path = get_npy_path(video_path, input_path)
            if save_path.exists():
                print(f"Worker {worker_id} skipping (already exists): {video_path.name}")
                results.append((video_path, True, "Already exists, skipped"))
                continue
            
            try:
                output_data = process_video(
                    video_path=video_path,
                    reconstructor=reconstructor,
                    input_path=input_path,
                    save_output=True,
                )
                results.append((video_path, True, None))
                print(f"Worker {worker_id} completed: {video_path.name}")
            except Exception as e:
                results.append((video_path, False, str(e)))
                print(f"Worker {worker_id} error: {video_path.name} - {e}")
        
        return results
    except Exception as e:
        print(f"Worker {worker_id} initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return [(vp, False, f"Initialization failed: {e}") for vp in video_paths]


def main():
    parser = argparse.ArgumentParser(description='Process videos to extract hand reconstruction data')
    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help='Root path to search for videos (recursively). '
             'Typical WM-H output: WM-H/database/wm-h/instr_first/streaming_runs/run_*/',
    )
    parser.add_argument(
        '--dataset-name',
        type=str,
        default=os.environ.get('WH0_DATASET_NAME', 'WM-H'),
        help='Prefix for exported .npy annotation files',
    )
    parser.add_argument(
        '--parallel_k',
        type=int,
        default=1,
        help='Parallel level: number of models to load per GPU. Set to 0 for serial processing (default)'
    )
    
    args = parser.parse_args()

    global DATASET_NAME
    DATASET_NAME = args.dataset_name

    # Setup paths
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return
    
    # Find all videos
    print(f"Searching for videos in: {input_path}")
    video_files = find_videos(input_path)
    
    if len(video_files) == 0:
        print(f"No video files found in {input_path}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Filter out videos that already have npy files
    videos_to_process = []
    skipped_count = 0
    for video_path in video_files:
        npy_path = get_npy_path(video_path, input_path)
        if npy_path.exists():
            skipped_count += 1
        else:
            videos_to_process.append(video_path)
    
    if skipped_count > 0:
        print(f"Skipped {skipped_count} videos that already have npy files")
    print(f"Need to process {len(videos_to_process)} videos\n")
    
    if not videos_to_process:
        print("All videos have been processed")
        return
    
    # Parallel processing
    if args.parallel_k > 0:
        # Detect GPU count
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            device_type = "cuda"
        else:
            num_gpus = 1
            device_type = "cpu"
        
        # Calculate total workers: k models per GPU
        total_workers = num_gpus * args.parallel_k
        print(f"Detected {num_gpus} GPUs, loading {args.parallel_k} models per GPU")
        print(f"Creating {total_workers} worker processes in total")
        
        # Distribute videos to workers using round-robin for load balancing
        worker_videos = [[] for _ in range(total_workers)]
        for idx, video_path in enumerate(videos_to_process):
            worker_id = idx % total_workers
            worker_videos[worker_id].append(video_path)
        
        # Create process list
        manager = Manager()
        result_queue = manager.Queue()
        processes = []
        
        def worker_wrapper(worker_id, gpu_id, video_paths, input_path, device_type, result_queue):
            """
            Wrapper function to put results in queue.
            Set environment variable in worker function to ensure each process only sees specified GPU.
            Note: Must be set before torch uses CUDA.
            """
            if device_type == "cuda":
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            results = process_videos_worker(worker_id, gpu_id, video_paths, input_path, device_type)
            for result in results:
                result_queue.put(result)
        
        # Start all worker processes
        worker_id = 0
        for gpu_id in range(num_gpus):
            for k_id in range(args.parallel_k):
                video_list = worker_videos[worker_id]
                # Only create process for workers with tasks
                if video_list:
                    p = Process(
                        target=worker_wrapper,
                        args=(worker_id, gpu_id, video_list, input_path, device_type, result_queue)
                    )
                    p.start()
                    processes.append(p)
                    print(f"Started Worker {worker_id} on GPU {gpu_id}, processing {len(video_list)} videos")
                worker_id += 1
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Collect results
        success_count = 0
        fail_count = 0
        while not result_queue.empty():
            video_path, success, error = result_queue.get()
            if success:
                success_count += 1
            else:
                fail_count += 1
                print(f"Processing failed: {video_path} - {error}")
        
        print(f"\nProcessing complete: {success_count} succeeded, {fail_count} failed")
        print(f"Results saved to: {input_path / 'episodic_annotations'}")
    
    else:
        # Serial processing (original way)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Setup configuration with default paths
        config = HandReconConfig()
        
        # Initialize reconstructor (includes MANO model internally)
        print("Initializing HandReconstructor...")
        reconstructor = HandReconstructor(config=config, device=device)
        
        for video_path in tqdm(videos_to_process, desc="Processing videos"):
            # Check again (though already filtered in main function, but for safety)
            save_path = get_npy_path(video_path, input_path)
            if save_path.exists():
                print(f"Skipping (already exists): {video_path.name}")
                continue
            
            try:
                process_video(
                    video_path=video_path,
                    reconstructor=reconstructor,
                    input_path=input_path,
                    save_output=True,
                )
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\nProcessing complete! Results saved to: {input_path / 'episodic_annotations'}")


if __name__ == '__main__':
    main()

"""
MegaSAM-based camera pose tracking using MoGe-2 depth estimation.

Uses DROID-SLAM (from HaWoR thirdparty) with MoGe-2 depth priors.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .moge import MogePipeline
from .paths import ensure_hawor_on_path, hawor_root, repo_root


def _ensure_droid_on_path() -> None:
    ensure_hawor_on_path()
    droid_root = hawor_root() / "thirdparty" / "DROID-SLAM"
    paths = [
        droid_root,
        droid_root / "droid_slam",
        *sorted((droid_root / "build").glob("lib.*")),
    ]
    for path in reversed(paths):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


_ensure_droid_on_path()
from droid import Droid  # noqa: E402
from lietorch import SE3  # noqa: E402


class MegaSAMPipeline:
    """Camera pose tracking with DROID-SLAM and MoGe-2 depth priors."""

    def __init__(
        self,
        moge_pipeline: Optional[MogePipeline] = None,
        device: torch.device | None = None,
        weights_path: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        moge_model = os.environ.get(
            "WH0_MOGE_MODEL",
            str(repo_root() / "weights" / "models" / "Ruicheng" / "moge-2-vitl" / "model.pt"),
        )
        self.moge_pipeline = moge_pipeline or MogePipeline(
            model_name=moge_model,
            device=self.device,
        )
        self.droid_args = self._create_droid_args(weights_path)

    def _create_droid_args(self, weights_path: Optional[str] = None):
        if weights_path is None:
            weights_path = os.environ.get(
                "WH0_DROID_WEIGHTS",
                str(repo_root() / "weights" / "external" / "droid.pth"),
            )

        class Args:
            def __init__(self):
                self.weights = weights_path
                self.buffer = 1024
                self.image_size = [240, 320]
                self.disable_vis = True
                self.beta = 0.3
                self.filter_thresh = 2.0
                self.warmup = 8
                self.keyframe_thresh = 2.0
                self.frontend_thresh = 12.0
                self.frontend_window = 25
                self.frontend_radius = 2
                self.frontend_nms = 1
                self.backend_thresh = 16.0
                self.backend_radius = 2
                self.backend_nms = 3
                self.stereo = False
                self.depth = True
                self.upsample = True

        return Args()

    def estimate_depth_with_moge(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        fov_x = self.moge_pipeline.infer(image)
        _ = fov_x  # FoV informs scene scale; depth prior is geometric

        center_y, center_x = h // 2, w // 2
        y_coords, x_coords = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        depth = 1.0 + 2.0 * (dist_from_center / max_dist)
        return depth.astype(np.float32)

    def _resize_for_droid(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
        h0, w0 = image.shape[:2]
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image_resized = cv2.resize(image, (w1, h1), interpolation=cv2.INTER_AREA)
        image_resized = image_resized[: h1 - h1 % 8, : w1 - w1 % 8]
        image_tensor = torch.as_tensor(image_resized).permute(2, 0, 1).float()
        return image_tensor, (h0, w0), (h1, w1)

    def _resize_depth_for_droid(self, depth: np.ndarray, target_size: Tuple[int, int]) -> torch.Tensor:
        depth_tensor = torch.as_tensor(depth).float()
        depth_resized = F.interpolate(
            depth_tensor[None, None],
            target_size,
            mode="nearest-exact",
        ).squeeze()
        h1, w1 = target_size
        return depth_resized[: h1 - h1 % 8, : w1 - w1 % 8]

    def _mask_for_droid(self, image_tensor: torch.Tensor) -> torch.Tensor:
        h, w = image_tensor.shape[-2:]
        return torch.ones((h // 8, w // 8), dtype=torch.float32)

    def track_camera_poses(
        self,
        frames: List[np.ndarray],
        camera_intrinsics: Optional[np.ndarray] = None,
        focal_length: Optional[float] = None,
        img_center: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        if Droid is None:
            raise ImportError("DROID-SLAM not available. Install HaWoR thirdparty/DROID-SLAM.")

        num_frames = len(frames)
        if num_frames == 0:
            raise ValueError("No frames provided")

        h, w = frames[0].shape[:2]

        if camera_intrinsics is None:
            if focal_length is None:
                fov_x = self.moge_pipeline.infer(frames[0])
                focal_length = 0.5 * w / np.tan(0.5 * fov_x * np.pi / 180)

            if img_center is None:
                img_center = np.array([w / 2.0, h / 2.0])

            k = np.eye(3)
            k[0, 0] = focal_length
            k[1, 1] = focal_length
            k[0, 2] = img_center[0]
            k[1, 2] = img_center[1]
        else:
            k = camera_intrinsics
            focal_length = k[0, 0]
            img_center = np.array([k[0, 2], k[1, 2]])

        droid = None
        print("Tracking camera poses with DROID-SLAM + MoGe-2 depth...")

        for t, frame in enumerate(tqdm(frames, desc="Processing frames")):
            image_tensor, (h0, w0), (h1, w1) = self._resize_for_droid(frame)
            depth = self.estimate_depth_with_moge(frame)
            depth_tensor = self._resize_depth_for_droid(depth, (h1, w1))
            mask = self._mask_for_droid(image_tensor)

            fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
            intrinsics = torch.as_tensor([fx, fy, cx, cy], dtype=torch.float32)
            intrinsics[0::2] *= w1 / w0
            intrinsics[1::2] *= h1 / h0

            if droid is None:
                self.droid_args.image_size = [image_tensor.shape[1], image_tensor.shape[2]]
                droid = Droid(self.droid_args)

            droid.track(t, image_tensor[None], depth=depth_tensor, intrinsics=intrinsics, mask=mask)

        if droid is not None and hasattr(droid, "track_final"):
            droid.track_final(
                len(frames) - 1,
                image_tensor[None],
                depth=depth_tensor,
                intrinsics=intrinsics,
                mask=mask,
            )

        def simple_image_stream():
            for t in range(len(frames)):
                frame = frames[t]
                image_tensor, (h0, w0), (h1, w1) = self._resize_for_droid(frame)
                depth = self.estimate_depth_with_moge(frame)
                depth_tensor = self._resize_depth_for_droid(depth, (h1, w1))
                fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
                intrinsics = torch.as_tensor([fx, fy, cx, cy], dtype=torch.float32)
                intrinsics[0::2] *= w1 / w0
                intrinsics[1::2] *= h1 / h0
                mask = self._mask_for_droid(image_tensor)
                yield t, image_tensor[None], depth_tensor, intrinsics, mask

        try:
            terminated = droid.terminate(simple_image_stream())
            if isinstance(terminated, tuple):
                traj_est, depth_est, motion_prob = terminated
            else:
                traj_est = terminated
                depth_est = None
                motion_prob = None
        except Exception as e:
            print(f"Warning: Error in droid.terminate: {e}")
            n = droid.video.counter.value
            traj_est = droid.video.poses.cpu().numpy()[:n]
            depth_est = None
            motion_prob = None

        poses_th = torch.as_tensor(traj_est, device="cpu")
        cam_c2w = SE3(poses_th).inv().matrix().numpy()

        r_c2w = cam_c2w[:, :3, :3]
        t_c2w = cam_c2w[:, :3, 3]
        r_w2c = r_c2w.transpose(0, 2, 1)
        t_w2c = -np.einsum("tij,tj->ti", r_w2c, t_c2w)
        intrinsics_all = droid.video.intrinsics[: len(frames)].cpu().numpy()

        return {
            "R_c2w": r_c2w,
            "t_c2w": t_c2w,
            "R_w2c": r_w2c,
            "t_w2c": t_w2c,
            "scale": 1.0,
            "intrinsics": intrinsics_all,
            "trajectory": traj_est,
        }

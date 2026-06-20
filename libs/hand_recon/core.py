from __future__ import annotations

import copy

import numpy as np
import torch

from .config import HandReconConfig
from .hawor import HaworPipeline
from .mano import MANO
from .moge import MogePipeline


class HandReconstructor:
    """
    3D hand reconstruction combining MoGe (FoV), HaWoR (pose), and MANO alignment.
    """

    def __init__(
        self,
        config: HandReconConfig | None = None,
        device: torch.device = torch.device("cuda"),
    ):
        config = config or HandReconConfig()
        self.device = device
        self.hawor_pipeline = HaworPipeline(
            model_path=config.hawor_model_path,
            detector_path=config.detector_path,
            device=device,
        )
        self.moge_pipeline = MogePipeline(model_name=config.moge_model_path, device=device)
        self.mano = MANO(model_path=config.mano_path).to(device)

    def recon(self, images: list) -> dict:
        n = len(images)
        if n == 0:
            return {"left": {}, "right": {}, "fov_x": None}

        h, w = images[0].shape[:2]

        all_fov_x = [self.moge_pipeline.infer(img) for img in images]
        fov_x = float(np.median(np.array(all_fov_x)))
        img_focal = 0.5 * w / np.tan(0.5 * fov_x * np.pi / 180)

        recon_results = self.hawor_pipeline.recon(images, img_focal, single_image=(n == 1))
        recon_results_new_transl = {"left": {}, "right": {}, "fov_x": fov_x}

        for img_idx in range(n):
            for hand_type in ["left", "right"]:
                hand_results = recon_results[hand_type]
                if img_idx not in hand_results:
                    continue
                result = hand_results[img_idx]

                betas = torch.from_numpy(result["beta"]).unsqueeze(0).to(self.device)
                hand_pose = torch.from_numpy(result["hand_pose"]).unsqueeze(0).to(self.device)
                transl = torch.from_numpy(result["transl"]).unsqueeze(0).to(self.device)

                model_output = self.mano(betas=betas, hand_pose=hand_pose)
                joints_m = model_output.joints[0]

                if hand_type == "left":
                    joints_m = joints_m.clone()
                    joints_m[:, 0] *= -1

                wrist = joints_m[0]
                transl_new = wrist + transl

                result_new_transl = copy.deepcopy(result)
                result_new_transl["transl"] = transl_new[0].cpu().numpy()
                recon_results_new_transl[hand_type][img_idx] = result_new_transl

        return recon_results_new_transl

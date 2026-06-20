from __future__ import annotations

import cv2
import numpy as np
import torch

from moge.model.v2 import MoGeModel as MoGeModelV2  # noqa: E402


class MogePipeline:
    """Estimate horizontal field-of-view with MoGe-2."""

    def __init__(
        self,
        model_name: str = "Ruicheng/moge-2-vitl",
        device: torch.device = torch.device("cuda"),
    ):
        self.device = device
        self.model = MoGeModelV2.from_pretrained(model_name).to(device)

    def infer(self, input_image: np.ndarray) -> float:
        input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_tensor = torch.tensor(
            input_image_rgb / 255.0,
            dtype=torch.float32,
            device=self.device,
        ).permute(2, 0, 1)

        output = self.model.infer(input_tensor, resolution_level=1)
        intrinsics = output["intrinsics"].cpu().numpy()
        fov_x_rad = 2 * np.arctan(intrinsics[0, 2] / intrinsics[0, 0])
        return float(np.rad2deg(fov_x_rad))

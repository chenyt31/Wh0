# MANO wrapper from https://github.com/geopavlakos/hamer/blob/main/hamer/models/mano_wrapper.py
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import smplx
import torch
from smplx.lbs import vertices2joints
from smplx.utils import MANOOutput, to_tensor
from smplx.vertex_ids import vertex_ids


class MANO(smplx.MANOLayer):
    def __init__(
        self,
        *args,
        model_path: Optional[str] = None,
        joint_regressor_extra: Optional[str] = None,
        **kwargs,
    ):
        if model_path is not None:
            model_dir = Path(model_path)
            kwargs.setdefault("model_path", str(model_dir))
            kwargs.setdefault("is_rhand", True)
            kwargs.setdefault("use_pca", False)
            kwargs.setdefault("flat_hand_mean", True)

        super().__init__(*args, **kwargs)
        mano_to_openpose = [
            0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20
        ]

        if joint_regressor_extra is not None:
            self.register_buffer(
                "joint_regressor_extra",
                torch.tensor(
                    pickle.load(open(joint_regressor_extra, "rb"), encoding="latin1"),
                    dtype=torch.float32,
                ),
            )
        self.register_buffer(
            "extra_joints_idxs",
            to_tensor(list(vertex_ids["mano"].values()), dtype=torch.long),
        )
        self.register_buffer("joint_map", torch.tensor(mano_to_openpose, dtype=torch.long))

    def forward(self, *args, **kwargs) -> MANOOutput:
        mano_output = super().forward(*args, **kwargs)
        extra_joints = torch.index_select(mano_output.vertices, 1, self.extra_joints_idxs)
        joints = torch.cat([mano_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        if hasattr(self, "joint_regressor_extra"):
            extra_joints = vertices2joints(self.joint_regressor_extra, mano_output.vertices)
            joints = torch.cat([joints, extra_joints], dim=1)
        mano_output.joints = joints
        return mano_output

    def query(self, hmr_output):
        batch_size = hmr_output["pred_rotmat"].shape[0]
        pred_rotmat = hmr_output["pred_rotmat"].reshape(batch_size, -1, 3, 3)
        pred_shape = hmr_output["pred_shape"].reshape(batch_size, 10)

        return self(
            global_orient=pred_rotmat[:, [0]],
            hand_pose=pred_rotmat[:, 1:],
            betas=pred_shape,
            pose2rot=False,
        )

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

from .paths import ensure_hawor_on_path, repo_root

for _name, _value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if not hasattr(np, _name):
        setattr(np, _name, _value)

ensure_hawor_on_path()
from hawor.configs import get_config  # noqa: E402
from hawor.utils.rotation import (  # noqa: E402
    angle_axis_to_rotation_matrix,
    rotation_matrix_to_angle_axis,
)
from lib.eval_utils.custom_utils import interpolate_bboxes  # noqa: E402
from lib.models.hawor import HAWOR  # noqa: E402
from lib.pipeline.tools import parse_chunks  # noqa: E402


def _ensure_mano_mean_params(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        pose=np.zeros(96, dtype=np.float32),
        shape=np.zeros(10, dtype=np.float32),
        cam=np.zeros(3, dtype=np.float32),
    )


def load_hawor(checkpoint_path: str):
    model_cfg_path = Path(checkpoint_path).parent.parent / "model_config.yaml"
    model_cfg = get_config(str(model_cfg_path), update_cachedir=True)
    model_cfg.defrost()
    model_cfg.MANO.MODEL_PATH = str(repo_root() / "weights")
    model_cfg.MANO.MEAN_PARAMS = str(repo_root() / "weights" / "mano" / "mano_mean_params.npz")
    _ensure_mano_mean_params(Path(model_cfg.MANO.MEAN_PARAMS))

    if (model_cfg.MODEL.BACKBONE.TYPE == "vit") and ("BBOX_SHAPE" not in model_cfg.MODEL):
        assert model_cfg.MODEL.IMAGE_SIZE == 256, (
            f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        )
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
    model_cfg.freeze()

    try:
        model = HAWOR.load_from_checkpoint(
            checkpoint_path,
            strict=False,
            cfg=model_cfg,
            weights_only=False,
        )
    except TypeError:
        model = HAWOR.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
    return model, model_cfg


class HaworPipeline:
    """Hand detection, tracking, and HaWoR motion estimation."""

    def __init__(
        self,
        model_path: str = "",
        detector_path: str = "",
        device: torch.device = torch.device("cuda"),
    ):
        self.device = device
        self.detector_path = detector_path

        model, _ = load_hawor(model_path)
        model = model.to(device)
        model.eval()
        self.model = model

    def recon(
        self,
        images: list,
        img_focal: float,
        thresh: float = 0.2,
        single_image: bool = False,
    ) -> dict:
        hand_det_model = YOLO(self.detector_path)
        _, tracks = detect_track(images, hand_det_model, thresh=thresh)
        recon_results = hawor_motion_estimation(
            images, tracks, self.model, img_focal, single_image=single_image
        )
        del hand_det_model
        return recon_results


# Adapted from https://github.com/ThunderVVV/HaWoR/blob/main/scripts/scripts_test_video/detect_track_video.py
def detect_track(imgfiles: list, hand_det_model: YOLO, thresh: float = 0.5) -> tuple:
    boxes_ = []
    tracks = {}

    for t, img_cv2 in enumerate(tqdm(imgfiles)):
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                results = hand_det_model.track(img_cv2, conf=thresh, persist=True, verbose=False)

                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                handedness = results[0].boxes.cls.cpu().numpy()
                if results[0].boxes.id is not None:
                    track_id = results[0].boxes.id.cpu().numpy()
                else:
                    track_id = [-1] * len(boxes)

                boxes = np.hstack([boxes, confs[:, None]])

                find_right = False
                find_left = False

                for idx, box in enumerate(boxes):
                    if track_id[idx] == -1:
                        id = int(10000) if handedness[[idx]] > 0 else int(5000)
                    else:
                        id = track_id[idx]
                    subj = {
                        "frame": t,
                        "det": True,
                        "det_box": boxes[[idx]],
                        "det_handedness": handedness[[idx]],
                    }

                    if (not find_right and handedness[[idx]] > 0) or (
                        not find_left and handedness[[idx]] == 0
                    ):
                        if id in tracks:
                            tracks[id].append(subj)
                        else:
                            tracks[id] = [subj]

                        if handedness[[idx]] > 0:
                            find_right = True
                        elif handedness[[idx]] == 0:
                            find_left = True

    return boxes_, tracks


# Adapted from https://github.com/ThunderVVV/HaWoR/blob/main/scripts/scripts_test_video/hawor_video.py
def hawor_motion_estimation(
    imgfiles: list,
    tracks: dict,
    model: HAWOR,
    img_focal: float,
    single_image: bool = False,
) -> dict:
    left_results = {}
    right_results = {}

    tid = np.array([tr for tr in tracks])
    left_trk = []
    right_trk = []
    for idx in tid:
        trk = tracks[idx]
        valid = np.array([t["det"] for t in trk])
        is_right = np.concatenate([t["det_handedness"] for t in trk])[valid]

        if is_right.sum() / len(is_right) < 0.5:
            left_trk.extend(trk)
        else:
            right_trk.extend(trk)
    left_trk = sorted(left_trk, key=lambda x: x["frame"])
    right_trk = sorted(right_trk, key=lambda x: x["frame"])
    final_tracks = {0: left_trk, 1: right_trk}
    tid = [0, 1]

    img = imgfiles[0]
    img_center = [img.shape[1] / 2, img.shape[0] / 2]
    H, W = img.shape[:2]

    for idx in tid:
        print(f"tracklet {idx}:")
        trk = final_tracks[idx]

        valid = np.array([t["det"] for t in trk])
        if not single_image:
            if valid.sum() < 2:
                continue
        else:
            if valid.sum() < 1:
                continue
        boxes = np.concatenate([t["det_box"] for t in trk])
        non_zero_indices = np.where(np.any(boxes != 0, axis=1))[0]
        first_non_zero = non_zero_indices[0]
        last_non_zero = non_zero_indices[-1]
        boxes[first_non_zero : last_non_zero + 1] = interpolate_bboxes(
            boxes[first_non_zero : last_non_zero + 1]
        )
        valid[first_non_zero : last_non_zero + 1] = True

        boxes = boxes[first_non_zero : last_non_zero + 1]
        is_right = np.concatenate([t["det_handedness"] for t in trk])[valid]
        frame = np.array([t["frame"] for t in trk])[valid]

        if is_right.sum() / len(is_right) < 0.5:
            is_right = np.zeros((len(boxes), 1))
        else:
            is_right = np.ones((len(boxes), 1))

        frame_chunks, boxes_chunks = parse_chunks(frame, boxes, min_len=1)
        if len(frame_chunks) == 0:
            continue

        for frame_ck, boxes_ck in zip(frame_chunks, boxes_chunks):
            print(f"inference from frame {frame_ck[0]} to {frame_ck[-1]}")
            img_ck = [imgfiles[i] for i in frame_ck]
            do_flip = is_right[0] <= 0

            results = model.inference(
                img_ck,
                boxes_ck,
                img_focal=img_focal,
                img_center=img_center,
                do_flip=do_flip,
            )

            data_out = {
                "init_root_orient": results["pred_rotmat"][None, :, 0],
                "init_hand_pose": results["pred_rotmat"][None, :, 1:],
                "init_trans": results["pred_trans"][None, :, 0],
                "init_betas": results["pred_shape"][None, :],
            }

            init_root = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
            init_hand_pose = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])
            if do_flip:
                init_root[..., 1] *= -1
                init_root[..., 2] *= -1
            data_out["init_root_orient"] = angle_axis_to_rotation_matrix(init_root)
            data_out["init_hand_pose"] = angle_axis_to_rotation_matrix(init_hand_pose)

            s_frame = frame_ck[0]
            e_frame = frame_ck[-1]

            for frame_id in range(s_frame, e_frame + 1):
                result = {
                    "beta": data_out["init_betas"][0, frame_id - s_frame].cpu().numpy(),
                    "hand_pose": data_out["init_hand_pose"][0, frame_id - s_frame].cpu().numpy(),
                    "global_orient": data_out["init_root_orient"][0, frame_id - s_frame].cpu().numpy(),
                    "transl": data_out["init_trans"][0, frame_id - s_frame].cpu().numpy(),
                }

                if idx == 0:
                    left_results[frame_id] = result
                else:
                    right_results[frame_id] = result

    return {"left": left_results, "right": right_results}

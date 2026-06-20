"""Replace human hands with robot hands in generated videos (Qwen-Image-Edit per frame)."""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm

from wm_h.box_editor import load_qwen_image_edit_pipeline

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "Change the visible person's hand into a realistic robotic hand with a white back shell, "
    "black palm, black fingertips, and a silver forearm. Only modify the hand skin/material; "
    "do not change pose, position, proportions, background, or composition. If the image shows "
    "one hand, keep exactly one hand and do not add another hand."
)
DEFAULT_NEGATIVE_PROMPT = (
    "cartoon, anime, comic, toy hand, plastic hand, CGI look, stylized, 3d render, "
    "illustration, low poly, fake robot hand, deformed fingers, extra fingers, extra hand, "
    "second hand, duplicate hand, blurry, low quality"
)

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def frame_indices_to_replace(num_frames: int, every_n: int = 4, offset: int = 1) -> List[int]:
    """Frame indices to edit; default every fourth frame starting at offset."""
    return [i for i in range(num_frames) if (i % every_n) == offset]


def _task_path_for_video(video_path: Path) -> Path | None:
    task_dir = video_path.parent.parent / "tasks"
    if not task_dir.is_dir():
        return None
    candidates = [task_dir / f"{video_path.stem}.json"]
    if "_" in video_path.stem:
        candidates.append(task_dir / f"{video_path.stem.split('_', 1)[1]}.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_task_metadata(video_path: Path) -> dict:
    task_path = _task_path_for_video(video_path)
    if task_path is None:
        return {}
    try:
        return json.loads(task_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("failed to read task metadata %s: %s", task_path, exc)
        return {}


def hand_mode_from_task(task: dict) -> str:
    hand = str(task.get("hand") or "").strip().lower()
    if bool(task.get("use_both_hands")) or hand in {"both", "bimanual", "two_hands", "two-hands"}:
        return "both"
    left = str(task.get("left_instruction") or "").strip().lower().rstrip(".")
    right = str(task.get("right_instruction") or "").strip().lower().rstrip(".")
    left_active = bool(left and left != "none")
    right_active = bool(right and right != "none")
    if left_active and right_active:
        return "both"
    if left_active or hand == "left":
        return "left"
    if right_active or hand == "right":
        return "right"
    return "visible"


def prompt_for_hand_mode(base_prompt: str, hand_mode: str) -> str:
    base = base_prompt.strip() or DEFAULT_PROMPT
    if hand_mode == "left":
        constraint = (
            "Edit only the visible LEFT hand. Keep the right hand absent if it is not visible. "
            "Do not add, reveal, or synthesize a second hand."
        )
    elif hand_mode == "right":
        constraint = (
            "Edit only the visible RIGHT hand. Keep the left hand absent if it is not visible. "
            "Do not add, reveal, or synthesize a second hand."
        )
    elif hand_mode == "both":
        constraint = (
            "Edit the two visible hands only. Do not add any extra hands or duplicate fingers."
        )
    else:
        constraint = (
            "Edit only the hand or hands already visible in the image. Do not add any new hand."
        )
    return f"{base} {constraint}"


def read_video_frames(video_path: Path) -> tuple[list[np.ndarray], float]:
    reader = imageio.get_reader(str(video_path), "ffmpeg")
    meta = reader.get_meta_data()
    fps = float(meta.get("fps") or 24.0)
    if fps <= 0:
        fps = 24.0
    frames: list[np.ndarray] = []
    for i in range(reader.count_frames()):
        frame = reader.get_data(i)
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.shape[-1] == 4:
            frame = frame[..., :3]
        frames.append(frame)
    reader.close()
    return frames, fps


def write_video(frames: Sequence[np.ndarray], save_path: Path, fps: float, quality: int = 9) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(save_path), fps=fps, quality=quality)
    for frame in frames:
        writer.append_data(np.asarray(frame))
    writer.close()


def edit_frame_robot_hand(
    pipe,
    frame_rgb: np.ndarray,
    *,
    prompt: str,
    negative_prompt: str,
    seed: int,
    num_inference_steps: int,
    cfg_scale: float,
    height: int,
    width: int,
) -> np.ndarray:
    frame_pil = Image.fromarray(frame_rgb.astype(np.uint8)).convert("RGB")
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        edit_image=[frame_pil],
        seed=seed,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
        height=height,
        width=width,
        edit_image_auto_resize=False,
        zero_cond_t=True,
    )
    return np.array(out)


def process_video(
    video_path: Path,
    output_path: Path,
    pipe,
    *,
    every_n: int = 4,
    offset: int = 1,
    prompt: str = DEFAULT_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    seed: int = 42,
    num_inference_steps: int = 4,
    cfg_scale: float = 1.0,
    quality: int = 9,
    skip_existing: bool = True,
) -> bool:
    if skip_existing and output_path.is_file():
        logger.info("skip existing %s", output_path)
        return False

    frames, fps = read_video_frames(video_path)
    if not frames:
        logger.warning("empty video: %s", video_path)
        return False

    task = load_task_metadata(video_path)
    edit_prompt = prompt_for_hand_mode(prompt, hand_mode_from_task(task))
    replace = set(frame_indices_to_replace(len(frames), every_n=every_n, offset=offset))
    h, w = frames[0].shape[:2]
    out_frames: list[np.ndarray] = []
    for i, frame in enumerate(
        tqdm(frames, desc=video_path.name, leave=False, disable=len(frames) < 4)
    ):
        if i in replace:
            out_frames.append(
                edit_frame_robot_hand(
                    pipe,
                    frame,
                    prompt=edit_prompt,
                    negative_prompt=negative_prompt,
                    seed=seed + i,
                    num_inference_steps=num_inference_steps,
                    cfg_scale=cfg_scale,
                    height=h,
                    width=w,
                )
            )
        else:
            out_frames.append(frame)

    write_video(out_frames, output_path, fps=fps, quality=quality)
    return True


def collect_videos(input_dir: Path) -> List[Path]:
    if not input_dir.is_dir():
        return []
    return sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def find_latest_videos_dir(base: Path) -> Optional[Path]:
    """Newest .../videos directory under instr_first streaming or batch runs."""
    candidates: list[Path] = []
    instr_root = base / "instr_first"
    for pattern in ("streaming_runs/*/videos", "runs/*/videos"):
        candidates.extend(instr_root.glob(pattern))
    gen_root = base / "video" / "generated"
    if gen_root.is_dir():
        candidates.extend(gen_root.glob("run_*/videos"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_io_dirs(data_cfg: dict, repo_root: Path) -> tuple[Path, Path]:
    input_raw = (data_cfg.get("input_dir") or "").strip()
    if input_raw:
        input_dir = Path(input_raw)
        if not input_dir.is_absolute():
            input_dir = (repo_root / input_dir).resolve()
    else:
        found = find_latest_videos_dir(repo_root / "database/wm-h")
        if found is None:
            raise FileNotFoundError(
                "No videos directory found; set data.input_dir in configs/video_hand_edit.yaml"
            )
        input_dir = found

    output_raw = data_cfg.get("output_dir")
    if output_raw:
        output_dir = Path(output_raw)
        if not output_dir.is_absolute():
            output_dir = (repo_root / output_dir).resolve()
    else:
        output_dir = input_dir.parent / f"{input_dir.name}_robot_hands"
    return input_dir, output_dir


def load_hand_edit_pipe(model_cfg: dict, device: str):
    model_path = model_cfg["model_path"]
    lora = model_cfg.get("lightning_lora_path") or None
    return load_qwen_image_edit_pipeline(model_path, device=device, lightning_lora_path=lora)

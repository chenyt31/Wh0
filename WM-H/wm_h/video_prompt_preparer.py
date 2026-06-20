"""
Stage 1 (video): Read instr-first manifest rows, add Qwen-VL augmentation and video_prompt.

Input:  database/wm-h/instr_first/runs/run_*/manifest.jsonl
Output: database/wm-h/video/manifests/video_manifest_YYYYMMDD_HHMMSS.jsonl

Stage 2: wm_h/run_video_generator.py
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from wm_h.video.cuda import isolate_cuda_device
from wm_h.video.preparer import VideoPromptPreparer
from wm_h.video.common import (
    CheckpointManager,
    QWEN3_VL_COORD_MAX,
    build_hand_trajectory_vl_prompt,
    draw_hand_trajectories_on_image,
    format_bimanual_instruction,
    load_config,
    parse_hand_trajectory_json,
    scene_aug_output_path,
)
from wm_h.video_prompts import (
    build_desk_synth_bimanual_video_prompt,
    build_desk_synth_text_augmentation_prompt,
    build_desk_synth_video_prompt,
    sanitize_augmented_desc,
    scene_object_labels,
)

PROJECT_ROOT = ROOT


def _hand_active(instr: str) -> bool:
    return (instr or "").strip().lower() not in ("none", "none.", "")


def infer_hand_mode(row: Dict) -> str:
    """Return bimanual | left | right from per-hand instructions."""
    left = (row.get("left_instruction") or "").strip()
    right = (row.get("right_instruction") or "").strip()
    left_on = _hand_active(left)
    right_on = _hand_active(right)
    if left_on and right_on:
        return "bimanual"
    if left_on:
        return "left"
    return "right"


def shared_object_label(row: Dict) -> str:
    target = row.get("target_object") or {}
    if not isinstance(target, dict):
        return ""
    adj = (target.get("adjective") or "").strip()
    noun = (target.get("noun") or "").strip()
    if adj and noun:
        return f"{adj} {noun}"
    return noun or adj


def resolve_input_image(row: Dict, project_root: Path = PROJECT_ROOT) -> str:
    """Prefer edited/working scene image for I2V (OOI edit or resized desktop)."""
    for key in ("edited_image_path", "image_path", "desktop_image_path"):
        raw = (row.get(key) or "").strip()
        if not raw:
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = project_root / path
        if path.exists():
            return str(path.resolve())
    raise FileNotFoundError(
        f"No input image for task {row.get('task_id')}: "
        f"edited_image_path={row.get('edited_image_path')!r}"
    )


def read_synth_manifest(manifest_path: str, limit: int = -1) -> List[Dict]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    rows: List[Dict] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit != -1 and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def find_all_synth_manifests(mode: str = "instr_first") -> List[Path]:
    """All manifest.jsonl under instr_first runs (sorted)."""
    base = PROJECT_ROOT / "database/wm-h" / "instr_first" / "runs"
    if not base.is_dir():
        return []
    return sorted(base.glob("run_*/manifest.jsonl"))


def find_latest_synth_manifest(mode: str = "instr_first") -> Optional[Path]:
    """Pick newest manifest.jsonl under instr_first runs."""
    manifests = find_all_synth_manifests(mode)
    if not manifests:
        return None
    return max(manifests, key=lambda p: p.stat().st_mtime)


def make_checkpoint_id(synth_manifest: str, task_id: str) -> str:
    return CheckpointManager.get_manifest_row_id(synth_manifest, task_id)


def load_synth_jobs(manifest_paths: List[Path], limit: int = -1) -> List[Dict]:
    """Flatten rows from multiple synth manifests into processing jobs."""
    jobs: List[Dict] = []
    for manifest_path in manifest_paths:
        manifest_str = str(manifest_path.resolve())
        rows = read_synth_manifest(manifest_str)
        for row in rows:
            jobs.append({
                "row": row,
                "synth_manifest": manifest_str,
                "global_idx": len(jobs),
            })
            if limit != -1 and len(jobs) >= limit:
                return jobs
    return jobs


def scene_aug_path_for_task(scene_aug_dir: str, image_path: str, task_id: str) -> Path:
    stem = Path(image_path).stem
    out_dir = Path(scene_aug_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = task_id.replace("/", "_")
    return out_dir / f"{stem}_{safe_id}.png"


class DeskSynthVideoPromptPreparer(VideoPromptPreparer):
    """Augment existing wm-h manifest rows for Wan I2V."""

    @torch.no_grad()
    def augment_prompt_with_qwen(
        self,
        image_path: str,
        instruction: str,
        hand: str = "right",
        task_description: str = "",
        visible_objects: Optional[List[str]] = None,
        output_language: str = "zh",
        include_retry: bool = False,
    ) -> str:
        text_prompt = build_desk_synth_text_augmentation_prompt(
            instruction,
            hand=hand,
            mode="single",
            shared_goal=task_description or instruction,
            visible_objects=visible_objects,
            output_language=output_language,
            include_retry=include_retry,
        )
        max_tokens = self.gen_cfg.get("augment_max_new_tokens", 384)
        text = self._vl_generate(image_path, text_prompt, max_tokens)
        if text:
            self.logger.info(f"[Augment] {text[:100]}...")
        return sanitize_augmented_desc(text)

    @torch.no_grad()
    def augment_bimanual(
        self,
        image_path: str,
        left_instr: str,
        right_instr: str,
        task_description: str = "",
        shared_object: str = "",
        coordination_type: str = "",
        visible_objects: Optional[List[str]] = None,
        output_language: str = "zh",
        include_retry: bool = False,
    ) -> str:
        text_prompt = build_desk_synth_text_augmentation_prompt(
            instruction="",
            mode="bimanual",
            left_instruction=left_instr,
            right_instruction=right_instr,
            shared_goal=task_description,
            shared_object=shared_object,
            coordination_type=coordination_type,
            visible_objects=visible_objects,
            output_language=output_language,
            include_retry=include_retry,
        )
        max_tokens = self.gen_cfg.get("bimanual_augment_max_new_tokens", 512)
        text = self._vl_generate(image_path, text_prompt, max_tokens)
        if text:
            self.logger.info(f"[Augment/bimanual] {text[:120]}...")
        return sanitize_augmented_desc(text)

    @torch.no_grad()
    def generate_bimanual_trajectories(
        self,
        image_path: str,
        left_instr: str,
        right_instr: str,
        task_description: str = "",
        num_points: int = 8,
        augmented_desc: Optional[str] = None,
    ) -> Dict[str, List]:
        with Image.open(image_path) as img:
            width, height = img.size
        task_summary = task_description or format_bimanual_instruction(left_instr, right_instr)
        text_prompt = build_hand_trajectory_vl_prompt(
            instruction=task_summary,
            image_width=width,
            image_height=height,
            num_points=num_points,
            hand="bimanual",
            left_instruction=left_instr,
            right_instruction=right_instr,
            augmented_desc=augmented_desc,
        )
        max_tokens = self.gen_cfg.get("trajectory_max_new_tokens", 512)
        raw = self._vl_generate(image_path, text_prompt, max_tokens)
        parsed = parse_hand_trajectory_json(raw, width, height, default_hand="right")

        def _needs_retry() -> bool:
            left_ok = not _hand_active(left_instr) or parsed.get("left_hand")
            right_ok = not _hand_active(right_instr) or parsed.get("right_hand")
            return not (left_ok and right_ok)

        if _needs_retry():
            self.logger.warning("[VisAug/bimanual] incomplete trajectories, retrying once")
            retry_prompt = (
                text_prompt
                + f"\nOutput valid JSON only. Use point_2d with integers 0-{QWEN3_VL_COORD_MAX}, "
                "not pixel coordinates. Include both left_hand and right_hand arrays."
            )
            raw = self._vl_generate(image_path, retry_prompt, max_tokens)
            parsed = parse_hand_trajectory_json(raw, width, height, default_hand="right")

        if not _hand_active(left_instr):
            parsed["left_hand"] = []
        if not _hand_active(right_instr):
            parsed["right_hand"] = []
        return {
            "coordinate_space": "qwen3_rel_0_1000",
            "image_width": width,
            "image_height": height,
            "left_hand": [list(p) for p in parsed.get("left_hand", [])],
            "right_hand": [list(p) for p in parsed.get("right_hand", [])],
        }

    def save_scene_aug_for_task(
        self,
        image_path: str,
        trajectories: Dict,
        scene_aug_dir: str,
        task_id: str,
    ) -> str:
        out_path = scene_aug_path_for_task(scene_aug_dir, image_path, task_id)
        traj_draw = {
            "left_hand": [tuple(p) for p in trajectories.get("left_hand", [])],
            "right_hand": [tuple(p) for p in trajectories.get("right_hand", [])],
        }
        with Image.open(image_path) as img:
            vis = draw_hand_trajectories_on_image(img, traj_draw)
            vis.save(out_path, quality=95)
        self.logger.info(f"[VisAug] saved {out_path}")
        return str(out_path.resolve())

    def prepare_row(
        self,
        row: Dict,
        *,
        seed: int,
        use_augmentation: bool = True,
        use_visual_augmentation: bool = False,
        scene_aug_dir: str = "database/wm-h/video/scene_aug",
        trajectory_num_points: int = 8,
        augment_output_language: str = "zh",
        synth_manifest_path: str = "",
        include_retry: bool = False,
    ) -> Optional[Dict]:
        task_id = row.get("task_id", "")
        if not task_id:
            return None

        try:
            image_str = resolve_input_image(row)
        except FileNotFoundError as exc:
            self.logger.error(str(exc))
            return None

        hand_mode = infer_hand_mode(row)
        left_instr = (row.get("left_instruction") or "None").strip()
        right_instr = (row.get("right_instruction") or "None").strip()
        if not _hand_active(left_instr):
            left_instr = "None"
        if not _hand_active(right_instr):
            right_instr = "None"
        instruction = row.get("instruction") or format_bimanual_instruction(left_instr, right_instr)
        task_description = (row.get("task_description") or "").strip()
        shared_object = shared_object_label(row)
        visible_objects = scene_object_labels(row)
        coordination_type = (row.get("coordination_type") or "").strip()

        augmented_desc = ""
        if use_augmentation:
            if hand_mode == "bimanual":
                augmented_desc = self.augment_bimanual(
                    image_str,
                    left_instr,
                    right_instr,
                    task_description,
                    shared_object,
                    coordination_type=coordination_type,
                    visible_objects=visible_objects,
                    output_language=augment_output_language,
                    include_retry=include_retry,
                )
            else:
                active_instr = left_instr if hand_mode == "left" else right_instr
                augmented_desc = self.augment_prompt_with_qwen(
                    image_str,
                    active_instr,
                    hand=hand_mode,
                    task_description=task_description,
                    visible_objects=visible_objects,
                    output_language=augment_output_language,
                    include_retry=include_retry,
                )

        traj_scene_desc = augmented_desc
        if use_visual_augmentation and not traj_scene_desc.strip():
            if hand_mode == "bimanual":
                traj_scene_desc = self.augment_bimanual(
                    image_str,
                    left_instr,
                    right_instr,
                    task_description,
                    shared_object,
                    coordination_type=coordination_type,
                    visible_objects=visible_objects,
                    output_language=augment_output_language,
                    include_retry=include_retry,
                )
            else:
                active_instr = left_instr if hand_mode == "left" else right_instr
                traj_scene_desc = self.augment_prompt_with_qwen(
                    image_str,
                    active_instr,
                    hand=hand_mode,
                    task_description=task_description,
                    visible_objects=visible_objects,
                    output_language=augment_output_language,
                    include_retry=include_retry,
                )

        hand_trajectories: Dict = {}
        scene_aug_path = ""
        if use_visual_augmentation:
            if hand_mode == "bimanual":
                hand_trajectories = self.generate_bimanual_trajectories(
                    image_str,
                    left_instr,
                    right_instr,
                    task_description=task_description,
                    num_points=trajectory_num_points,
                    augmented_desc=traj_scene_desc or None,
                )
            else:
                active_instr = left_instr if hand_mode == "left" else right_instr
                hand_trajectories = self.generate_hand_trajectories_with_qwen(
                    image_str,
                    active_instr,
                    hand=hand_mode,
                    num_points=trajectory_num_points,
                    augmented_desc=traj_scene_desc or None,
                )
            scene_aug_path = self.save_scene_aug_for_task(
                image_str,
                hand_trajectories,
                scene_aug_dir,
                task_id,
            )

        final_augmented = sanitize_augmented_desc(augmented_desc)
        if hand_mode == "bimanual":
            video_prompt = build_desk_synth_bimanual_video_prompt(
                left_instr,
                right_instr,
                task_description=task_description,
                augmented_desc=final_augmented or None,
                use_visual_trajectory=bool(scene_aug_path),
            )
        else:
            active_instr = left_instr if hand_mode == "left" else right_instr
            video_prompt = build_desk_synth_video_prompt(
                active_instr,
                augmented_desc=final_augmented or None,
                hand=hand_mode,
                use_visual_trajectory=bool(scene_aug_path),
            )

        out = dict(row)
        out.update({
            "synth_manifest": synth_manifest_path,
            "synth_run_dir": row.get("run_dir", ""),
            "source_image_path": row.get("image_path") or row.get("desktop_image_path", ""),
            "edited_image_path": row.get("edited_image_path", ""),
            "image_path": image_str,
            "instruction": instruction,
            "task_description": task_description,
            "left_instruction": left_instr,
            "right_instruction": right_instr,
            "shared_object": shared_object,
            "visible_objects": row.get("visible_objects", []),
            "coordination_type": row.get("coordination_type", ""),
            "hand": hand_mode,
            "augmented_text": final_augmented,
            "video_prompt": video_prompt,
            "use_visual_aug": bool(scene_aug_path),
            "scene_aug_path": scene_aug_path,
            "hand_trajectories": hand_trajectories,
            "include_retry": include_retry,
            "seed": seed,
        })
        return out


def _worker_prepare(args: Tuple) -> int:
    jobs, gpu_id, config, output_manifest, checkpoint_info = args
    device = isolate_cuda_device(gpu_id)
    print(f"[GPU {gpu_id}] Desk synth video preparer on {device}, {len(jobs)} rows")

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    task_cfg = config.get("task", {})
    settings_cfg = config.get("settings", {})
    gen_cfg = config.get("generation", {})
    checkpoint_cfg = config.get("checkpoint", {})

    seed = task_cfg.get("seed", 0)
    checkpoint_dir = checkpoint_info.get("checkpoint_dir", "database/log")
    checkpoint_name = checkpoint_cfg.get("checkpoint_name", "processed_desk_synth_video_prompts.txt")
    ckpt = CheckpointManager(checkpoint_dir, checkpoint_name=checkpoint_name)
    processed_ids = ckpt.load_processed_instructions() if checkpoint_info.get("enabled", True) else set()

    preparer = DeskSynthVideoPromptPreparer(
        qwen_vl_model_path=model_cfg["qwen_vl_model_path"],
        device=device,
        gen_cfg=gen_cfg,
        image_max_side=settings_cfg.get("image_max_side", 1024),
    )

    manifest_path = Path(output_manifest)
    success = 0
    with open(manifest_path, "a", encoding="utf-8") as out_f:
        for job in jobs:
            row = job["row"]
            synth_manifest = job["synth_manifest"]
            global_idx = job["global_idx"]
            task_id = row.get("task_id", "")
            ckpt_id = make_checkpoint_id(synth_manifest, task_id)
            if processed_ids is not None and ckpt_id in processed_ids:
                continue

            print(f"[GPU {gpu_id}] {task_id} ({Path(synth_manifest).parent.name})")
            entry = preparer.prepare_row(
                row,
                seed=seed + global_idx,
                use_augmentation=settings_cfg.get("aug", True),
                use_visual_augmentation=settings_cfg.get("visual_aug", False),
                scene_aug_dir=data_cfg.get(
                    "scene_aug_dir",
                    "database/wm-h/video/scene_aug",
                ),
                trajectory_num_points=settings_cfg.get("trajectory_num_points", 8),
                augment_output_language=settings_cfg.get("augment_output_language", "zh"),
                synth_manifest_path=synth_manifest,
                include_retry=settings_cfg.get("include_retry", False),
            )
            if not entry:
                continue

            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())
            if checkpoint_info.get("enabled", True):
                ckpt.mark_as_processed(ckpt_id, gpu_id)
                processed_ids.add(ckpt_id)
            success += 1

    print(f"[GPU {gpu_id}] Wrote {success} video manifest entries")
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Augment wm-h manifest with video prompts"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/video.yaml",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Single input synth manifest.jsonl (overrides default all-manifest scan)",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Only process the newest manifest.jsonl (default: all manifests)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "instr_first"],
        default="instr_first",
        help="Which synth mode(s) to scan (default: instr_first)",
    )
    parser.add_argument("--limit", type=int, default=-1, help="Process first N rows only")
    parser.add_argument("--reset-checkpoint", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    task_cfg = config.get("task", {})
    parallel_cfg = config.get("parallel", {})
    checkpoint_cfg = config.get("checkpoint", {})

    manifest_paths: List[Path] = []
    if args.manifest:
        manifest_paths = [Path(args.manifest)]
    elif data_cfg.get("synth_manifest"):
        manifest_paths = [Path(data_cfg["synth_manifest"])]
    elif args.latest:
        latest = find_latest_synth_manifest(args.mode)
        if latest:
            manifest_paths = [latest]
            print(f"Using latest synth manifest: {latest}")
    else:
        manifest_paths = find_all_synth_manifests(args.mode)
        if manifest_paths:
            print(f"Found {len(manifest_paths)} synth manifest(s):")
            for mp in manifest_paths:
                print(f"  - {mp}")

    if not manifest_paths:
        raise ValueError(
            "No synth manifest found. Run instr_first first, "
            "or pass --manifest PATH"
        )

    jobs = load_synth_jobs(manifest_paths, limit=args.limit)
    if not jobs:
        print("No rows in selected synth manifest(s)")
        return
    print(f"Loaded {len(jobs)} synth rows from {len(manifest_paths)} manifest(s)")

    manifest_dir = Path(
        data_cfg.get("manifest_dir", "database/wm-h/video/manifests")
    )
    manifest_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_manifest = manifest_dir / f"video_manifest_{timestamp}.jsonl"
    output_manifest.touch()
    print(f"Output: {output_manifest}")

    checkpoint_dir = checkpoint_cfg.get("dir", "database/log")
    checkpoint_name = checkpoint_cfg.get(
        "checkpoint_name", "processed_desk_synth_video_prompts.txt"
    )
    ckpt = CheckpointManager(checkpoint_dir, checkpoint_name=checkpoint_name)
    if args.reset_checkpoint:
        ckpt.reset()

    checkpoint_info = {
        "checkpoint_dir": checkpoint_dir,
        "enabled": checkpoint_cfg.get("enable", True),
    }

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible:
        visible_gpus = [int(x) for x in cuda_visible.split(",") if x.strip()]
    else:
        visible_gpus = list(range(torch.cuda.device_count()))
    if not visible_gpus:
        raise RuntimeError("No GPU available")

    enable_parallel = parallel_cfg.get("enable", True) and len(visible_gpus) > 1
    if not enable_parallel:
        n = _worker_prepare(
            (jobs, 0, config, str(output_manifest), checkpoint_info)
        )
        print(f"Done: {n} entries -> {output_manifest}")
        return

    jobs_per_gpu: List[List[Dict]] = [[] for _ in range(len(visible_gpus))]
    for idx, job in enumerate(jobs):
        jobs_per_gpu[idx % len(visible_gpus)].append(job)

    tasks = [
        (jobs_per_gpu[g], g, config, str(output_manifest), checkpoint_info)
        for g in range(len(visible_gpus))
        if jobs_per_gpu[g]
    ]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(tasks)) as pool:
        results = pool.map(_worker_prepare, tasks)
    print(f"Total: {sum(results)} entries -> {output_manifest}")


if __name__ == "__main__":
    main()

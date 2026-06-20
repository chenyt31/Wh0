"""
Stage 2: Generate Wan I2V videos from WM-H video manifests.

Handles mixed hand modes (left / right / bimanual) in one run.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import torch
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "DiffSynth-Studio"))

os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "sage_attention")

from diffsynth.utils.data import save_video
from PIL import Image, ImageOps

from wm_h.video.generator import ManifestVideoGenerator, read_manifest
from wm_h.video.cuda import isolate_cuda_device
from wm_h.video.common import (
    CheckpointManager,
    DEFAULT_NEGATIVE_PROMPT,
    expand_manifest_rollouts,
    load_config,
    resolve_manifest_image_path,
    resolve_rollout_generation_params,
)
from wm_h.video_prompts import build_desk_synth_task_json

def find_latest_video_manifest() -> Optional[Path]:
    base = ROOT / "database/wm-h/video/manifests"
    if not base.is_dir():
        return None
    files = list(base.glob("video_manifest_*.jsonl"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


class DeskSynthVideoGenerator(ManifestVideoGenerator):
    """Wan I2V for desk synth manifests — mirror only left-hand single-arm tasks."""

    def process_manifest_entry(
        self,
        entry: Dict,
        videos_dir: Path,
        tasks_dir: Path,
        video_steps: int,
        video_height: int,
        video_width: int,
        video_cfg_scale: float,
        video_sigma_shift: float,
        video_switch_dit_boundary: float,
        num_frames: int,
        fps: int,
        negative_prompt: str,
        input_prep_width: int = 0,
        input_prep_height: int = 0,
    ) -> Optional[Path]:
        task_id = entry["task_id"]
        image_path_str = resolve_manifest_image_path(entry)
        image_path = Path(image_path_str)
        video_prompt = entry["video_prompt"]
        hand = entry.get("hand", "right")
        use_visual_aug = entry.get("use_visual_aug", False)
        gen_params = resolve_rollout_generation_params(
            entry,
            {
                "video_sigma_shift": video_sigma_shift,
                "video_cfg_scale": video_cfg_scale,
            },
        )

        if not image_path.exists():
            self.logger.error(f"Image not found: {image_path}")
            return None

        if use_visual_aug:
            self.logger.info(f"Using scene_aug image: {image_path}")

        try:
            input_image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            self.logger.error(f"Failed to open image: {exc}")
            return None

        try:
            video = self.generate_video(
                input_image=input_image,
                prompt=video_prompt,
                negative_prompt=negative_prompt,
                seed=gen_params["seed"],
                num_inference_steps=video_steps,
                height=video_height,
                width=video_width,
                num_frames=num_frames,
                cfg_scale=gen_params["cfg_scale"],
                sigma_shift=gen_params["sigma_shift"],
                switch_DiT_boundary=video_switch_dit_boundary,
                rand_device=gen_params["rand_device"],
                input_prep_width=input_prep_width,
                input_prep_height=input_prep_height,
            )
        except Exception as exc:
            self.logger.error(f"Video generation failed for {task_id}: {exc}")
            return None

        if hand == "left":
            video = [ImageOps.mirror(frame) for frame in video]

        video_path = videos_dir / f"{task_id}.mp4"
        save_video(video, str(video_path), fps=fps, quality=5)
        self.logger.info(f"Saved {video_path} (hand={hand})")

        task_info = build_desk_synth_task_json(
            {**entry, "task_id": task_id},
            video_path=str(video_path.resolve()),
        )
        task_json = tasks_dir / f"{task_id}.json"
        with open(task_json, "w", encoding="utf-8") as f:
            json.dump(task_info, f, ensure_ascii=False, indent=2)

        return video_path


def _worker_generate(args: Tuple) -> int:
    entries, gpu_id, config, output_dirs, checkpoint_info, manifest_path = args
    device = isolate_cuda_device(gpu_id)
    print(f"[GPU {gpu_id}] Desk synth video gen on {device}, {len(entries)} entries")

    model_cfg = config.get("model", {})
    video_cfg = config.get("video", {})
    checkpoint_cfg = config.get("checkpoint", {})

    ckpt = CheckpointManager(
        checkpoint_info.get("checkpoint_dir", "database/log"),
        checkpoint_name=checkpoint_cfg.get(
            "video_checkpoint_name", "processed_desk_synth_video_manifest.txt"
        ),
    )
    processed_ids = (
        ckpt.load_processed_instructions()
        if checkpoint_info.get("enabled", True)
        else set()
    )

    gen = DeskSynthVideoGenerator(
        video_model_path=model_cfg["video_model_path"],
        device=device,
        lightx2v_lora_path=model_cfg.get("lightx2v_lora_path"),
        vram_management=model_cfg.get("vram_management") or {},
    )

    videos_dir = Path(output_dirs["videos_dir"])
    tasks_dir = Path(output_dirs["tasks_dir"])
    negative_prompt = config.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)

    success = 0
    for entry in entries:
        task_id = entry.get("task_id", "")
        ckpt_id = CheckpointManager.get_manifest_row_id(manifest_path, task_id)
        if processed_ids is not None and ckpt_id in processed_ids:
            print(f"[GPU {gpu_id}] skip (done): {task_id}")
            continue
        print(f"[GPU {gpu_id}] {task_id} (hand={entry.get('hand', '?')})")
        start_time = time.perf_counter()
        result = gen.process_manifest_entry(
            entry=entry,
            videos_dir=videos_dir,
            tasks_dir=tasks_dir,
            video_steps=video_cfg.get("video_steps", 4),
            video_height=video_cfg.get("video_height", 360),
            video_width=video_cfg.get("video_width", 640),
            video_cfg_scale=video_cfg.get("video_cfg_scale", 1.0),
            video_sigma_shift=video_cfg.get("video_sigma_shift", 5.0),
            video_switch_dit_boundary=video_cfg.get("video_switch_dit_boundary", 0.875),
            num_frames=video_cfg.get("num_frames", 81),
            fps=video_cfg.get("fps", 15),
            negative_prompt=negative_prompt,
            input_prep_width=int(video_cfg.get("input_prep_width", 0) or 0),
            input_prep_height=int(video_cfg.get("input_prep_height", 0) or 0),
        )
        elapsed_sec = time.perf_counter() - start_time
        print(
            f"[GPU {gpu_id}] TIMING task_id={task_id} "
            f"elapsed_sec={elapsed_sec:.3f} status={'ok' if result is not None else 'failed'}",
            flush=True,
        )
        if result is not None:
            success += 1
            if checkpoint_info.get("enabled", True):
                ckpt.mark_as_processed(ckpt_id, gpu_id)
                processed_ids.add(ckpt_id)
    print(f"[GPU {gpu_id}] Success {success}/{len(entries)}")
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: generate videos from WM-H video manifest"
    )
    parser.add_argument("--config", type=str, default="configs/video.yaml")
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--latest", action="store_true", help="Use newest video_manifest_*.jsonl")
    parser.add_argument("--reset-checkpoint", action="store_true")
    parser.add_argument("--rollouts", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.rollouts is not None:
        config.setdefault("task", {})["rollouts_per_task"] = max(1, args.rollouts)

    data_cfg = config.get("data", {})
    task_cfg = config.get("task", {})
    parallel_cfg = config.get("parallel", {})
    checkpoint_cfg = config.get("checkpoint", {})

    manifest = args.manifest or data_cfg.get("video_manifest")
    if not manifest:
        latest = find_latest_video_manifest()
        if latest:
            manifest = str(latest)
            print(f"Using latest video manifest (default): {manifest}")
    if not manifest:
        raise ValueError(
            "No video manifest found. Run video_prompt_preparer.py first, "
            "or pass --manifest PATH"
        )

    entries = read_manifest(manifest, k=task_cfg.get("k", -1))
    if not entries:
        return

    entries, _ = expand_manifest_rollouts(entries, config)

    output_base = data_cfg.get("output_dir", "database/wm-h/video/generated")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_base) / f"run_{timestamp}"
    videos_dir = run_dir / "videos"
    tasks_dir = run_dir / "tasks"
    videos_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {run_dir}")

    checkpoint_dir = checkpoint_cfg.get("dir", "database/log")
    checkpoint_info = {
        "checkpoint_dir": checkpoint_dir,
        "enabled": checkpoint_cfg.get("enable", True),
    }
    ckpt = CheckpointManager(
        checkpoint_dir,
        checkpoint_name=checkpoint_cfg.get(
            "video_checkpoint_name", "processed_desk_synth_video_manifest.txt"
        ),
    )
    if args.reset_checkpoint:
        ckpt.reset()

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible:
        visible_gpus = [int(x) for x in cuda_visible.split(",") if x.strip()]
    else:
        visible_gpus = list(range(torch.cuda.device_count()))
    if not visible_gpus:
        raise RuntimeError("No GPU available")

    enable_parallel = parallel_cfg.get("enable", True) and len(visible_gpus) > 1
    output_dirs = {"videos_dir": str(videos_dir), "tasks_dir": str(tasks_dir)}

    manifest_resolved = str(Path(manifest).resolve())

    if not enable_parallel:
        n = _worker_generate(
            (entries, 0, config, output_dirs, checkpoint_info, manifest_resolved)
        )
        print(f"Done: {n}/{len(entries)} -> {run_dir}")
        return

    jobs_per_gpu: List[List[Dict]] = [[] for _ in range(len(visible_gpus))]
    for idx, entry in enumerate(entries):
        jobs_per_gpu[idx % len(visible_gpus)].append(entry)

    tasks = [
        (jobs_per_gpu[g], g, config, output_dirs, checkpoint_info, manifest_resolved)
        for g in range(len(visible_gpus))
        if jobs_per_gpu[g]
    ]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(tasks)) as pool:
        results = pool.map(_worker_generate, tasks)
    print(f"Total: {sum(results)}/{len(entries)} -> {run_dir}")


if __name__ == "__main__":
    main()

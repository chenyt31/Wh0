"""Shared helpers for streaming instr-first synthesis into video generation."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from wm_h.video.common import DEFAULT_NEGATIVE_PROMPT
from wm_h.logging_utils import get_cli_logger

cli = get_cli_logger()


def safe_label(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def setup_run(
    base: Path,
    gpu_id: int,
    *,
    worker_label: str = "",
    run_label: str = "",
) -> Tuple[Path, Path, Path, Path, Path, Path]:
    tag = safe_label(run_label) if run_label else datetime.now().strftime("run_%Y%m%d_%H%M%S")
    suffix = (
        safe_label(worker_label)
        if worker_label
        else (f"gpu{gpu_id}" if gpu_id >= 0 else "single")
    )
    run_dir = base / "instr_first" / "streaming_runs" / f"{tag}_{suffix}"
    analysis_dir = run_dir / "analysis"
    edited_dir = run_dir / "edited_images"
    viz_dir = run_dir / "viz"
    videos_dir = run_dir / "videos"
    tasks_dir = run_dir / "tasks"
    for d in (analysis_dir, edited_dir, viz_dir, videos_dir, tasks_dir):
        d.mkdir(parents=True, exist_ok=True)
    return run_dir, analysis_dir, edited_dir, viz_dir, videos_dir, tasks_dir


def chunks(items: List[Any], size: int) -> Iterable[List[Any]]:
    step = max(1, size)
    for start in range(0, len(items), step):
        yield items[start : start + step]


def split_even_chunks(items: List[Any], n: int) -> List[List[Any]]:
    if n <= 0:
        return [items]
    k = len(items)
    if k == 0:
        return [[] for _ in range(n)]
    base = k // n
    rem = k % n
    out: List[List[Any]] = []
    idx = 0
    for i in range(n):
        size = base + (1 if i < rem else 0)
        out.append(items[idx : idx + size])
        idx += size
    return out


def video_preparer(video_config: Dict[str, Any], device: str) -> Any:
    import torch
    from wm_h.video_prompt_preparer import DeskSynthVideoPromptPreparer

    model_cfg = video_config.get("model", {})
    settings_cfg = video_config.get("settings", {})
    return DeskSynthVideoPromptPreparer(
        qwen_vl_model_path=model_cfg["qwen_vl_model_path"],
        device=device,
        torch_dtype=torch.bfloat16,
        gen_cfg=video_config.get("generation", {}),
        image_max_side=int(settings_cfg.get("image_max_side", 1024)),
        defer_vl_load=True,
    )


def video_generator(video_config: Dict[str, Any], device: str) -> Any:
    from wm_h.run_video_generator import DeskSynthVideoGenerator

    model_cfg = video_config.get("model", {})
    return DeskSynthVideoGenerator(
        video_model_path=model_cfg["video_model_path"],
        device=device,
        lightx2v_lora_path=model_cfg.get("lightx2v_lora_path"),
        vram_management=model_cfg.get("vram_management") or {},
    )


def prepare_video_rows(
    rows: List[Dict[str, Any]],
    *,
    preparer: Any,
    video_config: Dict[str, Any],
    synth_manifest: Path,
    seed_offset: int,
) -> List[Dict[str, Any]]:
    data_cfg = video_config.get("data", {})
    settings_cfg = video_config.get("settings", {})
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        entry = preparer.prepare_row(
            row,
            seed=seed_offset + idx,
            use_augmentation=bool(settings_cfg.get("aug", True)),
            use_visual_augmentation=bool(settings_cfg.get("visual_aug", False)),
            scene_aug_dir=data_cfg.get(
                "scene_aug_dir",
                "database/wm-h/video/scene_aug",
            ),
            trajectory_num_points=int(settings_cfg.get("trajectory_num_points", 8)),
            augment_output_language=settings_cfg.get("augment_output_language", "en"),
            synth_manifest_path=str(synth_manifest.resolve()),
            include_retry=bool(settings_cfg.get("include_retry", False)),
        )
        if entry is not None:
            out.append(entry)
    return out


def generate_video_rows(
    entries: List[Dict[str, Any]],
    *,
    generator: Any,
    video_config: Dict[str, Any],
    videos_dir: Path,
    tasks_dir: Path,
) -> int:
    video_cfg = video_config.get("video", {})
    negative_prompt = video_config.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    ok = 0
    for entry in entries:
        start = time.perf_counter()
        result = generator.process_manifest_entry(
            entry=entry,
            videos_dir=videos_dir,
            tasks_dir=tasks_dir,
            video_steps=video_cfg.get("video_steps", 4),
            video_height=video_cfg.get("video_height", 368),
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
        elapsed = time.perf_counter() - start
        cli.info(
            "TIMING streaming_video task_id=%s elapsed_sec=%.3f status=%s",
            entry.get("task_id", ""),
            elapsed,
            "ok" if result is not None else "failed",
        )
        ok += int(result is not None)
    return ok

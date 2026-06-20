#!/usr/bin/env python3
"""Single-machine producer/consumer runner for instr_first.

Producer GPUs run slot assembly + image edit and enqueue Wan jobs.
Consumer GPUs keep Wan resident and pull jobs from a shared multiprocessing queue.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import uuid
from queue import Empty
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "DiffSynth-Studio"))

from PIL import Image, ImageOps

from wm_h.video.cuda import isolate_cuda_device, visible_cuda_devices
from wm_h.video.common import (
    DEFAULT_NEGATIVE_PROMPT,
    list_images,
    load_config,
    resolve_manifest_image_path,
    resolve_rollout_generation_params,
)
from wm_h.logging_utils import get_cli_logger, setup_logging
from wm_h.streaming import (
    chunks,
    prepare_video_rows,
    setup_run,
    split_even_chunks,
    video_preparer as _video_preparer,
)
from wm_h.run_video_generator import DeskSynthVideoGenerator
from wm_h.video_prompt_preparer import read_synth_manifest
from wm_h.video_prompts import build_desk_synth_task_json

cli = get_cli_logger()


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _fmt_stats(values: List[float]) -> str:
    if not values:
        return "n=0"
    return (
        f"n={len(values)} avg={_mean(values):.3f}s "
        f"min={min(values):.3f}s max={max(values):.3f}s"
    )


def _video_duration_sec(video_config: Dict[str, Any]) -> float:
    video_cfg = video_config.get("video", {})
    fps = float(video_cfg.get("fps", 15) or 15)
    if fps <= 0:
        fps = 15.0
    return float(video_cfg.get("num_frames", 81) or 81) / fps


def _efficiency_stats(
    *,
    wall_sec: float,
    videos: int,
    active_gpus: int,
    video_duration_sec: float,
) -> Dict[str, float]:
    gpu_hours = max(0.0, wall_sec) * max(1, active_gpus) / 3600.0
    video_hours = max(0, videos) * max(0.0, video_duration_sec) / 3600.0
    return {
        "avg_sec_per_video": wall_sec / videos if videos else 0.0,
        "gpu_hours": gpu_hours,
        "video_hours": video_hours,
        "video_hours_per_gpu_hour": video_hours / gpu_hours if gpu_hours else 0.0,
        "gpu_hours_per_video_hour": gpu_hours / video_hours if video_hours else 0.0,
        "gpu_sec_per_video": wall_sec * max(1, active_gpus) / videos if videos else 0.0,
    }


def _parse_gpu_list(spec: str, visible: Sequence[str]) -> List[int]:
    """Parse physical GPU ids or visible-list indexes into isolate_cuda_device indexes."""
    if not spec.strip():
        return []
    out: List[int] = []
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        if token in visible:
            idx = list(visible).index(token)
        else:
            idx = int(token)
            if idx < 0 or idx >= len(visible):
                raise ValueError(
                    f"GPU {token!r} is neither a visible physical id nor a valid "
                    f"visible index; visible={','.join(visible)}"
                )
        if idx in out:
            raise ValueError(f"duplicate GPU in list: {token}")
        out.append(idx)
    return out


def _gpu_label(index: int, visible: Sequence[str]) -> str:
    if 0 <= index < len(visible):
        return str(visible[index])
    return str(index)


def _editor_timing(editor: Any) -> Dict[str, float]:
    if editor is None or not hasattr(editor, "timing_snapshot"):
        return {}
    timing = editor.timing_snapshot()
    return {
        "producer_image_edit_load_elapsed": float(timing.get("load_elapsed_sec", 0.0)),
        "producer_image_edit_infer_elapsed": float(timing.get("infer_elapsed_sec", 0.0)),
        "producer_image_edit_calls": float(timing.get("edit_calls", 0.0)),
    }


def _editor_timing_delta(editor: Any, before: Dict[str, float]) -> Dict[str, float]:
    after = _editor_timing(editor)
    return {key: max(0.0, after.get(key, 0.0) - before.get(key, 0.0)) for key in after}


def _append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _job_output_dirs(run_dir: Path) -> Dict[str, str]:
    videos_dir = run_dir / "videos"
    tasks_dir = run_dir / "tasks"
    frames_dir = run_dir / "frames"
    for path in (videos_dir, tasks_dir, frames_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "videos_dir": str(videos_dir),
        "tasks_dir": str(tasks_dir),
        "frames_dir": str(frames_dir),
    }


def _enqueue_video_rows(
    queue,
    rows: List[Dict[str, Any]],
    *,
    output_dirs: Dict[str, str],
    need_last_frame: bool = False,
    timing: Optional[Dict[str, Any]] = None,
    synth_manifest: Optional[Path] = None,
    video_manifest: Optional[Path] = None,
    seed_offset: int = 0,
) -> List[str]:
    job_ids: List[str] = []
    for idx, row in enumerate(rows):
        job_id = uuid.uuid4().hex
        job_ids.append(job_id)
        timing_payload = dict(timing or {})
        timing_payload["enqueue_perf"] = time.perf_counter()
        queue.put(
            {
                "job_id": job_id,
                "entry": row,
                "output_dirs": output_dirs,
                "need_last_frame": need_last_frame,
                "timing": timing_payload,
                "seed": seed_offset + idx,
                "synth_manifest": str(synth_manifest.resolve()) if synth_manifest else "",
                "video_manifest": str(video_manifest.resolve()) if video_manifest else "",
            }
        )
    return job_ids


def _prepare_consumer_video_entry(
    preparer: Any,
    row: Dict[str, Any],
    *,
    video_config: Dict[str, Any],
    seed: int,
    synth_manifest: str = "",
) -> Dict[str, Any]:
    if row.get("video_prompt") and row.get("image_path"):
        return row
    data_cfg = video_config.get("data", {})
    settings_cfg = video_config.get("settings", {})
    entry = preparer.prepare_row(
        row,
        seed=seed,
        use_augmentation=bool(settings_cfg.get("aug", True)),
        use_visual_augmentation=bool(settings_cfg.get("visual_aug", False)),
        scene_aug_dir=data_cfg.get(
            "scene_aug_dir",
            "database/wm-h/video/scene_aug",
        ),
        trajectory_num_points=int(settings_cfg.get("trajectory_num_points", 8)),
        augment_output_language=settings_cfg.get("augment_output_language", "en"),
        synth_manifest_path=synth_manifest,
        include_retry=bool(settings_cfg.get("include_retry", False)),
    )
    if entry is None:
        raise RuntimeError(f"video prompt preparation failed for {row.get('task_id', '')}")
    return entry


def _wait_for_job(result_store, job_id: str, *, poll_sec: float = 1.0) -> Dict[str, Any]:
    while True:
        result = result_store.pop(job_id, None)
        if result is not None:
            return result
        time.sleep(poll_sec)


def _generate_one_video(
    generator: DeskSynthVideoGenerator,
    entry: Dict[str, Any],
    *,
    video_config: Dict[str, Any],
    output_dirs: Dict[str, str],
    need_last_frame: bool = False,
) -> Dict[str, Any]:
    from diffsynth.utils.data import save_video

    video_cfg = video_config.get("video", {})
    negative_prompt = video_config.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    task_id = entry["task_id"]
    image_path = Path(resolve_manifest_image_path(entry))
    hand = entry.get("hand", "right")
    gen_params = resolve_rollout_generation_params(
        entry,
        {
            "video_sigma_shift": video_cfg.get("video_sigma_shift", 5.0),
            "video_cfg_scale": video_cfg.get("video_cfg_scale", 1.0),
        },
    )
    input_image = Image.open(image_path).convert("RGB")
    frames = generator.generate_video(
        input_image=input_image,
        prompt=entry["video_prompt"],
        negative_prompt=negative_prompt,
        seed=gen_params["seed"],
        num_inference_steps=video_cfg.get("video_steps", 4),
        height=video_cfg.get("video_height", 368),
        width=video_cfg.get("video_width", 640),
        num_frames=video_cfg.get("num_frames", 81),
        cfg_scale=gen_params["cfg_scale"],
        sigma_shift=gen_params["sigma_shift"],
        switch_DiT_boundary=video_cfg.get("video_switch_dit_boundary", 0.875),
        rand_device=gen_params["rand_device"],
        input_prep_width=int(video_cfg.get("input_prep_width", 0) or 0),
        input_prep_height=int(video_cfg.get("input_prep_height", 0) or 0),
    )
    if hand == "left":
        frames = [ImageOps.mirror(frame) for frame in frames]

    videos_dir = Path(output_dirs["videos_dir"])
    tasks_dir = Path(output_dirs["tasks_dir"])
    frames_dir = Path(output_dirs["frames_dir"])
    video_path = videos_dir / f"{task_id}.mp4"
    save_video(frames, str(video_path), fps=video_cfg.get("fps", 15), quality=5)
    task_info = build_desk_synth_task_json(
        {**entry, "task_id": task_id},
        video_path=str(video_path.resolve()),
    )
    (tasks_dir / f"{task_id}.json").write_text(
        json.dumps(task_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result = {
        "task_id": task_id,
        "video_path": str(video_path.resolve()),
        "status": "ok",
    }
    if need_last_frame:
        last_frame_path = frames_dir / f"{task_id}_last_frame.jpg"
        frames[-1].save(last_frame_path, quality=95)
        result["last_frame_path"] = str(last_frame_path.resolve())
    return result


def _consumer_worker(payload: Dict[str, Any]) -> None:
    gpu_id = int(payload["gpu_id"])
    device = isolate_cuda_device(gpu_id, payload["visible_gpus"])
    setup_logging(verbose=bool(payload.get("verbose", False)))
    video_config = load_config(payload["video_config"])
    consumer_aug_config = dict(video_config)
    consumer_aug_config["generation"] = dict(video_config.get("generation", {}))
    consumer_aug_config["model"] = dict(video_config.get("model", {}))
    settings_cfg = video_config.get("settings", {})
    consumer_aug_model = str(settings_cfg.get("consumer_augment_model_path") or "").strip()
    if consumer_aug_model:
        consumer_aug_config["model"]["qwen_vl_model_path"] = consumer_aug_model
    augment_gpu_util = settings_cfg.get(
        "consumer_augment_gpu_memory_utilization",
        settings_cfg.get("augment_gpu_memory_utilization", 0.10),
    )
    consumer_aug_config["generation"]["gpu_memory_utilization"] = float(augment_gpu_util)
    consumer_aug_config["generation"]["max_num_seqs"] = int(
        settings_cfg.get("consumer_augment_max_num_seqs", 1)
    )
    consumer_aug_config["generation"]["max_num_batched_tokens"] = int(
        settings_cfg.get("consumer_augment_max_num_batched_tokens", 4096)
    )
    consumer_aug_config["generation"]["use_v1"] = bool(
        settings_cfg.get("consumer_augment_use_v1", False)
    )
    consumer_aug_quant = settings_cfg.get("consumer_augment_quantization", None)
    if consumer_aug_quant is not None:
        if str(consumer_aug_quant).strip():
            consumer_aug_config["generation"]["quantization"] = consumer_aug_quant
        else:
            consumer_aug_config["generation"].pop("quantization", None)
    consumer_aug_config["settings"] = dict(settings_cfg)
    consumer_aug_config["settings"]["image_max_side"] = int(
        settings_cfg.get("consumer_augment_image_max_side", 512)
    )
    model_cfg = video_config.get("model", {})
    queue = payload["queue"]
    results = payload["results"]
    result_store = payload.get("result_store")
    manifest_lock = payload.get("manifest_lock")
    load_t0 = time.perf_counter()
    generator = DeskSynthVideoGenerator(
        video_model_path=model_cfg["video_model_path"],
        device=device,
        lightx2v_lora_path=model_cfg.get("lightx2v_lora_path"),
    )
    load_elapsed = time.perf_counter() - load_t0
    preparer = _video_preparer(consumer_aug_config, device)
    results.put(
        {
            "type": "consumer_ready",
            "gpu_id": gpu_id,
            "wan_load_elapsed_sec": load_elapsed,
        }
    )
    while True:
        job = queue.get()
        if job is None:
            break
        job_id = job["job_id"]
        entry = job["entry"]
        timing = dict(job.get("timing") or {})
        dequeue_perf = time.perf_counter()
        try:
            augment_t0 = time.perf_counter()
            entry = _prepare_consumer_video_entry(
                preparer,
                entry,
                video_config=consumer_aug_config,
                seed=int(job.get("seed", 0)),
                synth_manifest=str(job.get("synth_manifest") or ""),
            )
            augment_elapsed = time.perf_counter() - augment_t0
            video_manifest = str(job.get("video_manifest") or "")
            if video_manifest:
                if manifest_lock is not None:
                    with manifest_lock:
                        _append_jsonl(Path(video_manifest), [entry])
                else:
                    _append_jsonl(Path(video_manifest), [entry])
            t0 = time.perf_counter()
            result = _generate_one_video(
                generator,
                entry,
                video_config=video_config,
                output_dirs=job["output_dirs"],
                need_last_frame=bool(job.get("need_last_frame", False)),
            )
            done_perf = time.perf_counter()
            result["type"] = "video_result"
            result["elapsed_sec"] = done_perf - t0
            result["wan_elapsed_sec"] = result["elapsed_sec"]
            result["consumer_augment_elapsed_sec"] = augment_elapsed
            result["video_entry"] = entry
            if timing.get("enqueue_perf") is not None:
                result["queue_wait_sec"] = dequeue_perf - float(timing["enqueue_perf"])
            if timing.get("pipeline_start_perf") is not None:
                pipeline_elapsed = done_perf - float(timing["pipeline_start_perf"])
                image_edit_load = float(
                    timing.get("producer_image_edit_load_elapsed", 0.0)
                )
                result["pipeline_elapsed_sec"] = pipeline_elapsed
                result["pipeline_no_model_load_sec"] = max(
                    0.0, pipeline_elapsed - image_edit_load
                )
            if timing.get("producer_elapsed_before_enqueue") is not None:
                result["producer_elapsed_before_enqueue_sec"] = float(
                    timing["producer_elapsed_before_enqueue"]
                )
            for key, value in timing.items():
                if key.startswith("producer_") and key.endswith("_elapsed"):
                    result[f"{key}_sec"] = float(value)
            for key in (
                "producer_image_edit_load_elapsed",
                "producer_image_edit_infer_elapsed",
                "producer_image_edit_calls",
            ):
                if key in timing:
                    result[f"{key}_sec"] = float(timing[key])
            if timing.get("pipeline_mode"):
                result["pipeline_mode"] = timing["pipeline_mode"]
            if timing.get("batch_idx") is not None:
                result["batch_idx"] = timing["batch_idx"]
            if timing.get("step") is not None:
                result["step"] = timing["step"]
            result["job_id"] = job_id
            result["gpu_id"] = gpu_id
            if result_store is not None:
                result_store[job_id] = result
            results.put(result)
        except Exception as exc:
            result = {
                "type": "video_result",
                "job_id": job_id,
                "task_id": entry.get("task_id", ""),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "gpu_id": gpu_id,
            }
            if result_store is not None:
                result_store[job_id] = result
            results.put(result)


_chunks = chunks
_split_even_chunks = split_even_chunks
_setup_run = setup_run
_prepare_video_rows = prepare_video_rows


def _instr_first_producer(payload: Dict[str, Any]) -> str:
    gpu_id = int(payload["gpu_id"])
    device = isolate_cuda_device(gpu_id, payload["visible_gpus"])
    setup_logging(verbose=bool(payload.get("verbose", False)))

    from wm_h.instr_first_pipeline import InstrFirstPipeline

    full_config = load_config(payload["config"])
    if_cfg = full_config.get("instr_first", {})
    video_config = load_config(payload["video_config"])
    images = list_images(
        payload.get("image_dir") or if_cfg.get("data", {}).get("image_dir", ""),
        recursive=bool(payload.get("recursive", False)),
    )
    if not images:
        raise FileNotFoundError("instr_first producer found no source images")
    out_base = Path(if_cfg.get("output", {}).get("dir", "database/wm-h/instr_first"))
    run_dir, _, edited_dir, _, _, _ = _setup_run(
        out_base.parent if out_base.name == "instr_first" else out_base,
        gpu_id,
        worker_label=str(payload.get("worker_label") or ""),
        run_label=str(payload.get("run_label") or ""),
    )
    output_dirs = _job_output_dirs(run_dir)
    pipeline = InstrFirstPipeline(payload["config"], device=device)
    pipeline.run_dir = run_dir
    pipeline.edited_dir = edited_dir
    consumer_augment = bool(video_config.get("settings", {}).get("consumer_augment", False))
    preparer = None if consumer_augment else _video_preparer(video_config, device)

    manifest_all = run_dir / "manifest.jsonl"
    video_manifest_all = run_dir / "video_manifest.jsonl"
    batch_size = max(1, int(payload["batch_size"]))
    produced = 0
    target = int(payload["total_instructions"])
    next_instr_idx = 0
    batch_idx = 0
    empty_assemble_attempts = 0
    empty_edit_attempts = 0
    max_empty_assemble_attempts = int(payload.get("max_empty_assemble_attempts", 5))
    max_empty_edit_attempts = int(payload.get("max_empty_edit_attempts", 5))
    while produced < target:
        batch_start = time.perf_counter()
        n = min(batch_size, target - produced)
        assemble_t0 = time.perf_counter()
        assembled = pipeline._assemble_count(n)
        assemble_elapsed = time.perf_counter() - assemble_t0
        if not assembled:
            empty_assemble_attempts += 1
            cli.warning(
                "instr_first assemble produced 0 rows (%d/%d), retrying without writing an empty batch",
                empty_assemble_attempts,
                max_empty_assemble_attempts,
            )
            if empty_assemble_attempts >= max_empty_assemble_attempts:
                raise RuntimeError(
                    "instr_first assemble produced no rows repeatedly; "
                    "check slot DB usage caps and instr_first.database.base_path"
                )
            time.sleep(5)
            continue
        empty_assemble_attempts = 0
        batch_manifest = run_dir / f"manifest.batch{batch_idx:04d}.jsonl"
        editor_before = _editor_timing(pipeline.editor)
        edit_t0 = time.perf_counter()
        pipeline.edit_assembled(assembled, images, batch_manifest, start_idx=next_instr_idx)
        next_instr_idx += len(assembled)
        edit_elapsed = time.perf_counter() - edit_t0
        rows = read_synth_manifest(str(batch_manifest))
        if not rows:
            empty_edit_attempts += 1
            cli.warning(
                "instr_first edit produced 0 valid rows (%d/%d), retrying without enqueue",
                empty_edit_attempts,
                max_empty_edit_attempts,
            )
            if empty_edit_attempts >= max_empty_edit_attempts:
                raise RuntimeError(
                    "instr_first edit produced no valid rows repeatedly; "
                    "check image edit failures and source images"
                )
            batch_idx += 1
            time.sleep(5)
            continue
        empty_edit_attempts = 0
        _append_jsonl(manifest_all, rows)
        prepare_elapsed = 0.0
        enqueue_rows = rows
        if not consumer_augment:
            prepare_t0 = time.perf_counter()
            assert preparer is not None
            enqueue_rows = _prepare_video_rows(
                rows,
                preparer=preparer,
                video_config=video_config,
                synth_manifest=batch_manifest,
                seed_offset=batch_idx * 100000,
            )
            prepare_elapsed = time.perf_counter() - prepare_t0
            _append_jsonl(video_manifest_all, enqueue_rows)
        _enqueue_video_rows(
            payload["queue"],
            enqueue_rows,
            output_dirs=output_dirs,
            timing={
                "pipeline_mode": "instr_first",
                "pipeline_start_perf": batch_start,
                "producer_elapsed_before_enqueue": time.perf_counter() - batch_start,
                "producer_assemble_elapsed": assemble_elapsed,
                "producer_edit_elapsed": edit_elapsed,
                "producer_prepare_elapsed": prepare_elapsed,
                "batch_idx": batch_idx,
                **_editor_timing_delta(pipeline.editor, editor_before),
            },
            synth_manifest=batch_manifest,
            video_manifest=video_manifest_all if consumer_augment else None,
            seed_offset=batch_idx * 100000,
        )
        produced += len(rows)
        batch_idx += 1
    if payload.get("results") is not None:
        payload["results"].put({"type": "producer_done", "gpu_id": gpu_id})
    if payload.get("shutdown_event") is not None:
        payload["shutdown_event"].wait()
    return str(run_dir)


def _run_producers(ctx, target, payloads: List[Dict[str, Any]]) -> List[mp.Process]:
    procs: List[mp.Process] = []
    for payload in payloads:
        proc = ctx.Process(target=target, args=(payload,))
        proc.start()
        procs.append(proc)
    return procs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--video-config", default="configs/video.yaml")
    parser.add_argument("--image-dir", default="")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--total-instructions", type=int, default=8)
    parser.add_argument(
        "--producer-gpus",
        default="0",
        help=(
            "Comma-separated producer GPU physical ids or visible indexes. "
            "Example: 0 or 0,1. Producer runs slot/assembly/VL/image-edit."
        ),
    )
    parser.add_argument(
        "--consumer-gpus",
        default="",
        help=(
            "Comma-separated consumer GPU physical ids or visible indexes. "
            "Default: all visible GPUs not listed in --producer-gpus."
        ),
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=0,
        help=(
            "Unified scheduler backlog. 0 means unbounded so producers never "
            "wait for consumers; positive values apply backpressure."
        ),
    )
    parser.add_argument(
        "--max-video-failures",
        type=int,
        default=1,
        help=(
            "Abort after this many per-video failures. 0 means keep going and "
            "report failures at the end."
        ),
    )
    parser.add_argument(
        "--throughput-log-every",
        type=int,
        default=10,
        help=(
            "Log wall-clock average seconds per successful video every N videos. "
            "The timer starts after all consumers report ready, so Wan checkpoint "
            "loading is excluded."
        ),
    )
    parser.add_argument("--run-label", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    visible = visible_cuda_devices()
    if len(visible) < 2:
        raise RuntimeError("producer/consumer mode needs at least 2 visible GPUs")
    producer_indices = _parse_gpu_list(args.producer_gpus, visible)
    if not producer_indices:
        producer_indices = [0]
    if args.consumer_gpus.strip():
        consumer_indices = _parse_gpu_list(args.consumer_gpus, visible)
    else:
        producer_set = set(producer_indices)
        consumer_indices = [idx for idx in range(len(visible)) if idx not in producer_set]
    overlap = sorted(set(producer_indices).intersection(consumer_indices))
    if overlap:
        raise RuntimeError(
            "producer and consumer GPU sets overlap: "
            + ",".join(_gpu_label(idx, visible) for idx in overlap)
        )
    if not producer_indices or not consumer_indices:
        raise RuntimeError(
            "need at least one producer GPU and one consumer GPU; "
            f"producer={args.producer_gpus!r} consumer={args.consumer_gpus!r}"
        )
    producer_count = len(producer_indices)
    consumer_count = len(consumer_indices)
    active_gpu_count = producer_count + consumer_count
    queue_size = max(0, args.queue_size)
    run_video_config = load_config(args.video_config)
    video_duration_sec = _video_duration_sec(run_video_config)
    cli.info(
        "local PC instr_first producer_gpus=%s consumer_gpus=%s visible=%s queue_size=%d video_duration_sec=%.3f",
        ",".join(_gpu_label(idx, visible) for idx in producer_indices),
        ",".join(_gpu_label(idx, visible) for idx in consumer_indices),
        ",".join(visible),
        queue_size,
        video_duration_sec,
    )

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    queue = ctx.Queue(maxsize=queue_size)
    results = ctx.Queue()
    result_store = manager.dict()
    manifest_lock = ctx.Lock()
    producer_shutdown = ctx.Event()

    consumer_procs = []
    for gpu_idx in consumer_indices:
        consumer_procs.append(
            ctx.Process(
                target=_consumer_worker,
                args=(
                    {
                        "gpu_id": gpu_idx,
                        "visible_gpus": visible,
                        "video_config": args.video_config,
                        "queue": queue,
                        "results": results,
                        "result_store": result_store,
                        "manifest_lock": manifest_lock,
                        "verbose": args.verbose,
                    },
                ),
            )
        )
    for proc in consumer_procs:
        proc.start()
    cli.info("started %d consumer process(es); loading Wan resident models", len(consumer_procs))

    consumer_load_times: List[float] = []
    ready = 0
    all_consumers_ready_perf: Optional[float] = None
    pipeline_start_perf = time.perf_counter()

    per = (args.total_instructions + producer_count - 1) // producer_count
    payloads = [
        {
            "gpu_id": gpu_idx,
            "visible_gpus": visible,
            "config": args.config,
            "video_config": args.video_config,
            "image_dir": args.image_dir,
            "recursive": args.recursive,
            "total_instructions": min(per, args.total_instructions - offset * per),
            "batch_size": args.batch_size,
            "queue": queue,
            "results": results,
            "shutdown_event": producer_shutdown,
            "worker_label": f"producer_gpu{_gpu_label(gpu_idx, visible)}",
            "run_label": args.run_label or datetime.now().strftime("pc_%Y%m%d_%H%M%S"),
            "verbose": args.verbose,
        }
        for offset, gpu_idx in enumerate(producer_indices)
        if args.total_instructions - offset * per > 0
    ]
    producer_target = _instr_first_producer

    producer_procs = _run_producers(ctx, producer_target, payloads)

    done = 0
    fatal_failed = 0
    video_failed = 0
    video_results: List[Dict[str, Any]] = []
    producer_sentinels_sent = False
    producers_done = 0
    while True:
        for proc in consumer_procs:
            if not proc.is_alive() and proc.exitcode not in (None, 0):
                fatal_failed += 1
                cli.error("consumer pid=%s exited with %s", proc.pid, proc.exitcode)
                producer_shutdown.set()
                break
        if fatal_failed:
            break
        for proc in producer_procs:
            if not proc.is_alive() and proc.exitcode not in (None, 0):
                fatal_failed += 1
                cli.error("producer pid=%s exited with %s", proc.pid, proc.exitcode)
                producer_shutdown.set()
                break
        if fatal_failed:
            break
        if not producer_sentinels_sent and producers_done >= len(producer_procs):
            for _ in range(consumer_count):
                queue.put(None)
            producer_sentinels_sent = True

        try:
            result = results.get(timeout=2)
        except Empty:
            if (
                producer_sentinels_sent
                and all(not p.is_alive() for p in consumer_procs)
            ):
                break
            continue
        if result.get("type") == "consumer_ready":
            ready += 1
            load_elapsed = float(result.get("wan_load_elapsed_sec", 0.0))
            consumer_load_times.append(load_elapsed)
            if ready >= consumer_count and all_consumers_ready_perf is None:
                all_consumers_ready_perf = time.perf_counter()
            cli.info(
                "consumer ready gpu=%s wan_load_elapsed_sec=%.3f",
                result.get("gpu_id"),
                load_elapsed,
            )
            continue
        if result.get("type") == "producer_done":
            producers_done += 1
            cli.info(
                "producer done enqueue gpu=%s (%d/%d), keeping producer models resident",
                result.get("gpu_id"),
                producers_done,
                len(producer_procs),
            )
            continue
        if result.get("type") != "video_result":
            cli.warning("unknown result event: %s", result)
            continue
        if result.get("status") == "ok":
            done += 1
            video_results.append(result)
            cli.info(
                "video ok task_id=%s gpu=%s pipeline_elapsed_sec=%.3f pipeline_no_model_load_sec=%.3f wan_elapsed_sec=%.3f queue_wait_sec=%.3f",
                result.get("task_id"),
                result.get("gpu_id"),
                float(result.get("pipeline_elapsed_sec", 0.0)),
                float(result.get("pipeline_no_model_load_sec", result.get("pipeline_elapsed_sec", 0.0))),
                float(result.get("wan_elapsed_sec", result.get("elapsed_sec", 0.0))),
                float(result.get("queue_wait_sec", 0.0)),
            )
            log_every = max(0, int(args.throughput_log_every))
            if all_consumers_ready_perf is not None and log_every and done % log_every == 0:
                wall_elapsed = time.perf_counter() - all_consumers_ready_perf
                eff = _efficiency_stats(
                    wall_sec=wall_elapsed,
                    videos=done,
                    active_gpus=active_gpu_count,
                    video_duration_sec=video_duration_sec,
                )
                cli.info(
                    "EFFICIENCY ready_wall_sec=%.3f videos=%d video_duration_sec=%.3f avg_sec_per_video=%.3f gpu_hours=%.6f video_hours=%.6f video_hours_per_gpu_hour=%.6f gpu_hours_per_video_hour=%.3f gpu_sec_per_video=%.3f",
                    wall_elapsed,
                    done,
                    video_duration_sec,
                    eff["avg_sec_per_video"],
                    eff["gpu_hours"],
                    eff["video_hours"],
                    eff["video_hours_per_gpu_hour"],
                    eff["gpu_hours_per_video_hour"],
                    eff["gpu_sec_per_video"],
                )
        else:
            video_failed += 1
            cli.error("video failed task_id=%s error=%s", result.get("task_id"), result.get("error"))
            if args.max_video_failures > 0 and video_failed >= args.max_video_failures:
                cli.error(
                    "aborting after %d video failure(s); set --max-video-failures 0 to keep going",
                    video_failed,
                )
                fatal_failed += 1
                producer_shutdown.set()
                break

    if fatal_failed and not producer_sentinels_sent:
        producer_shutdown.set()
        for _ in range(consumer_count):
            queue.put(None)
        producer_sentinels_sent = True

    for proc in consumer_procs:
        proc.join()
        if proc.exitcode != 0:
            fatal_failed += 1
            cli.error("consumer pid=%s exited with %s", proc.pid, proc.exitcode)
    producer_shutdown.set()
    for proc in producer_procs:
        proc.join()
        if proc.exitcode != 0:
            fatal_failed += 1
            cli.error("producer pid=%s exited with %s", proc.pid, proc.exitcode)
    if fatal_failed:
        raise SystemExit(
            "producer/consumer run finished with "
            f"{fatal_failed} fatal failure(s), {video_failed} video failure(s)"
        )
    total_elapsed = time.perf_counter() - pipeline_start_perf
    cli.info("producer/consumer done: %d video(s)", done)
    cli.info("TIMING wan_load_excluded_from_pipeline %s", _fmt_stats(consumer_load_times))
    cli.info("TIMING pipeline_total_excluding_wan_load_sec=%.3f", total_elapsed)
    if all_consumers_ready_perf is not None and done:
        ready_elapsed = time.perf_counter() - all_consumers_ready_perf
        eff = _efficiency_stats(
            wall_sec=ready_elapsed,
            videos=done,
            active_gpus=active_gpu_count,
            video_duration_sec=video_duration_sec,
        )
        cli.info(
            "EFFICIENCY_FINAL ready_wall_sec=%.3f videos=%d video_duration_sec=%.3f avg_sec_per_video=%.3f gpu_hours=%.6f video_hours=%.6f video_hours_per_gpu_hour=%.6f gpu_hours_per_video_hour=%.3f gpu_sec_per_video=%.3f",
            ready_elapsed,
            done,
            video_duration_sec,
            eff["avg_sec_per_video"],
            eff["gpu_hours"],
            eff["video_hours"],
            eff["video_hours_per_gpu_hour"],
            eff["gpu_hours_per_video_hour"],
            eff["gpu_sec_per_video"],
        )
    cli.info(
        "TIMING per_video_pipeline_elapsed %s",
        _fmt_stats([float(r.get("pipeline_elapsed_sec", 0.0)) for r in video_results]),
    )
    cli.info(
        "TIMING per_video_pipeline_no_model_load %s",
        _fmt_stats(
            [
                float(r.get("pipeline_no_model_load_sec", r.get("pipeline_elapsed_sec", 0.0)))
                for r in video_results
            ]
        ),
    )
    cli.info(
        "TIMING per_video_image_edit_load %s",
        _fmt_stats([float(r.get("producer_image_edit_load_elapsed_sec", 0.0)) for r in video_results]),
    )
    cli.info(
        "TIMING per_video_image_edit_infer %s",
        _fmt_stats([float(r.get("producer_image_edit_infer_elapsed_sec", 0.0)) for r in video_results]),
    )
    cli.info(
        "TIMING per_video_wan_generate_save %s",
        _fmt_stats([float(r.get("wan_elapsed_sec", r.get("elapsed_sec", 0.0))) for r in video_results]),
    )
    cli.info(
        "TIMING per_video_consumer_augment %s",
        _fmt_stats([float(r.get("consumer_augment_elapsed_sec", 0.0)) for r in video_results]),
    )
    cli.info(
        "TIMING per_video_queue_wait %s",
        _fmt_stats([float(r.get("queue_wait_sec", 0.0)) for r in video_results]),
    )
    by_mode: Dict[str, List[float]] = {}
    for result in video_results:
        mode = str(result.get("pipeline_mode") or "instr_first")
        by_mode.setdefault(mode, []).append(float(result.get("pipeline_elapsed_sec", 0.0)))
    for mode, values in sorted(by_mode.items()):
        cli.info("TIMING mode=%s per_video_pipeline_elapsed %s", mode, _fmt_stats(values))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stream instr-first synthesis batches directly into Wan video generation."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "DiffSynth-Studio"))

from wm_h.video.cuda import isolate_cuda_device, visible_cuda_devices
from wm_h.video.common import load_config
from wm_h.io_utils import list_images
from wm_h.logging_utils import get_cli_logger, setup_logging
from wm_h.streaming import (
    chunks,
    generate_video_rows,
    prepare_video_rows,
    setup_run,
    split_even_chunks,
    video_generator,
    video_preparer,
)
from wm_h.video_prompt_preparer import read_synth_manifest

cli = get_cli_logger()


def _isolate_worker_cuda(gpu_id: int, visible_gpus: Optional[List[str]] = None) -> str:
    return isolate_cuda_device(gpu_id, visible_gpus)


def _visible_worker_count(requested: int, n_items: int) -> int:
    available = len(visible_cuda_devices())
    if available <= 0 or n_items <= 1:
        return 1
    want = available if requested <= 0 else requested
    return max(1, min(want, available, n_items))


def _offload_image_editor(pipeline: Any) -> None:
    editor = getattr(pipeline, "editor", None)
    if editor is not None and getattr(editor, "enabled", False):
        editor.offload_pipe()
        cli.info("offloaded image editor before VL/Wan (single-GPU mode)")


def _release_image_editor(pipeline: Any) -> None:
    editor = getattr(pipeline, "editor", None)
    if editor is not None and getattr(editor, "enabled", False):
        editor.release_pipe(force=True)
        cli.info("released image editor before VL/Wan")


def _run_instr_first_worker(payload: Dict[str, Any]) -> str:
    gpu_id = int(payload["gpu_id"])
    device = _isolate_worker_cuda(gpu_id, payload.get("visible_gpus"))
    setup_logging(verbose=bool(payload.get("verbose", False)))

    from wm_h.instr_first_pipeline import InstrFirstPipeline

    config_path = payload["config"]
    with open(config_path, encoding="utf-8") as f:
        full_config = yaml.safe_load(f)
    if_cfg = full_config.get("instr_first", {})
    video_config = load_config(payload["video_config"])
    batch_size = max(1, int(payload["batch_size"]))
    target = int(payload["total_instructions"])

    img_dir = payload.get("image_dir") or if_cfg.get("data", {}).get("image_dir", "")
    images = list_images(img_dir, recursive=bool(payload.get("recursive", False)))
    if not images:
        raise FileNotFoundError(f"No images in {img_dir}")

    out_base = Path(if_cfg.get("output", {}).get("dir", "database/wm-h/instr_first"))
    run_dir, _, edited_dir, _, videos_dir, tasks_dir = setup_run(
        out_base.parent if out_base.name == "instr_first" else out_base,
        gpu_id,
        worker_label=str(payload.get("worker_label") or ""),
        run_label=str(payload.get("run_label") or ""),
    )

    worker_config = full_config.copy()
    worker_if_cfg = dict(worker_config.get("instr_first", {}))
    worker_model_cfg = dict(worker_if_cfg.get("model", {}))
    worker_model_cfg["keep_model_loaded"] = False
    worker_if_cfg["model"] = worker_model_cfg
    worker_db_cfg = dict(worker_if_cfg.get("database", {}))
    worker_db_cfg["base_path"] = str(run_dir / "slots")
    worker_if_cfg["database"] = worker_db_cfg
    worker_output_cfg = dict(worker_if_cfg.get("output", {}))
    worker_output_cfg["instructions_file"] = str(run_dir / "vocab" / "generated_instructions.jsonl")
    worker_if_cfg["output"] = worker_output_cfg
    worker_config["instr_first"] = worker_if_cfg
    worker_config_path = run_dir / "streaming_worker_config.yaml"
    worker_config_path.write_text(yaml.safe_dump(worker_config, sort_keys=False), encoding="utf-8")

    pipeline = InstrFirstPipeline(str(worker_config_path), device=device)
    pipeline.run_dir = run_dir
    pipeline.edited_dir = edited_dir
    preparer = video_preparer(video_config, device)
    generator: Optional[Any] = None

    manifest_all = run_dir / "manifest.jsonl"
    video_manifest_all = run_dir / "video_manifest.jsonl"
    settings_cfg = video_config.get("settings", {})
    offload_between_stages = bool(settings_cfg.get("offload_between_stages", False))
    release_between_stages = bool(settings_cfg.get("release_between_stages", False))
    release_image_editor_after_edit = bool(
        settings_cfg.get("release_image_editor_after_edit", release_between_stages)
    )
    total_videos = 0
    produced = 0
    batch_idx = 0
    while produced < target:
        n = min(batch_size, target - produced)
        cli.info("instr_first batch %d: assemble/edit %d instruction(s)", batch_idx, n)
        assembled = pipeline._assemble_count(n)
        batch_manifest = run_dir / f"manifest.batch{batch_idx:04d}.jsonl"
        pipeline.edit_assembled(assembled, images, batch_manifest, start_idx=produced)
        if offload_between_stages:
            _offload_image_editor(pipeline)
        elif release_image_editor_after_edit:
            _release_image_editor(pipeline)

        rows = read_synth_manifest(str(batch_manifest))
        with open(manifest_all, "a", encoding="utf-8") as out_f:
            for row in rows:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if offload_between_stages and generator is not None:
            generator.offload_model()
            cli.info("offloaded Wan before VL augment (single-GPU mode)")

        preparer.reload_model()
        video_rows = prepare_video_rows(
            rows,
            preparer=preparer,
            video_config=video_config,
            synth_manifest=batch_manifest,
            seed_offset=batch_idx * 100000,
        )
        if offload_between_stages:
            preparer.release_model()
            cli.info("released VL augment model before Wan (single-GPU mode)")
        elif release_between_stages:
            preparer.release_model()
            cli.info("released VL augment model before Wan")

        with open(video_manifest_all, "a", encoding="utf-8") as out_f:
            for row in video_rows:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if generator is None:
            generator = video_generator(video_config, device)
        else:
            generator.onload_model()
        total_videos += generate_video_rows(
            video_rows,
            generator=generator,
            video_config=video_config,
            videos_dir=videos_dir,
            tasks_dir=tasks_dir,
        )
        if offload_between_stages:
            generator.offload_model()
        produced += n
        batch_idx += 1

    cli.info("instr_first streaming done: %d video(s) -> %s", total_videos, run_dir)
    return str(run_dir)


def _worker_entry(payload: Dict[str, Any], queue: Any) -> None:
    try:
        queue.put({"ok": True, "run_dir": _run_instr_first_worker(payload)})
    except BaseException as exc:
        queue.put(
            {
                "ok": False,
                "gpu_id": payload.get("gpu_id"),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-stream instr_first rows directly into video generation"
    )
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--video-config", default="configs/video.yaml")
    parser.add_argument("--image-dir", default="")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--total-instructions", type=int, default=8)
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--worker-label", default="")
    parser.add_argument("--run-label", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    visible_gpus = visible_cuda_devices()
    if visible_gpus:
        cli.info("streaming visible GPUs: %s", ",".join(visible_gpus))

    workers = _visible_worker_count(args.num_gpus, args.total_instructions)
    per_worker = math.ceil(args.total_instructions / workers)
    payloads = [
        {
            "gpu_id": gpu_id,
            "config": args.config,
            "video_config": args.video_config,
            "image_dir": args.image_dir,
            "recursive": args.recursive,
            "batch_size": args.batch_size,
            "total_instructions": min(per_worker, args.total_instructions - gpu_id * per_worker),
            "visible_gpus": visible_gpus,
            "worker_label": (
                args.worker_label if workers == 1 else f"{args.worker_label}_gpu{gpu_id}"
            )
            if args.worker_label
            else "",
            "run_label": args.run_label,
            "verbose": args.verbose,
        }
        for gpu_id in range(workers)
        if args.total_instructions - gpu_id * per_worker > 0
    ]

    if len(payloads) == 1:
        print(_run_instr_first_worker(payloads[0]))
        return

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = [ctx.Process(target=_worker_entry, args=(payload, queue)) for payload in payloads]
    for proc in processes:
        proc.start()

    results = [queue.get() for _ in processes]
    for proc in processes:
        proc.join()

    failures = [item for item in results if not item.get("ok")]
    if failures:
        for failure in failures:
            print(failure.get("traceback") or failure.get("error"), file=sys.stderr)
        raise SystemExit(1)
    for item in results:
        print(item["run_dir"])


if __name__ == "__main__":
    main()

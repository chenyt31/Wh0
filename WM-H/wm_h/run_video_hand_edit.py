#!/usr/bin/env python3
"""Post-process Wan videos: replace human hands with robot hands on selected frames."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "DiffSynth-Studio"))

os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "sage_attention")

from wm_h.logging_utils import get_cli_logger, setup_logging
from wm_h.video.common import load_config
from wm_h.video.cuda import isolate_cuda_device, visible_cuda_devices
from wm_h.parallel import resolve_worker_count, split_even_chunks

cli = get_cli_logger()


def _worker(payload: dict) -> int:
    gpu_id = int(payload["gpu_id"])
    device = isolate_cuda_device(gpu_id, payload.get("visible_gpus"))
    setup_logging(verbose=bool(payload.get("verbose", False)))

    from wm_h.video.hand_edit import load_hand_edit_pipe, process_video

    pipe = load_hand_edit_pipe(payload["model_cfg"], device)
    output_dir = Path(payload["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    for raw_video_path in payload["videos"]:
        video_path = Path(raw_video_path)
        out_path = output_dir / video_path.name
        if process_video(
            Path(video_path),
            out_path,
            pipe,
            every_n=int(payload["every_n"]),
            offset=int(payload["offset"]),
            prompt=str(payload["prompt"]),
            negative_prompt=str(payload["negative_prompt"]),
            seed=int(payload["seed"]),
            num_inference_steps=int(payload["steps"]),
            cfg_scale=float(payload["cfg_scale"]),
            quality=int(payload["quality"]),
            skip_existing=bool(payload.get("skip_existing", True)),
        ):
            done += 1
            cli.info("robot-hand edit saved %s", out_path)
    return done


def _worker_entry(payload: dict, queue: mp.Queue) -> None:
    try:
        queue.put({"ok": True, "count": _worker(payload)})
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/video_hand_edit.yaml")
    parser.add_argument("--input-dir", default="", help="Override data.input_dir")
    parser.add_argument("--output-dir", default="", help="Override data.output_dir")
    parser.add_argument("--every-n", type=int, default=0, help="Override edit.every_n")
    parser.add_argument("--offset", type=int, default=-1, help="Override edit.offset")
    parser.add_argument("--max-videos", type=int, default=-1, help="Process at most this many videos")
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    config = load_config(args.config)
    from wm_h.video.hand_edit import collect_videos, resolve_io_dirs

    data_cfg = dict(config.get("data") or {})
    if args.input_dir:
        data_cfg["input_dir"] = args.input_dir
    if args.output_dir:
        data_cfg["output_dir"] = args.output_dir

    input_dir, output_dir = resolve_io_dirs(data_cfg, ROOT)
    videos = collect_videos(input_dir)
    if args.max_videos >= 0:
        videos = videos[: args.max_videos]
    if not videos:
        raise SystemExit(f"No videos in {input_dir}")

    edit_cfg = config.get("edit") or {}
    if args.every_n > 0:
        edit_cfg["every_n"] = args.every_n
    if args.offset >= 0:
        edit_cfg["offset"] = args.offset
    settings = config.get("settings") or {}
    model_cfg = config.get("model") or {}
    visible = visible_cuda_devices()
    workers = resolve_worker_count(
        args.num_gpus or int(settings.get("num_gpus", 0)),
        len(videos),
    )
    if not bool(settings.get("multi_gpu", True)):
        workers = 1

    payload_base = {
        "model_cfg": model_cfg,
        "output_dir": str(output_dir),
        "every_n": int(edit_cfg.get("every_n", 4)),
        "offset": int(edit_cfg.get("offset", 1)),
        "prompt": str(edit_cfg.get("prompt", "")),
        "negative_prompt": str(edit_cfg.get("negative_prompt", "")),
        "steps": int(edit_cfg.get("steps", 4)),
        "cfg_scale": float(edit_cfg.get("cfg_scale", 1.0)),
        "seed": int(edit_cfg.get("seed", 42)),
        "quality": int(edit_cfg.get("quality", 9)),
        "skip_existing": not args.no_skip_existing,
        "verbose": args.verbose,
        "visible_gpus": visible,
    }
    if not payload_base["prompt"]:
        from wm_h.video.hand_edit import DEFAULT_PROMPT, DEFAULT_NEGATIVE_PROMPT

        payload_base["prompt"] = DEFAULT_PROMPT
        payload_base["negative_prompt"] = DEFAULT_NEGATIVE_PROMPT

    cli.info(
        "robot-hand video edit: %d video(s) from %s -> %s (%d worker(s))",
        len(videos),
        input_dir,
        output_dir,
        workers,
    )

    if workers <= 1:
        count = _worker(
            {
                **payload_base,
                "gpu_id": 0,
                "videos": [str(p) for p in videos],
            }
        )
    else:
        chunks = split_even_chunks([str(p) for p in videos], workers)
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        processes = [
            ctx.Process(
                target=_worker_entry,
                args=({**payload_base, "gpu_id": i, "videos": chunk}, queue),
            )
            for i, chunk in enumerate(chunks)
            if chunk
        ]
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
        count = sum(int(item.get("count", 0)) for item in results)

    cli.info("robot-hand edit done: %d/%d video(s) written to %s", count, len(videos), output_dir)


if __name__ == "__main__":
    main()

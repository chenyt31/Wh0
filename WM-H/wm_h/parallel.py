"""Multi-GPU parallel execution for the instr-first edit phase."""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from wm_h.video.cuda import isolate_cuda_device, visible_cuda_devices
from wm_h.video.common import CheckpointManager

from .logging_utils import get_cli_logger

logger = logging.getLogger(__name__)
cli = get_cli_logger()

ImageItem = Tuple[int, Path]


def _isolate_worker_cuda(gpu_id: int, visible_gpus: Optional[List[str]] = None) -> str:
    return isolate_cuda_device(gpu_id, visible_gpus)


def detect_cuda_device_count() -> int:
    return len(visible_cuda_devices())


def resolve_worker_count(num_gpus: int, n_items: int) -> int:
    """Resolve GPU worker count. num_gpus=0 uses all visible GPUs."""
    if n_items <= 1:
        return 1
    available = detect_cuda_device_count()
    if available <= 0:
        return 1
    requested = available if num_gpus <= 0 else num_gpus
    return max(1, min(requested, available, n_items))


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


def _setup_run_directories(output_base: Path) -> Tuple[Path, Path]:
    run_tag = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = output_base / "runs" / run_tag
    edited_dir = run_dir / "edited_images"
    edited_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, edited_dir


def _merge_manifests(run_dir: Path, manifest_path: Path, num_workers: int) -> int:
    rows: List[Dict[str, Any]] = []
    for gpu_id in range(num_workers):
        partial = run_dir / f"manifest.gpu{gpu_id}.jsonl"
        if not partial.exists():
            continue
        with open(partial, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        partial.unlink()
    rows.sort(key=lambda r: r["task_id"])
    with open(manifest_path, "w", encoding="utf-8") as out_f:
        for row in rows:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def _instr_first_worker(payload: Dict[str, Any]) -> None:
    from .instr_first_pipeline import InstrFirstPipeline
    from .logging_utils import setup_logging

    gpu_id = int(payload["gpu_id"])
    device = _isolate_worker_cuda(gpu_id, payload.get("visible_gpus"))
    setup_logging(verbose=bool(payload.get("verbose", False)))

    pipeline = InstrFirstPipeline(payload["config_path"], device=device)
    pipeline.run_dir = Path(payload["run_dir"])
    pipeline.edited_dir = Path(payload["edited_dir"])
    ckpt = (
        CheckpointManager(
            payload["checkpoint_dir"],
            checkpoint_name=payload["checkpoint_name"],
        )
        if payload.get("checkpoint_enabled")
        else None
    )
    processed_ids = ckpt.load_processed_instructions() if ckpt is not None else set()

    images = [Path(p) for p in payload["images"]]
    items: List[Tuple[int, Dict[str, Any]]] = [
        (int(idx), assembled) for idx, assembled in payload["items"]
    ]
    partial_manifest = Path(payload["partial_manifest"])

    with open(partial_manifest, "w", encoding="utf-8") as out_f:
        for instr_idx, assembled in items:
            ckpt_id = CheckpointManager.get_instr_first_id(assembled)
            if processed_ids is not None and ckpt_id in processed_ids:
                continue
            row = pipeline._edit_instruction_resilient(assembled, images, instr_idx)
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if ckpt is not None:
                ckpt.mark_as_processed(ckpt_id, gpu_id)
                processed_ids.add(ckpt_id)


def run_instr_first_multi_gpu(
    config_path: str,
    assembled_list: List[Dict[str, Any]],
    images: List[Path],
    *,
    run_dir: Path,
    manifest_path: Path,
    num_workers: int,
    seed: int = 0,
    verbose: bool = False,
    checkpoint: Optional[CheckpointManager] = None,
) -> int:
    """Edit pre-assembled instructions across GPU workers; returns row count."""
    items = list(enumerate(assembled_list))
    with open(config_path, encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)
    ckpt_cfg = (full_cfg.get("instr_first") or {}).get("checkpoint", {})
    ckpt_enabled = checkpoint is not None and ckpt_cfg.get("enable", True)
    chunks = split_even_chunks(items, num_workers)
    visible_gpus = visible_cuda_devices()
    ctx = mp.get_context("spawn")
    processes: List[mp.Process] = []
    edited_dir = run_dir / "edited_images"

    for gpu_id, chunk in enumerate(chunks):
        if not chunk:
            continue
        payload = {
            "gpu_id": gpu_id,
            "config_path": config_path,
            "items": [(idx, assembled) for idx, assembled in chunk],
            "images": [str(p) for p in images],
            "run_dir": str(run_dir),
            "edited_dir": str(edited_dir),
            "partial_manifest": str(run_dir / f"manifest.gpu{gpu_id}.jsonl"),
            "visible_gpus": visible_gpus,
            "seed": seed,
            "verbose": verbose,
            "checkpoint_enabled": ckpt_enabled,
            "checkpoint_dir": ckpt_cfg.get("dir", "database/log"),
            "checkpoint_name": ckpt_cfg.get(
                "checkpoint_name", "processed_instr_first.txt"
            ),
        }
        proc = ctx.Process(
            target=_instr_first_worker,
            args=(payload,),
            name=f"instr-first-gpu-{gpu_id}",
        )
        proc.start()
        processes.append(proc)

    exit_codes = []
    for proc in processes:
        proc.join()
        exit_codes.append(proc.exitcode)

    if not all(code == 0 for code in exit_codes):
        raise RuntimeError(f"instr_first GPU worker(s) failed: exit codes {exit_codes}")

    return _merge_manifests(run_dir, manifest_path, num_workers)

"""Helpers for mapping multi-process workers to one physical GPU each."""

from __future__ import annotations

import os
import re
import subprocess
from typing import Iterable, Optional


def visible_cuda_devices(limit: int = 0) -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        devices = [v.strip() for v in visible.split(",") if v.strip()]
    else:
        devices = _detect_physical_cuda_devices()
    if limit > 0:
        devices = devices[:limit]
    return devices


def _detect_physical_cuda_devices() -> list[str]:
    """Detect GPUs without importing torch, so worker isolation can happen first."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if proc.returncode == 0:
            ids = []
            for line in proc.stdout.splitlines():
                match = re.match(r"GPU\s+(\d+):", line.strip())
                if match:
                    ids.append(match.group(1))
            if ids:
                return ids
    except Exception:
        pass

    ids = []
    for name in os.listdir("/dev") if os.path.isdir("/dev") else []:
        match = re.fullmatch(r"nvidia(\d+)", name)
        if match:
            ids.append(int(match.group(1)))
    return [str(i) for i in sorted(ids)]


def isolate_cuda_device(gpu_id: int, visible_gpus: Optional[Iterable[int | str]] = None) -> str:
    """Expose exactly one physical GPU to this process and return in-process device."""
    if gpu_id < 0:
        return "cuda"
    if visible_gpus is None:
        devices = visible_cuda_devices()
    else:
        devices = [str(v) for v in visible_gpus]
    physical = devices[gpu_id] if 0 <= gpu_id < len(devices) else str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = physical
    device = "cuda:0"
    try:
        import torch

        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            logical = 0 if count <= 1 else min(max(gpu_id, 0), count - 1)
            torch.cuda.set_device(logical)
            device = f"cuda:{logical}"
    except Exception:
        pass
    return device

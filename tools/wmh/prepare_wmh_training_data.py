#!/usr/bin/env python3
"""Materialize one WM-H run as a VITRA WM-H training tree."""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.dataset_index.episodic import write_episode_frame_index


EPISODE_RE = re.compile(r"_ep_(\d{6})\.npy$")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _safe_link(target: Path, source: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    source = source.resolve()
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source:
            return
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.symlink_to(source, target_is_directory=source.is_dir())


def _video_files(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


def _annotation_for_video(annotation_dir: Path, video_stem: str) -> Path | None:
    candidates = sorted(annotation_dir.glob(f"*_{video_stem}_ep_*.npy"))
    if candidates:
        return candidates[0]
    candidates = sorted(path for path in annotation_dir.glob("*_ep_*.npy") if video_stem in path.stem)
    return candidates[0] if candidates else None


def _episode_suffix(path: Path) -> str:
    match = EPISODE_RE.search(path.name)
    return match.group(1) if match else "000000"


def prepare(
    run_dir: Path,
    output_root: Path,
    *,
    edited_video_dir: Path | None,
    robot_prob: float,
    seed: int,
    dataset_name: str,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    original_dir = run_dir / "videos"
    annotation_dir = run_dir / "episodic_annotations"
    edited_video_dir = (
        run_dir / "videos_robot_hands"
        if edited_video_dir is None
        else edited_video_dir.expanduser().resolve()
    )

    if not original_dir.is_dir():
        raise FileNotFoundError(f"Missing original video directory: {original_dir}")
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"Missing annotation directory: {annotation_dir}")

    video_root = output_root / "Video" / f"{dataset_name}_root"
    annotation_root = output_root / "Annotation" / dataset_name / "episodic_annotations"
    manifest: list[dict[str, Any]] = []

    for video in _video_files(original_dir):
        annotation = _annotation_for_video(annotation_dir, video.stem)
        if annotation is None:
            continue
        edited = edited_video_dir / video.name
        rng = random.Random(f"{seed}:{video.stem}")
        use_robot = edited.is_file() and rng.random() < robot_prob
        selected = edited if use_robot else video

        _safe_link(video_root / video.name, selected)
        episode = _episode_suffix(annotation)
        _safe_link(annotation_root / f"{dataset_name}_{video.stem}_ep_{episode}.npy", annotation)
        manifest.append(
            {
                "video": video.name,
                "annotation": annotation.name,
                "selected_source": "robot_hand_edit" if use_robot else "original",
                "selected_video": str(selected),
            }
        )

    if not manifest:
        raise RuntimeError(f"No video/annotation pairs found in {run_dir}")

    index_path = output_root / "Annotation" / dataset_name / "episode_frame_index.npz"
    write_episode_frame_index(annotation_root, index_path)
    manifest_path = output_root / "wmh_training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "output_root": str(output_root),
        "video_root": str(video_root),
        "annotation_root": str(annotation_root),
        "index_path": str(index_path),
        "manifest_path": str(manifest_path),
        "total": len(manifest),
        "robot_hand_edit": sum(1 for item in manifest if item["selected_source"] == "robot_hand_edit"),
        "original": sum(1 for item in manifest if item["selected_source"] == "original"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="WM-H run directory containing videos/ and episodic_annotations/")
    parser.add_argument("--output-root", default="", help="Default: <run_dir>/vitra_training_data")
    parser.add_argument("--edited-video-dir", default="", help="Default: <run_dir>/videos_robot_hands")
    parser.add_argument("--robot-prob", type=float, default=0.2, help="Probability of selecting edited video")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic selection seed")
    parser.add_argument("--dataset-name", default="WM-H", help="VITRA dataset folder/prefix")
    args = parser.parse_args()

    if not 0.0 <= args.robot_prob <= 1.0:
        raise SystemExit("--robot-prob must be in [0, 1]")
    run_dir = Path(args.run_dir)
    output_root = Path(args.output_root) if args.output_root else run_dir / "vitra_training_data"
    edited_video_dir = Path(args.edited_video_dir) if args.edited_video_dir else None
    result = prepare(
        run_dir,
        output_root,
        edited_video_dir=edited_video_dir,
        robot_prob=args.robot_prob,
        seed=args.seed,
        dataset_name=args.dataset_name,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

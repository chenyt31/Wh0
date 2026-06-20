from __future__ import annotations

import argparse
import os
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import imageio.v2 as imageio


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.eval_pipeline import save_annotation_frame, save_dataset_frame


def _run_checked(cmd: list[str]) -> None:
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    if result.stdout:
        print(result.stdout)


def _load_presets(path: str | None) -> dict[str, dict]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _apply_preset(argv: list[str]) -> list[str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--preset-file")
    parser.add_argument("--preset")
    known, remaining = parser.parse_known_args(argv)
    if not known.preset:
        return argv
    presets = _load_presets(known.preset_file)
    if known.preset not in presets:
        raise ValueError(f"Unknown preset: {known.preset}")
    preset_args = presets[known.preset]
    merged: list[str] = []
    for key, value in preset_args.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                merged.append(flag)
        else:
            merged.extend([flag, str(value)])
    return merged + remaining


def _resolve_annotation_path(video_path: str) -> str:
    video_name = Path(video_path).stem
    return str(REPO_ROOT / f"debug_data/Annotation/WM-H/episodic_annotations/WM-H_{video_name}_ep_000000.npy")


def _extract_raw_instruction(raw: str, use_left: bool, use_right: bool) -> str:
    raw = raw.strip()
    if "Left hand:" not in raw or "Right hand:" not in raw:
        return raw
    if use_right:
        parts = raw.split("Right hand:", 1)
        return parts[1].strip().rstrip(".,") if len(parts) > 1 else ""
    if use_left:
        parts = raw.split("Left hand:", 1)
        return parts[1].split("Right hand:")[0].strip().rstrip(".,") if len(parts) > 1 else ""
    return ""


def _build_instruction(info: dict, args: argparse.Namespace) -> str:
    if args.instruction:
        return args.instruction
    raw = info.get("instruction") or info.get("text", "")
    text = _extract_raw_instruction(raw, args.use_left, args.use_right)
    if args.left_template and args.right_template:
        return f"{args.left_template.format(text=text)}. {args.right_template.format(text=text)}."
    if args.left_template:
        return args.left_template.format(text=text)
    if args.right_template:
        return args.right_template.format(text=text)
    if args.use_left and args.use_right:
        return f"Left hand: {text}. Right hand: {text}."
    if args.use_left:
        return f"Left hand: {text}. Right hand: None."
    return f"Left hand: None. Right hand: {text}."


def _default_output_name(args: argparse.Namespace) -> str:
    if args.output_video:
        return args.output_video
    if args.mode.startswith("annotation"):
        suffix = "_pred2.mp4" if args.mode == "annotation2" else "_pred.mp4"
        return f"{Path(args.video_path).stem}_frame{args.frame_idx}{suffix}"
    return f"sample{args.sample_idx}_pred.mp4"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = _apply_preset(argv or sys.argv[1:])
    parser = argparse.ArgumentParser(description="Human prediction pipeline")
    parser.add_argument("--preset-file")
    parser.add_argument("--preset")
    parser.add_argument("--mode", required=True, choices=["annotation", "annotation2", "dataset"])
    parser.add_argument("--video-path")
    parser.add_argument("--annotation-npy")
    parser.add_argument("--frame-idx", type=int, default=0)
    parser.add_argument("--hand-index", type=int, default=1)
    parser.add_argument("--root-dir")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--statistics-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--output-videos-dir", default="output_videos")
    parser.add_argument("--output-video")
    parser.add_argument("--keep-extracted", action="store_true")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--use-left", action="store_true")
    parser.add_argument("--use-right", action="store_true")
    parser.add_argument("--sample-times", type=int, default=1)
    parser.add_argument("--mano-path")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--instruction")
    parser.add_argument("--render-gt", action="store_true")
    parser.add_argument("--no-render-gt", action="store_true")
    parser.add_argument("--left-template")
    parser.add_argument("--right-template")
    args = parser.parse_args(argv)

    if not args.use_left and not args.use_right:
        args.use_right = True
    if args.mode in {"annotation", "annotation2"} and not args.video_path:
        parser.error(f"{args.mode} mode requires --video-path")
    if args.mode == "dataset" and not args.root_dir:
        parser.error("dataset mode requires --root-dir")
    if args.mode in {"annotation", "annotation2"} and not args.annotation_npy:
        args.annotation_npy = _resolve_annotation_path(args.video_path)
    return args


def _run_extract(args: argparse.Namespace, extracted_dir: str) -> None:
    if args.mode == "dataset":
        save_dataset_frame(
            root_dir=args.root_dir,
            sample_idx=args.sample_idx,
            output_dir=extracted_dir,
            statistics_path=args.statistics_path,
        )
        return
    save_annotation_frame(
        video_path=args.video_path,
        annotation_npy_path=args.annotation_npy,
        frame_idx=args.frame_idx,
        output_dir=extracted_dir,
        hand_index=args.hand_index,
    )


def _run_inference(args: argparse.Namespace, extracted_dir: str, instruction: str, output_video: str) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "inference_human_prediction.py"),
        "--config_path",
        args.config,
        "--image_path",
        str(Path(extracted_dir) / "image.jpg"),
        "--instruction",
        instruction,
        "--video_path",
        output_video,
        "--sample_times",
        str(args.sample_times),
        "--fps",
        str(args.fps),
    ]
    if args.use_left:
        cmd.append("--use_left")
    if args.use_right:
        cmd.append("--use_right")
    if args.mano_path:
        cmd.extend(["--mano_path", args.mano_path])
    if args.statistics_path:
        cmd.extend(["--statistics_path", args.statistics_path])
    if args.model_path:
        cmd.extend(["--model_path", args.model_path])
    _run_checked(cmd)


def _gt_output_path(pred_video: str) -> Path:
    path = Path(pred_video)
    return path.with_name(f"{path.stem}_gt{path.suffix}")


def _compare_output_path(pred_video: str) -> Path:
    path = Path(pred_video)
    stem = path.stem
    if stem.endswith("_pred"):
        stem = stem[: -len("_pred")]
    return path.with_name(f"{stem}_pred_vs_gt{path.suffix}")


def _render_dataset_gt(args: argparse.Namespace, output_video: str) -> Path:
    gt_dir = Path(args.output_dir or tempfile.mkdtemp(prefix="human_gt_")) / "gt_render"
    cmd = [
        sys.executable,
        "-m",
        "vitra.tools.render_hand_dataset",
        "--root-dir",
        args.root_dir,
        "--sample-idx",
        str(args.sample_idx),
        "--num-frames",
        str(max(args.sample_times, 1) * 17),
        "--num-samples",
        "1",
        "--output-dir",
        str(gt_dir),
        "--fps",
        str(args.fps),
        "--no-action-compare",
    ]
    if args.statistics_path:
        cmd.extend(["--statistics-path", args.statistics_path])
    if args.mano_path:
        cmd.extend(["--mano-path", args.mano_path])
    _run_checked(cmd)
    source = gt_dir / "hand_rendered_video.mp4"
    target = _gt_output_path(output_video)
    shutil.copy2(source, target)
    return target


def _render_annotation_gt(args: argparse.Namespace, output_video: str) -> Path:
    gt_root = Path(args.output_dir or tempfile.mkdtemp(prefix="human_gt_")) / "gt_render"
    label_dir = gt_root / "labels"
    save_dir = gt_root / "videos"
    label_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    label_link = label_dir / Path(args.annotation_npy).name
    if label_link.exists() or label_link.is_symlink():
        label_link.unlink()
    os.symlink(Path(args.annotation_npy).resolve(), label_link)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "data" / "demo_visualization_epi.py"),
        "--video_root",
        str(Path(args.video_path).expanduser().parent.resolve()),
        "--label_root",
        str(label_dir),
        "--save_path",
        str(save_dir),
        "--mano_model_path",
        args.mano_path,
        "--modes",
        "cam",
        "--max_episodes",
        "1",
    ]
    _run_checked(cmd)
    rendered = sorted(save_dir.glob("*.mp4"))
    if not rendered:
        raise RuntimeError(f"No GT render was produced in {save_dir}")
    target = _gt_output_path(output_video)
    shutil.copy2(rendered[0], target)
    return target


def _resize_to_height(frame, height: int):
    if frame.shape[0] == height:
        return frame
    width = int(round(frame.shape[1] * height / frame.shape[0]))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _compose_pred_vs_gt(pred_video: str, gt_video: Path) -> Path:
    pred_reader = imageio.get_reader(pred_video)
    gt_reader = imageio.get_reader(str(gt_video))
    output = _compare_output_path(pred_video)
    output.parent.mkdir(parents=True, exist_ok=True)
    fps = pred_reader.get_meta_data().get("fps", 8)
    with imageio.get_writer(output, fps=fps) as writer:
        for pred_frame, gt_frame in zip(pred_reader, gt_reader):
            height = min(pred_frame.shape[0], gt_frame.shape[0])
            pred_frame = _resize_to_height(pred_frame, height)
            gt_frame = _resize_to_height(gt_frame, height)
            writer.append_data(cv2.hconcat([pred_frame, gt_frame]))
    pred_reader.close()
    gt_reader.close()
    return output


def _render_and_compose_gt(args: argparse.Namespace, output_video: str) -> None:
    if args.no_render_gt:
        return
    if args.mode == "dataset":
        gt_video = _render_dataset_gt(args, output_video)
    else:
        gt_video = _render_annotation_gt(args, output_video)
    compare_video = _compose_pred_vs_gt(output_video, gt_video)
    print(f"GT video: {gt_video}")
    print(f"Prediction vs GT video: {compare_video}")


def run_pipeline(args: argparse.Namespace) -> None:
    extracted_dir = args.output_dir or tempfile.mkdtemp(prefix="human_pred_")
    Path(extracted_dir).mkdir(parents=True, exist_ok=True)
    print(f"Working directory: {extracted_dir}")

    try:
        _run_extract(args, extracted_dir)
        info_path = Path(extracted_dir) / "info.json"
        with info_path.open("r", encoding="utf-8") as handle:
            info = json.load(handle)
        instruction = _build_instruction(info, args)
        output_dir = Path(args.output_videos_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_video = str(output_dir / _default_output_name(args))
        _run_inference(args, extracted_dir, instruction, output_video)
        _render_and_compose_gt(args, output_video)
        print(f"Output video: {output_video}")
    finally:
        if args.output_dir:
            if not args.keep_extracted:
                npy_path = Path(extracted_dir) / "image.npy"
                if npy_path.exists():
                    npy_path.unlink()
        elif not args.keep_extracted and Path(extracted_dir).exists():
            shutil.rmtree(extracted_dir)


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()

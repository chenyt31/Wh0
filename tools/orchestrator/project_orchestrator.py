#!/usr/bin/env python3
"""Run Wh0 setup and workflows from one YAML file."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import yaml


DEFAULT_FINETUNE_CONFIG = "vitra-wh0/vitra/configs/robot_finetune_wmh.json"
DEFAULT_PRETRAIN_CONFIG = "vitra-wh0/vitra/configs/human_pretrain.json"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, env=merged, check=True)


def repo_root(config: dict) -> Path:
    return Path(config["paths"]["repo_root"]).expanduser().resolve()


def setup_env(config: dict) -> None:
    if config["environment"].get("setup_with_uv", True):
        environment = config.get("environment", {})
        sources = config.get("package_sources", {})
        run(
            ["bash", "scripts/setup.sh"],
            cwd=repo_root(config),
            env={
                "CONFIG_PATH": "configs/project_request.yaml",
                "SYNC_THIRD_PARTY": "1" if environment.get("sync_third_party", True) else "0",
                "PYTHON_VERSION": str(environment.get("python_version", "3.10")),
                "CUDA_EXTRA": str(environment.get("cuda_extra", "auto")),
                "WH0_EXTRAS": ",".join(str(item) for item in environment.get("dependency_extras", ["policy", "wmh", "annotation", "dev"])),
                "INSTALL_VLLM": "1" if environment.get("install_vllm", False) else "0",
                "INSTALL_VISUALIZATION": "1" if environment.get("install_visualization", False) else "0",
                "PYPI_INDEX_URL": str(sources.get("pypi_index_url", "")),
                "PYPI_EXTRA_INDEX_URLS": ",".join(str(item) for item in sources.get("pypi_extra_index_urls", [])),
                "PYPI_FIND_LINKS": ",".join(str(item) for item in sources.get("pypi_find_links", [])),
                "UV_INDEX_STRATEGY": str(sources.get("uv_index_strategy", "")),
                "PYTORCH_INDEX_URL": str(sources.get("pytorch_index_url", "")),
                "WH0_HFD_URL": str(sources.get("hfd_script_url", "https://hf-mirror.com/hfd/hfd.sh")),
            },
        )


def sync_weights(config: dict) -> None:
    if not config["weights"].get("download_missing", True):
        return
    root = repo_root(config)
    weights_root = Path(config["paths"]["weights_root"]).expanduser().resolve()
    items = config["weights"]["items"]
    enabled = [name for name, payload in items.items() if payload.get("enabled", False)]
    cmd = [
        "uv",
        "run",
        "--with",
        "huggingface_hub",
        "--with",
        "pyyaml",
        "python",
        "tools/weights/manage_weights.py",
        "sync",
        "--hf-downloader",
        str(config["weights"].get("hf_downloader", "hfd")),
        "--weights-root",
        str(weights_root),
    ]
    if enabled:
        cmd.extend(["--only", *enabled])
    for name, payload in items.items():
        local_path = str(payload.get("local_path", "")).strip()
        if local_path:
            cmd.extend(["--local-path", f"{name}={local_path}"])
    run(cmd, cwd=root, env={"WH0_HFD_URL": str(config.get("package_sources", {}).get("hfd_script_url", "https://hf-mirror.com/hfd/hfd.sh"))})


def workflow_env(config: dict) -> dict[str, str]:
    env = {
        "INPUT_PATH": config["paths"].get("input_path", ""),
        "OUTPUT_PATH": config["paths"].get("output_path", ""),
        "WH0_DATASET_NAME": config["annotate"].get("dataset_name", "WM-H"),
        "PARALLEL_K": str(config["annotate"].get("parallel_k", 1)),
        "PROFILE": config["wmh"].get("profile", "default"),
        "TOTAL_INSTRUCTIONS": str(config["wmh"].get("total_instructions", 10)),
        "TASK": config["train"].get("task", "finetune"),
        "HAND_EDIT_EVERY_N": str(config.get("hand_edit", {}).get("every_n", 4)),
        "ROBOT_PROB": str(config.get("prepare_data", {}).get("robot_prob", 0.2)),
    }
    wmh = config.get("wmh", {})
    python_path = str(wmh.get("python", "")).strip()
    bin_prepend = str(wmh.get("bin_prepend", "")).strip()
    if python_path:
        env["WMH_PYTHON"] = python_path
    if bin_prepend:
        env["PATH"] = f"{bin_prepend}{os.pathsep}{os.environ.get('PATH', '')}"
    return env


def latest_wmh_run(root: Path) -> Path:
    run_root = root / "WM-H" / "database" / "wm-h" / "instr_first" / "streaming_runs"
    candidates = [p for p in run_root.glob("run_*") if (p / "videos").is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No WM-H run with videos found under {run_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _latest_wmh_run_group(root: Path) -> list[Path]:
    run_root = root / "WM-H" / "database" / "wm-h" / "instr_first" / "streaming_runs"
    candidates = [p for p in run_root.glob("run_*") if (p / "videos").is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No WM-H run with videos found under {run_root}")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    match = re.match(r"^(run_.+)_gpu\d+$", latest.name)
    if not match:
        return [latest]
    prefix = match.group(1)
    grouped = sorted(
        p for p in candidates if re.match(rf"^{re.escape(prefix)}_gpu\d+$", p.name)
    )
    return grouped or [latest]


def _safe_symlink(target: Path, source: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source.resolve():
            return
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def materialize_latest_wmh_run(root: Path) -> Path:
    group = _latest_wmh_run_group(root)
    if len(group) == 1:
        return group[0].resolve()

    prefix = re.sub(r"_gpu\d+$", "", group[0].name)
    merged = group[0].parent / f"{prefix}_merged"
    if merged.exists() or merged.is_symlink():
        if merged.is_dir() and not merged.is_symlink():
            shutil.rmtree(merged)
        else:
            merged.unlink()
    for dirname in ["videos", "tasks"]:
        (merged / dirname).mkdir(parents=True, exist_ok=True)
    metadata = {
        "sources": [str(path.resolve()) for path in group],
        "video_count": 0,
    }

    for source in group:
        worker_match = re.search(r"_(gpu\d+)$", source.name)
        worker = worker_match.group(1) if worker_match else source.name
        for video in sorted((source / "videos").glob("*")):
            if not video.is_file():
                continue
            target_name = f"{worker}_{video.name}"
            _safe_symlink(merged / "videos" / target_name, video)
            metadata["video_count"] += 1
        for task in sorted((source / "tasks").glob("*.json")):
            _safe_symlink(merged / "tasks" / f"{worker}_{task.name}", task)
        for filename in [
            "streaming_worker_config.yaml",
            "manifest.jsonl",
            "video_manifest.jsonl",
        ]:
            path = source / filename
            if path.is_file():
                _safe_symlink(merged / f"{worker}_{filename}", path)

    (merged / "merge_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Merged {len(group)} WM-H worker runs into {merged} ({metadata['video_count']} videos)")
    return merged.resolve()


def _training_config_path(config: dict, task: str) -> Path:
    train = config.get("train", {})
    configured = str(train.get("config_path", "")).strip()
    if task == "pretrain" and configured in {"", DEFAULT_FINETUNE_CONFIG}:
        path = Path(DEFAULT_PRETRAIN_CONFIG)
    elif task == "finetune" and configured in {"", DEFAULT_PRETRAIN_CONFIG}:
        path = Path(DEFAULT_FINETUNE_CONFIG)
    elif configured:
        path = Path(configured)
    elif task == "pretrain":
        path = Path(DEFAULT_PRETRAIN_CONFIG)
    else:
        path = Path(DEFAULT_FINETUNE_CONFIG)
    if not path.is_absolute():
        path = repo_root(config) / path
    return path.resolve()


def write_training_config_override(
    base_config_path: Path,
    output_path: Path,
    *,
    data_root_dir: Path | None = None,
    data_mix: str | None = None,
    max_steps: int | None = None,
    save_final_checkpoint: bool | None = None,
    augmentation: bool | None = None,
) -> Path:
    payload = json.loads(base_config_path.read_text(encoding="utf-8"))
    if data_root_dir is not None:
        payload.setdefault("train_dataset", {})["data_root_dir"] = str(data_root_dir)
    if data_mix is not None:
        payload.setdefault("train_dataset", {})["data_mix"] = data_mix
    if max_steps is not None:
        payload.setdefault("trainer", {})["max_steps"] = int(max_steps)
        payload.setdefault("trainer", {})["max_epochs"] = max(
            int(payload.get("trainer", {}).get("max_epochs", 1)),
            2,
        )
    if save_final_checkpoint is not None:
        payload["save_final_checkpoint"] = bool(save_final_checkpoint)
    if augmentation is not None:
        payload.setdefault("train_dataset", {})["augmentation"] = bool(augmentation)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=4), encoding="utf-8")
    return output_path.resolve()


def link_g1_dataset_for_generated_mix(root: Path, config: dict, training_root: Path) -> None:
    train_cfg = config.get("train", {})
    configured = str(train_cfg.get("g1_dataset_root", "")).strip()
    if configured:
        source = Path(configured).expanduser()
        if not source.is_absolute():
            source = (root / source).resolve()
    else:
        source = Path(config.get("paths", {}).get("debug_assets_root", root / "assets/debug_eval")) / "g1_dataset"
    source = source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Cannot mix G1 data; missing g1_dataset root: {source}")
    target = training_root / "g1_dataset"
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source:
            return
        if target.is_dir() and not target.is_symlink():
            raise FileExistsError(f"Refusing to replace existing non-symlink G1 dataset: {target}")
        target.unlink()
    target.symlink_to(source, target_is_directory=True)


def _dataset_names(data_mix: str) -> list[str]:
    mixtures = {
        "magic_mix": ["ego4d_cooking_and_cleaning", "egoexo4d", "epic", "ssv2", "ego4d_other"],
        "magic_mix_cooking_and_cleaning": ["ego4d_cooking_and_cleaning", "egoexo4d", "epic", "ssv2"],
        "real_only": ["g1_dataset"],
        "WM-H_50k": ["WM-H", "g1_dataset"],
    }
    return mixtures.get(data_mix, [data_mix])


def _human_annotation_paths(data_root: Path, dataset_name: str) -> tuple[Path, Path] | None:
    folder_names = {
        "ego4d_cooking_and_cleaning": "ego4d_cooking_and_cleaning",
        "egoexo4d": "egoexo4d",
        "epic": "epic",
        "ssv2": "ssv2",
        "ego4d_other": "ego4d_other",
        "WM-H": "WM-H",
        "hoi4d": "hoi4d",
        "hot3d": "hot3d",
    }
    folder = folder_names.get(dataset_name)
    if not folder:
        return None
    annotation_root = data_root / "Annotation" / folder / "episodic_annotations"
    index_path = data_root / "Annotation" / folder / "episode_frame_index.npz"
    return annotation_root, index_path


def ensure_training_indices(config: dict, task: str) -> None:
    train = config.get("train", {})
    if not train.get("auto_build_indices", True) or task == "eval":
        return
    root = repo_root(config)
    training_config = _training_config_path(config, task)
    if not training_config.exists():
        raise FileNotFoundError(f"Training config does not exist: {training_config}")
    payload = json.loads(training_config.read_text(encoding="utf-8"))
    dataset_cfg = payload.get("train_dataset", {})
    data_root = Path(dataset_cfg.get("data_root_dir", "")).expanduser()
    if not data_root.is_absolute():
        data_root = (training_config.parent / data_root).resolve()
    data_mix = str(dataset_cfg.get("data_mix", ""))
    required = bool(train.get("require_indices", True))
    missing: list[str] = []

    for dataset_name in _dataset_names(data_mix):
        if dataset_name == "g1_dataset":
            g1_root = data_root / "g1_dataset"
            if g1_root.exists():
                cmd = ["uv", "run", "python", "tools/dataset/build_g1_index.py", str(g1_root), "--verify"]
                if config.get("dataset_index", {}).get("use_all_frames", False):
                    cmd.append("--use-all-frames")
                else:
                    cmd.extend(["--target-frames", str(config.get("dataset_index", {}).get("target_frames", 81))])
                run(cmd, cwd=root)
            else:
                missing.append(f"{dataset_name}: expected {g1_root}")
            continue

        human_paths = _human_annotation_paths(data_root, dataset_name)
        if human_paths is None:
            missing.append(f"{dataset_name}: no index rule")
            continue
        annotation_root, index_path = human_paths
        if annotation_root.exists():
            run(
                [
                    "uv",
                    "run",
                    "python",
                    "tools/dataset/build_episode_index.py",
                    str(annotation_root),
                    "--output",
                    str(index_path),
                    "--verify",
                ],
                cwd=root,
            )
        else:
            missing.append(f"{dataset_name}: expected {annotation_root}")

    if missing and required:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Cannot start training because required dataset roots are missing:\n"
            f"  - {joined}\n"
            f"Update {training_config} train_dataset.data_root_dir/data_mix or set train.require_indices=false."
        )
    if missing:
        print("WARN: skipped missing dataset roots:\n  - " + "\n  - ".join(missing))


def _latest_checkpoint(root: Path) -> tuple[Path, Path]:
    candidates = sorted(
        (root / "outputs" / "vitra").glob("**/checkpoints/*/weights.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError("No VITRA checkpoint weights.pt found under outputs/vitra")
    ckpt = candidates[-1].resolve()
    config_path = ckpt.parents[2] / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing training config beside checkpoint: {config_path}")
    return ckpt, config_path.resolve()


def _annotation_for_video(run_dir: Path, video: Path) -> Path:
    annotation_dir = run_dir / "episodic_annotations"
    candidates = sorted(annotation_dir.glob(f"*_{video.stem}_ep_*.npy"))
    if not candidates:
        candidates = sorted(path for path in annotation_dir.glob("*_ep_*.npy") if video.stem in path.stem)
    if not candidates:
        raise FileNotFoundError(f"No annotation found for video {video.name} in {annotation_dir}")
    return candidates[0].resolve()


def run_post_training_checks(root: Path, run_dir: Path, config: dict, env: dict[str, str]) -> dict[str, str]:
    ckpt, train_config = _latest_checkpoint(root)
    g1_root = Path(config.get("train", {}).get("g1_dataset_root") or root / "assets" / "debug_eval" / "g1_dataset").resolve()
    wmh_vis = run_dir / "visualize_wmh_render_hand"
    g1_vis = root / "validation_outputs" / "g1_render_hand_episode"
    wmh_pred = root / "validation_outputs" / "pred_vs_gt_wmh"
    g1_pred = root / "validation_outputs" / "pred_vs_gt_g1"

    run(
        [
            "bash",
            "scripts/run_all.sh",
            "--stage",
            "visualize",
            "--input-path",
            str(run_dir / "vitra_training_data"),
            "--output-path",
            str(wmh_vis),
        ],
        cwd=root,
        env={**env, "RENDER_HAND": "1", "MAX_EPISODES": "1"},
    )
    run(
        [
            "bash",
            "scripts/run_all.sh",
            "--stage",
            "visualize",
            "--input-path",
            str(g1_root),
            "--output-path",
            str(g1_vis),
        ],
        cwd=root,
        env={**env, "RENDER_HAND": "1", "WHOLE_EPISODE": "1", "COMPARE_ACTION": "0", "NUM_SAMPLES": "1"},
    )

    # Keep the merged-run symlink path so the video stem still matches the
    # annotation naming convention, e.g. gpu0_instr000000 -> WM-H_gpu0_instr...
    video = sorted((run_dir / "videos").glob("*.mp4"))[0].absolute()
    annotation = _annotation_for_video(run_dir, video)
    run(
        [
            "bash",
            "scripts/run_eval_pipeline.sh",
            "annotation_default",
            "--mode",
            "annotation",
            "--video-path",
            str(video),
            "--annotation-npy",
            str(annotation),
            "--frame-idx",
            "0",
            "--config",
            str(train_config),
            "--model-path",
            str(ckpt),
            "--statistics-path",
            str(root / "weights" / "statistics" / "dataset_statistics.json"),
            "--output-dir",
            str(wmh_pred),
            "--output-videos-dir",
            str(wmh_pred),
            "--sample-times",
            "1",
            "--mano-path",
            "weights/mano",
        ],
        cwd=root,
        env={**env, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]},
    )
    run(
        [
            "bash",
            "scripts/run_eval_pipeline.sh",
            "annotation_default",
            "--mode",
            "dataset",
            "--root-dir",
            str(g1_root),
            "--sample-idx",
            "0",
            "--config",
            str(train_config),
            "--model-path",
            str(ckpt),
            "--statistics-path",
            str(g1_root / "g1_dataset_angle_statistics.json"),
            "--output-dir",
            str(g1_pred),
            "--output-videos-dir",
            str(g1_pred),
            "--sample-times",
            "1",
            "--mano-path",
            "weights/mano",
        ],
        cwd=root,
        env={**env, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]},
    )

    outputs = {
        "wmh_run": str(run_dir),
        "training_data": str(run_dir / "vitra_training_data"),
        "training_config": str(train_config),
        "checkpoint": str(ckpt),
        "wmh_render": str(next(wmh_vis.glob("*.mp4"))),
        "g1_render": str(g1_vis / "hand_rendered_video.mp4"),
        "wmh_pred_vs_gt": str(next(wmh_pred.glob("*_pred_vs_gt.mp4"))),
        "g1_pred_vs_gt": str(next(g1_pred.glob("*_pred_vs_gt.mp4"))),
    }
    summary_path = run_dir / "workflow_outputs.json"
    summary_path.write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Workflow outputs:")
    print(json.dumps(outputs, indent=2, ensure_ascii=False))
    return outputs


def run_objective(config: dict) -> None:
    root = repo_root(config)
    env = workflow_env(config)
    objective = config["project"]["objective"]

    if objective == "debug_eval":
        cmd = ["bash", "scripts/run_eval_pipeline.sh", config["debug_eval"].get("preset", "annotation_default")]
        cmd.extend(config["debug_eval"].get("extra_args", []))
        run(cmd, cwd=root, env=env)
        return
    if objective == "wmh":
        run(["bash", "scripts/run_all.sh", "--stage", "wmh", "--profile", env["PROFILE"], "--total-instructions", env["TOTAL_INSTRUCTIONS"]], cwd=root, env=env)
        return
    if objective == "annotate":
        run(["bash", "scripts/run_all.sh", "--stage", "annotate", "--input-path", env["INPUT_PATH"], "--parallel-k", env["PARALLEL_K"]], cwd=root, env=env)
        return
    if objective == "train_finetune":
        ensure_training_indices(config, "finetune")
        env["CONFIG"] = str(_training_config_path(config, "finetune"))
        run(["bash", "scripts/run_all.sh", "--stage", "train", "--task", "finetune"], cwd=root, env=env)
        return
    if objective == "train_pretrain":
        ensure_training_indices(config, "pretrain")
        env["CONFIG"] = str(_training_config_path(config, "pretrain"))
        run(["bash", "scripts/run_all.sh", "--stage", "train", "--task", "pretrain"], cwd=root, env=env)
        return
    if objective == "train_eval":
        run(["bash", "scripts/run_all.sh", "--stage", "train", "--task", "eval"], cwd=root, env=env)
        return
    if objective == "build_episode_index":
        target = config["dataset_index"].get("output_path", "")
        cmd = ["uv", "run", "python", "tools/dataset/build_episode_index.py", config["dataset_index"]["input_path"]]
        if target:
            cmd.extend(["--output", target])
        run(cmd, cwd=root, env=env)
        return
    if objective == "split_wmh_annotations":
        cmd = [
            "uv",
            "run",
            "python",
            "tools/dataset/split_wmh_episodes.py",
            config["dataset_index"]["input_path"],
            "--right-hand-threshold",
            str(config["dataset_index"].get("right_hand_threshold", 0.5)),
            "--max-episodes",
            str(config["dataset_index"].get("max_episodes", -1)),
        ]
        if config["dataset_index"].get("recursive", False):
            cmd.append("--recursive")
        run(cmd, cwd=root, env=env)
        return
    if objective == "build_g1_index":
        cmd = ["uv", "run", "python", "tools/dataset/build_g1_index.py", config["dataset_index"]["input_path"]]
        target = config["dataset_index"].get("output_path", "")
        if target:
            cmd.extend(["--output", target])
        if config["dataset_index"].get("use_all_frames", False):
            cmd.append("--use-all-frames")
        else:
            cmd.extend(["--target-frames", str(config["dataset_index"].get("target_frames", 81))])
        run(cmd, cwd=root, env=env)
        return
    if objective == "full_pipeline":
        if env["INPUT_PATH"]:
            run_dir = Path(env["INPUT_PATH"]).expanduser()
        else:
            run(["bash", "scripts/run_all.sh", "--stage", "wmh", "--profile", env["PROFILE"], "--total-instructions", env["TOTAL_INSTRUCTIONS"]], cwd=root, env=env)
            run_dir = materialize_latest_wmh_run(root)
        if not run_dir.is_absolute():
            run_dir = (root / run_dir).resolve()
        run(["bash", "scripts/run_all.sh", "--stage", "annotate", "--input-path", str(run_dir), "--parallel-k", env["PARALLEL_K"]], cwd=root, env=env)
        run(
            [
                "bash",
                "scripts/run_all.sh",
                "--stage",
                "hand_edit",
                "--input-path",
                str(run_dir / "videos"),
                "--hand-edit-every-n",
                env["HAND_EDIT_EVERY_N"],
            ],
            cwd=root,
            env=env,
        )
        run(
            [
                "bash",
                "scripts/run_all.sh",
                "--stage",
                "prepare_data",
                "--input-path",
                str(run_dir),
                "--robot-prob",
                env["ROBOT_PROB"],
            ],
            cwd=root,
            env=env,
        )
        print(f"Prepared WM-H training data: {run_dir / 'vitra_training_data'}")
        training_config = _training_config_path(config, env["TASK"])
        train_cfg = config.get("train", {})
        if env["TASK"] == "finetune" and train_cfg.get("use_generated_wmh_data", True):
            training_root = run_dir / "vitra_training_data"
            link_g1_dataset_for_generated_mix(root, config, training_root)
            training_config = write_training_config_override(
                training_config,
                run_dir / "vitra_training_config.json",
                data_root_dir=training_root,
                data_mix=str(train_cfg.get("generated_data_mix", "WM-H_50k")),
                max_steps=int(train_cfg["max_steps"]) if train_cfg.get("max_steps") is not None else None,
                save_final_checkpoint=bool(train_cfg.get("save_final_checkpoint", True)),
                augmentation=bool(train_cfg.get("augmentation", False)),
            )
            config = {
                **config,
                "train": {
                    **train_cfg,
                    "config_path": str(training_config),
                },
            }
        ensure_training_indices(config, env["TASK"])
        env["CONFIG"] = str(training_config)
        run(["bash", "scripts/run_all.sh", "--stage", "train", "--task", env["TASK"]], cwd=root, env=env)
        if env["TASK"] == "finetune":
            run_post_training_checks(root, run_dir, config, env)
        return
    raise ValueError(f"Unsupported objective: {objective}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Wh0 from one YAML config")
    parser.add_argument("action", choices=["summary", "setup", "weights", "run", "all"])
    parser.add_argument("--config", default="configs/project_request.yaml")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    if args.action == "summary":
        print(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        return
    if args.action in {"setup", "all"}:
        setup_env(config)
    if args.action in {"weights", "all"}:
        sync_weights(config)
    if args.action in {"run", "all"}:
        run_objective(config)


if __name__ == "__main__":
    main()

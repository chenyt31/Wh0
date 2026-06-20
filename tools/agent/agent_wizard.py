#!/usr/bin/env python3
"""Semi-automatic Wh0 setup flow for humans and AI agents."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/project_request.yaml"
DEFAULT_STATE = REPO_ROOT / ".wh0/agent_state.yaml"
DEFAULT_FINETUNE_CONFIG = "vitra-wh0/vitra/configs/robot_finetune_wmh.json"
DEFAULT_PRETRAIN_CONFIG = "vitra-wh0/vitra/configs/human_pretrain.json"


OBJECTIVES = {
    "debug_eval",
    "wmh",
    "annotate",
    "train_finetune",
    "train_pretrain",
    "train_eval",
    "build_episode_index",
    "split_wmh_annotations",
    "build_g1_index",
    "full_pipeline",
}


def load_yaml(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return default or {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload or {}


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def ask(prompt: str, current: Any = "", *, assume_defaults: bool = False) -> str:
    current_str = "" if current is None else str(current)
    suffix = f" [{current_str}]" if current_str else ""
    if assume_defaults or not sys.stdin.isatty():
        print(f"{prompt}{suffix}: {current_str}")
        return current_str
    answer = input(f"{prompt}{suffix}: ").strip()
    return answer if answer else current_str


def ask_bool(prompt: str, current: bool, *, assume_defaults: bool = False) -> bool:
    default = "y" if current else "n"
    answer = ask(f"{prompt} (y/n)", default, assume_defaults=assume_defaults).lower()
    return answer in {"y", "yes", "1", "true", "on"}


def set_if_answered(container: dict[str, Any], key: str, value: str) -> None:
    if value != "":
        container[key] = value


def configure(config: dict[str, Any], state: dict[str, Any], *, assume_defaults: bool, reconfigure: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    if state.get("confirmed") and not reconfigure:
        print(f"Using cached agent configuration from {DEFAULT_STATE}")
        return config, state

    config.setdefault("project", {})
    config.setdefault("environment", {})
    config.setdefault("package_sources", {})
    config.setdefault("paths", {})
    config.setdefault("weights", {}).setdefault("items", {})
    config.setdefault("wmh", {})
    config.setdefault("annotate", {})
    config.setdefault("train", {})
    config.setdefault("dataset_index", {})

    config["paths"]["repo_root"] = str(REPO_ROOT)
    config["paths"].setdefault("weights_root", str(REPO_ROOT / "weights"))
    config["paths"].setdefault("debug_assets_root", str(REPO_ROOT / "assets/debug_eval"))

    objective = ask("Workflow objective", config["project"].get("objective", "debug_eval"), assume_defaults=assume_defaults)
    if objective not in OBJECTIVES:
        raise SystemExit(f"Unsupported objective {objective!r}. Use one of: {', '.join(sorted(OBJECTIVES))}")
    config["project"]["objective"] = objective

    env = config["environment"]
    env["setup_with_uv"] = ask_bool("Install/sync the uv environment when running all", bool(env.get("setup_with_uv", True)), assume_defaults=assume_defaults)
    env["sync_third_party"] = ask_bool("Sync WM-H third-party repositories during setup", bool(env.get("sync_third_party", True)), assume_defaults=assume_defaults)
    set_if_answered(env, "python_version", ask("Python version", env.get("python_version", "3.10"), assume_defaults=assume_defaults))
    set_if_answered(env, "cuda_extra", ask("CUDA extra (auto/none/cu121/cu124/cu128)", env.get("cuda_extra", "auto"), assume_defaults=assume_defaults))
    env["install_vllm"] = ask_bool("Install optional vLLM backend", bool(env.get("install_vllm", False)), assume_defaults=assume_defaults)
    env["install_visualization"] = ask_bool("Install optional visualization dependencies", bool(env.get("install_visualization", False)), assume_defaults=assume_defaults)

    weights = config["weights"]
    weights["download_missing"] = ask_bool("Download missing enabled model weights", bool(weights.get("download_missing", True)), assume_defaults=assume_defaults)

    paths = config["paths"]
    if objective in {"annotate", "full_pipeline"}:
        set_if_answered(paths, "input_path", ask("Annotation input run path", paths.get("input_path", ""), assume_defaults=assume_defaults))
    if objective in {"build_episode_index", "split_wmh_annotations", "build_g1_index"}:
        set_if_answered(config["dataset_index"], "input_path", ask("Dataset/index input path", config["dataset_index"].get("input_path", ""), assume_defaults=assume_defaults))

    train = config["train"]
    if objective in {"train_finetune", "train_pretrain", "train_eval", "full_pipeline"}:
        default_task = train.get("task", "finetune")
        set_if_answered(train, "task", ask("Training task (finetune/pretrain/eval)", default_task, assume_defaults=assume_defaults))
        default_config = DEFAULT_PRETRAIN_CONFIG if train.get("task") == "pretrain" else DEFAULT_FINETUNE_CONFIG
        current_config = str(train.get("config_path", "")).strip()
        if train.get("task") == "pretrain" and current_config == DEFAULT_FINETUNE_CONFIG:
            current_config = default_config
        elif train.get("task") == "finetune" and current_config == DEFAULT_PRETRAIN_CONFIG:
            current_config = default_config
        set_if_answered(train, "config_path", ask("Training config path", current_config or default_config, assume_defaults=assume_defaults))
        train["auto_build_indices"] = ask_bool("Build/verify dataset indices before training", bool(train.get("auto_build_indices", True)), assume_defaults=assume_defaults)
        train["require_indices"] = ask_bool("Stop if configured training datasets are missing", bool(train.get("require_indices", True)), assume_defaults=assume_defaults)

    state.update(
        {
            "confirmed": True,
            "confirmed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "config_path": str(DEFAULT_CONFIG),
            "repo_root": str(REPO_ROOT),
            "objective": config["project"]["objective"],
        }
    )
    return config, state


def run_command(action: str, config_path: Path) -> int:
    if action == "quick":
        cmd = ["bash", "tools/validation/validate_environment.sh", "quick"]
        print("Running:", " ".join(cmd))
        return subprocess.run(cmd, cwd=REPO_ROOT, env=os.environ.copy()).returncode
    else:
        cmd = ["bash", "scripts/run_from_config.sh", action]
    print("Running:", " ".join(cmd))
    env = os.environ.copy()
    env["CONFIG_PATH"] = str(config_path.relative_to(REPO_ROOT) if config_path.is_relative_to(REPO_ROOT) else config_path)
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env).returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Configure Wh0 once, cache answers, and run the selected pipeline action.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Project request YAML path.")
    parser.add_argument("--state", default=str(DEFAULT_STATE), help="Agent state cache path.")
    parser.add_argument("--reconfigure", action="store_true", help="Ask questions again even if state is cached.")
    parser.add_argument("--assume-defaults", action="store_true", help="Do not prompt; accept current YAML values.")
    parser.add_argument(
        "--run",
        choices=["none", "summary", "setup", "weights", "run", "all", "quick"],
        default="none",
        help="Optional action to run after configuration.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    state_path = Path(args.state).expanduser()
    if not state_path.is_absolute():
        state_path = (REPO_ROOT / state_path).resolve()

    config = load_yaml(config_path)
    state = load_yaml(state_path)
    config, state = configure(config, state, assume_defaults=args.assume_defaults, reconfigure=args.reconfigure)
    save_yaml(config_path, config)
    save_yaml(state_path, state)
    print(f"Wrote {config_path}")
    print(f"Wrote {state_path}")

    if args.run == "none":
        print("Next: bash scripts/agent_run.sh --run quick")
        return 0
    return run_command(args.run, config_path)


if __name__ == "__main__":
    raise SystemExit(main())

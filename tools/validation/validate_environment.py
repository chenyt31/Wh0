#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""


class Validator:
    def __init__(self, level: str, config_path: Path):
        self.level = level
        self.config_path = config_path
        self.config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.settings = self.config.get("validation", {})
        self.results: list[CheckResult] = []
        self.output_root = Path(self.settings.get("output_root", REPO_ROOT / "validation_outputs")).expanduser()
        self.assets_root = Path(self.settings.get("debug_assets_root", REPO_ROOT / "assets/debug_eval")).expanduser()

    def record(self, name: str, status: str, detail: str = "") -> None:
        self.results.append(CheckResult(name=name, status=status, detail=detail))
        suffix = f" - {detail}" if detail else ""
        print(f"{status.upper():4} {name}{suffix}")

    def run(self, name: str, fn: Callable[[], str | None]) -> None:
        try:
            detail = fn() or ""
            self.record(name, "pass", detail)
        except SkipCheck as exc:
            self.record(name, "skip", str(exc))
        except Exception as exc:
            self.record(name, "fail", str(exc))

    def command(
        self,
        cmd: list[str],
        *,
        cwd: Path = REPO_ROOT,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> str:
        merged = os.environ.copy()
        if env:
            merged.update(env)
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=merged,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
            raise RuntimeError(output[-4000:] if output else f"command failed: {' '.join(cmd)}")
        return result.stdout.strip()

    def validate(self) -> int:
        self.output_root.mkdir(parents=True, exist_ok=True)
        for name, fn in self.quick_checks():
            self.run(name, fn)
        if self.level in {"runtime", "models"}:
            for name, fn in self.runtime_checks():
                self.run(name, fn)
        if self.level == "models":
            for name, fn in self.model_checks():
                self.run(name, fn)

        failed = [result for result in self.results if result.status == "fail"]
        summary = {"pass": 0, "skip": 0, "fail": 0}
        for result in self.results:
            summary[result.status] += 1
        report_path = self.output_root / f"validation_{self.level}.json"
        report_path.write_text(
            json.dumps(
                {"level": self.level, "summary": summary, "results": [result.__dict__ for result in self.results]},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nReport: {report_path}")
        print(f"Summary: pass={summary['pass']} skip={summary['skip']} fail={summary['fail']}")
        return 1 if failed else 0

    def quick_checks(self) -> list[tuple[str, Callable[[], str | None]]]:
        return [
            ("repo layout", self.check_repo_layout),
            ("validation config", self.check_validation_config),
            ("debug assets", self.check_debug_assets),
            ("extract debug g1 asset", self.extract_debug_g1_asset),
            ("shell syntax", self.check_shell_syntax),
            ("python compile", self.check_python_compile),
            ("WM-H word database unit test", self.check_wmh_unit_test),
            ("episode index builder", self.check_episode_index_builder),
            ("G1 index builder", self.check_g1_index_builder),
            ("third-party layout", self.check_third_party_layout),
            ("enabled weight paths", self.check_enabled_weight_paths),
        ]

    def runtime_checks(self) -> list[tuple[str, Callable[[], str | None]]]:
        return [
            ("uv python imports", self.check_uv_imports),
            ("RoboDatasetCore debug sample", self.check_robot_dataset_sample),
        ]

    def model_checks(self) -> list[tuple[str, Callable[[], str | None]]]:
        return [
            ("CUDA availability", self.check_cuda),
            ("model weight paths", self.check_model_weight_paths),
            ("debug visualization", self.check_debug_visualization),
            ("pretrained policy inference", self.check_pretrained_inference),
        ]

    def check_repo_layout(self) -> str:
        required = [
            "configs/project_request.yaml",
            "configs/project_request.example.yaml",
            "configs/weights_manifest.json",
            "configs/validation.yaml",
            "scripts/setup.sh",
            "scripts/run_from_config.sh",
            "tools/validation/validate_environment.sh",
            "tools/dataset/build_g1_index.py",
            "tools/dataset/build_episode_index.py",
            "scripts/run_eval_pipeline.sh",
            "scripts/run_human_inference.sh",
            "libs/dataset_index/g1.py",
            "vitra-wh0/vitra/datasets/robot_dataset.py",
            "WM-H/tests/test_word_database.py",
        ]
        missing = [path for path in required if not (REPO_ROOT / path).exists()]
        if missing:
            raise RuntimeError(f"missing: {missing}")
        return f"{len(required)} required paths"

    def check_validation_config(self) -> str:
        if "validation" not in self.config:
            raise RuntimeError("missing validation root")
        return str(self.config_path)

    def check_debug_assets(self) -> str:
        required = [
            self.assets_root / "wm-h/videos/gpu0_task000000.mp4",
            self.assets_root / "wm-h/annotations/WM-H_gpu0_task000000_ep_000000.npy",
            self.assets_root / "g1_dataset/episode_0001.tar.gz",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"missing: {missing}")
        return str(self.assets_root)

    def extract_debug_g1_asset(self) -> str:
        archive = self.assets_root / "g1_dataset/episode_0001.tar.gz"
        target = self.assets_root / "g1_dataset/episode_0001"
        if target.exists() and (target / "data.json").exists() and (target / "colors").exists():
            return "already extracted"
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(self.assets_root / "g1_dataset")
        return "extracted episode_0001"

    def check_shell_syntax(self) -> str:
        scripts = [
            "scripts/run_from_config.sh",
            "scripts/run_eval_pipeline.sh",
            "scripts/run_human_inference.sh",
            "tools/validation/validate_environment.sh",
            "scripts/run_train.sh",
            "scripts/run_all.sh",
        ]
        self.command(["bash", "-n", *scripts])
        return f"{len(scripts)} scripts"

    def check_python_compile(self) -> str:
        targets = [
            "tools/orchestrator/project_orchestrator.py",
            "tools/weights/manage_weights.py",
            "tools/validation/validate_environment.py",
            "tools/dataset/build_g1_index.py",
            "tools/dataset/build_episode_index.py",
            "tools/dataset/split_wmh_episodes.py",
            "libs/dataset_index/g1.py",
            "libs/dataset_index/episodic.py",
            "libs/dataset_index/wmh_split.py",
            "vitra-wh0/vitra/datasets/robot_dataset.py",
            "vitra-wh0/vitra/tools/human_prediction_pipeline.py",
        ]
        self.command([sys.executable, "-m", "py_compile", *targets])
        return f"{len(targets)} files"

    def check_wmh_unit_test(self) -> str:
        self.command([sys.executable, "tests/test_word_database.py"], cwd=REPO_ROOT / "WM-H")
        return "word database"

    def check_episode_index_builder(self) -> str:
        with tempfile.TemporaryDirectory(prefix="wh0_episode_index_") as tmp:
            output = Path(tmp) / "episode_frame_index.npz"
            self.command(
                [
                    sys.executable,
                    "tools/dataset/build_episode_index.py",
                    str(self.assets_root / "wm-h/annotations"),
                    "--output",
                    str(output),
                    "--verify",
                ]
            )
            if not output.exists():
                raise RuntimeError("index file was not written")
        return "wm-h annotation asset"

    def check_g1_index_builder(self) -> str:
        with tempfile.TemporaryDirectory(prefix="wh0_g1_index_") as tmp:
            tmp_root = Path(tmp) / "g1_dataset"
            shutil.copytree(self.assets_root / "g1_dataset/episode_0001", tmp_root / "episode_0001")
            output = Path(tmp) / "training_index.npz"
            self.command(
                [
                    sys.executable,
                    "tools/dataset/build_g1_index.py",
                    str(tmp_root),
                    "--output",
                    str(output),
                    "--target-frames",
                    "8",
                    "--verify",
                ]
            )
            if not output.exists():
                raise RuntimeError("training index was not written")
        return "debug G1 sample"

    def check_third_party_layout(self) -> str:
        required = [
            REPO_ROOT / "WM-H/third_party/DiffSynth-Studio",
            REPO_ROOT / "WM-H/third_party/HaWoR",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"run scripts/setup.sh first; missing {missing}")
        commit = self.command(["git", "rev-parse", "HEAD"], cwd=required[0])
        expected = "a84251704224d6189f695ce72c3d834b7f84557c"
        if commit != expected:
            raise RuntimeError(f"DiffSynth-Studio commit {commit} != {expected}")
        return "DiffSynth-Studio + HaWoR"

    def check_enabled_weight_paths(self) -> str:
        project_cfg = yaml.safe_load((REPO_ROOT / "configs/project_request.yaml").read_text(encoding="utf-8"))
        weights_root = Path(project_cfg["paths"]["weights_root"]).expanduser()
        manifest = json.loads((REPO_ROOT / "configs/weights_manifest.json").read_text(encoding="utf-8"))
        targets = {item["id"]: weights_root / item["target"] for item in manifest}
        enabled = [
            name
            for name, item in project_cfg.get("weights", {}).get("items", {}).items()
            if item.get("enabled", False)
        ]
        missing = [f"{name}:{targets[name]}" for name in enabled if name in targets and not targets[name].exists()]
        if missing:
            raise RuntimeError(f"missing enabled weights: {missing}")
        return f"{len(enabled)} enabled weights"

    def check_uv_imports(self) -> str:
        code = (
            "import torch\n"
            "import vitra\n"
            "from vitra.datasets.robot_dataset import RoboDatasetCore\n"
            "print('imports-ok')\n"
        )
        self.command(["uv", "run", "python", "-c", code], cwd=REPO_ROOT / "vitra-wh0", env={"UV_PROJECT": str(REPO_ROOT)})
        return "vitra-wh0 imports"

    def check_robot_dataset_sample(self) -> str:
        with tempfile.TemporaryDirectory(prefix="wh0_robot_dataset_") as tmp:
            tmp_root = Path(tmp) / "g1_dataset"
            shutil.copytree(self.assets_root / "g1_dataset/episode_0001", tmp_root / "episode_0001")
            index_path = Path(tmp) / "training_index.npz"
            self.command(
                [
                    sys.executable,
                    "tools/dataset/build_g1_index.py",
                    str(tmp_root),
                    "--output",
                    str(index_path),
                    "--target-frames",
                    "8",
                    "--verify",
                ]
            )
            shutil.copy(index_path, tmp_root / "training_index.npz")
            code = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT / "vitra-wh0")!r})
from vitra.datasets.robot_dataset import RoboDatasetCore
dataset = RoboDatasetCore({str(tmp_root)!r}, statistics_path=None, load_images=True, augmentation=False)
sample = dataset[0]
assert sample["image_list"].shape[0] == 1
assert sample["action_list"].shape[1] == 24
print(sample["image_list"].shape, sample["action_list"].shape)
"""
            self.command(["uv", "run", "python", "-c", code], env={"UV_PROJECT": str(REPO_ROOT)}, timeout=120)
        return "loaded one G1 sample"

    def check_cuda(self) -> str:
        if not self.settings.get("require_cuda", False):
            raise SkipCheck("validation.require_cuda is false")
        code = "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
        return self.command(
            ["uv", "run", "python", "-c", code],
            cwd=REPO_ROOT / "vitra-wh0",
            env={"UV_PROJECT": str(REPO_ROOT)},
            timeout=60,
        )

    def check_model_weight_paths(self) -> str:
        manifest = json.loads((REPO_ROOT / "configs/weights_manifest.json").read_text(encoding="utf-8"))
        weights_root = Path(self.settings.get("weights_root", REPO_ROOT / "weights")).expanduser()
        required_ids = {
            "mano_right",
            "hawor_detector",
            "hawor_checkpoint",
            "hawor_model_config",
            "moge_vit_l",
            "paligemma2_3b_mix_224",
        }
        targets = {item["id"]: weights_root / item["target"] for item in manifest}
        missing = [f"{item_id}:{targets[item_id]}" for item_id in sorted(required_ids) if not targets[item_id].exists()]
        if missing:
            raise RuntimeError(f"missing model validation weights: {missing}")
        return f"{len(required_ids)} required model weights"

    def check_debug_visualization(self) -> str:
        if not self.settings.get("run_visualization", False):
            raise SkipCheck("validation.run_visualization is false")
        mano_path = Path(self.settings.get("weights_root", REPO_ROOT / "weights")).expanduser() / "mano"
        output = self.output_root / "visualize"
        self.command(
            [
                "uv",
                "run",
                "python",
                "data/demo_visualization_epi.py",
                "--video_root",
                str(self.assets_root / "wm-h/videos"),
                "--label_root",
                str(self.assets_root / "wm-h/annotations"),
                "--save_path",
                str(output),
                "--mano_model_path",
                str(mano_path),
            ],
            cwd=REPO_ROOT / "vitra-wh0",
            env={"UV_PROJECT": str(REPO_ROOT)},
            timeout=300,
        )
        return str(output)

    def check_pretrained_inference(self) -> str:
        if not self.settings.get("run_model_inference", False):
            raise SkipCheck("validation.run_model_inference is false")
        config_path = str(self.settings.get("pretrained_config_path", "")).strip()
        if not config_path:
            raise RuntimeError("validation.pretrained_config_path is required")
        model_path = str(self.settings.get("pretrained_model_path", "")).strip()
        env = {
            "IMAGE_PATH": str(Path(self.settings["inference_image_path"]).expanduser()),
            "INSTRUCTION": str(self.settings["inference_instruction"]),
            "CONFIG_PATH": config_path,
            "OUTPUT_VIDEO": str(self.output_root / "human_inference.mp4"),
            "MANO_PATH": str(Path(self.settings.get("weights_root", REPO_ROOT / "weights")).expanduser() / "mano"),
        }
        if model_path:
            env["MODEL_PATH"] = model_path
        self.command(["bash", "scripts/run_human_inference.sh"], env=env, timeout=900)
        output = self.output_root / "human_inference.mp4"
        if not output.exists():
            raise RuntimeError("inference did not produce output video")
        return str(output)


class SkipCheck(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Wh0 environment after setup.")
    parser.add_argument("level", choices=["quick", "runtime", "models"], nargs="?", default="quick")
    parser.add_argument("--config", default="configs/validation.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return Validator(args.level, REPO_ROOT / args.config).validate()


if __name__ == "__main__":
    raise SystemExit(main())

"""
Stage 2: Generate videos from manifest produced by video_prompt_preparer.py.

Each manifest line is one (image, instruction) pair with video_prompt.
Runs Wan I2V on the original image — no image editing.
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import torch
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "sage_attention")

from wm_h.video.cuda import isolate_cuda_device
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig as VideoModelConfig
from diffsynth.utils.data import save_video
from PIL import Image, ImageOps

from wm_h.video.common import (
    CheckpointManager,
    DEFAULT_NEGATIVE_PROMPT,
    build_task_json_from_entry,
    expand_manifest_rollouts,
    load_config,
    prepare_i2v_input_image,
    resolve_manifest_image_path,
    resolve_rollout_generation_params,
)
from wm_h.box_editor import parse_diffsynth_vram_config

_WAN_OFFLOAD_MODULES = (
    "dit",
    "dit2",
    "vae",
    "text_encoder",
    "image_encoder",
    "motion_controller",
    "vace",
    "vace2",
)


class ManifestVideoGenerator:
    """Wan video generation from manifest entries."""

    def __init__(
        self,
        video_model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        lightx2v_lora_path: Optional[dict] = None,
        vram_management: Optional[dict[str, Any]] = None,
    ):
        self.logger = self._setup_logging()
        self.device = device
        self.torch_dtype = torch_dtype
        self.vram_management = dict(vram_management or {})
        self.video_pipe = self._load_video_model(
            video_model_path,
            device,
            torch_dtype,
            lightx2v_lora_path,
            vram_management,
        )
        self.logger.info("Video model loaded")

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )
        return logging.getLogger(__name__)

    def _load_video_model(
        self,
        model_path: str,
        device: str,
        torch_dtype: torch.dtype,
        lightx2v_lora_path: Optional[dict] = None,
        vram_management: Optional[dict[str, Any]] = None,
    ) -> WanVideoPipeline:
        model_path = Path(model_path)
        self.logger.info(f"Loading video model from: {model_path}")

        def find_model_files(subdir: str, pattern: str = "*.safetensors") -> list:
            dir_path = model_path / subdir
            if not dir_path.exists():
                return []
            files = list(dir_path.glob(pattern))
            if not files:
                files = list(dir_path.glob("*.pth"))
            return [str(f) for f in sorted(files)]

        def find_single_file(pattern: str) -> Optional[str]:
            files = list(model_path.glob(pattern))
            return str(files[0]) if files else None

        vram_kwargs, vram_limit = parse_diffsynth_vram_config(
            vram_management,
            device=device,
            torch_dtype=torch_dtype,
        )
        model_configs = []
        high_noise_files = find_model_files("high_noise_model", "diffusion_pytorch_model*.safetensors")
        if high_noise_files:
            model_configs.append(VideoModelConfig(
                path=high_noise_files if len(high_noise_files) > 1 else high_noise_files[0],
                **vram_kwargs,
            ))
        low_noise_files = find_model_files("low_noise_model", "diffusion_pytorch_model*.safetensors")
        if low_noise_files:
            model_configs.append(VideoModelConfig(
                path=low_noise_files if len(low_noise_files) > 1 else low_noise_files[0],
                **vram_kwargs,
            ))

        text_encoder_file = find_single_file("models_t5_umt5-xxl-enc-bf16.*")
        if text_encoder_file:
            model_configs.append(VideoModelConfig(path=text_encoder_file, **vram_kwargs))

        vae_file = find_single_file("Wan2.1_VAE.*") or find_single_file("Wan2.2_VAE.*")
        if vae_file:
            model_configs.append(VideoModelConfig(path=vae_file, **vram_kwargs))

        tokenizer_path = model_path / "google" / "umt5-xxl"
        if not tokenizer_path.exists():
            tokenizer_path = model_path / "tokenizer"
        tokenizer_config = VideoModelConfig(path=str(tokenizer_path)) if tokenizer_path.exists() else None

        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            vram_limit=vram_limit,
        )

        if lightx2v_lora_path:
            high = lightx2v_lora_path.get("high_noise", "").strip()
            low = lightx2v_lora_path.get("low_noise", "").strip()
            if high:
                pipe.load_lora(pipe.dit, VideoModelConfig(path=high), alpha=1.0)
            if low:
                pipe.load_lora(pipe.dit2, VideoModelConfig(path=low), alpha=1.0)
            self.logger.info("LightX2V LoRA loaded (4-step mode)")

        return pipe

    def offload_model(self) -> None:
        """Move Wan submodules to CPU without destroying the pipeline."""
        import gc

        if self.video_pipe is None:
            return
        if self.vram_management.get("enable", False):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return
        for name in _WAN_OFFLOAD_MODULES:
            module = getattr(self.video_pipe, name, None)
            if module is not None:
                module.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def onload_model(self) -> None:
        """Move Wan submodules back to the inference device."""
        if self.video_pipe is None:
            return
        if self.vram_management.get("enable", False):
            return
        for name in _WAN_OFFLOAD_MODULES:
            module = getattr(self.video_pipe, name, None)
            if module is not None:
                module.to(self.device)

    def release_model(self) -> None:
        import gc

        self.video_pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def generate_video(
        self,
        input_image: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        seed: int = 0,
        num_inference_steps: int = 4,
        height: int = 360,
        width: int = 640,
        num_frames: int = 81,
        cfg_scale: float = 1.0,
        switch_DiT_boundary: float = 0.875,
        sigma_shift: float = 5.0,
        tiled: bool = True,
        rand_device: str = "cuda",
        input_prep_width: int = 0,
        input_prep_height: int = 0,
    ) -> list:
        if negative_prompt is None:
            negative_prompt = DEFAULT_NEGATIVE_PROMPT
        resized = prepare_i2v_input_image(
            input_image,
            width,
            height,
            prep_width=input_prep_width,
            prep_height=input_prep_height,
        )
        if input_image.size != resized.size:
            self.logger.info(
                "Prepared I2V input: %dx%d -> %dx%d (prep=%dx%d)",
                input_image.size[0],
                input_image.size[1],
                resized.size[0],
                resized.size[1],
                input_prep_width or width,
                input_prep_height or height,
            )
        self.logger.info(
            f"Generating video: {width}x{height}, frames={num_frames}, "
            f"steps={num_inference_steps}, seed={seed}, sigma_shift={sigma_shift}"
        )
        return self.video_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            input_image=resized,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            num_frames=num_frames,
            cfg_scale=cfg_scale,
            switch_DiT_boundary=switch_DiT_boundary,
            sigma_shift=sigma_shift,
            tiled=tiled,
            rand_device=rand_device,
        )

    def process_manifest_entry(
        self,
        entry: Dict,
        videos_dir: Path,
        tasks_dir: Path,
        video_steps: int,
        video_height: int,
        video_width: int,
        video_cfg_scale: float,
        video_sigma_shift: float,
        video_switch_dit_boundary: float,
        num_frames: int,
        fps: int,
        negative_prompt: str,
        input_prep_width: int = 0,
        input_prep_height: int = 0,
    ) -> Optional[Path]:
        task_id = entry["task_id"]
        image_path_str = resolve_manifest_image_path(entry)
        image_path = Path(image_path_str)
        video_prompt = entry["video_prompt"]
        hand = entry.get("hand", "right")
        use_visual_aug = entry.get("use_visual_aug", False)
        gen_params = resolve_rollout_generation_params(
            entry,
            {
                "video_sigma_shift": video_sigma_shift,
                "video_cfg_scale": video_cfg_scale,
            },
        )

        if not image_path.exists():
            self.logger.error(f"Image not found: {image_path}")
            return None

        if use_visual_aug:
            self.logger.info(f"Using scene_aug image: {image_path}")

        try:
            input_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            self.logger.error(f"Failed to open image: {e}")
            return None

        try:
            video = self.generate_video(
                input_image=input_image,
                prompt=video_prompt,
                negative_prompt=negative_prompt,
                seed=gen_params["seed"],
                num_inference_steps=video_steps,
                height=video_height,
                width=video_width,
                num_frames=num_frames,
                cfg_scale=gen_params["cfg_scale"],
                sigma_shift=gen_params["sigma_shift"],
                switch_DiT_boundary=video_switch_dit_boundary,
                rand_device=gen_params["rand_device"],
                input_prep_width=input_prep_width,
                input_prep_height=input_prep_height,
            )
        except Exception as e:
            self.logger.exception("Video generation failed: %s", e)
            return None

        if hand == "left":
            video = [ImageOps.mirror(frame) for frame in video]

        video_path = videos_dir / f"{task_id}.mp4"
        save_video(video, str(video_path), fps=fps, quality=5)
        self.logger.info(f"Saved {video_path}")

        task_info = build_task_json_from_entry({**entry, "task_id": task_id})
        task_json = tasks_dir / f"{task_id}.json"
        with open(task_json, "w", encoding="utf-8") as f:
            json.dump(task_info, f, ensure_ascii=False, indent=2)

        return video_path


def read_manifest(manifest_path: str, k: int = -1) -> List[Dict]:
    path = Path(manifest_path)
    if not path.exists():
        print(f"Manifest not found: {path}")
        return []
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if k != -1 and i >= k:
                break
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Bad manifest line {i + 1}: {e}")
    print(f"Loaded {len(entries)} manifest entries")
    return entries


def _manifest_task_id(entry: Dict) -> str:
    return entry.get("task_id", "")


def _worker_generate(args: Tuple) -> int:
    entries, gpu_id, visible_gpus, config, output_dirs, checkpoint_info = args
    device = isolate_cuda_device(gpu_id, visible_gpus)
    print(f"[GPU {gpu_id}] Video worker on {device}, {len(entries)} entries")

    model_cfg = config.get("model", {})
    video_cfg = config.get("video", {})
    checkpoint_cfg = config.get("checkpoint", {})

    ckpt = CheckpointManager(
        checkpoint_info.get("checkpoint_dir", "database/log"),
        checkpoint_name=checkpoint_cfg.get("checkpoint_name", "processed_video_manifest.txt"),
    )

    gen = ManifestVideoGenerator(
        video_model_path=model_cfg["video_model_path"],
        device=device,
        lightx2v_lora_path=model_cfg.get("lightx2v_lora_path"),
        vram_management=model_cfg.get("vram_management") or {},
    )

    videos_dir = Path(output_dirs["videos_dir"])
    tasks_dir = Path(output_dirs["tasks_dir"])
    negative_prompt = config.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)

    success = 0
    for entry in entries:
        task_id = _manifest_task_id(entry)
        print(f"[GPU {gpu_id}] {task_id}")
        result = gen.process_manifest_entry(
            entry=entry,
            videos_dir=videos_dir,
            tasks_dir=tasks_dir,
            video_steps=video_cfg.get("video_steps", 4),
            video_height=video_cfg.get("video_height", 360),
            video_width=video_cfg.get("video_width", 640),
            video_cfg_scale=video_cfg.get("video_cfg_scale", 1.0),
            video_sigma_shift=video_cfg.get("video_sigma_shift", 5.0),
            video_switch_dit_boundary=video_cfg.get("video_switch_dit_boundary", 0.875),
            num_frames=video_cfg.get("num_frames", 81),
            fps=video_cfg.get("fps", 15),
            negative_prompt=negative_prompt,
            input_prep_width=int(video_cfg.get("input_prep_width", 0) or 0),
            input_prep_height=int(video_cfg.get("input_prep_height", 0) or 0),
        )
        if result is not None:
            success += 1
            if checkpoint_info.get("enabled", True):
                ckpt.mark_as_processed(task_id, gpu_id)
    print(f"[GPU {gpu_id}] Success {success}/{len(entries)}")
    return success


def main():
    parser = argparse.ArgumentParser(description="Stage 2: generate videos from manifest")
    parser.add_argument("--config", type=str, default="config/config_video_generator_manifest.yaml")
    parser.add_argument("--manifest", type=str, default=None, help="Override manifest JSONL path")
    parser.add_argument("--reset-checkpoint", action="store_true")
    parser.add_argument(
        "--rollouts",
        type=int,
        default=None,
        help="Override task.rollouts_per_task (multiple videos per manifest instruction)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.rollouts is not None:
        config.setdefault("task", {})["rollouts_per_task"] = max(1, args.rollouts)
    data_cfg = config.get("data", {})
    task_cfg = config.get("task", {})
    parallel_cfg = config.get("parallel", {})
    checkpoint_cfg = config.get("checkpoint", {})

    manifest = args.manifest or data_cfg.get("manifest")
    if not manifest:
        raise ValueError("Set data.manifest in config or pass --manifest")

    output_dir = data_cfg.get("output_dir", "database/generated_videos")
    k = task_cfg.get("k", -1)
    enable_parallel = parallel_cfg.get("enable", True)
    checkpoint_enabled = checkpoint_cfg.get("enable", True)
    checkpoint_dir = checkpoint_cfg.get("dir", "database/log")
    checkpoint_name = checkpoint_cfg.get("checkpoint_name", "processed_video_manifest.txt")

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible:
        visible_gpus = [int(x) for x in cuda_visible.split(",") if x.strip()]
    else:
        visible_gpus = list(range(torch.cuda.device_count()))
    if not visible_gpus:
        raise RuntimeError("No GPU available")

    num_gpus = len(visible_gpus)
    ckpt = CheckpointManager(checkpoint_dir, checkpoint_name=checkpoint_name)
    if args.reset_checkpoint:
        ckpt.checkpoint_file.write_text("")

    entries = read_manifest(manifest, k)
    if not entries:
        return

    n_instructions = len(entries)
    entries, rollouts_per_task = expand_manifest_rollouts(entries, config)
    if rollouts_per_task > 1:
        print(
            f"Rollouts: {n_instructions} instructions × {rollouts_per_task} "
            f"= {len(entries)} generation jobs"
        )

    if checkpoint_enabled:
        processed = ckpt.load_processed_instructions()
        if processed:
            before = len(entries)
            entries = [e for e in entries if _manifest_task_id(e) not in processed]
            print(f"[Checkpoint] Skipped {before - len(entries)}, remaining {len(entries)}")
    if not entries:
        print("All manifest entries processed")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"run_{timestamp}"
    videos_dir = run_dir / "videos"
    tasks_dir = run_dir / "tasks"
    videos_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {run_dir}")

    checkpoint_info = {"checkpoint_dir": checkpoint_dir, "enabled": checkpoint_enabled}
    output_dirs = {"videos_dir": str(videos_dir), "tasks_dir": str(tasks_dir)}

    if not enable_parallel or num_gpus == 1:
        n = _worker_generate((entries, 0, visible_gpus, config, output_dirs, checkpoint_info))
        print(f"Done: {n}/{len(entries)}")
        return

    per_gpu: List[List[Dict]] = [[] for _ in range(num_gpus)]
    for idx, entry in enumerate(entries):
        per_gpu[idx % num_gpus].append(entry)

    tasks = [
        (per_gpu[g], g, visible_gpus, config, output_dirs, checkpoint_info)
        for g in range(num_gpus)
        if per_gpu[g]
    ]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(tasks)) as pool:
        results = pool.map(_worker_generate, tasks)
    print(f"Total success: {sum(results)}/{len(entries)}")


if __name__ == "__main__":
    main()

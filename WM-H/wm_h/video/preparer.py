"""
Stage 1: For each image in a folder, generate k manipulation instructions via Qwen VL,
then text-augmented descriptions, optional visual trajectory augmentation (scene_aug),
and video prompts. Writes manifest for stage 2.
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import re
import sys
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from wm_h.video.cuda import isolate_cuda_device
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from PIL import Image

from wm_h.video.common import (
    CheckpointManager,
    DEFAULT_DIVERSITY_DB_PATH,
    build_hand_trajectory_vl_prompt,
    build_text_augmentation_prompt,
    build_video_prompt,
    draw_hand_trajectories_on_image,
    extract_primary_object_key,
    filter_valid_instructions,
    format_single_arm_manifest_fields,
    is_valid_manipulation_instruction,
    list_images,
    load_config,
    normalize_instruction_schema,
    parse_hand_trajectory_json,
    parse_vl_instructions_json,
    record_manifest_entry_in_db,
    scene_aug_output_path,
)
from wm_h.vllm_backend import VLLMGenerator


def _build_instruction_generation_prompt(
    k: int,
    hand: str,
    strict: bool = False,
    used_objects: Optional[List[str]] = None,
) -> str:
    hand_name = "RIGHT" if hand == "right" else "LEFT"
    reminder = ""
    if strict:
        reminder = (
            "\nIMPORTANT: Your previous output contained invalid instructions. "
            "Do NOT use hyphenated verbs like pick-then-place. "
            "Every instruction with 'place' or 'move' MUST include WHERE using "
            "on/onto/in/next to/beside/near/toward/etc. "
            "Every 'push'/'pull'/'slide' MUST include direction or target "
            "(toward/away from/to the left/next to/etc.).\n"
        )
    if used_objects:
        used_list = ", ".join(used_objects)
        reminder += (
            f"\nIMPORTANT: These primary objects are ALREADY used in this batch — "
            f"each new instruction MUST manipulate a DIFFERENT object: {used_list}\n"
        )
    return (
        f"You are viewing a top-down desktop scene image. "
        f"Generate exactly {k} diverse, realistic manipulation instructions that a person's {hand_name} hand "
        f"could perform on objects visible in THIS image. "
        f"Tasks are completed using only the {hand_name} hand. "
        f"All {k} instructions must be different from each other. "
        f"Each instruction must target a DIFFERENT primary manipulated object — "
        f"do NOT repeat the same object across instructions "
        f"(e.g. if one picks the bottle, others must use pen, book, mug, mouse, etc.). "
        f"Vary action types across instructions — include push, slide, open, rotate, press, etc., "
        f"not only pick/grasp/place. "
        f"Reference objects that are actually visible in the image.\n"
        f"{reminder}"
        "Allowed action verbs (use diverse types):\n"
        "- Pick/grasp/lift: pick, grasp, lift the <adj> <object>\n"
        "- Place/move (MUST include destination): place/move ... on/next to/beside/in/...; "
        "pick/grasp ... then place it ...\n"
        "- Push/pull/slide (MUST include direction or target): push/pull/slide ... toward/away from/"
        "next to/to the left of/...\n"
        "- Articulation: open, close, flip, rotate, turn, twist, unfold, fold the <object/part>\n"
        "- Press/tap: press, tap the <button/key/part>\n"
        "- Fine manipulation: align, insert, remove, unscrew, screw\n\n"
        "Rules for the \"instr\" field (STRICT):\n"
        "1. Write natural English only. NEVER use hyphenated verbs like \"pick-then-place\" or \"grasp-then-place\".\n"
        "2. Start with one allowed verb; compound steps use \"then\" (e.g. pick ... then place it ...).\n"
        "3. place/move: MUST include object AND destination/spatial target.\n"
        "4. push/pull/slide: MUST include direction or spatial target.\n"
        "5. open/close/rotate/turn/flip/twist/press/tap: object must be visible in the image.\n"
        "6. Destinations and references must refer to objects or areas visible in the image.\n"
        "Output a JSON array ONLY (no markdown, no explanation). Each element must have:\n"
        '- "verb": list of verbs used (e.g. ["push"] or ["pick", "place"])\n'
        '- "noun": list of object nouns\n'
        '- "adjective": list of adjectives (use empty string \"\" if none)\n'
        '- "instr": complete English instruction sentence following the rules above\n'
        "Example format:\n"
        '[{"verb": ["push"], "noun": ["mug"], "adjective": ["white"], '
        '"instr": "push the white mug toward the laptop"}, '
        '{"verb": ["open"], "noun": ["book"], "adjective": ["light blue"], '
        '"instr": "open the light blue book"}, '
        '{"verb": ["pick"], "noun": ["pen"], "adjective": ["black"], "instr": "pick the black pen"}, '
        '{"verb": ["pick", "place"], "noun": ["bottle", "mouse"], "adjective": ["clear", ""], '
        '"instr": "pick the clear bottle then place it beside the mouse"}]\n'
        f"Generate EXACTLY {k} instructions:"
    )


class VideoPromptPreparer:
    """Per-image: generate k instructions, then augment + video prompt for each."""

    def __init__(
        self,
        qwen_vl_model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        gen_cfg: Optional[dict] = None,
        image_max_side: int = 1024,
        defer_vl_load: bool = False,
    ):
        self.logger = self._setup_logging()
        self.device = device
        self._torch_dtype = torch_dtype
        self._qwen_vl_model_path = qwen_vl_model_path
        self.gen_cfg = gen_cfg or {}
        self.backend = str(self.gen_cfg.get("backend", "transformers")).lower()
        self.image_max_side = image_max_side
        self.model = None
        self.processor = None
        self._vllm = None
        if self.backend == "vllm":
            self._vllm = VLLMGenerator(
                qwen_vl_model_path,
                gen_cfg=self.gen_cfg,
                image_max_side=image_max_side,
            )
            self.logger.info(f"Using vLLM Qwen VL backend on {device}")
            return
        if defer_vl_load:
            self.logger.info("Qwen VL load deferred (will attach shared weights)")
        else:
            self.logger.info(f"Loading Qwen VL on {device}")
            self.model, self.processor = self._load_qwen_vl(
                qwen_vl_model_path, device, torch_dtype
            )

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )
        return logging.getLogger(__name__)

    def _load_qwen_vl(self, model_path: str, device: str, torch_dtype: torch.dtype):
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map={"": device},
            trust_remote_code=True,
            local_files_only=True,
        ).eval()
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        processor.tokenizer.padding_side = "left"
        if self.image_max_side > 0:
            max_pixels = self.image_max_side * self.image_max_side
            min_pixels = 256 * 28 * 28
            processor.image_processor.size = {
                "longest_edge": max_pixels,
                "shortest_edge": min_pixels,
            }
            self.logger.info(
                f"VL image max side ~{self.image_max_side}px (max_pixels={max_pixels})"
            )
        return model, processor

    def attach_vl_model(self, model, processor) -> None:
        """Borrow an existing VL instance (e.g. from ImageAnalyzer)."""
        if self.backend == "vllm":
            return
        self.model = model
        self.processor = processor

    def _model_on_gpu(self) -> bool:
        if self.backend == "vllm":
            return True
        if self.model is None:
            return False
        return next(self.model.parameters()).device.type != "cpu"

    def offload_model(self) -> None:
        if self.backend == "vllm":
            return
        if self.model is None or not self._model_on_gpu():
            return
        self.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def release_model(self, *, destroy: bool = True) -> None:
        if self.backend == "vllm":
            if destroy and self._vllm is not None:
                self._vllm.release()
            return
        if destroy:
            self.model = None
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reload_model(self) -> None:
        if self.backend == "vllm":
            if self._vllm is None:
                self._vllm = VLLMGenerator(
                    self._qwen_vl_model_path,
                    gen_cfg=self.gen_cfg,
                    image_max_side=self.image_max_side,
                )
            return
        if self.model is not None and self.processor is not None:
            if not self._model_on_gpu():
                self.model.to(self.device)
            return
        self.model, self.processor = self._load_qwen_vl(
            self._qwen_vl_model_path, self.device, self._torch_dtype
        )

    @torch.no_grad()
    def _vl_generate(self, image_path: str, text_prompt: str, max_new_tokens: int) -> str:
        if self.backend == "vllm":
            if self._vllm is None:
                self.reload_model()
            return self._vllm.generate_vision_texts(
                [image_path],
                [text_prompt],
                max_new_tokens=max_new_tokens,
            )[0]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            [messages],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=self.gen_cfg.get("do_sample", True),
            temperature=self.gen_cfg.get("temperature", 0.7),
            top_p=self.gen_cfg.get("top_p", 0.9),
        )
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        texts = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return texts[0].strip() if texts else ""

    @torch.no_grad()
    def generate_instructions_on_image(
        self, image_path: str, k: int, hand: str = "right"
    ) -> List[Dict]:
        max_retries = self.gen_cfg.get("instruction_max_retries", 2)
        max_tokens = self.gen_cfg.get("instruction_max_new_tokens", 1024)
        collected: List[Dict] = []
        seen_texts: set = set()
        seen_objects: set = set()

        for attempt in range(max_retries + 1):
            need = k - len(collected)
            if need <= 0:
                break
            strict = attempt > 0 or bool(seen_objects)
            used_objects = sorted(seen_objects) if seen_objects else None
            prompt = _build_instruction_generation_prompt(
                need, hand, strict=strict, used_objects=used_objects
            )
            raw = self._vl_generate(image_path, prompt, max_tokens)
            self.logger.info(f"[GenInstr] attempt {attempt + 1} raw ({len(raw)} chars): {raw[:200]}...")

            parsed = parse_vl_instructions_json(raw, max_count=need * 2)
            if len(parsed) < need:
                parsed.extend(self._fallback_parse_instructions(raw, need * 2))

            batch = filter_valid_instructions(parsed)
            rejected = len(parsed) - len(batch)
            if rejected:
                self.logger.warning(f"[GenInstr] rejected {rejected} invalid instruction(s) on attempt {attempt + 1}")

            for item in batch:
                text = item.get("instruction", "").strip()
                key = text.lower()
                if key in seen_texts:
                    continue
                obj_key = extract_primary_object_key(item)
                if obj_key and obj_key in seen_objects:
                    self.logger.debug(
                        f"[GenInstr] skip duplicate primary object '{obj_key}': {text[:60]}"
                    )
                    continue
                seen_texts.add(key)
                if obj_key:
                    seen_objects.add(obj_key)
                collected.append(item)
                if len(collected) >= k:
                    break

            if len(collected) >= k:
                break
            if attempt < max_retries:
                self.logger.warning(
                    f"[GenInstr] only {len(collected)}/{k} valid instructions, retrying"
                )

        if len(collected) < k:
            self.logger.error(f"[GenInstr] got {len(collected)}/{k} valid instructions after retries")
        return collected[:k]

    def _fallback_parse_instructions(self, text: str, k: int) -> List[Dict]:
        """Fallback: numbered lines or quoted instr strings."""
        results = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            m = re.search(r'"instr"\s*:\s*"([^"]+)"', line)
            if m:
                item = normalize_instruction_schema({"instr": m.group(1)})
                if is_valid_manipulation_instruction(item.get("instruction", "")):
                    results.append(item)
            else:
                m2 = re.match(r"^\d+[\.\)]\s*(.+)$", line)
                if m2 and len(m2.group(1)) > 10:
                    item = normalize_instruction_schema({"instr": m2.group(1)})
                    if is_valid_manipulation_instruction(item.get("instruction", "")):
                        results.append(item)
            if len(results) >= k:
                break
        return results

    @torch.no_grad()
    def augment_prompt_with_qwen(
        self,
        image_path: str,
        instruction: str,
        hand: str = "right",
        output_language: str = "zh",
    ) -> str:
        text_prompt = build_text_augmentation_prompt(
            instruction, hand=hand, mode="single", output_language=output_language
        )
        max_tokens = self.gen_cfg.get("augment_max_new_tokens", 256)
        text = self._vl_generate(image_path, text_prompt, max_tokens)
        if text:
            self.logger.info(f"[Augment] {text[:100]}...")
        return text

    @torch.no_grad()
    def generate_hand_trajectories_with_qwen(
        self,
        image_path: str,
        instruction: str,
        hand: str = "right",
        num_points: int = 8,
        augmented_desc: Optional[str] = None,
    ) -> Dict[str, List]:
        """Qwen VL: discrete 2D hand trajectories guided by text scene augmentation."""
        with Image.open(image_path) as img:
            width, height = img.size
        text_prompt = build_hand_trajectory_vl_prompt(
            instruction=instruction,
            image_width=width,
            image_height=height,
            num_points=num_points,
            hand=hand,
            augmented_desc=augmented_desc,
        )
        max_tokens = self.gen_cfg.get("trajectory_max_new_tokens", 512)
        raw = self._vl_generate(image_path, text_prompt, max_tokens)
        self.logger.info(f"[VisAug] trajectory raw ({len(raw)} chars): {raw[:200]}...")
        parsed = parse_hand_trajectory_json(raw, width, height, default_hand=hand)
        active_key = "right_hand" if hand == "right" else "left_hand"
        inactive_key = "left_hand" if active_key == "right_hand" else "right_hand"
        if not parsed.get(active_key):
            self.logger.warning(f"[VisAug] no points for {active_key}, retrying once")
            retry_prompt = (
                text_prompt
                + f"\nOutput valid JSON only. Use point_2d with integers 0-{1000}, "
                "not pixel coordinates."
            )
            raw = self._vl_generate(image_path, retry_prompt, max_tokens)
            parsed = parse_hand_trajectory_json(raw, width, height, default_hand=hand)
        parsed[inactive_key] = []
        self.logger.info(
            f"[VisAug] {active_key} pixels: {parsed.get(active_key, [])[:3]}..."
        )
        return {
            "coordinate_space": "qwen3_rel_0_1000",
            "image_width": width,
            "image_height": height,
            "left_hand": [list(p) for p in parsed.get("left_hand", [])],
            "right_hand": [list(p) for p in parsed.get("right_hand", [])],
        }

    def save_scene_aug_image(
        self,
        image_path: str,
        trajectories: Dict,
        scene_aug_dir: str,
        instr_idx: int,
    ) -> str:
        """Draw trajectories on scene image and save under database/scene_aug."""
        out_path = scene_aug_output_path(scene_aug_dir, image_path, instr_idx)
        traj_draw = {
            "left_hand": [tuple(p) for p in trajectories.get("left_hand", [])],
            "right_hand": [tuple(p) for p in trajectories.get("right_hand", [])],
        }
        with Image.open(image_path) as img:
            vis = draw_hand_trajectories_on_image(img, traj_draw)
            vis.save(out_path, quality=95)
        self.logger.info(f"[VisAug] saved {out_path}")
        return str(out_path.resolve())

    def prepare_image(
        self,
        image_path: Path,
        k: int,
        task_id_prefix: str,
        global_idx_base: int,
        seed: int,
        hand: str = "right",
        use_augmentation: bool = True,
        use_visual_augmentation: bool = False,
        scene_aug_dir: str = "database/scene_aug",
        trajectory_num_points: int = 8,
        processed_ids: Optional[set] = None,
    ) -> List[Dict]:
        """Generate k (image, instruction) manifest entries."""
        image_str = str(image_path.resolve())
        instructions = self.generate_instructions_on_image(image_str, k, hand=hand)
        if not instructions:
            self.logger.error(f"No instructions generated for {image_path.name}")
            return []

        entries = []
        for instr_idx, instruction in enumerate(instructions):
            ckpt_id = CheckpointManager.get_image_instr_id(image_str, instr_idx)
            if processed_ids is not None and ckpt_id in processed_ids:
                continue

            instr_text = instruction.get("instruction", "")
            if not instr_text:
                continue

            global_idx = global_idx_base + instr_idx
            task_id = f"{task_id_prefix}_instr{instr_idx:02d}"

            aug_lang = getattr(self, "augment_output_language", "zh")
            augmented_desc = ""
            if use_augmentation:
                augmented_desc = self.augment_prompt_with_qwen(
                    image_str, instr_text, hand=hand, output_language=aug_lang
                )

            traj_scene_desc = augmented_desc
            if use_visual_augmentation and not traj_scene_desc.strip():
                self.logger.info("[VisAug] generating scene text for trajectory planning")
                traj_scene_desc = self.augment_prompt_with_qwen(
                    image_str, instr_text, hand=hand, output_language=aug_lang
                )

            hand_trajectories: Dict = {}
            scene_aug_path = ""
            if use_visual_augmentation:
                hand_trajectories = self.generate_hand_trajectories_with_qwen(
                    image_str,
                    instr_text,
                    hand=hand,
                    num_points=trajectory_num_points,
                    augmented_desc=traj_scene_desc or None,
                )
                scene_aug_path = self.save_scene_aug_image(
                    image_str,
                    hand_trajectories,
                    scene_aug_dir,
                    instr_idx,
                )

            video_prompt = build_video_prompt(
                instr_text,
                instruction.get("caption"),
                augmented_desc or None,
                hand=hand,
                use_visual_trajectory=bool(scene_aug_path),
            )
            (
                combined_instruction,
                task_description,
                left_instruction,
                right_instruction,
                final_augmented,
            ) = format_single_arm_manifest_fields(instr_text, augmented_desc, hand)

            entries.append({
                "task_id": task_id,
                "global_idx": global_idx,
                "image_path": image_str,
                "image_name": image_path.name,
                "instr_idx": instr_idx,
                "instruction": combined_instruction,
                "task_description": task_description,
                "left_instruction": left_instruction,
                "right_instruction": right_instruction,
                "shared_object": "",
                "coordination_type": "",
                "nouns": instruction.get("nouns", []),
                "adjectives": instruction.get("adjectives", []),
                "verbs": instruction.get("verbs", []),
                "primary_object": extract_primary_object_key(instruction),
                "augmented_text": final_augmented,
                "video_prompt": video_prompt,
                "hand": hand,
                "use_visual_aug": bool(scene_aug_path),
                "scene_aug_path": scene_aug_path,
                "hand_trajectories": hand_trajectories,
                "seed": seed + global_idx,
                "_checkpoint_id": ckpt_id,
            })
        return entries


def _worker_prepare_images(args: Tuple) -> int:
    image_jobs, gpu_id, visible_gpus, config, output_manifest, checkpoint_info = args
    device = isolate_cuda_device(gpu_id, visible_gpus)
    print(f"[GPU {gpu_id}] Preparer on {device}, {len(image_jobs)} images")

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    task_cfg = config.get("task", {})
    settings_cfg = config.get("settings", {})
    gen_cfg = config.get("generation", {})
    checkpoint_cfg = config.get("checkpoint", {})

    k = task_cfg.get("k", 5)
    hand = settings_cfg.get("hand", "right")
    aug = settings_cfg.get("aug", True)
    visual_aug = settings_cfg.get("visual_aug", False)
    scene_aug_dir = data_cfg.get("scene_aug_dir", "database/scene_aug")
    trajectory_num_points = settings_cfg.get("trajectory_num_points", 8)
    seed = task_cfg.get("seed", 0)
    checkpoint_dir = checkpoint_info.get("checkpoint_dir", "database/log")
    checkpoint_name = checkpoint_cfg.get("checkpoint_name", "processed_image_prompts.txt")
    db_cfg = config.get("database", {})
    db_enabled = db_cfg.get("enable", True)
    db_path = db_cfg.get("path", DEFAULT_DIVERSITY_DB_PATH)
    ckpt = CheckpointManager(checkpoint_dir, checkpoint_name=checkpoint_name)
    processed_ids = ckpt.load_processed_instructions() if checkpoint_info.get("enabled", True) else set()

    preparer = VideoPromptPreparer(
        qwen_vl_model_path=model_cfg["qwen_vl_model_path"],
        device=device,
        gen_cfg=gen_cfg,
        image_max_side=settings_cfg.get("image_max_side", 1024),
    )
    preparer.augment_output_language = settings_cfg.get("augment_output_language", "zh")

    manifest_path = Path(output_manifest)
    success = 0

    with open(manifest_path, "a", encoding="utf-8") as out_f:
        for job in image_jobs:
            image_path = Path(job["image_path"])
            image_idx = job["image_idx"]
            global_idx_base = image_idx * k
            task_id_prefix = f"gpu{gpu_id}_img{image_idx:06d}"

            print(f"[GPU {gpu_id}] image {image_idx}: {image_path.name} (k={k})")
            entries = preparer.prepare_image(
                image_path=image_path,
                k=k,
                task_id_prefix=task_id_prefix,
                global_idx_base=global_idx_base,
                seed=seed,
                hand=hand,
                use_augmentation=aug,
                use_visual_augmentation=visual_aug,
                scene_aug_dir=scene_aug_dir,
                trajectory_num_points=trajectory_num_points,
                processed_ids=processed_ids,
            )

            for entry in entries:
                ckpt_id = entry.pop("_checkpoint_id")
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                out_f.flush()
                os.fsync(out_f.fileno())
                if db_enabled:
                    try:
                        record_manifest_entry_in_db(entry, "video_single", db_path)
                    except Exception as exc:
                        print(f"[GPU {gpu_id}] diversity DB write failed: {exc}")
                if checkpoint_info.get("enabled", True):
                    ckpt.mark_as_processed(ckpt_id, gpu_id)
                    processed_ids.add(ckpt_id)
                success += 1

    print(f"[GPU {gpu_id}] Wrote {success} manifest entries")
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: per-image generate k instructions + prompts"
    )
    parser.add_argument("--config", type=str, default="config/config_video_prompt_preparer.yaml")
    parser.add_argument("--reset-checkpoint", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    task_cfg = config.get("task", {})
    parallel_cfg = config.get("parallel", {})
    checkpoint_cfg = config.get("checkpoint", {})

    image_dir = data_cfg.get("image_dir")
    recursive = data_cfg.get("recursive", False)
    manifest_dir = Path(data_cfg.get("manifest_dir", "database/manifests"))
    k = task_cfg.get("k", 5)
    enable_parallel = parallel_cfg.get("enable", True)
    checkpoint_enabled = checkpoint_cfg.get("enable", True)
    checkpoint_dir = checkpoint_cfg.get("dir", "database/log")
    checkpoint_name = checkpoint_cfg.get("checkpoint_name", "processed_image_prompts.txt")

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible:
        visible_gpus = [int(x) for x in cuda_visible.split(",") if x.strip()]
    else:
        visible_gpus = list(range(torch.cuda.device_count()))
    if not visible_gpus:
        raise RuntimeError("No GPU available")

    num_gpus = len(visible_gpus)
    image_paths = list_images(image_dir, recursive=recursive)
    if not image_paths:
        print(f"No images in {image_dir}")
        return

    print(f"Found {len(image_paths)} images, k={k} instructions per image")
    print(f"Expected manifest entries: up to {len(image_paths) * k}")

    ckpt = CheckpointManager(checkpoint_dir, checkpoint_name=checkpoint_name)
    if args.reset_checkpoint:
        ckpt.checkpoint_file.write_text("")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = manifest_dir / f"manifest_{timestamp}.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.touch()
    print(f"Manifest: {manifest_path}")

    image_jobs = [{"image_path": str(p), "image_idx": idx} for idx, p in enumerate(image_paths)]
    checkpoint_info = {
        "checkpoint_dir": checkpoint_dir,
        "enabled": checkpoint_enabled,
    }

    if not enable_parallel or num_gpus == 1:
        n = _worker_prepare_images(
            (image_jobs, 0, visible_gpus, config, str(manifest_path), checkpoint_info)
        )
        print(f"Done: {n} entries -> {manifest_path}")
        return

    jobs_per_gpu: List[List[Dict]] = [[] for _ in range(num_gpus)]
    for idx, job in enumerate(image_jobs):
        jobs_per_gpu[idx % num_gpus].append(job)

    tasks = [
        (jobs_per_gpu[g], g, visible_gpus, config, str(manifest_path), checkpoint_info)
        for g in range(num_gpus)
        if jobs_per_gpu[g]
    ]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(tasks)) as pool:
        results = pool.map(_worker_prepare_images, tasks)
    print(f"Total: {sum(results)} entries -> {manifest_path}")


if __name__ == "__main__":
    main()

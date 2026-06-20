"""Instruction-first pipeline: slot assembly → box edit objects into scene images."""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from PIL import Image
from tqdm import tqdm

from wm_h.video.common import (
    CheckpointManager,
    ensure_working_image_file,
    resize_image_lanczos,
)

from .bbox_coords import BBOX_COORD_SYSTEM, pixel_rect_to_qwen_bbox
from .logging_utils import get_cli_logger
from .terminal_ui import tqdm_defaults
from .box_editor import (
    GUIDE_MODE_BBOX,
    GUIDE_MODE_POINT,
    images_meaningfully_differ,
    make_image_editor,
)
from .box_placement import assign_edit_rects
from .hand_detector import HandDetector, rect_overlaps_hands
from .instr_first import InstrFirstGenerator, SlotInstructionAssembler
from .io_utils import list_images, resolve_image_edit_config
from .schema import FeasibleInstruction, ObjectRef, format_instruction

logger = logging.getLogger("wm_h.instr_first_pipeline")
cli = get_cli_logger()

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _rel_path(path: str) -> str:
    if not path:
        return path
    p = Path(path).resolve()
    try:
        return p.relative_to(_REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(p)


class InstrFirstPipeline:
    """
    1. Assemble instructions from slot templates + word DB
    2. Paint each involved object onto the scene at random non-overlapping desk boxes
    3. Write manifest with verbs / nouns / adjectives / objects / edit bboxes
    """

    def __init__(self, config_path: str, *, device: Optional[str] = None):
        cfg_path = Path(config_path).resolve()
        with open(cfg_path, encoding="utf-8") as f:
            self.full_config = yaml.safe_load(f)
        self.config_path = str(cfg_path)
        self.if_cfg = self.full_config.get("instr_first", {})
        self.base = _REPO_ROOT

        settings = self.full_config.get("settings", {})
        self.device = device or settings.get("device", "cuda")
        self.seed = int(self.if_cfg.get("seed", 0))

        self._generator: Optional[InstrFirstGenerator] = None

        edit_cfg = resolve_image_edit_config(self.full_config)
        box_cfg = self.if_cfg.get("box", {})

        edit_width = int(
            edit_cfg.get("video_width", edit_cfg.get("edit_width", 0)) or 0
        )
        edit_height = int(
            edit_cfg.get("video_height", edit_cfg.get("edit_height", 0)) or 0
        )
        self.min_edit_box_px = int(edit_cfg.get("min_box_px", 16))
        self.mask_width = int(box_cfg.get("mask_width", edit_cfg.get("mask_width", 256)))
        self.mask_height = int(box_cfg.get("mask_height", edit_cfg.get("mask_height", 256)))
        wr = box_cfg.get("mask_width_range")
        hr = box_cfg.get("mask_height_range")
        self.width_range = (int(wr[0]), int(wr[1])) if wr and len(wr) >= 2 else None
        self.height_range = (int(hr[0]), int(hr[1])) if hr and len(hr) >= 2 else None
        self.center_x_range = tuple(
            edit_cfg.get("center_x_range", [0.2, 0.8])
        )
        self.center_y_range = tuple(
            edit_cfg.get("center_y_range", [0.2, 0.8])
        )
        hand_det_cfg = self.if_cfg.get("hand_detection", {})
        self.allow_hand_occlusion = bool(box_cfg.get("allow_hand_occlusion", True))
        self.hand_negative_prompt = str(
            edit_cfg.get("hand_negative_prompt", edit_cfg.get("negative_prompt", ""))
            or ""
        )
        self.default_negative_prompt = str(edit_cfg.get("negative_prompt", "") or "")
        self._hand_cache: Dict[str, List[Tuple[int, int, int, int]]] = {}
        self.hand_detector = HandDetector(
            model_path=str(hand_det_cfg.get("model_path", "")),
            device=str(hand_det_cfg.get("device", self.device)),
            conf_threshold=float(hand_det_cfg.get("conf_threshold", 0.35)),
            enable=bool(hand_det_cfg.get("enable", True)),
            padding_frac=float(hand_det_cfg.get("padding", 0.05)),
            fallback_zones_norm=list(
                box_cfg.get("hand_zones")
                or box_cfg.get("hand_preserve_rects")
                or box_cfg.get("hand_exclude_rects")
                or []
            ),
            use_fallback_zones=bool(hand_det_cfg.get("use_fallback_zones", False)),
        )
        self.size_skew = float(box_cfg.get("size_skew", 0.65))
        self.position_jitter = float(box_cfg.get("position_jitter", 0.14))
        self.placement_max_attempts = int(box_cfg.get("placement_max_attempts", 300))
        self.point_guide_ratio = max(
            0.0,
            min(1.0, float(edit_cfg.get("point_guide_ratio", 0.7))),
        )
        self.min_point_distance_frac = float(
            edit_cfg.get("min_point_distance_frac", 0.28)
        )

        self.editor = make_image_editor(
            edit_cfg,
            self.device,
            mask_width=self.mask_width,
            mask_height=self.mask_height,
            center_x_range=self.center_x_range,
            center_y_range=self.center_y_range,
        )

        out_cfg = self.if_cfg.get("output", {})
        self.output_base = Path(
            out_cfg.get("dir", "database/wm-h/instr_first")
        )
        self.run_dir: Optional[Path] = None
        self.edited_dir: Optional[Path] = None

    def _ensure_generator(self) -> InstrFirstGenerator:
        if self._generator is None:
            logger.debug("Loading text model (first load may take 1–3 min)")
            t0 = time.perf_counter()
            self._generator = InstrFirstGenerator(self.config_path, device=self.device)
            logger.debug("Text model ready (%.1fs)", time.perf_counter() - t0)
        return self._generator

    def _setup_run_directories(self) -> Path:
        run_tag = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_dir = self.output_base / "runs" / run_tag
        self.edited_dir = self.run_dir / "edited_images"
        self.edited_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir

    def _edit_coord_size(self, image_path: str) -> Tuple[int, int]:
        fixed = self.editor.fixed_edit_size()
        if fixed is not None:
            return fixed
        with Image.open(image_path) as im:
            return im.size

    def _ensure_working_image(
        self, image_path: str, stem: str, target_w: int, target_h: int
    ) -> str:
        if self.edited_dir is None:
            return str(Path(image_path).resolve())
        out_path = self.edited_dir / f"{stem}_desktop.jpg"
        return ensure_working_image_file(image_path, out_path, target_w, target_h)

    def _detected_hands(self, image_path: str) -> List[Tuple[int, int, int, int]]:
        key = str(Path(image_path).resolve())
        if key not in self._hand_cache:
            edit_w, edit_h = self._edit_coord_size(image_path)
            self._hand_cache[key] = self.hand_detector.detect(
                image_path,
                target_w=edit_w,
                target_h=edit_h,
            )
        return self._hand_cache[key]

    def _assemble_count(self, count: int) -> List[Dict]:
        gen = self._ensure_generator()
        assembled: List[Dict] = []
        force: List[str] = []
        attempts = 0

        gen._bootstrap_vocab(max_rounds=2)

        pbar = tqdm(
            total=count,
            desc="assemble",
            unit="instr",
            **tqdm_defaults(colour="green"),
        )
        while len(assembled) < count and attempts < 30:
            attempts += 1

            if gen._should_expand():
                pbar.set_postfix_str("vocab expand", refresh=True)
                gen._expand_slots(list(gen.cfg["slots"].keys()))

            batch, exhausted = gen.agent2.assemble_batch(
                count - len(assembled), force
            )
            if exhausted:
                pbar.set_postfix_str(f"expand {exhausted}", refresh=True)
                gen._expand_slots(exhausted)
                if not batch:
                    logger.warning(
                        "assemble exhausted %s (attempt %d), expanded and retrying",
                        exhausted,
                        attempts,
                    )
                    continue
            if batch:
                assembled.extend(batch)
                pbar.update(len(batch))
                pbar.set_postfix_str(
                    f"{len(assembled)}/{count}", refresh=True
                )
            if not batch:
                logger.warning("assemble stalled (attempt %d)", attempts)
                continue

        pbar.close()
        assembled = assembled[:count]
        bimanual_n = sum(1 for a in assembled if a.get("use_both_hands"))
        if bimanual_n:
            logger.debug("Bimanual decompose: %d instruction(s)", bimanual_n)
            gen.enrich_bimanual_assembled(assembled)
        return assembled

    def _instruction_center_jitter(
        self,
        instr_idx: int,
        placement_attempt: int,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Per-instruction shift so consecutive instructions don't cluster."""
        if (
            self.center_x_range[0] <= 0.02
            and self.center_x_range[1] >= 0.98
            and self.center_y_range[0] <= 0.02
            and self.center_y_range[1] >= 0.98
        ):
            return self.center_x_range, self.center_y_range
        rng = random.Random(self.seed + instr_idx * 137 + placement_attempt * 19)
        jx = rng.uniform(-self.position_jitter, self.position_jitter)
        jy = rng.uniform(-self.position_jitter, self.position_jitter)
        cx0 = max(0.03, min(0.94, self.center_x_range[0] + jx))
        cx1 = max(cx0 + 0.08, min(0.97, self.center_x_range[1] + jx))
        cy0 = max(0.04, min(0.86, self.center_y_range[0] + jy))
        cy1 = max(cy0 + 0.08, min(0.92, self.center_y_range[1] + jy))
        return (cx0, cx1), (cy0, cy1)

    def _placement_ranges_for_attempt(
        self,
        attempt: int,
        *,
        base_cx: Tuple[float, float],
        base_cy: Tuple[float, float],
    ) -> Tuple[
        Tuple[float, float],
        Tuple[float, float],
        Optional[Tuple[int, int]],
        Optional[Tuple[int, int]],
    ]:
        """Progressively relax box sampling when desk/hand zones are tight."""
        cx = base_cx
        cy = base_cy
        wr = self.width_range
        hr = self.height_range
        if attempt >= 8:
            cy = (
                max(0.04, cy[0] - 0.06),
                min(0.92, cy[1] + 0.06),
            )
        if attempt >= 16:
            if wr:
                wr = (wr[0], max(wr[0], int(wr[1] * 0.8)))
            if hr:
                hr = (hr[0], max(hr[0], int(hr[1] * 0.75)))
        if attempt >= 24:
            cx = (max(0.03, cx[0] - 0.08), min(0.97, cx[1] + 0.08))
        return cx, cy, wr, hr

    def _plan_placements(
        self,
        objects: List[ObjectRef],
        image_path: str,
        *,
        image_idx: int,
        instr_idx: int,
        placement_attempt: int,
    ) -> Tuple[List[Tuple[ObjectRef, Tuple[int, int, int, int]]], int, int]:
        edit_w, edit_h = self._edit_coord_size(image_path)
        chain_seed = (
            self.seed
            + image_idx * 1000
            + instr_idx * 10
            + placement_attempt * 31
        )
        base_cx, base_cy = self._instruction_center_jitter(
            instr_idx, placement_attempt
        )
        cx, cy, wr, hr = self._placement_ranges_for_attempt(
            placement_attempt,
            base_cx=base_cx,
            base_cy=base_cy,
        )
        placements = assign_edit_rects(
            objects,
            img_w=edit_w,
            img_h=edit_h,
            seed=chain_seed,
            mask_width=self.mask_width,
            mask_height=self.mask_height,
            width_range=wr,
            height_range=hr,
            center_x_range=cx,
            center_y_range=cy,
            min_box_px=self.min_edit_box_px,
            size_skew=self.size_skew,
            position_jitter=self.position_jitter * 0.6,
            allow_rect_overlap=True,
            max_attempts=self.placement_max_attempts,
        )
        if len(placements) != len(objects):
            return [], edit_w, edit_h
        if not self._placements_are_far_enough(placements, edit_w, edit_h):
            return [], edit_w, edit_h
        return placements, edit_w, edit_h

    def _placements_are_far_enough(
        self,
        placements: List[Tuple[ObjectRef, Tuple[int, int, int, int]]],
        edit_w: int,
        edit_h: int,
    ) -> bool:
        if len(placements) < 2 or self.min_point_distance_frac <= 0:
            return True
        min_dist = self.min_point_distance_frac * min(edit_w, edit_h)
        centers = [
            ((rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0)
            for _, rect in placements
        ]
        for i, (ax, ay) in enumerate(centers):
            for bx, by in centers[i + 1 :]:
                if ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 < min_dist:
                    return False
        return True

    def _guide_mode_for_object(
        self,
        *,
        image_idx: int,
        instr_idx: int,
        placement_attempt: int,
        step: int,
    ) -> str:
        rng = random.Random(
            self.seed
            + image_idx * 1000
            + instr_idx * 97
            + placement_attempt * 31
            + step * 1009
        )
        return (
            GUIDE_MODE_POINT
            if rng.random() < self.point_guide_ratio
            else GUIDE_MODE_BBOX
        )

    def _edit_instruction(
        self,
        image_path: str,
        assembled: Dict,
        stem: str,
        image_idx: int,
        instr_idx: int,
        *,
        placement_attempt: int = 0,
    ) -> Dict[str, Any]:
        feasible = FeasibleInstruction.from_dict(assembled["feasible"])
        objects = [o for o in feasible.objects if o.noun]
        edit_w, edit_h = self._edit_coord_size(image_path)
        desktop_image = self._ensure_working_image(image_path, stem, edit_w, edit_h)

        placements, edit_w, edit_h = self._plan_placements(
            objects,
            image_path,
            image_idx=image_idx,
            instr_idx=instr_idx,
            placement_attempt=placement_attempt,
        )

        if not placements:
            return self._manifest_row(
                image_path=image_path,
                edited_path=desktop_image,
                assembled=assembled,
                feasible=feasible,
                objects=objects,
                placements=[],
                image_idx=image_idx,
                instr_idx=instr_idx,
            )

        chain_seed = (
            self.seed
            + image_idx * 1000
            + instr_idx * 10
            + placement_attempt * 31
        )
        working_image = desktop_image
        exclude_rects: List[Tuple[int, int, int, int]] = []
        edited_path = desktop_image
        hand_rects = self._detected_hands(image_path)

        if self.editor.enabled and self.edited_dir is not None:
            final_placements: List[Tuple[ObjectRef, Tuple[int, int, int, int]]] = []
            guide_modes: List[str] = []
            for step, (obj, rect) in enumerate(placements):
                edit_name = f"{stem}_instr{instr_idx:02d}_obj{step:02d}_edited.jpg"
                out_path = str(self.edited_dir / edit_name)
                guide_mode = self._guide_mode_for_object(
                    image_idx=image_idx,
                    instr_idx=instr_idx,
                    placement_attempt=placement_attempt,
                    step=step,
                )
                occludes_hands = (
                    self.allow_hand_occlusion
                    and bool(hand_rects)
                    and rect_overlaps_hands(rect, hand_rects)
                )
                neg_prompt = (
                    self.hand_negative_prompt
                    if occludes_hands
                    else self.default_negative_prompt
                )
                edited_path, rect = self.editor.edit_object_into_image(
                    working_image,
                    obj,
                    out_path,
                    seed=chain_seed + step,
                    exclude_rects=exclude_rects,
                    target_rect_px=rect,
                    min_box_px=self.min_edit_box_px,
                    trust_assigned_bbox=True,
                    hand_occlusion=occludes_hands,
                    negative_prompt=neg_prompt,
                    guide_mode=guide_mode,
                )
                exclude_rects.append(rect)
                final_placements.append((obj, rect))
                guide_modes.append(guide_mode)
                working_image = edited_path
            placements = final_placements
        else:
            guide_modes = [
                self._guide_mode_for_object(
                    image_idx=image_idx,
                    instr_idx=instr_idx,
                    placement_attempt=placement_attempt,
                    step=step,
                )
                for step, _ in enumerate(placements)
            ]

        enriched_objects: List[ObjectRef] = []
        rect_by_key = {obj.key(): rect for obj, rect in placements}
        for obj in objects:
            rect = rect_by_key.get(obj.key())
            if rect:
                bbox = pixel_rect_to_qwen_bbox(rect, edit_w, edit_h)
                enriched_objects.append(
                    ObjectRef(
                        noun=obj.noun,
                        adjective=obj.adjective,
                        bbox_2d=bbox,
                    )
                )
            else:
                enriched_objects.append(obj)

        return self._manifest_row(
            image_path=image_path,
            edited_path=edited_path,
            assembled=assembled,
            feasible=feasible,
            objects=enriched_objects,
            placements=placements,
            guide_modes=guide_modes,
            image_idx=image_idx,
            instr_idx=instr_idx,
            edit_w=edit_w,
            edit_h=edit_h,
        )

    def _manifest_row(
        self,
        *,
        image_path: str,
        edited_path: str,
        assembled: Dict,
        feasible: FeasibleInstruction,
        objects: List[ObjectRef],
        placements: List[Tuple[ObjectRef, Tuple[int, int, int, int]]],
        image_idx: int,
        instr_idx: int,
        edit_w: int = 0,
        edit_h: int = 0,
        guide_modes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        slots = assembled.get("slots", {})
        edit_boxes_px = [list(r) for _, r in placements]
        edit_points_px = [
            [int((r[0] + r[2]) / 2), int((r[1] + r[3]) / 2)]
            for _, r in placements
        ]
        edit_bboxes_2d = (
            [pixel_rect_to_qwen_bbox(r, edit_w, edit_h) for _, r in placements]
            if edit_w and edit_h
            else []
        )
        edit_points_2d = (
            [
                [
                    int(round(point[0] / edit_w * 1000)),
                    int(round(point[1] / edit_h * 1000)),
                ]
                for point in edit_points_px
            ]
            if edit_w and edit_h
            else []
        )
        guide_modes = list(guide_modes or [])
        if len(guide_modes) < len(placements):
            guide_modes.extend([""] * (len(placements) - len(guide_modes)))

        target = objects[0] if objects else ObjectRef()
        reference = objects[1] if len(objects) > 1 else None

        desktop_rel = _rel_path(image_path)
        use_both = bool(
            assembled.get("use_both_hands") or feasible.use_both_hands
        )
        if use_both and feasible.use_both_hands:
            left_instr = feasible.left_instruction
            right_instr = feasible.right_instruction
        else:
            left_instr = "None"
            right_instr = feasible.task_description
        instruction_text = format_instruction(left_instr, right_instr)

        row: Dict[str, Any] = {
            "task_id": f"instr{instr_idx:06d}",
            "run_dir": _rel_path(str(self.run_dir)) if self.run_dir else "",
            "mode": "instr_first",
            "desktop_image_path": desktop_rel,
            "image_path": desktop_rel,
            "edited_image_path": _rel_path(edited_path),
            "template_id": assembled.get("template_id", ""),
            "use_both_hands": use_both and feasible.use_both_hands,
            "slots": dict(slots),
            "task_description": feasible.task_description,
            "instruction": instruction_text,
            "action_type": feasible.action_type,
            "verbs": list(feasible.verbs),
            "nouns": list(feasible.nouns),
            "adjectives": list(feasible.adjectives),
            "objects": [o.to_dict() for o in objects],
            "target_object": target.to_dict(),
            "reference_object": reference.to_dict() if reference else None,
            "left_instruction": left_instr,
            "right_instruction": right_instr,
            "target_in_image": True,
            "edit_boxes_px": edit_boxes_px,
            "edit_points_px": edit_points_px,
            "edit_bboxes_2d": edit_bboxes_2d,
            "edit_points_2d": edit_points_2d,
            "edit_guide_modes": guide_modes,
            "bbox_coord_system": BBOX_COORD_SYSTEM,
            "edit_boxes_coord_system": "pixel",
            "edit_bboxes_coord_system": BBOX_COORD_SYSTEM,
        }
        if use_both and feasible.use_both_hands:
            row["coordination_type"] = feasible.coordination_type
        return row

    @staticmethod
    def _has_edited_image(row: Dict[str, Any]) -> bool:
        src = row.get("desktop_image_path", "")
        edited = row.get("edited_image_path", "")
        if not edited or edited == src:
            return False
        if not row.get("edit_boxes_px"):
            return False
        try:
            with Image.open(src) as before, Image.open(edited) as after:
                before = before.convert("RGB")
                after = after.convert("RGB")
                if before.size != after.size:
                    before = resize_image_lanczos(before, after.size[0], after.size[1])
                return images_meaningfully_differ(before, after)
        except OSError:
            return False

    def _edit_instruction_resilient(
        self,
        assembled: Dict,
        images: List[Path],
        instr_idx: int,
    ) -> Optional[Dict[str, Any]]:
        """Retry placement/edit across seeds and desktop images until an edited image exists."""
        rng = random.Random(self.seed + instr_idx)
        if not self.editor.enabled:
            image_idx = rng.randrange(len(images))
            img = images[image_idx]
            return self._edit_instruction(
                str(img.resolve()),
                assembled,
                img.stem,
                image_idx=image_idx,
                instr_idx=instr_idx,
            )

        max_attempts = max(48, len(images) * 12)
        row: Optional[Dict[str, Any]] = None
        for attempt in range(max_attempts):
            image_idx = rng.randrange(len(images))
            img = images[image_idx]
            row = self._edit_instruction(
                str(img.resolve()),
                assembled,
                img.stem,
                image_idx=image_idx,
                instr_idx=instr_idx,
                placement_attempt=attempt,
            )
            if self._has_edited_image(row):
                return row

        task = assembled.get("feasible", {}).get("task_description", "")
        logger.warning(
            "Placement/edit failed after %d attempts for instr %d (%s); "
            "skipping this instruction",
            max_attempts,
            instr_idx,
            task,
        )
        return None

    def _write_manifest_batch(
        self,
        out_f,
        *,
        assembled_batch: List[Dict],
        images: List[Path],
        start_idx: int,
        checkpoint: Optional[CheckpointManager] = None,
        processed_ids: Optional[set] = None,
    ) -> int:
        """Edit one batch of instructions and append rows to manifest (1 instr → 1 edited image)."""
        written = 0
        for offset, assembled in enumerate(assembled_batch):
            instr_idx = start_idx + offset
            ckpt_id = CheckpointManager.get_instr_first_id(assembled)
            if processed_ids is not None and ckpt_id in processed_ids:
                continue
            row = self._edit_instruction_resilient(
                assembled,
                images,
                instr_idx,
            )
            if row is None:
                continue
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if checkpoint is not None:
                checkpoint.mark_as_processed(ckpt_id)
                if processed_ids is not None:
                    processed_ids.add(ckpt_id)
        out_f.flush()
        return written

    def _instr_first_checkpoint(
        self, *, reset: bool = False
    ) -> Tuple[Optional[CheckpointManager], Optional[set]]:
        ckpt_cfg = self.if_cfg.get("checkpoint", {})
        if not ckpt_cfg.get("enable", True):
            return None, None
        ckpt = CheckpointManager(
            ckpt_cfg.get("dir", "database/log"),
            checkpoint_name=ckpt_cfg.get(
                "checkpoint_name", "processed_desk_synth_instr_first.txt"
            ),
        )
        if reset:
            ckpt.reset()
        return ckpt, ckpt.load_processed_instructions()

    def edit_assembled(
        self,
        assembled_list: List[Dict],
        images: List[Path],
        manifest_path: Path,
        *,
        start_idx: int = 0,
        checkpoint: Optional[CheckpointManager] = None,
        processed_ids: Optional[set] = None,
    ) -> int:
        """Run the image-edit phase for pre-assembled instructions."""
        batch_cfg = self.if_cfg.get("batch", {})
        edit_batch_size = max(
            1,
            int(
                batch_cfg.get(
                    "edit_batch_size",
                    batch_cfg.get("agent2_batch_size", 1),
                )
            ),
        )
        total = 0
        mode = "a" if manifest_path.exists() and manifest_path.stat().st_size > 0 else "w"
        if self.editor.enabled and self._generator is not None:
            self._generator.release_text_model()
        with open(manifest_path, mode, encoding="utf-8") as out_f:
            batch_starts = range(0, len(assembled_list), edit_batch_size)
            batch_iter = batch_starts
            if self.editor.enabled:
                batch_iter = tqdm(
                    batch_starts,
                    desc="edit",
                    unit="batch",
                    **tqdm_defaults(colour="magenta"),
                )
            for batch_start in batch_iter:
                batch = assembled_list[batch_start : batch_start + edit_batch_size]
                total += self._write_manifest_batch(
                    out_f,
                    assembled_batch=batch,
                    images=images,
                    start_idx=start_idx + batch_start,
                    checkpoint=checkpoint,
                    processed_ids=processed_ids,
                )
        if self.editor.enabled:
            self.editor.release_pipe()
        return total

    def run(
        self,
        image_dir: Optional[str] = None,
        recursive: bool = False,
        total_instructions: Optional[int] = None,
        *,
        num_gpus: int = 0,
        reset_checkpoint: bool = False,
    ) -> str:
        data_cfg = self.if_cfg.get("data", {})
        img_dir = image_dir or data_cfg.get("image_dir", "")
        if not img_dir:
            raise ValueError(
                "image_dir required (config instr_first.data.image_dir or --image-dir)"
            )

        recursive = recursive or bool(data_cfg.get("recursive", False))
        batch_cfg = self.if_cfg.get("batch", {})
        target = int(
            total_instructions
            if total_instructions is not None
            else batch_cfg.get("total_instructions", 100)
        )

        images = list_images(img_dir, recursive=recursive)
        if not images:
            raise FileNotFoundError(f"No images in {img_dir}")

        run_dir = self._setup_run_directories()
        manifest_path = run_dir / "manifest.jsonl"

        cli.info(
            "instr_first: %d instructions, %d images → %s",
            target,
            len(images),
            run_dir,
        )

        t0 = time.perf_counter()
        assembled_list = self._assemble_count(target)
        logger.debug(
            "Assembled %d instruction(s) in %.1fs",
            len(assembled_list),
            time.perf_counter() - t0,
        )

        if not self.editor.enabled:
            logger.warning("Image edit disabled; manifest rows will use source images only")

        ckpt, processed_ids = self._instr_first_checkpoint(reset=reset_checkpoint)

        from .parallel import resolve_worker_count, run_instr_first_multi_gpu

        workers = resolve_worker_count(num_gpus, len(assembled_list))
        if workers > 1 and self.editor.enabled:
            if self._generator is not None:
                self._generator.db.close()
                self._generator = None
            cli.info(
                "instr_first edit: %d instructions across %d GPU worker(s)",
                len(assembled_list),
                workers,
            )
            total = run_instr_first_multi_gpu(
                self.config_path,
                assembled_list,
                images,
                run_dir=run_dir,
                manifest_path=manifest_path,
                num_workers=workers,
                seed=self.seed,
                checkpoint=ckpt,
            )
            cli.info(
                "instr_first done: %d rows (%d GPUs) → %s",
                total,
                workers,
                manifest_path,
            )
            return str(manifest_path)

        total = self.edit_assembled(
            assembled_list,
            images,
            manifest_path,
            checkpoint=ckpt,
            processed_ids=processed_ids,
        )

        if self._generator is not None:
            self._generator.db.close()

        cli.info("instr_first done: %d rows → %s", total, manifest_path)
        return str(manifest_path)

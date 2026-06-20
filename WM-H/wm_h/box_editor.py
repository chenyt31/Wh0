"""Box- and point-guided image editing for out-of-image target objects (Qwen-Image-Edit-2511)."""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageChops

from .bbox_coords import qwen_bbox_to_pixel_rect
from .schema import ObjectRef

logger = logging.getLogger(__name__)

BOX_NEGATIVE_PROMPT = (
    "black rectangle, bounding box, rectangle outline, rectangle border, guide box"
)
POINT_NEGATIVE_PROMPT = (
    "red dot, bright red circle, placement marker, guide dot, circle marker, "
    "red point, marker dot"
)

GUIDE_MODE_BBOX = "bbox"
GUIDE_MODE_POINT = "point"


def merge_negative_prompt(*parts: Optional[str]) -> str:
    seen = set()
    tokens: List[str] = []
    for part in parts:
        for chunk in (part or "").replace("，", ",").split(","):
            chunk = chunk.strip()
            if chunk and chunk not in seen:
                seen.add(chunk)
                tokens.append(chunk)
    return ", ".join(tokens)

# Qwen-Image-Edit VAE downsamples by 16, then the transformer patchifies by 2.
# Keep image H/W divisible by 32 so latent H/W stay even (e.g. 720→45 fails).
EDIT_DIM_ALIGN = 32

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DIFFSYNTH = _REPO_ROOT / "third_party" / "DiffSynth-Studio"


def _ensure_diffsynth_path() -> None:
    if str(_DIFFSYNTH) not in sys.path:
        sys.path.insert(0, str(_DIFFSYNTH))
    os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "sage_attention")


def align_edit_dim(n: int) -> int:
    """Round up to a supported edit dimension."""
    n = max(EDIT_DIM_ALIGN, int(n))
    return ((n + EDIT_DIM_ALIGN - 1) // EDIT_DIM_ALIGN) * EDIT_DIM_ALIGN


def resolve_lightning_lora_path(path: str) -> Optional[str]:
    """Resolve Lightning LoRA file from a file path or directory."""
    if not path:
        return None
    p = Path(path)
    if p.is_file():
        return str(p)
    if p.is_dir():
        files = sorted(p.glob("*.safetensors"))
        if files:
            return str(files[0])
    return None


def rects_overlap(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> bool:
    a_left, a_top, a_right, a_bottom = a
    b_left, b_top, b_right, b_bottom = b
    if a_right <= b_left or b_right <= a_left:
        return False
    if a_bottom <= b_top or b_bottom <= a_top:
        return False
    return True


def images_meaningfully_differ(
    before: Image.Image,
    after: Image.Image,
    *,
    min_mean_diff: float = 2.0,
) -> bool:
    """Return False when edit output is nearly pixel-identical to the source."""
    if before.size != after.size:
        return True
    a = before.convert("RGB")
    b = after.convert("RGB")
    diff = ImageChops.difference(a, b)
    hist = diff.histogram()
    total_pixels = a.size[0] * a.size[1]
    if total_pixels <= 0:
        return False
    total_diff = 0
    for channel in range(3):
        offset = channel * 256
        for value, count in enumerate(hist[offset : offset + 256]):
            total_diff += value * count
    mean_diff = total_diff / (total_pixels * 3)
    return mean_diff >= min_mean_diff


def create_random_edit_rect(
    image_width: int,
    image_height: int,
    mask_width: int,
    mask_height: int,
    center_x_range: Tuple[float, float],
    center_y_range: Tuple[float, float],
    seed: Optional[int] = None,
    exclude_rects: Optional[List[Tuple[int, int, int, int]]] = None,
    mask_width_range: Optional[Tuple[int, int]] = None,
    mask_height_range: Optional[Tuple[int, int]] = None,
    min_box_px: int = 16,
) -> Tuple[int, int, int, int]:
    rng = random.Random(seed)
    exclude_rects = exclude_rects or []
    left = top = right = bottom = 0
    for _ in range(200):
        if mask_width_range:
            bw = rng.randint(mask_width_range[0], mask_width_range[1])
        else:
            bw = mask_width
        if mask_height_range:
            bh = rng.randint(mask_height_range[0], mask_height_range[1])
        else:
            bh = mask_height
        bw = align_edit_dim(max(min_box_px, bw))
        bh = align_edit_dim(max(min_box_px, bh))
        cx = rng.uniform(center_x_range[0], center_x_range[1]) * image_width
        cy = rng.uniform(center_y_range[0], center_y_range[1]) * image_height
        left = int(cx - bw / 2)
        top = int(cy - bh / 2)
        left = max(0, min(left, image_width - bw))
        top = max(0, min(top, image_height - bh))
        right = left + bw
        bottom = top + bh
        rect = (left, top, right, bottom)
        if not any(rects_overlap(rect, r) for r in exclude_rects):
            break
    return (left, top, right, bottom)


def draw_box_on_image(
    image: Image.Image,
    rect: Tuple[int, int, int, int],
    line_width: int = 3,
    color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    left, top, right, bottom = rect
    for w in range(line_width):
        draw.rectangle(
            [left + w, top + w, right - w, bottom - w],
            outline=color,
            fill=None,
        )
    return out


def rect_center(rect: Tuple[int, int, int, int]) -> Tuple[int, int]:
    left, top, right, bottom = rect
    return (left + right) // 2, (top + bottom) // 2


def draw_point_on_image(
    image: Image.Image,
    cx: int,
    cy: int,
    radius: int = 12,
    fill_color: Tuple[int, int, int] = (255, 0, 0),
    outline_color: Tuple[int, int, int] = (0, 0, 0),
    outline_width: int = 2,
) -> Image.Image:
    """Draw a highly visible filled dot (placement guide for point-based edit)."""
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    r = max(4, int(radius))
    ow = max(1, int(outline_width))
    outer = (cx - r - ow, cy - r - ow, cx + r + ow, cy + r + ow)
    draw.ellipse(outer, fill=outline_color, outline=outline_color)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill_color, outline=fill_color)
    return out


def build_point_edit_prompt(
    obj: ObjectRef,
    *,
    avoid_hands: bool = False,
    hand_occlusion: bool = False,
) -> str:
    label = obj.label()
    item = f"a {label}" if label else "an object"
    prompt = (
        f"Add {item} at the red dot marker on the desktop, with a natural object size."
    )
    if hand_occlusion:
        prompt += (
            " The marker overlaps visible hand(s) or fingers; place the object behind "
            "them and keep the hands unchanged."
        )
    elif avoid_hands:
        prompt += (
            " Do not place the object on or overlapping visible hands or fingers."
        )
    prompt += " Remove the red dot marker from the final image."
    ctx = (obj.scene_context or "").strip()
    if ctx:
        prompt = f"{prompt} {ctx.rstrip('.')}."
    return prompt


def build_box_edit_prompt(
    obj: ObjectRef,
    *,
    avoid_hands: bool = False,
    hand_occlusion: bool = False,
) -> str:
    label = obj.label()
    item = f"a {label}" if label else "an object"
    prompt = (
        f"Add {item} on the desktop at the black rectangle in the image. "
        "The rectangle indicates the location but does not constrain the object size; "
        "use a natural object size."
    )
    if hand_occlusion:
        prompt += (
            " The rectangle overlaps visible hand(s) or fingers; place the object behind "
            "them and keep the hands unchanged."
        )
    elif avoid_hands:
        prompt += (
            " Do not place the object on or overlapping visible hands or fingers."
        )
    prompt += " Remove the black rectangle from the final image."
    ctx = (obj.scene_context or "").strip()
    if ctx:
        prompt = f"{prompt} {ctx.rstrip('.')}."
    return prompt


def load_qwen_image_edit_pipeline(
    model_path: str,
    device: str = "cuda",
    lightning_lora_path: Optional[str] = None,
    vram_management: Optional[dict[str, Any]] = None,
):
    """Load Qwen-Image-Edit-2511 via DiffSynth-Studio (optional Lightning LoRA)."""
    _ensure_diffsynth_path()
    import torch
    from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline

    root = Path(model_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Image edit model directory not found: {root}")

    def find_model_files(subdir: str, pattern: str = "*.safetensors") -> list:
        dir_path = root / subdir
        files = list(dir_path.glob(pattern))
        if not files:
            files = list(dir_path.glob("*.bin"))
        if not files:
            raise FileNotFoundError(f"No model files in {dir_path}")
        return [str(f) for f in sorted(files)]

    transformer_files = find_model_files(
        "transformer", "diffusion_pytorch_model*.safetensors"
    )
    text_encoder_files = find_model_files("text_encoder", "model*.safetensors")
    vae_files = find_model_files("vae", "diffusion_pytorch_model*.safetensors")

    processor_path = root / "processor"
    if not processor_path.exists():
        raise FileNotFoundError(f"Processor not found: {processor_path}")

    tokenizer_path = root / "tokenizer"
    tokenizer_config = (
        ModelConfig(path=str(tokenizer_path)) if tokenizer_path.exists() else None
    )
    vram_kwargs, vram_limit = parse_diffsynth_vram_config(
        vram_management,
        device=device,
        torch_dtype=torch.bfloat16,
    )

    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(
                path=transformer_files
                if len(transformer_files) > 1
                else transformer_files[0],
                **vram_kwargs,
            ),
            ModelConfig(
                path=text_encoder_files
                if len(text_encoder_files) > 1
                else text_encoder_files[0],
                **vram_kwargs,
            ),
            ModelConfig(
                path=vae_files if len(vae_files) > 1 else vae_files[0],
                **vram_kwargs,
            ),
        ],
        tokenizer_config=tokenizer_config,
        processor_config=ModelConfig(path=str(processor_path)),
        vram_limit=vram_limit,
    )

    lora = resolve_lightning_lora_path(lightning_lora_path or "")
    if lora:
        pipe.load_lora(pipe.dit, ModelConfig(path=lora), alpha=1.0)
        logger.debug("Loaded Lightning LoRA: %s", lora)

    logger.debug("Qwen-Image-Edit loaded from %s", root)
    return pipe


def _parse_torch_dtype(value: Any, torch_module: Any) -> Any:
    if value is None or value == "":
        return None
    if value == "disk":
        return "disk"
    if not isinstance(value, str):
        return value
    name = value.strip()
    if not name:
        return None
    if name.startswith("torch."):
        name = name.split(".", 1)[1]
    if not hasattr(torch_module, name):
        raise ValueError(f"Unsupported torch dtype in VRAM config: {value}")
    return getattr(torch_module, name)


def parse_diffsynth_vram_config(
    cfg: Optional[dict[str, Any]],
    *,
    device: str,
    torch_dtype: Any,
) -> tuple[dict[str, Any], Optional[float]]:
    if not cfg or not bool(cfg.get("enable", False)):
        return {}, None

    import torch

    mode = str(cfg.get("mode", "dynamic_cpu")).strip().lower()
    if mode == "dynamic_cpu":
        defaults = {
            "offload_dtype": "bfloat16",
            "offload_device": "cpu",
            "onload_dtype": "bfloat16",
            "onload_device": "cpu",
            "preparing_dtype": "bfloat16",
            "preparing_device": device,
            "computation_dtype": "bfloat16",
            "computation_device": device,
        }
    elif mode == "fp8_cpu":
        defaults = {
            "offload_dtype": "float8_e4m3fn",
            "offload_device": "cpu",
            "onload_dtype": "float8_e4m3fn",
            "onload_device": "cpu",
            "preparing_dtype": "float8_e4m3fn",
            "preparing_device": device,
            "computation_dtype": "bfloat16",
            "computation_device": device,
        }
    elif mode == "disk":
        defaults = {
            "offload_dtype": "disk",
            "offload_device": "disk",
            "onload_dtype": "disk",
            "onload_device": "disk",
            "preparing_dtype": "bfloat16",
            "preparing_device": device,
            "computation_dtype": "bfloat16",
            "computation_device": device,
        }
    else:
        raise ValueError(f"Unsupported DiffSynth VRAM mode: {mode}")

    merged = {**defaults, **{k: v for k, v in cfg.items() if k in defaults}}
    kwargs = {
        "offload_dtype": _parse_torch_dtype(merged["offload_dtype"], torch),
        "offload_device": merged["offload_device"],
        "onload_dtype": _parse_torch_dtype(merged["onload_dtype"], torch),
        "onload_device": merged["onload_device"],
        "preparing_dtype": _parse_torch_dtype(merged["preparing_dtype"], torch),
        "preparing_device": merged["preparing_device"],
        "computation_dtype": _parse_torch_dtype(
            merged.get("computation_dtype", torch_dtype), torch
        ),
        "computation_device": merged["computation_device"],
    }
    vram_limit = cfg.get("vram_limit_gb", None)
    if vram_limit is not None and vram_limit != "":
        vram_limit = float(vram_limit)
    else:
        vram_limit = None
    return kwargs, vram_limit


class BoxImageEditor:
    """Box-based Qwen-Image-Edit-2511 wrapper. Skips model load when disabled."""

    def __init__(
        self,
        model_path: str = "",
        device: str = "cuda",
        enabled: bool = True,
        lightning_lora_path: str = "",
        edit_steps: int = 4,
        edit_height: int = 0,
        edit_width: int = 0,
        edit_cfg_scale: float = 1.0,
        use_native_resolution: bool = True,
        negative_prompt: str = "",
        mask_width: int = 256,
        mask_height: int = 256,
        center_x_range: Tuple[float, float] = (0.25, 0.75),
        center_y_range: Tuple[float, float] = (0.25, 0.75),
        box_line_width: int = 3,
        max_edit_retries: int = 8,
        min_edit_mean_diff: float = 2.0,
        guide_mode: str = GUIDE_MODE_POINT,
        point_radius: int = 12,
        point_color: Tuple[int, int, int] = (255, 0, 0),
        point_outline_width: int = 2,
        keep_loaded: bool = False,
        vram_management: Optional[dict[str, Any]] = None,
    ):
        self.model_path = model_path
        self.device = device
        self.enabled = enabled and bool(model_path)
        self.lightning_lora_path = lightning_lora_path
        self.edit_steps = edit_steps
        raw_h, raw_w = edit_height, edit_width
        self.edit_height = align_edit_dim(edit_height) if edit_height else 0
        self.edit_width = align_edit_dim(edit_width) if edit_width else 0
        if (
            (raw_w, raw_h) != (self.edit_width, self.edit_height)
            and raw_w
            and raw_h
        ):
            logger.debug(
                "Aligned edit size %dx%d -> %dx%d (Qwen-Image-Edit needs multiples of %d)",
                raw_w,
                raw_h,
                self.edit_width,
                self.edit_height,
                EDIT_DIM_ALIGN,
            )
        self.edit_cfg_scale = edit_cfg_scale
        self.use_native_resolution = use_native_resolution
        self.negative_prompt = negative_prompt
        self.mask_width = mask_width
        self.mask_height = mask_height
        self.center_x_range = center_x_range
        self.center_y_range = center_y_range
        self.box_line_width = box_line_width
        self.max_edit_retries = max(1, int(max_edit_retries))
        self.min_edit_mean_diff = float(min_edit_mean_diff)
        mode = str(guide_mode or GUIDE_MODE_POINT).lower().strip()
        self.guide_mode = mode if mode in (GUIDE_MODE_BBOX, GUIDE_MODE_POINT) else GUIDE_MODE_POINT
        self.point_radius = max(4, int(point_radius))
        self.point_color = tuple(int(c) for c in point_color[:3])
        self.point_outline_width = max(1, int(point_outline_width))
        self.keep_loaded = bool(keep_loaded)
        self.vram_management = dict(vram_management or {})
        self._pipe = None
        self._load_on_init = False
        self.load_elapsed_sec = 0.0
        self.infer_elapsed_sec = 0.0
        self.edit_calls = 0

    def fixed_edit_size(self) -> Optional[Tuple[int, int]]:
        """(width, height) when editing at a fixed resolution, else None."""
        if self.use_native_resolution or not self.edit_width or not self.edit_height:
            return None
        return self.edit_width, self.edit_height

    def _resize_for_edit(self, image: Image.Image) -> Tuple[Image.Image, int, int]:
        fixed = self.fixed_edit_size()
        if fixed is None:
            w, h = image.size
            return image, w, h
        target_w, target_h = fixed
        if image.size != (target_w, target_h):
            logger.debug(
                "Resize for edit: %dx%d -> %dx%d",
                image.size[0],
                image.size[1],
                target_w,
                target_h,
            )
            image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        return image, target_w, target_h

    def _ensure_pipe(self) -> None:
        if not self.enabled:
            return
        if self._pipe is not None:
            if not self.vram_management.get("enable", False):
                self.onload_pipe()
            return
        t0 = time.perf_counter()
        self._pipe = load_qwen_image_edit_pipeline(
            self.model_path,
            device=self.device,
            lightning_lora_path=self.lightning_lora_path,
            vram_management=self.vram_management,
        )
        self.load_elapsed_sec += time.perf_counter() - t0

    def _move_pipe(self, device: str) -> None:
        if self._pipe is None:
            return
        for name in (
            "dit",
            "vae",
            "text_encoder",
            "image_encoder",
            "controlnet",
            "ipadapter",
        ):
            module = getattr(self._pipe, name, None)
            if module is not None:
                module.to(device)

    def offload_pipe(self) -> None:
        """Move image-edit modules to CPU without destroying loaded weights."""
        import gc

        import torch

        if self._pipe is None:
            return
        if self.vram_management.get("enable", False):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return
        self._move_pipe("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("Offloaded Qwen-Image-Edit model to CPU")

    def onload_pipe(self) -> None:
        """Move image-edit modules back to the inference device."""
        if self._pipe is None:
            return
        if self.vram_management.get("enable", False):
            return
        self._move_pipe(self.device)

    def release_pipe(self, *, force: bool = False) -> None:
        """Free image-edit weights before reloading VL on the next image."""
        import gc

        import torch

        if self._pipe is None:
            return
        if self.keep_loaded and not force:
            logger.debug("Keeping Qwen-Image-Edit model resident")
            return
        self._pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("Released Qwen-Image-Edit model from GPU")

    def timing_snapshot(self) -> dict:
        return {
            "load_elapsed_sec": self.load_elapsed_sec,
            "infer_elapsed_sec": self.infer_elapsed_sec,
            "edit_calls": self.edit_calls,
        }

    def edit_object_into_image(
        self,
        image_path: str,
        obj: ObjectRef,
        output_path: str,
        seed: Optional[int] = None,
        exclude_rects: Optional[List[Tuple[int, int, int, int]]] = None,
        target_rect_px: Optional[Tuple[int, int, int, int]] = None,
        min_box_px: int = 16,
        trust_assigned_bbox: bool = False,
        avoid_hands: bool = False,
        hand_occlusion: bool = False,
        negative_prompt: Optional[str] = None,
        guide_mode: Optional[str] = None,
    ) -> Tuple[str, Tuple[int, int, int, int]]:
        """
        Add `obj` into a box region. Returns (output_path, box_rect in pixels).

        If target_rect_px is set (from Stage-1 feasible bbox_2d), use it;
        otherwise fall back to random placement.
        """
        image = Image.open(image_path).convert("RGB")
        image, w, h = self._resize_for_edit(image)
        exclude_rects = exclude_rects or []
        base_seed = seed if seed is not None else 0
        max_attempts = self.max_edit_retries if self.enabled else 1

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        rect: Optional[Tuple[int, int, int, int]] = None
        if target_rect_px is not None:
            left, top, right, bottom = target_rect_px
            candidate = (
                max(0, left),
                max(0, top),
                min(w, right),
                min(h, bottom),
            )
            if candidate[2] - candidate[0] >= min_box_px and candidate[3] - candidate[1] >= min_box_px:
                if trust_assigned_bbox or not any(
                    rects_overlap(candidate, r) for r in exclude_rects
                ):
                    rect = candidate

        if rect is None and obj.bbox_2d:
            rect = qwen_bbox_to_pixel_rect(
                obj.bbox_2d, w, h, min_size_px=min_box_px
            )
            if rect and not trust_assigned_bbox and any(
                rects_overlap(rect, r) for r in exclude_rects
            ):
                rect = None

        fixed_rect = rect is not None
        if rect is None:
            rect = create_random_edit_rect(
                w,
                h,
                self.mask_width,
                self.mask_height,
                self.center_x_range,
                self.center_y_range,
                seed=base_seed,
                exclude_rects=exclude_rects,
            )
            logger.debug(
                "No valid VL bbox for %s — using random edit rect %s",
                obj.label(),
                rect,
            )

        mode = str(guide_mode or self.guide_mode).lower().strip()
        if mode not in (GUIDE_MODE_BBOX, GUIDE_MODE_POINT):
            mode = self.guide_mode

        prompt = (
            build_point_edit_prompt(
                obj,
                avoid_hands=avoid_hands and not hand_occlusion,
                hand_occlusion=hand_occlusion,
            )
            if mode == GUIDE_MODE_POINT
            else build_box_edit_prompt(
                obj,
                avoid_hands=avoid_hands and not hand_occlusion,
                hand_occlusion=hand_occlusion,
            )
        )
        logger.debug("%s edit prompt: %s", mode, prompt)

        last_rect = rect
        last_cleaned: Optional[Image.Image] = None
        use_point = mode == GUIDE_MODE_POINT

        for attempt in range(max_attempts):
            attempt_seed = base_seed + attempt * 104729
            if attempt > 0 and not fixed_rect:
                rect = create_random_edit_rect(
                    w,
                    h,
                    self.mask_width,
                    self.mask_height,
                    self.center_x_range,
                    self.center_y_range,
                    seed=attempt_seed,
                    exclude_rects=exclude_rects,
                )
            last_rect = rect
            cx, cy = rect_center(rect)

            if use_point:
                guided = draw_point_on_image(
                    image,
                    cx,
                    cy,
                    radius=self.point_radius,
                    fill_color=self.point_color,
                    outline_width=self.point_outline_width,
                )
            else:
                guided = draw_box_on_image(image, rect, line_width=self.box_line_width)

            if not self.enabled:
                guided.save(out_path)
                logger.warning(
                    "Image edit model disabled — saved %s guide only: %s",
                    mode,
                    out_path,
                )
                return str(out_path), rect

            self._ensure_pipe()
            out_w, out_h = w, h

            import os

            import torch

            prev_tqdm_disable = os.environ.get("TQDM_DISABLE")
            os.environ["TQDM_DISABLE"] = "1"
            try:
                with torch.no_grad():
                    base_neg = (
                        negative_prompt
                        if negative_prompt is not None
                        else self.negative_prompt
                    )
                    guide_neg = (
                        POINT_NEGATIVE_PROMPT if use_point else BOX_NEGATIVE_PROMPT
                    )
                    neg = merge_negative_prompt(base_neg, guide_neg)
                    infer_t0 = time.perf_counter()
                    edited = self._pipe(
                        prompt=prompt,
                        negative_prompt=neg or None,
                        edit_image=[guided],
                        seed=attempt_seed,
                        num_inference_steps=self.edit_steps,
                        height=out_h,
                        width=out_w,
                        cfg_scale=self.edit_cfg_scale,
                        edit_image_auto_resize=False,
                        zero_cond_t=True,
                    )
                    self.infer_elapsed_sec += time.perf_counter() - infer_t0
                    self.edit_calls += 1
            finally:
                if prev_tqdm_disable is None:
                    os.environ.pop("TQDM_DISABLE", None)
                else:
                    os.environ["TQDM_DISABLE"] = prev_tqdm_disable

            if isinstance(edited, list):
                edited = edited[0]

            cleaned = edited.convert("RGB")
            if cleaned.size != image.size:
                cleaned = cleaned.resize(image.size, Image.Resampling.LANCZOS)
            last_rect = rect
            last_cleaned = cleaned

            changed = images_meaningfully_differ(
                image, cleaned, min_mean_diff=self.min_edit_mean_diff
            )
            if changed:
                cleaned.save(out_path)
                logger.debug("Saved edited image: %s (%dx%d)", out_path, out_w, out_h)
                return str(out_path), rect

            if attempt + 1 < max_attempts:
                logger.warning(
                    "Edit unchanged for %s (attempt %d/%d); retrying with seed %d",
                    obj.label(),
                    attempt + 1,
                    max_attempts,
                    attempt_seed + 104729,
                )

        if last_cleaned is not None:
            last_cleaned.save(out_path)
            logger.warning(
                "Edit still unchanged for %s after %d attempts; saving last output",
                obj.label(),
                max_attempts,
            )
            return str(out_path), last_rect

        raise RuntimeError(f"Image edit failed for {obj.label()}")


def make_image_editor(
    edit_cfg: dict,
    device: str,
    *,
    enabled: Optional[bool] = None,
    mask_width: Optional[int] = None,
    mask_height: Optional[int] = None,
    center_x_range: Optional[Tuple[float, float]] = None,
    center_y_range: Optional[Tuple[float, float]] = None,
) -> BoxImageEditor:
    """Build BoxImageEditor from ``image_edit`` config (image_first / instr_first)."""
    cfg = dict(edit_cfg or {})
    edit_width = int(cfg.get("video_width", cfg.get("edit_width", 0)) or 0)
    edit_height = int(cfg.get("video_height", cfg.get("edit_height", 0)) or 0)
    point_color_raw = cfg.get("point_color", [255, 0, 0])
    if isinstance(point_color_raw, (list, tuple)) and len(point_color_raw) >= 3:
        point_color = tuple(int(c) for c in point_color_raw[:3])
    else:
        point_color = (255, 0, 0)

    return BoxImageEditor(
        model_path=str(cfg.get("model_path", "") or ""),
        device=device,
        enabled=bool(cfg.get("enable", False)) if enabled is None else enabled,
        lightning_lora_path=str(cfg.get("lightning_lora_path", "") or ""),
        edit_steps=int(cfg.get("edit_steps", 4)),
        edit_height=edit_height,
        edit_width=edit_width,
        edit_cfg_scale=float(cfg.get("edit_cfg_scale", 1.0)),
        use_native_resolution=bool(cfg.get("use_native_resolution", False)),
        negative_prompt=str(cfg.get("negative_prompt", "") or ""),
        mask_width=int(mask_width if mask_width is not None else cfg.get("mask_width", 256)),
        mask_height=int(mask_height if mask_height is not None else cfg.get("mask_height", 256)),
        center_x_range=tuple(
            center_x_range
            if center_x_range is not None
            else cfg.get("center_x_range", [0.25, 0.75])
        ),
        center_y_range=tuple(
            center_y_range
            if center_y_range is not None
            else cfg.get("center_y_range", [0.25, 0.75])
        ),
        box_line_width=int(cfg.get("box_line_width", 3)),
        max_edit_retries=int(cfg.get("max_edit_retries", 8)),
        min_edit_mean_diff=float(cfg.get("min_edit_mean_diff", 2.0)),
        guide_mode=str(cfg.get("guide_mode", GUIDE_MODE_POINT)),
        point_radius=int(cfg.get("point_radius", 12)),
        point_color=point_color,
        point_outline_width=int(cfg.get("point_outline_width", 2)),
        keep_loaded=bool(cfg.get("keep_loaded", False)),
        vram_management=cfg.get("vram_management") or {},
    )

"""Small vLLM wrapper for Qwen-style text and vision-language generation."""

from __future__ import annotations

import os
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image


IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def _qwen_chat_prompt(text: str, *, has_image: bool = False) -> str:
    prefix = IMAGE_PLACEHOLDER if has_image else ""
    return (
        f"<|im_start|>user\n{prefix}{text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _sampling_params(gen_cfg: Dict[str, Any], max_new_tokens: int):
    try:
        from vllm import SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "vLLM backend requested but vllm is not installed in this environment"
        ) from exc

    do_sample = bool(gen_cfg.get("do_sample", True))
    kwargs: Dict[str, Any] = {"max_tokens": int(max_new_tokens)}
    if do_sample:
        kwargs["temperature"] = float(gen_cfg.get("temperature", 0.7))
        kwargs["top_p"] = float(gen_cfg.get("top_p", 0.9))
        if "top_k" in gen_cfg:
            kwargs["top_k"] = int(gen_cfg["top_k"])
    else:
        kwargs["temperature"] = 0.0
    return SamplingParams(**kwargs)


class VLLMGenerator:
    """Lazy vLLM engine with Qwen chat prompt helpers."""

    def __init__(
        self,
        model_path: str,
        *,
        gen_cfg: Optional[Dict[str, Any]] = None,
        image_max_side: int = 1024,
    ):
        self.model_path = model_path
        self.gen_cfg = dict(gen_cfg or {})
        self.image_max_side = int(image_max_side)
        self._llm = None
        if self.gen_cfg.get("use_v1") is False:
            os.environ.setdefault("VLLM_USE_V1", "0")

    def _run_subprocess(
        self,
        *,
        mode: str,
        prompts: List[str],
        max_new_tokens: int,
        retry_attempt: int = 0,
        image_paths: Optional[List[str]] = None,
    ) -> List[str]:
        root = Path(__file__).resolve().parent.parent
        helper = root / "scripts" / "vllm_generate_batch.py"
        payload = {
            "mode": mode,
            "model_path": self.model_path,
            "gen_cfg": {**self.gen_cfg, "subprocess_per_call": False},
            "image_max_side": self.image_max_side,
            "prompts": prompts,
            "image_paths": image_paths or [],
            "max_new_tokens": int(max_new_tokens),
            "retry_attempt": int(retry_attempt),
        }
        with tempfile.TemporaryDirectory(prefix="dsp_vllm_") as tmp:
            req = Path(tmp) / "request.json"
            out = Path(tmp) / "output.json"
            req.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            env = os.environ.copy()
            env["DSP_VLLM_HELPER"] = "1"
            env["PYTHONPATH"] = (
                str(root)
                if not env.get("PYTHONPATH")
                else f"{root}:{env['PYTHONPATH']}"
            )
            subprocess.run(
                [sys.executable, str(helper), str(req), str(out)],
                check=True,
                env=env,
            )
            data = json.loads(out.read_text(encoding="utf-8"))
        return [str(x) for x in data.get("outputs", [])]

    def _ensure_llm(self):
        if self._llm is not None:
            return self._llm
        try:
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError(
                "vLLM backend requested but vllm is not installed in this environment"
            ) from exc

        max_pixels = self.image_max_side * self.image_max_side
        mm_kwargs = {
            "min_pixels": int(self.gen_cfg.get("min_pixels", 28 * 28)),
            "max_pixels": int(self.gen_cfg.get("max_pixels", max_pixels)),
        }
        engine_kwargs: Dict[str, Any] = {
            "model": self.model_path,
            "trust_remote_code": bool(self.gen_cfg.get("trust_remote_code", True)),
            "tensor_parallel_size": int(self.gen_cfg.get("tensor_parallel_size", 1)),
            "gpu_memory_utilization": float(
                self.gen_cfg.get("gpu_memory_utilization", 0.35)
            ),
            "limit_mm_per_prompt": {"image": 1},
            "mm_processor_kwargs": mm_kwargs,
            "enforce_eager": bool(self.gen_cfg.get("enforce_eager", True)),
        }
        if self.gen_cfg.get("max_model_len"):
            engine_kwargs["max_model_len"] = int(self.gen_cfg["max_model_len"])
        if self.gen_cfg.get("max_num_seqs"):
            engine_kwargs["max_num_seqs"] = int(self.gen_cfg["max_num_seqs"])
        if self.gen_cfg.get("dtype"):
            engine_kwargs["dtype"] = str(self.gen_cfg["dtype"])
        for key in (
            "quantization",
            "kv_cache_dtype",
            "load_format",
            "max_num_batched_tokens",
            "calculate_kv_scales",
        ):
            if key in self.gen_cfg and self.gen_cfg[key] not in (None, ""):
                engine_kwargs[key] = self.gen_cfg[key]
        if self.gen_cfg.get("mm_encoder_tp_mode"):
            engine_kwargs["mm_encoder_tp_mode"] = str(self.gen_cfg["mm_encoder_tp_mode"])

        try:
            self._llm = LLM(**engine_kwargs)
        except ValueError as exc:
            if "limit_mm_per_prompt" not in str(exc):
                raise
            # Some vLLM/model combinations reject multimodal processor kwargs even
            # for Qwen VL checkpoints. Retry without these caps; image payloads are
            # still supplied per request in generate_vision_texts().
            engine_kwargs.pop("limit_mm_per_prompt", None)
            engine_kwargs.pop("mm_processor_kwargs", None)
            self._llm = LLM(**engine_kwargs)
        return self._llm

    def generate_texts(
        self,
        prompts: List[str],
        *,
        max_new_tokens: int,
        retry_attempt: int = 0,
    ) -> List[str]:
        if not prompts:
            return []
        if self.gen_cfg.get("subprocess_per_call") and not os.environ.get("DSP_VLLM_HELPER"):
            return self._run_subprocess(
                mode="text",
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                retry_attempt=retry_attempt,
            )
        gen_cfg = dict(self.gen_cfg)
        if retry_attempt > 0:
            gen_cfg["do_sample"] = True
            gen_cfg["temperature"] = float(gen_cfg.get("retry_temperature", 0.85))
        sampling = _sampling_params(gen_cfg, max_new_tokens)
        llm_inputs = [{"prompt": _qwen_chat_prompt(prompt)} for prompt in prompts]
        outputs = self._ensure_llm().generate(llm_inputs, sampling_params=sampling)
        return [out.outputs[0].text.strip() if out.outputs else "" for out in outputs]

    def generate_vision_texts(
        self,
        image_paths: List[str],
        prompts: List[str],
        *,
        max_new_tokens: int,
        retry_attempt: int = 0,
    ) -> List[str]:
        if len(image_paths) != len(prompts):
            raise ValueError("image_paths and prompts length mismatch")
        if not image_paths:
            return []
        if self.gen_cfg.get("subprocess_per_call") and not os.environ.get("DSP_VLLM_HELPER"):
            return self._run_subprocess(
                mode="vision",
                prompts=prompts,
                image_paths=image_paths,
                max_new_tokens=max_new_tokens,
                retry_attempt=retry_attempt,
            )
        gen_cfg = dict(self.gen_cfg)
        if retry_attempt > 0:
            gen_cfg["do_sample"] = True
            gen_cfg["temperature"] = float(gen_cfg.get("retry_temperature", 0.85))
        sampling = _sampling_params(gen_cfg, max_new_tokens)
        llm_inputs = []
        for image_path, prompt in zip(image_paths, prompts):
            image = Image.open(Path(image_path)).convert("RGB")
            llm_inputs.append(
                {
                    "prompt": _qwen_chat_prompt(prompt, has_image=True),
                    "multi_modal_data": {"image": image},
                }
            )
        outputs = self._ensure_llm().generate(llm_inputs, sampling_params=sampling)
        return [out.outputs[0].text.strip() if out.outputs else "" for out in outputs]

    def release(self) -> None:
        if self._llm is not None:
            try:
                engine = getattr(self._llm, "llm_engine", None)
                engine_core = getattr(engine, "engine_core", None)
                if engine_core is not None and hasattr(engine_core, "shutdown"):
                    engine_core.shutdown()
            except Exception:
                pass
        self._llm = None
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

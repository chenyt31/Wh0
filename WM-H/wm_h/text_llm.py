"""Text-only LLM for instr_first: causal LM or Qwen3-VL (no image input)."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
)

from .vllm_backend import VLLMGenerator

logger = logging.getLogger("wm_h.text_llm")


def _strip_think(text: str) -> str:
    return re.sub(
        r"<think>.*?</think>\s*",
        "",
        text,
        flags=re.DOTALL,
    ).strip()


def detect_backend(model_path: str) -> str:
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.is_file():
        return "causal_lm"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    architectures = cfg.get("architectures") or []
    model_type = str(cfg.get("model_type", "")).lower()
    if any("Qwen3VL" in a for a in architectures) or "qwen3_vl" in model_type:
        return "qwen3_vl"
    return "causal_lm"


class InstrFirstTextLLM:
    """Loads causal_lm or qwen3_vl and runs batched text generation."""

    def __init__(self, model_cfg: dict, device: str = "cuda"):
        self.path = model_cfg["path"]
        backend = str(model_cfg.get("backend", "auto")).lower()
        if backend == "auto":
            backend = detect_backend(self.path)
        self.backend = backend
        self.device = device
        self.enable_thinking = bool(model_cfg.get("enable_thinking", False))
        self.model = None
        self.tokenizer = None
        self.processor = None
        self._vllm = None
        if self.backend == "vllm":
            self._vllm = VLLMGenerator(
                self.path,
                gen_cfg=model_cfg,
                image_max_side=int(model_cfg.get("image_max_side", 1024)),
            )
            logger.debug("vLLM ready (backend=vllm)")
        else:
            self._load(model_cfg)

    def _load(self, mc: dict) -> None:
        dtype_str = mc.get("torch_dtype", "auto")
        dtype = getattr(torch, dtype_str, "auto") if dtype_str != "auto" else "auto"
        trust = mc.get("trust_remote_code", True)
        local = mc.get("local_files_only", True)

        if self.backend == "qwen3_vl":
            logger.debug(
                "Loading Qwen3-VL for instr_first (8B, first load ~1-3 min): %s",
                self.path,
            )
            device_map = mc.get("device_map", "auto")
            if device_map == "auto" and str(self.device) != "cuda":
                device_map = {"": self.device}
            load_kwargs = {
                "torch_dtype": dtype,
                "trust_remote_code": trust,
                "local_files_only": local,
            }
            if device_map == "auto":
                load_kwargs["device_map"] = "auto"
            else:
                load_kwargs["device_map"] = {"": self.device}
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.path,
                **load_kwargs,
            ).eval()
            self.processor = AutoProcessor.from_pretrained(
                self.path,
                trust_remote_code=trust,
                local_files_only=local,
            )
            self.processor.tokenizer.padding_side = "left"
            if self.processor.tokenizer.pad_token_id is None:
                self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
            self.tokenizer = self.processor.tokenizer
            logger.debug("Qwen3-VL ready (backend=qwen3_vl)")
        else:
            logger.debug("Loading causal LM for instr_first: %s", self.path)
            device_map = mc.get("device_map", "auto")
            if device_map == "auto" and str(self.device) != "cuda":
                device_map = {"": self.device}
            self.model = AutoModelForCausalLM.from_pretrained(
                self.path,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=trust,
                local_files_only=local,
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.path,
                trust_remote_code=trust,
                local_files_only=local,
            )
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.debug("Causal LM ready (backend=causal_lm)")

    def release(self) -> None:
        import gc

        if self.backend == "vllm":
            if self._vllm is not None:
                self._vllm.release()
            return
        self.model = None
        self.tokenizer = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def offload(self) -> None:
        import gc

        if self.backend == "vllm":
            return
        if self.model is None:
            return
        try:
            self.model.to("cpu")
        except RuntimeError:
            logger.warning("Text LLM CPU offload failed; keeping current placement")
            return
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def onload(self) -> None:
        if self.backend == "vllm":
            return
        if self.model is None:
            return
        try:
            self.model.to(self.device)
        except RuntimeError:
            logger.warning("Text LLM onload failed; keeping current placement")

    @torch.no_grad()
    def infer_batch(self, prompts: List[str], gen_cfg: dict) -> List[str]:
        if not prompts:
            return []

        max_new = int(gen_cfg.get("max_new_tokens", 512))
        if self.backend == "vllm":
            if self._vllm is None:
                self._vllm = VLLMGenerator(
                    self.path,
                    gen_cfg={**gen_cfg, "model_path": self.path},
                )
            texts = self._vllm.generate_texts(
                prompts,
                max_new_tokens=max_new,
            )
            return [_strip_think(t) for t in texts]

        if self.model is None:
            return []

        do_sample = bool(gen_cfg.get("do_sample", True))
        temperature = float(gen_cfg.get("temperature", 0.7))
        top_p = float(gen_cfg.get("top_p", 0.8))
        top_k = int(gen_cfg.get("top_k", 50))
        enable_thinking = bool(
            gen_cfg.get("enable_thinking", self.enable_thinking)
        )

        gen_kw: dict = {"max_new_tokens": max_new, "do_sample": do_sample}
        if do_sample:
            gen_kw.update(
                temperature=temperature, top_p=top_p, top_k=top_k
            )

        if self.backend == "qwen3_vl" and self.processor is not None:
            messages_batch = [
                [{"role": "user", "content": [{"type": "text", "text": p}]}]
                for p in prompts
            ]
            inputs = self.processor.apply_chat_template(
                messages_batch,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
                enable_thinking=enable_thinking,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            in_ids = inputs["input_ids"]
            out = self.model.generate(**inputs, **gen_kw)
            trimmed = [o[len(i):] for i, o in zip(in_ids, out)]
            texts = self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        else:
            assert self.tokenizer is not None
            chat_texts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
                for p in prompts
            ]
            inputs = self.tokenizer(chat_texts, padding=True, return_tensors="pt")
            dev = next(self.model.parameters()).device
            inputs = {k: v.to(dev) for k, v in inputs.items()}
            in_len = inputs["input_ids"].shape[1]
            out = self.model.generate(**inputs, **gen_kw)
            texts = self.tokenizer.batch_decode(
                out[:, in_len:], skip_special_tokens=True
            )

        return [_strip_think(t) for t in texts]

    def infer_one(self, prompt: str, gen_cfg: dict) -> str:
        rows = self.infer_batch([prompt], gen_cfg)
        return rows[0] if rows else ""

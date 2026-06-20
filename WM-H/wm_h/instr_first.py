"""
Instruction-first generator: slot templates + per-slot word DB.

Agent1 (LLM): bulk-generate words per slot / grammatical type.
Agent2 (rules): sample low-usage words from DB → fill templates → dedupe.
"""

from __future__ import annotations

import json
import logging
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

from .instruction_builder import (
    is_cooperative_bimanual_decomposition,
    is_valid_bimanual_instruction,
)
from .schema import BIMANUAL_COORDINATION_TYPES, FeasibleInstruction, ObjectRef
from .schema import parse_json_from_llm
from .slot_database import SlotDatabase
from .slot_templates import SlotTemplateEngine
from .terminal_ui import slot_counts
from .text_llm import InstrFirstTextLLM

logger = logging.getLogger("wm_h.instr_first")

_REPO_ROOT = Path(__file__).resolve().parent.parent

_STOPWORDS = frozenset({
    "a", "an", "the", "it", "its", "is", "am", "are", "was", "were", "be",
    "on", "in", "at", "to", "by", "of", "and", "or", "but", "not", "no",
    "do", "did", "has", "had", "have", "can", "may", "will", "shall",
    "this", "that", "then", "than", "from", "with", "into", "onto", "for",
    "near", "beside", "next", "between", "under", "over", "above", "below",
    "up", "down", "out", "off", "away", "back", "here", "there", "where",
    "each", "every", "some", "any", "all", "both", "other", "another",
})

_NOUN_SUFFIXES = (
    "tion", "ness", "ment", "ity", "ure", "age", "ance", "ence",
    "er", "or", "ist", "ian", "ism", "dom", "ship", "hood",
)


def _strip_think(text: str) -> str:
    return re.sub(
        r"<think>.*?</think>\s*",
        "",
        text,
        flags=re.DOTALL,
    ).strip()


def _slot_type(slots_cfg: Dict[str, dict], slot_name: str) -> str:
    return str(slots_cfg.get(slot_name, {}).get("type", "")).lower()


def _valid_word(word: str, slot_type: str) -> bool:
    w = word.strip().lower()
    if not w:
        return False
    if slot_type in {"prep", "preposition", "p"}:
        return bool(re.match(r"^[a-z]+(?: [a-z]+)*$", w)) and len(w) <= 30
    return bool(re.match(r"^[a-z]{3,24}$", w)) and w not in _STOPWORDS


def _is_likely_noun(w: str) -> bool:
    for suffix in _NOUN_SUFFIXES:
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            return True
    return False


def _is_near_duplicate(w: str, pool: Set[str]) -> bool:
    for existing in pool:
        if w == existing:
            continue
        for i in range(4, min(len(w), len(existing))):
            if w.startswith(existing[:i]) or existing.startswith(w[:i]):
                return True
        if abs(len(w) - len(existing)) <= 2:
            diff = sum(c1 != c2 for c1, c2 in zip(w, existing))
            if diff <= 2:
                return True
    return False


def _normalize_template(raw: dict) -> dict:
    pattern = raw.get("task_description") or raw.get("pattern") or ""
    tid = str(raw.get("id") or "tpl")
    return {
        "id": tid,
        "pattern": str(pattern),
        "use_both_hands": bool(raw.get("use_both_hands", False)),
    }


def _format_pattern(pattern: str, samples: Dict[str, str]) -> str:
    text = pattern
    for slot, word in samples.items():
        if slot.startswith("d") and not word:
            text = text.replace("{" + slot + "} ", "")
            text = text.replace("{" + slot + "}", "")
        else:
            text = text.replace("{" + slot + "}", word)
    return " ".join(text.split())


def _assembled_to_feasible(
    instr: str,
    slots: Dict[str, str],
    template_id: str,
    pattern: str,
) -> FeasibleInstruction:
    nouns: List[str] = []
    adjectives: List[str] = []
    for key in sorted(slots.keys()):
        val = slots[key].strip()
        if not val:
            continue
        if key.startswith("n"):
            nouns.append(val)
        elif key.startswith("d"):
            adjectives.append(val)

    # Pair adjectives with nouns by index (d0→n0, d1→n1)
    paired_adjs: List[str] = []
    for i, noun in enumerate(nouns):
        adj_key = f"d{i}"
        paired_adjs.append(slots.get(adj_key, "").strip())

    objects = [
        ObjectRef(noun=n, adjective=a)
        for n, a in zip(nouns, paired_adjs)
    ]

    verbs: List[str] = []
    if slots.get("v0"):
        verbs = [slots["v0"].strip().lower()]
    else:
        verbs = SlotTemplateEngine.extract_verbs(pattern, slots)

    action_type = "two_object" if len(nouns) >= 2 else "single"
    return FeasibleInstruction(
        action_type=action_type,
        task_description=instr,
        verbs=verbs,
        nouns=nouns,
        adjectives=[a for a in paired_adjs if a],
        objects=objects,
    )


# ── Agent1: per-slot vocabulary expansion ──────────────────

class SlotVocabDiscoverer:
    """LLM generates new words for each slot (by grammatical / semantic type)."""

    # diversity hints for vocab expansion prompts
    HINTS = [
        "Focus on kitchen and dining items.",
        "Focus on office and study desk items.",
        "Focus on storage and organization items.",
        "Focus on common solid colors and materials.",
        "Focus on textures, sizes, and shapes.",
        "Focus on cleaning and personal care items.",
        "Focus on clothing and accessories.",
        "Focus on outdoor and sports items.",
        "Focus on manipulation verbs: pick, grasp, slide, place, push, pull.",
        "Focus on manipulation verbs: rotate, open, close, flip, tilt, press.",
        "Focus on spatial prepositions and relations.",
        "Focus on produce and food items.",
        "Focus on electronics and gadgets on a desk.",
        "Focus on workshop and tool items.",
        "Focus on toys and children's items.",
    ]

    def __init__(self, llm: InstrFirstTextLLM, template: str, gen_cfg: dict):
        self.llm = llm
        self.template = template
        self.gen_cfg = gen_cfg

    def _vocab_gen_cfg(self) -> dict:
        """扩词专用生成参数：短输出、无思考、贪心解码。"""
        base = dict(self.gen_cfg)
        vocab = dict(self.gen_cfg.get("vocab") or {})
        if "max_new_tokens" not in vocab and "vocab_max_new_tokens" in self.gen_cfg:
            vocab["max_new_tokens"] = self.gen_cfg["vocab_max_new_tokens"]
        base.update(vocab)
        base.setdefault("max_new_tokens", 512)
        base.setdefault("do_sample", False)
        base.setdefault("enable_thinking", False)
        return base

    def _build_prompt(
        self,
        slot_name: str,
        slot_cfg: dict,
        existing: Set[str],
        hint: str,
    ) -> str:
        examples = slot_cfg.get("examples") or slot_cfg.get("fallback") or []
        max_show = int(self.gen_cfg.get("existing_words_in_prompt", 60))
        shown = sorted(existing)
        if len(shown) > max_show:
            shown = shown[:max_show]
            exist_str = ", ".join(shown) + f" ... (+{len(existing) - max_show} more)"
        else:
            exist_str = ", ".join(shown) if shown else "(empty)"
        return self.template.format(
            slot_name=slot_name,
            type=slot_cfg.get("type", ""),
            gen_type=slot_cfg.get("gen_type", slot_cfg.get("type", "")),
            examples=", ".join(str(e) for e in examples[:12]),
            existing=exist_str,
            num_words=self.gen_cfg.get("words_per_slot", 50),
            hint=f"\n{hint}" if hint else "",
        )

    def _infer(self, prompts: List[str]) -> List[str]:
        return self.llm.infer_batch(prompts, self._vocab_gen_cfg())

    @staticmethod
    def _parse(text: str) -> List[str]:
        cleaned = _strip_think(text)
        cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
        try:
            s, e = cleaned.find("{"), cleaned.rfind("}")
            if s != -1 and e > s:
                data = json.loads(cleaned[s : e + 1])
                return [w.strip().lower() for w in data.get("words", []) if w.strip()]
        except json.JSONDecodeError:
            pass
        return []

    def discover_batch(
        self,
        slots: Dict[str, dict],
        existing_by_slot: Dict[str, Set[str]],
        target_per_slot: Optional[Dict[str, int]] = None,
    ) -> Dict[str, List[str]]:
        default_target = int(self.gen_cfg.get("min_new_words_per_slot", 80))
        prompts_per_slot = int(self.gen_cfg.get("prompts_per_slot", 3))
        max_rounds = int(self.gen_cfg.get("max_expand_rounds", 12))

        targets: Dict[str, int] = {}
        for slot_name in slots:
            if target_per_slot and slot_name in target_per_slot:
                targets[slot_name] = max(0, int(target_per_slot[slot_name]))
            else:
                targets[slot_name] = default_target

        new_by_slot: Dict[str, Set[str]] = {s: set() for s in slots}

        from .terminal_ui import VocabExpandBar

        expand_bar = VocabExpandBar(
            max_rounds,
            targets,
            tuple(sorted(slots.keys())),
        )
        for _round in range(max_rounds):
            if all(len(new_by_slot[s]) >= targets[s] for s in slots):
                break

            round_hints = random.sample(
                self.HINTS,
                min(prompts_per_slot, len(self.HINTS)),
            )
            all_prompts: List[str] = []
            prompt_slots: List[str] = []

            for slot_name, slot_cfg in slots.items():
                if len(new_by_slot[slot_name]) >= targets[slot_name]:
                    continue
                for hi in range(prompts_per_slot):
                    hint = round_hints[hi % len(round_hints)]
                    all_prompts.append(
                        self._build_prompt(
                            slot_name,
                            slot_cfg,
                            existing_by_slot.get(slot_name, set()),
                            hint,
                        )
                    )
                    prompt_slots.append(slot_name)

            if not all_prompts:
                break

            expand_bar.update_round(
                _round + 1,
                {s: len(new_by_slot[s]) for s in slots},
                len(all_prompts),
            )

            try:
                outputs = self._infer(all_prompts)
            except Exception as exc:
                logger.error("Vocab inference failed: %s", exc)
                break

            for slot_name, output in zip(prompt_slots, outputs):
                slot_type = _slot_type(slots, slot_name)
                parsed = self._parse(output)
                full_existing = (
                    existing_by_slot.get(slot_name, set()) | new_by_slot[slot_name]
                )
                for w in parsed:
                    if not _valid_word(w, slot_type):
                        continue
                    if slot_type in {"adj", "adjective"} and _is_likely_noun(w):
                        continue
                    if (
                        w in existing_by_slot.get(slot_name, set())
                        or w in new_by_slot[slot_name]
                        or _is_near_duplicate(w, full_existing)
                    ):
                        continue
                    new_by_slot[slot_name].add(w)
                    full_existing.add(w)

        expand_bar.close()

        for slot_name, slot_cfg in slots.items():
            if len(new_by_slot[slot_name]) >= targets[slot_name]:
                continue
            fallback = slot_cfg.get("fallback") or slot_cfg.get("examples") or []
            slot_type = _slot_type(slots, slot_name)
            full_existing = existing_by_slot.get(slot_name, set()) | new_by_slot[slot_name]
            shuffled = list(fallback)
            random.shuffle(shuffled)
            for w in shuffled:
                if len(new_by_slot[slot_name]) >= targets[slot_name]:
                    break
                wl = str(w).strip().lower()
                if (
                    _valid_word(wl, slot_type)
                    and wl not in full_existing
                    and not _is_near_duplicate(wl, full_existing)
                ):
                    new_by_slot[slot_name].add(wl)
                    full_existing.add(wl)

        return {s: list(new_by_slot[s]) for s in slots}


# ── Bimanual: LLM splits task into left / right sub-instructions ──

class BimanualDecomposer:
    """Text LLM: overall task_description → left/right + coordination_type."""

    COOPERATION_HINTS = [
        "Brace/pin the reference object so the other hand can place the target.",
        "Guide or align the target toward the reference — avoid 'hold steady'.",
        "Nudge or shift the reference to make space, then place the target.",
        "Tilt or rotate the reference slightly while the other hand places/slides.",
        "Present or receive: one hand readies the reference, the other inserts/places.",
        "Press or anchor the reference to the desk; the other hand completes placement.",
        "Lift or reposition the reference briefly so the target can be placed beneath/beside.",
        "Clear the area near the reference (pull/push) while the other hand places.",
        "Open, part, or separate the reference while the other hand places inside/beside.",
        "Simultaneous dual verbs: both hands do different actions (not two grasps).",
    ]

    def __init__(self, llm: InstrFirstTextLLM, template: str, gen_cfg: dict):
        self.llm = llm
        self.template = template
        self.gen_cfg = gen_cfg

    def _objects_summary(self, feasible: FeasibleInstruction) -> str:
        parts: List[str] = []
        for obj in feasible.objects:
            label = obj.label()
            if label:
                parts.append(label)
        if feasible.nouns:
            parts.extend(
                n for n in feasible.nouns if n not in parts
            )
        return ", ".join(parts) if parts else "(none listed)"

    def _role_labels(self, feasible: FeasibleInstruction) -> Tuple[str, str, str]:
        target = feasible.objects[0].label() if feasible.objects else "(none)"
        reference = (
            feasible.objects[1].label() if len(feasible.objects) > 1 else "(none)"
        )
        primary_verb = feasible.verbs[0] if feasible.verbs else "manipulate"
        return target, reference, primary_verb

    def _cooperation_hint(
        self,
        feasible: FeasibleInstruction,
        assembled: Optional[Dict],
        attempt: int,
    ) -> str:
        key = (
            f"{feasible.task_description}|"
            f"{assembled.get('template_id') if assembled else ''}"
        )
        idx = (hash(key) + attempt * 7) % len(self.COOPERATION_HINTS)
        return self.COOPERATION_HINTS[idx]

    def _build_prompt(
        self,
        feasible: FeasibleInstruction,
        assembled: Optional[Dict] = None,
        *,
        attempt: int = 0,
    ) -> str:
        assembled = assembled or {}
        slots = assembled.get("slots") or {}
        target, reference, primary_verb = self._role_labels(feasible)
        return self.template.format(
            task_description=feasible.task_description,
            objects_summary=self._objects_summary(feasible),
            template_id=str(assembled.get("template_id") or "unknown"),
            primary_verb=primary_verb,
            target_object=target,
            reference_object=reference,
            preposition=str(slots.get("p0") or "(none)"),
            cooperation_hint=self._cooperation_hint(feasible, assembled, attempt),
        )

    def _infer_one(self, prompt: str) -> str:
        return self.llm.infer_one(prompt, self.gen_cfg)

    @staticmethod
    def _parse(text: str) -> Optional[Dict[str, str]]:
        data = parse_json_from_llm(text)
        if not data:
            return None
        left = str(data.get("left_instruction", "") or "").strip()
        right = str(data.get("right_instruction", "") or "").strip()
        ctype = str(data.get("coordination_type", "") or "").strip().lower()
        if not left or not right:
            return None
        if ctype not in BIMANUAL_COORDINATION_TYPES:
            return None
        return {
            "left_instruction": left,
            "right_instruction": right,
            "coordination_type": ctype,
        }

    def decompose(
        self,
        feasible: FeasibleInstruction,
        assembled: Optional[Dict] = None,
        max_retries: int = 4,
    ) -> FeasibleInstruction:
        for attempt in range(max_retries + 1):
            try:
                raw = self._infer_one(
                    self._build_prompt(feasible, assembled, attempt=attempt)
                )
            except Exception as exc:
                logger.warning("Bimanual LLM failed: %s", exc)
                continue
            parsed = self._parse(raw)
            if not parsed:
                continue
            enriched = FeasibleInstruction(
                action_type=feasible.action_type,
                task_description=feasible.task_description,
                verbs=list(feasible.verbs),
                nouns=list(feasible.nouns),
                adjectives=list(feasible.adjectives),
                objects=list(feasible.objects),
                use_both_hands=True,
                left_instruction=parsed["left_instruction"],
                right_instruction=parsed["right_instruction"],
                coordination_type=parsed["coordination_type"],
            )
            if is_cooperative_bimanual_decomposition(
                enriched,
                reject_steady_hold=(attempt >= 2),
            ):
                return enriched
            logger.debug(
                "Bimanual parse rejected (attempt %d): %s / %s",
                attempt + 1,
                parsed["left_instruction"],
                parsed["right_instruction"],
            )
        logger.warning(
            "Bimanual decompose failed for: %s",
            feasible.task_description,
        )
        return feasible

    def enrich_assembled(self, assembled: Dict) -> Dict:
        if not assembled.get("use_both_hands"):
            return assembled
        feasible = FeasibleInstruction.from_dict(assembled.get("feasible"))
        enriched = self.decompose(feasible, assembled=assembled)
        if enriched.use_both_hands:
            assembled["feasible"] = enriched.to_dict()
        return assembled

    def enrich_batch(self, assembled_list: List[Dict]) -> List[Dict]:
        out: List[Dict] = []
        for item in assembled_list:
            out.append(self.enrich_assembled(item))
        return out


# ── Agent2: slot assembly from DB ──────────────────────────

class SlotInstructionAssembler:
    """Sample low-usage words per slot and fill instruction templates."""

    def __init__(
        self,
        db: SlotDatabase,
        templates: List[dict],
        slots_cfg: Dict[str, dict],
        *,
        adj_skip_probability: float = 0.25,
        object_max_usage: int = 10,
    ):
        self.db = db
        self.templates = [_normalize_template(t) for t in templates]
        self.slots_cfg = slots_cfg
        self.adj_skip_probability = adj_skip_probability
        self.object_max_usage = max(0, int(object_max_usage))
        self._object_usage: Dict[str, int] = {}

    @staticmethod
    def _slot_names(pattern: str) -> List[str]:
        return list(dict.fromkeys(re.findall(r"\{(\w+)\}", pattern)))

    def allows_instruction_duplicates(self) -> bool:
        """
        Only when there is a single template with no {slot} variables —
        intentional production of many identical instructions.
        """
        if len(self.templates) != 1:
            return False
        return not self._slot_names(self.templates[0]["pattern"])

    def _maybe_skip_adj(self, slot_name: str) -> bool:
        slot_type = _slot_type(self.slots_cfg, slot_name)
        if slot_type not in {"adj", "adjective"}:
            return False
        return random.random() < self.adj_skip_probability

    def _sample_slot_word(self, slot_name: str, n: int = 1) -> List[str]:
        slot_type = _slot_type(self.slots_cfg, slot_name)
        if slot_type not in {"noun", "n"} or self.object_max_usage <= 0:
            return self.db.sample_low_freq(slot_name, n)
        max_candidates = max(8, min(128, n * 16))
        candidates = self.db.sample_low_freq(slot_name, max_candidates)
        out: List[str] = []
        for word in candidates:
            key = word.strip().lower()
            if self._object_usage.get(key, 0) >= self.object_max_usage:
                continue
            out.append(word)
            if len(out) >= n:
                break
        return out

    def _object_keys(self, samples: Dict[str, str]) -> List[str]:
        keys: List[str] = []
        for slot, word in samples.items():
            if _slot_type(self.slots_cfg, slot) not in {"noun", "n"}:
                continue
            key = word.strip().lower()
            if key:
                keys.append(key)
        return keys

    def assemble_batch(
        self,
        count: int,
        force_expand_slots: Optional[List[str]] = None,
    ) -> Tuple[List[Dict], List[str]]:
        results: List[Dict] = []
        seen: Set[str] = set()
        exhausted: Set[str] = set()
        force_expand = set(force_expand_slots or [])
        allow_dup = self.allows_instruction_duplicates()
        attempts = 0
        max_attempts = count * 20 if not allow_dup else count + 5

        while len(results) < count and attempts < max_attempts:
            attempts += 1
            tpl = random.choice(self.templates)
            pattern = tpl["pattern"]
            slot_names = self._slot_names(pattern)

            if force_expand & set(slot_names):
                continue

            samples: Dict[str, str] = {}
            slot_exhausted: Set[str] = set()

            for slot in slot_names:
                if slot.startswith("d") and self._maybe_skip_adj(slot):
                    samples[slot] = ""
                    continue
                picked = self._sample_slot_word(slot, 1)
                if not picked:
                    slot_exhausted.add(slot)
                else:
                    samples[slot] = picked[0]

            if slot_exhausted:
                exhausted |= slot_exhausted
                break

            instr = _format_pattern(pattern, samples)
            norm = " ".join(instr.lower().split())
            if not allow_dup:
                if norm in seen or self.db.instruction_exists(instr):
                    continue
                seen.add(norm)
            elif len(results) >= count:
                break

            object_keys = self._object_keys(samples)
            if len(object_keys) != len(set(object_keys)):
                continue
            if (
                self.object_max_usage > 0
                and any(
                    self._object_usage.get(key, 0) + delta
                    > self.object_max_usage
                    for key, delta in Counter(object_keys).items()
                )
            ):
                continue

            if not allow_dup:
                self.db.add_instruction(instr)
            self.db.increment_usage(samples)
            for key in object_keys:
                self._object_usage[key] = self._object_usage.get(key, 0) + 1

            feasible = _assembled_to_feasible(
                instr, samples, tpl["id"], pattern
            )
            results.append({
                "instr": instr,
                "slots": dict(samples),
                "template_id": tpl["id"],
                "use_both_hands": bool(tpl.get("use_both_hands")),
                "feasible": feasible.to_dict(),
            })

        return results, sorted(exhausted)


# ── Main controller ──────────────────────────────────────────

class InstrFirstGenerator:
    """
    Dual-agent instruction-first generator:
    - Agent1 expands per-slot word pools via LLM
    - Agent2 assembles balanced instructions from templates
    """

    def __init__(self, config_path: str, device: Optional[str] = None):
        cfg_path = Path(config_path).resolve()
        with open(cfg_path, encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f)
        self.cfg = full_cfg.get("instr_first", full_cfg)
        self.base = _REPO_ROOT

        log_rel = self.cfg.get("output", {}).get(
            "log_file", "database/wm-h/logs/instr_first.log"
        )
        log_path = self.base / log_rel
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = logging.getLogger("wm_h.instr_first")
        self.log.setLevel(logging.INFO)
        log_abs = str(log_path.resolve())
        if not any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", "") == log_abs
            for h in self.log.handlers
        ):
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setFormatter(
                logging.Formatter("%(asctime)s [instr_first] %(levelname)s %(message)s")
            )
            self.log.addHandler(fh)

        mc = self.cfg.get("model", {})
        if not mc.get("path"):
            raise ValueError("instr_first.model.path is required in config")

        resolved_device = str(device or full_cfg.get("settings", {}).get("device", "cuda"))
        self.keep_model_loaded = bool(mc.get("keep_model_loaded", False))
        self.llm = InstrFirstTextLLM(mc, device=resolved_device)

        batch_cfg = self.cfg.get("batch", {})
        db_path = self.base / self.cfg.get("database", {}).get(
            "base_path", "database/wm-h/instr_first/slots"
        )
        max_usage_global = batch_cfg.get("max_usage_per_word", 50)
        max_usage_per_slot = {
            s: c["max_usage"]
            for s, c in self.cfg.get("slots", {}).items()
            if "max_usage" in c
        }
        self.db = SlotDatabase(
            str(db_path),
            self.cfg.get("slots", {}),
            sample_power=float(batch_cfg.get("sample_power", 1.5)),
            max_usage_global=max_usage_global,
            max_usage_per_slot=max_usage_per_slot,
        )

        min_vocab_boot = int(
            self.cfg.get("batch", {}).get("min_vocab_per_slot", 30)
        )
        for slot_name, slot_cfg in self.cfg.get("slots", {}).items():
            fallback = slot_cfg.get("fallback") or slot_cfg.get("examples") or []
            self.db.seed_if_empty(slot_name, fallback)
            self.db.seed_to_min(slot_name, min_vocab_boot, fallback)

        tpl_rel = self.cfg.get("prompt_template", {}).get("file", "")
        tpl_path = self.base / tpl_rel if tpl_rel else (
            Path(__file__).parent / "prompts" / "slot_expand_template.txt"
        )
        template = tpl_path.read_text(encoding="utf-8")
        self.agent1 = SlotVocabDiscoverer(
            self.llm, template, self.cfg.get("generation", {})
        )
        self.agent2 = SlotInstructionAssembler(
            self.db,
            self.cfg.get("instruction_templates", []),
            self.cfg.get("slots", {}),
            adj_skip_probability=float(batch_cfg.get("adj_skip_probability", 0.25)),
            object_max_usage=int(batch_cfg.get("object_max_usage", 10)),
        )

        bimanual_rel = self.cfg.get("prompt_template", {}).get("bimanual_file", "")
        bimanual_path = (
            self.base / bimanual_rel
            if bimanual_rel
            else Path(__file__).parent / "prompts" / "bimanual_decompose.txt"
        )
        self.bimanual = BimanualDecomposer(
            self.llm,
            bimanual_path.read_text(encoding="utf-8"),
            self.cfg.get("generation", {}),
        )

        out_rel = self.cfg.get("output", {}).get(
            "instructions_file",
            "database/wm-h/instr_first/vocab/generated_instructions.jsonl",
        )
        self.out_file = self.base / out_rel
        self.out_file.parent.mkdir(parents=True, exist_ok=True)

        self.log.debug("Init done. Slot counts: %s", self.db.get_counts())

    def enrich_bimanual_assembled(self, assembled_list: List[Dict]) -> None:
        """LLM split for templates marked use_both_hands."""
        bimanual = [a for a in assembled_list if a.get("use_both_hands")]
        if not bimanual:
            return
        self.log.debug("Bimanual decompose: %d instruction(s)", len(bimanual))
        self.bimanual.enrich_batch(bimanual)

    def _low_vocab_slots(self, slack: int = 0) -> List[str]:
        min_vocab = int(self.cfg.get("batch", {}).get("min_vocab_per_slot", 30))
        counts = self.db.get_counts()
        return [
            s
            for s in self.cfg.get("slots", {})
            if counts.get(s, 0) < min_vocab - slack
        ]

    def _bootstrap_vocab(self, max_rounds: int = 2) -> None:
        """Cold-start: seed fallbacks + short LLM expand; never block assembly forever."""
        min_vocab = int(self.cfg.get("batch", {}).get("min_vocab_per_slot", 30))
        low = self._low_vocab_slots()
        if not low:
            return

        logger.debug(
            "bootstrap · min_vocab=%d · %s",
            min_vocab,
            {s: self.db.get_counts().get(s, 0) for s in low},
        )

        for slot_name in low:
            slot_cfg = self.cfg.get("slots", {}).get(slot_name, {})
            fallback = slot_cfg.get("fallback") or slot_cfg.get("examples") or []
            self.db.seed_to_min(slot_name, min_vocab, fallback)

        for _ in range(max_rounds):
            low = self._low_vocab_slots()
            if not low:
                logger.debug(
                    "bootstrap done · %s", slot_counts(self.db.get_counts())
                )
                return

            before = self.db.get_counts()
            self._expand_slots(low, cold_start=True)
            after = self.db.get_counts()
            if any(after.get(s, 0) > before.get(s, 0) for s in low):
                continue

            self.log.warning(
                "Vocab bootstrap stalled (LLM added 0 words) for %s | %s → proceeding to assemble",
                low,
                {s: after.get(s, 0) for s in low},
            )
            return

        still_low = self._low_vocab_slots(slack=1)
        if still_low:
            self.log.warning(
                "Slots still slightly below min_vocab after bootstrap: %s — assemble anyway",
                {s: self.db.get_counts().get(s, 0) for s in still_low},
            )

    def _should_expand(self) -> bool:
        batch_cfg = self.cfg.get("batch", {})
        if not bool(batch_cfg.get("enable_runtime_expand", False)):
            return False
        min_uc = batch_cfg.get("vocab_expand_min_uc", 5)
        all_reached_threshold = True
        any_at_cap = False

        for slot_name in self.cfg.get("slots", {}):
            min_actual = self.db.min_usage(slot_name)
            if min_actual is None:
                return True
            if not self.db.is_verb_slot(slot_name):
                max_actual = self.db.max_usage(slot_name)
                cap = self.db.effective_max_usage(slot_name)
                if max_actual is not None and max_actual >= cap:
                    any_at_cap = True
            if min_actual < min_uc:
                all_reached_threshold = False

        return any_at_cap or all_reached_threshold

    def release_text_model(self) -> None:
        """Free text LLM weights before loading image-edit model."""
        if self.keep_model_loaded:
            self.log.info("Keeping instr_first text model resident")
            return
        if self.llm is not None:
            self.llm.release()

    def offload_text_model(self) -> None:
        """Move text LLM weights to CPU without destroying them."""
        if self.llm is not None:
            self.llm.offload()

    def onload_text_model(self) -> None:
        """Move text LLM weights back to the configured device."""
        if self.llm is not None:
            self.llm.onload()

    def _save(self, results: List[Dict]) -> None:
        with open(self.out_file, "a", encoding="utf-8") as f:
            for r in results:
                feasible = r.get("feasible") or {}
                row = {
                    "instr": r.get("instr"),
                    "slots": r.get("slots"),
                    "template_id": r.get("template_id"),
                    "task_description": feasible.get("task_description", r.get("instr")),
                    "verbs": feasible.get("verbs", []),
                    "nouns": feasible.get("nouns", []),
                    "adjectives": feasible.get("adjectives", []),
                    "objects": feasible.get("objects", []),
                    "feasible": feasible,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _assemble_all(
        self,
        remaining: int,
        force_expand_slots: List[str],
    ) -> Tuple[List[Dict], List[str]]:
        all_instrs: List[Dict] = []
        all_exhausted: Set[str] = set()
        attempts = 0
        while len(all_instrs) < remaining and attempts < 20:
            attempts += 1
            batch, exhausted = self.agent2.assemble_batch(
                remaining - len(all_instrs), force_expand_slots
            )
            if not batch:
                break
            all_instrs.extend(batch)
            all_exhausted |= set(exhausted)
        return all_instrs, sorted(all_exhausted)

    def _expand_slots(
        self,
        slot_names: List[str],
        *,
        cold_start: bool = False,
    ) -> None:
        import time

        batch_cfg = self.cfg.get("batch", {})
        min_vocab = int(batch_cfg.get("min_vocab_per_slot", 30))
        counts = self.db.get_counts()

        slots = {s: self.cfg["slots"][s] for s in slot_names if s in self.cfg["slots"]}
        if cold_start:
            low = [
                s for s in slots
                if counts.get(s, 0) < min_vocab
            ]
            slots = {s: slots[s] for s in low}
            if not slots:
                self.log.debug(
                    "Cold start: all slots >= %d words, skip LLM expand", min_vocab
                )
                return

        target_per_slot: Optional[Dict[str, int]] = None
        if cold_start:
            target_per_slot = {
                s: max(0, min_vocab - counts.get(s, 0)) for s in slots
            }
            slots = {s: slots[s] for s in slots if target_per_slot[s] > 0}
            if not slots:
                return

        vocab_cfg = self.agent1._vocab_gen_cfg()
        slot_list = ", ".join(slots)
        logger.debug(
            "LLM expand · %s · tokens=%s sample=%s think=%s",
            slot_list,
            vocab_cfg.get("max_new_tokens", 512),
            vocab_cfg.get("do_sample", False),
            vocab_cfg.get("enable_thinking", False),
        )
        if cold_start and target_per_slot:
            logger.debug("targets %s", target_per_slot)

        t0 = time.perf_counter()
        existing = {s: self.db.get_all_words(s) for s in slots}
        new_words = self.agent1.discover_batch(
            slots, existing, target_per_slot=target_per_slot
        )
        elapsed = time.perf_counter() - t0
        added_parts: List[str] = []
        for slot_name, words in new_words.items():
            added = self.db.add_words(slot_name, words)
            if added or words:
                added_parts.append(f"{slot_name}:+{added}")
        counts = self.db.get_counts()
        summary = slot_counts(counts)
        if added_parts:
            logger.debug(
                "expand %.1fs · %s · %s",
                elapsed,
                " ".join(added_parts),
                summary,
            )
        else:
            logger.debug("expand %.1fs · no new words · %s", elapsed, summary)

    def run(self) -> str:
        batch_cfg = self.cfg.get("batch", {})
        target = int(batch_cfg.get("total_instructions", 1000))
        bs2 = int(batch_cfg.get("agent2_batch_size", 50))

        generated = 0
        force_expand_slots: List[str] = []
        zero_rounds = 0
        pool = ThreadPoolExecutor(max_workers=1)

        self.log.debug("Target: %d instructions", target)
        self._bootstrap_vocab(max_rounds=2)

        while generated < target:
            remaining = min(bs2, target - generated)
            should_force = bool(force_expand_slots) or zero_rounds >= 2

            if self._should_expand() or should_force:
                fut = pool.submit(
                    self._expand_slots, list(self.cfg["slots"].keys())
                )
                instrs, exhausted = self._assemble_all(remaining, force_expand_slots)
                fut.result()

                if exhausted:
                    self.log.warning("Slots at usage cap, force expand: %s", exhausted)
                    self._expand_slots(exhausted)
                    more, still = self._assemble_all(remaining, force_expand_slots)
                    instrs.extend(more)
                    for s in still:
                        if s not in force_expand_slots:
                            force_expand_slots.append(s)
                    if not still:
                        force_expand_slots.clear()
                else:
                    force_expand_slots.clear()
                zero_rounds = 0
                self.log.debug(
                    "[expand+assemble] %d/%d | slots: %s | batch: %d",
                    generated, target, self.db.get_counts(), len(instrs),
                )
            else:
                instrs, exhausted = self._assemble_all(remaining, force_expand_slots)
                if exhausted:
                    self.log.warning("Slots exhausted: %s", exhausted)
                    self._expand_slots(exhausted)
                    more, still = self._assemble_all(remaining, force_expand_slots)
                    instrs.extend(more)
                    for s in still:
                        if s not in force_expand_slots:
                            force_expand_slots.append(s)
                    if not still:
                        force_expand_slots.clear()
                    zero_rounds = 0
                else:
                    force_expand_slots.clear()
                    if not instrs:
                        zero_rounds += 1
                        self.log.debug(
                            "[assemble only] 0 assembled (low-freq catch-up), round %d",
                            zero_rounds,
                        )
                    else:
                        zero_rounds = 0
                        self.log.debug("[assemble only] %d assembled", len(instrs))

            if instrs:
                self.enrich_bimanual_assembled(instrs)
                self._save(instrs)
                generated += len(instrs)

        pool.shutdown(wait=True)
        self.log.info("Done: %d instructions → %s", generated, self.out_file)
        self.db.close()
        return str(self.out_file)

    def generate_feasible_batch(self, count: int) -> List[FeasibleInstruction]:
        """One-shot assembly without LLM (for testing or DB-only use)."""
        results, _ = self.agent2.assemble_batch(count)
        return [
            FeasibleInstruction.from_dict(r["feasible"]) for r in results
        ]

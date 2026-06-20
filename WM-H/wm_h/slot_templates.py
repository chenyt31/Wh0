"""Slot-based task_description templates; per-hand text filled separately."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .schema import ObjectRef

_SLOT_PATTERN = re.compile(r"\{(\w+)\}")
_VERB_FROM_PATTERN = re.compile(r"^([a-z]+)\b", re.IGNORECASE)


@dataclass
class InstructionTemplate:
    id: str
    task_description: str
    when_holding: Optional[bool] = None  # None = any


class SlotTemplateEngine:
    """Select templates and fill task_description from LLM-provided slot values."""

    def __init__(self, slots_cfg: Dict[str, Any], templates_cfg: List[Dict]):
        self.slots_cfg = slots_cfg or {}
        self.templates = self._parse_templates(templates_cfg or [])

    @staticmethod
    def _parse_templates(raw: List[Dict]) -> List[InstructionTemplate]:
        templates: List[InstructionTemplate] = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            tid = str(item.get("id") or f"tpl_{i}")
            task = item.get("task_description") or item.get("pattern") or ""
            templates.append(
                InstructionTemplate(
                    id=tid,
                    task_description=str(task),
                    when_holding=item.get("when_holding"),
                )
            )
        return templates

    def slot_names_in_template(self, template: InstructionTemplate) -> List[str]:
        return list(dict.fromkeys(_SLOT_PATTERN.findall(template.task_description)))

    def format_slot_definitions(self, slot_names: List[str]) -> str:
        lines: List[str] = []
        for name in slot_names:
            slot = self.slots_cfg.get(name, {})
            slot_type = slot.get("type", "unknown")
            gen_type = slot.get("gen_type", slot_type)
            examples = slot.get("examples") or slot.get("fallback") or []
            ex_str = ", ".join(str(e) for e in examples[:8])
            lines.append(
                f'- "{name}": type={slot_type}, gen_type="{gen_type}", examples=[{ex_str}]'
            )
        return "\n".join(lines)

    def fallback_slot_values(
        self,
        template: InstructionTemplate,
        target: ObjectRef,
    ) -> Dict[str, str]:
        """Last-resort values when LLM fill fails."""
        values: Dict[str, str] = {}
        for name in self.slot_names_in_template(template):
            slot = self.slots_cfg.get(name, {})
            slot_type = str(slot.get("type", "")).lower()
            if slot_type == "verb":
                pool = slot.get("fallback") or slot.get("examples") or ["pick"]
                values[name] = str(random.choice(pool))
            elif slot_type == "adj":
                if name.endswith("0"):
                    values[name] = target.adjective or str(
                        random.choice(slot.get("fallback") or slot.get("examples") or ["red"])
                    )
                else:
                    values[name] = str(
                        random.choice(slot.get("fallback") or slot.get("examples") or [""])
                    )
            elif slot_type in {"noun", "n"}:
                if name.endswith("0"):
                    values[name] = target.noun or str(
                        random.choice(slot.get("fallback") or slot.get("examples") or ["object"])
                    )
                else:
                    values[name] = str(
                        random.choice(slot.get("fallback") or slot.get("examples") or ["laptop"])
                    )
            elif slot_type in {"prep", "preposition"}:
                pool = slot.get("fallback") or slot.get("examples") or ["beside"]
                values[name] = str(random.choice(pool))
            else:
                pool = slot.get("fallback") or slot.get("examples") or [""]
                values[name] = str(random.choice(pool))
        return values

    @staticmethod
    def fill(text: str, slots: Dict[str, str]) -> str:
        out = text
        for key, val in slots.items():
            out = out.replace("{" + key + "}", val)
        return out.strip()

    def select_template(self, *, hand_holding: bool) -> InstructionTemplate:
        candidates = [
            t
            for t in self.templates
            if t.when_holding is None or t.when_holding == hand_holding
        ]
        if not candidates:
            raise ValueError("No instruction template matches the current scenario")
        return random.choice(candidates)

    @staticmethod
    def extract_verbs(pattern: str, slot_values: Dict[str, str]) -> List[str]:
        if "{v0}" in pattern:
            v = (slot_values.get("v0") or "").strip().lower()
            return [v] if v else []
        match = _VERB_FROM_PATTERN.match(pattern.strip())
        if match:
            return [match.group(1).lower()]
        return []

    def render_task(
        self,
        template: InstructionTemplate,
        slot_values: Dict[str, str],
    ) -> Dict[str, Any]:
        task = self.fill(template.task_description, slot_values)
        verbs = self.extract_verbs(template.task_description, slot_values)
        return {
            "template_id": template.id,
            "task_description": task,
            "slots": dict(slot_values),
            "verb": verbs,
        }

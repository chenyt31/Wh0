"""JSON schema helpers for image-first instruction generation."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from .bbox_coords import BBOX_COORD_SYSTEM, parse_bbox_2d

logger = logging.getLogger(__name__)

_NON_OBJECT_NOUNS = frozenset({
    "hand", "hands", "finger", "fingers", "thumb", "palm", "wrist",
    "arm", "arms", "body", "human", "person", "people", "face",
})


def _parse_str_list(raw: object) -> List[str]:
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [str(x).strip() for x in raw if str(x).strip()]


@dataclass
class ObjectRef:
    noun: str = ""
    adjective: str = ""
    bbox_2d: Optional[List[int]] = None  # Qwen3-VL 0-1000 grid [x_min,y_min,x_max,y_max]
    scene_context: str = ""  # OOI only: spatial/occlusion hint for box image edit

    def label(self) -> str:
        adj = (self.adjective or "").strip()
        noun = (self.noun or "").strip()
        if adj and noun:
            return f"{adj} {noun}"
        return noun or adj

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "noun": self.noun,
            "adjective": self.adjective,
        }
        if self.bbox_2d:
            d["bbox_2d"] = list(self.bbox_2d)
            d["bbox_coord_system"] = BBOX_COORD_SYSTEM
        if self.scene_context:
            d["scene_context"] = self.scene_context
        return d

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> "ObjectRef":
        if not isinstance(data, dict):
            return cls()
        bbox = parse_bbox_2d(data.get("bbox_2d") or data.get("bbox"))
        return cls(
            noun=str(data.get("noun", "") or "").strip(),
            adjective=str(data.get("adjective", "") or "").strip(),
            bbox_2d=bbox,
            scene_context=str(data.get("scene_context", "") or "").strip(),
        )

    def key(self) -> str:
        """Dedup key: noun + adjective (lowercase)."""
        return f"{self.noun.lower()}|{self.adjective.lower()}"

    def noun_key(self) -> str:
        """Normalized noun for catalog overlap checks (ignores adjective)."""
        return normalize_noun(self.noun)


def normalize_noun(noun: str) -> str:
    """Lowercase noun with spaces/underscores collapsed."""
    return re.sub(r"[\s_]+", " ", (noun or "").strip().lower())


BIMANUAL_COORDINATION_TYPES = frozenset({
    "left_then_right",
    "right_then_left",
    "simultaneous",
})


@dataclass
class FeasibleInstruction:
    """One VLM-proposed manipulation instruction for this scene."""

    action_type: str = ""  # single | two_object
    task_description: str = ""
    verbs: List[str] = field(default_factory=list)
    nouns: List[str] = field(default_factory=list)
    adjectives: List[str] = field(default_factory=list)
    objects: List[ObjectRef] = field(default_factory=list)
    use_both_hands: bool = False
    left_instruction: str = ""
    right_instruction: str = ""
    coordination_type: str = ""  # left_then_right | right_then_left | simultaneous

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "action_type": self.action_type,
            "task_description": self.task_description,
            "verbs": list(self.verbs),
            "nouns": list(self.nouns),
            "adjectives": list(self.adjectives),
            "objects": [o.to_dict() for o in self.objects],
        }
        if self.use_both_hands:
            d["use_both_hands"] = True
            d["left_instruction"] = self.left_instruction
            d["right_instruction"] = self.right_instruction
            d["coordination_type"] = self.coordination_type
        return d

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> "FeasibleInstruction":
        if not isinstance(data, dict):
            return cls()
        action_type = str(data.get("action_type", "") or "").strip().lower()
        if action_type not in {"single", "two_object"}:
            # tolerate legacy aliases
            if action_type in {"two-object", "interaction", "dual"}:
                action_type = "two_object"
            elif action_type:
                action_type = "single" if "single" in action_type else action_type
        objs_raw = data.get("objects") or []
        objects = [ObjectRef.from_dict(o) for o in objs_raw if isinstance(o, dict)]
        use_both_hands = bool(data.get("use_both_hands", False))
        return cls(
            action_type=action_type,
            task_description=str(data.get("task_description", "") or "").strip(),
            verbs=[v.lower() for v in _parse_str_list(data.get("verbs"))],
            nouns=_parse_str_list(data.get("nouns")),
            adjectives=_parse_str_list(data.get("adjectives")),
            objects=objects,
            use_both_hands=use_both_hands,
            left_instruction=str(data.get("left_instruction", "") or "").strip(),
            right_instruction=str(data.get("right_instruction", "") or "").strip(),
            coordination_type=str(data.get("coordination_type", "") or "").strip().lower(),
        )


def _dedupe_objects(objects: List[ObjectRef]) -> List[ObjectRef]:
    seen: set = set()
    out: List[ObjectRef] = []
    for obj in objects:
        if not obj.noun:
            continue
        k = obj.key()
        if k in seen:
            continue
        seen.add(k)
        out.append(obj)
    return out


def visible_and_held_nouns(analysis: "ImageAnalysis") -> set:
    """Normalized nouns for objects already present in the image (desk or in-hand)."""
    nouns: set = set()
    for obj in analysis.visible_objects:
        nk = obj.noun_key()
        if nk:
            nouns.add(nk)
    for _, held in analysis.held_objects():
        nk = held.noun_key()
        if nk:
            nouns.add(nk)
    return nouns


def visible_and_held_keys(analysis: "ImageAnalysis") -> set:
    """Exact catalog keys for objects already present in the image."""
    keys: set = set()
    for obj in analysis.visible_objects:
        keys.add(obj.key())
    for _, held in analysis.held_objects():
        keys.add(held.key())
    return keys


def filter_ooi_duplicates(analysis: "ImageAnalysis") -> int:
    """Drop OOI entries that overlap visible/held by noun or exact key."""
    blocked_nouns = visible_and_held_nouns(analysis)
    blocked_keys = visible_and_held_keys(analysis)
    before = len(analysis.suggested_out_of_image_objects)
    kept: List[ObjectRef] = []
    seen_ooi_nouns: set = set()
    for obj in analysis.suggested_out_of_image_objects:
        nk = obj.noun_key()
        if not nk or nk in blocked_nouns or obj.key() in blocked_keys:
            continue
        if nk in seen_ooi_nouns:
            continue
        seen_ooi_nouns.add(nk)
        kept.append(obj)
    analysis.suggested_out_of_image_objects = kept
    removed = before - len(kept)
    if removed:
        blocked_list = ", ".join(sorted(blocked_nouns))
        logger.warning(
            "Removed %d suggested_out_of_image_objects overlapping "
            "visible/held catalog (forbidden nouns: %s)",
            removed,
            blocked_list,
        )
    return removed


def _is_non_object_noun(noun: str) -> bool:
    n = (noun or "").strip().lower()
    return n in _NON_OBJECT_NOUNS or n.endswith(" hand") or n.startswith("hand ")


def _sanitize_object_list(objects: List[ObjectRef]) -> List[ObjectRef]:
    return [o for o in objects if o.noun and not _is_non_object_noun(o.noun)]


def _sanitize_held_object(data: Optional[Dict]) -> Optional[ObjectRef]:
    if not data:
        return None
    obj = ObjectRef.from_dict(data)
    if not obj.noun or _is_non_object_noun(obj.noun):
        return None
    return obj


def _sanitize_hand_holding(
    left: Optional[ObjectRef],
    right: Optional[ObjectRef],
) -> Tuple[bool, bool, Optional[ObjectRef], Optional[ObjectRef]]:
    """Returns hand_holding, both_hands_holding, left, right."""
    if left and right and left.key() == right.key():
        right = None
    hand_holding = left is not None or right is not None
    both_hands_holding = left is not None and right is not None
    return hand_holding, both_hands_holding, left, right


def _bbox_area_qwen(bbox: List[int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _label_scene_tokens(obj: ObjectRef) -> List[str]:
    """Lowercase tokens for matching object mentions in scene_description."""
    tokens: List[str] = []
    label = obj.label().lower().replace("_", " ")
    if label:
        tokens.append(label)
    noun = (obj.noun or "").strip().lower().replace("_", " ")
    if noun and noun not in tokens:
        tokens.append(noun)
    adj = (obj.adjective or "").strip().lower().replace("_", " ")
    if adj and noun:
        tokens.append(f"{adj} {noun}")
    return [t for t in tokens if t]


_SCENE_TOUCH_RE = re.compile(
    r"\b("
    r"steady|stabiliz|brace|bracing|press(?:ed|ing)? against|"
    r"pin(?:ning)?|touch(?:ing|es)?|resting on|flat on|"
    r"on the (?:desk|counter|surface)"
    r")\b",
    re.I,
)
_SCENE_GRASP_RE = re.compile(
    r"\b(grip(?:s|ping)?|grasp(?:s|ing)?|hold(?:s|ing)?)\b",
    re.I,
)
_SCENE_LIFT_RE = re.compile(
    r"\b("
    r"lift(?:ed|ing|s)?|raised|carrying|in the air|"
    r"off the (?:desk|counter|surface|board)"
    r")\b",
    re.I,
)


def _held_likely_resting_on_surface(obj: ObjectRef) -> bool:
    """Large desk-surface items (e.g. cutting boards) are rarely lifted in-hand."""
    if not obj.bbox_2d:
        return False
    x1, y1, x2, y2 = obj.bbox_2d
    width = x2 - x1
    height = y2 - y1
    area = _bbox_area_qwen(obj.bbox_2d)
    if area >= 100_000 and width >= 300:
        return True
    if y2 >= 700 and width >= 250 and height >= 350:
        return True
    return False


def _scene_suggests_touch_not_grasp(
    scene: str,
    obj: ObjectRef,
    side: str,
) -> bool:
    """True when scene_description describes bracing/touching, not a firm lift."""
    if not scene:
        return False
    tokens = _label_scene_tokens(obj)
    if not tokens:
        return False
    side_word = f"{side} hand"
    for sentence in re.split(r"[.!?]\s+", scene):
        sent = sentence.lower()
        idx = sent.find(side_word)
        if idx < 0:
            continue
        clause = sent[idx:]
        if not any(tok in clause for tok in tokens):
            continue
        if _SCENE_LIFT_RE.search(clause):
            return False
        if _SCENE_TOUCH_RE.search(clause):
            return True
        if _SCENE_GRASP_RE.search(clause):
            return False
    return False


def reconcile_held_objects(analysis: "ImageAnalysis") -> "ImageAnalysis":
    """Demote touch/brace-on-surface items mistakenly labeled as held."""
    changed = False
    for side in ("left", "right"):
        attr = f"{side}_held_object"
        held = getattr(analysis, attr)
        if held is None:
            continue
        on_surface = _held_likely_resting_on_surface(held)
        touch_only = _scene_suggests_touch_not_grasp(
            analysis.scene_description,
            held,
            side,
        )
        if not on_surface and not touch_only:
            continue
        reason = "on-surface bbox" if on_surface else "touch/brace in scene_description"
        logger.info(
            "Demoting %s held_object %s to visible_objects (%s, not firm grasp)",
            side,
            held.label(),
            reason,
        )
        analysis.visible_objects.append(held)
        setattr(analysis, attr, None)
        changed = True
    if changed:
        analysis.visible_objects = _dedupe_objects(analysis.visible_objects)
        analysis.hand_holding = bool(
            analysis.left_held_object or analysis.right_held_object
        )
        analysis.both_hands_holding = bool(
            analysis.left_held_object and analysis.right_held_object
        )
    return analysis


def parse_feasible_instructions(raw: object) -> List[FeasibleInstruction]:
    if not isinstance(raw, list):
        return []
    out: List[FeasibleInstruction] = []
    seen: set = set()
    for item in raw:
        instr = FeasibleInstruction.from_dict(item)
        if not instr.task_description:
            continue
        key = instr.task_description.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(instr)
    return out


@dataclass
class ImageAnalysis:
    hand_holding: bool = False
    both_hands_holding: bool = False
    left_held_object: Optional[ObjectRef] = None
    right_held_object: Optional[ObjectRef] = None
    visible_objects: List[ObjectRef] = field(default_factory=list)
    suggested_out_of_image_objects: List[ObjectRef] = field(default_factory=list)
    feasible_instructions: List[FeasibleInstruction] = field(default_factory=list)
    scene_description: str = ""  # Stage-2: VLM narrative of the scene before instructions

    def all_catalog_objects(self) -> List[ObjectRef]:
        """Every object the instruction stage may reference."""
        out: List[ObjectRef] = []
        seen: set = set()
        for obj in (
            list(self.visible_objects)
            + [o for _, o in self.held_objects()]
            + list(self.suggested_out_of_image_objects)
        ):
            if not obj.noun:
                continue
            k = obj.key()
            if k in seen:
                continue
            seen.add(k)
            out.append(obj)
        return out

    def object_by_key(self, key: str) -> Optional[ObjectRef]:
        for obj in self.all_catalog_objects():
            if obj.key() == key:
                return obj
        return None

    def is_in_image(self, obj: ObjectRef) -> bool:
        """True when the object is already visible on the desk or held in hand."""
        key = obj.key()
        if any(o.key() == key for o in self.visible_objects):
            return True
        for _, held in self.held_objects():
            if held.key() == key:
                return True
        nk = obj.noun_key()
        if nk:
            if any(o.noun_key() == nk for o in self.visible_objects):
                return True
            for _, held in self.held_objects():
                if held.noun_key() == nk:
                    return True
        return False

    def is_out_of_image(self, obj: ObjectRef) -> bool:
        if self.is_in_image(obj):
            return False
        nk = obj.noun_key()
        for o in self.suggested_out_of_image_objects:
            if o.key() == obj.key():
                return True
            if nk and o.noun_key() == nk:
                return True
        return False

    def held_objects(self) -> List[Tuple[str, ObjectRef]]:
        """(hand_side, object) for each hand holding an item — left then right."""
        out: List[Tuple[str, ObjectRef]] = []
        if self.left_held_object:
            out.append(("left", self.left_held_object))
        if self.right_held_object:
            out.append(("right", self.right_held_object))
        return out

    @property
    def held_object(self) -> Optional[ObjectRef]:
        if self.right_held_object:
            return self.right_held_object
        return self.left_held_object

    def hand_for_held(self, obj: ObjectRef) -> Optional[str]:
        if self.left_held_object and self.left_held_object.key() == obj.key():
            return "left"
        if self.right_held_object and self.right_held_object.key() == obj.key():
            return "right"
        return None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "hand_holding": self.hand_holding,
            "both_hands_holding": self.both_hands_holding,
            "left_held_object": (
                self.left_held_object.to_dict() if self.left_held_object else None
            ),
            "right_held_object": (
                self.right_held_object.to_dict() if self.right_held_object else None
            ),
            "visible_objects": [o.to_dict() for o in self.visible_objects],
            "suggested_out_of_image_objects": [
                o.to_dict() for o in self.suggested_out_of_image_objects
            ],
            "feasible_instructions": [
                i.to_dict() for i in self.feasible_instructions
            ],
            "scene_description": self.scene_description,
            "bbox_coord_system": BBOX_COORD_SYSTEM,
        }
        if self.held_object and not self.both_hands_holding:
            d["held_object"] = self.held_object.to_dict()
        else:
            d["held_object"] = None
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> "ImageAnalysis":
        left = _sanitize_held_object(data.get("left_held_object"))
        right = _sanitize_held_object(data.get("right_held_object"))
        legacy = _sanitize_held_object(data.get("held_object"))
        if legacy and not left and not right:
            right = legacy
        elif legacy and not right:
            right = legacy
        hand_holding, both_hands_holding, left, right = _sanitize_hand_holding(
            left, right
        )

        visible_raw = list(data.get("visible_objects") or [])
        visible_raw.extend(data.get("operable_objects") or [])
        visible = _sanitize_object_list(
            _dedupe_objects([ObjectRef.from_dict(o) for o in visible_raw])
        )
        out_of_image = _sanitize_object_list(
            _dedupe_objects(
                [
                    ObjectRef.from_dict(o)
                    for o in (data.get("suggested_out_of_image_objects") or [])
                ]
            )
        )
        feasible = parse_feasible_instructions(data.get("feasible_instructions"))
        analysis = cls(
            hand_holding=hand_holding,
            both_hands_holding=both_hands_holding,
            left_held_object=left,
            right_held_object=right,
            visible_objects=visible,
            suggested_out_of_image_objects=out_of_image,
            feasible_instructions=feasible,
            scene_description=str(data.get("scene_description", "") or "").strip(),
        )
        filter_ooi_duplicates(analysis)
        reconcile_held_objects(analysis)
        return analysis


@dataclass
class InstructionRecord:
    """One selected instruction tied to an image."""

    task_description: str
    action_type: str
    verbs: List[str]
    nouns: List[str]
    adjectives: List[str]
    objects: List[ObjectRef]
    left_instruction: str
    right_instruction: str
    use_both_hands: bool
    target_in_image: bool
    target_object: ObjectRef
    reference_object: Optional[ObjectRef] = None
    target_hand: str = ""
    coordination_type: str = ""
    edited_image_path: str = ""
    edit_box: Optional[List[int]] = None
    edit_bbox_2d: Optional[List[int]] = None
    edit_boxes_px: Optional[List[List[int]]] = None
    edit_bboxes_2d: Optional[List[List[int]]] = None
    ooi_object_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["objects"] = [o.to_dict() for o in self.objects]
        d["target_object"] = self.target_object.to_dict()
        d["reference_object"] = (
            self.reference_object.to_dict() if self.reference_object else None
        )
        d.pop("edit_box", None)
        d.pop("edit_boxes_px", None)
        d.pop("edit_bboxes_2d", None)
        d["instruction"] = format_instruction(
            self.left_instruction, self.right_instruction
        )
        if self.edit_box:
            d["edit_box_px"] = list(self.edit_box)
            d["edit_box_coord_system"] = "pixel"
        if self.edit_bbox_2d:
            d["edit_bbox_2d"] = list(self.edit_bbox_2d)
            d["edit_bbox_coord_system"] = BBOX_COORD_SYSTEM
        if self.edit_boxes_px:
            d["edit_boxes_px"] = [list(b) for b in self.edit_boxes_px]
            d["edit_boxes_coord_system"] = "pixel"
        if self.edit_bboxes_2d:
            d["edit_bboxes_2d"] = [list(b) for b in self.edit_bboxes_2d]
            d["edit_bboxes_coord_system"] = BBOX_COORD_SYSTEM
        if self.ooi_object_count:
            d["ooi_object_count"] = self.ooi_object_count
        return d


def format_instruction(left: str, right: str) -> str:
    """Canonical per-hand instruction string."""
    left = (left or "").strip()
    right = (right or "").strip()
    if left.lower() in {"none", "none."}:
        left = "None"
    else:
        left = left.rstrip(".")
    if right.lower() in {"none", "none."}:
        right = "None"
    else:
        right = right.rstrip(".")
    return f"Left hand: {left}. Right hand: {right}."


def parse_json_from_llm(text: str) -> Optional[Dict]:
    """Extract a JSON object from VL/LLM output."""
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return None

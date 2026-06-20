"""Select and resolve VLM-proposed feasible instructions."""

from __future__ import annotations

import json
import logging
import math
import random
import re
from dataclasses import replace
from typing import Any, Dict, List, Optional, Set, Tuple

from .schema import (
    BIMANUAL_COORDINATION_TYPES,
    FeasibleInstruction,
    ImageAnalysis,
    InstructionRecord,
    ObjectRef,
    _label_scene_tokens,
)

logger = logging.getLogger(__name__)


def _bbox_area_qwen(bbox: List[int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def build_held_object_focus_hint(analysis: ImageAnalysis) -> str:
    """Prompt steer when hand(s) already hold item(s): most instructions must use them."""
    if not analysis.hand_holding:
        return ""
    held_parts = [
        f"{hand} hand holds {obj.label()}"
        for hand, obj in analysis.held_objects()
    ]
    if not held_parts:
        return ""
    held_summary = "; ".join(held_parts)
    first_held = analysis.held_objects()[0][1].label()
    return (
        f"=== CRITICAL: in-hand objects ({held_summary}) ===\n"
        f"Touching/bracing/stabilizing an object on the desk does NOT count as holding — "
        f"only items firmly grasped count.\n"
        f"When hand_holding is true, the held item MUST be the PRIMARY target:\n"
        f"- objects[0], nouns[0], adjectives[0] MUST be the held item.\n"
        f"- task_description MUST name the held item FIRST (before any desk/reference object).\n"
        f"- WRONG: \"place the black mouse beside the {first_held}\" (held item is passive second noun)\n"
        f"- RIGHT: \"place the {first_held} beside the black mouse\" (held item is active first noun)\n"
        f"- Target ≥70% of non-bimanual instructions with held item as objects[0].\n"
        f"- SEQUENCE while holding (in this order in feasible_instructions):\n"
        f"  1) ONE in-hand USE first — ONLY the gripped item can be \"used\" (it is the tool).\n"
        f"     MUST say what the use accomplishes: \"use the {first_held} to slice/cut/stir ... <target>\".\n"
        f"     RIGHT (food in catalog ON board): \"use the {first_held} to slice the <food> on the <board>\".\n"
        f"     RIGHT (no food on board): \"use the {first_held} to scrape the cutting board\".\n"
        f"     WRONG: \"slice the food\" when no food object is in visible_objects on the board.\n"
        f"     WRONG: \"use the {first_held} on the cutting board\" (no purpose).\n"
        f"     WRONG: \"use the food on the board\" (food is not in hand — cannot use).\n"
        f"  2) Then ONE OR TWO put-down options with DIFFERENT verbs "
        f"(put/set/lay/rest — at most one \"place\").\n"
        f"  3) Do NOT lead with put-down while still gripping — use first, then release.\n"
        f"- Put-down wording must imply finger release at the destination.\n"
        f"- At most ONE in-hand operation before put-down; then hands must release the item.\n"
        f"- NOT pick/grasp/lift (already in hand). NO pure desk-only tasks while holding.\n"
    )


def build_bimanual_side_object_hint(
    analysis: ImageAnalysis,
    rng: random.Random,
) -> str:
    """One-line prompt steer: bimanual task on a smaller peripheral item, not the largest."""
    objects = [o for o in analysis.all_catalog_objects() if o.bbox_2d]
    if len(objects) < 2:
        return (
            "For the ONE bimanual instruction, target a small peripheral desk item "
            "(pen, mouse, cup, etc.) — not the largest central object."
        )

    ranked = sorted(objects, key=lambda o: _bbox_area_qwen(o.bbox_2d), reverse=True)
    largest = ranked[0]
    side_pool = ranked[max(1, len(ranked) // 2):]
    if not side_pool:
        side_pool = ranked[1:]
    target = rng.choice(side_pool)
    return (
        f"For the ONE bimanual instruction, center cooperation on the {target.label()} "
        f"(a smaller side item) — not on the largest object ({largest.label()})."
    )


def _match_catalog_object(
    ref: ObjectRef,
    analysis: ImageAnalysis,
) -> Optional[ObjectRef]:
    """Match instruction ref to catalog; fuzzy noun match for OOI when adjective differs."""
    if not ref.noun:
        return None
    catalog = analysis.object_by_key(ref.key())
    if catalog:
        return catalog

    ref_noun = ref.noun.lower().strip()
    ref_adj = (ref.adjective or "").lower().strip()
    noun_matches = [
        o
        for o in analysis.all_catalog_objects()
        if o.noun.lower().strip() == ref_noun
    ]
    if ref_adj:
        adj_matches = [
            o for o in noun_matches if (o.adjective or "").lower().strip() == ref_adj
        ]
        if len(adj_matches) == 1:
            return adj_matches[0]
        if adj_matches:
            in_image = [o for o in adj_matches if analysis.is_in_image(o)]
            return in_image[0] if in_image else adj_matches[0]
    if len(noun_matches) == 1:
        return noun_matches[0]
    if noun_matches:
        in_image = [o for o in noun_matches if analysis.is_in_image(o)]
        return in_image[0] if in_image else noun_matches[0]
    return None


def resolve_instruction_objects(
    instruction: FeasibleInstruction,
    analysis: ImageAnalysis,
) -> List[ObjectRef]:
    """Match instruction object refs to catalog entries (with bbox/scene_context)."""
    resolved: List[ObjectRef] = []
    for ref in instruction.objects:
        if not ref.noun:
            continue
        catalog = _match_catalog_object(ref, analysis)
        if catalog:
            resolved.append(catalog)
        else:
            resolved.append(ref)
    return resolved


def _held_keys(analysis: ImageAnalysis) -> Set[str]:
    return {obj.key() for _, obj in analysis.held_objects()}


def ooi_objects_in_instruction(
    instruction: FeasibleInstruction,
    analysis: ImageAnalysis,
) -> List[ObjectRef]:
    """OOI catalog objects referenced by this instruction (deduped, stable order).

    Visible/held catalog items are never returned — they are already in the image
    and must not be box-edited, including in two_object instructions.
    """
    seen: Set[str] = set()
    seen_nouns: Set[str] = set()
    out: List[ObjectRef] = []
    for obj in resolve_instruction_objects(instruction, analysis):
        if analysis.is_in_image(obj):
            continue
        if not analysis.is_out_of_image(obj):
            continue
        key = obj.key()
        nk = obj.noun_key()
        if key in seen or (nk and nk in seen_nouns):
            continue
        seen.add(key)
        if nk:
            seen_nouns.add(nk)
        out.append(obj)
    return out


def instruction_needs_ooi_edit(
    instruction: FeasibleInstruction,
    analysis: ImageAnalysis,
) -> Tuple[bool, Optional[ObjectRef]]:
    """True when any involved object is from suggested_out_of_image_objects."""
    ooi = ooi_objects_in_instruction(instruction, analysis)
    if not ooi:
        return False, None
    return True, ooi[0]


def _resolved_held_objects(
    instruction: FeasibleInstruction,
    analysis: ImageAnalysis,
) -> List[ObjectRef]:
    held = _held_keys(analysis)
    if not held:
        return []
    return [
        obj
        for obj in resolve_instruction_objects(instruction, analysis)
        if obj.key() in held
    ]


def instruction_involves_held(
    instruction: FeasibleInstruction,
    analysis: ImageAnalysis,
) -> bool:
    return bool(_resolved_held_objects(instruction, analysis))


def instruction_primary_is_held(
    instruction: FeasibleInstruction,
    analysis: ImageAnalysis,
) -> bool:
    resolved = resolve_instruction_objects(instruction, analysis)
    if not resolved:
        return False
    held = _held_keys(analysis)
    return resolved[0].key() in held


def _task_indicates_put_down(task: str) -> bool:
    text = (task or "").strip().lower()
    if not text:
        return False
    return any(
        phrase in text
        for phrase in (
            " on the desk",
            " onto the desk",
            " on the counter",
            " on the table",
            " down on",
            " put down",
            " set down",
            " lay down",
        )
    )


def instruction_puts_down(instruction: FeasibleInstruction) -> bool:
    verbs = {v.lower().strip() for v in instruction.verbs if v}
    if verbs & _PUT_DOWN_VERBS:
        return True
    return _task_indicates_put_down(instruction.task_description)


_USE_ACTION_RE = re.compile(
    r"\buse\s+the\s+.+\s+to\s+"
    r"(?:slice|cut|chop|dice|mince|trim|peel|scoop|stir|mix|spread|wipe|clean|scrape|"
    r"press|poke|open|close|loosen|tighten|stir|divide|portion|carve|shave)\b",
    re.I,
)
_USE_TO_THE_RE = re.compile(
    r"\buse\s+the\s+.+\s+to\s+\w+(?:\s+\w+)?\s+the\s+",
    re.I,
)
_BARE_USE_ON_RE = re.compile(r"\buse\s+the\s+.+\s+on\s+the\s+", re.I)
_FOOD_NOUNS = frozenset({
    "food", "vegetable", "fruit", "cheese", "bread", "meat", "substance",
    "ingredient", "onion", "tomato", "carrot", "herb",
})
_SURFACE_NOUNS = frozenset({
    "cutting_board", "board", "plate", "bowl", "dish", "tray", "mat",
})
_SLICE_FOOD_TEXT_RE = re.compile(
    r"\b(?:slice|chop|dice|cut|carve|mince)\b.+\b(?:food|substance|ingredient|vegetable|fruit|meat|cheese|bread)\b",
    re.I,
)
_GENERIC_FOOD_RE = re.compile(r"\b(?:the\s+)?food\b", re.I)


def _is_food_like(obj: ObjectRef) -> bool:
    noun = (obj.noun or "").strip().lower()
    adj = (obj.adjective or "").strip().lower()
    if noun in _FOOD_NOUNS:
        return True
    return any(tok in adj for tok in ("food", "substance", "ingredient", "sliceable"))


def _is_work_surface(obj: ObjectRef) -> bool:
    return (obj.noun or "").strip().lower() in _SURFACE_NOUNS


def _food_objects_in_catalog(analysis: ImageAnalysis) -> List[ObjectRef]:
    seen: Set[str] = set()
    out: List[ObjectRef] = []
    for o in analysis.all_catalog_objects():
        if not _is_food_like(o):
            continue
        key = o.key()
        if key in seen:
            continue
        seen.add(key)
        out.append(o)
    return out


def _work_surfaces_in_catalog(analysis: ImageAnalysis) -> List[ObjectRef]:
    return [o for o in analysis.visible_objects if _is_work_surface(o)]


def food_on_surface(
    analysis: ImageAnalysis,
    surface: ObjectRef,
) -> List[ObjectRef]:
    """Food-like catalog items whose bbox lies on the given surface."""
    if not surface.bbox_2d:
        return []
    on_surface: List[ObjectRef] = []
    for food in _food_objects_in_catalog(analysis):
        if food.bbox_2d and _object_resting_on(food.bbox_2d, surface.bbox_2d):
            on_surface.append(food)
    return on_surface


def catalog_food_on_board(analysis: ImageAnalysis) -> Optional[Tuple[ObjectRef, ObjectRef]]:
    """(food, board) when a catalog food item is visibly on a work surface."""
    for surface in _work_surfaces_in_catalog(analysis):
        foods = food_on_surface(analysis, surface)
        if foods:
            return foods[0], surface
    return None


def instruction_uses_use_verb(instruction: FeasibleInstruction) -> bool:
    verbs = {v.lower().strip() for v in instruction.verbs if v}
    if "use" in verbs:
        return True
    return " use " in (instruction.task_description or "").lower()


def instruction_use_has_explicit_purpose(task: str) -> bool:
    """Reject bare 'use X on Y' — must state what the use accomplishes."""
    text = (task or "").strip()
    if not text:
        return False
    if _USE_ACTION_RE.search(text):
        return True
    if _USE_TO_THE_RE.search(text):
        return True
    if _BARE_USE_ON_RE.search(text) and " to " not in text.lower():
        return False
    return " to " in text.lower() and text.lower().startswith("use ")


def instruction_references_absent_food(
    instruction: FeasibleInstruction,
    analysis: ImageAnalysis,
) -> bool:
    """True when task slices/chops food that is not in the object catalog on the surface."""
    task = instruction.task_description or ""
    task_l = task.lower()
    if not re.search(r"\b(?:slice|chop|dice|cut|carve|mince)\b", task_l):
        return False

    catalog_foods = _food_objects_in_catalog(analysis)
    if _GENERIC_FOOD_RE.search(task) and not catalog_foods:
        return True
    if _SLICE_FOOD_TEXT_RE.search(task) and not catalog_foods:
        return True

    resolved = resolve_instruction_objects(instruction, analysis)
    food_refs = [o for o in resolved if _is_food_like(o)]
    if food_refs and not catalog_foods:
        return True

    catalog_keys = {o.key() for o in catalog_foods}
    for food in food_refs:
        if food.key() not in catalog_keys:
            return True

    surfaces = [o for o in resolved if _is_work_surface(o)]
    if not surfaces:
        surfaces = _work_surfaces_in_catalog(analysis)
    for surface in surfaces:
        on_board = {f.key() for f in food_on_surface(analysis, surface)}
        for food in food_refs:
            if food.key() not in on_board:
                return True
    return False


def is_valid_use_instruction(
    instruction: FeasibleInstruction,
    analysis: ImageAnalysis,
) -> bool:
    """use = only while gripping; held item is tool; task must say to <action> <target>."""
    if not instruction_uses_use_verb(instruction):
        return True
    if not analysis.hand_holding:
        return False
    if not instruction_primary_is_held(instruction, analysis):
        return False
    if not instruction_use_has_explicit_purpose(instruction.task_description):
        return False
    if instruction_references_absent_food(instruction, analysis):
        return False
    return True


def _pick_work_surface_ref(
    analysis: ImageAnalysis,
    ref: Optional[ObjectRef],
) -> Optional[ObjectRef]:
    if ref and _is_work_surface(ref):
        return ref
    boards = _work_surfaces_in_catalog(analysis)
    return boards[0] if boards else ref


def _knife_use_alternatives(
    held: ObjectRef,
    surface: ObjectRef,
    analysis: ImageAnalysis,
) -> List[str]:
    """Grounded use tasks when knife + board but no food on board."""
    label = held.label()
    surf = surface.label()
    paired = catalog_food_on_board(analysis)
    if paired:
        food, board = paired
        return [f"use the {label} to slice the {food.label()} on the {board.label()}"]

    return [
        f"use the {label} to scrape the {surf}",
        f"use the {label} to wipe the {surf}",
        f"use the {label} to press along the {surf}",
    ]


def _build_use_task(
    held: ObjectRef,
    ref: Optional[ObjectRef],
    analysis: ImageAnalysis,
    *,
    alt_index: int = 0,
) -> str:
    """Concrete use instruction grounded in catalog objects visible in the scene."""
    label = held.label()
    noun = (held.noun or "").lower()
    surface = _pick_work_surface_ref(analysis, ref)
    ref_label = surface.label() if surface else "the desk"

    if "knife" in noun:
        alts = _knife_use_alternatives(held, surface, analysis) if surface else [
            f"use the {label} to scrape the counter"
        ]
        return alts[alt_index % len(alts)]

    if noun in {"spoon", "ladle", "spatula"}:
        bowl = next(
            (o for o in analysis.visible_objects if o.noun in {"bowl", "dish", "cup", "pot"}),
            surface,
        )
        if bowl:
            contents = food_on_surface(analysis, bowl)
            if contents:
                return f"use the {label} to stir the {contents[0].label()} in the {bowl.label()}"
            return f"use the {label} to stir inside the {bowl.label()}"
        return f"use the {label} to stir in the bowl"

    if noun in {"screwdriver", "wrench"}:
        if surface:
            return f"use the {label} to loosen the screw on the {ref_label}"
        return f"use the {label} to tighten the fastener"

    if surface:
        return f"use the {label} to press along the {ref_label}"
    return f"use the {label} to manipulate the item on the desk"


def instruction_is_held_operation(
    instruction: FeasibleInstruction,
    analysis: ImageAnalysis,
) -> bool:
    """Held-item action that does not release the object (the one allowed pre-put-down step)."""
    if not analysis.hand_holding:
        return False
    if not instruction_primary_is_held(instruction, analysis):
        return False
    if instruction_uses_use_verb(instruction) and not is_valid_use_instruction(
        instruction, analysis
    ):
        return False
    return not instruction_puts_down(instruction)


def instruction_uses_grasp_on_held(
    instruction: FeasibleInstruction,
    analysis: ImageAnalysis,
) -> bool:
    """True when instruction tries to pick/grasp an item already in hand."""
    if not analysis.hand_holding:
        return False
    verbs = {v.lower().strip() for v in instruction.verbs if v}
    if not verbs & _GRASP_PICK_VERBS:
        return False
    return instruction_primary_is_held(instruction, analysis)


def _rewrite_held_instruction_verb(
    instruction: FeasibleInstruction,
    *,
    held: ObjectRef,
    new_verb: str,
    ref: Optional[ObjectRef],
    analysis: ImageAnalysis,
) -> FeasibleInstruction:
    """Rewrite a held-primary instruction to use a different manipulation verb."""
    label = held.label()
    if ref is not None and new_verb in _HELD_TWO_OBJECT_VERBS:
        ref_label = ref.label()
        if new_verb == "use":
            task = _build_use_task(held, ref, analysis)
        elif new_verb in ("place", "put", "set"):
            task = f"{new_verb} the {label} beside the {ref_label}"
        elif new_verb == "slide":
            task = f"slide the {label} toward the {ref_label}"
        else:
            task = f"{new_verb} the {label} next to the {ref_label}"
        return replace(
            instruction,
            action_type="two_object",
            task_description=task,
            verbs=[new_verb],
            nouns=[held.noun, ref.noun],
            adjectives=[held.adjective or "", ref.adjective or ""],
            objects=[
                ObjectRef(noun=held.noun, adjective=held.adjective, bbox_2d=held.bbox_2d),
                ObjectRef(noun=ref.noun, adjective=ref.adjective, bbox_2d=ref.bbox_2d),
            ],
        )

    if new_verb in ("place", "put"):
        task = f"{new_verb} the {label} on the desk"
    elif new_verb == "slide":
        task = f"slide the {label} across the desk"
    else:
        task = f"{new_verb} the {label}"
    return replace(
        instruction,
        action_type="single",
        task_description=task,
        verbs=[new_verb],
        nouns=[held.noun],
        adjectives=[held.adjective or ""],
        objects=[
            ObjectRef(noun=held.noun, adjective=held.adjective, bbox_2d=held.bbox_2d)
        ],
    )


def diversify_held_instruction_verbs(
    instructions: List[FeasibleInstruction],
    analysis: ImageAnalysis,
    *,
    seed: int = 0,
) -> List[FeasibleInstruction]:
    """Cap overused verbs (especially rotate/place) on held-object instructions."""
    if not analysis.hand_holding:
        return instructions

    rng = random.Random(seed)
    desk_refs = [o for o in analysis.visible_objects if o.noun]
    verb_count_by_held: Dict[str, Dict[str, int]] = {}
    seen_tasks: Set[str] = set()
    diversified: List[FeasibleInstruction] = []

    for instr in instructions:
        if is_valid_bimanual_instruction(instr):
            key = instr.task_description.lower()
            if key not in seen_tasks:
                seen_tasks.add(key)
                diversified.append(instr)
            continue

        if not instruction_primary_is_held(instr, analysis):
            key = instr.task_description.lower()
            if key not in seen_tasks:
                seen_tasks.add(key)
                diversified.append(instr)
            continue

        resolved = resolve_instruction_objects(instr, analysis)
        if not resolved:
            continue
        held = resolved[0]
        held_key = held.key()
        main_verb = (instr.verbs[0] if instr.verbs else "").lower().strip()
        counts = verb_count_by_held.setdefault(held_key, {})

        needs_rewrite = False
        if instruction_is_held_operation(instr, analysis):
            op_count = counts.get("__held_op__", 0) + 1
            counts["__held_op__"] = op_count
            if op_count > _MAX_HELD_OPERATION_PER_HELD:
                needs_rewrite = True
        elif instruction_puts_down(instr) and main_verb:
            counts[main_verb] = counts.get(main_verb, 0) + 1
            if counts[main_verb] > _MAX_SAME_VERB_PER_HELD:
                needs_rewrite = True

        candidate = instr
        if needs_rewrite:
            ref = rng.choice(desk_refs) if desk_refs else None
            if instruction_puts_down(instr):
                new_verb = _pick_alternate_put_down_verb(counts, rng)
            else:
                op_pool = [v for v in _HELD_OPERATION_VERBS if counts.get(v, 0) < 1]
                new_verb = rng.choice(op_pool or list(_HELD_OPERATION_VERBS))
            candidate = _rewrite_held_instruction_verb(
                instr,
                held=held,
                new_verb=new_verb,
                ref=ref,
                analysis=analysis,
            )
            counts[new_verb] = counts.get(new_verb, 0) + 1

        key = candidate.task_description.lower()
        if key in seen_tasks:
            continue
        seen_tasks.add(key)
        diversified.append(candidate)

    return diversified


def synthesize_held_fallback_instructions(
    analysis: ImageAnalysis,
    *,
    seed: int = 0,
) -> List[FeasibleInstruction]:
    """Template held-object instructions when VL output ignores in-hand constraint."""
    fallbacks: List[FeasibleInstruction] = []
    desk_refs = [o for o in analysis.visible_objects if o.noun]
    rng = random.Random(seed)
    for hand_idx, (_, held) in enumerate(analysis.held_objects()):
        adj = held.adjective or ""
        noun = held.noun or "object"
        held_ref = ObjectRef(noun=noun, adjective=adj, bbox_2d=held.bbox_2d)
        ref = desk_refs[(hand_idx + seed) % len(desk_refs)] if desk_refs else None

        use_ref = ref
        if held.noun == "knife":
            for preferred in ("cutting_board", "board", "plate", "dish", "tray"):
                match = next(
                    (d for d in desk_refs if d.noun == preferred),
                    None,
                )
                if match is not None:
                    use_ref = match
                    break
        put_a = _PUT_DOWN_VERB_POOL[(hand_idx + seed) % len(_PUT_DOWN_VERB_POOL)]
        put_b = _PUT_DOWN_VERB_POOL[(hand_idx + seed + 2) % len(_PUT_DOWN_VERB_POOL)]
        if put_b == put_a:
            put_b = _PUT_DOWN_VERB_POOL[(hand_idx + seed + 3) % len(_PUT_DOWN_VERB_POOL)]
        plans: List[tuple] = [("use", use_ref, 0), (put_a, None, 0), (put_b, ref, 0)]
        op_verb = _HELD_OPERATION_VERBS[
            (hand_idx + seed + 1) % len(_HELD_OPERATION_VERBS)
        ]
        if op_verb != "use" and use_ref is not None:
            plans.append((op_verb, use_ref, 0))
        for plan in plans:
            if len(plan) == 3:
                verb, planned_ref, use_alt = plan
            else:
                verb, planned_ref = plan
                use_alt = 0
            template = FeasibleInstruction(
                action_type="two_object" if planned_ref is not None else "single",
                task_description="",
                verbs=[verb],
                nouns=[noun],
                adjectives=[adj],
                objects=[held_ref],
            )
            if verb == "use":
                task = _build_use_task(held, planned_ref, analysis, alt_index=use_alt)
                objs = [held_ref]
                nouns = [held.noun]
                adjs = [adj]
                if planned_ref:
                    objs.append(
                        ObjectRef(
                            noun=planned_ref.noun,
                            adjective=planned_ref.adjective,
                            bbox_2d=planned_ref.bbox_2d,
                        )
                    )
                    nouns.append(planned_ref.noun)
                    adjs.append(planned_ref.adjective or "")
                fallbacks.append(
                    replace(
                        template,
                        action_type="two_object" if len(objs) > 1 else "single",
                        task_description=task,
                        verbs=["use"],
                        nouns=nouns,
                        adjectives=adjs,
                        objects=objs,
                    )
                )
            else:
                fallbacks.append(
                    _rewrite_held_instruction_verb(
                        template,
                        held=held,
                        new_verb=verb,
                        ref=planned_ref,
                        analysis=analysis,
                    )
                )
    return fallbacks


def _phrase_for_object(obj: ObjectRef) -> str:
    adj = (obj.adjective or "").strip()
    noun = (obj.noun or "").strip()
    if adj and noun:
        return f"the {adj} {noun}"
    if noun:
        return f"the {noun}"
    return ""


def _swap_object_phrases_in_text(text: str, obj_a: ObjectRef, obj_b: ObjectRef) -> str:
    """Swap 'the adj noun' phrases in task_description (case-insensitive)."""
    phrase_a = _phrase_for_object(obj_a)
    phrase_b = _phrase_for_object(obj_b)
    if not phrase_a or not phrase_b:
        return text
    if phrase_a.lower() == phrase_b.lower():
        return text
    pat_a = re.compile(re.escape(phrase_a), re.IGNORECASE)
    pat_b = re.compile(re.escape(phrase_b), re.IGNORECASE)
    if not pat_a.search(text) or not pat_b.search(text):
        return text
    token = "__OBJ_SWAP__"
    swapped = pat_a.sub(token, text, count=1)
    swapped = pat_b.sub(phrase_a, swapped, count=1)
    return swapped.replace(token, phrase_b, 1)


def promote_held_to_primary(
    instruction: FeasibleInstruction,
    analysis: ImageAnalysis,
) -> Optional[FeasibleInstruction]:
    """
    When hand_holding, ensure held item is objects[0] and first noun in task_description.
    Returns None if held appears only as a passive second object and cannot be promoted.
    """
    if not analysis.hand_holding or is_valid_bimanual_instruction(instruction):
        return instruction
    if instruction_primary_is_held(instruction, analysis):
        return instruction

    held_objs = _resolved_held_objects(instruction, analysis)
    if not held_objs:
        return instruction

    resolved = resolve_instruction_objects(instruction, analysis)
    if len(resolved) < 2:
        return None

    held_key = held_objs[0].key()
    held_idx = next(
        (i for i, obj in enumerate(resolved) if obj.key() == held_key),
        -1,
    )
    if held_idx <= 0:
        return instruction

    new_objects = list(instruction.objects)
    if held_idx < len(new_objects):
        new_objects[0], new_objects[held_idx] = new_objects[held_idx], new_objects[0]

    new_nouns = list(instruction.nouns)
    if len(new_nouns) >= 2 and held_idx < len(new_nouns):
        new_nouns[0], new_nouns[held_idx] = new_nouns[held_idx], new_nouns[0]

    new_adjectives = list(instruction.adjectives)
    if len(new_adjectives) >= 2 and held_idx < len(new_adjectives):
        new_adjectives[0], new_adjectives[held_idx] = (
            new_adjectives[held_idx],
            new_adjectives[0],
        )

    new_task = _swap_object_phrases_in_text(
        instruction.task_description,
        resolved[0],
        resolved[held_idx],
    )
    promoted = replace(
        instruction,
        objects=new_objects,
        nouns=new_nouns,
        adjectives=new_adjectives,
        task_description=new_task,
    )
    if not instruction_primary_is_held(promoted, analysis):
        logger.debug(
            "dropping instruction with held as passive second noun: %s",
            instruction.task_description,
        )
        return None
    if promoted.task_description.lower() != instruction.task_description.lower():
        logger.debug(
            "promoted held to first noun: %r -> %r",
            instruction.task_description,
            promoted.task_description,
        )
    return promoted


def ooi_instruction_quota(analysis: ImageAnalysis) -> int:
    """Minimum instructions that should reference suggested_out_of_image_objects."""
    n = len(analysis.suggested_out_of_image_objects)
    return max(1, n) if n else 0


def instruction_involves_ooi(
    instruction: FeasibleInstruction,
    analysis: ImageAnalysis,
) -> bool:
    for obj in resolve_instruction_objects(instruction, analysis):
        if analysis.is_out_of_image(obj):
            return True
    return False


def object_catalog_json(analysis: ImageAnalysis) -> str:
    """Serialize Stage-1 objects for the instruction-generation prompt."""
    ooi = analysis.suggested_out_of_image_objects
    catalog: dict = {
        "hand_holding": analysis.hand_holding,
        "both_hands_holding": analysis.both_hands_holding,
        "visible_objects": [o.to_dict() for o in analysis.visible_objects],
        "left_held_object": (
            analysis.left_held_object.to_dict()
            if analysis.left_held_object
            else None
        ),
        "right_held_object": (
            analysis.right_held_object.to_dict()
            if analysis.right_held_object
            else None
        ),
        "suggested_out_of_image_objects": [o.to_dict() for o in ooi],
    }
    if analysis.scene_description:
        catalog["scene_description"] = analysis.scene_description
    if ooi:
        catalog["_catalog_note"] = (
            "suggested_out_of_image_objects are ALREADY on the desk at their bbox_2d "
            "positions for instruction planning. Treat them like visible_objects in "
            "scene_description and feasible_instructions. Image editing later paints ONLY "
            "suggested_out_of_image_objects — visible_objects are already in the photo."
        )
    if analysis.hand_holding and analysis.held_objects():
        held_labels = ", ".join(obj.label() for _, obj in analysis.held_objects())
        catalog["_held_note"] = (
            f"hand_holding=true: {held_labels} already in hand — put held item(s) in "
            f"objects[0] / nouns[0]; task_description must name held item as the FIRST noun."
        )
    catalog["_layout_note"] = (
        "Do NOT propose instructions that restate the current layout. "
        "If an object is already on/beside another object in scene_description, "
        "skip place/put/slide tasks that only repeat that relation."
    )
    foods = _food_objects_in_catalog(analysis)
    boards = _work_surfaces_in_catalog(analysis)
    if boards:
        on_board = food_on_surface(analysis, boards[0])
        if on_board:
            catalog["_food_note"] = (
                f"Food on {boards[0].label()}: {on_board[0].label()} — "
                "slice/cut tasks may reference this food only."
            )
        else:
            catalog["_food_note"] = (
                f"No food item in catalog on {boards[0].label()}. "
                "Do NOT say slice/cut the food. "
                "Use feasible alternatives: scrape, wipe, or press along the board."
            )
    elif not foods:
        catalog["_food_note"] = (
            "No food-like object in catalog — do NOT invent food in instructions."
        )
    return json.dumps(catalog, ensure_ascii=False, indent=2)


_GRASP_PICK_ONLY = re.compile(
    r"^(?:grasp|pick(?:\s+up)?|grab|hold\s+onto)\s+(?:the\s+)?",
    re.IGNORECASE,
)
_GRASP_PICK_VERBS = frozenset({"pick", "grasp", "lift", "grab", "hold"})
_PUT_DOWN_VERBS = frozenset({"place", "put", "set", "drop", "release", "lay", "rest"})
_PUT_DOWN_VERB_POOL = ("put", "set", "lay", "rest", "place")
_VERB_SELECTION_CAPS = {"place": 1, "put": 1, "rotate": 1}
_ON_DEST_RE = re.compile(r"\b(?:on(?:to)?|upon|into|in(?:to)?|inside)\b", re.I)
_BESIDE_DEST_RE = re.compile(
    r"\b(?:beside|next to|alongside|near|by)\b",
    re.I,
)
_HELD_OPERATION_VERBS = (
    "move", "slide", "open", "close", "tilt", "press", "align", "use", "wipe",
    "pour", "insert", "remove", "rotate", "flip", "nudge", "turn", "scoop",
)
_HELD_SINGLE_VERBS = _PUT_DOWN_VERB_POOL + _HELD_OPERATION_VERBS
_HELD_TWO_OBJECT_VERBS = _PUT_DOWN_VERB_POOL + ("slide", "move", "align", "use")
_MAX_HELD_OPERATION_PER_HELD = 1
_MAX_SAME_VERB_PER_HELD = 1


def _bbox_center(bbox: List[int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _object_resting_on(inner: List[int], outer: List[int], *, margin: int = 35) -> bool:
    """True when inner object's center sits on/over outer surface bbox."""
    icx, icy = _bbox_center(inner)
    ox1, oy1, ox2, oy2 = outer
    if not (ox1 - margin <= icx <= ox2 + margin and oy1 - margin <= icy <= oy2 + margin):
        return False
    ix1, iy1, ix2, iy2 = inner
    overlap_w = max(0, min(ix2, ox2) - max(ix1, ox1))
    overlap_h = max(0, min(iy2, oy2) - max(iy1, oy1))
    overlap = overlap_w * overlap_h
    inner_area = max(1, (ix2 - ix1) * (iy2 - iy1))
    return overlap / inner_area >= 0.2


def _objects_adjacent(a: List[int], b: List[int], *, max_gap: int = 90) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    h_gap = max(0, max(bx1 - ax2, ax1 - bx2))
    v_gap = max(0, max(by1 - ay2, ay1 - by2))
    if min(h_gap, v_gap) >= max_gap:
        return False
    overlap_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    overlap_h = max(0, min(ay2, by2) - max(ay1, by1))
    if overlap_w * overlap_h > 0.15 * min(_bbox_area_qwen(a), _bbox_area_qwen(b)):
        return False
    return True


def _scene_indicates_spatial_relation(
    scene: str,
    target: ObjectRef,
    reference: ObjectRef,
    *,
    relation: str,
) -> bool:
    if not scene:
        return False
    scene_l = scene.lower()
    target_tokens = _label_scene_tokens(target)
    ref_tokens = _label_scene_tokens(reference)
    if not target_tokens or not ref_tokens:
        return False
    for tt in target_tokens:
        for rt in ref_tokens:
            if relation == "on":
                patterns = (
                    f"{tt} on {rt}",
                    f"{tt} on the {rt}",
                    f"{tt} atop the {rt}",
                    f"slicing on the {rt}",
                )
                if any(p in scene_l for p in patterns):
                    return True
                if tt in scene_l and rt in scene_l:
                    if re.search(
                        rf"{re.escape(tt)}.{{0,50}}\bon\b.{{0,30}}{re.escape(rt)}",
                        scene_l,
                    ):
                        return True
            elif relation == "beside":
                patterns = (
                    f"{tt} beside {rt}",
                    f"{tt} next to {rt}",
                    f"{tt} near the {rt}",
                )
                if any(p in scene_l for p in patterns):
                    return True
    return False


def _pick_alternate_put_down_verb(
    verb_counts: Dict[str, int],
    rng: random.Random,
) -> str:
    available = [v for v in _PUT_DOWN_VERB_POOL if verb_counts.get(v, 0) < 1]
    if not available:
        available = [
            v for v in _PUT_DOWN_VERB_POOL
            if verb_counts.get(v, 0) < _MAX_SAME_VERB_PER_HELD
        ]
    if not available:
        available = list(_PUT_DOWN_VERB_POOL)
    return rng.choice(available)


def instruction_already_satisfied(
    instruction: FeasibleInstruction,
    analysis: ImageAnalysis,
    *,
    target: Optional[ObjectRef] = None,
    reference: Optional[ObjectRef] = None,
) -> bool:
    """True when the task describes a layout/state that already holds in the scene."""
    task = (instruction.task_description or "").strip().lower()
    if not task:
        return False
    verbs = {v.lower().strip() for v in instruction.verbs if v}

    resolved = resolve_instruction_objects(instruction, analysis)
    tgt = target if target and target.noun else (resolved[0] if resolved else None)
    ref = (
        reference
        if reference and reference.noun
        else (resolved[1] if len(resolved) > 1 else None)
    )
    if tgt is None:
        return False

    if verbs & _GRASP_PICK_VERBS and analysis.hand_holding:
        if tgt.key() in _held_keys(analysis):
            return True

    if instruction_is_held_operation(instruction, analysis):
        return False

    if verbs & _PUT_DOWN_VERBS or verbs & {"slide", "move", "set", "align"}:
        if ref and tgt.bbox_2d and ref.bbox_2d:
            if _ON_DEST_RE.search(task) or " on " in task:
                if _object_resting_on(tgt.bbox_2d, ref.bbox_2d):
                    return True
            if _BESIDE_DEST_RE.search(task) or " beside " in task or " next to " in task:
                if _objects_adjacent(tgt.bbox_2d, ref.bbox_2d):
                    return True
        if ref and analysis.scene_description:
            if (_ON_DEST_RE.search(task) or " on " in task) and _scene_indicates_spatial_relation(
                analysis.scene_description, tgt, ref, relation="on"
            ):
                return True
            if _BESIDE_DEST_RE.search(task) and _scene_indicates_spatial_relation(
                analysis.scene_description, tgt, ref, relation="beside"
            ):
                return True

    return False


def filter_already_satisfied_instructions(
    instructions: List[FeasibleInstruction],
    analysis: ImageAnalysis,
) -> List[FeasibleInstruction]:
    filtered = [
        instr
        for instr in instructions
        if not instruction_already_satisfied(instr, analysis)
    ]
    dropped = len(instructions) - len(filtered)
    if dropped:
        logger.info(
            "Dropped %d instruction(s) that repeat the current scene layout",
            dropped,
        )
    return filtered if filtered else list(instructions)


def filter_already_satisfied_instruction_dicts(
    instructions: List[Dict[str, Any]],
    analysis: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not analysis:
        return list(instructions)
    ia = ImageAnalysis.from_dict(analysis)
    filtered: List[Dict[str, Any]] = []
    for row in instructions:
        fi = FeasibleInstruction.from_dict(row)
        target = ObjectRef.from_dict(row.get("target_object"))
        reference = ObjectRef.from_dict(row.get("reference_object"))
        if instruction_already_satisfied(
            fi,
            ia,
            target=target if target.noun else None,
            reference=reference if reference.noun else None,
        ):
            continue
        filtered.append(row)
    dropped = len(instructions) - len(filtered)
    if dropped:
        logger.info(
            "Dropped %d manifest instruction(s) that repeat the current scene layout",
            dropped,
        )
    return filtered if filtered else list(instructions)
_HOLD_STEADY = re.compile(r"\bhold\s+the\s+.+\s+steady\b", re.IGNORECASE)
_COOP_VERBS = frozenset({
    "hold", "steady", "stabilize", "stabilise", "support", "guide",
    "position", "align", "fix", "secure", "brace", "pin", "press",
    "anchor", "nudge", "tilt", "rotate", "shift", "slide", "pull",
    "push", "lift", "lower", "open", "part", "separate", "clear",
    "present", "receive", "catch", "reposition", "make",
})


def is_valid_bimanual_instruction(instr: FeasibleInstruction) -> bool:
    """True when both hands have distinct, coordinated sub-instructions."""
    if not instr.use_both_hands:
        return False
    left = (instr.left_instruction or "").strip()
    right = (instr.right_instruction or "").strip()
    if len(left) < 8 or len(right) < 8:
        return False
    if left.lower() in {"none", "none."} or right.lower() in {"none", "none."}:
        return False
    if left.lower() == right.lower():
        return False
    ctype = (instr.coordination_type or "").strip().lower()
    return ctype in BIMANUAL_COORDINATION_TYPES


def is_cooperative_bimanual_decomposition(
    instr: FeasibleInstruction,
    *,
    reject_steady_hold: bool = False,
) -> bool:
    """
    Stricter check for instr_first LLM splits: reject dual-grasp-only outputs
    that do not accomplish the overall manipulation verb.
    """
    if not is_valid_bimanual_instruction(instr):
        return False
    left = (instr.left_instruction or "").strip().lower()
    right = (instr.right_instruction or "").strip().lower()
    if _GRASP_PICK_ONLY.match(left) and _GRASP_PICK_ONLY.match(right):
        return False
    if reject_steady_hold and (
        _HOLD_STEADY.search(left) or _HOLD_STEADY.search(right)
    ):
        return False

    combined = f"{left} {right}"
    main_verbs = [v.lower() for v in instr.verbs if v]
    has_main_verb = any(v in combined for v in main_verbs) if main_verbs else True
    has_coop = any(v in combined for v in _COOP_VERBS)
    if main_verbs and not has_main_verb and not has_coop:
        return False
    return True


def feasibility_score(
    instruction: FeasibleInstruction,
    analysis: ImageAnalysis,
) -> float:
    """Higher = more physically plausible for the current scene."""
    score = 0.0
    if is_valid_bimanual_instruction(instruction):
        score += 50.0
    verbs = {v.lower().strip() for v in instruction.verbs if v}
    if analysis.hand_holding:
        if instruction_uses_grasp_on_held(instruction, analysis):
            score -= 100.0
        elif instruction_is_held_operation(instruction, analysis):
            score += 40.0
            if "use" in verbs:
                score += 20.0
        elif instruction_puts_down(instruction) and instruction_primary_is_held(
            instruction, analysis
        ):
            score += 18.0
        elif instruction_primary_is_held(instruction, analysis):
            score += 8.0
        elif instruction_involves_held(instruction, analysis):
            score += 4.0
        else:
            score -= 100.0
    elif verbs & _PUT_DOWN_VERBS:
        score += 4.0
    if instruction.action_type == "two_object":
        score += 3.0
    return score


def _rank_feasible_pool(
    pool: List[FeasibleInstruction],
    analysis: ImageAnalysis,
    seed: int,
) -> List[FeasibleInstruction]:
    rng = random.Random(seed)
    ranked = sorted(
        pool,
        key=lambda instr: (
            feasibility_score(instr, analysis) + rng.random() * 0.01
        ),
        reverse=True,
    )
    return ranked


def _verb_selection_blocked(verb: str, verb_counts: Dict[str, int], used_verbs: Set[str]) -> bool:
    if not verb:
        return False
    cap = _VERB_SELECTION_CAPS.get(verb)
    if cap is not None:
        return verb_counts.get(verb, 0) >= cap
    return verb in used_verbs


def _take_top_diverse(
    ranked: List[FeasibleInstruction],
    n: int,
    *,
    used_verbs: Set[str],
    verb_counts: Dict[str, int],
    seen: set,
    selected: List[FeasibleInstruction],
) -> int:
    """Take up to n instructions from ranked pool, preferring verb diversity."""
    taken = 0
    deferred: List[FeasibleInstruction] = []
    for instr in ranked:
        if taken >= n:
            break
        key = instr.task_description.lower()
        if key in seen:
            continue
        main_verb = (instr.verbs[0] if instr.verbs else "").lower().strip()
        if main_verb and _verb_selection_blocked(main_verb, verb_counts, used_verbs):
            deferred.append(instr)
            continue
        seen.add(key)
        selected.append(instr)
        if main_verb:
            used_verbs.add(main_verb)
            verb_counts[main_verb] = verb_counts.get(main_verb, 0) + 1
        taken += 1
    if taken < n:
        for instr in deferred:
            if taken >= n:
                break
            key = instr.task_description.lower()
            if key in seen:
                continue
            main_verb = (instr.verbs[0] if instr.verbs else "").lower().strip()
            if main_verb and _verb_selection_blocked(main_verb, verb_counts, used_verbs):
                continue
            seen.add(key)
            selected.append(instr)
            if main_verb:
                used_verbs.add(main_verb)
                verb_counts[main_verb] = verb_counts.get(main_verb, 0) + 1
            taken += 1
    return taken


def select_instructions(
    feasible: List[FeasibleInstruction],
    analysis: ImageAnalysis,
    k: int,
    out_of_image_ratio: float,
    seed: int,
    *,
    single_hand_only: bool = False,
) -> List[FeasibleInstruction]:
    """Pick the top-k most plausible instructions for this scene."""
    if not feasible:
        return []

    feasible = filter_already_satisfied_instructions(feasible, analysis)
    if single_hand_only:
        feasible = [i for i in feasible if not is_valid_bimanual_instruction(i)]

    reserved_bimanual: Optional[FeasibleInstruction] = None
    if not single_hand_only:
        bimanual_candidates = _rank_feasible_pool(
            [i for i in feasible if is_valid_bimanual_instruction(i)],
            analysis,
            seed,
        )
        if bimanual_candidates:
            reserved_bimanual = bimanual_candidates[0]

    if analysis.hand_holding:
        primary_held = _rank_feasible_pool(
            [i for i in feasible if instruction_primary_is_held(i, analysis)],
            analysis,
            seed + 1,
        )
        desk_only = _rank_feasible_pool(
            [i for i in feasible if not instruction_involves_held(i, analysis)],
            analysis,
            seed + 2,
        )
        pool = primary_held + desk_only
        effective_ooi_ratio = out_of_image_ratio
    else:
        pool = _rank_feasible_pool(list(feasible), analysis, seed)
        effective_ooi_ratio = out_of_image_ratio

    in_pool = [i for i in pool if not instruction_needs_ooi_edit(i, analysis)[0]]
    ooi_pool = [i for i in pool if instruction_needs_ooi_edit(i, analysis)[0]]

    selected: List[FeasibleInstruction] = []
    seen: set = set()
    used_verbs: Set[str] = set()
    verb_counts: Dict[str, int] = {}

    remaining_k = k
    if reserved_bimanual and remaining_k > 0:
        seen.add(reserved_bimanual.task_description.lower())
        selected.append(reserved_bimanual)
        for verb in reserved_bimanual.verbs:
            v = str(verb).lower().strip()
            used_verbs.add(v)
            verb_counts[v] = verb_counts.get(v, 0) + 1
        remaining_k -= 1

    if remaining_k > 0:
        if effective_ooi_ratio > 0 and ooi_pool:
            n_ooi = max(1, math.ceil(remaining_k * effective_ooi_ratio))
        else:
            n_ooi = int(round(remaining_k * effective_ooi_ratio))
        n_ooi = min(n_ooi, remaining_k, len(ooi_pool))
        n_in = remaining_k - n_ooi
        n_ooi = _take_top_diverse(
            ooi_pool,
            n_ooi,
            used_verbs=used_verbs,
            verb_counts=verb_counts,
            seen=seen,
            selected=selected,
        )
        n_in = _take_top_diverse(
            in_pool,
            n_in,
            used_verbs=used_verbs,
            verb_counts=verb_counts,
            seen=seen,
            selected=selected,
        )
        if len(selected) < k:
            _take_top_diverse(
                pool,
                k - len(selected),
                used_verbs=used_verbs,
                verb_counts=verb_counts,
                seen=seen,
                selected=selected,
            )

    selected = _rebalance_held_put_down_priority(selected, analysis, k=k)
    return selected[:k]


def _rebalance_held_put_down_priority(
    instructions: List[FeasibleInstruction],
    analysis: ImageAnalysis,
    *,
    k: Optional[int] = None,
) -> List[FeasibleInstruction]:
    """When holding, order candidates: in-hand use first, then put-down, then rest."""
    if not analysis.hand_holding:
        return instructions[:k] if k is not None else instructions

    limit = k if k is not None else len(instructions)
    put_down = [
        i
        for i in instructions
        if instruction_puts_down(i) and instruction_primary_is_held(i, analysis)
    ]
    manip = [
        i
        for i in instructions
        if instruction_is_held_operation(i, analysis)
        and not is_valid_bimanual_instruction(i)
    ]
    use_first = sorted(
        manip,
        key=lambda i: (
            0
            if "use" in {v.lower() for v in i.verbs}
            or " use " in i.task_description.lower()
            else 1
        ),
    )
    rest = [i for i in instructions if i not in put_down and i not in manip]

    ordered: List[FeasibleInstruction] = []
    seen: Set[str] = set()

    def _add(source: List[FeasibleInstruction], max_n: Optional[int] = None) -> None:
        added = 0
        for instr in source:
            if len(ordered) >= limit:
                return
            if max_n is not None and added >= max_n:
                return
            key = instr.task_description.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(instr)
            added += 1

    diversified_put_down = _diversify_put_down_verbs(put_down, analysis, seed=0)
    _add(use_first, _MAX_HELD_OPERATION_PER_HELD)
    _add(diversified_put_down, min(2, limit))
    _add(rest)
    _add(use_first)
    _add(diversified_put_down)
    return ordered[:limit]


def _diversify_put_down_verbs(
    instructions: List[FeasibleInstruction],
    analysis: ImageAnalysis,
    *,
    seed: int,
) -> List[FeasibleInstruction]:
    """Ensure put-down instructions use varied release verbs (put/set/lay/rest/place)."""
    rng = random.Random(seed)
    verb_counts: Dict[str, int] = {}
    diversified: List[FeasibleInstruction] = []
    for instr in instructions:
        if not instruction_puts_down(instr):
            diversified.append(instr)
            continue
        verb = (instr.verbs[0] if instr.verbs else "place").lower().strip()
        candidate = instr
        if verb_counts.get(verb, 0) >= 1 and instruction_primary_is_held(instr, analysis):
            resolved = resolve_instruction_objects(instr, analysis)
            if resolved:
                held = resolved[0]
                ref = resolved[1] if len(resolved) > 1 else None
                new_verb = _pick_alternate_put_down_verb(verb_counts, rng)
                candidate = _rewrite_held_instruction_verb(
                    instr,
                    held=held,
                    new_verb=new_verb,
                    ref=ref,
                    analysis=analysis,
                )
                verb = new_verb
        verb_counts[verb] = verb_counts.get(verb, 0) + 1
        diversified.append(candidate)
    return diversified


def filter_invalid_use_instructions(
    instructions: List[FeasibleInstruction],
    analysis: ImageAnalysis,
) -> List[FeasibleInstruction]:
    filtered = [
        instr
        for instr in instructions
        if is_valid_use_instruction(instr, analysis)
    ]
    dropped = len(instructions) - len(filtered)
    if dropped:
        logger.info(
            "Dropped %d invalid/ungrounded use instruction(s)",
            dropped,
        )
    return filtered if filtered else list(instructions)


def ensure_grounded_held_use_candidate(
    instructions: List[FeasibleInstruction],
    analysis: ImageAnalysis,
    *,
    seed: int = 0,
) -> List[FeasibleInstruction]:
    """Inject a catalog-grounded use fallback when holding but no valid use remains."""
    if not analysis.hand_holding:
        return instructions
    has_use = any(
        instruction_is_held_operation(instr, analysis)
        and is_valid_use_instruction(instr, analysis)
        for instr in instructions
    )
    if has_use:
        return instructions
    fallbacks = synthesize_held_fallback_instructions(analysis, seed=seed)
    use_fb = [
        fb
        for fb in fallbacks
        if instruction_uses_use_verb(fb) and is_valid_use_instruction(fb, analysis)
    ]
    if not use_fb:
        return instructions
    out = list(instructions)
    key = use_fb[0].task_description.lower()
    if not any(i.task_description.lower() == key for i in out):
        out.insert(0, use_fb[0])
        logger.info("Injected grounded use fallback: %s", use_fb[0].task_description)
    return out


def prioritize_held_instructions(
    feasible: List[FeasibleInstruction],
    analysis: ImageAnalysis,
    *,
    min_held_ratio: float = 0.7,
    seed: int = 0,
) -> List[FeasibleInstruction]:
    """
    When hand_holding, promote held item to objects[0]/first noun, reorder, and trim
    desk-only or passive-second-noun instructions.
    """
    if not analysis.hand_holding or not feasible:
        return feasible

    promoted: List[FeasibleInstruction] = []
    seen: Set[str] = set()
    dropped_passive = 0
    dropped_grasp = 0
    for instr in feasible:
        fixed = promote_held_to_primary(instr, analysis)
        if fixed is None:
            dropped_passive += 1
            continue
        if instruction_uses_grasp_on_held(fixed, analysis):
            dropped_grasp += 1
            continue
        if not is_valid_use_instruction(fixed, analysis):
            continue
        key = fixed.task_description.lower()
        if key in seen:
            continue
        seen.add(key)
        promoted.append(fixed)
    if dropped_passive:
        logger.debug(
            "hand_holding: dropped %d instruction(s) with held object as passive second noun",
            dropped_passive,
        )
    if dropped_grasp:
        logger.debug(
            "hand_holding: dropped %d pick/grasp/lift instruction(s) on in-hand items",
            dropped_grasp,
        )

    bimanual = [i for i in promoted if is_valid_bimanual_instruction(i)]
    non_bimanual = [i for i in promoted if not is_valid_bimanual_instruction(i)]

    primary_held = [
        i for i in non_bimanual if instruction_primary_is_held(i, analysis)
    ]
    desk_only = [
        i for i in non_bimanual if not instruction_involves_held(i, analysis)
    ]

    n_non_bi = len(non_bimanual)
    min_held_count = max(1, math.ceil(n_non_bi * min_held_ratio)) if n_non_bi else 0
    dropped = desk_only

    if len(primary_held) < min_held_count:
        logger.warning(
            "hand_holding: only %d/%d non-bimanual instructions have held as first object; "
            "injecting fallback held-object instructions",
            len(primary_held),
            min_held_count,
        )
        for fallback in synthesize_held_fallback_instructions(analysis, seed=seed):
            key = fallback.task_description.lower()
            if key in seen:
                continue
            seen.add(key)
            primary_held.append(fallback)

    if dropped:
        logger.debug(
            "hand_holding: dropped %d desk-only instruction(s) while hand holds item(s)",
            len(dropped),
        )

    ordered = primary_held
    for instr in bimanual:
        if instr not in ordered:
            ordered.append(instr)
    diversified = diversify_held_instruction_verbs(ordered, analysis, seed=seed)
    diversified = ensure_grounded_held_use_candidate(
        diversified, analysis, seed=seed
    )
    return _rebalance_held_put_down_priority(diversified, analysis)


def build_instruction_record(
    instruction: FeasibleInstruction,
    analysis: ImageAnalysis,
) -> InstructionRecord:
    objects = resolve_instruction_objects(instruction, analysis)
    ooi_objs = ooi_objects_in_instruction(instruction, analysis)

    held_in_objects = _resolved_held_objects(instruction, analysis)
    if held_in_objects and instruction_primary_is_held(instruction, analysis):
        target = objects[0] if objects else held_in_objects[0]
        desk_objects = [o for o in objects if o.key() not in _held_keys(analysis)]
        reference = desk_objects[0] if desk_objects else (
            objects[1] if len(objects) > 1 and objects[1].key() != target.key() else None
        )
    elif held_in_objects:
        target = held_in_objects[0]
        desk_objects = [o for o in objects if o.key() not in _held_keys(analysis)]
        reference = desk_objects[0] if desk_objects else (
            objects[1] if len(objects) > 1 and objects[1].key() != target.key() else None
        )
    elif len(ooi_objs) >= 2:
        target = ooi_objs[0]
        reference = ooi_objs[1]
    elif len(ooi_objs) == 1:
        target = ooi_objs[0]
        visible_refs = [o for o in objects if not analysis.is_out_of_image(o)]
        reference = visible_refs[0] if visible_refs else (
            objects[1] if len(objects) > 1 else None
        )
    else:
        target = objects[0] if objects else ObjectRef(noun="object", adjective="")
        reference = objects[1] if len(objects) > 1 else None

    target_hand = analysis.hand_for_held(target) or ""

    action_type = instruction.action_type or (
        "two_object" if len(objects) >= 2 else "single"
    )

    if is_valid_bimanual_instruction(instruction):
        left_instruction = instruction.left_instruction
        right_instruction = instruction.right_instruction
        use_both_hands = True
        coordination_type = instruction.coordination_type
    else:
        left_instruction = "None"
        right_instruction = instruction.task_description
        use_both_hands = False
        coordination_type = ""

    return InstructionRecord(
        task_description=instruction.task_description,
        action_type=action_type,
        verbs=list(instruction.verbs),
        nouns=list(instruction.nouns),
        adjectives=list(instruction.adjectives),
        objects=objects,
        left_instruction=left_instruction,
        right_instruction=right_instruction,
        use_both_hands=use_both_hands,
        target_in_image=not analysis.is_out_of_image(target),
        target_object=target,
        reference_object=reference,
        target_hand=target_hand,
        coordination_type=coordination_type,
        ooi_object_count=len(ooi_objs),
    )

"""Video prompt builders for wm_h (no hand-entry / no task-instruction clauses)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from wm_h.video.common import format_bimanual_instruction, normalize_task_fields

DESK_SYNTH_CAMERA_SINGLE = (
    "4k high definition, completely static fixed camera, first-person top-down view, "
    "no camera movement, no zoom, no pan, no tilt, camera locked throughout the entire video, "
    "showing human {hand} hand and objects on the desktop."
)

DESK_SYNTH_CAMERA_BIMANUAL = (
    "4k high definition, completely static fixed camera, first-person top-down view, "
    "no camera movement, no zoom, no pan, no tilt, camera locked throughout the entire video, "
    "showing both human hands and objects on the desktop."
)

_PLACE_PUT_RE = re.compile(
    r"\b(place|put|set|lay|rest|drop)\b",
    re.I,
)

_USE_TO_RE = re.compile(r"\buse\b.+\bto\b", re.I | re.DOTALL)


def is_place_put_instruction(text: str) -> bool:
    """True when the task is placing/putting an object down (needs explicit release)."""
    return bool(_PLACE_PUT_RE.search(text or ""))


def place_release_video_clause() -> str:
    """Static Wan prompt suffix so place tasks end with visible finger release."""
    return (
        "For place/put tasks the hand must visibly open its grip and release the object "
        "onto the destination surface, then withdraw empty and still — do not keep holding."
    )


def is_use_tool_instruction(text: str) -> bool:
    """True when the task is holding a tool/object and applying it (use X to Y)."""
    return bool(_USE_TO_RE.search(text or ""))


def retry_augment_rules(*, use_zh: bool, instruction: str) -> str:
    """Extra VL rules: one visible failure + retry before task success."""
    use_tool = is_use_tool_instruction(instruction)
    if use_zh:
        failure_detail = (
            "失败须写具体手物交互：滑脱、没抓稳、对准偏差、施力后物体回弹/倾斜、"
            "工具在目标上打滑等，禁止只写「失败了」而不写接触细节。"
        )
        if use_tool:
            return (
                "失败重试（额外必须遵守）：\n"
                "- 视频须包含一次失败的「use … to …」尝试，再成功完成同一任务。\n"
                f"- {failure_detail}\n"
                "- 手已握着工具/物体时：失败后手指保持握紧，不要张开或放下工具；"
                "稍作调整后再次执行 to … 动作。\n"
                "- 第二次尝试须写清与目标的接触、施力与效果，最后任务成功完成。\n"
                "- 失败与重试各写一次，成功步骤仍只写一次。\n"
            )
        return (
            "失败重试（额外必须遵守）：\n"
            "- 视频须包含一次失败尝试，再成功完成同一任务。\n"
            f"- {failure_detail}\n"
            "- 失败后手指逐根张开、松开物体或撤离接触，空手或复位后再重新接近并尝试。\n"
            "- 第二次尝试从接近/对准/抓取（或相应动作）写起，最后任务成功完成。\n"
            "- 失败与重试各写一次，成功步骤仍只写一次。\n"
        )

    failure_detail = (
        "Failures must show concrete hand-object contact: slip, mis-grasp, misalignment, "
        "object wobble/spring-back, tool skidding on target — not a vague \"failed\"."
    )
    if use_tool:
        return (
            "Failure + retry (REQUIRED):\n"
            "- Include one failed \"use … to …\" attempt, then a successful completion.\n"
            f"- {failure_detail}\n"
            "- While still gripping the tool/object: do NOT open the hand or put it down; "
            "adjust and retry the same \"to …\" action.\n"
            "- Second attempt needs visible contact, force, and effect; task ends completed.\n"
            "- Describe failure and retry once each; successful steps still happen once.\n"
        )
    return (
        "Failure + retry (REQUIRED):\n"
        "- Include one failed attempt, then a successful completion of the same task.\n"
        f"- {failure_detail}\n"
        "- After failure: fingers visibly open, release the object or break contact, "
        "then re-approach and try again.\n"
        "- Second attempt starts from approach/alignment/grasp (as appropriate) and succeeds.\n"
        "- Describe failure and retry once each; successful steps still happen once.\n"
    )


def place_release_augment_rules(*, use_zh: bool) -> str:
    """Extra VL augmentation rules for place/put-down motions."""
    if use_zh:
        return (
            "放置/放下任务（额外必须遵守）：\n"
            "- 若物体已在手中：先缓慢移到目标位置 → 降低物体直至接触台面 → "
            "手指逐根张开、虎口松开 → 明确写「松开/释放物体」→ 空手撤离并保持静止。\n"
            "- 禁止物体已放好但手仍紧握不放；视频结束时该手必须为空手。\n"
            "- 松开动作只写一次，放在放置之后、手静止之前。\n"
        )
    return (
        "Place/put-down rules (REQUIRED when task uses place/put/set):\n"
        "- If the object is already in hand: slow move to destination → lower until contact "
        "with the surface → fingers visibly open one by one → explicitly write "
        "\"releases the object\" / \"opens grip and lets go\" → hand withdraws empty and still.\n"
        "- Do NOT end while still gripping after the object rests at the destination.\n"
        "- Describe release exactly once, after placing and before the hand stops.\n"
    )


def desk_synth_motion_tail(bimanual: bool = False) -> str:
    """Short motion constraint + ending (replaces long slow_manipulation clause)."""
    hands = "Both hands" if bimanual else "The hand"
    return (
        f"Slow smooth motion, visible grasp contact, no snapping or teleporting. "
        f"Camera stays completely still. {hands} stops when done."
    )


def scene_object_labels(row: Dict[str, Any]) -> List[str]:
    """Collect human-readable object labels from a synth manifest row."""
    labels: List[str] = []
    seen: set = set()

    def _add(obj: Any) -> None:
        if not isinstance(obj, dict):
            return
        adj = (obj.get("adjective") or "").strip()
        noun = (obj.get("noun") or "").strip()
        if not noun:
            return
        label = f"{adj} {noun}".strip() if adj else noun
        key = label.lower()
        if key not in seen:
            seen.add(key)
            labels.append(label)

    for obj in row.get("objects") or []:
        _add(obj)
    _add(row.get("target_object"))
    _add(row.get("reference_object"))
    return labels


def build_desk_synth_task_json(entry: Dict[str, Any], *, video_path: str = "") -> Dict[str, Any]:
    """Rich task metadata written to tasks/{task_id}.json after video generation."""
    base = normalize_task_fields({
        "task_id": entry.get("task_id", ""),
        "hand": entry.get("hand", ""),
        "instruction": entry.get("instruction", ""),
        "task_description": entry.get("task_description", ""),
        "left_instruction": entry.get("left_instruction", ""),
        "right_instruction": entry.get("right_instruction", ""),
        "shared_object": entry.get("shared_object", ""),
        "coordination_type": entry.get("coordination_type", ""),
        "augmented_text": entry.get("augmented_text", ""),
    })
    out: Dict[str, Any] = {
        **base,
        "base_task_id": entry.get("base_task_id", entry.get("task_id", "")),
        "rollout_idx": entry.get("rollout_idx"),
        "generation_seed": entry.get("seed"),
        "video_prompt": entry.get("video_prompt", ""),
        "video_path": video_path,
        "image_path": entry.get("image_path", ""),
        "edited_image_path": entry.get("edited_image_path", ""),
        "source_image_path": entry.get("source_image_path", ""),
        "desktop_image_path": entry.get("desktop_image_path", ""),
        "synth_manifest": entry.get("synth_manifest", ""),
        "synth_run_dir": entry.get("synth_run_dir", entry.get("run_dir", "")),
        "mode": entry.get("mode", ""),
        "action_type": entry.get("action_type", ""),
        "use_both_hands": entry.get("use_both_hands", False),
        "target_in_image": entry.get("target_in_image"),
        "template_id": entry.get("template_id", ""),
        "verbs": entry.get("verbs", []),
        "nouns": entry.get("nouns", []),
        "adjectives": entry.get("adjectives", []),
        "objects": entry.get("objects", []),
        "visible_objects": entry.get("visible_objects", []),
        "target_object": entry.get("target_object"),
        "reference_object": entry.get("reference_object"),
        "use_visual_aug": entry.get("use_visual_aug", False),
        "scene_aug_path": entry.get("scene_aug_path", ""),
        "hand_trajectories": entry.get("hand_trajectories", {}),
        "ooi_object_count": entry.get("ooi_object_count", 0),
    }
    return out

# Strip fixed openings from Qwen augmented_text before appending to video_prompt.
_AUG_OPENING_PATTERNS = [
    re.compile(r"^双手从画面下方伸入画面[。.]?\s*"),
    re.compile(r"^人物使用双手协同完成动作[：:]\s*"),
    re.compile(r"^右手从画面右下角伸入画面[。.]?\s*"),
    re.compile(r"^左手从画面左下角伸入画面[。.]?\s*"),
    re.compile(r"^人物仅使用右手完成动作[：:]\s*"),
    re.compile(r"^人物仅使用左手完成动作[：:]\s*"),
    re.compile(
        r"^Both hands enter from the bottom of the frame[.]?\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^The person uses both hands cooperatively[：:]\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^The (?:RIGHT|LEFT) hand enters from the bottom-(?:right|left) of the frame[.]?\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^Only the (?:RIGHT|LEFT) hand is used[：:]\s*",
        re.IGNORECASE,
    ),
]


def sanitize_augmented_desc(text: str) -> str:
    """Remove mandated hand-entry openings from augmentation output."""
    out = (text or "").strip()
    if not out:
        return ""
    for _ in range(4):
        prev = out
        for pat in _AUG_OPENING_PATTERNS:
            out = pat.sub("", out).strip()
        if out == prev:
            break
    return out


def build_desk_synth_text_augmentation_prompt(
    instruction: str,
    hand: str = "right",
    *,
    mode: str = "single",
    left_instruction: str = "",
    right_instruction: str = "",
    shared_goal: str = "",
    shared_object: str = "",
    coordination_type: str = "",
    visible_objects: Optional[List[str]] = None,
    output_language: str = "zh",
    include_retry: bool = False,
) -> str:
    """Qwen VL augmentation grounded on the edited scene image."""
    lang = (output_language or "zh").strip().lower()
    use_zh = lang in ("zh", "cn", "chinese", "中文")
    obj_list = "、".join(visible_objects) if visible_objects else ("（见图中物体）" if use_zh else "(objects in image)")

    if use_zh:
        scene_intro = (
            "你正在观看用于视频生成的「编辑后场景图」——物体位置与图中一致。"
            "镜头固定俯视，全程禁止镜头移动、推拉、摇移、变焦。\n"
            f"图中任务相关物体：{obj_list}\n"
        )
        temporal_rules = (
            "写作规则（必须遵守）：\n"
            "1. 只描述图中可见物体的操作，禁止编造图中不存在的物体。\n"
            "2. 动作描述必须与任务一致，视频结束时任务须已完成。\n"
            "3. 不要复述完整指令句，用分步动作描述过程。\n"
            "4. 直接从「第一步：」开始，不要写手伸入画面等开场句。\n"
            "5. 禁止写镜头移动；画面视角始终固定。\n"
            "6. 节奏偏慢，抓取须分步写清接触与施力。\n"
            "7. place/put/set 类任务必须在结尾写清手指张开、松开物体、空手撤离。\n"
            "8. 一段连贯中文，不要 markdown，不要 JSON。"
        )
        lang_line = "请用中文输出。"
    else:
        scene_intro = (
            "You are viewing the EDITED scene image used for video generation. "
            "Objects match the image. Static top-down camera — no camera movement, zoom, pan, or tilt.\n"
            f"Task-relevant objects in image: {', '.join(visible_objects or ['(see image)'])}\n"
        )
        temporal_rules = (
            "Writing rules (STRICT):\n"
            "1. Only manipulate objects visible in this image; do not invent objects.\n"
            "2. Actions must accomplish the task; the video ends with task completed.\n"
            "3. Do not quote the instruction verbatim; describe step-by-step motion.\n"
            "4. Start with \"First,\" — no hand-entering-frame opening.\n"
            "5. Never describe camera movement; viewpoint stays fixed.\n"
            "6. Slow pace; grasping needs visible contact steps.\n"
            "7. place/put/set tasks MUST end with visible finger release, object resting, empty hand still.\n"
            "8. One coherent paragraph; no markdown."
        )
        lang_line = "Output in English."

    combined_instruction = " ".join(
        filter(
            None,
            [
                instruction,
                shared_goal,
                left_instruction,
                right_instruction,
            ],
        )
    )
    place_task = is_place_put_instruction(combined_instruction)
    retry_rules = (
        retry_augment_rules(use_zh=use_zh, instruction=combined_instruction)
        if include_retry
        else ""
    )

    if mode == "bimanual":
        combined = format_bimanual_instruction(left_instruction, right_instruction)
        coord = coordination_type or ("simultaneous" if use_zh else "simultaneous")
        if use_zh:
            context = (
                f"协同目标（必须达成）：{shared_goal or shared_object}\n"
                f"左手职责：{left_instruction}\n"
                f"右手职责：{right_instruction}\n"
                f"协调方式：{coord}\n"
                f"（分工参考，勿抄写）{combined}\n"
            )
            bimanual_rules = (
                "双手写作规则（额外）：\n"
                "- 每一步标明「左手：」「右手：」或「双手协同：」，不得混淆两手分工。\n"
                "- 左手步骤只写左手动作，右手步骤只写右手动作。\n"
                "- 按协调方式安排顺序（如先左后右、同时进行）。\n"
                "- 最终画面须达成协同目标。\n"
            )
            step_hint = (
                "从「第一步（左手）：」或「第一步（右手）：」开始，"
                "交替写清双手如何配合完成任务。"
            )
        else:
            context = (
                f"Shared goal (must achieve): {shared_goal or shared_object}\n"
                f"Left hand role: {left_instruction}\n"
                f"Right hand role: {right_instruction}\n"
                f"Coordination: {coord}\n"
                f"(Reference only) {combined}\n"
            )
            bimanual_rules = (
                "Bimanual rules:\n"
                "- Label each step Left hand: / Right hand: / Both hands: — never swap roles.\n"
                "- Follow coordination order; end with shared goal achieved.\n"
            )
            step_hint = 'Start with "First (left hand):" or "First (right hand):" and alternate clearly.'
        place_rules = place_release_augment_rules(use_zh=use_zh) if place_task else ""
        return (
            scene_intro
            + context
            + temporal_rules
            + ("\n" + place_rules if place_rules else "")
            + ("\n" + retry_rules if retry_rules else "")
            + "\n"
            + bimanual_rules
            + f"\n{step_hint}\n"
            + lang_line
            + (" 只输出描述正文。" if use_zh else " Output description only.")
        )

    hand_zh = "右手" if hand == "right" else "左手"
    goal = shared_goal or instruction
    if use_zh:
        context = (
            f"必须达成：{goal}\n"
            f"仅使用{hand_zh}完成（另一手保持静止）：{instruction}\n"
        )
        step_hint = "从「第一步：」开始，只写该手的动作步骤。"
        place_rules = place_release_augment_rules(use_zh=use_zh) if place_task else ""
        return (
            scene_intro
            + context
            + temporal_rules
            + ("\n" + place_rules if place_rules else "")
            + ("\n" + retry_rules if retry_rules else "")
            + f"\n{step_hint}\n"
            + lang_line
            + " 只输出描述正文。"
        )

    hand_name = "RIGHT" if hand == "right" else "LEFT"
    place_rules = place_release_augment_rules(use_zh=use_zh) if place_task else ""
    return (
        scene_intro
        + f"Must achieve: {goal}\n"
        + f"Only the {hand_name} hand acts (other hand stays still): {instruction}\n"
        + temporal_rules
        + ("\n" + place_rules if place_rules else "")
        + ("\n" + retry_rules if retry_rules else "")
        + '\nStart with "First," — describe only that hand\'s steps.\n'
        + f"{lang_line} Output description only."
    )


def build_desk_synth_video_prompt(
    instruction: str,
    caption: Optional[str] = None,
    augmented_desc: Optional[str] = None,
    hand: str = "right",
    use_visual_trajectory: bool = False,
) -> str:
    hand_lower = "right" if hand == "right" else "left"
    prompt_parts = []
    if caption:
        prompt_parts.append(caption)

    prompt_parts.append(DESK_SYNTH_CAMERA_SINGLE.format(hand=hand_lower))
    if use_visual_trajectory:
        prompt_parts.append(
            "The reference image shows colored hand trajectory overlays: cyan polyline with dots for the "
            "left hand, orange polyline with dots for the right hand. Green dot marks the trajectory end. "
            "During the video, each moving hand must closely follow its drawn path from the first waypoint "
            "to the last, avoiding obstacles and respecting the planned route on the desktop."
        )
    if augmented_desc:
        prompt_parts.append(sanitize_augmented_desc(augmented_desc))
    if use_visual_trajectory:
        traj_color = "orange" if hand == "right" else "cyan"
        prompt_parts.append(
            f"The {hand_lower} hand follows the {traj_color} trajectory overlay in the reference image."
        )
    if is_place_put_instruction(instruction):
        prompt_parts.append(place_release_video_clause())
    # prompt_parts.append(desk_synth_motion_tail(bimanual=False))
    return " ".join(prompt_parts)


def build_desk_synth_bimanual_video_prompt(
    left_instr: str,
    right_instr: str,
    task_description: str = "",
    caption: Optional[str] = None,
    augmented_desc: Optional[str] = None,
    use_visual_trajectory: bool = False,
) -> str:
    prompt_parts = []
    if caption:
        prompt_parts.append(caption)

    prompt_parts.append(DESK_SYNTH_CAMERA_BIMANUAL)
    if use_visual_trajectory:
        prompt_parts.append(
            "The reference image shows colored hand trajectory overlays: cyan polyline with dots for the "
            "left hand, orange polyline with dots for the right hand. Green dot marks each trajectory end. "
            "During the video, each moving hand must closely follow its drawn path from the first waypoint "
            "to the last, avoiding obstacles and respecting the planned route on the desktop."
        )
    if augmented_desc:
        prompt_parts.append(sanitize_augmented_desc(augmented_desc))
    if use_visual_trajectory:
        prompt_parts.append(
            "Left hand follows cyan trajectory; right hand follows orange trajectory in the reference image."
        )
    if is_place_put_instruction(
        " ".join(filter(None, [left_instr, right_instr, task_description]))
    ):
        prompt_parts.append(place_release_video_clause())
    # prompt_parts.append(desk_synth_motion_tail(bimanual=True))
    return " ".join(prompt_parts)

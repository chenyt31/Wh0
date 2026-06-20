"""
Shared utilities for decoupled video generation pipeline.
"""

import hashlib
import json
import fcntl
import os
import re
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image, ImageDraw


class CheckpointManager:
    """Checkpoint manager for fine-grained processed-item IDs (multi-process safe)."""

    def __init__(self, checkpoint_dir: str = "database/log", checkpoint_name: str = "processed_instructions.txt"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / checkpoint_name
        self.lock_file = self.checkpoint_dir / f"{checkpoint_name}.lock"
        if not self.checkpoint_file.exists():
            self.checkpoint_file.touch()

    @staticmethod
    def get_image_id(image_path: str) -> str:
        """Stable ID for a source image (image-first pipeline)."""
        return str(Path(image_path).resolve())

    @staticmethod
    def get_manifest_row_id(manifest_path: str, task_id: str) -> str:
        """Stable ID for one manifest row (synth or video manifest)."""
        return f"{Path(manifest_path).resolve()}:{task_id}"

    @staticmethod
    def get_video_frame_id(video_path: str, frame_index: int) -> str:
        """Stable ID for one sampled video frame (data_clean)."""
        return f"{Path(video_path).resolve()}:{int(frame_index)}"

    @staticmethod
    def get_long_horizon_step_id(run_dir: str, step: int) -> str:
        """Stable ID for one long-horizon chain step within a run."""
        return f"{Path(run_dir).resolve()}:step{int(step):02d}"

    @staticmethod
    def get_instr_first_id(assembled: Dict) -> str:
        """Stable ID for one instr-first assembled instruction."""
        stable = {
            "template_id": assembled.get("template_id"),
            "slots": assembled.get("slots"),
            "feasible": assembled.get("feasible"),
            "use_both_hands": assembled.get("use_both_hands"),
        }
        payload = json.dumps(stable, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def get_image_instr_id(image_path: str, instr_idx: int) -> str:
        key = f"{Path(image_path).resolve()}:{instr_idx}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    @staticmethod
    def get_instruction_id(instruction: Dict, jsonl_path: str = "") -> str:
        instr_copy = {k: v for k, v in instruction.items() if not k.startswith("_")}
        instr_str = json.dumps(instr_copy, sort_keys=True, ensure_ascii=False)
        if jsonl_path:
            instr_str = f"{Path(jsonl_path).name}:{instr_str}"
        return hashlib.md5(instr_str.encode("utf-8")).hexdigest()

    def load_processed_instructions(self) -> Set[str]:
        processed = set()
        if not self.checkpoint_file.exists():
            return processed
        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        processed.add(line)
        except Exception as e:
            print(f"[Checkpoint] Warning: Failed to load checkpoint file: {e}")
        return processed

    def mark_as_processed(self, instruction_id: str, gpu_id: int = 0) -> bool:
        try:
            with open(self.lock_file, "w") as lock_f:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
                try:
                    with open(self.checkpoint_file, "a", encoding="utf-8") as f:
                        f.write(f"{instruction_id}\n")
                        f.flush()
                        os.fsync(f.fileno())
                finally:
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
            return True
        except Exception as e:
            print(f"[GPU {gpu_id}] Warning: Failed to record checkpoint: {e}")
            return False

    def get_unprocessed_instructions(
        self, instructions: List[Dict], jsonl_path: str = ""
    ) -> Tuple[List[Dict], int]:
        processed = self.load_processed_instructions()
        if not processed:
            return instructions, 0
        unprocessed = []
        skipped = 0
        for instr in instructions:
            instr_id = self.get_instruction_id(instr, jsonl_path)
            if instr_id not in processed:
                unprocessed.append(instr)
            else:
                skipped += 1
        return unprocessed, skipped

    def is_processed(self, item_id: str) -> bool:
        return item_id in self.load_processed_instructions()

    def filter_unprocessed(
        self,
        items: List[Any],
        id_fn,
    ) -> Tuple[List[Any], int]:
        processed = self.load_processed_instructions()
        if not processed:
            return items, 0
        unprocessed: List[Any] = []
        skipped = 0
        for item in items:
            item_id = id_fn(item)
            if item_id not in processed:
                unprocessed.append(item)
            else:
                skipped += 1
        return unprocessed, skipped

    def reset(self) -> None:
        self.checkpoint_file.write_text("")

    def get_stats(self) -> Dict[str, int]:
        processed = self.load_processed_instructions()
        return {
            "total_processed": len(processed),
            "checkpoint_file": str(self.checkpoint_file),
        }


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_instruction_schema(instruction: Dict) -> Dict:
    """Normalize instruction fields (legacy vs v2 schema)."""
    if not isinstance(instruction, dict):
        return {"instruction": "", "nouns": [], "adjectives": []}

    normalized = dict(instruction)
    instr_text = normalized.get("instruction", "")
    if not instr_text:
        instr_text = normalized.get("instr", "")
    if instr_text is None:
        instr_text = ""
    normalized["instruction"] = str(instr_text)

    nouns = normalized.get("nouns")
    if nouns is None:
        nouns = normalized.get("noun", [])
    if isinstance(nouns, str):
        nouns = [nouns]
    elif not isinstance(nouns, list):
        nouns = [nouns] if nouns else []
    normalized["nouns"] = [str(n).strip() for n in nouns if str(n).strip()]

    adjectives = normalized.get("adjectives")
    if adjectives is None:
        adjectives = normalized.get("adjective", [])
    if isinstance(adjectives, str):
        adjectives = [adjectives]
    elif not isinstance(adjectives, list):
        adjectives = [adjectives] if adjectives else []

    normalized_adjectives = []
    for a in adjectives:
        if isinstance(a, list):
            cleaned = [str(x).strip() for x in a if str(x).strip()]
            normalized_adjectives.append(cleaned)
        else:
            normalized_adjectives.append("" if a is None else str(a).strip())
    normalized["adjectives"] = normalized_adjectives

    verbs = normalized.get("verbs")
    if verbs is None:
        verbs = normalized.get("verb", [])
    if isinstance(verbs, str):
        verbs = [verbs]
    elif not isinstance(verbs, list):
        verbs = [verbs] if verbs else []
    normalized["verbs"] = [str(v).strip().lower() for v in verbs if str(v).strip()]

    return normalized


_SPATIAL_PREP_PATTERN = (
    r"on(?:to)?|in(?:to|side)?|near|next to|beside|by|between|"
    r"to the left of|to the right of|above|below|against|at|over|under|"
    r"in front of|behind|around|top of"
)
_SINGLE_ARM_ALLOWED_VERBS = (
    "pick|grasp|place|push|pull|slide|open|close|rotate|turn|lift|lower|"
    "press|tap|move|align|insert|remove|unscrew|screw|flip|twist|unfold|fold"
)
_HYPHEN_COMPOUND_VERB_RE = re.compile(r"\b(pick|grasp)-then-place\b", re.IGNORECASE)
_PLACE_DESTINATION_RE = re.compile(
    rf"\bplace\b.+\b(?:{_SPATIAL_PREP_PATTERN})\b",
    re.IGNORECASE,
)
_SPATIAL_OR_DIRECTION_RE = re.compile(
    rf"\b(?:{_SPATIAL_PREP_PATTERN}|toward|towards|away from|closer to|farther from|"
    rf"to the left|to the right|leftward|rightward|forward|backward|aside|across)\b",
    re.IGNORECASE,
)
_ALLOWED_VERB_START_RE = re.compile(
    rf"^({_SINGLE_ARM_ALLOWED_VERBS})\b",
    re.IGNORECASE,
)


def is_valid_manipulation_instruction(instruction: str) -> bool:
    """Return True if instruction has complete, executable wording."""
    text = (instruction or "").strip()
    if len(text) < 8:
        return False
    if _HYPHEN_COMPOUND_VERB_RE.search(text):
        return False
    if not _ALLOWED_VERB_START_RE.match(text):
        return False
    if re.search(r"\bplace\b", text, re.IGNORECASE) and not _PLACE_DESTINATION_RE.search(text):
        return False
    if re.search(r"\bmove\b", text, re.IGNORECASE) and not _SPATIAL_OR_DIRECTION_RE.search(text):
        return False
    if (
        re.search(r"\b(?:push|pull|slide)\b", text, re.IGNORECASE)
        and not _SPATIAL_OR_DIRECTION_RE.search(text)
    ):
        return False
    return True


from utils.instruction_diversity import (  # noqa: F401
    DEFAULT_DIVERSITY_DB_PATH,
    backfill_manifests_to_db,
    record_manifest_entry_in_db,
)


def extract_primary_object_key(instruction: Dict) -> str:
    """Primary manipulated object (first noun), normalized for batch diversity checks."""
    item = normalize_instruction_schema(instruction)
    nouns = item.get("nouns") or []
    if nouns:
        return str(nouns[0]).strip().lower()
    return ""


def extract_bimanual_primary_object_key(instruction: Dict) -> str:
    """Primary shared object for bimanual batch diversity checks."""
    shared = (instruction.get("shared_object") or "").strip().lower()
    if shared:
        return shared
    left_nouns = instruction.get("left_nouns") or instruction.get("left_noun") or []
    if isinstance(left_nouns, str):
        left_nouns = [left_nouns]
    if left_nouns:
        return str(left_nouns[0]).strip().lower()
    return extract_primary_object_key(instruction)


def filter_valid_instructions(instructions: List[Dict]) -> List[Dict]:
    """Keep only instructions that pass validation; dedupe by instruction text."""
    seen = set()
    valid = []
    for item in instructions:
        text = item.get("instruction", "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        if is_valid_manipulation_instruction(text):
            seen.add(key)
            valid.append(item)
    return valid


def read_instructions_jsonl(jsonl_path: str, k: int = -1) -> List[Dict]:
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        print(f"JSONL file not found: {jsonl_path}")
        return []

    instructions = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if k != -1 and i >= k:
                break
            try:
                data = json.loads(line.strip())
                instructions.append(normalize_instruction_schema(data))
            except json.JSONDecodeError as e:
                print(f"Failed to parse line {i + 1}: {e}")
    print(f"Loaded {len(instructions)} instructions from {jsonl_path}")
    return instructions


def list_images(image_dir: str, recursive: bool = False) -> List[Path]:
    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"Image directory not found: {image_dir}")
        return []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    iterator = image_dir.rglob("*") if recursive else image_dir.iterdir()
    files = sorted(
        f for f in iterator
        if f.is_file() and f.suffix.lower() in extensions
    )
    return files


def parse_vl_instructions_json(text: str, max_count: Optional[int] = None) -> List[Dict]:
    """Parse Qwen VL JSON array output into normalized instruction dicts."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    raw_list = None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            raw_list = parsed
        elif isinstance(parsed, dict):
            raw_list = [parsed]
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end > start:
            try:
                raw_list = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

    if not raw_list:
        return []

    results = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        normalized = normalize_instruction_schema(item)
        if normalized.get("instruction"):
            results.append(normalized)
        if max_count is not None and len(results) >= max_count:
            break
    return results


def slow_manipulation_video_clause() -> str:
    """Wan I2V prompt fragment: slow motion with explicit grasp contact physics."""
    return (
        "The entire manipulation is slow, smooth, and deliberate — no sudden snaps, jumps, or teleporting. "
        "Approach the target object gradually with visible acceleration and deceleration. "
        "Grasping is the most important phase and must last visibly longer: fingers align first, "
        "then touch the object surface, then close slowly one joint at a time; show clear contact, "
        "slight object tilt or compression, and friction as grip force builds; the object lifts only "
        "after a secure grasp is established over multiple moments — never pop up instantly without "
        "contact or clear force transfer. Transport and place are also unhurried; release gently."
    )


def build_text_augmentation_prompt(
    instruction: str,
    hand: str = "right",
    *,
    mode: str = "single",
    left_instruction: str = "",
    right_instruction: str = "",
    shared_goal: str = "",
    shared_object: str = "",
    output_language: str = "zh",
) -> str:
    """
    Prompt for Qwen VL text augmentation: chronological video description only.
    The manipulation instruction is context for the model but must NOT appear in output.
    mode: "single" | "bimanual"
    """
    lang = (output_language or "zh").strip().lower()
    use_zh = lang in ("zh", "cn", "chinese", "中文")

    if use_zh:
        temporal_rules = (
            "写作规则（必须遵守）：\n"
            "1. 输出仅为「视频增强描述」，按时间顺序写画面里发生的事，不要复述任务指令原文。\n"
            "   禁止在输出中出现类似「拿起…放到…」「pick…then place…」等完整指令句。\n"
            "2. 每个动作或状态只写一次；已完成的动作后不得再写同一动作（例如松开物体只能写一次）。\n"
            "3. 用「第一步」「第二步」…分段叙述，顺序推进，最后一步之后只写静态结果。\n"
            "4. 整体节奏偏慢、匀速，禁止一闪完成或物体瞬移弹起。\n"
            "5. 抓取阶段最重要，必须拆成多步细写（至少 2–3 步）：缓慢对准 → 指腹/指节逐根接触物体 → "
            "可见施力（物体微倾、微移或轻微形变）→ 收拢握紧 → 确认抓牢后才缓慢抬起；"
            "禁止未体现接触受力就立刻离桌。\n"
            "6. 典型顺序：慢速接近 → 分步抓取（见上）→ 慢速移动（绕障）→ 轻放并松开（仅一次）→ 最终摆放 → 手静止。\n"
            "7. 一段连贯中文，不要 markdown，不要 JSON。"
        )
        lang_line = "请用中文输出。"
    else:
        temporal_rules = (
            "Writing rules (STRICT):\n"
            "1. Output is a video enhancement description only — chronological scene events.\n"
            "   Do NOT restate or quote the task instruction sentence in the output.\n"
            "2. Each action happens once; never repeat a completed action (e.g. release only once).\n"
            "3. Use First / Then / Next / Finally (or 第一步 if writing Chinese).\n"
            "4. Overall pace is slow and smooth; no snapping or instant object pop-up.\n"
            "5. Grasping is critical — split into multiple steps: slow alignment, finger-by-finger contact, "
            "visible force (object tilt/slip/compression), close grip, lift only after secure hold.\n"
            "6. Order: slow approach → detailed grasp steps → slow transport → gentle place/release → rest.\n"
            "7. One coherent paragraph; no markdown."
        )
        lang_line = "Output in English."

    if mode == "bimanual":
        combined = format_bimanual_instruction(left_instruction, right_instruction)
        context = (
            "（以下双手任务仅供你理解画面，不要写入输出）\n"
            f"{combined}\n"
            if use_zh
            else f"(Context only — do NOT copy into output)\nCooperative task: {combined}\n"
        )
        if shared_goal:
            context += f"{'协同目标' if use_zh else 'Shared goal'}: {shared_goal}\n"
        if shared_object:
            context += f"{'协作物体' if use_zh else 'Shared object'}: {shared_object}\n"
        opening = (
            "双手从画面下方伸入画面。人物使用双手协同完成动作："
            if use_zh
            else "Both hands enter from the bottom of the frame. The person uses both hands cooperatively: "
        )
        scene_line = "这是一张俯视桌面场景图。\n" if use_zh else "Top-down desktop scene image.\n"
        bimanual_example = ""
        if use_zh:
            bimanual_example = (
                "\n示例（结构参考，双手协同、抓取分步慢写）：\n"
                f"{opening}第一步：左手缓慢靠近物体一侧，右手在另一侧对准；"
                "第二步：双手指腹先后接触物体，手指逐根收拢，物体受力微倾；"
                "第三步：确认双手抓牢后同步缓缓抬起；第四步：…\n"
            )
        return (
            scene_line
            + context
            + temporal_rules
            + (
                f"\n输出必须以如下开场开头（照抄开场，然后接第一步）：\n{opening}\n"
                if use_zh
                else f"\nStart output with exactly this opening, then step 1:\n{opening}\n"
            )
            + bimanual_example
            + lang_line
            + (" 只输出描述正文，不要其他内容。" if use_zh else " Output description only.")
        )

    hand_zh = "右手" if hand == "right" else "左手"
    entry_zh = "右下角" if hand == "right" else "左下角"
    if use_zh:
        opening = f"{hand_zh}从画面{entry_zh}伸入画面。人物仅使用{hand_zh}完成动作："
        return (
            "这是一张俯视桌面场景图。\n"
            "（以下任务指令仅供你理解要做什么，不要写入输出）\n"
            f"指令：{instruction}\n"
            + temporal_rules
            + "\n输出必须以如下开场开头（照抄开场，然后直接写第一步，中间不要插入指令原文）：\n"
            f"{opening}第一步：…\n"
            "示例（结构参考，内容按图像与任务改写）：\n"
            f"{opening}第一步：{hand_zh}缓慢靠近浅蓝色书本；第二步：指腹接触书脊、手指逐根收拢，书本受力微倾；"
            "第三步：确认抓牢后缓缓抬起；…\n"
            + lang_line
            + " 只输出描述正文。"
        )

    hand_name = "RIGHT" if hand == "right" else "LEFT"
    entry_en = "bottom-right" if hand == "right" else "bottom-left"
    opening = (
        f"The {hand_name} hand enters from the {entry_en} of the frame. "
        f"Only the {hand_name} hand is used: "
    )
    return (
        "Top-down desktop scene image.\n"
        f"(Context only — do NOT copy into output) Task: {instruction}\n"
        f"{temporal_rules}\n"
        f"Start with exactly: \"{opening}First, ...\" — no instruction sentence after the colon.\n"
        f"{lang_line} Output description only."
    )


# Qwen3-VL 2D grounding uses relative coords in [0, 1000], not raw pixels.
# See: https://github.com/QwenLM/Qwen3-VL cookbooks/2d_grounding.ipynb
QWEN3_VL_COORD_MAX = 1000


def _strip_json_fence(text: str) -> str:
    text = (text or "").strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def qwen3_coord_to_pixel(x: float, y: float, width: int, height: int) -> Tuple[int, int]:
    """
    Map model coordinates to original-image pixels (Qwen3-VL cookbook plot_points).
    Priority: 0-1 fraction -> 0-1000 relative -> absolute pixel fallback.
    """
    ax, ay = abs(x), abs(y)
    if ax <= 1.0 and ay <= 1.0:
        px, py = x * width, y * height
    elif ax <= QWEN3_VL_COORD_MAX and ay <= QWEN3_VL_COORD_MAX:
        px = x / QWEN3_VL_COORD_MAX * width
        py = y / QWEN3_VL_COORD_MAX * height
    else:
        px, py = x, y
    xi = int(round(max(0, min(width - 1, px))))
    yi = int(round(max(0, min(height - 1, py))))
    return xi, yi


def _extract_rel_xy(item: Any) -> Optional[Tuple[float, float]]:
    """Read x,y from Qwen3 point_2d object, [x,y] list, or bbox_2d center."""
    if isinstance(item, dict):
        if "point_2d" in item:
            pt = item["point_2d"]
        elif "bbox_2d" in item:
            b = item["bbox_2d"]
            if not isinstance(b, (list, tuple)) or len(b) < 4:
                return None
            return (float(b[0]) + float(b[2])) / 2, (float(b[1]) + float(b[3])) / 2
        else:
            return None
    elif isinstance(item, (list, tuple)) and len(item) >= 2:
        pt = item
    else:
        return None
    try:
        return float(pt[0]), float(pt[1])
    except (TypeError, ValueError, IndexError):
        return None


def _parse_hand_point_list(
    items: Any,
    image_width: int,
    image_height: int,
) -> List[Tuple[int, int]]:
    if not isinstance(items, list):
        return []
    pixels: List[Tuple[int, int]] = []
    for item in items:
        rel = _extract_rel_xy(item)
        if rel is None:
            continue
        pixels.append(qwen3_coord_to_pixel(rel[0], rel[1], image_width, image_height))
    return pixels


def _load_trajectory_json(text: str) -> Any:
    text = _strip_json_fence(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


def parse_hand_trajectory_json(
    text: str,
    image_width: int,
    image_height: int,
    default_hand: str = "right",
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Parse Qwen3-VL grounding JSON into pixel trajectories on the original image.
    Supports official point_2d objects (0-1000 relative) and legacy [[x,y], ...] lists.
    """
    result: Dict[str, List[Tuple[int, int]]] = {"left_hand": [], "right_hand": []}
    parsed = _load_trajectory_json(text)
    if parsed is None:
        return result

    if isinstance(parsed, list):
        pts = _parse_hand_point_list(parsed, image_width, image_height)
        key = "right_hand" if default_hand == "right" else "left_hand"
        result[key] = pts
        return result

    if not isinstance(parsed, dict):
        return result

    for key, out_key in (("left_hand", "left_hand"), ("right_hand", "right_hand")):
        hand_items = parsed.get(key)
        if hand_items is None and key == "left_hand":
            hand_items = parsed.get("left")
        if hand_items is None and key == "right_hand":
            hand_items = parsed.get("right")
        result[out_key] = _parse_hand_point_list(hand_items, image_width, image_height)

    if not result["left_hand"] and not result["right_hand"]:
        for key, out_key in (("left", "left_hand"), ("right", "right_hand")):
            if key in parsed:
                result[out_key] = _parse_hand_point_list(
                    parsed[key], image_width, image_height
                )
    return result


def build_hand_trajectory_vl_prompt(
    instruction: str,
    image_width: int,
    image_height: int,
    num_points: int,
    hand: str = "right",
    left_instruction: str = "None",
    right_instruction: str = "",
    augmented_desc: Optional[str] = None,
) -> str:
    """Prompt Qwen3-VL for hand waypoints using official 0-1000 point_2d JSON format."""
    is_bimanual = hand == "bimanual"
    active = "right" if hand == "right" else "left"
    inactive = "left" if active == "right" else "right"
    active_name = active.upper()
    entry_hint = (
        "near the bottom-right edge of the frame (hand enters from bottom-right)"
        if active == "right"
        else "near the bottom-left edge of the frame (hand enters from bottom-left)"
    )
    example_start = [820, 920] if active == "right" else [180, 920]
    example_mid = [600, 650] if active == "right" else [400, 650]
    example_end = [450, 500] if active == "right" else [550, 500]
    lines = [
        "You are viewing a top-down desktop manipulation scene. Plan precise hand motion paths.",
        "",
        f"Manipulation task: {instruction}",
    ]
    if is_bimanual:
        lines.append(f"Left hand action: {left_instruction}")
        lines.append(f"Right hand action: {right_instruction}")
        lines.append(
            "Both hands cooperate on one shared task. Plan a separate trajectory per moving hand."
        )
    elif hand in ("left", "right"):
        lines.append(f"Only the {active_name} hand moves; the other hand stays idle (empty array).")

    if augmented_desc and augmented_desc.strip():
        scene_guidance = [
            "",
            "Scene understanding from prior visual-language analysis (USE THIS for planning):",
            augmented_desc.strip(),
            "",
            "Use the scene understanding above together with the image to:",
        ]
        if is_bimanual:
            scene_guidance.extend([
                "- Left hand: waypoint_0 near bottom-left entry; route around obstacles; "
                "final waypoint at the left-hand target.",
                "- Right hand: waypoint_0 near bottom-right entry; route around obstacles; "
                "final waypoint at the right-hand target.",
                "- Keep both polylines on collision-free paths; hands must not cross through objects.",
            ])
        else:
            scene_guidance.extend([
                "- Place waypoint_0 at the hand rest/entry area (" + entry_hint + ").",
                "- Route intermediate waypoints around obstacles (laptop, monitor, bottles, books, hub, etc.).",
                "- Place the final waypoint at the grasp/place target described above.",
                "- Keep every waypoint on collision-free paths over the desk, not through objects.",
            ])
        lines.extend(scene_guidance)
    else:
        if is_bimanual:
            lines.extend([
                "",
                "Study the image. Plan detours for each moving hand from its bottom-corner entry "
                "to its target object.",
            ])
        else:
            lines.extend([
                "",
                "Study the image carefully. Identify obstacle objects on the desk and plan detours.",
                f"- waypoint_0: hand entry/rest ({entry_hint}).",
                "- Middle waypoints: arc around obstacles, staying above the desk surface in image space.",
                "- Final waypoint: target object for grasp or place.",
            ])

    lines.extend([
        "",
        "Trajectory rules:",
        f"- Output exactly {num_points} waypoints per moving hand, in temporal order.",
        "- Labels: waypoint_0 (approach), then waypoint_1..N-2 (via/detour), waypoint_N-1 (target).",
        "- Space waypoints so each polyline visibly avoids blocking objects.",
        "",
        "IMPORTANT — Qwen3-VL coordinate rules:",
        f"- Use relative coordinates from 0 to {QWEN3_VL_COORD_MAX} (NOT pixel coordinates).",
        "- Origin (0,0) is top-left; x increases right; y increases down.",
        '- Each waypoint: {"point_2d": [x, y], "label": "waypoint_k"}.',
        "- Report JSON only (no markdown).",
    ])
    if is_bimanual:
        lines.extend([
            "",
            "Output format example:",
            "{",
            f'  "left_hand": [{{"point_2d": [180, 920], "label": "waypoint_0"}}, ...],',
            f'  "right_hand": [{{"point_2d": [820, 920], "label": "waypoint_0"}}, ...]',
            f"}}  // exactly {num_points} points per hand that moves; [] for an idle hand",
        ])
    else:
        lines.extend([
            "",
            "Output format example:",
            "{",
            f'  "{active}_hand": [',
            f'    {{"point_2d": [{example_start[0]}, {example_start[1]}], "label": "waypoint_0"}},',
            f'    {{"point_2d": [{example_mid[0]}, {example_mid[1]}], "label": "waypoint_1"}},',
            f'    {{"point_2d": [{example_end[0]}, {example_end[1]}], "label": "waypoint_2"}}',
            f"    // ... exactly {num_points} total",
            "  ],",
            f'  "{inactive}_hand": []',
            "}",
            f'For the idle {inactive} hand output [].',
        ])
    return "\n".join(lines)


def build_bimanual_video_prompt(
    left_instr: str,
    right_instr: str,
    task_description: str = "",
    caption: Optional[str] = None,
    augmented_desc: Optional[str] = None,
    use_visual_trajectory: bool = False,
) -> str:
    """Build Wan video prompt for coordinated two-hand manipulation."""
    combined = format_bimanual_instruction(left_instr, right_instr)
    prompt_parts = []
    if caption:
        prompt_parts.append(caption)
    prompt_parts.append(
        "4k high definition, fixed camera, first-person top-down view, showing both human hands "
        "and objects on the desktop. The left hand extends from the bottom left corner of the frame. "
        "The right hand extends from the bottom right corner of the frame."
    )
    if use_visual_trajectory:
        prompt_parts.append(
            "The reference image shows colored hand trajectory overlays: cyan polyline with dots for the "
            "left hand, orange polyline with dots for the right hand. Green dot marks each trajectory end. "
            "During the video, each moving hand must closely follow its drawn path from the first waypoint "
            "to the last, avoiding obstacles and respecting the planned route on the desktop."
        )
    goal_clause = f"to {task_description} " if task_description else ""
    prompt_parts.append(
        f"A person performs a coordinated two-hand manipulation task {goal_clause}in the frame: "
        f"{combined} Both hands must cooperate on the same task, not perform unrelated actions."
    )
    if augmented_desc:
        prompt_parts.append(augmented_desc)
    prompt_parts.append(slow_manipulation_video_clause())
    if use_visual_trajectory:
        prompt_parts.append(
            "The left hand must follow the cyan trajectory overlay; the right hand must follow the "
            "orange trajectory overlay shown in the reference image from start to end."
        )
    prompt_parts.append(
        "After the shared task is completed, both hands stop, with no extra movements."
    )
    return " ".join(prompt_parts)


def draw_hand_trajectories_on_image(
    image: Image.Image,
    trajectories: Dict[str, List[Tuple[int, int]]],
) -> Image.Image:
    """Overlay left (cyan) and right (orange) polylines with waypoint markers."""
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out, "RGBA")
    styles = {
        "left_hand": {"color": (0, 206, 209, 220), "width": 4},
        "right_hand": {"color": (255, 102, 0, 220), "width": 4},
    }
    for hand_key, style in styles.items():
        pts = trajectories.get(hand_key) or []
        if len(pts) < 1:
            continue
        if len(pts) >= 2:
            draw.line(pts, fill=style["color"], width=style["width"])
        r = max(6, min(out.size) // 80)
        for i, (x, y) in enumerate(pts):
            fill = style["color"]
            if i == 0:
                fill = tuple(min(255, c + 30) for c in fill[:3]) + (fill[3],)
            elif i == len(pts) - 1:
                fill = (50, 205, 50, 240)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=(255, 255, 255, 200))
    return out


def scene_aug_output_path(
    scene_aug_dir: str,
    image_path: str,
    instr_idx: int,
    suffix: str = ".png",
) -> Path:
    stem = Path(image_path).stem
    out_dir = Path(scene_aug_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}_instr{instr_idx:02d}{suffix}"


def build_video_prompt(
    instruction: str,
    caption: Optional[str] = None,
    augmented_desc: Optional[str] = None,
    hand: str = "right",
    use_visual_trajectory: bool = False,
) -> str:
    prompt_parts = []
    if caption:
        prompt_parts.append(caption)

    hand_name = "RIGHT" if hand == "right" else "LEFT"
    hand_lower = "right" if hand == "right" else "left"
    prompt_parts.append(
        f"4k high definition, fixed camera, first-person top-down view, showing human {hand_lower} hand "
        f"and objects on the desktop. The {hand_lower} hand extends from the bottom right corner of the frame."
    )
    if use_visual_trajectory:
        prompt_parts.append(
            "The reference image shows colored hand trajectory overlays: cyan polyline with dots for the "
            "left hand, orange polyline with dots for the right hand. Green dot marks the trajectory end. "
            "During the video, each moving hand must closely follow its drawn path from the first waypoint "
            "to the last, avoiding obstacles and respecting the planned route on the desktop."
        )
    prompt_parts.append(
        f"A person's {hand_name} hand performs the following action in the frame: {instruction}. "
        f"Only the {hand_lower} hand is used to complete this task."
    )
    if augmented_desc:
        prompt_parts.append(augmented_desc)
    prompt_parts.append(slow_manipulation_video_clause())
    if use_visual_trajectory:
        traj_color = "orange" if hand == "right" else "cyan"
        prompt_parts.append(
            f"The {hand_lower} hand motion must closely follow the {traj_color} trajectory overlay "
            "shown in the reference image from start to end."
        )
    prompt_parts.append("After the action is completed, the hands stop, with no extra movements.")
    return " ".join(prompt_parts)


def resize_image_lanczos(image: Image.Image, width: int, height: int) -> Image.Image:
    """Resize to exact (width, height) with LANCZOS."""
    if image.size == (width, height):
        return image
    return image.resize((width, height), Image.Resampling.LANCZOS)


def ensure_working_image_file(
    src_path: str,
    out_path: Path,
    width: int,
    height: int,
    *,
    jpeg_quality: int = 95,
) -> str:
    """Save a fixed-size working copy when the source differs; return usable path."""
    src = Path(src_path)
    if width <= 0 or height <= 0:
        return str(src.resolve())
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        if im.size == (width, height):
            if not out_path.exists():
                im.save(out_path, quality=jpeg_quality)
            return str(out_path.resolve())
        resized = resize_image_lanczos(im, width, height)
        resized.save(out_path, quality=jpeg_quality)
    return str(out_path.resolve())


def prepare_i2v_input_image(
    image: Image.Image,
    width: int,
    height: int,
    *,
    prep_width: int = 0,
    prep_height: int = 0,
) -> Image.Image:
    """Downscale large sources in two LANCZOS steps to reduce blur from one-shot shrink."""
    img = image.convert("RGB")
    pw = prep_width or width
    ph = prep_height or height
    if (
        prep_width > 0
        and prep_height > 0
        and max(img.size) > max(pw, ph)
        and img.size != (pw, ph)
    ):
        img = resize_image_lanczos(img, pw, ph)
    if img.size != (width, height):
        img = resize_image_lanczos(img, width, height)
    return img


def resolve_manifest_image_path(entry: Dict) -> str:
    """Prefer scene_aug image for video generation when visual augmentation was applied."""
    aug = (entry.get("scene_aug_path") or "").strip()
    if entry.get("use_visual_aug") and aug:
        return aug
    for key in ("image_path", "edited_image_path", "desktop_image_path"):
        raw = (entry.get(key) or "").strip()
        if raw:
            return raw
    return ""


def _ensure_period(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    return text if text.endswith(".") else text + "."


def format_bimanual_instruction(left_instr: str, right_instr: str) -> str:
    """Canonical per-hand instruction line: Left hand: ... Right hand: ..."""
    left = left_instr.strip()
    right = right_instr.strip()
    if left.lower() in {"none", "none."}:
        left = "None"
    else:
        left = _ensure_period(left).rstrip(".")
    if right.lower() in {"none", "none."}:
        right = "None"
    else:
        right = _ensure_period(right).rstrip(".")
    return f"Left hand: {left}. Right hand: {right}."


def format_single_arm_manifest_fields(
    instr_text: str,
    augmented_desc: str,
    hand: str,
) -> Tuple[str, str, str, str, str]:
    """
    Returns (instruction, task_description, left_instruction, right_instruction, augmented_text).
    instruction: per-hand lines; task_description: short action goal only.
    """
    goal = (instr_text or "").strip().rstrip(".")
    final_augmented = augmented_desc or ""
    if hand == "left":
        if final_augmented:
            for old, new in [
                ("right hand", "left hand"),
                ("Right hand", "Left hand"),
                ("RIGHT hand", "LEFT hand"),
                ("RIGHT HAND", "LEFT HAND"),
            ]:
                final_augmented = final_augmented.replace(old, new)
        left_instr, right_instr = goal, "None"
    else:
        left_instr, right_instr = "None", goal
    instruction = format_bimanual_instruction(left_instr, right_instr)
    return instruction, goal, left_instr, right_instr, final_augmented


def format_task_description(instr_text: str, augmented_desc: str, hand: str) -> Tuple[str, str]:
    """Backward-compatible wrapper: returns (instruction, augmented_text)."""
    instruction, _, _, _, final_augmented = format_single_arm_manifest_fields(
        instr_text, augmented_desc, hand
    )
    return instruction, final_augmented


def _parse_hand_instructions_from_combined(combined: str) -> Tuple[str, str]:
    left_instr, right_instr = "", ""
    left_match = re.search(
        r"left hand:\s*(.+?)(?=\s*right hand:|$)",
        combined,
        re.IGNORECASE | re.DOTALL,
    )
    right_match = re.search(
        r"right hand:\s*(.+?)$",
        combined,
        re.IGNORECASE | re.DOTALL,
    )
    if left_match:
        left_instr = left_match.group(1).strip().rstrip(".")
    if right_match:
        right_instr = right_match.group(1).strip().rstrip(".")
    return left_instr, right_instr


def normalize_task_fields(record: Dict) -> Dict:
    """Fill unified task fields from legacy layouts (eval / old task.json)."""
    out = {
        "task_id": record.get("task_id", ""),
        "hand": (record.get("hand") or "").strip().lower(),
        "instruction": (record.get("instruction") or "").strip(),
        "task_description": (record.get("task_description") or "").strip(),
        "left_instruction": (record.get("left_instruction") or "").strip(),
        "right_instruction": (record.get("right_instruction") or "").strip(),
        "shared_object": (record.get("shared_object") or "").strip(),
        "coordination_type": (record.get("coordination_type") or "").strip(),
        "augmented_text": (record.get("augmented_text") or "").strip(),
    }
    hand = out["hand"]
    inst = out["instruction"]
    task_desc = out["task_description"]

    # Legacy: goal duplicated inside task_description before instruction
    if hand == "bimanual" and inst and task_desc and inst in task_desc:
        prefix = task_desc.replace(inst, "").strip()
        if prefix:
            out["task_description"] = prefix

    # Legacy single-arm: hand lines stored in task_description only
    if hand in ("left", "right") and not inst and task_desc:
        low = task_desc.lower()
        if "left hand:" in low or "right hand:" in low:
            out["instruction"] = task_desc
            left_i, right_i = _parse_hand_instructions_from_combined(task_desc)
            active = left_i if hand == "left" else right_i
            if active.lower() not in ("none", ""):
                out["task_description"] = active

    # Legacy bimanual: combined string in task_description
    if hand == "bimanual" and not inst and "left hand:" in task_desc.lower():
        out["instruction"] = task_desc
        if out["task_description"] == task_desc:
            out["task_description"] = ""

    if not out["left_instruction"] and out["instruction"]:
        out["left_instruction"], out["right_instruction"] = _parse_hand_instructions_from_combined(
            out["instruction"]
        )

    if hand in ("left", "right") and not out["left_instruction"]:
        if hand == "left":
            out["left_instruction"] = out["task_description"]
            out["right_instruction"] = "None"
        else:
            out["left_instruction"] = "None"
            out["right_instruction"] = out["task_description"]

    # Single-arm: short instruction field -> canonical hand lines + goal-only task_description
    if hand in ("left", "right") and out.get("instruction"):
        if "left hand:" not in out["instruction"].lower():
            goal = out["instruction"]
            if (
                out["task_description"]
                and "left hand:" in out["task_description"].lower()
            ):
                left_i, right_i = _parse_hand_instructions_from_combined(
                    out["task_description"]
                )
                goal = left_i if hand == "left" else right_i
                if goal.lower() in ("none", ""):
                    goal = out["instruction"]
            (
                combined,
                task_goal,
                left_i,
                right_i,
                _,
            ) = format_single_arm_manifest_fields(
                goal, out.get("augmented_text", ""), hand
            )
            out["instruction"] = combined
            out["task_description"] = task_goal
            out["left_instruction"] = left_i
            out["right_instruction"] = right_i

    # shared_goal / task_summary -> task_description
    if not out["task_description"]:
        legacy_goal = (
            record.get("shared_goal") or record.get("task_summary") or ""
        ).strip()
        if legacy_goal:
            out["task_description"] = legacy_goal

    return out


def rollout_task_id(base_task_id: str, rollout_idx: int, num_rollouts: int) -> str:
    if num_rollouts <= 1:
        return base_task_id
    return f"{base_task_id}_roll{rollout_idx:02d}"


def compute_rollout_seed(base_seed: int, rollout_idx: int, seed_stride: int) -> int:
    return int(base_seed) + rollout_idx * int(seed_stride)


def expand_manifest_rollouts(entries: List[Dict], config: Dict) -> Tuple[List[Dict], int]:
    """
    Duplicate each manifest row into rollouts_per_task jobs with distinct seeds
    (and optional sigma_shift / cfg_scale offsets).
    """
    task_cfg = config.get("task", {}) or {}
    rollout_cfg = config.get("rollout", {}) or {}
    video_cfg = config.get("video", {}) or {}
    n = max(1, int(task_cfg.get("rollouts_per_task", 1)))
    if n <= 1:
        return list(entries), 1

    seed_stride = int(rollout_cfg.get("seed_stride", 100_007))
    seed_mode = str(rollout_cfg.get("seed_mode", "stride")).lower()
    sigma_offsets = rollout_cfg.get("sigma_shift_offsets")
    cfg_offsets = rollout_cfg.get("cfg_scale_offsets")
    base_sigma = float(video_cfg.get("video_sigma_shift", 5.0))
    base_cfg = float(video_cfg.get("video_cfg_scale", 1.0))
    rand_device = rollout_cfg.get("rand_device") or video_cfg.get("rand_device")

    expanded: List[Dict] = []
    for entry in entries:
        base_id = entry.get("task_id") or ""
        base_seed = int(entry.get("seed", 0))
        for r in range(n):
            e = dict(entry)
            e["base_task_id"] = base_id
            e["rollout_idx"] = r
            e["task_id"] = rollout_task_id(base_id, r, n)
            if seed_mode == "hash":
                import zlib

                h = zlib.crc32(f"{base_id}:{r}".encode()) & 0xFFFFFFFF
                e["seed"] = (base_seed + h) % (2**31 - 1)
            else:
                e["seed"] = compute_rollout_seed(base_seed, r, seed_stride)
            if sigma_offsets is not None:
                e["sigma_shift_override"] = base_sigma + float(
                    sigma_offsets[r % len(sigma_offsets)]
                )
            if cfg_offsets is not None:
                e["cfg_scale_override"] = base_cfg + float(cfg_offsets[r % len(cfg_offsets)])
            if rand_device:
                e["rand_device"] = rand_device
            expanded.append(e)
    return expanded, n


def resolve_rollout_generation_params(
    entry: Dict,
    video_cfg: Dict,
    rollout_cfg: Optional[Dict] = None,
) -> Dict:
    rollout_cfg = rollout_cfg or {}
    sigma = float(
        entry.get(
            "sigma_shift_override",
            video_cfg.get("video_sigma_shift", 5.0),
        )
    )
    cfg = float(
        entry.get(
            "cfg_scale_override",
            video_cfg.get("video_cfg_scale", 1.0),
        )
    )
    rand_device = (
        entry.get("rand_device")
        or rollout_cfg.get("rand_device")
        or video_cfg.get("rand_device", "cuda")
    )
    return {
        "seed": int(entry.get("seed", 0)),
        "sigma_shift": sigma,
        "cfg_scale": cfg,
        "rand_device": str(rand_device),
    }


def build_task_json_from_entry(entry: Dict) -> Dict:
    """Unified fields written to tasks/{task_id}.json after video generation."""
    base = {
        "task_id": entry.get("task_id", ""),
        "base_task_id": entry.get("base_task_id", entry.get("task_id", "")),
        "rollout_idx": entry.get("rollout_idx"),
        "generation_seed": entry.get("seed"),
        "hand": entry.get("hand", ""),
        "instruction": entry.get("instruction", ""),
        "task_description": entry.get("task_description", ""),
        "left_instruction": entry.get("left_instruction", ""),
        "right_instruction": entry.get("right_instruction", ""),
        "shared_object": entry.get("shared_object", ""),
        "coordination_type": entry.get("coordination_type", ""),
        "augmented_text": entry.get("augmented_text", ""),
        "use_visual_aug": entry.get("use_visual_aug", False),
        "scene_aug_path": entry.get("scene_aug_path", ""),
        "hand_trajectories": entry.get("hand_trajectories", {}),
    }
    return normalize_task_fields(base)


def resolve_eval_prompt_text(
    dimension: str,
    task_info: Optional[Dict] = None,
    manifest_entry: Optional[Dict] = None,
) -> str:
    """
    Build evaluator [PROMPT] text per dimension.
    - text_alignment: augmented_text (video generation augmentation)
    - visual_quality / motion_quality: task_description + instruction only
    """
    merged: Dict = {}
    if manifest_entry:
        merged.update(manifest_entry)
    if task_info:
        for k, v in task_info.items():
            if v or k not in merged:
                merged[k] = v
    ctx = normalize_task_fields(merged)

    if dimension == "text_alignment":
        aug = ctx.get("augmented_text", "")
        if aug:
            return aug
        parts = []
        if ctx.get("task_description"):
            parts.append(f"Task goal: {ctx['task_description']}")
        if ctx.get("instruction"):
            parts.append(f"Hand actions: {ctx['instruction']}")
        return "\n".join(parts)

    parts = []
    if ctx.get("task_description"):
        parts.append(f"Task goal: {ctx['task_description']}")
    if ctx.get("instruction"):
        parts.append(f"Hand actions: {ctx['instruction']}")
    return "\n".join(parts)


DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

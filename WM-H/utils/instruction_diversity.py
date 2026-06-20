"""
Unified instruction diversity database helpers (video + text pipelines).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from utils.word_database import WordDatabase

DEFAULT_DIVERSITY_DB_PATH = "database/language/instruction_diversity.db"


def record_manifest_entry_in_db(entry: Dict, mode: str, db_path: str) -> bool:
    """Persist one manifest entry into the unified instruction diversity database."""
    instruction = (
        entry.get("task_description")
        or entry.get("instruction")
        or ""
    ).strip()
    if not instruction:
        return False

    nouns = entry.get("nouns") or []
    adjectives = entry.get("adjectives") or []
    verbs = entry.get("verbs") or []
    if not verbs:
        verbs = WordDatabase._verbs_from_instruction_text(instruction)

    primary = (
        entry.get("primary_object")
        or entry.get("shared_object")
        or ""
    )
    if not primary and nouns:
        primary = str(nouns[0]).strip().lower()

    with WordDatabase(db_path) as db:
        return db.add_instruction_record(
            instruction,
            mode,
            verbs=verbs,
            nouns=nouns,
            adjectives=adjectives,
            primary_object=primary,
            hand=entry.get("hand", ""),
            coordination_type=entry.get("coordination_type", ""),
            shared_object=entry.get("shared_object", ""),
            source_image=entry.get("image_name") or entry.get("image_path", ""),
            task_id=entry.get("task_id", ""),
            skip_if_exists=True,
        )


def backfill_manifests_to_db(
    manifest_dir: str,
    db_path: str,
    *,
    patterns: Optional[List[str]] = None,
) -> Dict[str, int]:
    """Import existing manifest JSONL files into the diversity database."""
    root = Path(manifest_dir)
    if not root.is_dir():
        return {"files": 0, "lines": 0, "inserted": 0}

    globs = patterns or ["manifest_*.jsonl", "bimanual_manifest_*.jsonl"]
    files = []
    for pat in globs:
        files.extend(sorted(root.glob(pat)))
    files = sorted(set(files))

    stats = {"files": len(files), "lines": 0, "inserted": 0}
    with WordDatabase(db_path) as db:
        for path in files:
            mode = (
                "video_bimanual"
                if path.name.startswith("bimanual_")
                else "video_single"
            )
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    stats["lines"] += 1
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    instruction = (
                        entry.get("task_description")
                        or entry.get("instruction")
                        or ""
                    ).strip()
                    if not instruction:
                        continue
                    nouns = entry.get("nouns") or []
                    verbs = entry.get("verbs") or WordDatabase._verbs_from_instruction_text(
                        instruction
                    )
                    primary = entry.get("primary_object") or entry.get("shared_object") or (
                        str(nouns[0]).lower() if nouns else ""
                    )
                    if db.add_instruction_record(
                        instruction,
                        mode,
                        verbs=verbs,
                        nouns=nouns,
                        adjectives=entry.get("adjectives") or [],
                        primary_object=primary,
                        hand=entry.get("hand", ""),
                        coordination_type=entry.get("coordination_type", ""),
                        shared_object=entry.get("shared_object", ""),
                        source_image=entry.get("image_name")
                        or entry.get("image_path", ""),
                        task_id=entry.get("task_id", ""),
                        skip_if_exists=True,
                    ):
                        stats["inserted"] += 1
    return stats

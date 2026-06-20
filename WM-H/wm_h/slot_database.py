"""Per-slot word pools with usage tracking for balanced instruction assembly."""

from __future__ import annotations

import random
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class SlotDatabase:
    """
    Each slot name maps to an independent SQLite DB:
      {base_path}/slot_{name}.db

    Tables:
      words(word, usage_count)
      instructions(instruction_normalized)
    """

    def __init__(
        self,
        base_path: str,
        slots_cfg: Dict[str, dict],
        *,
        sample_power: float = 1.5,
        max_usage_global: int = 50,
        max_usage_per_slot: Optional[Dict[str, int]] = None,
    ):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.slots_cfg = slots_cfg
        self.sample_power = sample_power
        self._max_usage_global = max_usage_global
        self._max_usage: Dict[str, int] = {
            name: cfg.get("max_usage", max_usage_global)
            for name, cfg in slots_cfg.items()
        }
        self._verb_types = frozenset({"verb", "v"})
        if max_usage_per_slot:
            self._max_usage.update(max_usage_per_slot)
        self._conns: Dict[str, sqlite3.Connection] = {}
        self._lock = threading.Lock()
        for name in slots_cfg:
            self._init_slot(name)

    def _db_path(self, slot_name: str) -> Path:
        return self.base_path / f"slot_{slot_name}.db"

    def _get_conn(self, slot_name: str) -> sqlite3.Connection:
        if slot_name not in self._conns:
            conn = sqlite3.connect(str(self._db_path(slot_name)), timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            self._conns[slot_name] = conn
        return self._conns[slot_name]

    def _init_slot(self, slot_name: str) -> None:
        conn = self._get_conn(slot_name)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS words (
                word TEXT PRIMARY KEY,
                usage_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS instructions (
                instruction_normalized TEXT PRIMARY KEY
            )
            """
        )
        conn.commit()

    def close(self) -> None:
        for conn in self._conns.values():
            conn.close()
        self._conns.clear()

    def get_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for name in self.slots_cfg:
            row = self._get_conn(name).execute("SELECT COUNT(*) FROM words").fetchone()
            counts[name] = int(row[0]) if row else 0
        return counts

    def get_all_words(self, slot_name: str) -> Set[str]:
        rows = self._get_conn(slot_name).execute("SELECT word FROM words").fetchall()
        return {r[0] for r in rows}

    def seed_if_empty(self, slot_name: str, words: List[str]) -> int:
        existing = self.get_all_words(slot_name)
        if existing:
            return 0
        return self.add_words(slot_name, words)

    def seed_to_min(
        self, slot_name: str, min_count: int, pool: List[str]
    ) -> int:
        """Fill slot from fallback pool until min_count (no LLM)."""
        count = self.get_counts().get(slot_name, 0)
        if count >= min_count or not pool:
            return 0
        existing = self.get_all_words(slot_name)
        shuffled = list(pool)
        random.shuffle(shuffled)
        to_add: List[str] = []
        for w in shuffled:
            if count + len(to_add) >= min_count:
                break
            wl = w.strip().lower()
            if wl and wl not in existing:
                to_add.append(wl)
                existing.add(wl)
        return self.add_words(slot_name, to_add)

    def add_words(self, slot_name: str, words: List[str]) -> int:
        if not words:
            return 0
        conn = self._get_conn(slot_name)
        added = 0
        with self._lock:
            for w in words:
                word = w.strip().lower()
                if not word:
                    continue
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO words (word, usage_count) VALUES (?, 0)",
                        (word,),
                    )
                    if conn.total_changes:
                        added += 1
                except sqlite3.Error:
                    continue
            conn.commit()
        return added

    def instruction_exists(self, instruction: str) -> bool:
        norm = " ".join(instruction.lower().split())
        conn = self._get_conn(next(iter(self.slots_cfg)))
        row = conn.execute(
            "SELECT 1 FROM instructions WHERE instruction_normalized = ?",
            (norm,),
        ).fetchone()
        return row is not None

    def add_instruction(self, instruction: str) -> None:
        norm = " ".join(instruction.lower().split())
        conn = self._get_conn(next(iter(self.slots_cfg)))
        with self._lock:
            conn.execute(
                "INSERT OR IGNORE INTO instructions (instruction_normalized) VALUES (?)",
                (norm,),
            )
            conn.commit()

    def is_verb_slot(self, slot_name: str) -> bool:
        t = str(self.slots_cfg.get(slot_name, {}).get("type", "")).lower()
        return t in self._verb_types

    def effective_max_usage(self, slot_name: str) -> int:
        """Non-verb slots cap at max_usage; verb slots are never capped."""
        if self.is_verb_slot(slot_name):
            return 2**31 - 1
        return self._max_usage.get(slot_name, self._max_usage_global)

    def increment_usage(self, samples: Dict[str, str]) -> None:
        with self._lock:
            for slot_name, word in samples.items():
                if not word or slot_name not in self.slots_cfg:
                    continue
                conn = self._get_conn(slot_name)
                conn.execute(
                    "UPDATE words SET usage_count = usage_count + 1 WHERE word = ?",
                    (word.strip().lower(),),
                )
                conn.commit()

    def sample_low_freq(self, slot_name: str, n: int) -> List[str]:
        """Sample n words with low usage_count; weight ~ 1/(usage+1)^power."""
        if n <= 0:
            return []
        max_uc = self.effective_max_usage(slot_name)
        rows = self._get_conn(slot_name).execute(
            "SELECT word, usage_count FROM words WHERE usage_count < ?",
            (max_uc,),
        ).fetchall()
        if not rows:
            return []

        import random

        picks: List[str] = []
        pool = list(rows)
        while pool and len(picks) < n:
            weights = [
                1.0 / ((int(uc) + 1.0) ** self.sample_power)
                for _, uc in pool
            ]
            idx = random.choices(range(len(pool)), weights=weights, k=1)[0]
            word, _ = pool.pop(idx)
            picks.append(word)
        return picks

    def min_usage(self, slot_name: str) -> Optional[int]:
        row = self._get_conn(slot_name).execute(
            "SELECT MIN(usage_count) FROM words"
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return int(row[0])

    def max_usage(self, slot_name: str) -> Optional[int]:
        row = self._get_conn(slot_name).execute(
            "SELECT MAX(usage_count) FROM words"
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return int(row[0])

    def slots_at_usage_cap(self) -> List[str]:
        capped: List[str] = []
        for name in self.slots_cfg:
            if self.is_verb_slot(name):
                continue
            mx = self.max_usage(name)
            cap = self.effective_max_usage(name)
            if mx is not None and mx >= cap:
                capped.append(name)
        return capped

"""
Word Frequency Database Manager
维护动词、名词、形容词的词频统计
"""

import json
import math
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class WordDatabase:
    """管理词频数据库"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=30000")
        self._create_tables()
    
    def _create_tables(self):
        """创建数据库表"""
        cursor = self.conn.cursor()
        
        # 词频表 - 使用REAL类型支持浮点数增量
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS word_frequency (
                word TEXT PRIMARY KEY,
                word_type TEXT NOT NULL,
                frequency REAL DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 指令历史表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS instruction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instruction TEXT NOT NULL,
                instruction_normalized TEXT,
                verbs TEXT,
                nouns TEXT,
                adjectives TEXT,
                mode TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 为instruction字段创建索引以加快重复检查
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_instruction 
            ON instruction_history(instruction)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_instruction_normalized 
            ON instruction_history(instruction_normalized)
        """)

        self._ensure_instruction_columns(cursor)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_instruction_mode
            ON instruction_history(mode)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_instruction_primary_object
            ON instruction_history(primary_object)
        """)

        self.conn.commit()

    def _ensure_instruction_columns(self, cursor) -> None:
        """Migrate instruction_history with diversity dimension columns."""
        cursor.execute("PRAGMA table_info(instruction_history)")
        existing = {row[1] for row in cursor.fetchall()}
        migrations = [
            ("primary_object", "TEXT DEFAULT ''"),
            ("hand", "TEXT DEFAULT ''"),
            ("coordination_type", "TEXT DEFAULT ''"),
            ("shared_object", "TEXT DEFAULT ''"),
            ("source_image", "TEXT DEFAULT ''"),
            ("task_id", "TEXT DEFAULT ''"),
        ]
        for col, col_type in migrations:
            if col not in existing:
                cursor.execute(
                    f"ALTER TABLE instruction_history ADD COLUMN {col} {col_type}"
                )
    
    def initialize_with_seeds(self, seed_verbs: List[str], 
                             seed_nouns: List[str], 
                             seed_adjectives: List[str]):
        """用种子词初始化数据库"""
        cursor = self.conn.cursor()
        
        for verb in seed_verbs:
            cursor.execute(
                "INSERT OR IGNORE INTO word_frequency (word, word_type, frequency) VALUES (?, ?, ?)",
                (verb.lower(), 'verb', 1)  # 初始频率设为1，避免0频率导致的问题
            )
        
        for noun in seed_nouns:
            cursor.execute(
                "INSERT OR IGNORE INTO word_frequency (word, word_type, frequency) VALUES (?, ?, ?)",
                (noun.lower(), 'noun', 1)  # 初始频率设为1
            )
        
        for adj in seed_adjectives:
            cursor.execute(
                "INSERT OR IGNORE INTO word_frequency (word, word_type, frequency) VALUES (?, ?, ?)",
                (adj.lower(), 'adjective', 1)  # 初始频率设为1
            )
        
        self.conn.commit()
    
    def add_word(self, word: str, word_type: str, increment: float = 1.0):
        """
        添加或更新单个词的频率
        
        Args:
            word: 词汇
            word_type: 词性类型 ('verb', 'noun', 'adjective')
            increment: 频率增量（默认1.0）。注意：新词首次添加时固定为1，后续增量可使用小数来减缓高频词增长
        """
        if word_type not in ['verb', 'noun', 'adjective']:
            raise ValueError(f"Invalid word_type: {word_type}. Must be 'verb', 'noun', or 'adjective'")
        
        word = word.lower().strip()
        if not word:
            return
        
        cursor = self.conn.cursor()
        
        # 检查词是否已存在
        cursor.execute(
            "SELECT frequency FROM word_frequency WHERE word = ?",
            (word,)
        )
        existing = cursor.fetchone()
        
        if existing is None:
            # 新词：首次添加时频率固定为1
            cursor.execute(
                """
                INSERT INTO word_frequency (word, word_type, frequency)
                VALUES (?, ?, 1)
                """,
                (word, word_type)
            )
        else:
            # 已有词：使用指定的增量
            cursor.execute(
                """
                UPDATE word_frequency 
                SET frequency = frequency + ?, last_updated = CURRENT_TIMESTAMP
                WHERE word = ?
                """,
                (increment, word)
            )
        
        self.conn.commit()
    
    def _normalize_instruction(self, instruction: str) -> str:
        """
        规范化指令文本用于比较
        - 转小写
        - 去除多余空格
        - 去除首尾空格
        """
        return ' '.join(instruction.lower().split())
    
    def add_instruction(self, instruction: str, mode: str) -> Dict[str, List[str]]:
        """
        添加一条指令并更新词频
        返回提取的词汇
        """
        # 简单的词性提取（这里使用规则，实际可以用 NLP 工具）
        words = self._extract_words(instruction)
        
        # 规范化指令用于去重
        normalized = self._normalize_instruction(instruction)
        
        cursor = self.conn.cursor()
        
        # 更新词频
        for word_type, word_list in words.items():
            for word in word_list:
                word = word.lower()
                cursor.execute(
                    """
                    UPDATE word_frequency
                    SET frequency = frequency + 1,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE word = ?
                    """,
                    (word,)
                )
                if cursor.rowcount == 0:
                    cursor.execute(
                        """
                        INSERT INTO word_frequency (word, word_type, frequency)
                        VALUES (?, ?, 1)
                        """,
                        (word, word_type)
                    )
        
        # 记录指令（同时存储原始和规范化版本）
        cursor.execute(
            """
            INSERT INTO instruction_history 
            (instruction, instruction_normalized, verbs, nouns, adjectives, mode)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                instruction,
                normalized,
                json.dumps(words.get('verb', [])),
                json.dumps(words.get('noun', [])),
                json.dumps(words.get('adjective', [])),
                mode
            )
        )
        
        self.conn.commit()
        return words

    def add_instruction_record(
        self,
        instruction: str,
        mode: str,
        *,
        verbs: Optional[List[str]] = None,
        nouns: Optional[List[str]] = None,
        adjectives: Optional[List[str]] = None,
        primary_object: str = "",
        hand: str = "",
        coordination_type: str = "",
        shared_object: str = "",
        source_image: str = "",
        task_id: str = "",
        skip_if_exists: bool = False,
    ) -> bool:
        """
        Record one instruction with explicit dimension fields (video pipeline / manifests).
        Updates word_frequency for verbs, nouns, adjectives.
        Returns True if a new row was inserted.
        """
        instruction = (instruction or "").strip()
        if not instruction:
            return False

        normalized = self._normalize_instruction(instruction)
        if skip_if_exists and self.instruction_exists(instruction):
            return False

        verbs = [str(v).strip().lower() for v in (verbs or []) if str(v).strip()]
        nouns = [str(n).strip().lower() for n in (nouns or []) if str(n).strip()]
        adjectives = [
            str(a).strip().lower() for a in (adjectives or []) if str(a).strip()
        ]

        if not verbs:
            verbs = self._verbs_from_instruction_text(instruction)

        cursor = self.conn.cursor()
        for verb in verbs:
            self.add_word(verb, "verb")
        for noun in nouns:
            self.add_word(noun, "noun")
        for adj in adjectives:
            self.add_word(adj, "adjective")

        cursor.execute(
            """
            INSERT INTO instruction_history
            (instruction, instruction_normalized, verbs, nouns, adjectives, mode,
             primary_object, hand, coordination_type, shared_object, source_image, task_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                instruction,
                normalized,
                json.dumps(verbs),
                json.dumps(nouns),
                json.dumps(adjectives),
                mode,
                (primary_object or "").strip().lower(),
                (hand or "").strip().lower(),
                (coordination_type or "").strip().lower(),
                (shared_object or "").strip().lower(),
                (source_image or "").strip(),
                (task_id or "").strip(),
            ),
        )
        self.conn.commit()
        return True

    @staticmethod
    def _verbs_from_instruction_text(instruction: str) -> List[str]:
        """Extract verb tokens from task text (supports 'then' compounds)."""
        text = (instruction or "").strip().lower()
        if not text:
            return []
        verb_pattern = (
            r"^(pick|grasp|place|push|pull|slide|open|close|rotate|turn|lift|lower|"
            r"press|tap|move|align|insert|remove|unscrew|screw|flip|twist|unfold|fold)\b"
        )
        verbs: List[str] = []
        for segment in re.split(r"\s+then\s+", text):
            segment = segment.strip()
            match = re.match(verb_pattern, segment)
            if match:
                verbs.append(match.group(1))
        return verbs

    def _extract_words(self, instruction: str) -> Dict[str, List[str]]:
        """
        从指令中提取动词、名词、形容词
        这是一个简化版本，实际应该使用 spaCy 或其他 NLP 工具
        """
        # 这里提供一个简单的实现
        # 实际使用时建议用 spaCy 进行词性标注
        words = instruction.lower().split()
        
        # 获取数据库中已知的词
        cursor = self.conn.cursor()
        cursor.execute("SELECT word, word_type FROM word_frequency")
        known_words = {word: word_type for word, word_type in cursor.fetchall()}
        
        result = defaultdict(list)
        
        for word in words:
            # 清理标点，但保留连字符（例如 pencil-case）
            clean_word = re.sub(r'[^a-z\-]', '', word.lower()).strip('-')
            if clean_word in known_words:
                result[known_words[clean_word]].append(clean_word)
            else:
                # 对于新词，使用简单的启发式规则
                # 实际应该用 NLP 工具
                pass
        
        return dict(result)
    
    def get_statistics(self) -> Dict:
        """获取数据库统计信息"""
        cursor = self.conn.cursor()
        
        stats = {}
        
        for word_type in ['verb', 'noun', 'adjective']:
            cursor.execute(
                "SELECT COUNT(*), AVG(frequency), MAX(frequency) FROM word_frequency WHERE word_type = ?",
                (word_type,)
            )
            count, avg_freq, max_freq = cursor.fetchone()
            stats[word_type] = {
                'count': count or 0,
                'avg_frequency': round(avg_freq or 0, 2),
                'max_frequency': max_freq or 0
            }
        
        cursor.execute("SELECT COUNT(*) FROM instruction_history")
        stats['total_instructions'] = cursor.fetchone()[0]

        return stats

    @staticmethod
    def _coefficient_of_variation(values: List[float]) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        if mean <= 0:
            return 0.0
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return math.sqrt(variance) / mean

    def get_diversity_overview(self, top_n: int = 30) -> Dict[str, Any]:
        """Aggregate diversity dimensions for dashboard visualization."""
        cursor = self.conn.cursor()
        overview: Dict[str, Any] = {
            "db_path": str(self.db_path),
            "summary": {},
            "balance": {},
            "by_mode": [],
            "verbs": [],
            "nouns": [],
            "adjectives": [],
            "primary_objects": [],
            "coordination_types": [],
            "hands": [],
            "recent_instructions": [],
        }

        stats = self.get_statistics()
        overview["summary"] = {
            "total_instructions": stats.get("total_instructions", 0),
            "unique_verbs": stats.get("verb", {}).get("count", 0),
            "unique_nouns": stats.get("noun", {}).get("count", 0),
            "unique_adjectives": stats.get("adjective", {}).get("count", 0),
        }

        for word_type, key in [
            ("verb", "verbs"),
            ("noun", "nouns"),
            ("adjective", "adjectives"),
        ]:
            rows = self.get_words_by_type(word_type)
            freqs = [float(r[1]) for r in rows]
            overview["balance"][f"{word_type}_cv"] = round(
                self._coefficient_of_variation(freqs), 3
            )
            top_rows = sorted(rows, key=lambda x: x[1], reverse=True)[:top_n]
            overview[key] = [
                {"word": w, "frequency": int(f)} for w, f in top_rows
            ]

        cursor.execute(
            """
            SELECT mode, COUNT(*) AS cnt
            FROM instruction_history
            GROUP BY mode ORDER BY cnt DESC
            """
        )
        overview["by_mode"] = [
            {"mode": row[0] or "unknown", "count": row[1]} for row in cursor.fetchall()
        ]

        for col, key in [
            ("primary_object", "primary_objects"),
            ("coordination_type", "coordination_types"),
            ("hand", "hands"),
        ]:
            cursor.execute(
                f"""
                SELECT {col}, COUNT(*) AS cnt
                FROM instruction_history
                WHERE {col} IS NOT NULL AND TRIM({col}) != ''
                GROUP BY {col} ORDER BY cnt DESC LIMIT ?
                """,
                (top_n,),
            )
            overview[key] = [
                {"label": row[0], "count": row[1]} for row in cursor.fetchall()
            ]

        cursor.execute(
            """
            SELECT instruction, mode, primary_object, hand, coordination_type,
                   shared_object, source_image, task_id, created_at
            FROM instruction_history
            ORDER BY id DESC LIMIT ?
            """,
            (min(top_n, 20),),
        )
        overview["recent_instructions"] = [
            {
                "instruction": row[0],
                "mode": row[1],
                "primary_object": row[2],
                "hand": row[3],
                "coordination_type": row[4],
                "shared_object": row[5],
                "source_image": row[6],
                "task_id": row[7],
                "created_at": row[8],
            }
            for row in cursor.fetchall()
        ]

        return overview
    
    def get_words_by_type(self, word_type: str, limit: int = None) -> List[Tuple[str, int]]:
        """获取指定类型的所有词及其频率"""
        cursor = self.conn.cursor()
        
        if limit:
            cursor.execute(
                "SELECT word, frequency FROM word_frequency WHERE word_type = ? ORDER BY frequency ASC LIMIT ?",
                (word_type, limit)
            )
        else:
            cursor.execute(
                "SELECT word, frequency FROM word_frequency WHERE word_type = ? ORDER BY frequency ASC",
                (word_type,)
            )
        
        return cursor.fetchall()
    
    def get_low_frequency_words(self, word_type: str = None, threshold: int = 3, limit: int = 20) -> List[str]:
        """获取低频词（频率低于阈值的词）"""
        cursor = self.conn.cursor()
        
        if word_type:
            cursor.execute(
                """
                SELECT word FROM word_frequency 
                WHERE word_type = ? AND frequency < ?
                ORDER BY frequency ASC
                LIMIT ?
                """,
                (word_type, threshold, limit)
            )
        else:
            cursor.execute(
                """
                SELECT word, word_type FROM word_frequency 
                WHERE frequency < ?
                ORDER BY frequency ASC
                LIMIT ?
                """,
                (threshold, limit)
            )
        
        return [row[0] for row in cursor.fetchall()]
    
    def get_lowest_frequency_words(self, word_type: str, limit: int = 20) -> List[str]:
        """
        获取频率最低的词（不使用固定阈值，而是获取频率最低的N个词）
        
        Args:
            word_type: 词性类型 ('verb', 'noun', 'adjective')
            limit: 返回数量，默认返回最低频的20个词
            
        Returns:
            低频词列表（按频率升序排列）
        """
        cursor = self.conn.cursor()
        
        # 获取该类型的所有词，按频率升序排列
        cursor.execute(
            """
            SELECT word FROM word_frequency 
            WHERE word_type = ?
            ORDER BY frequency ASC, word ASC
            LIMIT ?
            """,
            (word_type, limit)
        )
        
        return [row[0] for row in cursor.fetchall()]
    
    def get_all_words_sorted_by_frequency(self, word_type: str) -> List[Tuple[str, int]]:
        """
        获取某种词性的所有词，按频率升序排列
        
        Args:
            word_type: 词性类型 ('verb', 'noun', 'adjective')
            
        Returns:
            按频率排序的词列表 [(word, frequency), ...]
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT word, frequency FROM word_frequency 
            WHERE word_type = ?
            ORDER BY frequency ASC, word ASC
            """,
            (word_type,)
        )
        
        return cursor.fetchall()
    
    def word_exists(self, word: str, word_type: str = None) -> bool:
        """
        检查词是否在数据库中
        
        Args:
            word: 词汇
            word_type: 词性类型（可选，如果提供则只检查该类型）
            
        Returns:
            如果词存在返回True，否则返回False
        """
        word = word.lower().strip()
        if not word:
            return False
        
        cursor = self.conn.cursor()
        
        if word_type:
            cursor.execute(
                "SELECT COUNT(*) FROM word_frequency WHERE word = ? AND word_type = ?",
                (word, word_type)
            )
        else:
            cursor.execute(
                "SELECT COUNT(*) FROM word_frequency WHERE word = ?",
                (word,)
            )
        
        count = cursor.fetchone()[0]
        return count > 0
    
    def get_all_words_set(self, word_type: str) -> set:
        """
        获取某种词性的所有词（作为集合，用于快速查找）
        
        Args:
            word_type: 词性类型 ('verb', 'noun', 'adjective')
            
        Returns:
            词的集合
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT word FROM word_frequency WHERE word_type = ?",
            (word_type,)
        )
        
        return {row[0] for row in cursor.fetchall()}
    
    def get_frequency_percentile(self, word_type: str, percentile: float = 0.3) -> float:
        """
        获取某种词性频率的指定百分位数
        
        Args:
            word_type: 词性类型
            percentile: 百分位（0-1），默认0.3表示30%分位
            
        Returns:
            百分位对应的频率值
        """
        cursor = self.conn.cursor()
        
        # 获取所有该类型词的频率
        cursor.execute(
            """
            SELECT frequency FROM word_frequency 
            WHERE word_type = ?
            ORDER BY frequency ASC
            """,
            (word_type,)
        )
        
        frequencies = [row[0] for row in cursor.fetchall()]
        
        if not frequencies:
            return 0
        
        # 计算百分位索引
        idx = int(len(frequencies) * percentile)
        idx = min(idx, len(frequencies) - 1)
        
        return frequencies[idx]
    
    def get_weighted_sample_words(self, word_type: str, limit: int = 20, 
                                   max_frequency: int = None) -> List[Tuple[str, int]]:
        """
        获取加权采样的词汇（频率越低，权重越高）
        使用 (max_freq - frequency + 1) 作为权重，确保低频词更容易被选中
        
        Args:
            word_type: 词性类型 ('verb', 'noun', 'adjective')
            limit: 返回数量
            max_frequency: 最大频率限制（可选，用于限定采样范围）
            
        Returns:
            词汇列表 [(word, frequency), ...]
        """
        cursor = self.conn.cursor()
        
        if max_frequency:
            cursor.execute(
                """
                SELECT word, frequency FROM word_frequency 
                WHERE word_type = ? AND frequency <= ?
                """,
                (word_type, max_frequency)
            )
        else:
            cursor.execute(
                """
                SELECT word, frequency FROM word_frequency 
                WHERE word_type = ?
                """,
                (word_type,)
            )
        
        words = cursor.fetchall()
        if not words:
            return []
        
        # 计算权重：频率越低权重越高
        # 权重 = (max_freq - freq + 1)，确保最低频的词权重最高
        max_freq = max(freq for _, freq in words)
        weights = [(max_freq - freq + 1) for _, freq in words]
        
        # 加权随机采样
        import random
        # 使用真实随机
        sampled = random.choices(words, weights=weights, k=min(limit, len(words)))
        
        return sampled
    
    def get_balanced_sample_words(self, word_type: str, limit: int = 20) -> List[Tuple[str, int]]:
        """
        获取均衡采样的词汇（确保每个词都有机会被选中）
        使用轮询 + 随机的方式，确保长期均衡
        
        Args:
            word_type: 词性类型 ('verb', 'noun', 'adjective')
            limit: 返回数量
            
        Returns:
            词汇列表 [(word, frequency), ...]
        """
        cursor = self.conn.cursor()
        
        # 按频率分组，每组随机选择一个
        cursor.execute(
            """
            SELECT word, frequency FROM word_frequency 
            WHERE word_type = ?
            ORDER BY frequency ASC
            """,
            (word_type,)
        )
        
        words = cursor.fetchall()
        if not words:
            return []
        
        # 将词汇按频率分成若干组
        freq_groups = {}
        for word, freq in words:
            if freq not in freq_groups:
                freq_groups[freq] = []
            freq_groups[freq].append((word, freq))
        
        # 从每个频率组中均匀采样
        result = []
        frequencies = sorted(freq_groups.keys())
        
        # 计算每组应该采样的数量
        samples_per_group = max(1, limit // len(frequencies))
        
        import random
        for freq in frequencies:
            group = freq_groups[freq]
            # 随机打乱后取样
            random.shuffle(group)
            samples = group[:samples_per_group]
            result.extend(samples)
        
        # 打乱最终结果并返回
        random.shuffle(result)
        return result[:limit]
    
    def instruction_exists(self, instruction: str) -> bool:
        """
        检查指令是否已存在于数据库中（使用规范化比较）
        
        Args:
            instruction: 指令文本
            
        Returns:
            如果指令已存在返回True，否则返回False
        """
        normalized = self._normalize_instruction(instruction)
        cursor = self.conn.cursor()
        
        # 优先使用规范化字段查询（如果存在）
        cursor.execute(
            "SELECT COUNT(*) FROM instruction_history WHERE instruction_normalized = ?",
            (normalized,)
        )
        count = cursor.fetchone()[0]
        
        # 如果规范化字段没有匹配，也检查原始字段（兼容旧数据）
        if count == 0:
            cursor.execute(
                "SELECT COUNT(*) FROM instruction_history WHERE instruction = ?",
                (instruction,)
            )
            count = cursor.fetchone()[0]
        
        return count > 0
    
    def close(self):
        """关闭数据库连接"""
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

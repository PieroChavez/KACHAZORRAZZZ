"""Market Memory System
Tracks price level interactions over time to build institutional memory.
Each level records: visits, bounces, breaks, and associated patterns.
Provides level reliability scores that feed into pattern detection.
"""
import logging
import sqlite3
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

from src.utils.helpers import atr, find_swing_points

logger = logging.getLogger(__name__)


@dataclass
class LevelMemory:
    price: float
    symbol: str
    visit_count: int = 0
    bounce_count: int = 0
    break_count: int = 0
    total_touches: int = 0
    last_visit: Optional[datetime] = None
    last_outcome: Optional[str] = None
    pattern_types: List[str] = field(default_factory=list)

    @property
    def reliability(self) -> float:
        if self.total_touches == 0:
            return 0.0
        ratio = (self.bounce_count + 1) / (self.total_touches + 2)
        return min(1.0, ratio * (1.0 - 1.0 / (self.total_touches + 1)))

    @property
    def is_reliable_support(self) -> bool:
        return self.reliability >= 0.6 and self.bounce_count > self.break_count

    @property
    def is_reliable_resistance(self) -> bool:
        return self.reliability >= 0.6 and self.bounce_count > self.break_count


class MarketMemory:
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "data" / "market_memory.db"
        self.db_path = db_path
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_db()
        self._cache: Dict[str, List[LevelMemory]] = {}
        self._max_cache_size = 1000

    def _init_db(self):
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS level_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    visit_count INTEGER DEFAULT 0,
                    bounce_count INTEGER DEFAULT 0,
                    break_count INTEGER DEFAULT 0,
                    total_touches INTEGER DEFAULT 0,
                    last_visit TEXT,
                    last_outcome TEXT,
                    pattern_types TEXT DEFAULT '[]',
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(symbol, price)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS level_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    pattern_type TEXT,
                    distance_pips REAL,
                    context_regime TEXT
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def record_interaction(self, symbol: str, price: float, outcome: str,
                           pattern_type: Optional[str] = None,
                           ltf_df: Optional[pd.DataFrame] = None,
                           regime: Optional[str] = None):
        conn = sqlite3.connect(str(self.db_path))
        try:
            price_rounded = round(price, 5)
            conn.execute("""
                INSERT INTO level_interactions
                (symbol, price, timestamp, outcome, pattern_type, context_regime)
                VALUES (?, ?, datetime('now'), ?, ?, ?)
            """, (symbol, price_rounded, outcome, pattern_type, regime))

            existing = conn.execute(
                "SELECT id, visit_count, bounce_count, break_count, total_touches, pattern_types FROM level_memory WHERE symbol=? AND price=?",
                (symbol, price_rounded)
            ).fetchone()

            if existing:
                new_visits = existing[1] + 1
                new_bounces = existing[2] + (1 if outcome == "bounce" else 0)
                new_breaks = existing[3] + (1 if outcome == "break" else 0)
                new_touches = existing[4] + 1
                old_patterns = json.loads(existing[5])
                if pattern_type and pattern_type not in old_patterns:
                    old_patterns.append(pattern_type)
                conn.execute("""
                    UPDATE level_memory SET
                        visit_count=?, bounce_count=?, break_count=?,
                        total_touches=?, last_visit=datetime('now'),
                        last_outcome=?, pattern_types=?,
                        updated_at=datetime('now')
                    WHERE id=?
                """, (new_visits, new_bounces, new_breaks, new_touches,
                      outcome, json.dumps(old_patterns), existing[0]))
            else:
                pats = json.dumps([pattern_type] if pattern_type else [])
                conn.execute("""
                    INSERT INTO level_memory
                    (symbol, price, visit_count, bounce_count, break_count,
                     total_touches, last_visit, last_outcome, pattern_types)
                    VALUES (?, ?, 1, ?, ?, 1, datetime('now'), ?, ?)
                """, (symbol, price_rounded,
                      1 if outcome == "bounce" else 0,
                      1 if outcome == "break" else 0,
                      outcome, pats))

            conn.commit()
            self._invalidate_cache(symbol)
        finally:
            conn.close()

    def get_level_reliability(self, symbol: str, price: float) -> LevelMemory:
        conn = sqlite3.connect(str(self.db_path))
        try:
            price_rounded = round(price, 5)
            row = conn.execute(
                "SELECT price, symbol, visit_count, bounce_count, break_count, total_touches, last_visit, last_outcome, pattern_types FROM level_memory WHERE symbol=? AND price=?",
                (symbol, price_rounded)
            ).fetchone()
            if row:
                return LevelMemory(
                    price=row[0], symbol=row[1], visit_count=row[2],
                    bounce_count=row[3], break_count=row[4],
                    total_touches=row[5],
                    last_visit=datetime.fromisoformat(row[6]) if row[6] else None,
                    last_outcome=row[7],
                    pattern_types=json.loads(row[8]),
                )
            return LevelMemory(price=price, symbol=symbol)
        finally:
            conn.close()

    def get_nearby_levels(self, symbol: str, price: float,
                           atr_value: float, max_levels: int = 5) -> List[LevelMemory]:
        cache_key = f"{symbol}_{round(price, 2)}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        conn = sqlite3.connect(str(self.db_path))
        try:
            min_p = price - atr_value * 2.0
            max_p = price + atr_value * 2.0
            rows = conn.execute("""
                SELECT price, symbol, visit_count, bounce_count, break_count,
                       total_touches, last_visit, last_outcome, pattern_types
                FROM level_memory
                WHERE symbol=? AND price BETWEEN ? AND ?
                ORDER BY total_touches DESC, bounce_count DESC
                LIMIT ?
            """, (symbol, min_p, max_p, max_levels)).fetchall()

            levels = []
            for row in rows:
                level = LevelMemory(
                    price=row[0], symbol=row[1], visit_count=row[2],
                    bounce_count=row[3], break_count=row[4],
                    total_touches=row[5],
                    last_visit=datetime.fromisoformat(row[6]) if row[6] else None,
                    last_outcome=row[7],
                    pattern_types=json.loads(row[8]),
                )
                levels.append(level)

            self._cache[cache_key] = levels
            if len(self._cache) > self._max_cache_size:
                self._cache.clear()

            return levels
        finally:
            conn.close()

    def get_regime_level_stats(self, symbol: str, regime_type: str) -> Dict:
        conn = sqlite3.connect(str(self.db_path))
        try:
            rows = conn.execute("""
                SELECT outcome, COUNT(*) as cnt
                FROM level_interactions
                WHERE symbol=? AND context_regime=?
                GROUP BY outcome
            """, (symbol, regime_type)).fetchall()
            stats = {"bounce": 0, "break": 0, "total": 0}
            for outcome, cnt in rows:
                stats[outcome] = cnt
            stats["total"] = stats["bounce"] + stats["break"]
            stats["bounce_rate"] = stats["bounce"] / stats["total"] if stats["total"] > 0 else 0.0
            return stats
        finally:
            conn.close()

    def get_consolidated_levels(self, symbol: str, atr_value: float) -> List[LevelMemory]:
        conn = sqlite3.connect(str(self.db_path))
        try:
            rows = conn.execute("""
                SELECT price, symbol, visit_count, bounce_count, break_count,
                       total_touches, last_visit, last_outcome, pattern_types
                FROM level_memory
                WHERE symbol=? AND total_touches >= 3
                ORDER BY total_touches DESC
                LIMIT 20
            """, (symbol,)).fetchall()

            levels = []
            for row in rows:
                level = LevelMemory(
                    price=row[0], symbol=row[1], visit_count=row[2],
                    bounce_count=row[3], break_count=row[4],
                    total_touches=row[5],
                    last_visit=datetime.fromisoformat(row[6]) if row[6] else None,
                    last_outcome=row[7],
                    pattern_types=json.loads(row[8]),
                )
                if level.reliability >= 0.5:
                    levels.append(level)

            merged = self._merge_close_levels(levels, atr_value)
            return merged[:10]
        finally:
            conn.close()

    def _merge_close_levels(self, levels: List[LevelMemory], atr_value: float) -> List[LevelMemory]:
        if not levels:
            return []
        sorted_levels = sorted(levels, key=lambda x: x.price)
        merged = [sorted_levels[0]]
        for level in sorted_levels[1:]:
            if abs(level.price - merged[-1].price) <= atr_value * 0.3:
                existing = merged[-1]
                existing.visit_count += level.visit_count
                existing.bounce_count += level.bounce_count
                existing.break_count += level.break_count
                existing.total_touches += level.total_touches
                for pt in level.pattern_types:
                    if pt not in existing.pattern_types:
                        existing.pattern_types.append(pt)
            else:
                merged.append(level)
        return merged

    def scan_current_levels(self, symbol: str, ltf_df: pd.DataFrame) -> List[LevelMemory]:
        if ltf_df is None or len(ltf_df) < 20:
            return []
        atr_val = atr(ltf_df, 14).iloc[-1]
        highs_idx, lows_idx = find_swing_points(ltf_df, lookback=3)
        current_levels = []
        for idx in highs_idx[-5:]:
            price = ltf_df["high"].iloc[idx]
            level = self.get_level_reliability(symbol, price)
            if level.total_touches > 0:
                current_levels.append(level)
        for idx in lows_idx[-5:]:
            price = ltf_df["low"].iloc[idx]
            level = self.get_level_reliability(symbol, price)
            if level.total_touches > 0:
                current_levels.append(level)
        return current_levels

    def get_level_bias(self, symbol: str, current_price: float,
                        atr_value: float) -> Optional[str]:
        levels = self.get_nearby_levels(symbol, current_price, atr_value, max_levels=3)
        if not levels:
            return None
        support_count = sum(1 for l in levels if l.price < current_price and l.is_reliable_support)
        resistance_count = sum(1 for l in levels if l.price > current_price and l.is_reliable_resistance)
        if support_count >= 2 and resistance_count == 0:
            return "BULLISH_BIAS"
        if resistance_count >= 2 and support_count == 0:
            return "BEARISH_BIAS"
        if support_count >= 2 and resistance_count >= 2:
            return "RANGE_BIAS"
        return None

    def _invalidate_cache(self, symbol: str):
        keys_to_remove = [k for k in self._cache if k.startswith(symbol)]
        for k in keys_to_remove:
            del self._cache[k]

    def cleanup_old_entries(self, days: int = 30):
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                DELETE FROM level_interactions
                WHERE timestamp < datetime('now', '-' || ? || ' days')
            """, (days,))
            conn.execute("""
                DELETE FROM level_memory
                WHERE total_touches = 0
                AND updated_at < datetime('now', '-' || ? || ' days')
            """, (days,))
            conn.commit()
            logger.info(f"Limpieza de market_memory: entradas anteriores a {days} días eliminadas")
        finally:
            conn.close()

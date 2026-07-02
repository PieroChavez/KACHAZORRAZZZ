"""Fractal Database — tracks structural swing points across timeframes
Stores valid fractals (CHoCH/BOS-based ranges) and their Fibonacci 0.72 levels.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict
from pathlib import Path
import sqlite3
import json
import threading
from loguru import logger


@dataclass
class Fractal:
    id: int = 0
    symbol: str = ""
    timeframe: str = ""
    direction: str = ""            # "bullish" (buy) or "bearish" (sell)
    level0: float = 0.0            # end of move (fib 0)
    level1: float = 0.0            # start of move (fib 1 / SL origin)
    fib_072: float = 0.0           # 0.72 retracement entry level
    swing_high: float = 0.0
    swing_low: float = 0.0
    bos_index: int = 0             # candle index where BOS occurred
    bos_time: Optional[datetime] = None
    active: bool = True
    created_at: Optional[datetime] = None
    hit_entry: bool = False        # True once price touches fib_072
    entry_price: float = 0.0
    sl_price: float = 0.0
    is_subfractal: bool = False    # True if detected on 5M (independent hunting)
    note: str = ""


class FractalDB:
    def __init__(self, symbol: str, db_path: Optional[Path] = None):
        self.symbol = symbol
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "data" / "db" / symbol / "fractal_state.db"
        db_path.parent.mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._init_db()
        self._cache: Dict[int, Fractal] = {}
        self._load_cache()

    def _init_db(self):
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS fractals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    timeframe TEXT,
                    direction TEXT,
                    level0 REAL,
                    level1 REAL,
                    fib_072 REAL,
                    swing_high REAL,
                    swing_low REAL,
                    bos_index INTEGER,
                    bos_time TEXT,
                    active INTEGER DEFAULT 1,
                    created_at TEXT,
                    hit_entry INTEGER DEFAULT 0,
                    entry_price REAL DEFAULT 0,
                    sl_price REAL DEFAULT 0,
                    is_subfractal INTEGER DEFAULT 0,
                    note TEXT DEFAULT ''
                )
            """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS swing_cache (
                    timeframe TEXT PRIMARY KEY,
                    last_high_idx INTEGER DEFAULT 0,
                    last_low_idx INTEGER DEFAULT 0,
                    last_high_price REAL DEFAULT 0,
                    last_low_price REAL DEFAULT 0,
                    updated_at TEXT
                )
            """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS config (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            try:
                self._conn.execute("ALTER TABLE fractals ADD COLUMN is_subfractal INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass
            try:
                self._conn.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS uq_fractal
                    ON fractals(symbol, timeframe, direction, bos_index, bos_time)
                """)
            except sqlite3.OperationalError:
                pass
            self._conn.commit()

    def _load_cache(self):
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM fractals WHERE symbol=? AND active=1", (self.symbol,)
            ).fetchall()
            for r in rows:
                f = self._row_to_fractal(r)
                self._cache[f.id] = f

    @staticmethod
    def _row_to_fractal(row) -> Fractal:
        is_sub = bool(row[17]) if len(row) > 17 else False
        return Fractal(
            id=row[0], symbol=row[1], timeframe=row[2], direction=row[3],
            level0=row[4], level1=row[5], fib_072=row[6],
            swing_high=row[7], swing_low=row[8], bos_index=row[9],
            bos_time=datetime.fromisoformat(row[10]) if row[10] else None,
            active=bool(row[11]),
            created_at=datetime.fromisoformat(row[12]) if row[12] else None,
            hit_entry=bool(row[13]), entry_price=row[14], sl_price=row[15],
            is_subfractal=is_sub,
            note=row[16] or "",
        )

    def add_fractal(self, f: Fractal) -> int:
        with self._lock:
            now = datetime.utcnow()
            cur = self._conn.execute("""
                INSERT OR IGNORE INTO fractals (symbol, timeframe, direction, level0, level1,
                    fib_072, swing_high, swing_low, bos_index, bos_time,
                    created_at, entry_price, sl_price, is_subfractal, note)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (f.symbol, f.timeframe, f.direction, f.level0, f.level1,
                  f.fib_072, f.swing_high, f.swing_low, f.bos_index,
                  f.bos_time.isoformat() if f.bos_time else None,
                  now.isoformat(), f.entry_price, f.sl_price,
                  int(f.is_subfractal), f.note))
            self._conn.commit()
            if cur.rowcount == 0:
                logger.debug(f"[{f.symbol}] Fractal duplicado ignorado "
                             f"{f.timeframe} {f.direction} idx={f.bos_index}")
                return 0
            f.id = cur.lastrowid
            f.created_at = now
            f.active = True
            self._cache[f.id] = f
            logger.info(f"[{f.symbol}] {f.timeframe} {f.direction.upper()} fractal #{f.id} "
                         f"L1={f.level1:.2f} L0={f.level0:.2f} 0.72={f.fib_072:.2f}")
            return f.id

    def invalidate(self, fractal_id: int):
        with self._lock:
            self._conn.execute(
                "UPDATE fractals SET active=0 WHERE id=?", (fractal_id,)
            )
            self._conn.commit()
            if fractal_id in self._cache:
                self._cache[fractal_id].active = False
                logger.info(f"Fractal #{fractal_id} invalidated")

    def mark_entry_hit(self, fractal_id: int, entry_price: float, sl_price: float):
        with self._lock:
            self._conn.execute("""
                UPDATE fractals SET hit_entry=1, entry_price=?, sl_price=?
                WHERE id=?
            """, (entry_price, sl_price, fractal_id))
            self._conn.commit()
            if fractal_id in self._cache:
                f = self._cache[fractal_id]
                f.hit_entry = True
                f.entry_price = entry_price
                f.sl_price = sl_price

    def get_active_fractals(self) -> List[Fractal]:
        return [f for f in self._cache.values() if f.active]

    def get_active_by_timeframe(self, tf: str) -> List[Fractal]:
        return [f for f in self._cache.values() if f.active and f.timeframe == tf]

    def get_active_not_hit(self) -> List[Fractal]:
        return [f for f in self._cache.values() if f.active and not f.hit_entry]

    def get_by_id(self, fid: int) -> Optional[Fractal]:
        return self._cache.get(fid)

    def count_active(self) -> int:
        return sum(1 for f in self._cache.values() if f.active)

    def clean_inactive(self):
        self._cache = {k: v for k, v in self._cache.items() if v.active}

    def get_config(self, key: str, default: str = "") -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM config WHERE key=?", (key,)
            ).fetchone()
            return row[0] if row else default

    def set_config(self, key: str, value: str):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO config (key, value) VALUES (?,?)",
                (key, value)
            )
            self._conn.commit()

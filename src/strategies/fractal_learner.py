"""Fractal Learner — ML feedback for structural trades
Records every fractal pack (entry + exit) with context: TF, direction, range,
hour, day. Periodically analyzes win rate per feature and adjusts volume
multiplier and filters automatically.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import sqlite3
import json
from statistics import mean, stdev
from loguru import logger


@dataclass
class FractalTradeRecord:
    pack_id: int
    fractal_id: int
    symbol: str
    timeframe: str
    direction: str
    is_subfractal: bool
    entry_price: float
    sl_price: float
    tp_price: float
    volume: float
    range_size: float
    fib_level: float
    entry_hour: int
    entry_day: int
    outcome: str = ""         # "win" | "loss" | "open"
    exit_price: float = 0.0
    profit: float = 0.0
    duration_hours: float = 0.0
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None


@dataclass
class PerformanceStats:
    total: int = 0
    wins: int = 0
    losses: int = 0
    total_profit: float = 0.0
    avg_profit: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    volume_mult: float = 1.0

    def compute(self):
        self.win_rate = self.wins / self.total if self.total > 0 else 0.0
        self.avg_profit = self.total_profit / self.total if self.total > 0 else 0.0
        loss_total = abs(self.total_profit - self.wins * self.avg_profit)
        self.profit_factor = abs(self.total_profit / loss_total) if loss_total > 0 else 999.0


class FractalLearner:
    def __init__(self, symbol: str, db_path: Optional[Path] = None):
        self.symbol = symbol
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "data" / "db" / symbol / "fractal_learner.db"
        db_path.parent.mkdir(exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._init_db()
        self._learning_enabled = True
        self._last_analysis: Optional[datetime] = None
        self._analysis_interval = timedelta(hours=4)
        self._volume_multipliers: Dict[str, float] = self._load_multipliers()
        self._filter_blacklist: List[str] = []

    def _init_db(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS fractal_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pack_id INTEGER, fractal_id INTEGER,
                symbol TEXT, timeframe TEXT, direction TEXT,
                is_subfractal INTEGER, entry_price REAL,
                sl_price REAL, tp_price REAL, volume REAL,
                range_size REAL, fib_level REAL,
                entry_hour INTEGER, entry_day INTEGER,
                session TEXT DEFAULT '',
                outcome TEXT DEFAULT 'open',
                exit_price REAL DEFAULT 0,
                profit REAL DEFAULT 0,
                duration_hours REAL DEFAULT 0,
                opened_at TEXT, closed_at TEXT
            )
        """)
        try:
            self._conn.execute("ALTER TABLE fractal_trades ADD COLUMN session TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS learner_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)
        self._conn.commit()

    def _load_multipliers(self) -> Dict[str, float]:
        row = self._conn.execute(
            "SELECT value FROM learner_config WHERE key='volume_multipliers'"
        ).fetchone()
        if row:
            return json.loads(row[0])
        return {}

    def _save_multipliers(self):
        self._conn.execute("""
            INSERT OR REPLACE INTO learner_config (key, value, updated_at)
            VALUES (?, ?, datetime('now'))
        """, ("volume_multipliers", json.dumps(self._volume_multipliers)))
        self._conn.commit()

    def record_entry(self, pack_id: int, fractal_id: int, timeframe: str,
                     direction: str, is_subfractal: bool, entry_price: float,
                     sl_price: float, tp_price: float, volume: float,
                     range_size: float, fib_level: float,
                     session: str = ""):
        if not self._learning_enabled:
            return
        now = datetime.utcnow()
        self._conn.execute("""
            INSERT INTO fractal_trades (pack_id, fractal_id, symbol, timeframe,
                direction, is_subfractal, entry_price, sl_price, tp_price,
                volume, range_size, fib_level, entry_hour, entry_day, session, opened_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (pack_id, fractal_id, self.symbol, timeframe, direction,
              int(is_subfractal), entry_price, sl_price, tp_price, volume,
              range_size, fib_level, now.hour, now.weekday(), session,
              now.isoformat()))
        self._conn.commit()

    def record_exit(self, pack_id: int, outcome: str, exit_price: float,
                    profit: float):
        if not self._learning_enabled:
            return
        now = datetime.utcnow()
        row = self._conn.execute(
            "SELECT opened_at FROM fractal_trades WHERE pack_id=?", (pack_id,)
        ).fetchone()
        dur = 0.0
        if row and row[0]:
            opened = datetime.fromisoformat(row[0])
            dur = (now - opened).total_seconds() / 3600
        self._conn.execute("""
            UPDATE fractal_trades SET outcome=?, exit_price=?, profit=?,
                duration_hours=?, closed_at=?
            WHERE pack_id=?
        """, (outcome, exit_price, profit, dur, now.isoformat(), pack_id))
        self._conn.commit()

    def analyze(self, force: bool = False) -> Dict:
        now = datetime.utcnow()
        if self._last_analysis and not force:
            if now - self._last_analysis < self._analysis_interval:
                return {"analyzed": False, "reason": "intervalo no vencido"}
        count = self._conn.execute(
            "SELECT COUNT(*) FROM fractal_trades WHERE outcome != 'open'"
        ).fetchone()[0]
        if count < 5:
            return {"analyzed": False, "reason": f"solo {count} trades, mínimo 5"}

        results = {"analyzed": True, "by_timeframe": {}, "by_direction": {},
                   "adjustments": [], "recommendations": []}

        # Per timeframe
        for tf in ("4H", "2H", "30min", "15min", "5min"):
            stats = self._stats_for("timeframe=? AND symbol=?", (tf, self.symbol))
            if stats and stats.total >= 2:
                results["by_timeframe"][tf] = stats.__dict__
                self._adjust_volume_for(tf, stats, results)

        # Per direction
        for direction in ("bullish", "bearish"):
            stats = self._stats_for("direction=? AND symbol=?", (direction, self.symbol))
            if stats and stats.total >= 2:
                results["by_direction"][direction] = stats.__dict__

        # Per subfractal vs macro
        for sub in (0, 1):
            label = "subfractal_5m" if sub else "macro"
            stats = self._stats_for("is_subfractal=? AND symbol=?", (sub, self.symbol))
            if stats and stats.total >= 2:
                results[label] = stats.__dict__

        # Per session
        sess_rows = self._conn.execute("""
            SELECT DISTINCT session FROM fractal_trades
            WHERE outcome != 'open' AND symbol=? AND session != ''
        """, (self.symbol,)).fetchall()
        for (session,) in sess_rows:
            stats = self._stats_for("session=? AND symbol=?", (session, self.symbol))
            if stats and stats.total >= 2:
                results[f"session:{session}"] = stats.__dict__
                self._adjust_volume_for(f"session:{session}", stats, results)

        self._save_multipliers()
        self._last_analysis = now
        logger.info(f"[FractalLearner] Análisis completado: {count} trades, "
                     f"{len(results['adjustments'])} ajustes")
        return results

    def _stats_for(self, where_clause: str, params: tuple) -> Optional[PerformanceStats]:
        rows = self._conn.execute(f"""
            SELECT COUNT(*),
                   SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN profit <= 0 THEN 1 ELSE 0 END),
                   SUM(profit)
            FROM fractal_trades
            WHERE outcome != 'open' AND {where_clause}
        """, params).fetchone()
        if not rows or rows[0] == 0:
            return None
        total, wins, losses, total_profit = rows
        wins = wins or 0
        losses = losses or 0
        total_profit = total_profit or 0.0
        key = str(params)  # unique key for this group
        stats = PerformanceStats(
            total=total, wins=wins, losses=losses, total_profit=total_profit,
            volume_mult=self._volume_multipliers.get(key, 1.0)
        )
        stats.compute()
        return stats

    def _adjust_volume_for(self, group: str, stats: PerformanceStats,
                            results: Dict):
        key = group if group.startswith("session:") else f"tf:{group}"
        old_mult = self._volume_multipliers.get(key, 1.0)
        new_mult = old_mult

        if stats.total >= 3:
            if stats.win_rate >= 0.65:
                new_mult = round(min(old_mult * 1.15, 2.0), 2)
            elif stats.win_rate <= 0.35:
                new_mult = round(max(old_mult * 0.8, 0.3), 2)

        if new_mult != old_mult:
            self._volume_multipliers[key] = new_mult
            results["adjustments"].append(
                f"{group}: vol_mult {old_mult}→{new_mult} "
                f"(wr={stats.win_rate:.0%}, n={stats.total})"
            )

    def get_volume_mult(self, timeframe: str, direction: str,
                        is_subfractal: bool,
                        session: str = "") -> float:
        tf_mult = self._volume_multipliers.get(f"tf:{timeframe}", 1.0)
        sess_mult = self._volume_multipliers.get(f"session:{session}", 1.0) if session else 1.0
        return round(tf_mult * sess_mult, 2)

    def get_summary(self) -> Dict:
        total = self._conn.execute(
            "SELECT COUNT(*) FROM fractal_trades"
        ).fetchone()[0]
        closed = self._conn.execute(
            "SELECT COUNT(*) FROM fractal_trades WHERE outcome != 'open'"
        ).fetchone()[0]
        wins = self._conn.execute(
            "SELECT COUNT(*) FROM fractal_trades WHERE outcome='win'"
        ).fetchone()[0]
        profit = self._conn.execute(
            "SELECT SUM(profit) FROM fractal_trades WHERE outcome != 'open'"
        ).fetchone()[0] or 0.0
        return {
            "total_trades": total,
            "closed_trades": closed,
            "wins": wins,
            "win_rate": wins / closed if closed > 0 else 0.0,
            "total_profit": round(profit, 2),
            "volume_multipliers": self._volume_multipliers,
        }

    def close(self):
        self._conn.close()

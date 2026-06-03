"""Kelly Criterion Risk Manager
Replaces fixed risk percentage with dynamic Kelly fraction.
Learns optimal bet size from historical win rate and avg risk/reward.
"""
import logging
import sqlite3
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KellyState:
    win_rate: float = 0.5
    avg_win: float = 1.0
    avg_loss: float = 1.0
    kelly_fraction: float = 0.02
    conservative_mult: float = 0.25
    total_trades: int = 0
    recent_trades: List[float] = field(default_factory=list)


class KellyRiskManager:
    def __init__(self, db_path: Optional[Path] = None,
                 initial_risk: float = 0.02,
                 max_risk: float = 0.04,
                 min_risk: float = 0.005):
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "data" / "kelly_state.db"
        self.db_path = db_path
        self.db_path.parent.mkdir(exist_ok=True)
        self.state = KellyState(
            kelly_fraction=initial_risk,
            conservative_mult=0.25,
        )
        self.initial_risk = initial_risk
        self.max_risk = max_risk
        self.min_risk = min_risk
        self._init_db()
        self._load()

    def _init_db(self):
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kelly_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    direction TEXT,
                    profit REAL,
                    conviction REAL,
                    regime TEXT,
                    timestamp TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kelly_state (
                    key TEXT PRIMARY KEY,
                    value REAL
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def _load(self):
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            rows = conn.execute("SELECT * FROM kelly_state").fetchall()
            for key, value in rows:
                if hasattr(self.state, key):
                    setattr(self.state, key, value)
            trade_rows = conn.execute("""
                SELECT profit FROM kelly_trades ORDER BY id DESC LIMIT 50
            """).fetchall()
            self.state.recent_trades = [r[0] for r in reversed(trade_rows)]
            self.state.total_trades = len(self.state.recent_trades)
        finally:
            conn.close()

    def _save_state(self):
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            for key, value in self.state.__dict__.items():
                if key == "recent_trades":
                    continue
                if isinstance(value, (int, float)):
                    conn.execute("""
                        INSERT OR REPLACE INTO kelly_state (key, value) VALUES (?, ?)
                    """, (key, float(value)))
            conn.commit()
        finally:
            conn.close()

    def record_trade(self, symbol: str, direction: str, profit: float,
                     conviction: float, regime: str):
        self.state.recent_trades.append(profit)
        if len(self.state.recent_trades) > 50:
            self.state.recent_trades = self.state.recent_trades[-50:]
        self.state.total_trades = len(self.state.recent_trades)

        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            conn.execute("""
                INSERT INTO kelly_trades (symbol, direction, profit, conviction, regime, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (symbol, direction, profit, conviction, regime, datetime.now().isoformat()))
            conn.commit()
        finally:
            conn.close()

        if self.state.total_trades >= 5:
            self._recompute_kelly()
        self._save_state()

    def _recompute_kelly(self):
        trades = self.state.recent_trades
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]

        total = len(trades)
        win_rate = len(wins) / total if total > 0 else 0.5
        avg_win = sum(wins) / len(wins) if wins else 1.0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 1.0

        if avg_loss == 0:
            avg_loss = 1.0

        b = avg_win / avg_loss
        q = 1 - win_rate
        p = win_rate

        kelly = (p * b - q) / b if b > 0 else 0
        kelly = max(0, kelly)

        self.state.win_rate = win_rate
        self.state.avg_win = avg_win
        self.state.avg_loss = avg_loss
        self.state.kelly_fraction = max(
            self.min_risk,
            min(self.max_risk, kelly * self.state.conservative_mult),
        )

        logger.info(
            f"Kelly updated: WR={win_rate:.2f} avg_win=${avg_win:.2f} "
            f"avg_loss=${avg_loss:.2f} b={b:.2f} kelly={kelly:.4f} "
            f"→ risk={self.state.kelly_fraction:.4f}"
        )

    def get_risk_fraction(self, conviction: float) -> float:
        base = self.state.kelly_fraction
        conviction_mult = 0.5 + conviction * 0.5
        return max(self.min_risk, min(self.max_risk, base * conviction_mult))

    def get_adjusted_volume_mult(self, conviction: float) -> float:
        base = self.state.kelly_fraction
        if self.state.total_trades < 5:
            return 1.0
        conviction_mult = 0.5 + conviction * 0.5
        adjusted = base * conviction_mult
        return adjusted / self.initial_risk if self.initial_risk > 0 else 1.0


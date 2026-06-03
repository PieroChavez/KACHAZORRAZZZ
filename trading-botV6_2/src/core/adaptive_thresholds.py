"""Adaptive Threshold Engine: ajusta min_reversal_score y min_score dinámicamente
según la tasa de acierto histórica de reversals por nivel de score.
No modifica ninguna lógica del bot, solo produce thresholds recomendados."""
import sqlite3
import time
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from loguru import logger


class AdaptiveThresholdEngine:
    """Ventana deslizante de reversals → win rate por bin de score → threshold óptimo."""

    def __init__(self, db_path: Path, window_size: int = 100, min_samples: int = 10):
        self.db_path = db_path
        self.window_size = window_size
        self.min_samples = min_samples
        self._conn: Optional[sqlite3.Connection] = None

    def initialize(self):
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS reversal_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                score REAL, conviction REAL, regime TEXT, direction TEXT,
                profit REAL, won INTEGER, timestamp REAL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS threshold_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL, reversal_threshold REAL, loss_threshold REAL,
                window_size INTEGER, samples_used INTEGER
            )
        """)
        self._conn.commit()

    def record_reversal(self, score: float, conviction: float, regime: str,
                        direction: str, profit: float):
        if self._conn is None:
            return
        won = 1 if profit >= 0 else 0
        self._conn.execute("""
            INSERT INTO reversal_outcomes (score, conviction, regime, direction, profit, won, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (score, conviction, regime, direction, profit, won, time.time()))
        self._conn.commit()

    def get_optimal_thresholds(self) -> Tuple[float, float]:
        """Retorna (reversal_threshold, loss_threshold) recomendados."""
        if self._conn is None:
            return 80.0, 50.0

        cur = self._conn.execute("""
            SELECT score, won FROM reversal_outcomes
            ORDER BY timestamp DESC LIMIT ?
        """, (self.window_size,))
        rows = cur.fetchall()
        if len(rows) < self.min_samples:
            return 80.0, 50.0

        scores = np.array([r["score"] for r in rows])
        outcomes = np.array([r["won"] for r in rows])

        bins = list(range(0, 101, 5))
        best_reversal = 80.0
        best_loss = 50.0

        for threshold_candidate in range(95, 50, -5):
            mask = scores >= threshold_candidate
            if mask.sum() < self.min_samples:
                continue
            win_rate = outcomes[mask].mean()
            if win_rate >= 0.55:
                best_reversal = float(threshold_candidate)
                break

        for threshold_candidate in range(65, 25, -5):
            mask = scores >= threshold_candidate
            if mask.sum() < self.min_samples:
                continue
            win_rate = outcomes[mask].mean()
            if win_rate >= 0.40:
                best_loss = float(threshold_candidate)
                break

        self._conn.execute("""
            INSERT INTO threshold_history (timestamp, reversal_threshold, loss_threshold, window_size, samples_used)
            VALUES (?, ?, ?, ?, ?)
        """, (time.time(), best_reversal, best_loss, self.window_size, len(rows)))
        self._conn.commit()

        if best_reversal != 80.0 or best_loss != 50.0:
            logger.info(f"[AdaptiveThreshold] optimal: reversal={best_reversal:.0f}, loss={best_loss:.0f} "
                        f"(n={len(rows)})")

        return best_reversal, best_loss

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None


"""Meta-Learning System
Records every trade with full context (regime, patterns, conviction, outcome).
Periodically analyzes performance by regime and pattern type to auto-adjust
scoring weights, regime thresholds, and pattern multipliers.
"""
import logging
import sqlite3
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

from src.core.regime_detector import RegimeType, REGIME_PATTERN_MULTIPLIERS
from src.core.strategy_engine import ScoringConfig

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    volume: float = 0.0
    profit: float = 0.0
    score: float = 0.0
    conviction: float = 0.0
    regime: str = "UNKNOWN"
    session: str = "UNKNOWN"
    primary_pattern: Optional[str] = None
    patterns_found: List[str] = field(default_factory=list)
    regime_confidence: float = 0.0
    exit_reason: str = ""
    duration_minutes: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    timing: str = "UNKNOWN"


@dataclass
class RegimePerformance:
    regime: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_profit: float
    avg_profit: float
    profit_factor: float
    patterns_best: List[Tuple[str, float]]
    patterns_worst: List[Tuple[str, float]]


@dataclass
class PatternPerformance:
    pattern_type: str
    total_occurrences: int
    as_primary_count: int
    win_rate: float
    avg_profit: float
    total_profit: float
    best_regime: Optional[str]
    worst_regime: Optional[str]


# Mapping from pattern_type (DB) to REGIME_PATTERN_MULTIPLIERS group key
PATTERN_TO_GROUP: Dict[str, str] = {
    "interval_point": "INTERVAL_POINT",
    "price_interaction": "PRICE_INTERACTION",
    "void_scalp": "VOID_SCALP",
    "bos_zone_retest": "BOS_ZONE",
    "breaker_3_touch": "BREAKER",
    "cycle_full": "CYCLE",
    "liquidity_sweep_ltf": "SWEEP",
    "sweep_wick": "SWEEP",
    "ltf_sweep_confirmation": "SWEEP",
    "sequence_123": "SEQUENCE",
    "fvg_detected": "FVG",
    "order_block_valid": "OB",
    "wyckoff_aligned": "WYCKOFF",
    "harmonic_cycle_aligned": "HARMONIC_CYCLE",
    "pressure_zone_bonus": "PRESSURE_ZONE",
    "triple_confluence": "CYCLE",
}


PATTERN_TYPE_NAME_TO_GROUP: Dict[str, str] = {
    "FVG_BULLISH": "FVG", "FVG_BEARISH": "FVG",
    "OB_BULLISH": "OB", "OB_BEARISH": "OB",
    "BREAKER_BULLISH": "BREAKER", "BREAKER_BEARISH": "BREAKER",
    "SWEEP_BULLISH": "SWEEP", "SWEEP_BEARISH": "SWEEP",
    "CYCLE_BULLISH": "CYCLE", "CYCLE_BEARISH": "CYCLE",
    "SPRING_BULLISH": "WYCKOFF", "UTAD_BEARISH": "WYCKOFF",
    "VOID_SCALP": "VOID_SCALP",
    "BOS_ZONE_RETEST": "BOS_ZONE",
    "HARMONIC_CYCLE_BULLISH": "HARMONIC_CYCLE",
    "HARMONIC_CYCLE_BEARISH": "HARMONIC_CYCLE",
    "PRESSURE_ZONE_BULLISH": "PRESSURE_ZONE",
    "PRESSURE_ZONE_BEARISH": "PRESSURE_ZONE",
    "INTERVAL_POINT_BULLISH": "INTERVAL_POINT",
    "INTERVAL_POINT_BEARISH": "INTERVAL_POINT",
    "PRICE_INTERACTION_BULLISH": "PRICE_INTERACTION",
    "PRICE_INTERACTION_BEARISH": "PRICE_INTERACTION",
    "BOS_ZONE_RETEST_BULLISH": "BOS_ZONE",
    "BOS_ZONE_RETEST_BEARISH": "BOS_ZONE",
}


class MetaLearner:
    def __init__(self, db_path: Optional[Path] = None,
                 base_weights: Optional[ScoringConfig] = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "data" / "meta_learning.db"
        self.db_path = db_path
        self.db_path.parent.mkdir(exist_ok=True)
        self.base_weights = base_weights
        self._init_db()
        self._load_persisted_multipliers()
        self._learning_enabled = True
        self._adjustment_log: List[str] = []
        self._analysis_interval = timedelta(hours=2)
        self._last_analysis: Optional[datetime] = None

    def _init_db(self):
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL,
                    exit_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    volume REAL,
                    profit REAL,
                    score REAL,
                    conviction REAL,
                    regime TEXT,
                    session TEXT,
                    primary_pattern TEXT,
                    patterns_found TEXT DEFAULT '[]',
                    regime_confidence REAL,
                    exit_reason TEXT,
                    duration_minutes INTEGER,
                    timestamp TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            # Migrate old rows that lack the new columns
            for col, col_type in (("stop_loss", "REAL"), ("take_profit", "REAL"), ("timing", "TEXT")):
                try:
                    conn.execute(f"ALTER TABLE trade_records ADD COLUMN {col} {col_type}")
                except Exception:
                    pass
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pattern_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    regime TEXT,
                    total_occurrences INTEGER DEFAULT 0,
                    as_primary INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    total_profit REAL DEFAULT 0.0,
                    updated_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(pattern_type, regime)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS weight_adjustments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    weight_name TEXT NOT NULL,
                    old_value REAL,
                    new_value REAL,
                    reason TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS multiplier_adjustments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_group TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    old_mult REAL,
                    new_mult REAL,
                    reason TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS meta_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS skipped_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    score REAL,
                    conviction REAL,
                    regime TEXT,
                    session TEXT,
                    reason TEXT NOT NULL,
                    pattern_type TEXT,
                    price REAL,
                    timestamp TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS skipped_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    score REAL,
                    conviction REAL,
                    regime TEXT,
                    session TEXT,
                    reason TEXT,
                    pattern_type TEXT,
                    atr_val REAL,
                    pip_value REAL,
                    outcome TEXT DEFAULT 'PENDING',
                    hit_price REAL,
                    hit_at TEXT,
                    timestamp TEXT NOT NULL,
                    resolved_at TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_skipped_outcomes_pending
                ON skipped_outcomes(symbol, outcome)
            """)
            conn.commit()
        finally:
            conn.close()

    def _serialize_multipliers(self) -> str:
        """Serializa REGIME_PATTERN_MULTIPLIERS a JSON con keys de Enum como strings."""
        data: Dict[str, Dict[str, float]] = {}
        for pattern_group, regime_map in REGIME_PATTERN_MULTIPLIERS.items():
            data[pattern_group] = {
                regime.value: mult
                for regime, mult in regime_map.items()
            }
        return json.dumps(data)

    def _deserialize_multipliers(self, raw: str) -> Dict[str, Dict[str, float]]:
        """Deserializa JSON a estructura de multipliers."""
        data: Dict[str, Dict[str, float]] = json.loads(raw)
        return data

    def _save_persisted_multipliers(self):
        """Guarda el estado actual de REGIME_PATTERN_MULTIPLIERS en meta_settings."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            serialized = self._serialize_multipliers()
            conn.execute("""
                INSERT OR REPLACE INTO meta_settings (key, value, updated_at)
                VALUES (?, ?, datetime('now'))
            """, ("regime_pattern_multipliers", serialized))
            conn.commit()
        finally:
            conn.close()

    def _load_persisted_multipliers(self):
        """Carga multipliers persistidos y los aplica a REGIME_PATTERN_MULTIPLIERS."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            row = conn.execute(
                "SELECT value FROM meta_settings WHERE key = ?",
                ("regime_pattern_multipliers",),
            ).fetchone()
            if row is None:
                return
            data = self._deserialize_multipliers(row[0])
            for pattern_group, regime_map in data.items():
                if pattern_group in REGIME_PATTERN_MULTIPLIERS:
                    for regime_str, mult in regime_map.items():
                        try:
                            regime_enum = RegimeType(regime_str)
                            REGIME_PATTERN_MULTIPLIERS[pattern_group][regime_enum] = mult
                        except ValueError:
                            continue
                else:
                    REGIME_PATTERN_MULTIPLIERS[pattern_group] = {}
                    for regime_str, mult in regime_map.items():
                        try:
                            regime_enum = RegimeType(regime_str)
                            REGIME_PATTERN_MULTIPLIERS[pattern_group][regime_enum] = mult
                        except ValueError:
                            continue
            logger.info(
                f"Meta-Learning: {sum(len(v) for v in data.values())} multipliers "
                f"persistidos cargados desde DB"
            )
        finally:
            conn.close()

    def record_trade(self, trade: TradeRecord):
        if not self._learning_enabled:
            return
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            conn.execute("""
                INSERT INTO trade_records
                (symbol, direction, entry_price, exit_price,
                 stop_loss, take_profit,
                 volume, profit,
                 score, conviction, regime, session, primary_pattern,
                 patterns_found, regime_confidence, exit_reason,
                 duration_minutes, timestamp, timing)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.symbol, trade.direction, trade.entry_price,
                trade.exit_price, trade.stop_loss, trade.take_profit,
                trade.volume, trade.profit,
                trade.score, trade.conviction, trade.regime,
                trade.session, trade.primary_pattern,
                json.dumps(trade.patterns_found),
                trade.regime_confidence, trade.exit_reason,
                trade.duration_minutes,
                trade.timestamp.isoformat() if hasattr(trade.timestamp, 'isoformat') else str(trade.timestamp),
                trade.timing,
            ))

            for pattern in trade.patterns_found:
                is_primary = pattern == trade.primary_pattern
                conn.execute("""
                    INSERT INTO pattern_performance
                    (pattern_type, regime, total_occurrences, as_primary,
                     wins, losses, total_profit, updated_at)
                    VALUES (?, ?, 1, ?, ?, ?, ?, datetime('now'))
                    ON CONFLICT(pattern_type, regime) DO UPDATE SET
                        total_occurrences = total_occurrences + excluded.total_occurrences,
                        as_primary = as_primary + excluded.as_primary,
                        wins = wins + excluded.wins,
                        losses = losses + excluded.losses,
                        total_profit = total_profit + excluded.total_profit,
                        updated_at = datetime('now')
                """, (
                    pattern, trade.regime,
                    1 if is_primary else 0,
                    1 if trade.profit > 0 else 0,
                    1 if trade.profit <= 0 else 0,
                    trade.profit,
                ))

            conn.commit()
        finally:
            conn.close()

    def analyze_performance(self, force: bool = False) -> Dict:
        now = datetime.now()
        if self._last_analysis and not force:
            if now - self._last_analysis < self._analysis_interval:
                return {"analyzed": False, "message": "Análisis aún no debido"}

        if not self._has_sufficient_data():
            return {"analyzed": False, "message": "Datos insuficientes para análisis"}

        results = {
            "analyzed": True,
            "by_regime": self._analyze_by_regime(),
            "by_pattern": self._analyze_by_pattern(),
            "adjustments": [],
        }

        regime_adjustments = self._compute_regime_adjustments(results["by_regime"])
        results["adjustments"].extend(regime_adjustments)

        beta_adjustments = self._compute_beta_pattern_adjustments()
        results["adjustments"].extend(beta_adjustments)

        pattern_adjustments = self._compute_pattern_adjustments(results["by_pattern"])
        results["adjustments"].extend(pattern_adjustments)

        self._last_analysis = now
        self._adjustment_log = results["adjustments"]

        if results["adjustments"]:
            logger.info(f"Meta-Learning: {len(results['adjustments'])} ajustes realizados")
            for adj in results["adjustments"]:
                logger.info(f"  Ajuste: {adj}")

        return results

    def _has_sufficient_data(self) -> bool:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            count = conn.execute("SELECT COUNT(*) FROM trade_records").fetchone()[0]
            return count >= 5
        finally:
            conn.close()

    def _analyze_by_regime(self) -> List[RegimePerformance]:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            rows = conn.execute("""
                SELECT regime,
                       COUNT(*) as total,
                       SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN profit <= 0 THEN 1 ELSE 0 END) as losses,
                       SUM(profit) as total_profit,
                       AVG(profit) as avg_profit
                FROM trade_records
                WHERE regime IS NOT NULL
                GROUP BY regime
                ORDER BY total DESC
            """).fetchall()

            performances = []
            for row in rows:
                regime, total, wins, losses, total_profit, avg_profit = row
                wins = wins or 0
                losses = losses or 0
                total_profit = total_profit or 0.0
                avg_profit = avg_profit or 0.0
                win_rate = wins / total if total > 0 else 0.0
                denominator = total_profit - wins * avg_profit
                profit_factor = abs(total_profit / denominator) if abs(denominator) > 1e-9 else 999.0

                pat_rows = conn.execute("""
                    SELECT pattern_type, AVG(total_profit) as avg_pnl
                    FROM pattern_performance
                    WHERE regime = ?
                    GROUP BY pattern_type
                    ORDER BY avg_pnl DESC
                """, (regime,)).fetchall()

                patterns_best = [(p[0], p[1]) for p in pat_rows[:3]]
                patterns_worst = [(p[0], p[1]) for p in pat_rows[-3:]] if len(pat_rows) > 3 else []

                performances.append(RegimePerformance(
                    regime=regime, total_trades=total, wins=wins, losses=losses,
                    win_rate=win_rate, total_profit=total_profit,
                    avg_profit=avg_profit, profit_factor=profit_factor,
                    patterns_best=patterns_best, patterns_worst=patterns_worst,
                ))

            return performances
        finally:
            conn.close()

    def _analyze_by_pattern(self) -> List[PatternPerformance]:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            rows = conn.execute("""
                SELECT pattern_type,
                       SUM(total_occurrences) as total,
                       SUM(as_primary) as primary_count,
                       SUM(wins) as wins,
                       SUM(losses) as losses,
                       SUM(total_profit) as total_profit
                FROM pattern_performance
                GROUP BY pattern_type
                HAVING total >= 3
                ORDER BY total DESC
            """).fetchall()

            performances = []
            for row in rows:
                pattern_type, total, primary_count, wins, losses, total_profit = row
                wins = wins or 0
                losses = losses or 0
                total_profit = total_profit or 0.0
                win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0
                avg_profit = total_profit / total if total > 0 else 0.0

                regime_rows = conn.execute("""
                    SELECT regime, AVG(total_profit) as avg_pnl
                    FROM pattern_performance
                    WHERE pattern_type = ?
                    GROUP BY regime
                    ORDER BY avg_pnl DESC
                """, (pattern_type,)).fetchall()

                best_regime = regime_rows[0][0] if regime_rows else None
                worst_regime = regime_rows[-1][0] if len(regime_rows) > 1 else None

                performances.append(PatternPerformance(
                    pattern_type=pattern_type, total_occurrences=total,
                    as_primary_count=primary_count, win_rate=win_rate,
                    avg_profit=avg_profit, total_profit=total_profit,
                    best_regime=best_regime, worst_regime=worst_regime,
                ))

            return performances
        finally:
            conn.close()

    def _compute_regime_adjustments(self, performances: List[RegimePerformance]) -> List[str]:
        adjustments = []
        for perf in performances:
            if perf.total_trades < 3:
                continue
            try:
                regime = RegimeType(perf.regime)
            except ValueError:
                continue

            if perf.win_rate < 0.3 and perf.total_trades >= 3:
                for pattern_group in REGIME_PATTERN_MULTIPLIERS:
                    old_mult = REGIME_PATTERN_MULTIPLIERS[pattern_group].get(regime, 1.0)
                    new_mult = round(max(old_mult * 0.85, 0.3), 2)
                    REGIME_PATTERN_MULTIPLIERS[pattern_group][regime] = new_mult
                    adjustments.append(
                        f"{pattern_group}@{regime.value}: {old_mult}→{new_mult} (win_rate={perf.win_rate:.0%})"
                    )

            elif perf.win_rate > 0.7 and perf.total_trades >= 3:
                for pattern_group in REGIME_PATTERN_MULTIPLIERS:
                    old_mult = REGIME_PATTERN_MULTIPLIERS[pattern_group].get(regime, 1.0)
                    new_mult = round(min(old_mult * 1.1, 2.0), 2)
                    REGIME_PATTERN_MULTIPLIERS[pattern_group][regime] = new_mult
                    adjustments.append(
                        f"{pattern_group}@{regime.value}: {old_mult}→{new_mult} (win_rate={perf.win_rate:.0%})"
                    )

        if adjustments:
            self._save_persisted_multipliers()

        return adjustments

    def _compute_pattern_adjustments(self, performances: List[PatternPerformance]) -> List[str]:
        adjustments = []
        for perf in performances:
            if perf.total_occurrences < 3:
                continue

            if perf.win_rate < 0.25 and perf.avg_profit < 0:
                adjustments.append(
                    f"PATTERN:{perf.pattern_type} win_rate={perf.win_rate:.0%} profit={perf.avg_profit:.2f} — considerar reducir peso"
                )

            if perf.win_rate > 0.75 and perf.avg_profit > 0:
                adjustments.append(
                    f"PATTERN:{perf.pattern_type} win_rate={perf.win_rate:.0%} profit={perf.avg_profit:.2f} — considerar aumentar peso"
                )

        return adjustments

    def _compute_beta_pattern_adjustments(
        self,
        learning_rate: float = 0.3,
        saturation: int = 20,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        min_samples: int = 3,
    ) -> List[str]:
        adjustments = []
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            rows = conn.execute("""
                SELECT pattern_type, regime,
                       SUM(total_occurrences) as total,
                       SUM(wins) as wins,
                       SUM(losses) as losses
                FROM pattern_performance
                GROUP BY pattern_type, regime
                HAVING total >= ?
            """, (min_samples,)).fetchall()
        finally:
            conn.close()

        combined: Dict[Tuple[str, str], dict] = defaultdict(
            lambda: {"wins": 0, "losses": 0, "n": 0}
        )

        for pat_type, regime_str, total, wins, losses in rows:
            group = PATTERN_TO_GROUP.get(pat_type)
            if group is None:
                continue
            key = (group, regime_str)
            combined[key]["wins"] += (wins or 0)
            combined[key]["losses"] += (losses or 0)
            combined[key]["n"] += total

        conn2 = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            for (group, regime_str), stats in combined.items():
                if stats["n"] < min_samples:
                    continue
                wins = stats["wins"]
                losses = stats["losses"]
                n = stats["n"]

                posterior_mean = (prior_alpha + wins) / (prior_alpha + prior_beta + wins + losses)
                deviation = posterior_mean - 0.5
                scale = min(1.0, n / saturation)

                adjustment = 1.0 + deviation * 2.0 * learning_rate * scale

                try:
                    regime_enum = RegimeType(regime_str)
                except ValueError:
                    continue

                if group not in REGIME_PATTERN_MULTIPLIERS:
                    continue
                if regime_enum not in REGIME_PATTERN_MULTIPLIERS[group]:
                    continue

                old_mult = REGIME_PATTERN_MULTIPLIERS[group][regime_enum]
                new_mult = round(max(0.3, min(2.0, old_mult * adjustment)), 2)

                if abs(new_mult - old_mult) < 0.01:
                    continue

                REGIME_PATTERN_MULTIPLIERS[group][regime_enum] = new_mult

                adjustments.append(
                    f"BETA:{group}@{regime_str}: {old_mult}→{new_mult} "
                    f"(posterior={posterior_mean:.3f}, n={n})"
                )

                conn2.execute("""
                    INSERT INTO multiplier_adjustments
                    (pattern_group, regime, old_mult, new_mult, reason, created_at)
                    VALUES (?, ?, ?, ?, ?, datetime('now'))
                """, (group, regime_str, old_mult, new_mult,
                      f"beta posterior={posterior_mean:.3f} n={n}"))

            # ── Timing-specific adjustments ──
            timing_rows = conn2.execute("""
                SELECT primary_pattern, regime, timing,
                       COUNT(*) as total,
                       SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN profit <= 0 THEN 1 ELSE 0 END) as losses
                FROM trade_records
                WHERE primary_pattern IS NOT NULL
                  AND timing IS NOT NULL
                  AND timing != 'UNKNOWN'
                  AND regime IS NOT NULL
                GROUP BY primary_pattern, regime, timing
                HAVING total >= ?
            """, (min_samples,)).fetchall()

            for pat_name, regime_str, timing, total, wins, losses in timing_rows:
                group = PATTERN_TYPE_NAME_TO_GROUP.get(pat_name)
                if group is None:
                    continue
                try:
                    regime_enum = RegimeType(regime_str)
                except ValueError:
                    continue
                timing_key = f"{group}|{timing}"
                if timing_key not in REGIME_PATTERN_MULTIPLIERS:
                    REGIME_PATTERN_MULTIPLIERS[timing_key] = {
                        r: 1.0 for r in RegimeType
                    }
                if regime_enum not in REGIME_PATTERN_MULTIPLIERS[timing_key]:
                    REGIME_PATTERN_MULTIPLIERS[timing_key][regime_enum] = 1.0

                posterior_mean = (prior_alpha + wins) / (prior_alpha + prior_beta + wins + losses)
                deviation = posterior_mean - 0.5
                scale = min(1.0, total / saturation)
                adjustment = 1.0 + deviation * 2.0 * learning_rate * scale

                old_mult = REGIME_PATTERN_MULTIPLIERS[timing_key][regime_enum]
                new_mult = round(max(0.3, min(2.0, old_mult * adjustment)), 2)
                if abs(new_mult - old_mult) < 0.01:
                    continue
                REGIME_PATTERN_MULTIPLIERS[timing_key][regime_enum] = new_mult
                adjustments.append(
                    f"BETA_TIMING:{timing_key}@{regime_str}: {old_mult}→{new_mult} "
                    f"(posterior={posterior_mean:.3f}, n={total})"
                )
                conn2.execute("""
                    INSERT INTO multiplier_adjustments
                    (pattern_group, regime, old_mult, new_mult, reason, created_at)
                    VALUES (?, ?, ?, ?, ?, datetime('now'))
                """, (timing_key, regime_str, old_mult, new_mult,
                      f"timing beta posterior={posterior_mean:.3f} n={total}"))

            # ── Session-specific adjustments ──
            session_rows = conn2.execute("""
                SELECT primary_pattern, regime, session,
                       COUNT(*) as total,
                       SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN profit <= 0 THEN 1 ELSE 0 END) as losses
                FROM trade_records
                WHERE primary_pattern IS NOT NULL
                  AND session IS NOT NULL
                  AND session != ''
                  AND regime IS NOT NULL
                GROUP BY primary_pattern, regime, session
                HAVING total >= ?
            """, (min_samples,)).fetchall()

            for pat_name, regime_str, session, total, wins, losses in session_rows:
                group = PATTERN_TYPE_NAME_TO_GROUP.get(pat_name)
                if group is None:
                    continue
                try:
                    regime_enum = RegimeType(regime_str)
                except ValueError:
                    continue
                session_key = f"{group}|{session}"
                if session_key not in REGIME_PATTERN_MULTIPLIERS:
                    REGIME_PATTERN_MULTIPLIERS[session_key] = {
                        r: 1.0 for r in RegimeType
                    }
                if regime_enum not in REGIME_PATTERN_MULTIPLIERS[session_key]:
                    REGIME_PATTERN_MULTIPLIERS[session_key][regime_enum] = 1.0

                posterior_mean = (prior_alpha + wins) / (prior_alpha + prior_beta + wins + losses)
                deviation = posterior_mean - 0.5
                scale = min(1.0, total / saturation)
                adjustment = 1.0 + deviation * 2.0 * learning_rate * scale

                old_mult = REGIME_PATTERN_MULTIPLIERS[session_key][regime_enum]
                new_mult = round(max(0.3, min(2.0, old_mult * adjustment)), 2)
                if abs(new_mult - old_mult) < 0.01:
                    continue
                REGIME_PATTERN_MULTIPLIERS[session_key][regime_enum] = new_mult
                adjustments.append(
                    f"BETA_SESSION:{session_key}@{regime_str}: {old_mult}→{new_mult} "
                    f"(posterior={posterior_mean:.3f}, n={total})"
                )
                conn2.execute("""
                    INSERT INTO multiplier_adjustments
                    (pattern_group, regime, old_mult, new_mult, reason, created_at)
                    VALUES (?, ?, ?, ?, ?, datetime('now'))
                """, (session_key, regime_str, old_mult, new_mult,
                      f"session beta posterior={posterior_mean:.3f} n={total}"))

            if adjustments:
                conn2.commit()
                self._save_persisted_multipliers()
        finally:
            conn2.close()

        return adjustments

    def get_best_patterns_for_regime(self, regime: str, top_n: int = 3) -> List[str]:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            rows = conn.execute("""
                SELECT pattern_type, AVG(total_profit) as avg_pnl
                FROM pattern_performance
                WHERE regime = ?
                GROUP BY pattern_type
                HAVING SUM(total_occurrences) >= 3
                ORDER BY avg_pnl DESC
                LIMIT ?
            """, (regime, top_n)).fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()

    def get_worst_patterns_for_regime(self, regime: str, bottom_n: int = 3) -> List[str]:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            rows = conn.execute("""
                SELECT pattern_type, AVG(total_profit) as avg_pnl
                FROM pattern_performance
                WHERE regime = ?
                GROUP BY pattern_type
                HAVING SUM(total_occurrences) >= 3
                ORDER BY avg_pnl ASC
                LIMIT ?
            """, (regime, bottom_n)).fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()

    def get_regime_summary(self, regime: str) -> Dict:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            stats = conn.execute("""
                SELECT COUNT(*),
                       SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END),
                       SUM(profit),
                       AVG(profit)
                FROM trade_records
                WHERE regime = ?
            """, (regime,)).fetchone()

            if not stats or stats[0] == 0:
                return {"regime": regime, "total": 0}

            total, wins, total_profit, avg_profit = stats
            wins = wins or 0
            total_profit = total_profit or 0.0
            avg_profit = avg_profit or 0.0

            return {
                "regime": regime,
                "total": total,
                "wins": wins,
                "win_rate": wins / total if total > 0 else 0.0,
                "total_profit": total_profit,
                "avg_profit": avg_profit,
                "best_patterns": self.get_best_patterns_for_regime(regime),
                "worst_patterns": self.get_worst_patterns_for_regime(regime),
            }
        finally:
            conn.close()

    def get_learning_status(self) -> Dict:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            trade_count = conn.execute("SELECT COUNT(*) FROM trade_records").fetchone()[0]
            pattern_count = conn.execute("SELECT COUNT(DISTINCT pattern_type) FROM pattern_performance").fetchone()[0]
            last_trade = conn.execute("SELECT MAX(timestamp) FROM trade_records").fetchone()[0]

            return {
                "enabled": self._learning_enabled,
                "total_trades_recorded": trade_count,
                "pattern_types_tracked": pattern_count,
                "last_trade": last_trade,
                "last_analysis": self._last_analysis.isoformat() if self._last_analysis else None,
                "recent_adjustments": self._adjustment_log[-5:] if self._adjustment_log else [],
            }
        finally:
            conn.close()

    def cleanup_old_data(self, days: int = 90):
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            conn.execute("""
                DELETE FROM trade_records
                WHERE timestamp < datetime('now', '-' || ? || ' days')
            """, (days,))
            conn.execute("""
                DELETE FROM skipped_signals
                WHERE timestamp < datetime('now', '-' || ? || ' days')
            """, (days,))
            conn.commit()
            logger.info(f"Meta-Learning: datos anteriores a {days} días eliminados")
        finally:
            conn.close()

    def disable_learning(self):
        self._learning_enabled = False
        logger.info("Meta-Learning desactivado")

    def enable_learning(self):
        self._learning_enabled = True
        logger.info("Meta-Learning activado")

    @staticmethod
    def classify_timing(entry_price: float, direction: str,
                        ltf_df: Optional[pd.DataFrame],
                        atr_val: float) -> str:
        if ltf_df is None or ltf_df.empty or atr_val <= 0:
            return "UNKNOWN"
        lookback = min(30, len(ltf_df))
        df_slice = ltf_df.iloc[-lookback:]
        if direction == "BUY":
            lows = df_slice["low"].values
            swing_lows = []
            for i in range(1, len(lows) - 1):
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    swing_lows.append(lows[i])
            if not swing_lows:
                return "UNKNOWN"
            nearest = max(swing_lows)
            distance = max(0.0, entry_price - nearest)
        else:
            highs = df_slice["high"].values
            swing_highs = []
            for i in range(1, len(highs) - 1):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    swing_highs.append(highs[i])
            if not swing_highs:
                return "UNKNOWN"
            nearest = min(swing_highs)
            distance = max(0.0, nearest - entry_price)
        ratio = distance / atr_val
        if ratio < 0.5:
            return "EARLY_IMPULSE"
        if ratio < 1.5:
            return "MID_IMPULSE"
        return "LATE_IMPULSE"

    def record_skipped_signal(self, symbol: str, direction: str, score: float,
                               conviction: float, regime: str, session: str,
                               reason: str, pattern_type: Optional[str] = None,
                               price: Optional[float] = None,
                               entry_price: Optional[float] = None,
                               atr_val: Optional[float] = None,
                               pip_value: Optional[float] = None):
        """
        Registra una señal que fue evaluada pero no se ejecutó.
        Si además se proveen entry_price + atr_val + pip_value, computa
        SL/TP hipotético para seguimiento de outcome.
        """
        if not self._learning_enabled:
            return
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            conn.execute("""
                INSERT INTO skipped_signals
                (symbol, direction, score, conviction, regime, session,
                 reason, pattern_type, price, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """, (symbol, direction, score, conviction, regime, session,
                  reason, pattern_type, price))

            if all(v is not None for v in [entry_price, atr_val, pip_value]) and atr_val > 0 and pip_value > 0:
                if direction == "BUY":
                    sl = entry_price - atr_val * 1.5
                    tp = entry_price + atr_val * 2.0
                else:
                    sl = entry_price + atr_val * 1.5
                    tp = entry_price - atr_val * 2.0

                conn.execute("""
                    INSERT INTO skipped_outcomes
                    (symbol, direction, entry_price, stop_loss, take_profit,
                     score, conviction, regime, session, reason, pattern_type,
                     atr_val, pip_value, outcome, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', datetime('now'))
                """, (symbol, direction, entry_price, sl, tp,
                      score, conviction, regime, session, reason, pattern_type,
                      atr_val, pip_value))
                logger.info(
                    f"[SkippedOutcome] {symbol} {direction} @ {entry_price:.5f}: "
                    f"SL={sl:.5f} TP={tp:.5f} (ATR={atr_val:.5f}) — {reason}"
                )

            conn.commit()
        finally:
            conn.close()

    def poll_skipped_outcomes(self, symbol: str, current_price: float):
        """
        Revisa skipped_outcomes PENDING para *symbol* y los resuelve
        si el precio alcanzó SL o TP.
        """
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            rows = conn.execute("""
                SELECT id, direction, entry_price, stop_loss, take_profit
                FROM skipped_outcomes
                WHERE symbol = ? AND outcome = 'PENDING'
                  AND stop_loss IS NOT NULL AND take_profit IS NOT NULL
            """, (symbol,)).fetchall()

            if not rows:
                return

            now_str = datetime.now().isoformat()
            resolved = 0
            for row_id, direction, entry, sl, tp in rows:
                if sl is None or tp is None:
                    continue
                outcome = None
                hit_price = None
                if direction == "BUY":
                    if current_price <= sl:
                        outcome = "LOSS"
                        hit_price = sl
                    elif current_price >= tp:
                        outcome = "WIN"
                        hit_price = tp
                else:
                    if current_price >= sl:
                        outcome = "LOSS"
                        hit_price = sl
                    elif current_price <= tp:
                        outcome = "WIN"
                        hit_price = tp

                if outcome is None:
                    continue

                conn.execute("""
                    UPDATE skipped_outcomes
                    SET outcome = ?, hit_price = ?, hit_at = ?,
                        resolved_at = datetime('now')
                    WHERE id = ?
                """, (outcome, hit_price, now_str, row_id))
                resolved += 1
                logger.info(
                    f"[SkippedOutcome] {symbol} {direction} {outcome}: "
                    f"entry={entry:.5f} SL={sl:.5f} TP={tp:.5f} hit={hit_price:.5f}"
                )

            if resolved:
                conn.commit()
        finally:
            conn.close()

    def get_skipped_outcomes_analysis(self, symbol: Optional[str] = None) -> Dict:
        """
        Analiza outcomes de señales saltadas agrupando por razón y régimen,
        comparando win rate vs señales reales.
        """
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            total_pending = conn.execute(
                "SELECT COUNT(*) FROM skipped_outcomes WHERE outcome = 'PENDING'"
                + ("" if not symbol else " AND symbol = ?"),
                () if not symbol else (symbol,)
            ).fetchone()[0]

            total_resolved = conn.execute(
                "SELECT COUNT(*) FROM skipped_outcomes WHERE outcome IN ('WIN','LOSS')"
                + ("" if not symbol else " AND symbol = ?"),
                () if not symbol else (symbol,)
            ).fetchone()[0]

            wins = conn.execute(
                "SELECT COUNT(*) FROM skipped_outcomes WHERE outcome = 'WIN'"
                + ("" if not symbol else " AND symbol = ?"),
                () if not symbol else (symbol,)
            ).fetchone()[0]

            by_reason = conn.execute("""
                SELECT reason,
                       COUNT(*) as total,
                       SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses
                FROM skipped_outcomes
                WHERE outcome IN ('WIN','LOSS')
                """ + (" AND symbol = ?" if symbol else "") + """
                GROUP BY reason
                ORDER BY total DESC
            """, () if not symbol else (symbol,)).fetchall()

            by_regime = conn.execute("""
                SELECT regime,
                       COUNT(*) as total,
                       SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses
                FROM skipped_outcomes
                WHERE outcome IN ('WIN','LOSS')
                """ + (" AND symbol = ?" if symbol else "") + """
                GROUP BY regime
                ORDER BY total DESC
            """, () if not symbol else (symbol,)).fetchall()

            return {
                "total_outcomes_tracked": total_pending + total_resolved,
                "pending": total_pending,
                "resolved": total_resolved,
                "wins": wins,
                "losses": total_resolved - wins,
                "win_rate": round(wins / total_resolved, 3) if total_resolved > 0 else None,
                "by_reason": [
                    {"reason": r[0], "total": r[1], "wins": r[2], "losses": r[3],
                     "win_rate": round(r[2] / r[1], 2) if r[1] > 0 else None}
                    for r in by_reason
                ],
                "by_regime": [
                    {"regime": r[0], "total": r[1], "wins": r[2], "losses": r[3],
                     "win_rate": round(r[2] / r[1], 2) if r[1] > 0 else None}
                    for r in by_regime
                ],
            }
        finally:
            conn.close()

    def get_skipped_analysis(self, symbol: Optional[str] = None,
                              min_conviction: float = 0.0) -> Dict:
        """Analiza señales no tomadas agrupadas por razón."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            query = """
                SELECT reason,
                       COUNT(*) as total,
                       AVG(conviction) as avg_conviction,
                       AVG(score) as avg_score,
                       COUNT(DISTINCT pattern_type) as unique_patterns
                FROM skipped_signals
                WHERE conviction >= ?
            """
            params: list = [min_conviction]
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            query += " GROUP BY reason ORDER BY total DESC"

            rows = conn.execute(query, params).fetchall()
            total = sum(r[1] for r in rows)
            return {
                "total_skipped": total,
                "by_reason": [
                    {
                        "reason": r[0], "count": r[1],
                        "avg_conviction": round(r[2], 2) if r[2] else 0,
                        "avg_score": round(r[3], 1) if r[3] else 0,
                        "unique_patterns": r[4],
                    }
                    for r in rows
                ],
            }
        finally:
            conn.close()


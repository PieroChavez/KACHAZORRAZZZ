"""Bayesian Weight Learner
Learns optimal scoring weights from trade outcomes using Beta-Bernoulli
conjugate prior models. Each scoring feature is tracked independently.
Posterior expected values replace static ScoringConfig weights.
"""
import logging
import sqlite3
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureStats:
    name: str
    alpha: float = 1.0
    beta: float = 1.0
    total_weight: float = 0.0
    occurrences: int = 0
    positive_outcomes: int = 0
    negative_outcomes: int = 0
    avg_contribution: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.alpha / (self.alpha + self.beta) if (self.alpha + self.beta) > 0 else 0.5

    @property
    def expected_weight(self) -> float:
        wr = self.win_rate
        return (wr - 0.5) * 2.0

    @property
    def uncertainty(self) -> float:
        n = self.alpha + self.beta
        if n <= 2:
            return 1.0
        p = self.win_rate
        return math.sqrt(p * (1 - p) / n) * 2.0

    def update(self, contributed: float, was_profitable: bool):
        self.occurrences += 1
        self.total_weight += abs(contributed)
        self.avg_contribution = self.total_weight / self.occurrences
        if was_profitable:
            self.alpha += 1.0
            self.positive_outcomes += 1
        else:
            self.beta += 1.0
            self.negative_outcomes += 1


FEATURE_WEIGHT_MAP = {
    "htf_trend_aligned": "HTF_ALIGN",
    "in_discount_zone": "DISCOUNT_ZONE",
    "in_premium_zone": "DISCOUNT_ZONE",
    "valid_market_structure": "MSB",
    "fvg_detected": "FVG",
    "order_block_valid": "OB",
    "breaker_retest": "BREAKER",
    "cycle_full": "CYCLE",
    "wyckoff_full": "WYCKOFF",
    "sequence_123": "SEQUENCE",
    "void_scalp": "VOID_SCALP",
    "bos_zone_retest": "BOS_ZONE",
    "price_est": "PRICE_EST",
    "3rd_movement": "SUB_FRACTAL",
    "psych_price": "PSYCH_PRICE",
    "ltf_sweep_confirmation": "SWEEP_CONF",
    "multiframe_alignment": "MF_ALIGN",
    "triple_confluence": "TRIPLE_CONF",
    "price_interaction": "PRICE_INTERACTION",
    "interval_point": "INTERVAL_POINT",
    "harmonic_cycle": "HARMONIC_CYCLE",
    "pressure_zone": "PRESSURE_ZONE",
    "pause_continuation": "PAUSE",
    "price_trend": "PRICE_TREND",
    "killzone": "KILLZONE",
    "dxy_aligned": "DXY",
    "trb_manipulation": "TRB",
    "vsa_volume_confirmation": "VSA_VOL_CONF",
    "vsa_absorption": "VSA_ABSORPTION",
    "vsa_low_volume_pullback": "VSA_LV_PULLBACK",
}

OUTCOME_WEIGHT_MAP = {
    "body_close_invalid": "BODY_INVALID",
    "no_sweep": "NO_SWEEP",
    "fvg_burned": "FVG_BURNED",
    "spring_body_close": "SPRING_FAIL",
    "ob_mitigated": "OB_MITIGATED",
    "breaker_mitigated": "BREAKER_MITIGATED",
    "breaker_3_touch": "BREAKER_TOUCH_LIMIT",
    "lps_mitigated": "LPS_MITIGATED",
    "retracement_penalty": "RETRACEMENT",
    "order_flow_irregular": "OF_IRREGULAR",
    "price_trend_penalty": "TREND_AGAINST",
    "vsa_climax": "VSA_CLIMAX",
    "vsa_volume_divergence": "VSA_DIVERGENCE",
    "vsa_no_demand_supply": "VSA_NO_DEMAND",
    "dxy_conflict": "DXY_CONFLICT",
    "htf_misalignment_penalty": "HTF_MISALIGN",
    "pattern_confluence_penalty": "OVERLAP_PENALTY",
}


class BayesianWeightLearner:
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "data" / "bayesian_weights.db"
        self.db_path = db_path
        self.db_path.parent.mkdir(exist_ok=True)
        self._features: Dict[str, FeatureStats] = {}
        self._init_db()
        self._load()
        self._min_samples_for_override = 5

    def _init_db(self):
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_stats (
                    name TEXT PRIMARY KEY,
                    alpha REAL DEFAULT 1.0,
                    beta REAL DEFAULT 1.0,
                    total_weight REAL DEFAULT 0.0,
                    occurrences INTEGER DEFAULT 0,
                    positive_outcomes INTEGER DEFAULT 0,
                    negative_outcomes INTEGER DEFAULT 0,
                    avg_contribution REAL DEFAULT 0.0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS outcome_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    signal_id TEXT,
                    symbol TEXT,
                    direction TEXT,
                    profit REAL,
                    feature_key TEXT,
                    contribution REAL
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def _load(self):
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            rows = conn.execute("SELECT * FROM feature_stats").fetchall()
            for row in rows:
                fs = FeatureStats(
                    name=row[0], alpha=row[1], beta=row[2],
                    total_weight=row[3], occurrences=row[4],
                    positive_outcomes=row[5], negative_outcomes=row[6],
                    avg_contribution=row[7],
                )
                self._features[fs.name] = fs
        finally:
            conn.close()

    def _save(self):
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            for name, fs in self._features.items():
                conn.execute("""
                    INSERT OR REPLACE INTO feature_stats
                    (name, alpha, beta, total_weight, occurrences, positive_outcomes, negative_outcomes, avg_contribution)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (fs.name, fs.alpha, fs.beta, fs.total_weight, fs.occurrences,
                      fs.positive_outcomes, fs.negative_outcomes, fs.avg_contribution))
            conn.commit()
        finally:
            conn.close()

    def record_outcome(self, signal_id: str, symbol: str, direction: str,
                       profit: float, breakdown: dict, score_net: float):
        was_profitable = profit > 0
        for key, contrib in breakdown.items():
            if abs(contrib) < 0.5:
                continue
            group = FEATURE_WEIGHT_MAP.get(key) or OUTCOME_WEIGHT_MAP.get(key)
            if group is None:
                if contrib > 0:
                    group = "OTHER_POS"
                elif contrib < 0:
                    group = "OTHER_NEG"
                else:
                    continue
            if contrib > 0:
                feature_name = f"{group}_POS"
            else:
                feature_name = f"{group}_NEG"
            if feature_name not in self._features:
                self._features[feature_name] = FeatureStats(name=feature_name)
            self._features[feature_name].update(abs(contrib), was_profitable)

        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            for key, contrib in breakdown.items():
                if abs(contrib) < 0.5:
                    continue
                conn.execute("""
                    INSERT INTO outcome_log (timestamp, signal_id, symbol, direction, profit, feature_key, contribution)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (datetime.now().isoformat(), signal_id, symbol, direction, profit, key, contrib))
            conn.commit()
        finally:
            conn.close()

        self._save()

    def get_feature_multiplier(self, feature_group: str, base_weight: float) -> float:
        pos_name = f"{feature_group}_POS"
        neg_name = f"{feature_group}_NEG"
        pos_stats = self._features.get(pos_name)
        neg_stats = self._features.get(neg_name)

        if pos_stats and pos_stats.occurrences >= 3:
            pos_wr = pos_stats.win_rate
        else:
            pos_wr = 0.55

        if neg_stats and neg_stats.occurrences >= 3:
            neg_wr = neg_stats.win_rate
        else:
            neg_wr = 0.45

        if pos_stats and pos_stats.occurrences >= self._min_samples_for_override:
            bias = (pos_wr - 0.5) * 2.0
            mult = 1.0 + bias * 0.5
            return max(0.3, min(2.0, mult))

        if neg_stats and neg_stats.occurrences >= self._min_samples_for_override:
            bias = (neg_wr - 0.5) * 2.0
            mult = 1.0 + bias * 0.5
            return max(0.3, min(2.0, mult))

        return 1.0

    def get_conviction_adjustment(self, breakdown: dict, direction: str) -> float:
        if not self._features:
            return 1.0
        total_pos_features = 0
        positive_evidence = 0
        for key, contrib in breakdown.items():
            if abs(contrib) < 0.5:
                continue
            group = FEATURE_WEIGHT_MAP.get(key)
            if group is None:
                continue
            if contrib <= 0:
                continue
            pos_name = f"{group}_POS"
            stats = self._features.get(pos_name)
            if stats and stats.occurrences >= 3:
                total_pos_features += 1
                if stats.win_rate > 0.55:
                    positive_evidence += 1
        if total_pos_features > 0:
            evidence_ratio = positive_evidence / total_pos_features
            return 0.5 + 0.5 * evidence_ratio
        return 1.0

    def get_kelly_fraction(self) -> float:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            rows = conn.execute("""
                SELECT profit FROM outcome_log ORDER BY id DESC LIMIT 50
            """).fetchall()
        finally:
            conn.close()

        if len(rows) < 10:
            return 0.02

        wins = sum(1 for r in rows if r[0] > 0)
        losses = sum(1 for r in rows if r[0] <= 0)
        total = len(rows)

        win_rate = wins / total if total > 0 else 0.5
        avg_win = abs(sum(r[0] for r in rows if r[0] > 0)) / wins if wins > 0 else 1
        avg_loss = abs(sum(r[0] for r in rows if r[0] <= 0)) / losses if losses > 0 else 1

        if avg_loss == 0:
            avg_loss = 1

        b = avg_win / avg_loss
        q = 1 - win_rate
        p = win_rate

        kelly = (p * b - q) / b if b > 0 else 0
        return max(0.01, min(0.05, kelly * 0.25))

    def get_summary(self) -> dict:
        summary = {}
        for name, fs in sorted(self._features.items()):
            if fs.occurrences >= 3:
                summary[name] = {
                    "occurrences": fs.occurrences,
                    "win_rate": round(fs.win_rate, 3),
                    "expected_weight": round(fs.expected_weight, 3),
                    "uncertainty": round(fs.uncertainty, 3),
                }
        return summary


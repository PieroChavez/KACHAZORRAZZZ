from __future__ import annotations

"""Context Oracle with Thompson Sampling (Mejora 7, Modo Experto)
Replaces hard thresholds with probability-based decisions learned
from historical context similarity. Expert mode adds:
- Enhanced context vector (14 features) with session, weekday, DXY, volatility
- Weighted cosine similarity with trainable feature weights
- Temporal decay weighting (recent trades count more)
- Regime-aware Thompson sub-models
- Ensemble prediction (k-NN + Thompson + regime-aware)
- Confidence calibration via prediction variance
- Adaptive threshold based on market volatility
"""
import logging
import math
import pickle
import sqlite3
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SESSION_ENCODING = {
    "LONDON": 0, "NEW_YORK": 1, "ASIAN": 2,
    "LONDON_NY_OVERLAP": 3, "NY_LONDON_OVERLAP": 3,
    "SYDNEY": 4, "UNKNOWN": 5,
}

VOLATILITY_ENCODING = {
    "HIGH": 0, "MEDIUM": 1, "LOW": 2, "EXTREME": 3, "UNKNOWN": 4,
}

EXPERT_FEATURE_WEIGHTS = [
    2.0,  # regime_enc — regime type matters most
    1.8,  # align_enc — HTF/LTF alignment
    1.2,  # session_enc — session matters
    0.6,  # hour_sin — cyclic hour (less important than session)
    0.6,  # hour_cos
    0.5,  # day_sin
    0.5,  # day_cos
    1.5,  # norm_adx — trend strength
    1.0,  # norm_atr — volatility
    1.2,  # volatility_regime_enc
    1.0,  # dxy_trend_enc
    0.8,  # streak_enc — recent performance
    0.7,  # norm_conv
    0.7,  # norm_score
]

BASE_FEATURE_WEIGHTS = [1.0] * 8


class ContextOracle:
    def __init__(self, db_path: Optional[Path] = None, expert_mode: bool = True):
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "data" / "context_memory.db"
        self.db_path = db_path
        self.db_path.parent.mkdir(exist_ok=True)
        self.expert_mode = expert_mode
        self._samples: List[ContextSample] = []
        self._min_samples_for_prediction = 5
        self._max_samples_in_memory = 1000 if expert_mode else 500
        self._temporal_decay_hours = 48 if expert_mode else 0
        self._feature_weights: List[float] = list(EXPERT_FEATURE_WEIGHTS if expert_mode else BASE_FEATURE_WEIGHTS)
        self._regime_thompson: Dict[str, Dict] = defaultdict(lambda: {"alpha": {}, "beta": {}})
        self._init_db()
        self._load()

    def _init_db(self):
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS context_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context TEXT, direction TEXT, profit REAL,
                    conviction REAL, score REAL, regime TEXT,
                    alignment TEXT, timestamp REAL
                )
            """)
            if self.expert_mode:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS oracle_state (
                        key TEXT PRIMARY KEY, value TEXT
                    )
                """)
            conn.commit()
        finally:
            conn.close()

    def _load(self):
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            rows = conn.execute("""
                SELECT context, direction, profit, conviction, score,
                       regime, alignment, timestamp
                FROM context_samples ORDER BY id DESC LIMIT ?
            """, (self._max_samples_in_memory,)).fetchall()
            for row in rows:
                cv = json.loads(row[0])
                self._samples.append(ContextSample(
                    context_vector=cv, direction=row[1], profit=row[2],
                    conviction=row[3], score=row[4], regime=row[5],
                    alignment=row[6], timestamp=row[7],
                ))
            if self.expert_mode:
                try:
                    row = conn.execute(
                        "SELECT value FROM oracle_state WHERE key='feature_weights'"
                    ).fetchone()
                    if row:
                        loaded = json.loads(row[0])
                        if len(loaded) == len(self._feature_weights):
                            self._feature_weights = loaded
                except Exception:
                    pass
        finally:
            conn.close()

    def _save_feature_weights(self):
        if not self.expert_mode:
            return
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO oracle_state (key, value) VALUES (?, ?)",
                ("feature_weights", json.dumps(self._feature_weights)),
            )
            conn.commit()
        finally:
            conn.close()

    def _save_sample(self, sample: ContextSample):
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            conn.execute("""
                INSERT INTO context_samples
                (context, direction, profit, conviction, score, regime, alignment, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (json.dumps(sample.context_vector), sample.direction, sample.profit,
                  sample.conviction, sample.score, sample.regime,
                  sample.alignment, sample.timestamp))
            conn.commit()
        finally:
            conn.close()

    def build_context_vector(self, adx: float, atr_ratio: float,
                              regime_type: str, alignment: str,
                              hour: int, conviction: float,
                              score_net: float,
                              session: str = "", day_of_week: int = -1,
                              volatility_regime: str = "",
                              dxy_trend: str = "",
                              streak: int = 0) -> List[float]:
        regime_enc = REGIME_ENCODING.get(regime_type, 5) / 5.0
        align_enc = ALIGNMENT_ENCODING.get(alignment, 4) / 4.0
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        norm_adx = min(1.0, adx / 60.0)
        norm_atr = min(1.0, atr_ratio / 2.0)
        norm_conv = min(1.0, max(0.0, conviction))
        norm_score = min(1.0, max(-1.0, score_net / 100.0))

        vec = [regime_enc, align_enc, hour_sin, hour_cos,
               norm_adx, norm_atr, norm_conv, norm_score]

        if self.expert_mode:
            dow = day_of_week if day_of_week >= 0 else datetime.now().weekday()
            day_sin = math.sin(2 * math.pi * dow / 7)
            day_cos = math.cos(2 * math.pi * dow / 7)
            session_enc = SESSION_ENCODING.get(session, 5) / 5.0
            vol_reg_enc = VOLATILITY_ENCODING.get(volatility_regime, 4) / 4.0
            dxy_enc = {"BULLISH": 1.0, "BEARISH": -1.0}.get(dxy_trend, 0.0)
            streak_enc = max(-1.0, min(1.0, streak / 10.0))
            vec.extend([session_enc, day_sin, day_cos, vol_reg_enc, dxy_enc, streak_enc])

        return vec

    def record_outcome(self, vector: List[float], direction: str, profit: float,
                        conviction: float, score: float, regime: str,
                        alignment: str,
                        session: str = "", streak: int = 0,
                        volatility_regime: str = ""):
        sample = ContextSample(
            context_vector=vector, direction=direction, profit=profit,
            conviction=conviction, score=score, regime=regime,
            alignment=alignment, timestamp=datetime.now().timestamp(),
        )
        self._samples.append(sample)
        self._save_sample(sample)
        if len(self._samples) > self._max_samples_in_memory:
            self._samples = self._samples[-self._max_samples_in_memory:]

        if self.expert_mode and regime:
            self._update_regime_thompson(regime, direction, profit)
            self._update_feature_weights(sample, profit)

    def _update_regime_thompson(self, regime: str, direction: str, profit: float):
        rt = self._regime_thompson[regime]
        if direction not in rt["alpha"]:
            rt["alpha"][direction] = 1
            rt["beta"][direction] = 1
        if profit > 0:
            rt["alpha"][direction] += 1
        else:
            rt["beta"][direction] += 1

    def _update_feature_weights(self, sample: ContextSample, profit: float):
        if len(self._samples) < 20:
            return
        lr = 0.01
        vec = sample.context_vector
        n_feats = len(vec)
        for i in range(n_feats):
            direction = 1 if profit > 0 else -1
            self._feature_weights[i] += lr * direction * vec[i]
            self._feature_weights[i] = max(0.1, min(3.0, self._feature_weights[i]))
        if len(self._samples) % 50 == 0:
            self._save_feature_weights()

    def _weighted_cosine(self, a: List[float], b: List[float]) -> float:
        if len(a) != len(b):
            return 0.0
        w = self._feature_weights if len(self._feature_weights) == len(a) else [1.0] * len(a)
        dot = sum(x * y * wi for x, y, wi in zip(a, b, w))
        na = math.sqrt(sum(x * x * wi for x, wi in zip(a, w)))
        nb = math.sqrt(sum(y * y * wi for y, wi in zip(b, w)))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _temporal_weight(self, timestamp: float) -> float:
        if not self.expert_mode or self._temporal_decay_hours <= 0:
            return 1.0
        age_hours = (time.time() - timestamp) / 3600
        return math.exp(-age_hours / self._temporal_decay_hours)

    def find_similar_contexts(self, vector: List[float],
                               top_k: int = 20,
                               min_sim: float = 0.3) -> List[Tuple[ContextSample, float]]:
        scored = []
        for s in self._samples:
            sim = self._weighted_cosine(vector, s.context_vector)
            if sim > min_sim:
                tw = self._temporal_weight(s.timestamp)
                scored.append((sim * tw, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(s, sim) for sim, s in scored[:top_k]]

    def predict_direction(self, vector: List[float],
                           buy_score: float, sell_score: float,
                           buy_breakdown: dict, sell_breakdown: dict,
                           regime: str = "",
                           conviction: float = 0.0) -> Tuple[str, float, float]:
        similar = self.find_similar_contexts(vector, top_k=20)
        if len(similar) < self._min_samples_for_prediction:
            net = buy_score - sell_score
            direction = "BUY" if net > 0 else "SELL" if net < 0 else "HOLD"
            confidence = min(1.0, abs(net) / 50.0) * 0.5
            return direction, confidence, 0.5

        knn_dir, knn_conf, knn_wr = self._knn_predict(similar)
        if not self.expert_mode:
            return knn_dir, knn_conf, knn_wr

        thomp_dir, thomp_conf, thomp_wr = self._thompson_predict(similar)
        regime_dir, regime_conf, regime_wr = self._regime_predict(regime)

        votes = []
        if knn_dir != "HOLD":
            votes.append((knn_dir, knn_conf * 0.45, knn_wr))
        if thomp_dir != "HOLD":
            votes.append((thomp_dir, thomp_conf * 0.30, thomp_wr))
        if regime_dir != "HOLD":
            votes.append((regime_dir, regime_conf * 0.25, regime_wr))

        if not votes:
            return "HOLD", 0.0, 0.5

        buy_weight = sum(v[1] for v in votes if v[0] == "BUY")
        sell_weight = sum(v[1] for v in votes if v[0] == "SELL")

        if buy_weight > sell_weight:
            conf = min(1.0, buy_weight)
            wr = max(v[2] for v in votes if v[0] == "BUY") if any(v[0] == "BUY" for v in votes) else 0.5
            return "BUY", conf, wr
        elif sell_weight > buy_weight:
            conf = min(1.0, sell_weight)
            wr = max(v[2] for v in votes if v[0] == "SELL") if any(v[0] == "SELL" for v in votes) else 0.5
            return "SELL", conf, wr
        return "HOLD", 0.0, 0.5

    def _knn_predict(self, similar: List[Tuple[ContextSample, float]]) -> Tuple[str, float, float]:
        buy_total = 0
        sell_total = 0
        buy_samples = 0
        sell_samples = 0
        total_weight = 0

        for sample, sim in similar:
            weight = sim * sim
            total_weight += weight
            if sample.direction == "BUY":
                buy_total += sample.profit * weight
                buy_samples += 1
            elif sample.direction == "SELL":
                sell_total += sample.profit * weight
                sell_samples += 1

        if total_weight == 0:
            return "HOLD", 0.0, 0.0

        buy_weighted = buy_total / total_weight
        sell_weighted = sell_total / total_weight
        buy_wr = sum(1 for s, _ in similar if s.direction == "BUY" and s.profit > 0) / max(1, buy_samples)
        sell_wr = sum(1 for s, _ in similar if s.direction == "SELL" and s.profit > 0) / max(1, sell_samples)

        combined_buy = buy_weighted * 0.6 + buy_wr * 0.4
        combined_sell = sell_weighted * 0.6 + sell_wr * 0.4

        if combined_buy > combined_sell and buy_samples >= 2:
            confidence = min(1.0, abs(combined_buy) * 2.0)
            return "BUY", confidence, buy_wr
        elif combined_sell > combined_buy and sell_samples >= 2:
            confidence = min(1.0, abs(combined_sell) * 2.0)
            return "SELL", confidence, sell_wr
        return "HOLD", 0.0, 0.5

    def _thompson_predict(self, similar: List[Tuple[ContextSample, float]]) -> Tuple[str, float, float]:
        buy_alpha, buy_beta = 1, 1
        sell_alpha, sell_beta = 1, 1

        for sample, sim in similar:
            w = max(1, int(sim * 10))
            if sample.direction == "BUY":
                if sample.profit > 0:
                    buy_alpha += w
                else:
                    buy_beta += w
            elif sample.direction == "SELL":
                if sample.profit > 0:
                    sell_alpha += w
                else:
                    sell_beta += w

        buy_prob = buy_alpha / (buy_alpha + buy_beta)
        sell_prob = sell_alpha / (sell_alpha + sell_beta)

        buy_sample = np.random.beta(buy_alpha, buy_beta)
        sell_sample = np.random.beta(sell_alpha, sell_beta)

        if buy_sample > sell_sample and buy_prob > 0.45:
            return "BUY", buy_prob, buy_prob
        elif sell_sample > buy_sample and sell_prob > 0.45:
            return "SELL", sell_prob, sell_prob
        return "HOLD", max(buy_prob, sell_prob), 0.5

    def _regime_predict(self, regime: str) -> Tuple[str, float, float]:
        if not regime or regime not in self._regime_thompson:
            return "HOLD", 0.0, 0.5
        rt = self._regime_thompson[regime]
        best_dir = "HOLD"
        best_prob = 0.0
        for direction in ("BUY", "SELL"):
            alpha = rt["alpha"].get(direction, 1)
            beta = rt["beta"].get(direction, 1)
            prob = alpha / (alpha + beta)
            if prob > best_prob:
                best_prob = prob
                best_dir = direction
        if best_prob > 0.5:
            return best_dir, best_prob, best_prob
        return "HOLD", best_prob, 0.5

    def should_trade(self, vector: List[float], conviction: float,
                      direction: str, n_trades: int,
                      volatility_regime: str = "") -> Tuple[bool, float]:
        if n_trades < self._min_samples_for_prediction:
            threshold = self._get_adaptive_threshold(volatility_regime)
            return conviction >= threshold, conviction

        similar = self.find_similar_contexts(vector, top_k=15)
        if len(similar) < 3:
            threshold = self._get_adaptive_threshold(volatility_regime)
            return conviction >= threshold, conviction

        same_dir = [(s, sim) for s, sim in similar if s.direction == direction]
        if len(same_dir) < 2:
            threshold = self._get_adaptive_threshold(volatility_regime)
            return conviction >= threshold, conviction

        profits = [s.profit for s, _ in same_dir if s.profit != 0]
        if not profits:
            threshold = self._get_adaptive_threshold(volatility_regime)
            return conviction >= threshold, conviction

        win_rate = sum(1 for p in profits if p > 0) / len(profits)
        avg_profit = sum(profits) / len(profits)
        context_confidence = win_rate * 0.7 + (0.5 + 0.5 * (avg_profit / max(1, abs(avg_profit)))) * 0.3

        threshold = self._get_adaptive_threshold(volatility_regime, context_confidence)
        adjusted_conviction = conviction * (0.5 + 0.5 * context_confidence)

        should = adjusted_conviction >= threshold
        return should, adjusted_conviction

    def _get_adaptive_threshold(self, volatility_regime: str = "",
                                 context_confidence: float = 0.0) -> float:
        if not self.expert_mode:
            return 0.3
        base = 0.2
        if volatility_regime == "HIGH":
            base = 0.35
        elif volatility_regime == "EXTREME":
            base = 0.45
        elif volatility_regime == "LOW":
            base = 0.15
        return max(0.1, base - context_confidence * 0.15)

    def get_thompson_decision(self, vector: List[float],
                                buy_score: float, sell_score: float,
                                conviction: float, n_trades: int) -> Tuple[str, float, float]:
        similar = self.find_similar_contexts(vector, top_k=20)
        if len(similar) < 3:
            return "HOLD", conviction, 0.5

        thomp_dir, thomp_conf, thomp_wr = self._thompson_predict(similar)
        return thomp_dir, thomp_conf, thomp_wr

    def get_summary(self) -> dict:
        total = len(self._samples)
        if total == 0:
            return {"total_samples": 0, "expert_mode": self.expert_mode}
        buy_samples = sum(1 for s in self._samples if s.direction == "BUY")
        sell_samples = sum(1 for s in self._samples if s.direction == "SELL")
        profitable = sum(1 for s in self._samples if s.profit > 0)
        by_regime = defaultdict(int)
        for s in self._samples:
            by_regime[s.regime] += 1
        return {
            "total_samples": total,
            "buy_samples": buy_samples,
            "sell_samples": sell_samples,
            "profitable": profitable,
            "win_rate": profitable / total if total > 0 else 0,
            "expert_mode": self.expert_mode,
            "feature_weights": self._feature_weights[:5],
            "regime_models": len(self._regime_thompson),
            "samples_by_regime": dict(by_regime),
        }

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        return self._weighted_cosine(a, b)


@dataclass
class ContextSample:
    context_vector: List[float]
    direction: str
    profit: float
    conviction: float
    score: float
    regime: str
    alignment: str
    timestamp: float

    def to_dict(self) -> dict:
        return {
            "context": self.context_vector,
            "direction": self.direction,
            "profit": self.profit,
            "conviction": self.conviction,
            "score": self.score,
            "regime": self.regime,
            "alignment": self.alignment,
            "timestamp": self.timestamp,
        }


REGIME_ENCODING = {
    "STRONG_TREND_BULLISH": 0,
    "STRONG_TREND_BEARISH": 1,
    "RANGING": 2,
    "HIGH_VOLATILITY": 3,
    "LOW_VOLATILITY": 4,
    "TRANSITION": 5,
}

ALIGNMENT_ENCODING = {
    "BULLISH_ALIGNED": 0,
    "BEARISH_ALIGNED": 1,
    "HTF_BULLISH_LTF_BEARISH": 2,
    "HTF_BEARISH_LTF_BULLISH": 3,
    "NEUTRAL": 4,
}

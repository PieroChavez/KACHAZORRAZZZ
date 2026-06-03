"""Bayesian Ensemble — Sistema Experto Probabilístico
Mantiene distribuciones Beta por cada "experto" (MarketMap, OrderFlow,
MicroPredictor, RegimeAlignment) y las actualiza post-trade.
Produce convicción calibrada + aprendizaje adaptativo de SL/TP.

No bloquea trades — solo ajusta convicción y logra recomendaciones.
"""
import logging
import sqlite3
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExpertState:
    name: str
    alpha: float = 1.0
    beta: float = 1.0
    total_calls: int = 0
    last_weight: float = 0.5

    @property
    def weight(self) -> float:
        n = self.alpha + self.beta
        if n <= 2:
            return 0.5
        return self.alpha / n

    @property
    def uncertainty(self) -> float:
        n = self.alpha + self.beta
        if n <= 2:
            return 1.0
        return np.sqrt(self.alpha * self.beta / (n * n * (n + 1)))

    def update(self, success: float, strength: float = 1.0):
        self.alpha += success * strength
        self.beta += (1.0 - success) * strength
        self.total_calls += 1


@dataclass
class SLTPLearning:
    regime: str
    sl_distance_pips: List[float] = field(default_factory=list)
    tp_distance_pips: List[float] = field(default_factory=list)
    adverse_excursion_pips: List[float] = field(default_factory=list)
    favorable_excursion_pips: List[float] = field(default_factory=list)
    outcomes: List[float] = field(default_factory=list)

    @property
    def optimal_sl_pips(self) -> Optional[float]:
        if len(self.adverse_excursion_pips) < 3:
            return None
        p90 = np.percentile(self.adverse_excursion_pips, 90)
        return p90 * 1.2

    @property
    def optimal_tp_pips(self) -> Optional[float]:
        if len(self.favorable_excursion_pips) < 3:
            return None
        p50 = np.percentile(self.favorable_excursion_pips, 50)
        return p50


class BayesianEnsemble:
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "data" / "bayesian_ensemble.db"
        self.db_path = db_path
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_db()
        self._experts: Dict[str, ExpertState] = {}
        self._sl_tp_learning: Dict[str, SLTPLearning] = {}
        self._load_state()

    def _init_db(self):
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS expert_state (
                    name TEXT PRIMARY KEY,
                    alpha REAL DEFAULT 1.0,
                    beta REAL DEFAULT 1.0,
                    total_calls INTEGER DEFAULT 0,
                    updated_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sltp_learning (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    sl_pips REAL,
                    tp_pips REAL,
                    adverse_excursion REAL,
                    favorable_excursion REAL,
                    profit REAL,
                    exit_reason TEXT,
                    pattern_type TEXT,
                    timestamp TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ensemble_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    direction TEXT,
                    raw_conviction REAL,
                    adjusted_conviction REAL,
                    expert_weights TEXT,
                    uncertainty REAL,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def _load_state(self):
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            rows = conn.execute("SELECT name, alpha, beta, total_calls FROM expert_state").fetchall()
            for name, alpha, beta, total in rows:
                self._experts[name] = ExpertState(
                    name=name, alpha=alpha, beta=beta, total_calls=total,
                )
            logger.info(f"BayesianEnsemble: {len(self._experts)} expertos cargados")

            rows = conn.execute("""
                SELECT symbol, regime, sl_pips, tp_pips, adverse_excursion,
                       favorable_excursion, profit, exit_reason
                FROM sltp_learning
                ORDER BY timestamp
            """).fetchall()
            for symbol, regime, sl_pips, tp_pips, adv_exc, fav_exc, profit, reason in rows:
                key = f"{symbol}|{regime}"
                if key not in self._sl_tp_learning:
                    self._sl_tp_learning[key] = SLTPLearning(regime=regime)
                rec = self._sl_tp_learning[key]
                if sl_pips and sl_pips > 0:
                    rec.sl_distance_pips.append(sl_pips)
                if tp_pips and tp_pips > 0:
                    rec.tp_distance_pips.append(tp_pips)
                if adv_exc is not None:
                    rec.adverse_excursion_pips.append(adv_exc)
                if fav_exc is not None:
                    rec.favorable_excursion_pips.append(fav_exc)
                rec.outcomes.append(profit)
            logger.info(f"BayesianEnsemble: {len(self._sl_tp_learning)} configs SL/TP cargadas")
        finally:
            conn.close()

    def _save_expert(self, name: str, state: ExpertState):
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            conn.execute("""
                INSERT INTO expert_state (name, alpha, beta, total_calls, updated_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                ON CONFLICT(name) DO UPDATE SET
                    alpha = excluded.alpha,
                    beta = excluded.beta,
                    total_calls = excluded.total_calls,
                    updated_at = datetime('now')
            """, (name, state.alpha, state.beta, state.total_calls))
            conn.commit()
        finally:
            conn.close()

    def get_expert(self, name: str) -> ExpertState:
        if name not in self._experts:
            self._experts[name] = ExpertState(name=name)
        return self._experts[name]

    def evaluate(self, symbol: str, direction: str, regime: str,
                 market_map_conf: float, of_confidence: float,
                 micro_conf: float, regime_alignment: float,
                 raw_conviction: float) -> Dict:
        experts = {
            "market_map": self.get_expert(f"{symbol}|market_map"),
            "order_flow": self.get_expert(f"{symbol}|order_flow"),
            "micro_predictor": self.get_expert(f"{symbol}|micro_predictor"),
            "regime_alignment": self.get_expert(f"{symbol}|regime_alignment"),
        }
        raw_confs = {
            "market_map": market_map_conf,
            "order_flow": of_confidence,
            "micro_predictor": micro_conf,
            "regime_alignment": regime_alignment,
        }
        weights = {name: exp.weight for name, exp in experts.items()}
        total_weight = sum(weights.values())
        if total_weight <= 0:
            return {"adjusted_conviction": raw_conviction, "uncertainty": 1.0,
                    "expert_weights": weights, "experts_active": 0}

        weighted_sum = sum(raw_confs[name] * weights[name] for name in experts)
        ensemble_conf = weighted_sum / total_weight
        confs = list(raw_confs.values())
        uncertainty = float(np.std(confs)) if len(confs) > 1 else 0.5
        n_experts = sum(1 for w in weights.values() if w > 0.1)

        if n_experts < 2 or uncertainty > 0.3:
            adjusted = raw_conviction * 0.85
            logger.info(
                f"[Bayesian] Baja convergencia: {n_experts}/4 expertos, "
                f"uncert={uncertainty:.2f} → conv {raw_conviction:.2f} → {adjusted:.2f}"
            )
        else:
            blend = ensemble_conf * 0.4 + raw_conviction * 0.6
            adjusted = min(1.0, max(0.01, blend))

        result = {
            "adjusted_conviction": round(adjusted, 4),
            "uncertainty": round(uncertainty, 3),
            "expert_weights": {k: round(v, 3) for k, v in weights.items()},
            "ensemble_conf": round(ensemble_conf, 3),
            "experts_active": n_experts,
        }

        self._log_prediction(symbol, direction, raw_conviction, adjusted,
                             weights, uncertainty)
        return result

    def update_post_trade(self, symbol: str, direction: str, regime: str,
                          profit: float, exit_reason: str,
                          market_map_conf: float, of_confidence: float,
                          micro_conf: float, regime_alignment: float,
                          entry_price: float, stop_loss: float,
                          take_profit: float, exit_price: float,
                          pattern_type: Optional[str] = None,
                          pip_size_val: float = 0.0001):
        was_profitable = profit > 0
        sl_pips = abs(entry_price - stop_loss) / pip_size_val if stop_loss else 0
        tp_pips = abs(take_profit - entry_price) / pip_size_val if take_profit else 0

        adverse = 0.0
        favorable = 0.0
        price_range = abs(exit_price - entry_price)
        if direction == "BUY":
            adverse = max(0, entry_price - min(entry_price, exit_price))
            favorable = max(0, max(entry_price, exit_price) - entry_price)
        else:
            adverse = max(0, max(entry_price, exit_price) - entry_price)
            favorable = max(0, entry_price - min(entry_price, exit_price))

        adverse_pips = adverse / pip_size_val
        favorable_pips = favorable / pip_size_val

        experts = {
            "market_map": (market_map_conf, "market_map"),
            "order_flow": (of_confidence, "order_flow"),
            "micro_predictor": (micro_conf, "micro_predictor"),
            "regime_alignment": (regime_alignment, "regime_alignment"),
        }

        for exp_name, (conf, key) in experts.items():
            state = self.get_expert(f"{symbol}|{key}")
            pred_weight = conf * 0.5 + 0.25
            success = 1.0 if was_profitable else 0.0
            state.update(success, strength=pred_weight)
            self._save_expert(f"{symbol}|{key}", state)

        key = f"{symbol}|{regime}"
        if key not in self._sl_tp_learning:
            self._sl_tp_learning[key] = SLTPLearning(regime=regime)
        rec = self._sl_tp_learning[key]
        if sl_pips > 0:
            rec.sl_distance_pips.append(sl_pips)
        if tp_pips > 0:
            rec.tp_distance_pips.append(tp_pips)
        rec.adverse_excursion_pips.append(adverse_pips)
        rec.favorable_excursion_pips.append(favorable_pips)
        rec.outcomes.append(profit)

        self._record_sltp(symbol, regime, direction, sl_pips, tp_pips,
                          adverse_pips, favorable_pips, profit, exit_reason,
                          pattern_type)

        optimal_sl = rec.optimal_sl_pips
        optimal_tp = rec.optimal_tp_pips

        if optimal_sl is not None and sl_pips > 0:
            ratio = optimal_sl / sl_pips
            if ratio > 1.5:
                logger.info(
                    f"[Bayesian] SL {symbol}@{regime}: actual {sl_pips:.0f}p, "
                    f"recomendado ~{optimal_sl:.0f}p ({ratio:.1f}x más amplio)"
                )
            elif ratio < 0.5:
                logger.info(
                    f"[Bayesian] SL {symbol}@{regime}: actual {sl_pips:.0f}p, "
                    f"recomendado ~{optimal_sl:.0f}p ({ratio:.1f}x más justo)"
                )

        if optimal_tp is not None and tp_pips > 0:
            tp_ratio = optimal_tp / tp_pips
            if tp_ratio > 1.5:
                logger.info(
                    f"[Bayesian] TP {symbol}@{regime}: actual {tp_pips:.0f}p, "
                    f"recomendado ~{optimal_tp:.0f}p ({tp_ratio:.1f}x más lejos)"
                )

        ew = {
            "market_map": self.get_expert(f"{symbol}|market_map").weight,
            "order_flow": self.get_expert(f"{symbol}|order_flow").weight,
            "micro_predictor": self.get_expert(f"{symbol}|micro_predictor").weight,
            "regime_alignment": self.get_expert(f"{symbol}|regime_alignment").weight,
        }
        logger.info(
            f"[Bayesian] Post-trade {symbol} {direction} {'WIN' if was_profitable else 'LOSS'}: "
            f"pesos={ {k: f'{v:.2f}' for k, v in ew.items()} }, "
            f"SL_opt={optimal_sl}, TP_opt={optimal_tp}"
        )

    def _log_prediction(self, symbol: str, direction: str,
                        raw_conv: float, adj_conv: float,
                        weights: Dict, uncertainty: float):
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            conn.execute("""
                INSERT INTO ensemble_predictions
                (symbol, direction, raw_conviction, adjusted_conviction,
                 expert_weights, uncertainty, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            """, (symbol, direction, raw_conv, adj_conv,
                  json.dumps(weights), uncertainty))
            conn.commit()
        finally:
            conn.close()

    def _record_sltp(self, symbol: str, regime: str, direction: str,
                     sl_pips: float, tp_pips: float,
                     adverse: float, favorable: float,
                     profit: float, exit_reason: str,
                     pattern_type: Optional[str] = None):
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            conn.execute("""
                INSERT INTO sltp_learning
                (symbol, regime, direction, sl_pips, tp_pips,
                 adverse_excursion, favorable_excursion,
                 profit, exit_reason, pattern_type, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """, (symbol, regime, direction,
                  sl_pips if sl_pips > 0 else None,
                  tp_pips if tp_pips > 0 else None,
                  adverse, favorable, profit, exit_reason, pattern_type))
            conn.commit()
        finally:
            conn.close()

    def get_sltp_recommendation(self, symbol: str, regime: str) -> Dict:
        key = f"{symbol}|{regime}"
        rec = self._sl_tp_learning.get(key)
        if rec is None or len(rec.outcomes) < 3:
            return {"has_data": False}
        return {
            "has_data": True,
            "trades": len(rec.outcomes),
            "optimal_sl_pips": rec.optimal_sl_pips,
            "optimal_tp_pips": rec.optimal_tp_pips,
            "mean_sl_pips": np.mean(rec.sl_distance_pips) if rec.sl_distance_pips else None,
            "mean_adverse_pips": np.mean(rec.adverse_excursion_pips) if rec.adverse_excursion_pips else None,
            "mean_favorable_pips": np.mean(rec.favorable_excursion_pips) if rec.favorable_excursion_pips else None,
        }

    def get_expert_weights(self, symbol: str) -> Dict:
        return {
            name: self.get_expert(f"{symbol}|{name}").weight
            for name in ["market_map", "order_flow", "micro_predictor", "regime_alignment"]
        }

    def get_summary(self) -> Dict:
        return {
            "experts": {
                name: {
                    "alpha": round(s.alpha, 1),
                    "beta": round(s.beta, 1),
                    "weight": round(s.weight, 3),
                    "calls": s.total_calls,
                }
                for name, s in self._experts.items()
            },
            "sltp_configs": len(self._sl_tp_learning),
        }

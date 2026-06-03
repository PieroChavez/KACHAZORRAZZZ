"""Fitness Evaluation for Genetic Optimization (Mejora 5 + Mejora 9)
Uses the full SMC BacktestEngine instead of a simple trend-following simulator,
so the genetic optimizer tunes parameters against the REAL SMC strategy logic.
"""
import json
import logging
from typing import Dict, Optional, Tuple

import pandas as pd

from src.backtest import BacktestEngine, compute_metrics, metrics_to_fitness

logger = logging.getLogger(__name__)

_BT_CACHE: Dict[str, BacktestEngine] = {}


def evaluate_fitness(
    params: Dict,
    df: pd.DataFrame,
    config: Optional[Dict] = None,
    data_bundle: Optional[Dict[str, pd.DataFrame]] = None,
) -> Tuple[float, Dict]:
    """Run SMC backtest with given params and return fitness score.

    Args:
        params: parameter dict (from chromosome decoding)
        df: primary LTF DataFrame (used if data_bundle not provided)
        config: base config dict
        data_bundle: optional dict of {tf_name: DataFrame} for multi-timeframe

    Returns:
        (fitness_score, metrics_dict)
    """
    if config is None:
        config = _default_config(params)

    merged = dict(config)
    merged.update(params)
    engine_key = json.dumps(merged, sort_keys=True)
    if engine_key not in _BT_CACHE:
        engine = BacktestEngine(config=merged)
        _BT_CACHE[engine_key] = engine
    else:
        engine = _BT_CACHE[engine_key]
        engine._reconfigure(merged)

    if data_bundle is None:
        data_bundle = {_detect_tf_name(df): df}

    try:
        result = engine.run(
            data=data_bundle,
            params=params,
            step=5,
            initial_balance=10_000.0,
            max_trades=500,
        )
    except Exception as e:
        logger.warning(f"Backtest error: {e}")
        return -1000.0, {"error": str(e)}

    if not result.trades:
        return -500.0, {"trades": 0, "reason": "no_trades"}

    m = result.metrics
    details = {
        "trades": m.total_trades,
        "total_return_pct": round(m.total_return_pct, 2),
        "win_rate": round(m.win_rate, 4),
        "profit_factor": round(m.profit_factor, 4),
        "sharpe": round(m.sharpe_ratio, 4),
        "sortino": round(m.sortino_ratio, 4),
        "max_drawdown_pct": round(m.max_drawdown_pct * 100, 2),
        "avg_rr": round(m.avg_risk_reward, 4),
        "expectancy": round(m.expectancy, 4),
        "kelly": round(m.kelly_fraction, 4),
    }
    return round(result.fitness, 4), details


def _default_config(params: Dict) -> Dict:
    """Build a minimal config dict from params for the BacktestEngine."""
    strategy_cfg = {
        "adaptive": {
            "min_score_to_trade": params.get("min_score_to_trade", 60),
            "high_confidence_score": params.get("high_confidence_score", 80),
            "min_net_score": params.get("min_net_score", 5),
            "min_reversal_score": params.get("min_reversal_score", 65),
            "conviction_threshold": params.get("conviction_threshold", 0.12),
            "min_conviction_to_trade": 0.15,
            "cooldown_bars_m1": params.get("cooldown_bars_m1", 0),
        },
    }
    return {
        "strategy": strategy_cfg,
        "risk_per_trade": params.get("risk_per_trade", 0.02),
        "atr_multiplier_sl": params.get("atr_multiplier_sl", 1.5),
        "atr_multiplier_tp": params.get("atr_multiplier_tp", 2.0),
        "min_reward_risk_ratio": params.get("min_reward_risk_ratio", 1.5),
    }


def _detect_tf_name(df: pd.DataFrame) -> str:
    """Heuristic to detect timeframe from index frequency."""
    if df is None or df.index is None:
        return "5min"
    try:
        freq = pd.infer_freq(df.index)
        if freq:
            mapping = {
                "1min": "1min", "3min": "3min", "5min": "5min",
                "15min": "15min", "30min": "30min",
                "1H": "1H", "2H": "2H", "4H": "4H",
            }
            for k, v in mapping.items():
                if freq.startswith(k):
                    return v
        delta = (df.index[-1] - df.index[-2]).total_seconds() if len(df) >= 2 else 300
        if delta <= 60:
            return "1min"
        elif delta <= 180:
            return "3min"
        elif delta <= 300:
            return "5min"
        elif delta <= 900:
            return "15min"
        elif delta <= 1800:
            return "30min"
        elif delta <= 3600:
            return "1H"
        elif delta <= 14400:
            return "4H"
        return "5min"
    except Exception:
        return "5min"

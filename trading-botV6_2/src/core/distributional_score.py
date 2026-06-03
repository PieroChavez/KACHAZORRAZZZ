"""Distributional Score System
Represents score as a distribution (mean, std, per-TF scores, convergence)
instead of a single number. Enables conviction-based decisions.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from src.core.regime_detector import RegimeContext
from src.core.pattern_detector import Pattern

logger = logging.getLogger(__name__)


def _categorize_hint(key: str) -> str:
    """Categoriza una clave de score_breakdown en HTF/MID/LTF según su origen."""
    htf_prefixes = (
        "htf_", "dxy_", "wyckoff_", "vsa_", "harmonic_",
        "pressure_zone", "price_trend",
    )
    htf_exact = {
        "in_discount_zone", "in_premium_zone", "valid_market_structure",
        "regime_expansion", "regime_not_accumulation",
    }
    ltf_prefixes = (
        "ltf_", "liquidity_", "void_", "micro_", "sweep_",
        "wick_", "spring_", "no_sweep", "fvg_burned",
        "body_close", "fvg_fresh", "price_est",
    )
    ltf_exact = {
        "fvg_detected", "order_block_valid", "breaker_retest",
        "cycle_full", "sequence_123", "bos_zone_retest",
        "3rd_movement", "psych_price", "triple_confluence",
        "breakr_3_touch",
    }
    if key in htf_exact:
        return "HTF"
    if key in ltf_exact:
        return "LTF"
    for p in htf_prefixes:
        if key.startswith(p):
            return "HTF"
    for p in ltf_prefixes:
        if key.startswith(p):
            return "LTF"
    return "MID"


@dataclass
class DistributionalScore:
    mean: float
    std: float
    per_timeframe: Dict[str, float] = field(default_factory=dict)
    per_group: Dict[str, float] = field(default_factory=dict)
    group_std: Dict[str, float] = field(default_factory=dict)
    convergence: float = 0.0
    conviction: float = 0.0
    ht_lt_alignment: str = "NEUTRAL"
    direction: str = "HOLD"
    notes: List[str] = field(default_factory=list)

    @property
    def is_high_conviction(self) -> bool:
        return self.conviction >= 0.7

    @property
    def is_medium_conviction(self) -> bool:
        return 0.4 <= self.conviction < 0.7

    @property
    def is_low_conviction(self) -> bool:
        return self.conviction < 0.4

    @property
    def risk_adjusted_score(self) -> float:
        return self.mean / (1.0 + self.std) if self.std >= 0 else self.mean


class DistributionalScorer:
    def __init__(self):
        pass

    def compute(self, buy_score: float, sell_score: float,
                 tf_scores: Dict[str, Tuple[float, float]],
                 regime: RegimeContext) -> DistributionalScore:
        net = buy_score - sell_score
        direction = "BUY" if net > 0 else "SELL" if net < 0 else "HOLD"
        abs_net = abs(net)

        per_tf = {}
        per_group = {"HTF": 0.0, "MID": 0.0, "LTF": 0.0}
        group_scores: Dict[str, List[float]] = {"HTF": [], "MID": [], "LTF": []}

        for tf_name, (tf_buy, tf_sell) in tf_scores.items():
            tf_net = tf_buy - tf_sell
            per_tf[tf_name] = tf_net
            if tf_name in ("HTF", "MID", "LTF"):
                group_scores[tf_name].append(tf_net)
            else:
                for group_name, tfs in (("HTF", ["4H", "3H", "2H", "1H"]),
                                         ("MID", ["30min", "15min", "10min"]),
                                         ("LTF", ["5min", "3min", "1min"])):
                    if tf_name in tfs:
                        group_scores[group_name].append(tf_net)
                        break

        for group_name, scores in group_scores.items():
            if scores:
                per_group[group_name] = np.mean(scores)
                per_group[f"{group_name}_std"] = np.std(scores) if len(scores) > 1 else 0.0

        all_nets = list(per_tf.values())
        std = np.std(all_nets) if len(all_nets) > 1 else 0.0
        mean = np.mean(all_nets) if all_nets else 0.0

        convergence = self._compute_convergence(per_group, all_nets)
        conviction = self._compute_conviction(abs_net, std, convergence, regime)
        ht_lt_alignment = self._detect_ht_lt_alignment(per_group)

        if direction != "HOLD":
            multiplier = regime.get_multiplier("FVG")
            conviction = min(1.0, conviction * (0.7 + 0.3 * multiplier))

        notes = [
            f"Score neto: {mean:.1f} ± {std:.1f}",
            f"Convergencia TF: {convergence:.0%}",
            f"Convicción: {conviction:.0%}",
            f"Alineación HTF/LTF: {ht_lt_alignment}",
        ]

        group_std = {}
        for g in ["HTF", "MID", "LTF"]:
            scores = group_scores.get(g, [])
            group_std[g] = np.std(scores) if len(scores) > 1 else 0.0

        return DistributionalScore(
            mean=mean, std=std,
            per_timeframe=per_tf,
            per_group=per_group,
            group_std=group_std,
            convergence=convergence,
            conviction=conviction,
            ht_lt_alignment=ht_lt_alignment,
            direction=direction,
            notes=notes,
        )

    def _compute_convergence(self, per_group: Dict[str, float], all_nets: List[float]) -> float:
        if not all_nets:
            return 0.0
        htf_dir = 1 if per_group.get("HTF", 0) > 0 else -1 if per_group.get("HTF", 0) < 0 else 0
        mid_dir = 1 if per_group.get("MID", 0) > 0 else -1 if per_group.get("MID", 0) < 0 else 0
        ltf_dir = 1 if per_group.get("LTF", 0) > 0 else -1 if per_group.get("LTF", 0) < 0 else 0

        if htf_dir == 0 and mid_dir == 0 and ltf_dir == 0:
            return 0.0

        directions = [d for d in [htf_dir, mid_dir, ltf_dir] if d != 0]
        if not directions:
            return 0.0

        aligned = all(d == directions[0] for d in directions)
        if aligned and len(directions) == 3:
            return 1.0
        if aligned and len(directions) == 2:
            return 0.8

        htf = per_group.get("HTF", 0)
        ltf = per_group.get("LTF", 0)
        mid = per_group.get("MID", 0)

        if ((htf > 0 and ltf < 0) or (htf < 0 and ltf > 0)):
            total_score = abs(htf) + abs(mid) + abs(ltf)
            if total_score <= 0:
                return 0.0
            positive = 0
            if htf > 0:      positive += abs(htf)
            if mid > 0:      positive += abs(mid)
            if ltf > 0:      positive += abs(ltf)
            mid_factor = max(0.5, abs(mid) / max(abs(htf), abs(ltf), 0.001))
            return (positive / total_score) * mid_factor

        total_score = abs(htf) + abs(mid) + abs(ltf)
        positive = 0
        if htf > 0:
            positive += abs(htf)
        if per_group.get("MID", 0) > 0:
            positive += abs(per_group["MID"])
        if ltf > 0:
            positive += abs(ltf)

        return positive / total_score if total_score > 0 else 0.0

    def _compute_conviction(self, abs_net: float, std: float,
                             convergence: float, regime: RegimeContext) -> float:
        if abs_net == 0:
            return 0.0
        adx = getattr(regime, 'adx_value', 0) or 0
        alignment = getattr(regime, 'trend_alignment', 'NEUTRAL') or 'NEUTRAL'

        if adx > 40 and alignment in ("BULLISH_ALIGNED", "BEARISH_ALIGNED"):
            w_mag, w_stab, w_conv, w_reg = 0.40, 0.10, 0.10, 0.40
        elif adx > 25:
            w_mag, w_stab, w_conv, w_reg = 0.35, 0.15, 0.20, 0.30
        else:
            w_mag, w_stab, w_conv, w_reg = 0.38, 0.15, 0.22, 0.25

        score_magnitude = min(1.0, abs_net / 100.0)
        stability = 1.0 / (1.0 + std) if std >= 0 else 1.0
        conv = w_mag * score_magnitude + w_stab * stability + w_conv * convergence + w_reg * regime.confidence
        return min(1.0, max(0.0, conv))

    def _detect_ht_lt_alignment(self, per_group: Dict[str, float]) -> str:
        htf = per_group.get("HTF", 0)
        ltf = per_group.get("LTF", 0)
        if htf > 0 and ltf > 0:
            return "BULLISH_ALIGNED"
        if htf < 0 and ltf < 0:
            return "BEARISH_ALIGNED"
        if htf > 0 and ltf < 0:
            return "HTF_BULLISH_LTF_BEARISH"
        if htf < 0 and ltf > 0:
            return "HTF_BEARISH_LTF_BULLISH"
        return "NEUTRAL"


def _build_group_scores_from_breakdown(
    buy_signal, sell_signal,
) -> Dict[str, Tuple[float, float]]:
    """Construye tf_scores agrupando entradas de score_breakdown en HTF/MID/LTF."""
    groups = {"HTF", "MID", "LTF"}
    group_scores: Dict[str, List[float]] = {g: [] for g in groups}

    for sig in (buy_signal, sell_signal):
        if sig is None:
            continue
        bd = getattr(sig, "score_breakdown", {}) or {}
        for key, value in bd.items():
            group = _categorize_hint(key)
            group_scores.setdefault(group, []).append(value)

    result: Dict[str, Tuple[float, float]] = {}
    for group in groups:
        scores = group_scores.get(group, [])
        if scores:
            buy = sum(s for s in scores if s > 0)
            sell = abs(sum(s for s in scores if s < 0))  # positive magnitude for subtraction
            result[group] = (buy, sell)
    return result


def merge_signals(buy_signal, sell_signal, regime: RegimeContext,
                  scorer: DistributionalScorer, min_conviction: float = 0.4) -> Tuple[DistributionalScore, dict, dict]:
    buy_score = buy_signal.score if buy_signal else 0.0
    sell_score = sell_signal.score if sell_signal else 0.0

    net = buy_score - sell_score
    best_signal = buy_signal if abs(buy_score) >= abs(sell_score) else sell_signal

    tf_scores = _build_group_scores_from_breakdown(buy_signal, sell_signal)

    dist = scorer.compute(buy_score, sell_score, tf_scores, regime)

    if not tf_scores and best_signal:
        bd = getattr(best_signal, "score_breakdown", {}) or {}
        values = list(bd.values())
        dist.mean = net
        dist.std = float(np.std(values)) if len(values) > 1 else 0.0
        if net != 0 and values:
            agreeing = sum(1 for v in values if (v > 0 and net > 0) or (v < 0 and net < 0))
            dist.convergence = agreeing / len(values)
        else:
            dist.convergence = 0.0
        dist.conviction = scorer._compute_conviction(
            abs(net), dist.std, dist.convergence, regime,
        )

    buy_breakdown = buy_signal.score_breakdown if buy_signal else {}
    sell_breakdown = sell_signal.score_breakdown if sell_signal else {}

    return dist, buy_breakdown, sell_breakdown

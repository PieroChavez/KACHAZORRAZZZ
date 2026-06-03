"""Adaptive Scoring System
Transforms base static weights into context-aware dynamic weights
based on market regime, pattern confidence, and inter-TF convergence.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import pandas as pd

from src.core.regime_detector import RegimeContext, RegimeType
from src.core.pattern_detector import Pattern

if TYPE_CHECKING:
    from src.core.strategy_engine import ScoringConfig

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveWeights:
    base: ScoringConfig
    regime: RegimeContext
    adjusted: Dict[str, float] = field(default_factory=dict)
    multipliers: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def get(self, weight_name: str) -> float:
        return self.adjusted.get(weight_name, getattr(self.base, weight_name, 0.0))


PATTERN_TO_WEIGHT_MAP = {
    "fvg_detected": "FVG",
    "order_block_valid": "OB",
    "breaker_retest": "BREAKER",
    "liquidity_sweep_ltf": "SWEEP",
    "wyckoff_phase_c_spring": "WYCKOFF",
    "wyckoff_phase_c_utad": "WYCKOFF",
    "void_scalp_confirmed": "VOID_SCALP",
    "bos_zone_retest": "BOS_ZONE",
    "slip_memory_present": "CYCLE",
    "equidad_sweep_confirmed": "CYCLE",
    "eslabon_breakout": "CYCLE",
    "ltf_bos_with_body": "SEQUENCE",
    "interval_point_bonus": "INTERVAL_POINT",
    "price_interaction_bonus": "PRICE_INTERACTION",
    "harmonic_cycle_aligned": "HARMONIC_CYCLE",
    "pressure_zone_bonus": "PRESSURE_ZONE",
    "trb_manipulation_detected": "TRB",
    "trb_displacement": "TRB",
    "trb_retest": "TRB",
}

CONTEXT_WEIGHT_ADJUSTMENTS = {
    "htf_trend_aligned": {
        RegimeType.STRONG_TREND_BULLISH: 1.3,
        RegimeType.STRONG_TREND_BEARISH: 1.3,
        RegimeType.RANGING: 0.7,
        RegimeType.HIGH_VOLATILITY: 0.8,
        RegimeType.LOW_VOLATILITY: 1.1,
        RegimeType.TRANSITION: 0.8,
    },
    "in_discount_premium_zone": {
        RegimeType.STRONG_TREND_BULLISH: 1.2,
        RegimeType.STRONG_TREND_BEARISH: 1.2,
        RegimeType.RANGING: 0.8,
        RegimeType.HIGH_VOLATILITY: 0.8,
        RegimeType.LOW_VOLATILITY: 1.2,
        RegimeType.TRANSITION: 0.9,
    },
    "valid_market_structure": {
        RegimeType.STRONG_TREND_BULLISH: 1.2,
        RegimeType.STRONG_TREND_BEARISH: 1.2,
        RegimeType.RANGING: 0.7,
        RegimeType.HIGH_VOLATILITY: 0.7,
        RegimeType.LOW_VOLATILITY: 1.2,
        RegimeType.TRANSITION: 0.8,
    },
    "ltf_sweep_confirmation": {
        RegimeType.STRONG_TREND_BULLISH: 1.3,
        RegimeType.STRONG_TREND_BEARISH: 1.3,
        RegimeType.RANGING: 1.2,
        RegimeType.HIGH_VOLATILITY: 1.2,
        RegimeType.LOW_VOLATILITY: 0.9,
        RegimeType.TRANSITION: 1.1,
    },
    "multiframe_alignment": {
        RegimeType.STRONG_TREND_BULLISH: 1.4,
        RegimeType.STRONG_TREND_BEARISH: 1.4,
        RegimeType.RANGING: 0.6,
        RegimeType.HIGH_VOLATILITY: 1.0,
        RegimeType.LOW_VOLATILITY: 0.7,
        RegimeType.TRANSITION: 0.6,
    },
}


class AdaptiveScorer:
    def __init__(self, base_weights: ScoringConfig):
        self.base_weights = base_weights
        self._cache: Dict[str, AdaptiveWeights] = {}

    def compute_weights(self, regime: RegimeContext) -> AdaptiveWeights:
        cache_key = f"{regime.regime.value}_{regime.confidence:.2f}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        adjusted = {}
        multipliers = {}

        for weight_name in dir(self.base_weights):
            if weight_name.startswith("_"):
                continue
            base_val = getattr(self.base_weights, weight_name)
            if not isinstance(base_val, (int, float)):
                continue

            mult = 1.0

            if weight_name in PATTERN_TO_WEIGHT_MAP:
                pattern_group = PATTERN_TO_WEIGHT_MAP[weight_name]
                mult = regime.get_multiplier(pattern_group)

            elif weight_name in CONTEXT_WEIGHT_ADJUSTMENTS:
                regime_adjust = CONTEXT_WEIGHT_ADJUSTMENTS[weight_name]
                mult = regime_adjust.get(regime.regime, 1.0)

            adjusted_val = base_val * mult
            adjusted[weight_name] = round(adjusted_val, 1)
            multipliers[weight_name] = mult

        result = AdaptiveWeights(
            base=self.base_weights,
            regime=regime,
            adjusted=adjusted,
            multipliers=multipliers,
            notes=[f"Pesos ajustados por régimen {regime.regime.value} (confianza={regime.confidence:.0%})"],
        )

        self._cache[cache_key] = result
        if len(self._cache) > 50:
            self._cache.clear()

        return result

    def get_pattern_regime_multiplier(self, pattern_type_name: str, regime: RegimeContext) -> float:
        for weight_name, pattern_group in PATTERN_TO_WEIGHT_MAP.items():
            if pattern_type_name.upper().startswith(pattern_group) or pattern_group in pattern_type_name.upper():
                return regime.get_multiplier(pattern_group)
        return 1.0

    def adjust_for_confidence(self, base_score: float, pattern: Optional[Pattern],
                               regime: RegimeContext, convergence: float) -> float:
        confidence_factor = 0.5 + 0.5 * regime.confidence
        pattern_factor = pattern.confidence if pattern is not None else 0.5
        convergence_factor = 0.5 + 0.5 * convergence
        combined = confidence_factor * 0.4 + pattern_factor * 0.3 + convergence_factor * 0.3
        return base_score * combined

"""Continuous Decision System
Replaces binary trade decisions with continuous conviction-based sizing.
Position size is a smooth function of conviction, regime, and risk context.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd

from src.core.distributional_score import DistributionalScore
from src.core.regime_detector import RegimeContext, RegimeType
from src.utils.helpers import atr

logger = logging.getLogger(__name__)


@dataclass
class ContinuousDecision:
    direction: str
    conviction: float
    suggested_volume_pct: float
    sl_width_multiplier: float
    tp_width_multiplier: float
    scale_entries_count: int
    entry_aggressiveness: str
    max_risk_pct: float
    notes: List[str] = field(default_factory=list)

    @property
    def should_trade(self) -> bool:
        return self.conviction >= 0.3 and self.direction in ("BUY", "SELL")


CONVICTION_VOLUME_MAP = [
    (0.0, 0.0),
    (0.2, 0.2),
    (0.3, 0.4),
    (0.4, 0.6),
    (0.5, 0.8),
    (0.6, 1.0),
    (0.7, 1.2),
    (0.8, 1.5),
    (0.9, 2.0),
    (1.0, 3.0),
]


class ContinuousDecider:
    def __init__(self, base_risk_pct: float = 0.02):
        self.base_risk_pct = base_risk_pct

    def decide(self, dist: Optional[DistributionalScore], regime: RegimeContext,
                profile=None, ltf_df: Optional[pd.DataFrame] = None) -> ContinuousDecision:
        if dist is None:
            return ContinuousDecision(
                direction="HOLD", conviction=0.0,
                suggested_volume_pct=0.0, sl_width_multiplier=1.0,
                tp_width_multiplier=1.0, scale_entries_count=0,
                entry_aggressiveness="none", max_risk_pct=0.0,
                notes=["Sin distribución — no operar"],
            )
        conviction = dist.conviction
        direction = dist.direction

        if conviction < 0.2:
            return ContinuousDecision(
                direction="HOLD", conviction=conviction,
                suggested_volume_pct=0.0, sl_width_multiplier=1.0,
                tp_width_multiplier=1.0, scale_entries_count=0,
                entry_aggressiveness="none", max_risk_pct=0.0,
                notes=["Convicción demasiado baja para operar"],
            )

        volume_pct = self._conviction_to_volume(conviction)
        sl_mult, tp_mult = self._get_sl_tp_multipliers(conviction, regime)
        scale_count = self._get_scale_count(conviction, regime)
        aggressiveness = self._get_aggressiveness(conviction, regime)
        max_risk = self._get_max_risk_pct(conviction, regime)

        sl_mult, tp_mult = self._adjust_for_regime(sl_mult, tp_mult, regime)

        if regime.regime == RegimeType.HIGH_VOLATILITY:
            sl_mult *= 1.3
            tp_mult *= 1.5
            volume_pct *= 0.6
            notes = [f"Alta volatilidad (ATR ratio={regime.atr_ratio:.1f}): SL ampliado, volumen reducido"]
        elif regime.regime == RegimeType.LOW_VOLATILITY:
            sl_mult *= 0.8
            tp_mult *= 0.9
            scale_count = max(1, scale_count - 1)
            notes = [f"Baja volatilidad: SL más ajustado, escalas reducidas"]
        elif regime.regime in (RegimeType.STRONG_TREND_BULLISH, RegimeType.STRONG_TREND_BEARISH):
            tp_mult *= 1.3
            volume_pct *= 1.2
            notes = [f"Tendencia fuerte: TP extendido, volumen +20%"]
        elif regime.regime == RegimeType.RANGING:
            sl_mult *= 0.9
            tp_mult *= 0.7
            volume_pct *= 0.7
            notes = [f"Mercado en rango: reducción general de exposición"]
        else:
            notes = [f"Régimen {regime.regime.value}: decisiones estándar"]

        if dist.std > dist.mean * 0.5 and dist.mean > 0:
            volume_pct *= 0.7
            notes.append("Alta dispersión entre TFs: volumen reducido")

        entry_notes = [
            f"Convicción: {conviction:.0%}",
            f"Volumen sugerido: {volume_pct:.1f}× base",
            f"SL mult: {sl_mult:.1f}, TP mult: {tp_mult:.1f}",
            f"Escalas: {scale_count}",
            f"Agresividad: {aggressiveness}",
        ]

        return ContinuousDecision(
            direction=direction,
            conviction=conviction,
            suggested_volume_pct=volume_pct,
            sl_width_multiplier=sl_mult,
            tp_width_multiplier=tp_mult,
            scale_entries_count=scale_count,
            entry_aggressiveness=aggressiveness,
            max_risk_pct=max_risk,
            notes=notes + entry_notes,
        )

    def _conviction_to_volume(self, conviction: float) -> float:
        for threshold, volume in reversed(CONVICTION_VOLUME_MAP):
            if conviction >= threshold:
                return volume
        return 0.0

    def _get_sl_tp_multipliers(self, conviction: float, regime: RegimeContext) -> Tuple[float, float]:
        if conviction >= 0.8:
            return 0.8, 1.5
        if conviction >= 0.6:
            return 0.9, 1.2
        if conviction >= 0.4:
            return 1.0, 1.0
        return 1.2, 0.8

    def _get_scale_count(self, conviction: float, regime: RegimeContext) -> int:
        if conviction >= 0.9:
            return 5
        if conviction >= 0.7:
            return 3
        if conviction >= 0.5:
            return 2
        return 1

    def _get_aggressiveness(self, conviction: float, regime: RegimeContext) -> str:
        if conviction >= 0.8:
            return "aggressive"
        if conviction >= 0.5:
            return "moderate"
        return "conservative"

    def _get_max_risk_pct(self, conviction: float, regime: RegimeContext) -> float:
        if regime.regime == RegimeType.HIGH_VOLATILITY:
            return min(0.02, self.base_risk_pct * 0.6)
        if conviction >= 0.8:
            return min(0.04, self.base_risk_pct * 2.0)
        if conviction >= 0.6:
            return self.base_risk_pct * 1.5
        return self.base_risk_pct

    def _adjust_for_regime(self, sl_mult: float, tp_mult: float,
                            regime: RegimeContext) -> Tuple[float, float]:
        if regime.is_expanding:
            return sl_mult * 1.2, tp_mult * 1.3
        if regime.is_compressed:
            return sl_mult * 0.8, tp_mult * 0.9
        return sl_mult, tp_mult

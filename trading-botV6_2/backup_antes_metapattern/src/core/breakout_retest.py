"""Breakout Retest Detector — Consolidación→Ruptura→Retest (MEJORA SCALPING)
Detecta zonas consolidadas (3+ toques en el mismo nivel) que son rotas
y luego retestadas. Genera señales de alta probabilidad en el retest.
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import pandas as pd

from src.core.zone_state_tracker import ZoneStateTracker, ZoneRecord, ZoneStatus

logger = logging.getLogger(__name__)


@dataclass
class BreakoutRetestSignal:
    active: bool
    direction: str
    breakout_level: float
    retest_level: float
    zone_touches: int
    confidence: float
    score_bonus: float
    notes: List[str] = field(default_factory=list)


class BreakoutRetestDetector:
    """Detecta secuencias de consolidación → ruptura → retest.

    Una zona que ha sido tocada 3+ veces, luego el precio la rompe
    (cierre fuera), y luego vuelve a testearla, genera una señal
    forzada de alta probabilidad en la dirección de la ruptura.
    """

    def __init__(self, min_touches_for_consolidation: int = 3,
                 retest_tolerance_pct: float = 0.10,
                 breakout_confirm_candles: int = 2,
                 score_bonus: float = 30.0):
        self._min_touches = min_touches_for_consolidation
        self._retest_tolerance = retest_tolerance_pct
        self._confirm_candles = breakout_confirm_candles
        self._score_bonus = score_bonus

    def check(self, df: pd.DataFrame,
              zone_tracker: ZoneStateTracker,
              symbol: str) -> List[BreakoutRetestSignal]:
        signals = []
        broken = zone_tracker.get_broken_zones(symbol, retest_only=False)
        for zone in broken:
            if zone.break_direction is None:
                continue
            retesting = zone.status == ZoneStatus.BROKEN_RETEST
            if not retesting:
                continue
            dist = abs(df["close"].iloc[-1] - zone.price_level)
            atr_val = self._atr(df)
            tolerance = atr_val * self._retest_tolerance
            if dist > tolerance:
                continue
            retest_confirmed = self._confirm_retest(df, zone)
            if not retest_confirmed:
                continue
            direction = zone.break_direction
            conf = min(0.95, 0.5 + zone.touch_count * 0.12)
            sig = BreakoutRetestSignal(
                active=True,
                direction=direction,
                breakout_level=zone.break_price or zone.price_level,
                retest_level=zone.price_level,
                zone_touches=zone.touch_count,
                confidence=conf,
                score_bonus=self._score_bonus,
                notes=[
                    f"Breakout retest: zona tocada {zone.touch_count}x, rota {direction}, "
                    f"retest en {zone.price_level:.5f}",
                ],
            )
            signals.append(sig)
            logger.info(f"[{symbol}] BreakoutRetest {direction} @ {zone.price_level:.5f} "
                        f"(touches={zone.touch_count}, confidence={conf:.0%})")

        exhausted = zone_tracker.get_exhausted_zones(symbol)
        for zone in exhausted:
            dist = abs(df["close"].iloc[-1] - zone.price_level)
            atr_val = self._atr(df)
            tolerance = atr_val * self._retest_tolerance
            if dist > tolerance * 2:
                continue
            will_break = self._detect_breakout(df, zone)
            if not will_break:
                continue
            dir_vote = "BUY" if df["close"].iloc[-1] > zone.zone_high else "SELL"
            zone_tracker.detect_breakout(symbol, df["close"].iloc[-1], dir_vote)
            signals.append(BreakoutRetestSignal(
                active=False,
                direction=dir_vote,
                breakout_level=zone.price_level,
                retest_level=zone.price_level,
                zone_touches=zone.touch_count,
                confidence=0.4,
                score_bonus=0,
                notes=[f"Breakout detectado desde zona exhausta, esperando retest"],
            ))

        return signals

    def _confirm_retest(self, df: pd.DataFrame, zone: ZoneRecord) -> bool:
        if len(df) < self._confirm_candles + 1:
            return False
        for i in range(self._confirm_candles):
            idx = -(i + 1)
            c = df.iloc[idx]
            in_zone = zone.zone_low <= c["close"] <= zone.zone_high
            if not in_zone:
                return False
        return True

    def _detect_breakout(self, df: pd.DataFrame, zone: ZoneRecord) -> bool:
        if len(df) < self._confirm_candles:
            return False
        close = df["close"].values[-self._confirm_candles:]
        if zone.direction == "BUY":
            return all(c > zone.zone_high for c in close)
        else:
            return all(c < zone.zone_low for c in close)

    def _atr(self, df: pd.DataFrame, period: int = 14) -> float:
        from src.utils.helpers import atr as _atr_fn
        if len(df) < period + 1:
            return 0.0
        return float(_atr_fn(df, period).iloc[-1])

"""Micro Phase Detector — Fase del micro-movimiento M1 (MEJORA SCALPING)
Detecta en qué punto del micro-movimiento nos encontramos usando las últimas
3-5 velas M1. Permite saber si estamos al inicio de un impulso, en medio de
un retroceso, o en una fase madura donde no conviene entrar.
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

from src.utils.helpers import atr

logger = logging.getLogger(__name__)


class MicroPhase(Enum):
    SWEEP_JUST_COMPLETED = "SWEEP_JUST_COMPLETED"
    FIRST_RETRACE_CANDLE = "FIRST_RETRACE_CANDLE"
    RETRACE_CONFIRMED = "RETRACE_CONFIRMED"
    IMPULSE_STARTING = "IMPULSE_STARTING"
    IMPULSE_MATURE = "IMPULSE_MATURE"
    RETRACE_MID = "RETRACE_MID"
    INDECISION = "INDECISION"
    BREAKOUT = "BREAKOUT"
    BREAKOUT_RETEST = "BREAKOUT_RETEST"


ENTRY_ALLOWED_PHASES = {
    MicroPhase.SWEEP_JUST_COMPLETED,
    MicroPhase.FIRST_RETRACE_CANDLE,
    MicroPhase.RETRACE_CONFIRMED,
    MicroPhase.IMPULSE_STARTING,
    MicroPhase.BREAKOUT_RETEST,
}

PHASE_WEIGHT = {
    MicroPhase.IMPULSE_STARTING: 1.2,
    MicroPhase.FIRST_RETRACE_CANDLE: 1.1,
    MicroPhase.RETRACE_CONFIRMED: 1.0,
    MicroPhase.SWEEP_JUST_COMPLETED: 1.0,
    MicroPhase.BREAKOUT_RETEST: 1.3,
    MicroPhase.BREAKOUT: 0.6,
    MicroPhase.IMPULSE_MATURE: 0.3,
    MicroPhase.RETRACE_MID: 0.2,
    MicroPhase.INDECISION: 0.1,
}


@dataclass
class PhaseResult:
    phase: MicroPhase
    direction: str
    confidence: float
    sweep_level: Optional[float] = None
    retrace_depth_pct: float = 0.0
    fvg_detected: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def allows_entry(self) -> bool:
        return self.phase in ENTRY_ALLOWED_PHASES

    @property
    def weight(self) -> float:
        return PHASE_WEIGHT.get(self.phase, 0.5)

    def phase_bonus(self, score: float) -> float:
        if not self.allows_entry:
            return 0.0
        if self.phase == MicroPhase.BREAKOUT_RETEST:
            return score * 0.25
        return score * (self.weight - 1.0)


_SAFE_PHASE_RESULT = PhaseResult(MicroPhase.INDECISION, "NEUTRAL", 0.0, notes=["Fallback seguro"])


class MicroPhaseDetector:
    """Analiza las últimas N velas M1 y clasifica la micro-fase actual."""

    def __init__(self, lookback: int = 5, swing_window: int = 3,
                 retrace_threshold: float = 0.382,
                 sweep_body_ratio: float = 0.6):
        self._lookback = lookback
        self._swing_window = swing_window
        self._retrace_threshold = retrace_threshold
        self._sweep_body_ratio = sweep_body_ratio

    def detect(self, df: pd.DataFrame) -> PhaseResult:
        try:
            if df is None or len(df) < self._lookback + 2:
                return PhaseResult(MicroPhase.INDECISION, "NEUTRAL", 0.0, notes=["Datos insuficientes"])

            close = df["close"].values
            high = df["high"].values
            low = df["low"].values
            open_p = df["open"].values

            sweep_high = self._detect_sweep_high(df)
            sweep_low = self._detect_sweep_low(df)

            fvg_bullish = self._detect_bullish_fvg(df)
            fvg_bearish = self._detect_bearish_fvg(df)

            if sweep_high and fvg_bearish:
                return PhaseResult(
                    MicroPhase.IMPULSE_STARTING, "SELL", 0.85,
                    sweep_level=sweep_high, fvg_detected=True,
                    notes=["Sweep alto + FVG bajista → inicio impulso bajista"],
                )
            if sweep_low and fvg_bullish:
                return PhaseResult(
                    MicroPhase.IMPULSE_STARTING, "BUY", 0.85,
                    sweep_level=sweep_low, fvg_detected=True,
                    notes=["Sweep bajo + FVG alcista → inicio impulso alcista"],
                )

            if sweep_high:
                return PhaseResult(
                    MicroPhase.SWEEP_JUST_COMPLETED, "SELL", 0.70,
                    sweep_level=sweep_high,
                    notes=["Sweep de máximo recién completado"],
                )
            if sweep_low:
                return PhaseResult(
                    MicroPhase.SWEEP_JUST_COMPLETED, "BUY", 0.70,
                    sweep_level=sweep_low,
                    notes=["Sweep de mínimo recién completado"],
                )

            if self._is_retrace_just_started(df):
                dir = "BUY" if close[-1] > close[-2] else "SELL"
                return PhaseResult(
                    MicroPhase.FIRST_RETRACE_CANDLE, dir, 0.60,
                    notes=["Primera vela de retroceso detectada"],
                )

            if self._is_retrace_confirmed(df):
                dir = "BUY" if close[-1] > close[-2] else "SELL"
                depth = self._calc_retrace_depth(df)
                return PhaseResult(
                    MicroPhase.RETRACE_CONFIRMED, dir, 0.65,
                    retrace_depth_pct=depth,
                    notes=[f"Retroceso confirmado ({depth:.0%} del impulso anterior)"],
                )

            if self._is_in_retrace_mid(df):
                dir = "BUY" if close[-1] > close[-2] else "SELL"
                return PhaseResult(
                    MicroPhase.RETRACE_MID, dir, 0.30,
                    notes=["Retroceso a medio camino — no entrar"],
                )

            if self._is_impulse_mature(df):
                dir = "BUY" if close[-1] > close[-2] else "SELL"
                return PhaseResult(
                    MicroPhase.IMPULSE_MATURE, dir, 0.25,
                    notes=["Impulso maduro — no entrar"],
                )

            dir = "BUY" if close[-1] > close[-2] else "SELL" if close[-1] < close[-2] else "NEUTRAL"
            return PhaseResult(MicroPhase.INDECISION, dir, 0.20,
                               notes=["Micro-fase indecisa — esperar"])

        except Exception:
            logger.exception("MicroPhaseDetector.detect() fallback")
            return _SAFE_PHASE_RESULT

    def _detect_sweep_high(self, df: pd.DataFrame) -> Optional[float]:
        if len(df) < self._swing_window + 2:
            return None
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        open_p = df["open"].values
        n = len(df)
        for i in range(self._swing_window, n):
            prev_highs = high[i - self._swing_window:i]
            if high[i] > max(prev_highs) and close[i] < open_p[i]:
                body = abs(close[i] - open_p[i])
                rng = high[i] - low[i]
                if rng > 0 and body / rng > self._sweep_body_ratio:
                    return high[i]
        return None

    def _detect_sweep_low(self, df: pd.DataFrame) -> Optional[float]:
        if len(df) < self._swing_window + 2:
            return None
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        open_p = df["open"].values
        n = len(df)
        for i in range(self._swing_window, n):
            prev_lows = low[i - self._swing_window:i]
            if low[i] < min(prev_lows) and close[i] > open_p[i]:
                body = abs(close[i] - open_p[i])
                rng = high[i] - low[i]
                if rng > 0 and body / rng > self._sweep_body_ratio:
                    return low[i]
        return None

    def _detect_bullish_fvg(self, df: pd.DataFrame) -> bool:
        if len(df) < 4:
            return False
        c1, c2, c3 = df.iloc[-4], df.iloc[-3], df.iloc[-2]
        return c1["high"] < c2["low"]

    def _detect_bearish_fvg(self, df: pd.DataFrame) -> bool:
        if len(df) < 4:
            return False
        c2, c3 = df.iloc[-3], df.iloc[-2]
        return c2["low"] > c3["high"]

    def _is_retrace_just_started(self, df: pd.DataFrame) -> bool:
        if len(df) < 4:
            return False
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        c1, c2, c3 = close[-4], close[-3], close[-2]
        nc = close[-1]
        imv = (high[-3] - low[-3]) * 2
        if abs(nc - c2) < imv and abs(c2 - c1) > imv * 1.5:
            dir_change = (nc > c2 and c2 < c1) or (nc < c2 and c2 > c1)
            return dir_change
        return False

    def _is_retrace_confirmed(self, df: pd.DataFrame) -> bool:
        if len(df) < 5:
            return False
        close = df["close"].values
        c1, c2, c3 = close[-5], close[-4], close[-3]
        nc1, nc2 = close[-2], close[-1]
        trend_dir = "UP" if c3 > c2 > c1 else "DOWN" if c3 < c2 < c1 else None
        if trend_dir is None:
            return False
        if trend_dir == "UP":
            retrace_start = nc1 < c3
            retrace_confirm = nc2 < nc1 and nc2 < c3
        else:
            retrace_start = nc1 > c3
            retrace_confirm = nc2 > nc1 and nc2 > c3
        return retrace_start and retrace_confirm

    def _is_in_retrace_mid(self, df: pd.DataFrame) -> bool:
        if len(df) < 6:
            return False
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        mid = len(df) // 2
        left_range = high[mid - 2] - low[mid - 2]
        right_range = high[-1] - low[-1]
        left_dir = "UP" if close[mid - 1] > close[mid - 2] else "DOWN"
        right_dir = "UP" if close[-1] > close[-2] else "DOWN"
        return left_dir != right_dir and right_range < left_range * 0.7

    def _is_impulse_mature(self, df: pd.DataFrame) -> bool:
        if len(df) < 6:
            return False
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        mid = len(df) // 2
        left_impulse = abs(close[mid] - close[mid - 3])
        right_impulse = abs(close[-1] - close[-3])
        left_range = high[mid] - low[mid - 3]
        right_range = high[-1] - low[-3]
        return right_impulse > left_impulse * 1.5 and right_range > left_range * 1.5

    def _calc_retrace_depth(self, df: pd.DataFrame) -> float:
        if len(df) < 5:
            return 0.0
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        c1, c2 = close[-5], close[-4]
        impulse = abs(c2 - c1)
        if impulse == 0:
            return 0.0
        retrace = abs(close[-1] - c2)
        return min(1.0, retrace / impulse)

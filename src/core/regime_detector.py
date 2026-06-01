"""Market Regime Detector
Classifies the market into 6 regimes based on multi-factor analysis.
Each regime has a confidence score and produces pattern multipliers.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

import pandas as pd
import numpy as np

from src.utils.helpers import atr, find_swing_points

logger = logging.getLogger(__name__)


class RegimeType(Enum):
    STRONG_TREND_BULLISH = "STRONG_TREND_BULLISH"
    STRONG_TREND_BEARISH = "STRONG_TREND_BEARISH"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    TRANSITION = "TRANSITION"


REGIME_PATTERN_MULTIPLIERS = {
    "FVG": {
        RegimeType.STRONG_TREND_BULLISH: 1.8,
        RegimeType.STRONG_TREND_BEARISH: 1.8,
        RegimeType.RANGING: 0.5,
        RegimeType.HIGH_VOLATILITY: 1.3,
        RegimeType.LOW_VOLATILITY: 0.7,
        RegimeType.TRANSITION: 0.9,
    },
    "OB": {
        RegimeType.STRONG_TREND_BULLISH: 1.2,
        RegimeType.STRONG_TREND_BEARISH: 1.2,
        RegimeType.RANGING: 1.5,
        RegimeType.HIGH_VOLATILITY: 1.1,
        RegimeType.LOW_VOLATILITY: 1.3,
        RegimeType.TRANSITION: 1.4,
    },
    "BREAKER": {
        RegimeType.STRONG_TREND_BULLISH: 1.3,
        RegimeType.STRONG_TREND_BEARISH: 1.3,
        RegimeType.RANGING: 0.8,
        RegimeType.HIGH_VOLATILITY: 0.6,
        RegimeType.LOW_VOLATILITY: 1.1,
        RegimeType.TRANSITION: 1.5,
    },
    "SWEEP": {
        RegimeType.STRONG_TREND_BULLISH: 1.5,
        RegimeType.STRONG_TREND_BEARISH: 1.5,
        RegimeType.RANGING: 1.8,
        RegimeType.HIGH_VOLATILITY: 1.2,
        RegimeType.LOW_VOLATILITY: 1.4,
        RegimeType.TRANSITION: 1.6,
    },
    "WYCKOFF": {
        RegimeType.STRONG_TREND_BULLISH: 0.6,
        RegimeType.STRONG_TREND_BEARISH: 0.6,
        RegimeType.RANGING: 1.8,
        RegimeType.HIGH_VOLATILITY: 0.5,
        RegimeType.LOW_VOLATILITY: 1.6,
        RegimeType.TRANSITION: 1.5,
    },
    "VOID_SCALP": {
        RegimeType.STRONG_TREND_BULLISH: 1.6,
        RegimeType.STRONG_TREND_BEARISH: 1.6,
        RegimeType.RANGING: 0.4,
        RegimeType.HIGH_VOLATILITY: 1.5,
        RegimeType.LOW_VOLATILITY: 0.6,
        RegimeType.TRANSITION: 0.8,
    },
    "BOS_ZONE": {
        RegimeType.STRONG_TREND_BULLISH: 1.7,
        RegimeType.STRONG_TREND_BEARISH: 1.7,
        RegimeType.RANGING: 0.6,
        RegimeType.HIGH_VOLATILITY: 1.4,
        RegimeType.LOW_VOLATILITY: 0.8,
        RegimeType.TRANSITION: 0.7,
    },
    "CYCLE": {
        RegimeType.STRONG_TREND_BULLISH: 1.4,
        RegimeType.STRONG_TREND_BEARISH: 1.4,
        RegimeType.RANGING: 0.7,
        RegimeType.HIGH_VOLATILITY: 1.2,
        RegimeType.LOW_VOLATILITY: 0.9,
        RegimeType.TRANSITION: 1.3,
    },
    "SEQUENCE": {
        RegimeType.STRONG_TREND_BULLISH: 1.3,
        RegimeType.STRONG_TREND_BEARISH: 1.3,
        RegimeType.RANGING: 0.5,
        RegimeType.HIGH_VOLATILITY: 1.1,
        RegimeType.LOW_VOLATILITY: 0.6,
        RegimeType.TRANSITION: 0.8,
    },
    "INTERVAL_POINT": {
        RegimeType.STRONG_TREND_BULLISH: 1.1,
        RegimeType.STRONG_TREND_BEARISH: 1.1,
        RegimeType.RANGING: 1.6,
        RegimeType.HIGH_VOLATILITY: 0.8,
        RegimeType.LOW_VOLATILITY: 1.5,
        RegimeType.TRANSITION: 1.4,
    },
    "PRICE_INTERACTION": {
        RegimeType.STRONG_TREND_BULLISH: 1.2,
        RegimeType.STRONG_TREND_BEARISH: 1.2,
        RegimeType.RANGING: 1.4,
        RegimeType.HIGH_VOLATILITY: 0.9,
        RegimeType.LOW_VOLATILITY: 1.3,
        RegimeType.TRANSITION: 1.3,
    },
    "HARMONIC_CYCLE": {
        RegimeType.STRONG_TREND_BULLISH: 1.5,
        RegimeType.STRONG_TREND_BEARISH: 1.5,
        RegimeType.RANGING: 0.7,
        RegimeType.HIGH_VOLATILITY: 1.2,
        RegimeType.LOW_VOLATILITY: 0.8,
        RegimeType.TRANSITION: 0.9,
    },
    "PRESSURE_ZONE": {
        RegimeType.STRONG_TREND_BULLISH: 1.3,
        RegimeType.STRONG_TREND_BEARISH: 1.3,
        RegimeType.RANGING: 1.7,
        RegimeType.HIGH_VOLATILITY: 0.7,
        RegimeType.LOW_VOLATILITY: 1.5,
        RegimeType.TRANSITION: 1.6,
    },
    "TRB": {
        RegimeType.STRONG_TREND_BULLISH: 0.5,
        RegimeType.STRONG_TREND_BEARISH: 0.5,
        RegimeType.RANGING: 1.9,
        RegimeType.HIGH_VOLATILITY: 0.4,
        RegimeType.LOW_VOLATILITY: 1.7,
        RegimeType.TRANSITION: 1.2,
    },
}


@dataclass
class RegimeContext:
    regime: RegimeType
    confidence: float
    strength: float
    atr_ratio: float
    adx_value: float
    is_compressed: bool
    is_expanding: bool
    trend_alignment: str
    pattern_multipliers: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def get_multiplier(self, pattern_type: str) -> float:
        return self.pattern_multipliers.get(pattern_type, 1.0)


class RegimeDetector:
    def __init__(self, atr_period: int = 14, adx_period: int = 14):
        self.atr_period = atr_period
        self.adx_period = adx_period
        self._prev_regime: Optional[RegimeType] = None
        self._regime_stability: int = 0

    def detect(self, htf_df: pd.DataFrame, ltf_df: pd.DataFrame) -> RegimeContext:
        if htf_df is None or len(htf_df) < 30 or ltf_df is None or len(ltf_df) < 20:
            return self._fallback_regime()

        atr_series = atr(ltf_df, self.atr_period)
        recent_atr = atr_series.iloc[-1]
        avg_atr = atr_series.iloc[-50:].mean() if len(atr_series) >= 50 else atr_series.mean()
        atr_ratio = recent_atr / avg_atr if avg_atr > 0 else 1.0

        adx = self._compute_adx(ltf_df)
        trend_strength, ema_slope = self._measure_trend_strength(htf_df, ltf_df)
        is_compressed = atr_ratio < 0.7
        is_expanding = atr_ratio > 1.5

        regime, confidence = self._classify_regime(
            atr_ratio, adx, trend_strength, is_compressed, is_expanding,
            ema_slope=ema_slope,
        )

        self._regime_stability = self._regime_stability + 1 if regime == self._prev_regime else 0
        self._prev_regime = regime

        if self._regime_stability < 3:
            confidence = max(0.3, confidence * 0.8)

        multipliers = {}
        for pattern_name, regime_map in REGIME_PATTERN_MULTIPLIERS.items():
            base_mult = regime_map.get(regime, 1.0)
            adjusted = base_mult * (0.8 + 0.4 * confidence)
            multipliers[pattern_name] = round(adjusted, 2)

        trend_alignment = self._detect_trend_alignment(htf_df, ltf_df)

        notes = self._build_notes(regime, atr_ratio, adx, is_compressed, is_expanding, confidence)

        return RegimeContext(
            regime=regime,
            confidence=confidence,
            strength=trend_strength,
            atr_ratio=atr_ratio,
            adx_value=adx,
            is_compressed=is_compressed,
            is_expanding=is_expanding,
            trend_alignment=trend_alignment,
            pattern_multipliers=multipliers,
            notes=notes,
        )

    def _compute_adx(self, df: pd.DataFrame) -> float:
        if df is None or len(df) < self.adx_period * 2:
            return 0.0
        high, low, close = df["high"].values, df["low"].values, df["close"].values
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]),
                                   np.abs(low[1:] - close[:-1])))
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        def ema(x, period):
            alpha = 2.0 / (period + 1)
            result = np.zeros_like(x)
            result[0] = x[0]
            for i in range(1, len(x)):
                result[i] = x[i] * alpha + result[i - 1] * (1 - alpha)
            return result

        tr_ema = ema(tr, self.adx_period)
        plus_ema = ema(plus_dm, self.adx_period)
        minus_ema = ema(minus_dm, self.adx_period)

        plus_di = 100 * plus_ema / np.where(tr_ema > 0, tr_ema, 1)
        minus_di = 100 * minus_ema / np.where(tr_ema > 0, tr_ema, 1)
        dx = 100 * np.abs(plus_di - minus_di) / np.where(plus_di + minus_di > 0, plus_di + minus_di, 1)

        if len(dx) >= self.adx_period:
            return float(np.mean(dx[-self.adx_period:]))
        return float(np.mean(dx)) if len(dx) > 0 else 0.0

    def _measure_trend_strength(self, htf_df: pd.DataFrame, ltf_df: pd.DataFrame) -> Tuple[float, float]:
        ema8 = ltf_df["close"].ewm(span=8).mean()
        ema21 = ltf_df["close"].ewm(span=21).mean()
        ema_slope = (ema8.iloc[-1] - ema8.iloc[-5]) / ema8.iloc[-5] if len(ema8) >= 5 else 0
        ema_dist = abs(ema8.iloc[-1] - ema21.iloc[-1]) / ltf_df["close"].iloc[-1] if len(ema21) > 0 else 0

        highs_idx, lows_idx = find_swing_points(htf_df, lookback=5)
        swing_strength = 0.0
        if len(highs_idx) >= 3 and len(lows_idx) >= 3:
            hh = htf_df["high"].iloc[highs_idx[-1]] > htf_df["high"].iloc[highs_idx[-2]] > htf_df["high"].iloc[highs_idx[-3]]
            ll = htf_df["low"].iloc[lows_idx[-1]] < htf_df["low"].iloc[lows_idx[-2]] < htf_df["low"].iloc[lows_idx[-3]]
            hl = htf_df["low"].iloc[lows_idx[-1]] > htf_df["low"].iloc[lows_idx[-2]] > htf_df["low"].iloc[lows_idx[-3]]
            lh = htf_df["high"].iloc[highs_idx[-1]] < htf_df["high"].iloc[highs_idx[-2]] < htf_df["high"].iloc[highs_idx[-3]]
            if hh and hl:
                swing_strength = 1.0
            elif ll and lh:
                swing_strength = 1.0
            elif hh or ll:
                swing_strength = 0.5

        strength = min(1.0, abs(ema_slope) * 500 + ema_dist * 200 + swing_strength * 0.5)
        return strength, ema_slope

    def _classify_regime(self, atr_ratio: float, adx: float, trend_strength: float,
                          is_compressed: bool, is_expanding: bool,
                          ema_slope: float = 0.0) -> Tuple[RegimeType, float]:
        if is_expanding and adx < 25:
            return RegimeType.HIGH_VOLATILITY, min(1.0, atr_ratio / 2.0)

        if adx >= 30 and trend_strength >= 0.6:
            is_bullish = ema_slope > 0
            if is_bullish:
                return RegimeType.STRONG_TREND_BULLISH, min(1.0, adx / 60.0)
            else:
                return RegimeType.STRONG_TREND_BEARISH, min(1.0, adx / 60.0)

        if is_compressed:
            return RegimeType.LOW_VOLATILITY, min(1.0, (1.0 - atr_ratio) / 0.5)

        if 20 <= adx < 30 and 0.7 <= atr_ratio <= 1.3:
            return RegimeType.RANGING, min(1.0, (30 - adx) / 15.0)

        return RegimeType.TRANSITION, 0.5

    def _detect_trend_alignment(self, htf_df: pd.DataFrame, ltf_df: pd.DataFrame) -> str:
        htf_ema8 = htf_df["close"].ewm(span=8).mean()
        htf_ema21 = htf_df["close"].ewm(span=21).mean()
        ltf_ema8 = ltf_df["close"].ewm(span=8).mean()
        ltf_ema21 = ltf_df["close"].ewm(span=21).mean()

        htf_bull = htf_ema8.iloc[-1] > htf_ema21.iloc[-1] if len(htf_ema21) > 0 else False
        htf_bear = htf_ema8.iloc[-1] < htf_ema21.iloc[-1] if len(htf_ema21) > 0 else False
        ltf_bull = ltf_ema8.iloc[-1] > ltf_ema21.iloc[-1] if len(ltf_ema21) > 0 else False
        ltf_bear = ltf_ema8.iloc[-1] < ltf_ema21.iloc[-1] if len(ltf_ema21) > 0 else False

        if htf_bull and ltf_bull:
            return "BULLISH_ALIGNED"
        if htf_bear and ltf_bear:
            return "BEARISH_ALIGNED"
        if htf_bull and ltf_bear:
            return "HTF_BULLISH_LTF_BEARISH"
        if htf_bear and ltf_bull:
            return "HTF_BEARISH_LTF_BULLISH"
        return "NEUTRAL"

    def _build_notes(self, regime: RegimeType, atr_ratio: float, adx: float,
                      is_compressed: bool, is_expanding: bool, confidence: float) -> List[str]:
        notes = [f"Régimen: {regime.value} (confianza={confidence:.0%}, ADX={adx:.0f})"]
        if is_compressed:
            notes.append("Mercado comprimido — anticipar expansión")
        if is_expanding:
            notes.append("Volatilidad elevada — reducir tamaño")
        if adx >= 30:
            notes.append(f"Tendencia fuerte (ADX={adx:.0f}) — priorizar FVG/BOS")
        if adx < 20:
            notes.append("Sin tendencia clara — priorizar OB/Sweeps/Rango")
        return notes

    def _fallback_regime(self) -> RegimeContext:
        return RegimeContext(
            regime=RegimeType.RANGING, confidence=0.3, strength=0.0,
            atr_ratio=1.0, adx_value=0.0, is_compressed=False, is_expanding=False,
            trend_alignment="NEUTRAL",
            pattern_multipliers={k: 1.0 for k in REGIME_PATTERN_MULTIPLIERS},
            notes=["Sin datos suficientes — régimen por defecto"],
        )

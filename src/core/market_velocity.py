"""Market Velocity & Accumulation Detector
Mide velocidad (momentum), aceleración y compresión ATR
para clasificar el régimen del mercado: ACCUMULATION / EXPANSION / NEUTRAL.
"""
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class VelocityResult:
    momentum: float = 0.0
    velocity_raw: float = 0.0
    acceleration: float = 0.0
    atr_ratio: float = 1.0
    is_compressed: bool = False
    is_expanding: bool = False
    regime: str = "NEUTRAL"
    accumulation_score: float = 0.0


class MarketVelocityDetector:
    def __init__(self, atr_period: int = 5,
                 atr_smoothing: int = 40,
                 compression_threshold: float = 0.85,
                 expansion_threshold: float = 1.15,
                 velocity_period: int = 5):
        self.atr_period = atr_period
        self.atr_smoothing = atr_smoothing
        self.compression_threshold = compression_threshold
        self.expansion_threshold = expansion_threshold
        self.velocity_period = velocity_period

    def detect(self, df: Optional[pd.DataFrame]) -> VelocityResult:
        if df is None or len(df) < max(self.atr_period + 5, self.velocity_period + 5):
            return VelocityResult()

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        # ── ATR ratio (short / longer) ─────────────────────────────────
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        )
        period_s = min(self.atr_period, len(tr))
        period_l = min(self.atr_smoothing, len(tr))
        atr_short = float(np.mean(tr[-period_s:]))
        atr_long = float(np.mean(tr[-period_l:]))
        atr_ratio = atr_short / atr_long if atr_long > 0 else 1.0

        # ── Price velocity (linear regression slope, normalized) ──────
        n = min(self.velocity_period, len(close))
        prices = close[-n:]
        x = np.arange(n)
        slope = float(np.polyfit(x, prices, 1)[0]) if n > 1 else 0.0
        velocity_raw = slope
        momentum = slope / (float(close[-1]) + 1e-10) * 100.0

        # ── Acceleration ──────────────────────────────────────────────
        half = n // 2
        if len(close) > n + half and half >= 2:
            prices_prev = close[-(n + half):-half]
            x_prev = np.arange(len(prices_prev))
            prev_slope = float(np.polyfit(x_prev, prices_prev, 1)[0]) if len(prices_prev) > 1 else 0.0
            acceleration = slope - prev_slope
        else:
            acceleration = 0.0

        # ── Accumulation score ────────────────────────────────────────
        mom_score = max(0.0, 1.0 - abs(momentum) / 0.2)
        comp_score = max(0.0, 1.0 - atr_ratio) if atr_ratio < 1.0 else 0.0
        recent_high = float(np.max(high[-n:]))
        recent_low = float(np.min(low[-n:]))
        recent_range = recent_high - recent_low
        range_score = max(0.0, 1.0 - recent_range / (atr_long * 2.0)) if atr_long > 0 else 0.0
        accumulation_score = mom_score * 0.4 + comp_score * 0.3 + range_score * 0.3

        # ── Regime classification ─────────────────────────────────────
        is_compressed = atr_ratio < self.compression_threshold
        is_expanding = atr_ratio > self.expansion_threshold

        if accumulation_score > 0.50 or (is_compressed and mom_score > 0.7):
            regime = "ACCUMULATION"
        elif is_expanding or abs(momentum) > 0.2:
            regime = "EXPANSION"
        else:
            regime = "NEUTRAL"

        return VelocityResult(
            momentum=round(momentum, 4),
            velocity_raw=round(velocity_raw, 6),
            acceleration=round(acceleration, 6),
            atr_ratio=round(atr_ratio, 3),
            is_compressed=is_compressed,
            is_expanding=is_expanding,
            regime=regime,
            accumulation_score=round(accumulation_score, 3),
        )

"""Volume Spread Analysis (VSA) Detector
Detects climax, absorption, low-volume pullback,
volume confirmation, volume divergence, and no demand/supply.
"""
import logging
from typing import Optional

import pandas as pd

from src.utils.helpers import find_swing_points

logger = logging.getLogger(__name__)


class VSADetector:
    """VSA pattern detection using tick_volume from MT5 DataFrames.
    NOTE: MT5 tick_volume is an ESTIMATE (tick count), not real volume (contracts).
    Forex has no centralized exchange, so real volume is unavailable without special feeds.
    All signals include a volume_quality factor (0-1) indicating data reliability.
    """

    @staticmethod
    def _volume_quality(df: pd.DataFrame, lookback: int = 20) -> float:
        """Estimates how reliable tick_volume is as a proxy for real activity.
        Checks correlation between volume and candle range over recent bars.
        If tick volume is high on narrow candles, it's mostly noise → low quality.
        Returns 0.0 (unreliable) to 1.0 (good proxy).
        """
        if df is None or len(df) < lookback:
            return 0.3
        recent = df.iloc[-lookback:]
        ranges = recent["high"] - recent["low"]
        vol = recent["volume"]
        if ranges.std() == 0 or vol.std() == 0:
            return 0.3
        corr = ranges.corr(vol)
        last_rng = ranges.iloc[-1]
        last_vol = vol.iloc[-1]
        mean_rng = ranges.mean()
        mean_vol = vol.mean()
        extreme_ratio = (last_vol / mean_vol) / max(1e-6, last_rng / max(1e-6, mean_rng))
        quality = 0.3 + 0.4 * max(0, corr) + 0.3 * min(1.0, 1.0 / max(1.0, abs(extreme_ratio - 1.0)))
        return round(min(1.0, max(0.0, quality)), 2)

    @staticmethod
    def _sma_volume(df: pd.DataFrame, period: int = 20) -> float:
        if "volume" in df.columns and len(df) >= period:
            return float(df["volume"].iloc[-period:].mean())
        return 0.0

    @staticmethod
    def _avg_range(df: pd.DataFrame, period: int = 14) -> float:
        if len(df) >= period:
            return float((df["high"].iloc[-period:] - df["low"].iloc[-period:]).mean())
        return 0.0

    @classmethod
    def check_volume_confirmation(cls, df: pd.DataFrame, direction: str) -> bool:
        """Breakout candle with volume >= 1.5x SMA(20) and dominant body."""
        if df is None or len(df) < 21:
            return False
        avg_vol = cls._sma_volume(df)
        if avg_vol <= 0:
            return False
        last = df.iloc[-1]
        if last["volume"] < avg_vol * 1.5:
            return False
        spread = last["high"] - last["low"]
        if spread == 0:
            return False
        body = abs(last["close"] - last["open"])
        body_ratio = body / spread
        if body_ratio < 0.4:
            return False
        if direction == "BUY":
            return last["close"] > last["open"]
        return last["close"] < last["open"]

    @classmethod
    def check_climax(cls, df: pd.DataFrame, direction: str) -> bool:
        """Very high volume (>= 2.5x SMA) + wide spread (>= 1.5x avg range) =
        exhaustion climax. Penalize entering in the same direction."""
        if df is None or len(df) < 21:
            return False
        avg_vol = cls._sma_volume(df)
        avg_rng = cls._avg_range(df)
        if avg_vol <= 0 or avg_rng <= 0:
            return False
        last = df.iloc[-1]
        vol_ratio = last["volume"] / avg_vol
        spread_ratio = (last["high"] - last["low"]) / avg_rng
        if vol_ratio < 2.5 or spread_ratio < 1.5:
            return False
        if direction == "BUY":
            return last["close"] > last["open"] and last["close"] >= df["high"].iloc[-5:].max() * 0.995
        return last["close"] < last["open"] and last["close"] <= df["low"].iloc[-5:].min() * 1.005

    @classmethod
    def check_absorption(cls, df: pd.DataFrame, direction: str) -> bool:
        """High volume (>= 1.8x SMA) but narrow spread (<= 0.7x avg range)
        = smart money absorbing orders."""
        if df is None or len(df) < 21:
            return False
        avg_vol = cls._sma_volume(df)
        avg_rng = cls._avg_range(df)
        if avg_vol <= 0 or avg_rng <= 0:
            return False
        last = df.iloc[-1]
        if last["volume"] < avg_vol * 1.8:
            return False
        spread = last["high"] - last["low"]
        if spread > avg_rng * 0.7:
            return False
        body = abs(last["close"] - last["open"])
        body_ratio = body / spread if spread > 0 else 0
        if direction == "BUY":
            return last["close"] > df["low"].iloc[-5:].min() and body_ratio < 0.6
        return last["close"] < df["high"].iloc[-5:].max() and body_ratio < 0.6

    @classmethod
    def check_low_volume_pullback(cls, df: pd.DataFrame, direction: str) -> bool:
        """Low volume (< 0.7x SMA) retracement candles in the opposite
        direction of the trade = healthy pullback."""
        if df is None or len(df) < 21:
            return False
        avg_vol = cls._sma_volume(df)
        if avg_vol <= 0:
            return False
        lookback = min(5, len(df) - 1)
        retrace = df.iloc[-lookback:]
        if direction == "BUY":
            bearish = retrace[retrace["close"] < retrace["open"]]
            if bearish.empty:
                return False
            max_vol = bearish["volume"].max()
        else:
            bullish = retrace[retrace["close"] > retrace["open"]]
            if bullish.empty:
                return False
            max_vol = bullish["volume"].max()
        return max_vol < avg_vol * 0.7

    @classmethod
    def check_volume_divergence(cls, df: pd.DataFrame, direction: str) -> bool:
        """Price makes higher high / lower low but volume declines >= 20%."""
        if df is None or len(df) < 30:
            return False
        highs, lows = find_swing_points(df, lookback=5)
        if len(highs) < 2 or len(lows) < 2:
            return False
        if direction == "BUY":
            if len(highs) >= 2:
                last_hh = df.iloc[highs[-1]]
                prev_hh = df.iloc[highs[-2]]
                return last_hh["high"] > prev_hh["high"] and last_hh["volume"] < prev_hh["volume"] * 0.8
            return False
        if len(lows) >= 2:
            last_ll = df.iloc[lows[-1]]
            prev_ll = df.iloc[lows[-2]]
            return last_ll["low"] < prev_ll["low"] and last_ll["volume"] < prev_ll["volume"] * 0.8
        return False

    @classmethod
    def check_no_demand_supply(cls, df: pd.DataFrame, direction: str) -> bool:
        """Very low volume near price extreme = lack of follow-through."""
        if df is None or len(df) < 21:
            return False
        avg_vol = cls._sma_volume(df)
        if avg_vol <= 0:
            return False
        lookback = min(5, len(df) - 1)
        max_recent_vol = df["volume"].iloc[-lookback:].max()
        if max_recent_vol > avg_vol * 0.5:
            return False
        if direction == "BUY":
            return df["close"].iloc[-1] >= df["high"].iloc[-10:].max() * 0.995
        return df["close"].iloc[-1] <= df["low"].iloc[-10:].min() * 1.005

    @classmethod
    def analyze(cls, df: pd.DataFrame, direction: str) -> dict:
        """Run all VSA checks for a given direction. Returns dict of signal->bool
        plus volume_quality (0-1) indicating how reliable tick_volume is.
        """
        vol_quality = cls._volume_quality(df)
        return {
            "volume_confirmation": cls.check_volume_confirmation(df, direction),
            "climax": cls.check_climax(df, direction),
            "absorption": cls.check_absorption(df, direction),
            "low_volume_pullback": cls.check_low_volume_pullback(df, direction),
            "volume_divergence": cls.check_volume_divergence(df, direction),
            "no_demand_supply": cls.check_no_demand_supply(df, direction),
            "volume_quality": vol_quality,
        }

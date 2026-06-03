"""Market Context Analyzer
Analyzes market regime, structure, and fractal patterns
based on Smart Money Concepts (Classes 1-3).
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum

import pandas as pd
import numpy as np

from src.utils.helpers import atr, find_swing_points, candle_is_body_dominant

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    RANGING = "RANGING"


class MarketRegime(Enum):
    ACCUMULATION = "ACCUMULATION"
    EXPANSION = "EXPANSION"
    RETRACEMENT = "RETRACEMENT"
    UNDEFINED = "UNDEFINED"


@dataclass
class SwingPoint:
    index: int
    price: float
    time: pd.Timestamp
    kind: str


@dataclass
class MarketStructure:
    trend: TrendDirection
    last_swing_high: Optional[SwingPoint]
    last_swing_low: Optional[SwingPoint]
    valid_range_high: float
    valid_range_low: float
    in_discount_zone: bool
    in_premium_zone: bool
    equilibrium_level: float
    last_bos_index: Optional[int]
    last_bos_direction: Optional[str]


@dataclass
class FractalABC:
    a: Optional[SwingPoint] = None
    b: Optional[SwingPoint] = None
    c: Optional[SwingPoint] = None
    valid: bool = False
    direction: Optional[str] = None


@dataclass
class MarketContext:
    regime: MarketRegime
    htf_structure: MarketStructure
    ltf_structure: MarketStructure
    fractal_htf: FractalABC
    atr_value: float
    is_compressed: bool
    is_expanding: bool
    notes: List[str] = field(default_factory=list)


class MarketAnalyzer:
    def __init__(self, params):
        self.params = params

    def detect_regime(self, df: pd.DataFrame) -> Tuple[MarketRegime, bool, bool]:
        if df is None or len(df) < 30:
            return MarketRegime.UNDEFINED, False, False

        atr_series = atr(df, 14)
        recent_atr = atr_series.iloc[-1]
        avg_atr = atr_series.iloc[-50:].mean() if len(atr_series) >= 50 else atr_series.mean()

        is_compressed = recent_atr < (avg_atr * self.params.consolidation_max_atr_ratio)
        is_expanding = recent_atr > (avg_atr * self.params.expansion_tick_acceleration)

        last_close = df["close"].iloc[-2]
        last_open = df["open"].iloc[-1]
        gap_size = abs(last_open - last_close)
        has_gap = gap_size > (avg_atr * 0.5)

        if is_expanding or has_gap:
            return MarketRegime.EXPANSION, is_compressed, True
        if is_compressed:
            return MarketRegime.ACCUMULATION, True, False

        recent_change = df["close"].iloc[-1] - df["close"].iloc[-10]
        prior_change = df["close"].iloc[-10] - df["close"].iloc[-20]
        if recent_change * prior_change < 0 and abs(recent_change) < abs(prior_change):
            return MarketRegime.RETRACEMENT, False, False

        return MarketRegime.UNDEFINED, is_compressed, is_expanding

    def analyze_structure(self, df: pd.DataFrame, lookback: int = 5) -> MarketStructure:
        if df is None or len(df) < lookback * 4:
            return self._empty_structure()

        highs_idx, lows_idx = find_swing_points(df, lookback=lookback)

        last_high = None
        if highs_idx:
            i = highs_idx[-1]
            last_high = SwingPoint(index=i, price=df["high"].iloc[i], time=df["time"].iloc[i], kind="HIGH")
        last_low = None
        if lows_idx:
            i = lows_idx[-1]
            last_low = SwingPoint(index=i, price=df["low"].iloc[i], time=df["time"].iloc[i], kind="LOW")

        valid_high, valid_low = self._validate_range(df, highs_idx, lows_idx)

        if valid_high is None or valid_low is None:
            return self._empty_structure()

        rng = valid_high - valid_low
        if rng <= 0:
            return self._empty_structure()
        eq = valid_low + rng * 0.5
        current_price = df["close"].iloc[-1]

        in_discount = current_price <= eq
        in_premium = current_price >= eq

        trend, last_bos_idx, last_bos_dir = self._detect_trend_with_bos(df, highs_idx, lows_idx)

        return MarketStructure(
            trend=trend,
            last_swing_high=last_high,
            last_swing_low=last_low,
            valid_range_high=valid_high,
            valid_range_low=valid_low,
            in_discount_zone=in_discount,
            in_premium_zone=in_premium,
            equilibrium_level=eq,
            last_bos_index=last_bos_idx,
            last_bos_direction=last_bos_dir,
        )

    def _empty_structure(self) -> MarketStructure:
        return MarketStructure(
            trend=TrendDirection.RANGING,
            last_swing_high=None, last_swing_low=None,
            valid_range_high=0.0, valid_range_low=0.0,
            in_discount_zone=False, in_premium_zone=False,
            equilibrium_level=0.0,
            last_bos_index=None, last_bos_direction=None,
        )

    def _validate_range(self, df: pd.DataFrame, highs_idx: List[int], lows_idx: List[int]) -> Tuple[Optional[float], Optional[float]]:
        if not highs_idx or not lows_idx:
            return None, None

        last_high_idx = highs_idx[-1]
        last_low_idx = lows_idx[-1]

        if last_high_idx > last_low_idx:
            high_price = df["high"].iloc[last_high_idx]
            low_price = df["low"].iloc[last_low_idx]
            origin_low_idx = next((i for i in reversed(lows_idx) if i < last_high_idx), None)
            if origin_low_idx is None:
                return None, None
            origin_low = df["low"].iloc[origin_low_idx]
            impulse_range = high_price - origin_low
            retracement = high_price - low_price
            if impulse_range > 0 and retracement / impulse_range >= self.params.min_retracement_level:
                return high_price, origin_low
            return None, None
        else:
            low_price = df["low"].iloc[last_low_idx]
            high_price = df["high"].iloc[last_high_idx]
            origin_high_idx = next((i for i in reversed(highs_idx) if i < last_low_idx), None)
            if origin_high_idx is None:
                return None, None
            origin_high = df["high"].iloc[origin_high_idx]
            impulse_range = origin_high - low_price
            retracement = high_price - low_price
            if impulse_range > 0 and retracement / impulse_range >= self.params.min_retracement_level:
                return origin_high, low_price
            return None, None

    def _detect_trend_with_bos(self, df: pd.DataFrame, highs_idx: List[int], lows_idx: List[int]) -> Tuple[TrendDirection, Optional[int], Optional[str]]:
        if len(highs_idx) < 2 or len(lows_idx) < 2:
            return TrendDirection.RANGING, None, None

        last_bos_idx = None
        last_bos_dir = None
        for i in range(len(df) - 1, max(highs_idx[0], lows_idx[0]), -1):
            close_i = df["close"].iloc[i]
            prev_highs = [h for h in highs_idx if h < i]
            prev_lows = [l for l in lows_idx if l < i]
            if prev_highs:
                last_h_idx = prev_highs[-1]
                last_h = df["high"].iloc[last_h_idx]
                if close_i > last_h and self._validate_bos_retracement(df, last_h_idx, i, highs_idx, lows_idx, "BULLISH"):
                    last_bos_idx = i
                    last_bos_dir = "BULLISH"
                    break
            if prev_lows and last_bos_dir is None:
                last_l_idx = prev_lows[-1]
                last_l = df["low"].iloc[last_l_idx]
                if close_i < last_l and self._validate_bos_retracement(df, last_l_idx, i, highs_idx, lows_idx, "BEARISH"):
                    last_bos_idx = i
                    last_bos_dir = "BEARISH"
                    break

        if last_bos_dir == "BULLISH":
            return TrendDirection.BULLISH, last_bos_idx, "BULLISH"
        if last_bos_dir == "BEARISH":
            return TrendDirection.BEARISH, last_bos_idx, "BEARISH"
        return TrendDirection.RANGING, None, None

    def _validate_bos_retracement(self, df: pd.DataFrame, swing_idx: int, current_idx: int,
                                    highs_idx: List[int], lows_idx: List[int], direction: str) -> bool:
        """Valida que el BOS tenga respaldo de retroceso ≥50%.
        Excepción: 3er impulso (manual §19.5).
        Además valida cuerpo dominante y volumen (BOS_Zone_Retest.pine)."""
        if not self._validate_bos_volume_body(df, current_idx, direction):
            return False
        if direction == "BULLISH":
            origin_lows = [l for l in lows_idx if l < swing_idx]
            if not origin_lows:
                return False
            origin_low = df["low"].iloc[origin_lows[-1]]
            retrace_lows = [l for l in lows_idx if swing_idx < l < current_idx]
            if not retrace_lows:
                return self._es_tercer_impulso(df, swing_idx, current_idx, lows_idx)
            impulse = df["high"].iloc[swing_idx] - origin_low
            retrace = df["high"].iloc[swing_idx] - df["low"].iloc[retrace_lows[-1]]
            return impulse > 0 and retrace / impulse >= 0.50
        else:
            origin_highs = [h for h in highs_idx if h < swing_idx]
            if not origin_highs:
                return False
            origin_high = df["high"].iloc[origin_highs[-1]]
            retrace_highs = [h for h in highs_idx if swing_idx < h < current_idx]
            if not retrace_highs:
                return self._es_tercer_impulso(df, swing_idx, current_idx, highs_idx)
            impulse = origin_high - df["low"].iloc[swing_idx]
            retrace = df["high"].iloc[retrace_highs[-1]] - df["low"].iloc[swing_idx]
            return impulse > 0 and retrace / impulse >= 0.50

    def _validate_bos_volume_body(self, df: pd.DataFrame, idx: int, direction: str) -> bool:
        """Valida que la vela de ruptura tenga cuerpo dominante y volumen adecuado.
        Basado en BOS_Zone_Retest.pine: body ≥55% rango, volumen entre 1.2x y 4.0x SMA(20)."""
        if idx < 0 or idx >= len(df):
            return True
        candle = df.iloc[idx]
        o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
        rng = h - l
        if rng <= 0:
            return False
        body_ratio = abs(c - o) / rng
        if body_ratio < 0.55:
            return False
        if direction == "BULLISH" and c <= o:
            return False
        if direction == "BEARISH" and c >= o:
            return False
        if "volume" in df.columns and df["volume"].iloc[-20:].sum() > 0:
            vol_sma = df["volume"].iloc[-20:].mean()
            vol = candle.get("volume", 0)
            if vol > 0 and (vol < vol_sma * 1.2 or vol > vol_sma * 4.0):
                return False
        return True

    def _es_tercer_impulso(self, df: pd.DataFrame, swing_idx: int, current_idx: int, opposite_swings: List[int]) -> bool:
        """Excepción: el 3er movimiento no da retroceso clásico (manual §19.5)."""
        relevant = [s for s in opposite_swings if swing_idx < s < current_idx]
        if len(relevant) >= 2:
            return True
        if current_idx - swing_idx <= 8:
            from src.utils.helpers import atr
            atr_val = atr(df, 14).iloc[-1]
            price_move = abs(df["close"].iloc[current_idx] - df["close"].iloc[swing_idx])
            if price_move > atr_val * 1.2:
                return True
        return False

    def detect_trb_manipulation(self, df: pd.DataFrame, lookback_bars: int = 30) -> dict:
        """Detección de manipulación TRB (Trading Range Breakout).
        Retorna dict con: fake_breakout, range_high, range_low, displacement, displacement_size.
        Basado en MODELO_TRB_Analisis_Cuantitativo.docx (4 fases)."""
        result = {"fake_breakout": False, "range_high": None, "range_low": None,
                  "displacement": False, "displacement_size": 0.0, "direction": None}
        if df is None or len(df) < lookback_bars:
            return result
        from src.utils.helpers import atr
        atr_val = atr(df, 14).iloc[-1]
        n = len(df)
        recent = df.iloc[-lookback_bars:]
        range_high = recent["high"].max()
        range_low = recent["low"].min()
        range_size = range_high - range_low
        if range_size > atr_val * 4 or range_size < atr_val * 0.3:
            return result
        result["range_high"] = range_high
        result["range_low"] = range_low
        mid = (range_high + range_low) / 2
        lookback_vol = df["volume"].iloc[-lookback_bars:].mean() if "volume" in df.columns else 0
        for i in range(n - lookback_bars, n):
            close_i = df["close"].iloc[i]
            high_i = df["high"].iloc[i]
            low_i = df["low"].iloc[i]
            vol_i = df["volume"].iloc[i] if "volume" in df.columns else 0
            vol_ok = vol_i > lookback_vol * 1.5 if lookback_vol > 0 else True
            if high_i > range_high and close_i < range_high and vol_ok:
                atr_now = atr_val * 0.3
                displace = df["close"].iloc[i + 1:].min() if i + 1 < n else low_i
                displace_size = (high_i - displace) if displace < close_i else 0
                result["fake_breakout"] = True
                result["displacement"] = displace_size > atr_now * 1.5
                result["displacement_size"] = displace_size
                result["direction"] = "SELL"
                result["displace_low"] = displace
                break
            if low_i < range_low and close_i > range_low and vol_ok:
                atr_now = atr_val * 0.3
                displace = df["close"].iloc[i + 1:].max() if i + 1 < n else high_i
                displace_size = (displace - low_i) if displace > close_i else 0
                result["fake_breakout"] = True
                result["displacement"] = displace_size > atr_now * 1.5
                result["displacement_size"] = displace_size
                result["direction"] = "BUY"
                result["displace_high"] = displace
                break
        return result

    def count_wick_rejections(self, df: pd.DataFrame, level: float, direction: str,
                               lookback: int = 8, min_rejections: int = 2) -> dict:
        """Cuenta rechazos de mecha en un nivel (puntitos dorados).
        Basado en XAUUSD_M3_Analisis_Cuantitativo.docx.
        Retorna: count, avg_rejection_depth, valid."""
        n = min(lookback, len(df))
        count = 0
        total_depth = 0.0
        for i in range(-1, -n - 1, -1):
            candle = df.iloc[i]
            body_high = max(candle["open"], candle["close"])
            body_low = min(candle["open"], candle["close"])
            if direction == "BUY":
                if candle["low"] <= level <= body_low:
                    count += 1
                    total_depth += level - candle["low"]
            else:
                if candle["high"] >= level >= body_high:
                    count += 1
                    total_depth += candle["high"] - level
        avg_depth = total_depth / count if count > 0 else 0.0
        score_mult = 1.0
        if count >= 4:
            score_mult = 1.5
        elif count >= 3:
            score_mult = 1.2
        return {"count": count, "avg_depth": avg_depth,
                "valid": count >= min_rejections, "score_mult": score_mult,
                "intensity": "HIGH" if count >= 4 else "MEDIUM" if count >= 3 else "LOW"}

    def detect_fractal_abc(self, df: pd.DataFrame, lookback: int = 5) -> FractalABC:
        if df is None or len(df) < lookback * 4:
            return FractalABC()

        highs_idx, lows_idx = find_swing_points(df, lookback=lookback)
        if len(highs_idx) < 1 or len(lows_idx) < 1:
            return FractalABC()

        if highs_idx and lows_idx:
            last_high_idx = highs_idx[-1]
            origin_low_idx = next((i for i in reversed(lows_idx) if i < last_high_idx), None)
            current_low_idx = next((i for i in reversed(lows_idx) if i > last_high_idx), None)

            if origin_low_idx is not None and current_low_idx is not None:
                a = df["low"].iloc[origin_low_idx]
                b = df["high"].iloc[last_high_idx]
                c = df["low"].iloc[current_low_idx]
                impulse = b - a
                retr = b - c
                if impulse > 0 and retr / impulse >= self.params.min_retracement_level:
                    return FractalABC(
                        a=SwingPoint(origin_low_idx, a, df["time"].iloc[origin_low_idx], "LOW"),
                        b=SwingPoint(last_high_idx, b, df["time"].iloc[last_high_idx], "HIGH"),
                        c=SwingPoint(current_low_idx, c, df["time"].iloc[current_low_idx], "LOW"),
                        valid=True, direction="Bullish",
                    )

            last_low_idx = lows_idx[-1]
            origin_high_idx = next((i for i in reversed(highs_idx) if i < last_low_idx), None)
            current_high_idx = next((i for i in reversed(highs_idx) if i > last_low_idx), None)

            if origin_high_idx is not None and current_high_idx is not None:
                a = df["high"].iloc[origin_high_idx]
                b = df["low"].iloc[last_low_idx]
                c = df["high"].iloc[current_high_idx]
                impulse = a - b
                retr = c - b
                if impulse > 0 and retr / impulse >= self.params.min_retracement_level:
                    return FractalABC(
                        a=SwingPoint(origin_high_idx, a, df["time"].iloc[origin_high_idx], "HIGH"),
                        b=SwingPoint(last_low_idx, b, df["time"].iloc[last_low_idx], "LOW"),
                        c=SwingPoint(current_high_idx, c, df["time"].iloc[current_high_idx], "HIGH"),
                        valid=True, direction="Bearish",
                    )

        return FractalABC()

    def analyze_full_context(self, htf_df: pd.DataFrame, ltf_df: pd.DataFrame) -> MarketContext:
        regime, compressed, expanding = self.detect_regime(ltf_df)
        htf_struct = self.analyze_structure(htf_df, lookback=5)
        ltf_struct = self.analyze_structure(ltf_df, lookback=3)
        fractal = self.detect_fractal_abc(htf_df, lookback=5)
        atr_value = atr(ltf_df, 14).iloc[-1] if ltf_df is not None and len(ltf_df) > 14 else 0.0

        notes = []
        if compressed:
            notes.append("Mercado comprimido (acumulación)")
        if expanding:
            notes.append("Expansión institucional detectada")

        return MarketContext(
            regime=regime,
            htf_structure=htf_struct,
            ltf_structure=ltf_struct,
            fractal_htf=fractal,
            atr_value=atr_value,
            is_compressed=compressed,
            is_expanding=expanding,
            notes=notes,
        )

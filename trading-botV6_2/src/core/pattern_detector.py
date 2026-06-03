"""Pattern Detector for Smart Money Concepts
Detects FVG, Order Blocks, Breakers, Liquidity Sweeps,
Wyckoff Spring/UTAD, and Cycle patterns.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum

import pandas as pd
import numpy as np

from src.utils.helpers import (
    atr, candle_body_size, candle_is_body_dominant,
    find_swing_points, pip_size,
)

logger = logging.getLogger(__name__)


class PatternType(Enum):
    FVG_BULLISH = "FVG_BULLISH"
    FVG_BEARISH = "FVG_BEARISH"
    OB_BULLISH = "OB_BULLISH"
    OB_BEARISH = "OB_BEARISH"
    BREAKER_BULLISH = "BREAKER_BULLISH"
    BREAKER_BEARISH = "BREAKER_BEARISH"
    SWEEP_BULLISH = "SWEEP_BULLISH"
    SWEEP_BEARISH = "SWEEP_BEARISH"
    CYCLE_BULLISH = "CYCLE_BULLISH"
    CYCLE_BEARISH = "CYCLE_BEARISH"
    SPRING_BULLISH = "SPRING_BULLISH"
    UTAD_BEARISH = "UTAD_BEARISH"
    SEQUENCE_BULLISH = "SEQUENCE_BULLISH"
    SEQUENCE_BEARISH = "SEQUENCE_BEARISH"
    VOID_SCALP_BULLISH = "VOID_SCALP_BULLISH"
    VOID_SCALP_BEARISH = "VOID_SCALP_BEARISH"
    SOS_BULLISH = "SOS_BULLISH"
    SOW_BEARISH = "SOW_BEARISH"
    BOS_ZONE_RETEST_BULLISH = "BOS_ZONE_RETEST_BULLISH"
    BOS_ZONE_RETEST_BEARISH = "BOS_ZONE_RETEST_BEARISH"
    PRICE_ESTABLISHMENT_LONG = "PRICE_ESTABLISHMENT_LONG"
    PRICE_ESTABLISHMENT_SHORT = "PRICE_ESTABLISHMENT_SHORT"

    INTERVAL_POINT_BULLISH = "INTERVAL_POINT_BULLISH"
    INTERVAL_POINT_BEARISH = "INTERVAL_POINT_BEARISH"
    PRICE_INTERACTION_BULLISH = "PRICE_INTERACTION_BULLISH"
    PRICE_INTERACTION_BEARISH = "PRICE_INTERACTION_BEARISH"
    HARMONIC_CYCLE_BULLISH = "HARMONIC_CYCLE_BULLISH"
    HARMONIC_CYCLE_BEARISH = "HARMONIC_CYCLE_BEARISH"
    PRESSURE_ZONE_BULLISH = "PRESSURE_ZONE_BULLISH"
    PRESSURE_ZONE_BEARISH = "PRESSURE_ZONE_BEARISH"


@dataclass
class Pattern:
    type: PatternType
    direction: str
    high: float
    low: float
    mid: float
    index: int
    time: pd.Timestamp
    confidence: float = 1.0
    extra: dict = field(default_factory=dict)


class PatternDetector:
    def __init__(self, params, symbol: str):
        self.params = params
        self.symbol = symbol
        self.pip = pip_size(symbol)

    def detect_fvg(self, df: pd.DataFrame, last_n_bars: int = 100) -> List[Pattern]:
        patterns: List[Pattern] = []
        if df is None or len(df) < 3:
            return patterns

        start = max(2, len(df) - last_n_bars)
        min_size = self.params.fvg_min_size_atr_ratio * atr(df, 14).iloc[-1]
        min_size_pips = self.params.void_min_size_pips * self.pip
        threshold = max(min_size, min_size_pips * 0.5)

        for i in range(start, len(df)):
            high_i2 = df["high"].iloc[i - 2]
            low_i2 = df["low"].iloc[i - 2]
            high_i = df["high"].iloc[i]
            low_i = df["low"].iloc[i]

            if low_i > high_i2:
                gap = low_i - high_i2
                if gap >= threshold:
                    mid = high_i2 + gap * self.params.fvg_entry_level
                    patterns.append(Pattern(
                        type=PatternType.FVG_BULLISH, direction="BUY",
                        high=low_i, low=high_i2, mid=mid,
                        index=i, time=df["time"].iloc[i],
                        confidence=min(1.0, gap / (threshold * 2)),
                        extra={"size": gap},
                    ))

            if high_i < low_i2:
                gap = low_i2 - high_i
                if gap >= threshold:
                    mid = high_i + gap * self.params.fvg_entry_level
                    patterns.append(Pattern(
                        type=PatternType.FVG_BEARISH, direction="SELL",
                        high=low_i2, low=high_i, mid=mid,
                        index=i, time=df["time"].iloc[i],
                        confidence=min(1.0, gap / (threshold * 2)),
                        extra={"size": gap},
                    ))

        return self._filter_unmitigated_fvg(df, patterns)

    def _filter_unmitigated_fvg(self, df: pd.DataFrame, patterns: List[Pattern]) -> List[Pattern]:
        valid = []
        for p in patterns:
            after = df.iloc[p.index + 1:]
            if after.empty:
                valid.append(p)
                continue
            if p.direction == "BUY":
                if after["low"].min() > p.mid:
                    valid.append(p)
            else:
                if after["high"].max() < p.mid:
                    valid.append(p)
        return valid

    def detect_order_blocks(self, df: pd.DataFrame, last_n_bars: int = 80) -> List[Pattern]:
        patterns: List[Pattern] = []
        if df is None or len(df) < 6:
            return patterns

        atr_val = atr(df, 14).iloc[-1]
        start = max(3, len(df) - last_n_bars)

        for i in range(start, len(df) - 2):
            body_i1 = candle_body_size(df["open"].iloc[i + 1], df["close"].iloc[i + 1])

            if body_i1 < atr_val * 0.8:
                continue

            is_bearish_i = df["close"].iloc[i] < df["open"].iloc[i]
            is_bullish_i1 = df["close"].iloc[i + 1] > df["open"].iloc[i + 1]
            if is_bearish_i and is_bullish_i1:
                ob_high = df["high"].iloc[i]
                ob_low = df["low"].iloc[i]
                mid = ob_low + (ob_high - ob_low) * 0.5
                patterns.append(Pattern(
                    type=PatternType.OB_BULLISH, direction="BUY",
                    high=ob_high, low=ob_low, mid=mid,
                    index=i, time=df["time"].iloc[i],
                    confidence=min(1.0, body_i1 / (atr_val * 1.5)),
                ))

            is_bullish_i = df["close"].iloc[i] > df["open"].iloc[i]
            is_bearish_i1 = df["close"].iloc[i + 1] < df["open"].iloc[i + 1]
            if is_bullish_i and is_bearish_i1:
                ob_high = df["high"].iloc[i]
                ob_low = df["low"].iloc[i]
                mid = ob_low + (ob_high - ob_low) * 0.5
                patterns.append(Pattern(
                    type=PatternType.OB_BEARISH, direction="SELL",
                    high=ob_high, low=ob_low, mid=mid,
                    index=i, time=df["time"].iloc[i],
                    confidence=min(1.0, body_i1 / (atr_val * 1.5)),
                ))

        return self._filter_unmitigated_ob(df, patterns)

    def _filter_unmitigated_ob(self, df: pd.DataFrame, patterns: List[Pattern]) -> List[Pattern]:
        valid = []
        for p in patterns:
            after = df.iloc[p.index + 2:]
            if after.empty:
                valid.append(p)
                continue
            if p.direction == "BUY":
                if after["low"].min() > p.high:
                    valid.append(p)
            else:
                if after["high"].max() < p.low:
                    valid.append(p)
        return valid

    def detect_breakers(self, df: pd.DataFrame, last_n_bars: int = 100) -> List[Pattern]:
        patterns: List[Pattern] = []
        if df is None or len(df) < 20:
            return patterns

        highs_idx, lows_idx = find_swing_points(df, lookback=4)
        atr_val = atr(df, 14).iloc[-1]
        start = max(20, len(df) - last_n_bars)
        seen_levels = {}  # dedup by mid price cluster

        for i in range(start, len(df)):
            close_i = df["close"].iloc[i]
            open_i = df["open"].iloc[i]
            body = abs(close_i - open_i)
            if body < atr_val * 0.7:
                continue

            prev_highs = [h for h in highs_idx if h < i and h > i - 50]
            for hi in reversed(prev_highs):
                level = df["high"].iloc[hi]
                if close_i > level and df["close"].iloc[i - 1] < level:
                    dup = False
                    for seen_level in seen_levels:
                        if abs(level - seen_level) <= atr_val * 0.5:
                            dup = True
                            break
                    if not dup:
                        seen_levels[level] = True
                        patterns.append(Pattern(
                            type=PatternType.BREAKER_BULLISH, direction="BUY",
                            high=level + atr_val * 0.5,
                            low=level - atr_val * 0.2,
                            mid=level, index=i, time=df["time"].iloc[i],
                            confidence=min(1.0, body / (atr_val * 1.5)),
                        ))
                    break

            prev_lows = [l for l in lows_idx if l < i and l > i - 50]
            for li in reversed(prev_lows):
                level = df["low"].iloc[li]
                if close_i < level and df["close"].iloc[i - 1] > level:
                    dup = False
                    for seen_level in seen_levels:
                        if abs(level - seen_level) <= atr_val * 0.5:
                            dup = True
                            break
                    if not dup:
                        seen_levels[level] = True
                        patterns.append(Pattern(
                            type=PatternType.BREAKER_BEARISH, direction="SELL",
                            high=level + atr_val * 0.2,
                            low=level - atr_val * 0.5,
                            mid=level, index=i, time=df["time"].iloc[i],
                            confidence=min(1.0, body / (atr_val * 1.5)),
                        ))
                    break

        return patterns

    def detect_liquidity_sweep(self, df: pd.DataFrame, last_n_bars: int = 30) -> List[Pattern]:
        patterns: List[Pattern] = []
        if df is None or len(df) < 12:
            return patterns

        highs_idx, lows_idx = find_swing_points(df, lookback=3)
        start = max(5, len(df) - last_n_bars)

        for i in range(start, len(df)):
            low_i = df["low"].iloc[i]
            high_i = df["high"].iloc[i]
            close_i = df["close"].iloc[i]

            recent_lows = [l for l in lows_idx if l < i and l >= i - 30]
            if recent_lows:
                prev_low_idx = recent_lows[-1]
                prev_low = df["low"].iloc[prev_low_idx]
                if low_i < prev_low and close_i > prev_low:
                    patterns.append(Pattern(
                        type=PatternType.SWEEP_BULLISH, direction="BUY",
                        high=close_i, low=low_i, mid=prev_low,
                        index=i, time=df["time"].iloc[i],
                        extra={"swept_level": prev_low},
                    ))

            recent_highs = [h for h in highs_idx if h < i and h >= i - 30]
            if recent_highs:
                prev_high_idx = recent_highs[-1]
                prev_high = df["high"].iloc[prev_high_idx]
                if high_i > prev_high and close_i < prev_high:
                    patterns.append(Pattern(
                        type=PatternType.SWEEP_BEARISH, direction="SELL",
                        high=high_i, low=close_i, mid=prev_high,
                        index=i, time=df["time"].iloc[i],
                        extra={"swept_level": prev_high},
                    ))

        return patterns

    def detect_cycle(self, df: pd.DataFrame, last_n_bars: int = 80) -> List[Pattern]:
        patterns: List[Pattern] = []
        if df is None or len(df) < 30:
            return patterns

        highs_idx, lows_idx = find_swing_points(df, lookback=self.params.pivot_length // 2 or 5)
        if len(highs_idx) < 2 or len(lows_idx) < 2:
            return patterns

        atr_val = atr(df, 14).iloc[-1]

        for j in range(1, len(highs_idx)):
            slip_idx = highs_idx[j - 1]
            equi_idx = highs_idx[j]
            slip_price = df["high"].iloc[slip_idx]
            equi_price = df["high"].iloc[equi_idx]
            if equi_price <= slip_price:
                continue
            close_equi = df["close"].iloc[equi_idx]
            if close_equi >= slip_price:
                continue
            internal_low = df["low"].iloc[slip_idx:equi_idx + 1].min()
            for k in range(equi_idx + 1, min(equi_idx + 30, len(df))):
                close_k = df["close"].iloc[k]
                body_k = abs(df["close"].iloc[k] - df["open"].iloc[k])
                if close_k < internal_low and body_k > atr_val * 0.7:
                    void_high = equi_price
                    void_low = slip_price
                    mid = void_low + (void_high - void_low) * 0.5
                    patterns.append(Pattern(
                        type=PatternType.CYCLE_BEARISH, direction="SELL",
                        high=void_high, low=void_low, mid=mid,
                        index=k, time=df["time"].iloc[k],
                        extra={"slip_idx": slip_idx, "equidad_idx": equi_idx, "eslabon_idx": k},
                    ))
                    break

        for j in range(1, len(lows_idx)):
            slip_idx = lows_idx[j - 1]
            equi_idx = lows_idx[j]
            slip_price = df["low"].iloc[slip_idx]
            equi_price = df["low"].iloc[equi_idx]
            if equi_price >= slip_price:
                continue
            close_equi = df["close"].iloc[equi_idx]
            if close_equi <= slip_price:
                continue
            internal_high = df["high"].iloc[slip_idx:equi_idx + 1].max()
            for k in range(equi_idx + 1, min(equi_idx + 30, len(df))):
                close_k = df["close"].iloc[k]
                body_k = abs(df["close"].iloc[k] - df["open"].iloc[k])
                if close_k > internal_high and body_k > atr_val * 0.7:
                    void_high = slip_price
                    void_low = equi_price
                    mid = void_low + (void_high - void_low) * 0.5
                    patterns.append(Pattern(
                        type=PatternType.CYCLE_BULLISH, direction="BUY",
                        high=void_high, low=void_low, mid=mid,
                        index=k, time=df["time"].iloc[k],
                        extra={"slip_idx": slip_idx, "equidad_idx": equi_idx, "eslabon_idx": k},
                    ))
                    break

        return patterns

    def detect_wyckoff(self, df: pd.DataFrame, last_n_bars: int = 120) -> List[Pattern]:
        patterns: List[Pattern] = []
        if df is None or len(df) < 40:
            return patterns

        atr_val = atr(df, 14).iloc[-1]
        n = len(df)
        window = self.params.wyckoff_min_phase_bars

        for i in range(window, n - 3):
            box = df.iloc[i - window:i]
            support = box["low"].min()
            resistance = box["high"].max()
            box_range = resistance - support
            if box_range < atr_val * 1.0 or box_range > atr_val * 8:
                continue

            low_i = df["low"].iloc[i]
            close_i = df["close"].iloc[i]
            if low_i < support and close_i > support:
                for k in range(i + 1, min(i + 6, n)):
                    body_k = abs(df["close"].iloc[k] - df["open"].iloc[k])
                    if df["close"].iloc[k] > resistance and body_k > atr_val * 0.7:
                        patterns.append(Pattern(
                            type=PatternType.SPRING_BULLISH, direction="BUY",
                            high=resistance, low=low_i, mid=support,
                            index=k, time=df["time"].iloc[k],
                            extra={"spring_low": low_i, "support": support, "resistance": resistance},
                        ))
                        break

            high_i = df["high"].iloc[i]
            if high_i > resistance and close_i < resistance:
                for k in range(i + 1, min(i + 6, n)):
                    body_k = abs(df["close"].iloc[k] - df["open"].iloc[k])
                    if df["close"].iloc[k] < support and body_k > atr_val * 0.7:
                        patterns.append(Pattern(
                            type=PatternType.UTAD_BEARISH, direction="SELL",
                            high=high_i, low=support, mid=resistance,
                            index=k, time=df["time"].iloc[k],
                            extra={"utad_high": high_i, "support": support, "resistance": resistance},
                        ))
                        break

        return patterns

    def detect_sequence_123(self, df: pd.DataFrame, last_n_bars: int = 50) -> List[Pattern]:
        patterns: List[Pattern] = []
        if df is None or len(df) < 10:
            return patterns

        seq_len = self.params.sequence_length
        atr_val = atr(df, 14).iloc[-1]
        start = max(seq_len, len(df) - last_n_bars)

        for i in range(start, len(df)):
            colors_up = all(df["close"].iloc[i - j] > df["open"].iloc[i - j] for j in range(seq_len))
            colors_dn = all(df["close"].iloc[i - j] < df["open"].iloc[i - j] for j in range(seq_len))

            if colors_up:
                v1 = i - (seq_len - 1)
                v2 = i - (seq_len - 2)
                if not candle_is_body_dominant(
                    df["open"].iloc[v1], df["high"].iloc[v1],
                    df["low"].iloc[v1], df["close"].iloc[v1],
                ):
                    continue
                mid_v2 = (df["high"].iloc[v2] + df["low"].iloc[v2]) / 2
                patterns.append(Pattern(
                    type=PatternType.SEQUENCE_BULLISH, direction="BUY",
                    high=df["high"].iloc[v2], low=df["low"].iloc[v2],
                    mid=mid_v2, index=v2, time=df["time"].iloc[v2],
                ))

            if colors_dn:
                v1 = i - (seq_len - 1)
                v2 = i - (seq_len - 2)
                if not candle_is_body_dominant(
                    df["open"].iloc[v1], df["high"].iloc[v1],
                    df["low"].iloc[v1], df["close"].iloc[v1],
                ):
                    continue
                mid_v2 = (df["high"].iloc[v2] + df["low"].iloc[v2]) / 2
                patterns.append(Pattern(
                    type=PatternType.SEQUENCE_BEARISH, direction="SELL",
                    high=df["high"].iloc[v2], low=df["low"].iloc[v2],
                    mid=mid_v2, index=v2, time=df["time"].iloc[v2],
                ))

        return patterns

    def detect_void_scalp(self, df: pd.DataFrame, last_n_bars: int = 60) -> List[Pattern]:
        patterns: List[Pattern] = []
        if df is None or len(df) < 20:
            return patterns

        highs_idx, lows_idx = find_swing_points(df, lookback=4)
        if len(highs_idx) < 3 or len(lows_idx) < 3:
            return patterns

        atr_val = atr(df, 14).iloc[-1]
        start = max(20, len(df) - last_n_bars)

        for j in range(1, len(highs_idx)):
            slip_idx = highs_idx[j - 1]
            equi_idx = highs_idx[j]
            slip_price = df["high"].iloc[slip_idx]
            equi_price = df["high"].iloc[equi_idx]

            if equi_price <= slip_price:
                continue
            close_equi = df["close"].iloc[equi_idx]
            if close_equi >= slip_price:
                continue

            internal_low = df["low"].iloc[slip_idx:equi_idx + 1].min()
            void_size = equi_price - slip_price
            if void_size < atr_val * 0.15:
                continue

            for k in range(equi_idx + 1, min(equi_idx + 20, len(df))):
                close_k = df["close"].iloc[k]
                body_k = abs(df["close"].iloc[k] - df["open"].iloc[k])
                if close_k < internal_low and body_k > atr_val * 0.6:
                    void_entry = slip_price + void_size * 0.5
                    patterns.append(Pattern(
                        type=PatternType.VOID_SCALP_BEARISH, direction="SELL",
                        high=equi_price, low=slip_price, mid=void_entry,
                        index=k, time=df["time"].iloc[k],
                        confidence=min(1.0, void_size / (atr_val * 0.5)),
                        extra={"slip_idx": slip_idx, "equidad_idx": equi_idx,
                               "void_size": void_size, "eslabon_idx": k},
                    ))
                    break

        for j in range(1, len(lows_idx)):
            slip_idx = lows_idx[j - 1]
            equi_idx = lows_idx[j]
            slip_price = df["low"].iloc[slip_idx]
            equi_price = df["low"].iloc[equi_idx]

            if equi_price >= slip_price:
                continue
            close_equi = df["close"].iloc[equi_idx]
            if close_equi <= slip_price:
                continue

            internal_high = df["high"].iloc[slip_idx:equi_idx + 1].max()
            void_size = slip_price - equi_price
            if void_size < atr_val * 0.15:
                continue

            for k in range(equi_idx + 1, min(equi_idx + 20, len(df))):
                close_k = df["close"].iloc[k]
                body_k = abs(df["close"].iloc[k] - df["open"].iloc[k])
                if close_k > internal_high and body_k > atr_val * 0.6:
                    void_entry = equi_price + void_size * 0.5
                    patterns.append(Pattern(
                        type=PatternType.VOID_SCALP_BULLISH, direction="BUY",
                        high=slip_price, low=equi_price, mid=void_entry,
                        index=k, time=df["time"].iloc[k],
                        confidence=min(1.0, void_size / (atr_val * 0.5)),
                        extra={"slip_idx": slip_idx, "equidad_idx": equi_idx,
                               "void_size": void_size, "eslabon_idx": k},
                    ))
                    break

        return patterns

    def detect_wyckoff_phase_d(self, df: pd.DataFrame, wyckoff_patterns: List[Pattern]) -> List[Pattern]:
        """Detect Wyckoff Phase D (SOS/SOW) after a Spring/UTAD.
        SOS = close above resistance with body > 0.7*ATR and volume > 20-period avg.
        Returns LPS/LPSY level as pattern."""
        phase_d: List[Pattern] = []
        if df is None or len(df) < 30:
            return phase_d
        atr_val = atr(df, 14).iloc[-1]
        volume_col = "volume" if "volume" in df.columns else "tick_volume"
        avg_vol = df[volume_col].iloc[-20:].mean() if volume_col in df.columns else 0

        for wp in wyckoff_patterns:
            if wp.type not in (PatternType.SPRING_BULLISH, PatternType.UTAD_BEARISH):
                continue
            resistance = wp.extra.get("resistance", wp.high)
            support = wp.extra.get("support", wp.low)
            spring_low = wp.extra.get("spring_low", wp.low)
            utad_high = wp.extra.get("utad_high", wp.high)

            for k in range(wp.index + 1, min(wp.index + 10, len(df))):
                close_k = df["close"].iloc[k]
                body_k = abs(df["close"].iloc[k] - df["open"].iloc[k])
                vol_ok = True
                if avg_vol > 0 and volume_col in df.columns:
                    vol_ok = df[volume_col].iloc[k] > avg_vol

                if wp.type == PatternType.SPRING_BULLISH:
                    if close_k > resistance and body_k > atr_val * 0.7 and vol_ok:
                        for l in range(k + 1, min(k + 8, len(df))):
                            if df["close"].iloc[l] < df["high"].iloc[k]:
                                lps_price = df["low"].iloc[k] + (df["high"].iloc[k] - df["low"].iloc[k]) * 0.5
                                phase_d.append(Pattern(
                                    type=PatternType.SOS_BULLISH, direction="BUY",
                                    high=resistance, low=support, mid=lps_price,
                                    index=l, time=df["time"].iloc[l],
                                    confidence=min(1.0, body_k / (atr_val * 1.5)),
                                    extra={"sos_idx": k, "lps_level": lps_price,
                                           "spring_low": spring_low, "support": support},
                                ))
                                break
                        break

                elif wp.type == PatternType.UTAD_BEARISH:
                    if close_k < support and body_k > atr_val * 0.7 and vol_ok:
                        for l in range(k + 1, min(k + 8, len(df))):
                            if df["close"].iloc[l] > df["low"].iloc[k]:
                                lps_price = df["low"].iloc[k] + (df["high"].iloc[k] - df["low"].iloc[k]) * 0.5
                                phase_d.append(Pattern(
                                    type=PatternType.SOW_BEARISH, direction="SELL",
                                    high=resistance, low=support, mid=lps_price,
                                    index=l, time=df["time"].iloc[l],
                                    confidence=min(1.0, body_k / (atr_val * 1.5)),
                                    extra={"sow_idx": k, "lpsy_level": lps_price,
                                           "utad_high": utad_high, "resistance": resistance},
                                ))
                                break
                        break

        return phase_d

    def _fvg_mitigation_pct(self, df: pd.DataFrame, fvg: Pattern) -> float:
        """Calculate what % of FVG has been mitigated by subsequent price."""
        after = df.iloc[fvg.index + 1:]
        if after.empty:
            return 0.0
        if fvg.direction == "BUY":
            lowest = after["low"].min()
            if lowest <= fvg.low:
                return 1.0
            if lowest >= fvg.high:
                return 0.0
            return (lowest - fvg.low) / (fvg.high - fvg.low)
        else:
            highest = after["high"].max()
            if highest >= fvg.high:
                return 1.0
            if highest <= fvg.low:
                return 0.0
            return (fvg.high - highest) / (fvg.high - fvg.low)

    def _validate_dual_block(self, df: pd.DataFrame, ob_idx: int, direction: str) -> bool:
        """Dual Block Anatomy: second candle body must engulf first candle wick."""
        if ob_idx + 1 >= len(df):
            return True
        c1 = df.iloc[ob_idx]
        c2 = df.iloc[ob_idx + 1]
        body2 = abs(c2["close"] - c2["open"])
        wick1 = c1["high"] - c1["low"]
        if body2 == 0 or wick1 == 0:
            return False
        engulf_ratio = body2 / wick1
        return engulf_ratio >= 0.7

    def check_body_close_invalidation(self, df: pd.DataFrame, patterns: List[Pattern]) -> bool:
        """Body Close Rule: if Spring/UTAD closed with BODY outside range → real breakout, not spring."""
        for p in patterns:
            if p.type not in (PatternType.SPRING_BULLISH, PatternType.UTAD_BEARISH):
                continue
            support = p.extra.get("support", p.low)
            resistance = p.extra.get("resistance", p.high)
            spring_low = p.extra.get("spring_low", p.low)
            utad_high = p.extra.get("utad_high", p.high)
            spring_idx = p.extra.get("index", p.index)
            if spring_idx >= len(df):
                continue
            candle = df.iloc[spring_idx]
            if p.type == PatternType.SPRING_BULLISH:
                if candle["close"] < support:  # body closed below support → real breakout
                    return True
            elif p.type == PatternType.UTAD_BEARISH:
                if candle["close"] > resistance:  # body closed above resistance → real breakout
                    return True
        return False

    def _swing_highs(self, df: pd.DataFrame, lookback: int = 5) -> List[int]:
        idxs = []
        n = len(df)
        for i in range(lookback, n - lookback):
            win = df["high"].iloc[i - lookback:i + lookback + 1]
            if df["high"].iloc[i] == win.max():
                idxs.append(i)
        return idxs

    def _swing_lows(self, df: pd.DataFrame, lookback: int = 5) -> List[int]:
        idxs = []
        n = len(df)
        for i in range(lookback, n - lookback):
            win = df["low"].iloc[i - lookback:i + lookback + 1]
            if df["low"].iloc[i] == win.min():
                idxs.append(i)
        return idxs

    def detect_bos_zone_retest(self, df: pd.DataFrame, lookback_indec: int = 10,
                                swing_len: int = 5, body_ratio: float = 0.45) -> List[Pattern]:
        patterns: List[Pattern] = []
        if df is None or len(df) < 30:
            return patterns

        atr_val = atr(df, 14).iloc[-1]
        vol_sma = df["volume"].iloc[-20:].mean() if "volume" in df.columns else 0
        if vol_sma == 0:
            vol_sma = atr_val * 10

        current_close = df["close"].iloc[-1]
        current_low = df["low"].iloc[-1]
        current_high = df["high"].iloc[-1]

        highs_idx = self._swing_highs(df, swing_len)
        lows_idx = self._swing_lows(df, swing_len)

        for i in range(max(5, len(df) - lookback_indec * 3), len(df)):
            zone_min = float("inf")
            zone_max = float("-inf")
            count_indec = 0

            for j in range(i - lookback_indec, i):
                if j < 0:
                    continue
                o, h, l, c = df["open"].iloc[j], df["high"].iloc[j], df["low"].iloc[j], df["close"].iloc[j]
                body = abs(c - o)
                rng = h - l
                if rng > 0 and (body / rng) < (1.0 - body_ratio):
                    count_indec += 1
                    zone_min = min(zone_min, l)
                    zone_max = max(zone_max, h)

            if count_indec == 0 or zone_min == float("inf") or zone_max == float("-inf"):
                continue

            if i + 1 >= len(df):
                break
            c1 = df.iloc[i]
            body1 = abs(c1["close"] - c1["open"])
            rng1 = c1["high"] - c1["low"]
            is_body = (rng1 > 0 and (body1 / rng1) >= body_ratio)
            vol_ok = (not ("volume" in df.columns)) or (
                c1["volume"] >= vol_sma * 1.2 and c1["volume"] <= vol_sma * 4.0
            )

            if not (is_body and vol_ok):
                continue

            is_bearish_break = c1["close"] < c1["open"] and c1["close"] < zone_min
            is_bullish_break = c1["close"] > c1["open"] and c1["close"] > zone_max

            if is_bullish_break:
                valid_trend = any(
                    idx < i - lookback_indec and df["high"].iloc[idx] > zone_max
                    for idx in highs_idx
                )

                retesting = (current_low <= zone_max * 1.001 and
                            current_close > zone_max * 0.995)
                if retesting:
                    sl_price = c1["low"] - atr_val * 0.15
                    patterns.append(Pattern(
                        type=PatternType.BOS_ZONE_RETEST_BULLISH, direction="BUY",
                        high=zone_max, low=zone_min, mid=zone_max,
                        index=i, time=df["time"].iloc[i],
                        confidence=min(1.0, body1 / (atr_val * 2)),
                        extra={"bos_high": c1["high"], "bos_low": c1["low"],
                               "sl_price": sl_price, "zone_min": zone_min,
                               "zone_max": zone_max, "indec_count": count_indec},
                    ))

            elif is_bearish_break:
                valid_trend = any(
                    idx < i - lookback_indec and df["low"].iloc[idx] < zone_min
                    for idx in lows_idx
                )

                retesting = (current_high >= zone_min * 0.999 and
                            current_close < zone_min * 1.005)
                if retesting:
                    sl_price = c1["high"] + atr_val * 0.15
                    patterns.append(Pattern(
                        type=PatternType.BOS_ZONE_RETEST_BEARISH, direction="SELL",
                        high=zone_max, low=zone_min, mid=zone_min,
                        index=i, time=df["time"].iloc[i],
                        confidence=min(1.0, body1 / (atr_val * 2)),
                        extra={"bos_high": c1["high"], "bos_low": c1["low"],
                               "sl_price": sl_price, "zone_min": zone_min,
                               "zone_max": zone_max, "indec_count": count_indec},
                    ))
        return patterns

    def detect_price_establishment(self, df: pd.DataFrame, lookback: int = 20) -> List[Pattern]:
        patterns: List[Pattern] = []
        if df is None or len(df) < lookback + 5:
            return patterns

        from collections import Counter
        levels = Counter()
        for i in range(max(5, len(df) - lookback), len(df)):
            h = round(df["high"].iloc[i], 2)
            l = round(df["low"].iloc[i], 2)
            c = round(df["close"].iloc[i], 2)
            levels[h] += 1
            levels[l] += 1
            levels[c] += 1

        common = [level for level, count in levels.most_common(5) if count >= 3]

        current_close = df["close"].iloc[-1]
        current_low = df["low"].iloc[-1]
        current_high = df["high"].iloc[-1]

        for level in common:
            below = [i for i in range(max(0, len(df) - 10), len(df))
                     if df["low"].iloc[i] <= level <= df["high"].iloc[i]]
            if len(below) >= 3:
                reject_up = all(df["close"].iloc[i] <= level * 1.001 for i in below)
                if reject_up and abs(current_high - level) / level < 0.002:
                    patterns.append(Pattern(
                        type=PatternType.PRICE_ESTABLISHMENT_SHORT, direction="SELL",
                        high=level * 1.002, low=level * 0.998, mid=level,
                        index=len(df) - 1, time=df["time"].iloc[-1],
                        confidence=min(1.0, len(below) / 6),
                        extra={"level": level, "touches": len(below)},
                    ))

                reject_down = all(df["close"].iloc[i] >= level * 0.999 for i in below)
                if reject_down and abs(current_low - level) / level < 0.002:
                    patterns.append(Pattern(
                        type=PatternType.PRICE_ESTABLISHMENT_LONG, direction="BUY",
                        high=level * 1.002, low=level * 0.998, mid=level,
                        index=len(df) - 1, time=df["time"].iloc[-1],
                        confidence=min(1.0, len(below) / 6),
                        extra={"level": level, "touches": len(below)},
                    ))
        return patterns

    def count_sub_fractals(self, df: pd.DataFrame, lookback: int = 40) -> dict:
        result = {"count": 0, "direction": None, "third_movement_ready": False, "fractals": []}
        if df is None or len(df) < lookback:
            return result

        highs_idx = self._swing_highs(df, 3)
        lows_idx = self._swing_lows(df, 3)
        swing_points = []
        for idx in highs_idx:
            if idx >= len(df) - lookback:
                swing_points.append((idx, df["high"].iloc[idx], "high"))
        for idx in lows_idx:
            if idx >= len(df) - lookback:
                swing_points.append((idx, df["low"].iloc[idx], "low"))
        swing_points.sort(key=lambda x: x[0])
        if len(swing_points) < 4:
            return result

        for i in range(2, len(swing_points) - 1):
            p0, p1, p2 = swing_points[i - 2], swing_points[i - 1], swing_points[i]
            if p0[2] == "high" and p1[2] == "low" and p2[2] == "high":
                if p1[1] < p0[1] and p2[1] > p1[1]:
                    result["count"] += 1
                    result["fractals"].append({"type": "bullish", "start": p0[1], "mid": p1[1], "end": p2[1]})
            elif p0[2] == "low" and p1[2] == "high" and p2[2] == "low":
                if p1[1] > p0[1] and p2[1] < p1[1]:
                    result["count"] += 1
                    result["fractals"].append({"type": "bearish", "start": p0[1], "mid": p1[1], "end": p2[1]})

        if result["fractals"]:
            last = result["fractals"][-1]
            result["direction"] = last["type"]

        last_prices = []
        for idx in highs_idx[-3:]:
            if idx >= 0:
                last_prices.append(df["high"].iloc[idx])
        for idx in lows_idx[-3:]:
            if idx >= 0:
                last_prices.append(df["low"].iloc[idx])

        if len(result["fractals"]) >= 2:
            last_two = result["fractals"][-2:]
            all_same = all(f["type"] == last_two[0]["type"] for f in last_two)
            if all_same:
                result["third_movement_ready"] = True
                result["direction"] = last_two[0]["type"]

        return result

    def detect_interval_points(self, df: pd.DataFrame, last_n_bars: int = 80) -> List[Pattern]:
        """Punto de Intervalo (Clase 13, 15): vela de indecisión con mechas ≥ 2× el cuerpo.
        Actúa como límite de precio institucional.
        Requisito: mechas deben duplicar o triplicar el tamaño del cuerpo."""
        patterns: List[Pattern] = []
        if df is None or len(df) < 10:
            return patterns
        start = max(2, len(df) - last_n_bars)
        for i in range(start, len(df)):
            o, h, l, c = df["open"].iloc[i], df["high"].iloc[i], df["low"].iloc[i], df["close"].iloc[i]
            body = abs(c - o)
            rng = h - l
            if rng <= 0 or body <= 0:
                continue
            wick_ratio = (rng - body) / body
            if wick_ratio < 2.0:
                continue
            mid = (h + l) * 0.5
            if c >= o:
                patterns.append(Pattern(
                    type=PatternType.INTERVAL_POINT_BULLISH, direction="BUY",
                    high=h, low=l, mid=mid, index=i, time=df["time"].iloc[i],
                    confidence=min(1.0, wick_ratio / 4.0),
                    extra={"wick_ratio": wick_ratio, "is_bullish": True},
                ))
            else:
                patterns.append(Pattern(
                    type=PatternType.INTERVAL_POINT_BEARISH, direction="SELL",
                    high=h, low=l, mid=mid, index=i, time=df["time"].iloc[i],
                    confidence=min(1.0, wick_ratio / 4.0),
                    extra={"wick_ratio": wick_ratio, "is_bullish": False},
                ))
        return patterns

    def detect_price_interaction(self, df: pd.DataFrame, last_n_bars: int = 60) -> List[Pattern]:
        """Interacción de Precio (Clase 12, 16, 23, 24): el precio actual interactúa
        con un nivel previo (mecha a mecha, cuerpo a mecha).
        Busca la 'vela responsable' que toca un swing previo y es absorbida con volumen."""
        patterns: List[Pattern] = []
        if df is None or len(df) < 20:
            return patterns
        atr_val = atr(df, 14).iloc[-1]
        highs_idx, lows_idx = find_swing_points(df, lookback=3)
        start = max(10, len(df) - last_n_bars)
        for i in range(start, len(df)):
            if i < 2:
                continue
            o_i, h_i, l_i, c_i = df["open"].iloc[i], df["high"].iloc[i], df["low"].iloc[i], df["close"].iloc[i]
            body_i = abs(c_i - o_i)
            if body_i < atr_val * 0.4:
                continue
            for hi in reversed(highs_idx):
                if hi >= i or i - hi > 30:
                    continue
                level = df["high"].iloc[hi]
                if abs(h_i - level) <= atr_val * 0.15 and c_i < level:
                    mid = (h_i + level) * 0.5
                    patterns.append(Pattern(
                        type=PatternType.PRICE_INTERACTION_BEARISH, direction="SELL",
                        high=h_i, low=min(l_i, level), mid=mid,
                        index=i, time=df["time"].iloc[i],
                        confidence=min(1.0, body_i / (atr_val * 1.2)),
                        extra={"interacted_level": level, "interaction_type": "wick_to_wick"},
                    ))
                    break
            for li in reversed(lows_idx):
                if li >= i or i - li > 30:
                    continue
                level = df["low"].iloc[li]
                if abs(l_i - level) <= atr_val * 0.15 and c_i > level:
                    mid = (l_i + level) * 0.5
                    patterns.append(Pattern(
                        type=PatternType.PRICE_INTERACTION_BULLISH, direction="BUY",
                        high=max(h_i, level), low=l_i, mid=mid,
                        index=i, time=df["time"].iloc[i],
                        confidence=min(1.0, body_i / (atr_val * 1.2)),
                        extra={"interacted_level": level, "interaction_type": "wick_to_wick"},
                    ))
                    break
        return patterns

    def detect_harmonic_cycle(self, df: pd.DataFrame, last_n_bars: int = 100) -> List[Pattern]:
        """Ciclo Armónico (Clase 11, 18): calcula el 50% del último impulso completo
        (swing low → swing high para alcista, swing high → swing low para bajista).
        El 50% del rango es nivel de descuento institucional."""
        patterns: List[Pattern] = []
        if df is None or len(df) < 30:
            return patterns
        atr_val = atr(df, 14).iloc[-1]
        highs_idx, lows_idx = find_swing_points(df, lookback=4)
        current_close = df["close"].iloc[-1]
        current_price = df["close"].iloc[-1]
        if len(highs_idx) >= 2 and len(lows_idx) >= 2:
            last_high_idx = highs_idx[-1]
            last_low_idx = lows_idx[-1]
            if last_low_idx > last_high_idx:
                swing_low = df["low"].iloc[last_low_idx]
                swing_high = df["high"].iloc[last_high_idx]
                for hi in reversed(highs_idx):
                    if hi > last_low_idx:
                        continue
                    swing_high = df["high"].iloc[hi]
                    break
                harmonic_50 = swing_low + (swing_high - swing_low) * 0.5
                if abs(current_price - harmonic_50) <= atr_val * 1.5:
                    patterns.append(Pattern(
                        type=PatternType.HARMONIC_CYCLE_BULLISH, direction="BUY",
                        high=swing_high, low=swing_low, mid=harmonic_50,
                        index=len(df) - 1, time=df["time"].iloc[-1],
                        confidence=max(0.3, 1.0 - abs(current_price - harmonic_50) / (atr_val * 1.5)),
                        extra={"harmonic_level": harmonic_50, "swing_high": swing_high, "swing_low": swing_low},
                    ))
            elif last_high_idx > last_low_idx:
                swing_high = df["high"].iloc[last_high_idx]
                swing_low = df["low"].iloc[last_low_idx]
                for li in reversed(lows_idx):
                    if li > last_high_idx:
                        continue
                    swing_low = df["low"].iloc[li]
                    break
                harmonic_50 = swing_low + (swing_high - swing_low) * 0.5
                if abs(current_price - harmonic_50) <= atr_val * 1.5:
                    patterns.append(Pattern(
                        type=PatternType.HARMONIC_CYCLE_BEARISH, direction="SELL",
                        high=swing_high, low=swing_low, mid=harmonic_50,
                        index=len(df) - 1, time=df["time"].iloc[-1],
                        confidence=max(0.3, 1.0 - abs(current_price - harmonic_50) / (atr_val * 1.5)),
                        extra={"harmonic_level": harmonic_50, "swing_high": swing_high, "swing_low": swing_low},
                    ))
        return patterns

    def detect_pressure_zones(self, df: pd.DataFrame, last_n_bars: int = 50) -> List[Pattern]:
        """Zonas de Presión de Oferta/Demanda POD (Clase 25):
        precio falla 1-3 veces un nivel, creando consolidación estrecha con bajo volumen,
        luego expansión direccional."""
        patterns: List[Pattern] = []
        if df is None or len(df) < 20:
            return patterns
        atr_val = atr(df, 14).iloc[-1]
        start = max(10, len(df) - last_n_bars)
        vol_col = "volume" if "volume" in df.columns else "tick_volume"
        for i in range(start, len(df) - 3):
            zone_high = df["high"].iloc[i - 5:i + 1].max()
            zone_low = df["low"].iloc[i - 5:i + 1].min()
            zone_range = zone_high - zone_low
            if zone_range < atr_val * 0.15 or zone_range > atr_val * 2.0:
                continue
            avg_vol = df[vol_col].iloc[i - 5:i + 1].mean() if vol_col in df.columns else 0
            touches = 0
            for j in range(i - 5, i + 1):
                if j < 0:
                    continue
                if abs(df["high"].iloc[j] - zone_high) <= atr_val * 0.1:
                    touches += 1
                if abs(df["low"].iloc[j] - zone_low) <= atr_val * 0.1:
                    touches += 1
            if touches < 2:
                continue
            body_fwd = abs(df["close"].iloc[i + 1] - df["open"].iloc[i + 1])
            if body_fwd > atr_val * 0.6:
                fwd_close = df["close"].iloc[i + 1]
                fwd_open = df["open"].iloc[i + 1]
                fwd_low = df["low"].iloc[i + 1]
                fwd_high = df["high"].iloc[i + 1]
                if avg_vol > 0 and vol_col in df.columns:
                    fwd_vol = df[vol_col].iloc[i + 1]
                    if fwd_vol < avg_vol * 0.8:
                        continue
                if fwd_close > fwd_open and fwd_low <= zone_low + atr_val * 0.1:
                    mid = zone_low + zone_range * 0.5
                    patterns.append(Pattern(
                        type=PatternType.PRESSURE_ZONE_BULLISH, direction="BUY",
                        high=zone_high, low=zone_low, mid=mid,
                        index=i + 1, time=df["time"].iloc[i + 1],
                        confidence=min(1.0, touches / 5.0),
                        extra={"touches": touches, "zone_range": zone_range},
                    ))
                elif fwd_close < fwd_open and fwd_high >= zone_high - atr_val * 0.1:
                    mid = zone_low + zone_range * 0.5
                    patterns.append(Pattern(
                        type=PatternType.PRESSURE_ZONE_BEARISH, direction="SELL",
                        high=zone_high, low=zone_low, mid=mid,
                        index=i + 1, time=df["time"].iloc[i + 1],
                        confidence=min(1.0, touches / 5.0),
                        extra={"touches": touches, "zone_range": zone_range},
                    ))
        return patterns

    def scan_all(self, df: pd.DataFrame) -> dict:
        fvg = self.detect_fvg(df)
        ob = self.detect_order_blocks(df)
        breaker = self.detect_breakers(df)
        sweep = self.detect_liquidity_sweep(df)
        cycle = self.detect_cycle(df)
        wyckoff = self.detect_wyckoff(df)
        sequence = self.detect_sequence_123(df)
        void_scalp = self.detect_void_scalp(df)
        wyckoff_phase_d = self.detect_wyckoff_phase_d(df, wyckoff)
        bos_zone = self.detect_bos_zone_retest(df)
        price_est = self.detect_price_establishment(df)
        sub_fractals = self.count_sub_fractals(df)
        interval_points = self.detect_interval_points(df)
        price_interaction = self.detect_price_interaction(df)
        harmonic_cycle = self.detect_harmonic_cycle(df)
        pressure_zones = self.detect_pressure_zones(df)

        return {
            "fvg": fvg,
            "ob": ob,
            "breaker": breaker,
            "sweep": sweep,
            "cycle": cycle,
            "wyckoff": wyckoff,
            "wyckoff_phase_d": wyckoff_phase_d,
            "sequence": sequence,
            "void_scalp": void_scalp,
            "bos_zone_retest": bos_zone,
            "price_establishment": price_est,
            "sub_fractals": sub_fractals,
            "interval_points": interval_points,
            "price_interaction": price_interaction,
            "harmonic_cycle": harmonic_cycle,
            "pressure_zones": pressure_zones,
        }

"""Advanced Anticipation Module
Divergences, Order Flow Imbalance, Volume Profile, and enhanced DXY correlation.
Provides leading signals that anticipate reversals BEFORE price prints them.
"""
import logging
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from src.utils.helpers import atr, find_swing_points

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# RSI / MACD DIVERGENCE DETECTION
# ─────────────────────────────────────────────

def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    close = df["close"]
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def compute_macd(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal
    return macd_line, signal, histogram


@dataclass
class DivergenceSignal:
    type: str  # "REGULAR_BULLISH", "REGULAR_BEARISH", "HIDDEN_BULLISH", "HIDDEN_BEARISH"
    source: str  # "RSI" or "MACD"
    strength: float  # 0.0-1.0
    price_divergence_pct: float
    oscillator_divergence_pct: float
    last_swing_index: int
    notes: str = ""


class DivergenceDetector:
    """Detects regular and hidden divergences between price and RSI/MACD.
    Regular divergence: price makes HH/LL while oscillator makes LH/HL → trend reversal.
    Hidden divergence: price makes LH/HL while oscillator makes HH/LL → trend continuation.
    """

    @staticmethod
    def detect_all(df: pd.DataFrame, lookback: int = 60) -> Dict[str, List[DivergenceSignal]]:
        if df is None or len(df) < lookback:
            return {"bullish": [], "bearish": []}
        rsi = compute_rsi(df)
        macd_line, _, _ = compute_macd(df)
        results: Dict[str, List[DivergenceSignal]] = {"bullish": [], "bearish": []}
        for source_name, oscillator in [("RSI", rsi), ("MACD", macd_line)]:
            sigs = DivergenceDetector._detect_on_oscillator(df, oscillator, source_name, lookback)
            for s in sigs:
                if "BULLISH" in s.type:
                    results["bullish"].append(s)
                else:
                    results["bearish"].append(s)
        return results

    @staticmethod
    def _detect_on_oscillator(
        df: pd.DataFrame, oscillator: pd.Series,
        source: str, lookback: int,
    ) -> List[DivergenceSignal]:
        signals: List[DivergenceSignal] = []
        segment = df.iloc[-lookback:]
        osc_seg = oscillator.iloc[-lookback:]
        highs_idx, lows_idx = find_swing_points(segment, lookback=3)
        if len(highs_idx) < 2 or len(lows_idx) < 2:
            return signals
        # Price swing high points → look for bearish divergences
        for i in range(len(highs_idx) - 1):
            p1_idx = highs_idx[i]
            p2_idx = highs_idx[i + 1]
            p1_price = segment["high"].iloc[p1_idx]
            p2_price = segment["high"].iloc[p2_idx]
            p1_osc = osc_seg.iloc[p1_idx]
            p2_osc = osc_seg.iloc[p2_idx]
            price_change = (p2_price - p1_price) / p1_price if p1_price != 0 else 0
            osc_change = (p2_osc - p1_osc) / abs(p1_osc) if abs(p1_osc) > 1e-10 else 0
            if price_change > 0.0005 and osc_change < -0.01:
                strength = min(1.0, abs(osc_change) * 5 + abs(price_change) * 200)
                signals.append(DivergenceSignal(
                    type="REGULAR_BEARISH", source=source,
                    strength=strength,
                    price_divergence_pct=price_change * 100,
                    oscillator_divergence_pct=osc_change * 100,
                    last_swing_index=p2_idx,
                    notes=f"Price HH ({p1_price:.2f}→{p2_price:.2f}) while {source} HL ({p1_osc:.1f}→{p2_osc:.1f})",
                ))
            elif price_change < -0.0005 and osc_change > 0.01:
                strength = min(1.0, abs(osc_change) * 5 + abs(price_change) * 200)
                signals.append(DivergenceSignal(
                    type="HIDDEN_BEARISH", source=source,
                    strength=strength,
                    price_divergence_pct=price_change * 100,
                    oscillator_divergence_pct=osc_change * 100,
                    last_swing_index=p2_idx,
                    notes=f"Price LH ({p1_price:.2f}→{p2_price:.2f}) while {source} HH ({p1_osc:.1f}→{p2_osc:.1f})",
                ))
        # Price swing low points → look for bullish divergences
        for i in range(len(lows_idx) - 1):
            p1_idx = lows_idx[i]
            p2_idx = lows_idx[i + 1]
            p1_price = segment["low"].iloc[p1_idx]
            p2_price = segment["low"].iloc[p2_idx]
            p1_osc = osc_seg.iloc[p1_idx]
            p2_osc = osc_seg.iloc[p2_idx]
            price_change = (p2_price - p1_price) / p1_price if p1_price != 0 else 0
            osc_change = (p2_osc - p1_osc) / abs(p1_osc) if abs(p1_osc) > 1e-10 else 0
            if price_change < -0.0005 and osc_change > 0.01:
                strength = min(1.0, abs(osc_change) * 5 + abs(price_change) * 200)
                signals.append(DivergenceSignal(
                    type="REGULAR_BULLISH", source=source,
                    strength=strength,
                    price_divergence_pct=price_change * 100,
                    oscillator_divergence_pct=osc_change * 100,
                    last_swing_index=p2_idx,
                    notes=f"Price LL ({p1_price:.2f}→{p2_price:.2f}) while {source} HH ({p1_osc:.1f}→{p2_osc:.1f})",
                ))
            elif price_change > 0.0005 and osc_change < -0.01:
                strength = min(1.0, abs(osc_change) * 5 + abs(price_change) * 200)
                signals.append(DivergenceSignal(
                    type="HIDDEN_BULLISH", source=source,
                    strength=strength,
                    price_divergence_pct=price_change * 100,
                    oscillator_divergence_pct=osc_change * 100,
                    last_swing_index=p2_idx,
                    notes=f"Price LH ({p1_price:.2f}→{p2_price:.2f}) while {source} LL ({p1_osc:.1f}→{p2_osc:.1f})",
                ))
        return signals

    @staticmethod
    def best_divergence(
        divergences: Dict[str, List[DivergenceSignal]], direction: str,
    ) -> Tuple[Optional[DivergenceSignal], str]:
        """Returns (best_signal, divergence_type) for a given direction.
        divergence_type: "regular", "hidden", or ""."""
        if direction == "BUY":
            candidates = divergences.get("bullish", [])
        else:
            candidates = divergences.get("bearish", [])
        if not candidates:
            return None, ""
        best = max(candidates, key=lambda d: d.strength)
        div_type = "regular" if "REGULAR" in best.type else "hidden"
        return best, div_type


# ─────────────────────────────────────────────
# ORDER FLOW IMBALANCE (tick volume proxy)
# ─────────────────────────────────────────────

@dataclass
class OrderFlowResult:
    delta_imbalance: float  # -1.0 (sell) to 1.0 (buy)
    cumulative_delta: float
    buying_pressure: float  # 0-1 ratio
    selling_pressure: float  # 0-1 ratio
    divergence_active: bool
    absorption_active: bool
    exhaustion_active: bool
    notes: List[str] = field(default_factory=list)


class OrderFlowAnalyzer:
    """Analyzes tick-volume order flow imbalance.
    Without real market depth, we approximate delta using:
    - Up-tick volume = volume on bullish candles (close > open)
    - Down-tick volume = volume on bearish candles (close < open)
    - Indecision volume = volume on doji candles (close ~ open)
    
    This is a REASONABLE proxy for delta in forex where real volume isn't available.
    """

    @staticmethod
    def analyze(df: pd.DataFrame, lookback: int = 30) -> OrderFlowResult:
        if df is None or len(df) < max(lookback, 10):
            return OrderFlowResult(0, 0, 0, 0, False, False, False, ["No data"])
        segment = df.iloc[-lookback:]
        up_vol = 0.0
        down_vol = 0.0
        total_vol = 0.0
        for _, row in segment.iterrows():
            v = row["volume"]
            total_vol += v
            spread = row["high"] - row["low"]
            body = abs(row["close"] - row["open"])
            if spread == 0:
                continue
            body_ratio = body / spread
            if body_ratio < 0.15:
                continue
            if row["close"] > row["open"]:
                up_vol += v
            elif row["close"] < row["open"]:
                down_vol += v
        if total_vol == 0:
            return OrderFlowResult(0, 0, 0, 0, False, False, False, ["No volume"])
        buying_pressure = up_vol / total_vol
        selling_pressure = down_vol / total_vol
        delta = up_vol - down_vol
        delta_imbalance = delta / total_vol if total_vol > 0 else 0.0
        cumulative_delta = delta_imbalance
        # Divergence: price direction vs delta direction
        price_change = segment["close"].iloc[-1] - segment["open"].iloc[0]
        divergence_active = (price_change > 0 and delta_imbalance < -0.1) or \
                            (price_change < 0 and delta_imbalance > 0.1)
        # Absorption: high volume but indecisive delta (smart money absorbing)
        avg_vol = segment["volume"].mean()
        last_vol = segment["volume"].iloc[-1]
        absorption_active = last_vol > avg_vol * 1.5 and abs(delta_imbalance) < 0.15
        # Exhaustion: buying climax with declining delta
        exhaustion_active = buying_pressure > 0.7 and delta_imbalance < 0.2
        notes = []
        if divergence_active:
            notes.append(f"Delta-price divergence: price {price_change:.2f} vs delta {delta_imbalance:.0%}")
        if absorption_active:
            notes.append(f"Volume absorption: vol {last_vol:.0f} > {avg_vol:.0f} avg, delta neutral {delta_imbalance:.0%}")
        if exhaustion_active:
            notes.append(f"Buying exhaustion: pressure {buying_pressure:.0%} but delta {delta_imbalance:.0%}")
        return OrderFlowResult(
            delta_imbalance=round(delta_imbalance, 4),
            cumulative_delta=round(cumulative_delta, 4),
            buying_pressure=round(buying_pressure, 4),
            selling_pressure=round(selling_pressure, 4),
            divergence_active=divergence_active,
            absorption_active=absorption_active,
            exhaustion_active=exhaustion_active,
            notes=notes,
        )


# ─────────────────────────────────────────────
# VOLUME PROFILE (tick volume approximation)
# ─────────────────────────────────────────────

@dataclass
class VolumeProfileResult:
    value_area_high: float
    value_area_low: float
    poc: float
    poc_volume: float
    high_volume_nodes: List[Tuple[float, float]]  # (price, volume)
    low_volume_nodes: List[Tuple[float, float]]   # (price, volume)
    value_area_width_pips: float
    is_compressed: bool  # tight value area relative to ATR
    poc_shift_pips: float  # POC shift from previous period
    notes: List[str] = field(default_factory=list)


class VolumeProfile:
    """Computes a simplified Volume Profile from tick volume data.
    Uses a rolling window to identify POC (Point of Control),
    Value Area (70% of volume), HVN, and LVN.
    
    Parameters:
        num_bins: number of price bins for the profile
    """

    @staticmethod
    def compute(df: pd.DataFrame, num_bins: int = 24, value_area_pct: float = 0.70) -> VolumeProfileResult:
        if df is None or len(df) < num_bins:
            return VolumeProfileResult(0, 0, 0, 0, [], [], 0, False, 0, ["No data"])
        price_min = df["low"].min()
        price_max = df["high"].max()
        if price_max <= price_min:
            return VolumeProfileResult(0, 0, 0, 0, [], [], 0, False, 0, ["Flat market"])
        bin_size = (price_max - price_min) / num_bins
        if bin_size <= 0:
            return VolumeProfileResult(0, 0, 0, 0, [], [], 0, False, 0, ["No range"])
        bins = [[] for _ in range(num_bins)]
        for _, row in df.iterrows():
            vol = row["volume"]
            if vol == 0:
                continue
            lo = row["low"]
            hi = row["high"]
            lo_bin = int((lo - price_min) / bin_size)
            hi_bin = int((hi - price_min) / bin_size)
            lo_bin = max(0, min(num_bins - 1, lo_bin))
            hi_bin = max(0, min(num_bins - 1, hi_bin))
            vol_per_bin = vol / max(1, hi_bin - lo_bin + 1)
            for b in range(lo_bin, hi_bin + 1):
                bins[b].append(vol_per_bin)
        bin_volumes = [sum(v) for v in bins]
        total_volume = sum(bin_volumes)
        if total_volume == 0:
            return VolumeProfileResult(0, 0, 0, 0, [], [], 0, False, 0, ["Zero volume"])
        poc_bin = int(np.argmax(bin_volumes))
        poc_price = price_min + (poc_bin + 0.5) * bin_size
        poc_vol = bin_volumes[poc_bin]
        # Value Area: bins from POC outward until reaching value_area_pct of volume
        sorted_bins = sorted(
            range(num_bins), key=lambda i: bin_volumes[i], reverse=True,
        )
        cumulative = 0.0
        value_bins = set()
        target_volume = total_volume * value_area_pct
        for b in sorted_bins:
            value_bins.add(b)
            cumulative += bin_volumes[b]
            if cumulative >= target_volume:
                break
        if not value_bins:
            return VolumeProfileResult(0, 0, 0, 0, [], [], 0, False, 0, ["No value area"])
        val_bin = min(value_bins)
        vah_bin = max(value_bins)
        val_price = price_min + val_bin * bin_size
        vah_price = price_min + (vah_bin + 1) * bin_size
        # HVN: bins with >= 70% of POC volume
        hvn_threshold = poc_vol * 0.7
        hvn = []
        for b in value_bins:
            if bin_volumes[b] >= hvn_threshold and b != poc_bin:
                p = price_min + (b + 0.5) * bin_size
                hvn.append((round(p, 4), round(bin_volumes[b], 0)))
        # LVN: bins outside value area with low volume
        lvn_threshold = total_volume / num_bins * 0.3
        lvn = []
        for b in range(num_bins):
            if b not in value_bins and bin_volumes[b] < lvn_threshold and bin_volumes[b] > 0:
                p = price_min + (b + 0.5) * bin_size
                lvn.append((round(p, 4), round(bin_volumes[b], 0)))
        pip = 0.1  # XAU default
        va_width_pips = (vah_price - val_price) / pip if pip > 0 else 0
        atr_val = float(atr(df, 14).iloc[-1]) if len(df) >= 14 else 0
        is_compressed = va_width_pips > 0 and atr_val > 0 and (vah_price - val_price) < atr_val * 0.6
        poc_shift_pips = 0.0
        if len(df) >= num_bins * 2:
            half = len(df) // 2
            first_half = df.iloc[:half]
            second_half = df.iloc[half:]
            poc1 = VolumeProfile._find_poc_price(first_half, num_bins, value_area_pct)
            poc2 = VolumeProfile._find_poc_price(second_half, num_bins, value_area_pct)
            if poc1 and poc2:
                poc_shift_pips = (poc2 - poc1) / pip if pip > 0 else 0
        notes = []
        if is_compressed:
            notes.append(f"Value area compressed ({va_width_pips:.0f}p < {atr_val/pip*0.6:.0f}p ATR) — expansion likely")
        if abs(poc_shift_pips) > 15:
            notes.append(f"POC shift {poc_shift_pips:.0f}p — strong directional bias")
        if hvn:
            notes.append(f"{len(hvn)} HVN clusters — institutional interest zones")
        if lvn:
            notes.append(f"{len(lvn)} LVN gaps — low resistance zones")
        return VolumeProfileResult(
            value_area_high=round(vah_price, 4),
            value_area_low=round(val_price, 4),
            poc=round(poc_price, 4),
            poc_volume=round(poc_vol, 0),
            high_volume_nodes=[(round(p, 4), round(v, 0)) for p, v in hvn],
            low_volume_nodes=[(round(p, 4), round(v, 0)) for p, v in lvn],
            value_area_width_pips=round(va_width_pips, 1),
            is_compressed=is_compressed,
            poc_shift_pips=round(poc_shift_pips, 1),
            notes=notes,
        )

    @staticmethod
    def _find_poc_price(df: pd.DataFrame, num_bins: int, value_area_pct: float) -> Optional[float]:
        if df.empty:
            return None
        price_min = df["low"].min()
        price_max = df["high"].max()
        if price_max <= price_min:
            return None
        bin_size = (price_max - price_min) / num_bins
        if bin_size <= 0:
            return None
        bins = [[] for _ in range(num_bins)]
        for _, row in df.iterrows():
            v = row["volume"]
            if v == 0:
                continue
            lo = row["low"]
            hi = row["high"]
            lo_bin = max(0, min(num_bins - 1, int((lo - price_min) / bin_size)))
            hi_bin = max(0, min(num_bins - 1, int((hi - price_min) / bin_size)))
            vol_per_bin = v / max(1, hi_bin - lo_bin + 1)
            for b in range(lo_bin, hi_bin + 1):
                bins[b].append(vol_per_bin)
        bin_volumes = [sum(v) for v in bins]
        poc_bin = int(np.argmax(bin_volumes))
        return price_min + (poc_bin + 0.5) * bin_size

    @staticmethod
    def assess_direction(
        vp: VolumeProfileResult, current_price: float, direction: str, pip: float,
    ) -> Tuple[float, bool, str]:
        """Returns (score_bonus, is_valid, note) for direction assessment."""
        if vp.poc == 0:
            return 0, False, ""
        score = 0.0
        note = ""
        is_valid = False
        price_pct_vs_va = (current_price - vp.value_area_low) / max(1, vp.value_area_high - vp.value_area_low)
        if direction == "BUY":
            if current_price <= vp.poc:
                score = 10.0
                note = f"Price below POC {vp.poc:.2f} — discount zone"
                is_valid = True
            if current_price < vp.value_area_low and vp.low_volume_nodes:
                score += 8.0
                note += " | LVN gap below VAL — low resistance upward"
                is_valid = True
            if vp.is_compressed:
                score += 5.0
                note += " | Compressed VA — expansion upward"
        else:
            if current_price >= vp.poc:
                score = 10.0
                note = f"Price above POC {vp.poc:.2f} — premium zone"
                is_valid = True
            if current_price > vp.value_area_high and vp.low_volume_nodes:
                score += 8.0
                note += " | LVN gap above VAH — low resistance downward"
                is_valid = True
            if vp.is_compressed:
                score += 5.0
                note += " | Compressed VA — expansion downward"
        if abs(vp.poc_shift_pips) > 20:
            aligned = (vp.poc_shift_pips > 0 and direction == "BUY") or \
                      (vp.poc_shift_pips < 0 and direction == "SELL")
            if aligned:
                score += 7.0
                note += f" | POC shift {vp.poc_shift_pips:.0f}p aligned"
            else:
                score -= 8.0
                note += f" | POC shift {vp.poc_shift_pips:.0f}p against"
        return score, is_valid, note

    @staticmethod
    def get_absorption_levels(vp: VolumeProfileResult) -> List[float]:
        """Return price levels where institutional absorption is likely (HVN + POC)."""
        levels = []
        if vp.poc > 0:
            levels.append(vp.poc)
        for p, v in vp.high_volume_nodes:
            levels.append(p)
        return sorted(set(levels))


# ─────────────────────────────────────────────
# ENHANCED DXY CORRELATION + DIVERGENCE
# ─────────────────────────────────────────────

@dataclass
class DXYAnalysis:
    trend: str  # "bullish", "bearish", "neutral"
    strength: float  # 0-1
    correlation: float  # -1 to 1 (should be negative for XAU)
    divergence_active: bool  # XAU and DXY moving in same direction = divergence
    divergence_strength: float  # 0-1
    regime: str  # "normal", "divergent", "breakdown"
    notes: List[str] = field(default_factory=list)


class DXYCorrelationAnalyzer:
    @staticmethod
    def analyze(dxy_df: Optional[pd.DataFrame], xau_df: Optional[pd.DataFrame]) -> DXYAnalysis:
        if dxy_df is None or len(dxy_df) < 30:
            return DXYAnalysis("neutral", 0, 0, False, 0, "normal", ["No DXY data"])
        dxy_close = dxy_df["close"]
        dxy_sma20 = dxy_close.iloc[-20:].mean()
        dxy_current = dxy_close.iloc[-1]
        dxy_pct = (dxy_current - dxy_sma20) / dxy_sma20
        if abs(dxy_pct) < 0.0005:
            dxy_trend = "neutral"
            dxy_strength = 0
        elif dxy_pct > 0:
            dxy_trend = "bullish"
            dxy_strength = min(1.0, abs(dxy_pct) * 50)
        else:
            dxy_trend = "bearish"
            dxy_strength = min(1.0, abs(dxy_pct) * 50)
        dxy_roc = dxy_close.diff(5).iloc[-1] / dxy_close.iloc[-5] if len(dxy_close) >= 5 else 0
        # Correlation: running 20-bar correlation
        dxy_recent = dxy_close.iloc[-20:]
        correlation = 0.0
        if xau_df is not None and len(xau_df) >= 20:
            xau_close = xau_df["close"].iloc[-20:]
            if len(dxy_recent) == len(xau_close):
                corr = dxy_recent.corr(xau_close)
                correlation = round(corr, 3) if not pd.isna(corr) else 0.0
        # Divergence: XAU moving same direction as DXY = anomaly
        xau_pct = 0
        if xau_df is not None and len(xau_df) >= 5:
            xau_pct = (xau_df["close"].iloc[-1] - xau_df["close"].iloc[-5]) / xau_df["close"].iloc[-5]
        divergence_active = (dxy_roc > 0 and xau_pct > 0) or (dxy_roc < 0 and xau_pct < 0)
        div_strength = abs(dxy_roc) * 50 + abs(xau_pct) * 200 if divergence_active else 0
        div_strength = min(1.0, div_strength)
        if divergence_active and div_strength > 0.5:
            regime = "divergent"
        elif abs(dxy_pct) < 0.001:
            regime = "breakdown"
        else:
            regime = "normal"
        notes = []
        notes.append(f"DXY trend: {dxy_trend} ({dxy_pct*100:.2f}%)")
        if correlation < -0.3:
            notes.append(f"Inverse correlation active: r={correlation:.2f}")
        elif correlation > 0.3:
            notes.append(f"Correlation anomaly: r={correlation:.2f} (expected negative)")
        if divergence_active:
            notes.append(f"DXY-XAU divergence: both moving {dxy_trend} — breakdown possible")
        return DXYAnalysis(
            trend=dxy_trend, strength=round(dxy_strength, 2),
            correlation=round(correlation, 3),
            divergence_active=divergence_active,
            divergence_strength=round(div_strength, 2),
            regime=regime, notes=notes,
        )


# ─────────────────────────────────────────────
# CONSOLIDATED ANTICIPATION RESULT
# ─────────────────────────────────────────────

@dataclass
class AnticipationResult:
    divergences: Dict[str, List[DivergenceSignal]]
    best_bullish_divergence: Optional[DivergenceSignal]
    best_bearish_divergence: Optional[DivergenceSignal]
    bullish_div_type: str  # "regular", "hidden", or ""
    bearish_div_type: str
    order_flow: OrderFlowResult
    volume_profile: VolumeProfileResult
    dxy: DXYAnalysis
    buy_score: float  # accumulated bonus from all anticipation signals
    sell_score: float
    notes: List[str] = field(default_factory=list)


def analyze_all(
    df: pd.DataFrame, dxy_df: Optional[pd.DataFrame],
    direction_buy: bool = True, direction_sell: bool = True,
) -> AnticipationResult:
    """Run ALL anticipation analyses on the given dataframe.
    Returns scores and structured data for integration into the scoring engine."""
    notes: List[str] = []
    # 1. Divergences
    divergences = DivergenceDetector.detect_all(df)
    best_bull, bull_div_type = DivergenceDetector.best_divergence(divergences, "BUY")
    best_bear, bear_div_type = DivergenceDetector.best_divergence(divergences, "SELL")
    # 2. Order flow
    of = OrderFlowAnalyzer.analyze(df)
    # 3. Volume profile
    vp = VolumeProfile.compute(df)
    pip_val = 0.1  # will be overridden by caller
    # 4. DXY analysis
    dxy = DXYCorrelationAnalyzer.analyze(dxy_df, df)
    # 5. Compute accumulated scores
    buy_score = 0.0
    sell_score = 0.0
    if best_bull:
        score_div = 12 * best_bull.strength * (1.5 if bull_div_type == "regular" else 0.8)
        buy_score += score_div
        notes.append(f"Divergence BUY: {bull_div_type} {best_bull.source} (str={best_bull.strength:.0%})")
    if best_bear:
        score_div = 12 * best_bear.strength * (1.5 if bear_div_type == "regular" else 0.8)
        sell_score += score_div
        notes.append(f"Divergence SELL: {bear_div_type} {best_bear.source} (str={best_bear.strength:.0%})")
    # Order flow bonuses
    if of.buying_pressure > of.selling_pressure * 1.5:
        buy_score += 8.0 * of.buying_pressure
        notes.append(f"Order flow bullish: {of.buying_pressure:.0%} buying pressure")
    elif of.selling_pressure > of.buying_pressure * 1.5:
        sell_score += 8.0 * of.selling_pressure
        notes.append(f"Order flow bearish: {of.selling_pressure:.0%} selling pressure")
    if of.divergence_active:
        if of.delta_imbalance > 0:
            buy_score += 10.0
            notes.append("OF divergence: delta up while price down — bullish")
        else:
            sell_score += 10.0
            notes.append("OF divergence: delta down while price up — bearish")
    if of.exhaustion_active:
        sell_score += 6.0  # buying exhaustion = bearish
        notes.append("OF exhaustion: buying climax fading")
    if of.absorption_active:
        buy_score += 5.0
        sell_score += 5.0
        notes.append("OF absorption: large volume, neutral delta — smart money")
    # Volume profile bonuses
    current_price = df["close"].iloc[-1]
    vp_score_buy, vp_valid_buy, vp_note_buy = VolumeProfile.assess_direction(
        vp, current_price, "BUY", pip_val,
    )
    vp_score_sell, vp_valid_sell, vp_note_sell = VolumeProfile.assess_direction(
        vp, current_price, "SELL", pip_val,
    )
    if vp_valid_buy:
        buy_score += vp_score_buy
        notes.append(f"VP BUY: {vp_note_buy}")
    if vp_valid_sell:
        sell_score += vp_score_sell
        notes.append(f"VP SELL: {vp_note_sell}")
    # DXY bonuses
    if dxy.trend == "bearish" and dxy.strength > 0.3:
        buy_score += 8.0 * dxy.strength
        notes.append(f"DXY bearish ({dxy.strength:.0%}) — tailwind for XAU")
    elif dxy.trend == "bullish" and dxy.strength > 0.3:
        sell_score += 8.0 * dxy.strength
        notes.append(f"DXY bullish ({dxy.strength:.0%}) — tailwind for SELL")
    if dxy.divergence_active:
        if xau_pct_positive(df):
            sell_score += 10.0 * dxy.divergence_strength
            notes.append(f"DXY-XAU divergence: both up — XAU SELL signal")
        else:
            buy_score += 10.0 * dxy.divergence_strength
            notes.append(f"DXY-XAU divergence: both down — XAU BUY signal")
    return AnticipationResult(
        divergences=divergences,
        best_bullish_divergence=best_bull,
        best_bearish_divergence=best_bear,
        bullish_div_type=bull_div_type,
        bearish_div_type=bear_div_type,
        order_flow=of,
        volume_profile=vp,
        dxy=dxy,
        buy_score=round(buy_score, 1),
        sell_score=round(sell_score, 1),
        notes=notes,
    )


def xau_pct_positive(df: pd.DataFrame) -> bool:
    if df is None or len(df) < 5:
        return False
    return df["close"].iloc[-1] > df["close"].iloc[-5]


# ─────────────────────────────────────────────
# ORDER FLOW ENGINE INTEGRATION (Modo Experto)
# Bridge between new OrderFlowEngine and AnticipationResult
# ─────────────────────────────────────────────

def integrate_order_flow_signal(
    of_signal: 'OrderFlowSignal',
    current_price: float,
    direction_buy: bool = True,
    direction_sell: bool = True,
    pip: float = 0.0001,
) -> Tuple[float, float, List[str]]:
    """Integrate OrderFlowEngine results into anticipation scoring.
    
    Returns (buy_score_boost, sell_score_boost, notes)
    """
    from src.core.order_flow import OrderFlowSignal as OFS
    if not isinstance(of_signal, OFS):
        return 0.0, 0.0, ["No OrderFlow signal"]

    buy_boost = 0.0
    sell_boost = 0.0
    notes = []

    if direction_buy:
        buy_bonus, buy_reason = _approx_contribution(of_signal, "BUY")
        buy_boost += buy_bonus
        if buy_reason:
            notes.append(buy_reason)
    if direction_sell:
        sell_bonus, sell_reason = _approx_contribution(of_signal, "SELL")
        sell_boost += sell_bonus
        if sell_reason:
            notes.append(sell_reason)

    # Cross-reference absorption levels with current price
    if of_signal.absorption_active and of_signal.absorption_levels:
        nearest_abs = min(of_signal.absorption_levels, key=lambda p: abs(p - current_price))
        dist = abs(nearest_abs - current_price)
        if dist < pip * 10:
            if nearest_abs < current_price:
                buy_boost += 6.0
                notes.append(f"Absorption cluster near price: support @ {nearest_abs:.5f}")
            else:
                sell_boost += 6.0
                notes.append(f"Absorption cluster near price: resistance @ {nearest_abs:.5f}")

    # Stop-run confirmed near current price = strong signal
    if of_signal.stop_run_detected:
        sl_dist = abs(of_signal.stop_run_level - current_price) / max(pip, 0.0001)
        if sl_dist < pip * 20:
            if of_signal.stop_run_direction == "BUY":
                buy_boost += 8.0
                notes.append(f"Stop-run BUY confirmed @ {of_signal.stop_run_level:.5f}")
            else:
                sell_boost += 8.0
                notes.append(f"Stop-run SELL confirmed @ {of_signal.stop_run_level:.5f}")

    return round(buy_boost, 1), round(sell_boost, 1), notes


def _approx_contribution(signal: 'OrderFlowSignal', direction: str) -> Tuple[float, str]:
    """Approximate signal contribution without needing the engine instance."""
    score = 0.0
    parts = []

    if direction == "BUY":
        if signal.delta > 0.15:
            s = 8.0 * min(1.0, signal.delta * 3)
            score += s
            parts.append(f"Delta {signal.delta:+.0%}")
        if signal.imbalance_ratio > 0.20:
            s = 10.0 * min(1.0, signal.imbalance_ratio * 2)
            score += s
            parts.append(f"DOM bid={signal.imbalance_ratio:+.0%}")
        if signal.absorption_active:
            score += 12.0
            parts.append("Absorción")
        if signal.iceberg_detected and signal.iceberg_chunks >= 2:
            score += 10.0
            parts.append("Iceberg")
        if signal.stop_run_detected and signal.stop_run_direction == "BUY":
            score += 14.0
            parts.append("Stop-run shorts")
        if signal.delta_macd and signal.delta_macd.divergence_bullish:
            score += 12.0
            parts.append("Delta-MACD bull")
        if signal.exhaustion_active and signal.exhaustion_side == "SELL_CLIMAX":
            score += 10.0
            parts.append("Selling climax")
    else:
        if signal.delta < -0.15:
            s = 8.0 * min(1.0, abs(signal.delta) * 3)
            score += s
            parts.append(f"Delta {signal.delta:+.0%}")
        if signal.imbalance_ratio < -0.20:
            s = 10.0 * min(1.0, abs(signal.imbalance_ratio) * 2)
            score += s
            parts.append(f"DOM ask={signal.imbalance_ratio:+.0%}")
        if signal.absorption_active:
            score += 12.0
            parts.append("Absorción")
        if signal.iceberg_detected and signal.iceberg_chunks >= 2:
            score += 10.0
            parts.append("Iceberg")
        if signal.stop_run_detected and signal.stop_run_direction == "SELL":
            score += 14.0
            parts.append("Stop-run longs")
        if signal.delta_macd and signal.delta_macd.divergence_bearish:
            score += 12.0
            parts.append("Delta-MACD bear")
        if signal.exhaustion_active and signal.exhaustion_side == "BUY_CLIMAX":
            score += 10.0
            parts.append("Buying climax")

    return score, " | ".join(parts[:5]) if parts else ""

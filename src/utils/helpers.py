"""Utility functions for trading strategy"""
from __future__ import annotations
from datetime import datetime
from typing import Tuple, List, Optional, TYPE_CHECKING
from enum import Enum
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from src.core.pattern_detector import Pattern


class Killzone(Enum):
    ASIAN = "ASIAN"
    LONDON_OPEN = "LONDON_OPEN"
    NY_OPEN = "NY_OPEN"
    LONDON_NY_OVERLAP = "LONDON_NY_OVERLAP"
    NONE = "NONE"


def detect_killzone(current_dt: datetime) -> Killzone:
    hour = current_dt.hour
    if 8 <= hour < 9:
        return Killzone.LONDON_OPEN
    if 13 <= hour < 17:
        if 13 <= hour < 14:
            return Killzone.NY_OPEN
        return Killzone.LONDON_NY_OVERLAP
    if 0 <= hour < 8:
        return Killzone.ASIAN
    return Killzone.NONE


def pip_size(symbol: str) -> float:
    s = symbol.upper()
    if "JPY" in s:
        return 0.01
    if "XAU" in s or s == "GOLD":
        return 0.1
    if "XAG" in s or s == "SILVER":
        return 0.01
    if s in ("NAS100", "US100", "NDX", "DJI30", "US30", "SPX500", "USTEC", "USTEC_x100"):
        return 1.0
    return 0.0001


def pips_to_price(symbol: str, pips: float) -> float:
    return pips * pip_size(symbol)


def price_to_pips(symbol: str, price_diff: float) -> float:
    ps = pip_size(symbol)
    return price_diff / ps if ps > 0 else 0.0


def is_in_session(current_dt: datetime, sessions: List[Tuple[int, int]]) -> bool:
    if not sessions:
        return True
    hour = current_dt.hour
    for start, end in sessions:
        if start <= hour < end:
            return True
    return False


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def classify_retracement(df: pd.DataFrame, trend_direction: str, max_pause_bars: int = 4) -> str:
    """Clasifica si el movimiento actual es Pausa (≤4 velas en contra) o Retroceso (≥5).
    Clase 12: ≤4 velas contrarias = pausa y continuación del impulso.
    ≥5 velas = retroceso estructural (esperar reacumulación).
    Returns: "pause", "retracement", or "unknown"."""
    if df is None or len(df) < max_pause_bars + 2:
        return "unknown"
    last_n = df.tail(max_pause_bars + 5)
    if trend_direction.upper() == "BUY":
        contrary = (last_n["close"] < last_n["open"]).sum()
    elif trend_direction.upper() == "SELL":
        contrary = (last_n["close"] > last_n["open"]).sum()
    else:
        return "unknown"
    if contrary <= max_pause_bars:
        return "pause"
    return "retracement"


def classify_order_flow(df: pd.DataFrame, lookback: int = 20) -> str:
    """Clasifica Order Flow como Regular (estructura fractal limpia) o Irregular (por zonas).
    Clase 6, 12: Regular = máximos/mínimos crecientes/decrecientes consistentes.
    Irregular = precio vuelve al mismo rango repetidamente.
    Returns: "regular", "irregular", or "unknown"."""
    if df is None or len(df) < lookback + 5:
        return "unknown"
    segment = df.iloc[-lookback:]
    highs = segment["high"].values
    lows = segment["low"].values
    zone_visits = 0
    for i in range(2, len(segment) - 2):
        ref_high = highs[i]
        ref_low = lows[i]
        range_size = ref_high - ref_low
        if range_size <= 0:
            continue
        for j in range(i + 1, min(i + 6, len(segment))):
            if abs(highs[j] - ref_high) <= range_size * 0.5 and abs(lows[j] - ref_low) <= range_size * 0.5:
                zone_visits += 1
                break
    if zone_visits >= 3:
        return "irregular"
    # Check clean fractal: consistent HH/HL or LH/LL
    rising = sum(1 for i in range(1, len(segment)) if highs[i] > highs[i - 1] and lows[i] > lows[i - 1])
    falling = sum(1 for i in range(1, len(segment)) if highs[i] < highs[i - 1] and lows[i] < lows[i - 1])
    total = len(segment) - 1
    if rising / total >= 0.6 or falling / total >= 0.6:
        return "regular"
    return "irregular"


def breaker_touch_count(pattern: Pattern, df: pd.DataFrame) -> int:
    """Cuenta cuántas veces el precio ha interactuado con un nivel de Breaker
    después de su formación. Clase 10: máximo 3 toques válidos."""
    if pattern is None or df is None:
        return 0
    level = pattern.mid
    after = df.iloc[pattern.index + 1:]
    if after.empty:
        return 0
    touches = 0
    for i in range(len(after)):
        row = after.iloc[i]
        if row["low"] <= level <= row["high"]:
            touches += 1
    return touches


def find_swing_points(df: pd.DataFrame, lookback: int = 5) -> Tuple[List[int], List[int]]:
    highs, lows = [], []
    n = len(df)
    for i in range(lookback, n - lookback):
        win_high = df["high"].iloc[i - lookback:i + lookback + 1]
        win_low = df["low"].iloc[i - lookback:i + lookback + 1]
        if df["high"].iloc[i] == win_high.max():
            highs.append(i)
        if df["low"].iloc[i] == win_low.min():
            lows.append(i)
    return highs, lows


def round_to_tick(price: float, tick_size: float) -> float:
    if tick_size <= 0:
        return price
    return round(price / tick_size) * tick_size


def candle_body_size(open_, close) -> float:
    return abs(close - open_)


def candle_body_pct(open_, high, low, close) -> float:
    rng = high - low
    if rng == 0:
        return 0.0
    return abs(close - open_) / rng


def candle_is_body_dominant(open_, high, low, close) -> bool:
    body = abs(close - open_)
    wick = (high - low) - body
    return body > wick

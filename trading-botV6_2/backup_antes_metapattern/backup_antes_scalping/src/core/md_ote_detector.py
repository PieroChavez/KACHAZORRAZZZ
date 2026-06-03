"""Fibonacci OTE 75-79% Detector — Optimal Trade Entry en descuento profundo
Después de un impulso direccional fuerte, el 50% de retroceso es una trampa.
La entrada real está en el descuento profundo: 75-79% del movimiento total.
Requiere confluencia estructural y agotamiento en el retroceso.

Reglas:
  - Ignorar 50% (reacción temporal, no continuación segura)
  - Entrar en 75% o 79% (Fibonacci OTE)
  - Exige confluencia con estructura previa

Referencia: MD Classes 18, 22
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from src.core.md_concepts import MDConcept, MDDetection

logger = logging.getLogger(__name__)

MIN_IMPULSE_CANDLES = 3
MAX_RETRACE_CANDLES_RATIO = 0.40
MIN_IMPULSE_BODY_RATIO = 0.55
MIN_IMPULSE_VOLUME = 1.25

OTE_LEVELS = [0.75, 0.79]
OTE_TOLERANCE = 0.03


@dataclass
class OTEResult:
    found: bool
    direction: str
    confidence: float
    entry_price: Optional[float] = None
    impulse_start: Optional[int] = None
    impulse_end: Optional[int] = None
    impulse_range: float = 0.0
    ote_level: float = 0.0
    retrace_index: Optional[int] = None
    avg_body_ratio: float = 0.0
    avg_volume_ratio: float = 1.0
    has_confluence: bool = False
    notes: List[str] = field(default_factory=list)

    def to_detection(self) -> Optional[MDDetection]:
        if not self.found or self.entry_price is None:
            return None
        return MDDetection(
            concept=MDConcept.OTE_75_79,
            direction=self.direction,
            confidence=self.confidence,
            suggested_price=self.entry_price,
            timeframe="M15",
            metadata={
                "impulse_range": self.impulse_range,
                "ote_level": self.ote_level,
                "has_confluence": self.has_confluence,
            },
        )


class OTEDetector:
    """Detecta entradas en descuento profundo Fibonacci (OTE 75-79%).

    Pipeline:
      1. Identificar impulso direccional (3+ velas seguidas, cuerpos fuertes)
      2. Calcular rango del impulso (origen → extremo)
      3. Calcular niveles 75% y 79% de retroceso
      4. Verificar si el precio está retrocediendo en esa zona
      5. Buscar confluencia estructural (FVG previo, breaker, etc.)
    """

    def __init__(self, lookback: int = 60,
                 min_impulse_candles: int = MIN_IMPULSE_CANDLES,
                 max_retrace_candle_ratio: float = MAX_RETRACE_CANDLES_RATIO,
                 ote_levels: Optional[List[float]] = None,
                 ote_tolerance: float = OTE_TOLERANCE):
        self._lookback = lookback
        self._min_impulse = min_impulse_candles
        self._max_retrace_ratio = max_retrace_candle_ratio
        self._ote_levels = ote_levels or OTE_LEVELS
        self._ote_tol = ote_tolerance

    def detect(self, df: pd.DataFrame) -> OTEResult:
        if df is None or len(df) < self._lookback:
            return OTEResult(found=False, direction="NEUTRAL", confidence=0.0)

        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        n = len(df)

        atr_val = self._calc_atr(df)
        if atr_val == 0:
            return OTEResult(found=False, direction="NEUTRAL", confidence=0.0)

        for start in range(max(2, n - self._lookback), n - self._min_impulse - 2):
            direction = "BUY" if closes[start] > opens[start] else "SELL"
            body_run = 0
            total_body_ratio = 0.0
            total_vol_ratio = 0.0
            impulse_high = highs[start]
            impulse_low = lows[start]

            for k in range(self._min_impulse):
                idx = start + k
                if idx >= n - 1:
                    break
                same = (direction == "BUY" and closes[idx] > opens[idx]) or \
                       (direction == "SELL" and closes[idx] < opens[idx])
                if not same:
                    break
                body_run += 1
                c_range = highs[idx] - lows[idx]
                if c_range > 0:
                    total_body_ratio += abs(closes[idx] - opens[idx]) / c_range
                vol = self._volume_ratio(df, idx)
                total_vol_ratio += vol
                impulse_high = max(impulse_high, highs[idx])
                impulse_low = min(impulse_low, lows[idx])

            if body_run < self._min_impulse:
                continue

            avg_body = total_body_ratio / body_run
            avg_vol = total_vol_ratio / body_run
            if avg_body < MIN_IMPULSE_BODY_RATIO:
                continue
            if avg_vol < MIN_IMPULSE_VOLUME:
                continue

            impulse_start_idx = start
            impulse_end_idx = start + body_run - 1
            impulse_range = impulse_high - impulse_low
            if impulse_range < atr_val * 0.3:
                continue

            origin = impulse_low if direction == "BUY" else impulse_high
            extreme = impulse_high if direction == "BUY" else impulse_low

            current_price = closes[-1]
            total_move = abs(extreme - origin)
            if total_move == 0:
                continue

            retrace = abs(current_price - extreme)
            retrace_ratio = retrace / total_move

            hit_level = None
            for level in sorted(self._ote_levels):
                low_bound = level - self._ote_tol
                high_bound = level + self._ote_tol
                if low_bound <= retrace_ratio <= high_bound:
                    hit_level = level
                    break

            if hit_level is None:
                continue

            retrace_high = 0.0
            retrace_low = 0.0
            retrace_index = None
            extreme_idx = impulse_end_idx + 1
            for t in range(impulse_end_idx + 1, n):
                if highs[t] > retrace_high:
                    retrace_high = highs[t]
                    extreme_idx = t
                retrace_low = min(retrace_low if retrace_low else lows[t], lows[t])
                if direction == "SELL" and lows[t] == retrace_low:
                    extreme_idx = t
                retrace_index = t

            if retrace_index is None or extreme_idx is None:
                continue

            if direction == "BUY":
                ote_price = origin + total_move * hit_level
                entry_price = ote_price
            else:
                ote_price = origin - total_move * hit_level
                entry_price = ote_price

            confluence = self._check_confluence(df, impulse_end_idx, direction)

            conf = 0.30
            conf += min(0.20, avg_body * 0.25)
            vol_conf = min(0.15, (avg_vol - 1.0) * 0.20)
            conf += vol_conf
            if confluence:
                conf += 0.10
            conf = min(0.85, conf)

            notes = [
                f"OTE {direction}: impulso [{impulse_start_idx}-{impulse_end_idx}]",
                f"Retroceso {retrace_ratio:.0%} → nivel {hit_level:.0%}",
                f"Entry={entry_price:.5f} conf={conf:.0%}",
            ]

            return OTEResult(
                found=True,
                direction=direction,
                confidence=conf,
                entry_price=entry_price,
                impulse_start=impulse_start_idx,
                impulse_end=impulse_end_idx,
                impulse_range=impulse_range,
                ote_level=hit_level,
                retrace_index=retrace_index,
                avg_body_ratio=avg_body,
                avg_volume_ratio=avg_vol,
                has_confluence=confluence,
                notes=notes,
            )

        return OTEResult(found=False, direction="NEUTRAL", confidence=0.0)

    def _check_confluence(self, df: pd.DataFrame, end_idx: int,
                          direction: str) -> bool:
        """Verifica confluencia estructural: FVG previo en la zona OTE."""
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        n = len(df)

        for i in range(max(2, end_idx - 15), end_idx - 1):
            c1_range = highs[i] - lows[i]
            if c1_range == 0:
                continue
            body1 = abs(closes[i] - opens[i])
            body1_ratio = body1 / c1_range
            if body1_ratio < 0.4:
                continue
            if direction == "BUY" and closes[i] < opens[i]:
                if i + 1 < n and highs[i] < lows[i + 1]:
                    return True
            elif direction == "SELL" and closes[i] > opens[i]:
                if i + 1 < n and lows[i] > highs[i + 1]:
                    return True

        return False

    def _volume_ratio(self, df: pd.DataFrame, idx: int) -> float:
        if "tick_volume" not in df.columns and "volume" not in df.columns:
            return 1.0
        vol_col = "tick_volume" if "tick_volume" in df.columns else "volume"
        volumes = df[vol_col].values
        if idx < 5 or idx >= len(volumes):
            return 1.0
        recent = volumes[max(0, idx - 5):idx]
        avg_vol = float(np.mean(recent))
        if avg_vol == 0:
            return 1.0
        return min(3.0, float(volumes[idx]) / avg_vol)

    def _calc_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < period + 1:
            return 0.0
        high, low, close = df["high"].values, df["low"].values, df["close"].values
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]),
                                   np.abs(low[1:] - close[:-1])))
        return float(np.mean(tr[-period:]))

"""Interval Detector — Punto Interactivo (Interval)
Busca velas con CUERPO mínimo y mechas largas en AMBOS extremos.
El Intervalo es una pausa institucional: atrapa compradores y vendedores.
El mercado sale con violencia (dejando FVG) y luego retorna al 50%.
Entrada en el 50% del rango total de la vela.

Regla anatómica estricta: body < upper_wick AND body < lower_wick.
Si el cuerpo > cualquiera de las mechas → es TRAMPA, no intervalo.
Requiere volumen en la misma vela o la siguiente.

Referencia: MD Classes 23, 24, 26, 27
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.md_concepts import MDConcept, MDDetection

logger = logging.getLogger(__name__)

MIN_BODY_RATIO_ANATOMY = 0.35
MIN_VOLUME_SPIKE = 1.3
MAX_INTERVAL_RANGE_ATR = 1.5


@dataclass
class IntervalResult:
    found: bool
    direction: str
    confidence: float
    interval_index: Optional[int] = None
    entry_price: Optional[float] = None
    interval_high: Optional[float] = None
    interval_low: Optional[float] = None
    body_size: float = 0.0
    upper_wick: float = 0.0
    lower_wick: float = 0.0
    body_ratio: float = 0.0
    volume_ratio: float = 1.0
    fvg_detected: bool = False
    notes: List[str] = field(default_factory=list)

    def to_detection(self) -> Optional[MDDetection]:
        if not self.found or self.entry_price is None:
            return None
        return MDDetection(
            concept=MDConcept.INTERVAL,
            direction=self.direction,
            confidence=self.confidence,
            suggested_price=self.entry_price,
            timeframe="M15",
            metadata={
                "interval_high": self.interval_high,
                "interval_low": self.interval_low,
                "body_ratio": self.body_ratio,
                "volume_ratio": self.volume_ratio,
                "fvg_detected": self.fvg_detected,
            },
        )


class IntervalDetector:
    """Detecta Intervalos (Interactive Points) en velas.

    Pipeline:
      1. Buscar vela con body < ambas mechas (individualmente)
      2. Verificar volumen en vela actual o siguiente
      3. Verificar que la siguiente vela se expanda en dirección opuesta
      4. Detectar FVG dejado por la expansión
      5. Marcar 50% del rango como entrada
    """

    def __init__(self, lookback: int = 60,
                 max_body_to_total: float = MIN_BODY_RATIO_ANATOMY,
                 min_volume_ratio: float = MIN_VOLUME_SPIKE):
        self._lookback = lookback
        self._max_body_ratio = max_body_to_total
        self._min_volume = min_volume_ratio

    def detect(self, df: pd.DataFrame) -> IntervalResult:
        if df is None or len(df) < self._lookback:
            return IntervalResult(found=False, direction="NEUTRAL", confidence=0.0)

        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        n = len(df)

        atr_val = self._calc_atr(df)
        if atr_val == 0:
            return IntervalResult(found=False, direction="NEUTRAL", confidence=0.0)

        for i in range(max(3, n - self._lookback), n - 1):
            body = abs(closes[i] - opens[i])
            candle_range = highs[i] - lows[i]
            if candle_range == 0:
                continue

            upper_wick = highs[i] - max(opens[i], closes[i])
            lower_wick = min(opens[i], closes[i]) - lows[i]

            # MD Rule: body must be smaller than BOTH wicks individually
            if upper_wick <= 0 or lower_wick <= 0:
                continue
            if body / upper_wick > 1.0 or body / lower_wick > 1.0:
                continue

            body_to_total = body / candle_range
            if body_to_total > self._max_body_ratio:
                continue

            # Check range not too large
            if candle_range > MAX_INTERVAL_RANGE_ATR * atr_val:
                continue

            # Volume validation: current candle or next must have volume
            vol_ratio = self._volume_ratio(df, i)
            next_vol_ratio = self._volume_ratio(df, i + 1) if i + 1 < n else 1.0
            max_vol = max(vol_ratio, next_vol_ratio)

            if max_vol < self._min_volume:
                continue

            # Determine direction: look at next candle breakout direction
            next_bullish = closes[i + 1] > opens[i + 1] if i + 1 < n else False
            next_bearish = closes[i + 1] < opens[i + 1] if i + 1 < n else False

            if next_bullish and next_bearish:
                continue

            direction = "BUY" if next_bullish else "SELL"
            entry = (highs[i] + lows[i]) / 2.0

            fvg = False
            if i + 2 < n:
                if direction == "BUY":
                    fvg = highs[i] < lows[i + 1]
                else:
                    fvg = lows[i] > highs[i + 1]

            conf = 0.4
            conf += max(0, 0.3 * (1.0 - body_to_total))
            conf += min(0.15, (max_vol - 1.0) * 0.1)
            if fvg:
                conf += 0.1
            conf = min(0.90, conf)

            notes = [
                f"Interval en [{i}] body={body:.2f} up_wick={upper_wick:.2f} low_wick={lower_wick:.2f}",
                f"Body/total={body_to_total:.0%} vol_ratio={max_vol:.1f}",
                f"Dirección: {direction} FVG={fvg}",
            ]

            return IntervalResult(
                found=True,
                direction=direction,
                confidence=conf,
                interval_index=i,
                entry_price=entry,
                interval_high=highs[i],
                interval_low=lows[i],
                body_size=body,
                upper_wick=upper_wick,
                lower_wick=lower_wick,
                body_ratio=body_to_total,
                volume_ratio=max_vol,
                fvg_detected=fvg,
                notes=notes,
            )

        return IntervalResult(found=False, direction="NEUTRAL", confidence=0.0)

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

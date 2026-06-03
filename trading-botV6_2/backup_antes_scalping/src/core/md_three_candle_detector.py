"""ThreeCandle Detector — Conteo de 3 velas consecutivas
Tras una consolidación lateral, el mercado rompe con 3 velas del mismo color
sin retrocesos profundos. Regla estricta:
  - Ignorar vela 1 (origen del breakout)
  - Ignorar vela 3 (aceleración final)
  - ENTRAR al 50% de la vela 2 (vela intermedia)
  - Exige confluencia estructural y volumen

Solo para activos volátiles: XAUUSD, BTCUSD, US100, XAGUSD.

Referencia: MD Class 22 — Modelo 2 (Conteo por Tres Velas)
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from src.core.md_concepts import MDConcept, MDDetection

logger = logging.getLogger(__name__)

MIN_BODY_RATIO = 0.45
MAX_RETRACE_BODY_RATIO = 0.35
MIN_VOLUME_SPIKE = 1.25
CONSOLIDATION_LOOKBACK = 8
MAX_CONSOLIDATION_RANGE_ATR_RATIO = 0.6


@dataclass
class ThreeCandleResult:
    found: bool
    direction: str
    confidence: float
    entry_price: Optional[float] = None
    candle1_index: Optional[int] = None
    candle2_index: Optional[int] = None
    candle3_index: Optional[int] = None
    candle2_high: Optional[float] = None
    candle2_low: Optional[float] = None
    consolidation_range: float = 0.0
    avg_body_ratio: float = 0.0
    avg_volume_ratio: float = 1.0
    notes: List[str] = field(default_factory=list)

    def to_detection(self) -> Optional[MDDetection]:
        if not self.found or self.entry_price is None:
            return None
        return MDDetection(
            concept=MDConcept.THREE_CANDLE,
            direction=self.direction,
            confidence=self.confidence,
            suggested_price=self.entry_price,
            timeframe="M15",
            metadata={
                "candle1": self.candle1_index,
                "candle2": self.candle2_index,
                "candle3": self.candle3_index,
                "consolidation_range": self.consolidation_range,
            },
        )


class ThreeCandleDetector:
    """Detecta el patrón Conteo de 3 Velas.

    Pipeline:
      1. Buscar zona de consolidación lateral (rango pequeño, velas sin dirección)
      2. Detectar 3 velas consecutivas del mismo color rompiendo la zona
      3. Validar: cuerpos fuertes, sin retrocesos profundos entre ellas
      4. Aislar vela 2, marcar entry al 50% de su rango
      5. Verificar volumen en las 3 velas vs media
    """

    def __init__(self, lookback: int = 60,
                 min_body_ratio: float = MIN_BODY_RATIO,
                 max_retrace_body_ratio: float = MAX_RETRACE_BODY_RATIO,
                 min_volume: float = MIN_VOLUME_SPIKE):
        self._lookback = lookback
        self._min_body = min_body_ratio
        self._max_retrace = max_retrace_body_ratio
        self._min_vol = min_volume

    def detect(self, df: pd.DataFrame) -> ThreeCandleResult:
        if df is None or len(df) < self._lookback:
            return ThreeCandleResult(found=False, direction="NEUTRAL", confidence=0.0)

        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        n = len(df)

        atr_val = self._calc_atr(df)
        if atr_val == 0:
            return ThreeCandleResult(found=False, direction="NEUTRAL", confidence=0.0)

        for start in range(max(CONSOLIDATION_LOOKBACK + 3, n - self._lookback), n - 4):

            consolidation_range = (float(np.max(highs[start - CONSOLIDATION_LOOKBACK:start])) -
                                   float(np.min(lows[start - CONSOLIDATION_LOOKBACK:start])))
            if consolidation_range > MAX_CONSOLIDATION_RANGE_ATR_RATIO * atr_val:
                continue

            if start + 2 >= n:
                continue

            candle1_bull = closes[start] > opens[start]
            candle2_bull = closes[start + 1] > opens[start + 1]
            candle3_bull = closes[start + 2] > opens[start + 2]
            candle1_bear = closes[start] < opens[start]
            candle2_bear = closes[start + 1] < opens[start + 1]
            candle3_bear = closes[start + 2] < opens[start + 2]

            if candle1_bull and candle2_bull and candle3_bull:
                direction = "BUY"
            elif candle1_bear and candle2_bear and candle3_bear:
                direction = "SELL"
            else:
                continue

            for offset in range(3):
                idx = start + offset
                if idx >= n:
                    break
                body = abs(closes[idx] - opens[idx])
                c_range = highs[idx] - lows[idx]
                if c_range == 0:
                    break
                if body / c_range < self._min_body:
                    offset = -1
                    break
            if offset == -1:
                continue

            body1 = abs(closes[start] - opens[start])
            if direction == "BUY":
                retrace_12 = closes[start] - opens[start + 1]
                if retrace_12 > 0 and retrace_12 / body1 > self._max_retrace:
                    continue
            else:
                retrace_12 = opens[start + 1] - closes[start]
                if retrace_12 > 0 and retrace_12 / body1 > self._max_retrace:
                    continue

            c2_body = abs(closes[start + 1] - opens[start + 1])
            c2_range = highs[start + 1] - lows[start + 1]
            c2_body_ratio = c2_body / c2_range if c2_range > 0 else 0

            c3_body = abs(closes[start + 2] - opens[start + 2])
            c3_range = highs[start + 2] - lows[start + 2]
            c3_body_ratio = c3_body / c3_range if c3_range > 0 else 0

            if c2_range == 0:
                continue

            entry_price = (highs[start + 1] + lows[start + 1]) / 2.0

            vol_ratios = []
            for offset in range(3):
                idx = start + offset
                if idx < n:
                    vol_ratios.append(self._volume_ratio(df, idx))
            avg_vol = float(np.mean(vol_ratios)) if vol_ratios else 1.0
            if avg_vol < self._min_vol:
                continue

            avg_body = (c2_body_ratio + c3_body_ratio) / 2.0

            conf = 0.35
            conf += min(0.15, avg_body * 0.15)
            conf += min(0.15, (avg_vol - 1.0) * 0.15)
            conf += min(0.10, consolidation_range / atr_val * 0.20)
            conf = min(0.85, conf)

            notes = [
                f"3Candle {direction}: [{start}-{start+2}]",
                f"Consolidación={consolidation_range:.1f} body_ratio={avg_body:.0%}",
                f"Entry al 50% vela 2 @{entry_price:.5f}",
            ]

            return ThreeCandleResult(
                found=True,
                direction=direction,
                confidence=conf,
                entry_price=entry_price,
                candle1_index=start,
                candle2_index=start + 1,
                candle3_index=start + 2,
                candle2_high=highs[start + 1],
                candle2_low=lows[start + 1],
                consolidation_range=consolidation_range,
                avg_body_ratio=avg_body,
                avg_volume_ratio=avg_vol,
                notes=notes,
            )

        return ThreeCandleResult(found=False, direction="NEUTRAL", confidence=0.0)

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

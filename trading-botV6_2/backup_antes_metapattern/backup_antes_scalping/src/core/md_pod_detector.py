"""POD Detector — Traditional Block + Internal Debt (POD)
Busca la secuencia OB → Bloque Tradicional (cobertura con cuerpo) → POD gap.
El POD (Punto de Presión de Oferta/Demanda) es el espacio no negociado
entre el cierre del OB y la apertura del Bloque Tradicional.
El algoritmo DEBE retroceder a pagar esa deuda antes de continuar.

Referencia: MD Classes 19, 21, 22, 25, 26
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.md_concepts import MDConcept, MDDetection

logger = logging.getLogger(__name__)

MIN_BODY_RATIO = 0.4
MAX_COVER_CANDLES = 2


@dataclass
class PODResult:
    found: bool
    direction: str
    confidence: float
    pod_50_price: Optional[float] = None
    pod_low: Optional[float] = None
    pod_high: Optional[float] = None
    ob_index: Optional[int] = None
    tb_index: Optional[int] = None
    ob_body_low: Optional[float] = None
    ob_body_high: Optional[float] = None
    tb_body_low: Optional[float] = None
    tb_body_high: Optional[float] = None
    cover_ratio: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_detection(self) -> Optional[MDDetection]:
        if not self.found or self.pod_50_price is None:
            return None
        return MDDetection(
            concept=MDConcept.POD,
            direction=self.direction,
            confidence=self.confidence,
            suggested_price=self.pod_50_price,
            timeframe="M15",
            metadata={
                "pod_low": self.pod_low,
                "pod_high": self.pod_high,
                "cover_ratio": self.cover_ratio,
            },
        )


class PODDetector:
    """Detecta secuencias Order Block → Traditional Block → POD.

    Pipeline por cada OB:
      1. Localizar OB (última vela contraria antes del impulso)
      2. Verificar las siguientes 1-2 velas
      3. ¿Alguna cubre el rango del OB con CUERPO (no mecha)?
      4. Si sí → Bloque Tradicional validado
      5. POD = gap entre OB.close y TB.open
      6. Entrada al 50% del POD
    """

    def __init__(self, lookback: int = 60,
                 min_cover_ratio: float = 0.6,
                 max_pod_height_atr: float = 2.0):
        self._lookback = lookback
        self._min_cover = min_cover_ratio
        self._max_pod_atr = max_pod_height_atr

    def detect(self, df: pd.DataFrame) -> PODResult:
        if df is None or len(df) < self._lookback:
            return PODResult(found=False, direction="NEUTRAL", confidence=0.0)

        closes = df["close"].values
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        n = len(df)

        atr_val = self._calc_atr(df)
        if atr_val == 0:
            return PODResult(found=False, direction="NEUTRAL", confidence=0.0)

        for i in range(max(3, n - self._lookback), n - MAX_COVER_CANDLES):
            ob_body = abs(closes[i] - opens[i])
            ob_range = highs[i] - lows[i]
            if ob_range == 0:
                continue
            ob_body_ratio = ob_body / ob_range
            if ob_body_ratio < MIN_BODY_RATIO:
                continue

            ob_direction = "BUY" if closes[i] > opens[i] else "SELL"

            for offset in range(1, MAX_COVER_CANDLES + 1):
                j = i + offset
                if j >= n:
                    continue

                tb_body = abs(closes[j] - opens[j])
                tb_range = highs[j] - lows[j]
                if tb_range == 0:
                    continue
                tb_body_ratio = tb_body / tb_range
                if tb_body_ratio < MIN_BODY_RATIO:
                    continue

                tb_direction = "BUY" if closes[j] > opens[j] else "SELL"
                if tb_direction == ob_direction:
                    continue

                ob_body_low = min(opens[i], closes[i])
                ob_body_high = max(opens[i], closes[i])
                tb_body_low = min(opens[j], closes[j])
                tb_body_high = max(opens[j], closes[j])

                cover = min(ob_body_high, tb_body_high) - max(ob_body_low, tb_body_low)
                ob_body_size = ob_body_high - ob_body_low
                if ob_body_size == 0:
                    continue
                cover_ratio = cover / ob_body_size
                if cover_ratio < self._min_cover:
                    continue

                if tb_direction == "BUY":
                    pod_low = closes[i]
                    pod_high = opens[j]
                else:
                    pod_low = opens[j]
                    pod_high = closes[i]

                if pod_low is None or pod_high is None:
                    continue

                if pod_high <= pod_low:
                    continue

                pod_height = abs(pod_high - pod_low)
                if pod_height > self._max_pod_atr * atr_val:
                    continue

                pod_50 = (pod_low + pod_high) / 2.0

                rvol = self._relative_volume(df, j)
                conf = min(0.85, 0.4 + cover_ratio * 0.3 + rvol * 0.15)

                notes = [
                    f"OB[{i}]→TB[{j}] ({tb_direction})",
                    f"Cover={cover_ratio:.0%} rVol={rvol:.1f}",
                    f"POD gap: {pod_low:.5f}-{pod_high:.5f} entry={pod_50:.5f}",
                ]

                return PODResult(
                    found=True,
                    direction=tb_direction,
                    confidence=conf,
                    pod_50_price=pod_50,
                    pod_low=pod_low,
                    pod_high=pod_high,
                    ob_index=i,
                    tb_index=j,
                    ob_body_low=ob_body_low,
                    ob_body_high=ob_body_high,
                    tb_body_low=tb_body_low,
                    tb_body_high=tb_body_high,
                    cover_ratio=cover_ratio,
                    notes=notes,
                )

        return PODResult(found=False, direction="NEUTRAL", confidence=0.0)

    def _relative_volume(self, df: pd.DataFrame, idx: int) -> float:
        if "tick_volume" not in df.columns and "volume" not in df.columns:
            return 1.0
        vol_col = "tick_volume" if "tick_volume" in df.columns else "volume"
        volumes = df[vol_col].values
        if idx < 5:
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

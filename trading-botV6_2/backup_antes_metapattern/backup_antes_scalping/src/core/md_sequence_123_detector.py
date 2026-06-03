"""Sequence 1-2-3 Detector — Conteo estructural de 3 toques
Después de respetar una zona (Intervalo/LimitPrice), el mercado ejecuta:
  Punto 1: Ruptura local a favor de dirección esperada
  Punto 2: Retroceso que forma pivot de inducción (carnada de liquidez)
  Punto 3: Retesteo final sobre micro-vacío apoyado en la zona mayor

Entry = rechazo del Punto 3.

Referencia: MD Classes 15, 24
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.md_concepts import MDConcept, MDDetection

logger = logging.getLogger(__name__)

LOOKBACK_WINDOW = 30
MIN_POINT1_BODY_RATIO = 0.5
MIN_POINT2_RETRACE_RATIO = 0.3
MAX_POINT3_CANDLES = 6


@dataclass
class Sequence123Result:
    found: bool
    direction: str
    confidence: float
    entry_price: Optional[float] = None
    point1_index: Optional[int] = None
    point2_index: Optional[int] = None
    point3_index: Optional[int] = None
    zone_high: Optional[float] = None
    zone_low: Optional[float] = None
    point2_low: Optional[float] = None
    point2_high: Optional[float] = None
    retrace_ratio: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_detection(self) -> Optional[MDDetection]:
        if not self.found or self.entry_price is None:
            return None
        return MDDetection(
            concept=MDConcept.SEQUENCE_123,
            direction=self.direction,
            confidence=self.confidence,
            suggested_price=self.entry_price,
            timeframe="M15",
            metadata={
                "point1": self.point1_index,
                "point2": self.point2_index,
                "point3": self.point3_index,
                "retrace_ratio": self.retrace_ratio,
            },
        )


class Sequence123Detector:
    """Detecta el patrón 1-2-3 de conteo estructural.

    Pipeline:
      1. Identificar una zona de referencia donde el precio hizo pausa/respeto
      2. Punto 1: Vela que rompe limpiamente de la zona
      3. Punto 2: Retroceso que forma un nuevo extremo (inducement)
      4. Punto 3: Retorno a la zona, dejando un mechazo de rechazo
      5. Entry en el rechazo del Punto 3
    """

    def __init__(self, lookback: int = LOOKBACK_WINDOW,
                 min_point1_body: float = MIN_POINT1_BODY_RATIO,
                 min_retrace_ratio: float = MIN_POINT2_RETRACE_RATIO):
        self._lookback = lookback
        self._min_p1_body = min_point1_body
        self._min_retrace = min_retrace_ratio

    def detect(self, df: pd.DataFrame) -> Sequence123Result:
        if df is None or len(df) < self._lookback:
            return Sequence123Result(found=False, direction="NEUTRAL", confidence=0.0)

        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        n = len(df)

        for zone_end in range(max(5, n - self._lookback), n - 4):

            zone_high = float(np.max(highs[max(0, zone_end - 3):zone_end + 1]))
            zone_low = float(np.min(lows[max(0, zone_end - 3):zone_end + 1]))
            zone_avg = (zone_high + zone_low) / 2.0
            zone_range = zone_high - zone_low
            if zone_range == 0:
                continue

            # Punto 1: breakout de la zona
            p1 = zone_end + 1
            if p1 >= n - 3:
                continue
            p1_body = abs(closes[p1] - opens[p1])
            p1_range = highs[p1] - lows[p1]
            if p1_range == 0:
                continue
            if p1_body / p1_range < self._min_p1_body:
                continue

            if closes[p1] > opens[p1] and closes[p1] > zone_high:
                direction = "BUY"
            elif closes[p1] < opens[p1] and closes[p1] < zone_low:
                direction = "SELL"
            else:
                continue

            # Punto 2: retroceso que forma un nuevo extremo (inducement)
            p2_found = False
            p2_idx = None
            for k in range(p1 + 1, min(n - 2, p1 + 8)):
                if direction == "BUY":
                    if closes[k] < opens[k] and lows[k] < zone_avg:
                        p2_found = True
                        p2_idx = k
                        break
                else:
                    if closes[k] > opens[k] and highs[k] > zone_avg:
                        p2_found = True
                        p2_idx = k
                        break

            if not p2_found or p2_idx is None:
                continue

            p2_low = float(np.min(lows[p1:p2_idx + 1]))
            p2_high = float(np.max(highs[p1:p2_idx + 1]))

            p1_extreme = highs[p1] if direction == "BUY" else lows[p1]
            retrace_dist = abs(p2_high - p2_low)
            if retrace_dist == 0:
                continue
            p1_move = abs(p1_extreme - (zone_high if direction == "BUY" else zone_low))
            if p1_move == 0:
                continue
            retrace_ratio = retrace_dist / p1_move
            if retrace_ratio < self._min_retrace:
                continue

            # Punto 3: retorno a la zona con mechazo de rechazo
            p3_found = False
            p3_idx = None
            for k in range(p2_idx + 1, min(n, p2_idx + MAX_POINT3_CANDLES + 1)):
                if direction == "BUY":
                    if lows[k] <= zone_high and closes[k] > opens[k]:
                        if k + 1 < n and closes[k + 1] > closes[k]:
                            p3_found = True
                            p3_idx = k
                            break
                else:
                    if highs[k] >= zone_low and closes[k] < opens[k]:
                        if k + 1 < n and closes[k + 1] < closes[k]:
                            p3_found = True
                            p3_idx = k
                            break

            if not p3_found or p3_idx is None:
                continue

            entry_price = lows[p3_idx] if direction == "BUY" else highs[p3_idx]

            conf = 0.35
            point1_vol = self._volume_ratio(df, p1)
            if point1_vol > 1.0:
                conf += min(0.15, (point1_vol - 1.0) * 0.15)
            conf += min(0.10, retrace_ratio * 0.15)

            p3_wick = 0.0
            if direction == "BUY":
                p3_wick = min(opens[p3_idx], closes[p3_idx]) - lows[p3_idx]
            else:
                p3_wick = highs[p3_idx] - max(opens[p3_idx], closes[p3_idx])
            if p3_wick > zone_range * 0.3:
                conf += 0.10

            conf = min(0.85, conf)

            notes = [
                f"1-2-3 {direction}: P1[{p1}] P2[{p2_idx}] P3[{p3_idx}]",
                f"Zona: {zone_low:.5f}-{zone_high:.5f} retrace={retrace_ratio:.0%}",
                f"Entry={entry_price:.5f} conf={conf:.0%}",
            ]

            return Sequence123Result(
                found=True,
                direction=direction,
                confidence=conf,
                entry_price=entry_price,
                point1_index=p1,
                point2_index=p2_idx,
                point3_index=p3_idx,
                zone_high=zone_high,
                zone_low=zone_low,
                point2_low=p2_low,
                point2_high=p2_high,
                retrace_ratio=retrace_ratio,
                notes=notes,
            )

        return Sequence123Result(found=False, direction="NEUTRAL", confidence=0.0)

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

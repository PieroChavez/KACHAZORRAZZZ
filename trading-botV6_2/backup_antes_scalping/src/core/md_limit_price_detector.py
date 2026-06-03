"""LimitPrice / Equity Detector — Mecha extrema defendida por cuerpos siguientes
Una vela con mecha larga + cuerpo pequeño marca un "límite de precio".
Las velas posteriores deben cerrar sus cuerpos SIN violar ese límite.
Si el límite se defiende -> el precio debe retroceder desde ahí.

El SL se protege 6-10 pips detrás del límite.
Referencia: MD Classes 24, 26, 27
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from src.core.md_concepts import MDConcept, MDDetection

logger = logging.getLogger(__name__)

MAX_DEFEND_CANDLES = 3
MIN_WICK_BODY_RATIO = 1.5


@dataclass
class LimitPriceResult:
    found: bool
    direction: str
    confidence: float
    limit_price: Optional[float] = None
    wick_index: Optional[int] = None
    wick_high: Optional[float] = None
    wick_low: Optional[float] = None
    body_high: Optional[float] = None
    body_low: Optional[float] = None
    defend_count: int = 0
    wick_body_ratio: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_detection(self) -> Optional[MDDetection]:
        if not self.found or self.limit_price is None:
            return None
        return MDDetection(
            concept=MDConcept.LIMIT_PRICE,
            direction=self.direction,
            confidence=self.confidence,
            suggested_price=self.limit_price,
            timeframe="M15",
            metadata={
                "limit_price": self.limit_price,
                "defend_count": self.defend_count,
                "wick_body_ratio": self.wick_body_ratio,
            },
        )


class LimitPriceDetector:
    """Detecta mechas extremas defendidas por cuerpos de velas siguientes.

    Pipeline:
      1. Buscar vela con mecha >= 1.5x el cuerpo
      2. Verificar siguientes velas: ¿cierran su cuerpo SIN violar el body-edge?
      3. Si se defiende -> marca el body-edge como LimitPrice
      4. Dirección: contraria a la mecha (mecha arriba = SELL, mecha abajo = BUY)
    """

    def __init__(self, lookback: int = 60,
                 min_wick_body_ratio: float = MIN_WICK_BODY_RATIO,
                 max_defend_candles: int = MAX_DEFEND_CANDLES):
        self._lookback = lookback
        self._min_wick_ratio = min_wick_body_ratio
        self._max_defend = max_defend_candles

    def detect(self, df: pd.DataFrame) -> LimitPriceResult:
        if df is None or len(df) < self._lookback:
            return LimitPriceResult(found=False, direction="NEUTRAL", confidence=0.0)

        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        n = len(df)

        for i in range(max(3, n - self._lookback), n - 2):
            body = abs(closes[i] - opens[i])
            candle_range = highs[i] - lows[i]
            if candle_range == 0 or body == 0:
                continue

            upper_wick = highs[i] - max(opens[i], closes[i])
            lower_wick = min(opens[i], closes[i]) - lows[i]
            is_bearish = closes[i] < opens[i]

            body_low = min(opens[i], closes[i])
            body_high = max(opens[i], closes[i])

            # BUY setup: large lower wick (rejection of selling), body is limit
            if lower_wick >= body * self._min_wick_ratio and lower_wick > upper_wick:
                limit_price = body_low
                direction = "BUY"
                is_mecha_sell = True
            # SELL setup: large upper wick (rejection of buying), body is limit
            elif upper_wick >= body * self._min_wick_ratio and upper_wick > lower_wick:
                limit_price = body_high
                direction = "SELL"
                is_mecha_sell = False
            else:
                continue

            defend_count = 0
            for offset in range(1, min(self._max_defend + 1, n - i)):
                j = i + offset
                c_low = min(opens[j], closes[j])
                c_high = max(opens[j], closes[j])

                if direction == "BUY":
                    if c_low >= limit_price:
                        defend_count += 1
                    else:
                        break
                else:
                    if c_high <= limit_price:
                        defend_count += 1
                    else:
                        break

            if defend_count < 1:
                continue

            max_wick = upper_wick if is_mecha_sell else lower_wick
            wick_ratio = max_wick / body if body != 0 else 1.0

            conf = 0.35
            conf += min(0.20, 0.10 * defend_count)
            conf += min(0.15, (wick_ratio - 1.0) * 0.10)
            vol_boost = self._volume_ratio(df, i)
            if vol_boost > 1.0:
                conf += min(0.10, (vol_boost - 1.0) * 0.10)
            conf = min(0.85, conf)

            notes = [
                f"LimitPrice[{i}] {direction} wick_ratio={wick_ratio:.1f}",
                f"Defend={defend_count}/{self._max_defend} limit={limit_price:.5f}",
                f"Conf={conf:.0%} vol_ratio={vol_boost:.1f}",
            ]

            return LimitPriceResult(
                found=True,
                direction=direction,
                confidence=conf,
                limit_price=limit_price,
                wick_index=i,
                wick_high=highs[i],
                wick_low=lows[i],
                body_high=body_high,
                body_low=body_low,
                defend_count=defend_count,
                wick_body_ratio=wick_ratio,
                notes=notes,
            )

        return LimitPriceResult(found=False, direction="NEUTRAL", confidence=0.0)

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

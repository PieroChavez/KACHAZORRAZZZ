"""PriceCapture Detector — Micro-acumulación previa a breakout masivo
Antes de romper un nivel con violencia, las instituciones dejan una
micro-acumulación (vela Doji o cuerpo mínimo, pausa). Esa es la "huella"
del dinero. El trader espera que el precio retorne a esa huella.

Entry = extremo de la micro-acumulación en dirección del breakout.
Requiere: vela Doji/pausa + vela breakout grande con volumen.

Referencia: MD Class 27
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from src.core.md_concepts import MDConcept, MDDetection

logger = logging.getLogger(__name__)

MAX_DOJI_BODY_RATIO = 0.25
MIN_BREAKOUT_BODY_RATIO = 0.55
MIN_BREAKOUT_MULTIPLIER = 1.8
MIN_VOLUME_SPIKE = 1.3


@dataclass
class PriceCaptureResult:
    found: bool
    direction: str
    confidence: float
    entry_price: Optional[float] = None
    doji_index: Optional[int] = None
    breakout_index: Optional[int] = None
    doji_high: Optional[float] = None
    doji_low: Optional[float] = None
    breakout_body_ratio: float = 0.0
    breakout_multiplier: float = 0.0
    volume_ratio: float = 1.0
    fvg_detected: bool = False
    notes: List[str] = field(default_factory=list)

    def to_detection(self) -> Optional[MDDetection]:
        if not self.found or self.entry_price is None:
            return None
        return MDDetection(
            concept=MDConcept.PRICE_CAPTURE,
            direction=self.direction,
            confidence=self.confidence,
            suggested_price=self.entry_price,
            timeframe="M15",
            metadata={
                "doji_high": self.doji_high,
                "doji_low": self.doji_low,
                "breakout_multiplier": self.breakout_multiplier,
                "fvg_detected": self.fvg_detected,
            },
        )


class PriceCaptureDetector:
    """Detecta micro-acumulaciones previas a breakout masivo.

    Pipeline:
      1. Buscar una vela de pausa (cuerpo <= 25% del rango, DOJI-like)
      2. La siguiente vela debe ser breakout: cuerpo grande, >= 1.8x rango de la pausa
      3. Volumen del breakout >= 1.3x la media reciente
      4. Dirección = dirección del breakout
      5. Entry = extremo de la vela de pausa en dirección del breakout
    """

    def __init__(self, lookback: int = 60,
                 max_doji_ratio: float = MAX_DOJI_BODY_RATIO,
                 min_breakout_ratio: float = MIN_BREAKOUT_BODY_RATIO,
                 min_breakout_mult: float = MIN_BREAKOUT_MULTIPLIER,
                 min_vol_spike: float = MIN_VOLUME_SPIKE):
        self._lookback = lookback
        self._max_doji = max_doji_ratio
        self._min_breakout_ratio = min_breakout_ratio
        self._min_mult = min_breakout_mult
        self._min_vol = min_vol_spike

    def detect(self, df: pd.DataFrame) -> PriceCaptureResult:
        if df is None or len(df) < self._lookback:
            return PriceCaptureResult(found=False, direction="NEUTRAL", confidence=0.0)

        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        n = len(df)

        for i in range(max(3, n - self._lookback), n - 1):
            doji_body = abs(closes[i] - opens[i])
            doji_range = highs[i] - lows[i]
            if doji_range == 0:
                continue

            body_ratio = doji_body / doji_range
            if body_ratio > self._max_doji:
                continue

            j = i + 1
            bk_body = abs(closes[j] - opens[j])
            bk_range = highs[j] - lows[j]
            if bk_range == 0:
                continue

            bk_body_ratio = bk_body / bk_range
            if bk_body_ratio < self._min_breakout_ratio:
                continue

            bk_mult = bk_range / doji_range
            if bk_mult < self._min_mult:
                continue

            direction = "BUY" if closes[j] > opens[j] else "SELL"
            entry_price = lows[i] if direction == "BUY" else highs[i]
            vol_ratio = self._volume_ratio(df, j)
            if vol_ratio < self._min_vol:
                continue

            fvg = False
            if direction == "BUY" and i >= 1:
                fvg = highs[i - 1] < lows[j]
            elif direction == "SELL" and i >= 1:
                fvg = lows[i - 1] > highs[j]

            conf = 0.30
            conf += min(0.20, (bk_mult - 1.5) * 0.10)
            conf += min(0.15, bk_body_ratio * 0.15)
            conf += min(0.10, (vol_ratio - 1.0) * 0.10)
            if fvg:
                conf += 0.10
            conf = min(0.85, conf)

            notes = [
                f"PriceCapture[{i}]: Doji->Breakout en [{j}]",
                f"Mult={bk_mult:.1f}x Vol={vol_ratio:.1f} FVG={fvg}",
                f"Entry {direction} @{entry_price:.5f}",
            ]

            return PriceCaptureResult(
                found=True,
                direction=direction,
                confidence=conf,
                entry_price=entry_price,
                doji_index=i,
                breakout_index=j,
                doji_high=highs[i],
                doji_low=lows[i],
                breakout_body_ratio=bk_body_ratio,
                breakout_multiplier=bk_mult,
                volume_ratio=vol_ratio,
                fvg_detected=fvg,
                notes=notes,
            )

        return PriceCaptureResult(found=False, direction="NEUTRAL", confidence=0.0)

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

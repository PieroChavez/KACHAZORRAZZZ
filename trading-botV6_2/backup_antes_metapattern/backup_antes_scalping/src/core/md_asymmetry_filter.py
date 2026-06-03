"""Asymmetry Filter — Validación de asimetría matemática del setup
La metodología MD exige que toda entrada tenga asimetría matemática:
  1. Impulso rápido (pocas velas) vs retroceso lento (más velas)
  2. Ratio R:R mínimo (SL milimétrico vs TP estructural)
  3. Si el retroceso toma las mismas o menos velas que el impulso → no hay
     agotamiento → setup inválido

No es un detector de patrones, sino un FILTRO que evalúa la calidad del setup.
Retorna un factor multiplicador (0.0 = no pasar, 0.5-1.5 = ajuste de confianza).

Referencia: MD Classes 1, 3, 10, 11, 27 (asimetría matemática como pilar)
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from src.core.md_concepts import MDConcept, MDDetection

logger = logging.getLogger(__name__)

MIN_CANDLE_RATIO_RETRACE_IMPULSE = 1.5
MIN_RR_RATIO = 3.0
MIN_IMPULSE_CANDLES = 2


@dataclass
class AsymmetryResult:
    passed: bool
    asymmetry_factor: float
    impulse_candles: int = 0
    retrace_candles: int = 0
    candle_ratio: float = 0.0
    rr_ratio: float = 0.0
    confidence_bonus: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_detection(self, direction: str) -> Optional[MDDetection]:
        if not self.passed:
            return None
        return MDDetection(
            concept=MDConcept.ASYMMETRY,
            direction=direction,
            confidence=min(0.85, 0.5 + self.asymmetry_factor * 0.3),
            timeframe="M15",
            metadata={
                "impulse_candles": self.impulse_candles,
                "retrace_candles": self.retrace_candles,
                "candle_ratio": self.candle_ratio,
                "rr_ratio": self.rr_ratio,
                "asymmetry_factor": self.asymmetry_factor,
            },
        )


class AsymmetryFilter:
    """Evalúa la asimetría matemática de un setup candidato.

    Args:
        min_candle_ratio: mínimo ratio retrace/impulse (ej. 1.5 = retrace
                          toma 50% más velas que el impulse)
        min_rr_ratio: mínimo riesgo/beneficio (ej. 5.0 = 1:5)
    """

    def __init__(self, min_candle_ratio: float = MIN_CANDLE_RATIO_RETRACE_IMPULSE,
                 min_rr_ratio: float = MIN_RR_RATIO):
        self._min_candles = min_candle_ratio
        self._min_rr = min_rr_ratio

    def evaluate(self, df: pd.DataFrame,
                 direction: str,
                 entry_price: Optional[float] = None,
                 stop_loss: Optional[float] = None,
                 target_price: Optional[float] = None) -> AsymmetryResult:
        """Evalúa la asimetría del setup.

        Args:
            df: DataFrame con velas OHLC
            direction: BUY o SELL
            entry_price: precio de entrada propuesto
            stop_loss: precio del Stop Loss
            target_price: precio objetivo (TP)
        """
        if df is None or len(df) < 10:
            return AsymmetryResult(passed=False, asymmetry_factor=0.0)

        closes = df["close"].values
        opens = df["open"].values
        n = len(df)

        # ── 1. Asimetría temporal: retrace (reciente) vs impulso (anterior) ──
        retrace_end = n - 1
        retrace_start = retrace_end
        for i in range(retrace_end, max(1, retrace_end - 15), -1):
            opposite = (direction == "BUY" and closes[i] < opens[i]) or \
                       (direction == "SELL" and closes[i] > opens[i])
            if not opposite:
                break
            retrace_start = i

        retrace_candles = retrace_end - retrace_start + 1

        impulse_end = retrace_start - 1
        impulse_start = impulse_end
        for i in range(impulse_end, max(1, impulse_end - 20), -1):
            same = (direction == "BUY" and closes[i] > opens[i]) or \
                   (direction == "SELL" and closes[i] < opens[i])
            if not same:
                break
            impulse_start = i

        impulse_candles = impulse_end - impulse_start + 1

        candle_ratio = 0.0
        if retrace_candles > 0 and impulse_candles > 0:
            candle_ratio = retrace_candles / impulse_candles

        # ── 2. Asimetría R:R ──
        rr_ratio = 0.0
        if (entry_price is not None and stop_loss is not None
                and target_price is not None and stop_loss != entry_price):
            risk = abs(entry_price - stop_loss)
            reward = abs(target_price - entry_price)
            if risk > 0:
                rr_ratio = reward / risk

        # ── 3. Factor combinado ──
        temp_factor = 0.0
        if candle_ratio >= self._min_candles:
            temp_factor += 0.4
        elif candle_ratio >= 1.0:
            temp_factor += 0.2

        if rr_ratio >= self._min_rr:
            temp_factor += 0.4
        elif rr_ratio >= 3.0:
            temp_factor += 0.2

        if impulse_candles >= MIN_IMPULSE_CANDLES:
            temp_factor += 0.2

        passed = temp_factor >= 0.6

        notes = [
            f"Asymmetry: impulse={impulse_candles} retrace={retrace_candles} "
            f"ratio={candle_ratio:.1f}",
            f"R:R={rr_ratio:.1f} factor={temp_factor:.1f} passed={passed}",
        ]

        confidence_bonus = temp_factor * 0.1 if passed else -0.1

        return AsymmetryResult(
            passed=passed,
            asymmetry_factor=temp_factor,
            impulse_candles=impulse_candles,
            retrace_candles=retrace_candles,
            candle_ratio=candle_ratio,
            rr_ratio=rr_ratio,
            confidence_bonus=confidence_bonus,
            notes=notes,
        )

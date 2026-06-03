"""Candle Confirmer — Valida tesis en cierres de 1H/4H (MEJORA SCALPING)
Después de que MarketMap da una señal en M1/M5, este módulo espera
el cierre de la vela de 1H y 4H para confirmar que la dirección
sigue siendo válida antes de permitir la entrada completa.
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from src.core.liquidity_mapper import MarketMap

logger = logging.getLogger(__name__)


class ConfirmerStatus(Enum):
    NOT_READY = "NOT_READY"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"
    PENDING = "PENDING"


@dataclass
class CandleConfirmerResult:
    status: ConfirmerStatus
    direction: str
    timeframe: str
    last_close_time: Optional[pd.Timestamp] = None
    current_candle_ratio: float = 0.0
    validation_score: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def is_confirmed(self) -> bool:
        return self.status == ConfirmerStatus.CONFIRMED

    @property
    def is_rejected(self) -> bool:
        return self.status == ConfirmerStatus.REJECTED


class CandleConfirmer:
    """Valida la tesis direccional tras cierre de velas HTF.

    Flujo:
      1. MarketMap genera tesis direccional en M1.
      2. CandleConfirmer espera el cierre de 1H (o 4H) para verificar
         que la vela cierra a favor de la dirección.
      3. Si la vela cierra en contra → REJECTED.
      4. Si cierra a favor → CONFIRMED (se permite entrada).
    """

    def __init__(self, min_close_ratio: float = 0.6,
                 lookback_candles: int = 3):
        self._min_close_ratio = min_close_ratio
        self._lookback = lookback_candles

    def check(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame,
              direction: str, market_map: MarketMap) -> Dict[str, CandleConfirmerResult]:
        return {
            "1h": self._check_single(df_1h, direction, "1h", market_map),
            "4h": self._check_single(df_4h, direction, "4h", market_map),
        }

    def _check_single(self, df: pd.DataFrame, direction: str,
                      tf_label: str, market_map: MarketMap) -> CandleConfirmerResult:
        if df is None or len(df) < self._lookback + 1:
            return CandleConfirmerResult(
                status=ConfirmerStatus.NOT_READY,
                direction=direction,
                timeframe=tf_label,
                notes=[f"Insuficientes datos ({len(df) if df is not None else 0} velas)"],
            )

        last = df.iloc[-1]
        prev = df.iloc[-2]
        last_time = last.name if isinstance(last.name, pd.Timestamp) else None

        ratio = self._candle_body_ratio(last)
        close = last["close"]
        open_ = last["open"]
        high = last["high"]
        low = last["low"]

        prev_dir = "BUY" if prev["close"] > prev["open"] else "SELL" if prev["close"] < prev["open"] else "NEUTRAL"

        bullish = close > open_
        bearish = close < open_

        validation_score = 0.0
        notes = []

        dominant = market_map.dominant_direction if hasattr(market_map, 'dominant_direction') else "NEUTRAL"

        if direction == "BUY" and bullish:
            validation_score = ratio
            if ratio >= self._min_close_ratio:
                notes.append(f"Vela {tf_label} alcista (ratio={ratio:.0%})")
                if prev_dir == "BUY":
                    validation_score = min(1.0, validation_score + 0.15)
                    notes.append(f"2 velas alcistas consecutivas {tf_label}")
                if low <= market_map.nearest_liquidity_below if market_map.nearest_liquidity_below else low > 0:
                    pass
                return CandleConfirmerResult(
                    status=ConfirmerStatus.CONFIRMED,
                    direction=direction,
                    timeframe=tf_label,
                    last_close_time=last_time,
                    current_candle_ratio=ratio,
                    validation_score=validation_score,
                    notes=notes + [f"Tesis {direction} confirmada en {tf_label}"],
                )
            else:
                notes.append(f"Vela alcista pero cuerpo pequeño (ratio={ratio:.0%})")
                return CandleConfirmerResult(
                    status=ConfirmerStatus.PENDING,
                    direction=direction,
                    timeframe=tf_label,
                    last_close_time=last_time,
                    current_candle_ratio=ratio,
                    validation_score=ratio,
                    notes=notes,
                )

        elif direction == "SELL" and bearish:
            validation_score = ratio
            if ratio >= self._min_close_ratio:
                notes.append(f"Vela {tf_label} bajista (ratio={ratio:.0%})")
                if prev_dir == "SELL":
                    validation_score = min(1.0, validation_score + 0.15)
                    notes.append(f"2 velas bajistas consecutivas {tf_label}")
                return CandleConfirmerResult(
                    status=ConfirmerStatus.CONFIRMED,
                    direction=direction,
                    timeframe=tf_label,
                    last_close_time=last_time,
                    current_candle_ratio=ratio,
                    validation_score=validation_score,
                    notes=notes + [f"Tesis {direction} confirmada en {tf_label}"],
                )
            else:
                notes.append(f"Vela bajista pero cuerpo pequeño (ratio={ratio:.0%})")
                return CandleConfirmerResult(
                    status=ConfirmerStatus.PENDING,
                    direction=direction,
                    timeframe=tf_label,
                    last_close_time=last_time,
                    current_candle_ratio=ratio,
                    validation_score=ratio,
                    notes=notes,
                )

        else:
            notes.append(f"Vela {tf_label} en contra ({'alcista' if close > open_ else 'bajista'}) vs tesis {direction}")
            return CandleConfirmerResult(
                status=ConfirmerStatus.REJECTED,
                direction=direction,
                timeframe=tf_label,
                last_close_time=last_time,
                current_candle_ratio=ratio,
                validation_score=0.0,
                notes=notes + [f"Tesis {direction} rechazada en {tf_label}"],
            )

    def _candle_body_ratio(self, candle: pd.Series) -> float:
        body = abs(candle["close"] - candle["open"])
        total = candle["high"] - candle["low"]
        if total == 0:
            return 0.0
        return min(1.0, body / total)

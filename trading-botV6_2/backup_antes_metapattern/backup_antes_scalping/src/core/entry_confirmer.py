"""Entry Confirmer — Timing de entrada con Tick/DOM/StopRun (MEJORA SCALPING)
Valida que el timing de entrada sea correcto usando:
  - Tick delta divergence (precio vs volumen comprador/vendedor)
  - DOM absorption (órdenes grandes siendo absorbidas)
  - Stop-run detection (ruptura falsa seguida de reversión)
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from src.core.micro_phase import MicroPhase, PhaseResult

logger = logging.getLogger(__name__)


@dataclass
class EntryConfirmation:
    valid: bool
    confidence: float
    reason: str
    suggested_entry: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_type: str = "MARKET"
    confirmers_positive: int = 0
    confirmers_total: int = 0
    notes: List[str] = field(default_factory=list)


class TickDeltaAnalyzer:
    """Analiza tick data para detectar divergencia y absorción."""

    def __init__(self, divergence_window: int = 20,
                 delta_threshold: float = 0.6):
        self._window = divergence_window
        self._threshold = delta_threshold

    def analyze(self, ticks: Optional[List[dict]], direction: str) -> Tuple[bool, float, str]:
        if ticks is None or len(ticks) < self._window:
            return False, 0.0, "insufficient_tick_data"

        prices = []
        volumes = []
        aggr = []
        for t in ticks[-self._window:]:
            prices.append(t.get("price", t.get("bid", 0)))
            volumes.append(t.get("volume", t.get("tick_volume", 1)))
            aggr.append(t.get("aggressor", 0))

        if not prices or not volumes:
            return False, 0.0, "no_tick_data"

        price_chg = prices[-1] - prices[self._window // 2]
        delta_buy = sum(v for i, v in enumerate(volumes) if i < len(aggr) and aggr[i] > 0)
        delta_sell = sum(v for i, v in enumerate(volumes) if i < len(aggr) and aggr[i] < 0)
        total_delta = delta_buy + delta_sell
        if total_delta == 0:
            return False, 0.0, "zero_delta"

        delta_ratio = (delta_buy - delta_sell) / total_delta

        if direction == "BUY":
            divergence = price_chg < 0 and delta_ratio > self._threshold
            conf = abs(delta_ratio)
            reason = "bullish_divergence" if divergence else "no_divergence"
        else:
            divergence = price_chg > 0 and delta_ratio < -self._threshold
            conf = abs(delta_ratio)
            reason = "bearish_divergence" if divergence else "no_divergence"

        return divergence, min(1.0, conf), reason


class DOMAnalyzer:
    """Analiza DOM para detectar absorción y órdenes grandes."""

    def __init__(self, absorption_depth: int = 5,
                 absorption_multiple: float = 3.0):
        self._depth = absorption_depth
        self._multiple = absorption_multiple

    def analyze(self, dom: Optional[dict]) -> Tuple[bool, float, str]:
        if dom is None:
            return False, 0.0, "no_dom_data"

        bids = dom.get("bids", [])
        asks = dom.get("asks", [])
        if not bids or not asks:
            return False, 0.0, "empty_dom"

        bid_volumes = [b.get("volume", b.get("size", 0)) for b in bids[:self._depth]]
        ask_volumes = [a.get("volume", a.get("size", 0)) for a in asks[:self._depth]]

        if not bid_volumes or not ask_volumes:
            return False, 0.0, "no_volume"

        avg_bid = np.mean(bid_volumes) if bid_volumes else 0
        avg_ask = np.mean(ask_volumes) if ask_volumes else 0

        if avg_bid > 0 and avg_ask > 0:
            imbalance = abs(avg_bid - avg_ask) / max(avg_bid, avg_ask)
            if imbalance > 0.5:
                return True, min(1.0, imbalance), "dom_imbalance"
        return False, 0.0, "neutral_dom"


class StopRunDetector:
    """Detecta cazas de stops: ruptura de nivel seguida de reversión rápida."""

    def __init__(self, lookback: int = 5, stop_run_range: float = 0.10,
                 reversal_candles: int = 2):
        self._lookback = lookback
        self._range = stop_run_range
        self._reversal_candles = reversal_candles

    def analyze(self, df: pd.DataFrame) -> Tuple[bool, float, str]:
        if df is None or len(df) < self._lookback + self._reversal_candles:
            return False, 0.0, "insufficient_data"

        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        n = len(df)
        mid = n - self._lookback - self._reversal_candles

        before_max = max(high[mid:mid + self._lookback])
        before_min = min(low[mid:mid + self._lookback])

        low_break_val = min(low[-self._reversal_candles:])
        high_break_val = max(high[-self._reversal_candles:])
        low_break = low_break_val < before_min
        high_break = high_break_val > before_max

        if low_break and close[-1] > before_min:
            atr_val = self._atr(df)
            strength = min(1.0, (before_min - low_break_val) / max(atr_val, 0.0001))
            return True, strength, "stop_run_bullish"
        elif high_break and close[-1] < before_max:
            atr_val = self._atr(df)
            strength = min(1.0, (high_break_val - before_max) / max(atr_val, 0.0001))
            return True, strength, "stop_run_bearish"

        return False, 0.0, "no_stop_run"

    def _atr(self, df: pd.DataFrame, period: int = 14) -> float:
        from src.utils.helpers import atr
        if len(df) < period + 1:
            return 0.0
        return float(atr(df, period).iloc[-1])


class EntryConfirmer:
    """Orquestador de confirmadores. Requiere 2 de 3 para validar entrada."""

    def __init__(self, min_confirmers: int = 2,
                 tick_analyzer: Optional[TickDeltaAnalyzer] = None,
                 dom_analyzer: Optional[DOMAnalyzer] = None,
                 stop_run: Optional[StopRunDetector] = None):
        self._min_confirmers = min_confirmers
        self._tick = tick_analyzer or TickDeltaAnalyzer()
        self._dom = dom_analyzer or DOMAnalyzer()
        self._stop_run = stop_run or StopRunDetector()

    def confirm(self, symbol: str, direction: str,
                phase: PhaseResult, df: pd.DataFrame,
                ticks: Optional[List[dict]] = None,
                dom: Optional[dict] = None) -> EntryConfirmation:
        if not phase.allows_entry:
            return EntryConfirmation(False, 0.0, f"phase_{phase.phase.value}_no_entry")

        results = []
        tick_ok, tick_conf, tick_reason = self._tick.analyze(ticks, direction)
        dom_ok, dom_conf, dom_reason = self._dom.analyze(dom)
        stop_ok, stop_conf, stop_reason = self._stop_run.analyze(df)

        if tick_ok:
            results.append(("tick_delta", tick_conf, tick_reason))
        if dom_ok:
            results.append(("dom_absorption", dom_conf, dom_reason))
        if stop_ok:
            results.append(("stop_run", stop_conf, stop_reason))

        positives = len(results)
        total = 3
        valid = positives >= self._min_confirmers

        if not valid and phase.phase in (MicroPhase.IMPULSE_STARTING, MicroPhase.BREAKOUT_RETEST):
            if positives >= 1 and phase.confidence >= 0.7:
                valid = True
                results.append(("phase_override", 0.6, "high_confidence_phase_override"))

        avg_conf = np.mean([r[1] for r in results]) if results else phase.confidence * 0.3
        suggested_entry = df["close"].iloc[-1] if df is not None and len(df) > 0 else None

        notes = [f"Confirmers: {positives}/{total}"]
        for name, conf, reason in results:
            notes.append(f"  {name}: confidence={conf:.0%}, reason={reason}")

        return EntryConfirmation(
            valid=valid,
            confidence=avg_conf,
            reason=f"{positives}/{total} confirmers positive" if valid else "insufficient_confirmation",
            suggested_entry=suggested_entry,
            confirmers_positive=positives,
            confirmers_total=total,
            notes=notes,
        )

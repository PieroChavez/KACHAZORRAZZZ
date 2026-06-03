"""Trailing Stop Inteligente y Gestión de Salidas — MEJORA 11 (Modo Experto)
Módulo dedicado con múltiples algoritmos de trailing profesional:
  - Chandelier Exit (ATR desde máximo/mínimo)
  - Parabolic SAR
  - Fractal Trailing (estructura de mercado)
  - Breakeven + Lock progresivo por ratio R:R
  - Partial Take Profit inteligente
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TrailAlgorithm(Enum):
    CHANDELIER = "chandelier"
    PARABOLIC_SAR = "parabolic_sar"
    FRACTAL = "fractal"


class TrailStage(Enum):
    OFF = 0
    WIDE = 1
    MEDIUM = 2
    TIGHT = 3
    LOCK = 4


STAGE_NAMES = {0: "off", 1: "wide", 2: "medium", 3: "tight", 4: "lock"}
STAGE_MULTIPLIERS = {1: 3.5, 2: 3.0, 3: 2.0, 4: 1.5}


@dataclass
class TrailingConfig:
    algorithm: TrailAlgorithm = TrailAlgorithm.CHANDELIER
    atr_period: int = 14
    chandelier_mult: float = 3.0
    chandelier_base_mult: float = 3.0
    sar_acceleration: float = 0.02
    sar_max_acceleration: float = 0.2
    fractal_lookback: int = 3
    be_at_ratio: float = 0.20
    lock1_at_ratio: float = 0.40
    lock2_at_ratio: float = 0.60
    lock3_at_ratio: float = 0.80
    trail_stage_1_mult: float = 3.5
    trail_stage_2_mult: float = 3.0
    trail_stage_3_mult: float = 2.0
    trail_stage_4_mult: float = 1.5
    partial_tp1_ratio: float = 0.30
    partial_tp1_close_pct: float = 0.50
    partial_tp2_ratio: float = 0.60
    partial_tp2_close_pct: float = 0.30
    min_trail_distance_pips: float = 15.0
    recalc_on_new_signal: bool = True
    immediate_trail_atr_mult: float = 3.0
    close_on_reversal_score: float = 65.0
    close_on_loss_reversal_score: float = 55.0

    @classmethod
    def default(cls) -> "TrailingConfig":
        return cls()


@dataclass
class TrailingState:
    symbol: str
    is_long: bool
    entry_price: float
    original_sl: float
    original_tp: float
    original_tp_distance: float
    ticket: int
    volume: float
    be_activated: bool = False
    trail_activated: bool = False
    trail_stage: int = 0
    trail_algorithm: str = "chandelier"
    tp_expanded: int = 0
    managed_tp: Optional[float] = None
    partial_tp1_done: bool = False
    partial_tp2_done: bool = False
    recalculated: bool = False
    last_price: float = 0.0
    last_profit: float = 0.0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "is_long": self.is_long,
            "entry_price": self.entry_price, "original_sl": self.original_sl,
            "original_tp": self.original_tp,
            "original_tp_distance": self.original_tp_distance,
            "be_activated": self.be_activated,
            "trail_activated": self.trail_activated,
            "trail_stage": self.trail_stage,
            "trail_algorithm": self.trail_algorithm,
            "tp_expanded": self.tp_expanded,
            "managed_tp": self.managed_tp,
            "partial_tp1_done": self.partial_tp1_done,
            "partial_tp2_done": self.partial_tp2_done,
            "recalculated": self.recalculated,
            "last_profit": self.last_profit,
            "last_price": self.last_price,
            "volume": self.volume,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrailingState":
        return cls(
            symbol=d["symbol"], is_long=d["is_long"],
            entry_price=d["entry_price"],
            original_sl=d["original_sl"], original_tp=d["original_tp"],
            original_tp_distance=d["original_tp_distance"],
            ticket=d.get("ticket", 0), volume=d.get("volume", 0),
            be_activated=d.get("be_activated", False),
            trail_activated=d.get("trail_activated", False),
            trail_stage=d.get("trail_stage", 0),
            trail_algorithm=d.get("trail_algorithm", "chandelier"),
            tp_expanded=d.get("tp_expanded", 0),
            managed_tp=d.get("managed_tp"),
            partial_tp1_done=d.get("partial_tp1_done", False),
            partial_tp2_done=d.get("partial_tp2_done", False),
            recalculated=d.get("recalculated", False),
            last_price=d.get("last_price", 0.0),
            last_profit=d.get("last_profit", 0.0),
        )


@dataclass
class TrailResult:
    new_sl: float
    new_tp: Optional[float]
    updated_state: TrailingState
    actions: List[str] = field(default_factory=list)
    should_close: bool = False
    close_reason: str = ""


@dataclass
class PartialTPResult:
    close_pct: float
    reason: str
    remaining_volume: float
    update_tp: Optional[float] = None


class TrailingStopManager:
    """Multi-algorithm trailing stop and exit management."""

    def __init__(self, config: Optional[TrailingConfig] = None):
        self.config = config or TrailingConfig.default()
        self._states: Dict[int, TrailingState] = {}

    def initialize_state(
        self, ticket: int, symbol: str, is_long: bool,
        entry_price: float, current_sl: float, current_tp: float,
        current_price: float, profit: float, volume: float,
    ) -> TrailingState:
        tp_distance = abs(current_tp - entry_price) if current_tp else 0
        price_profit = current_price - entry_price if is_long else entry_price - current_price

        be_activated = False
        trail_stage = 0
        trail_activated = False

        if tp_distance > 0 and price_profit > 0:
            pct_tp = price_profit / tp_distance
            if pct_tp >= self.config.be_at_ratio:
                be_activated = True
            if pct_tp >= self.config.lock1_at_ratio:
                trail_activated = True
                if pct_tp < self.config.lock2_at_ratio:
                    trail_stage = 2
                elif pct_tp < self.config.lock3_at_ratio:
                    trail_stage = 3
                else:
                    trail_stage = 4

        state = TrailingState(
            symbol=symbol, is_long=is_long,
            entry_price=entry_price, original_sl=current_sl,
            original_tp=current_tp, original_tp_distance=tp_distance,
            ticket=ticket, volume=volume,
            be_activated=be_activated,
            trail_activated=trail_activated,
            trail_stage=trail_stage,
            managed_tp=current_tp,
            last_price=current_price, last_profit=profit,
        )
        self._states[ticket] = state
        return state

    def get_state(self, ticket: int) -> Optional[TrailingState]:
        return self._states.get(ticket)

    def update_state(self, ticket: int, profit: float, price: float):
        state = self._states.get(ticket)
        if state:
            state.last_profit = profit
            state.last_price = price

    def remove_state(self, ticket: int):
        self._states.pop(ticket, None)

    def compute(
        self,
        ticket: int,
        state: TrailingState,
        ltf_df: pd.DataFrame,
        current_price: float,
        current_sl: float,
        current_tp: float,
        pip: float,
        digit: int,
        fresh_signal_dir: str = "HOLD",
        fresh_signal_score: float = 0.0,
        profile: Optional[object] = None,
    ) -> TrailResult:
        """Compute new SL/TP based on configured algorithm and current market state."""
        actions = []
        new_sl = current_sl
        new_tp = current_tp
        should_close = False
        close_reason = ""

        if ltf_df is None or len(ltf_df) < self.config.atr_period + 5:
            return TrailResult(
                new_sl=new_sl, new_tp=new_tp,
                updated_state=state, actions=["insufficient_data"],
            )

        atr_val = self._compute_atr(ltf_df)
        if atr_val <= 0:
            return TrailResult(
                new_sl=new_sl, new_tp=new_tp,
                updated_state=state, actions=["atr_zero"],
            )

        entry_price = state.entry_price
        is_long = state.is_long
        price_profit = current_price - entry_price if is_long else entry_price - current_price

        algorithm = TrailAlgorithm(self.config.algorithm)

        # ── 2. Algorithm-specific trailing ──
        if algorithm == TrailAlgorithm.CHANDELIER:
            new_sl = self._chandelier_exit(
                ltf_df, current_price, entry_price, is_long,
                atr_val, pip, digit, state, actions,
            )
        elif algorithm == TrailAlgorithm.PARABOLIC_SAR:
            new_sl = self._parabolic_sar_trail(
                ltf_df, current_price, is_long, digit, state, actions,
            )
        elif algorithm == TrailAlgorithm.FRACTAL:
            new_sl = self._fractal_trail(
                ltf_df, current_price, is_long,
                atr_val, pip, digit, state, actions,
            )

        # ── 3. Breakeven activation ──
        new_sl = self._apply_breakeven(
            state, entry_price, current_price, is_long,
            new_sl, pip, digit, actions,
        )

        # ── 4. Stage-based trailing override if trail already active ──
        if state.trail_activated:
            stage_sl = self._stage_trail(
                current_price, is_long, atr_val,
                state.trail_stage, pip, digit,
            )
            if is_long:
                new_sl = max(new_sl, stage_sl)
            else:
                new_sl = min(new_sl, stage_sl)
            actions.append(f"Stage trail ({STAGE_NAMES[state.trail_stage]}): SL→{new_sl}")

        return TrailResult(
            new_sl=round(new_sl, digit) if new_sl else new_sl,
            new_tp=round(new_tp, digit) if new_tp else new_tp,
            updated_state=state,
            actions=actions,
            should_close=should_close,
            close_reason=close_reason,
        )

    def compute_partial_tp(
        self, state: TrailingState, current_price: float,
        is_long: bool,
    ) -> Optional[PartialTPResult]:
        """Check partial take-profit levels and return close instructions."""
        entry = state.entry_price
        tp_distance = state.original_tp_distance
        if tp_distance <= 0:
            return None

        price_profit = current_price - entry if is_long else entry - current_price
        pct_of_tp = price_profit / tp_distance

        if not state.partial_tp1_done and pct_of_tp >= self.config.partial_tp1_ratio:
            remaining = 1.0 - self.config.partial_tp1_close_pct
            state.partial_tp1_done = True
            return PartialTPResult(
                close_pct=self.config.partial_tp1_close_pct,
                reason=f"TP1 at {pct_of_tp:.0%} of TP ({price_profit:.1f}p)",
                remaining_volume=remaining,
            )

        if not state.partial_tp2_done and pct_of_tp >= self.config.partial_tp2_ratio:
            remaining = 1.0 - self.config.partial_tp2_close_pct
            state.partial_tp2_done = True
            return PartialTPResult(
                close_pct=self.config.partial_tp2_close_pct,
                reason=f"TP2 at {pct_of_tp:.0%} of TP ({price_profit:.1f}p)",
                remaining_volume=remaining,
            )

        return None

    def recalculate_sl_tp(
        self,
        state: TrailingState,
        ltf_df: pd.DataFrame,
        fresh_tp: float,
        profile: object,
        pip: float,
        digit: int,
        sl_mult: float,
        tp_mult: float,
        base_sl_pips: float,
        base_tp_pips: float,
    ) -> TrailResult:
        """Recalculate SL/TP from a fresh same-direction signal (VolatilityScaler)."""
        entry_price = state.entry_price
        is_long = state.is_long
        atr_val = self._compute_atr(ltf_df)

        sl_min_dist = getattr(profile, "sl_min_pips", 5.0) * pip
        sl_max_dist = getattr(profile, "sl_max_pips", 20.0) * pip
        sl_distance = atr_val * sl_mult
        sl_distance = max(min(sl_distance, sl_max_dist), sl_min_dist)
        tp_distance = atr_val * tp_mult

        if is_long:
            new_sl = round(entry_price - sl_distance, digit)
            tp_min = max(
                state.last_price + tp_distance,
                entry_price + sl_distance * 2,
            ) if state.last_price > 0 else entry_price + tp_distance
            new_tp = max(fresh_tp, tp_min)
        else:
            new_sl = round(entry_price + sl_distance, digit)
            tp_min = min(
                state.last_price - tp_distance,
                entry_price - sl_distance * 2,
            ) if state.last_price > 0 else entry_price - tp_distance
            new_tp = min(fresh_tp, tp_min)

        new_tp_dist = abs(new_tp - entry_price)
        state.be_activated = False
        state.trail_activated = False
        state.trail_stage = 0
        state.tp_expanded = 0
        state.original_tp_distance = new_tp_dist
        state.managed_tp = new_tp
        state.original_sl = new_sl
        state.recalculated = True

        actions = [f"SL/TP recalc: SL {new_sl} TP {new_tp} (ATR×{sl_mult:.1f}/{tp_mult:.1f})"]

        # Immediate trailing if deep in profit
        if state.last_price > 0:
            price_profit = state.last_price - entry_price if is_long else entry_price - state.last_price
            if price_profit > atr_val * 3:
                trail_pct = price_profit / new_tp_dist if new_tp_dist > 0 else 0
                imm_stage = self._classify_immediate_stage(trail_pct)
                imm_mult = STAGE_MULTIPLIERS.get(imm_stage, 3.0)
                trail_atr = max(atr_val * imm_mult, pip * 15)
                if is_long:
                    trail_sl = round(max(state.last_price - trail_atr, new_sl), digit)
                    if trail_sl > new_sl:
                        new_sl = trail_sl
                        state.trail_activated = True
                        state.trail_stage = imm_stage
                        actions.append(f"Immediate trail stage={imm_stage}({STAGE_NAMES[imm_stage]}) SL→{new_sl}")
                else:
                    trail_sl = round(min(state.last_price + trail_atr, new_sl), digit)
                    if trail_sl < new_sl:
                        new_sl = trail_sl
                        state.trail_activated = True
                        state.trail_stage = imm_stage
                        actions.append(f"Immediate trail stage={imm_stage}({STAGE_NAMES[imm_stage]}) SL→{new_sl}")

        return TrailResult(
            new_sl=round(new_sl, digit), new_tp=round(new_tp, digit),
            updated_state=state, actions=actions,
        )

    def generate_partial_close_plan(
        self, state: TrailingState, current_price: float, is_long: bool,
        total_volume: float,
    ) -> List[PartialTPResult]:
        """Generate full partial close plan for a scaling-in strategy."""
        results = []
        tp_distance = state.original_tp_distance
        if tp_distance <= 0:
            return results
        price_profit = current_price - state.entry_price if is_long else state.entry_price - current_price
        pct = price_profit / tp_distance

        levels = [
            (0.30, 0.50, "TP1"),
            (0.60, 0.30, "TP2"),
            (0.80, 0.20, "TP3"),
        ]
        cumulative = 0.0
        for level_pct, close_pct, label in levels:
            if pct >= level_pct and cumulative < 0.95:
                actual_close = min(close_pct, 1.0 - cumulative)
                if actual_close > 0.01:
                    results.append(PartialTPResult(
                        close_pct=actual_close,
                        reason=f"{label} at {pct:.0%} of TP",
                        remaining_volume=1.0 - cumulative - actual_close,
                    ))
                    cumulative += actual_close
        return results

    # ── Private algorithm implementations ──

    def _chandelier_exit(
        self, df: pd.DataFrame, current_price: float,
        entry_price: float, is_long: bool, atr_val: float,
        pip: float, digit: int, state: TrailingState, actions: List[str],
    ) -> float:
        mult = self.config.chandelier_mult
        if is_long:
            lookback = 22
            highest = df["high"].iloc[-lookback:].max() if len(df) >= lookback else df["high"].max()
            chandelier = highest - atr_val * mult
            result = max(chandelier, current_price - atr_val * mult)
            if result > (state.original_sl or 0):
                actions.append(f"Chandelier: high={highest:.5f} ATR×{mult:.1f} SL→{result}")
            return result
        else:
            lookback = 22
            lowest = df["low"].iloc[-lookback:].min() if len(df) >= lookback else df["low"].min()
            chandelier = lowest + atr_val * mult
            result = min(chandelier, current_price + atr_val * mult)
            if result < (state.original_sl or 999):
                actions.append(f"Chandelier: low={lowest:.5f} ATR×{mult:.1f} SL→{result}")
            return result

    def _parabolic_sar_trail(
        self, df: pd.DataFrame, current_price: float,
        is_long: bool, digit: int, state: TrailingState, actions: List[str],
    ) -> float:
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        acc = self.config.sar_acceleration
        max_acc = self.config.sar_max_acceleration

        psar = self._compute_psar(high, low, close, acc, max_acc)
        sar = psar[-1] if len(psar) > 0 else (current_price - current_price * 0.01)

        if is_long:
            result = max(sar, state.original_sl or 0)
        else:
            result = min(sar, state.original_sl or 999)

        actions.append(f"PSAR: sar={sar:.5f} SL→{result}")
        return round(result, digit)

    def _fractal_trail(
        self, df: pd.DataFrame, current_price: float,
        is_long: bool, atr_val: float, pip: float,
        digit: int, state: TrailingState, actions: List[str],
    ) -> float:
        from src.utils.helpers import find_swing_points
        lookback = self.config.fractal_lookback
        highs_idx, lows_idx = find_swing_points(df, lookback=lookback)

        if is_long and lows_idx:
            last_low = df["low"].iloc[lows_idx[-1]]
            buffer = atr_val * 0.5
            result = max(last_low - buffer, current_price - atr_val * self.config.chandelier_mult)
            if result > (state.original_sl or 0):
                actions.append(f"Fractal: swing low={last_low:.5f} SL→{result}")
            return max(result, state.original_sl or 0)
        elif not is_long and highs_idx:
            last_high = df["high"].iloc[highs_idx[-1]]
            buffer = atr_val * 0.5
            result = min(last_high + buffer, current_price + atr_val * self.config.chandelier_mult)
            if result < (state.original_sl or 999):
                actions.append(f"Fractal: swing high={last_high:.5f} SL→{result}")
            return min(result, state.original_sl or 999)

        return self._chandelier_exit(df, current_price, state.entry_price, is_long, atr_val, pip, digit, state, actions)

    def _apply_breakeven(
        self, state: TrailingState, entry_price: float,
        current_price: float, is_long: float, current_sl: float,
        pip: float, digit: int, actions: List[str],
    ) -> float:
        if state.be_activated:
            return current_sl

        price_profit = current_price - entry_price if is_long else entry_price - current_price
        tp_distance = state.original_tp_distance
        if tp_distance <= 0:
            return current_sl
        pct_of_tp = price_profit / tp_distance

        if pct_of_tp >= self.config.be_at_ratio:
            min_pip_pips = 5.0
            buffer = max(pip * min_pip_pips, pip * 0.05)
            be_sl = round(entry_price + buffer, digit) if is_long else round(entry_price - buffer, digit)

            if (is_long and be_sl > current_sl) or (not is_long and be_sl < current_sl):
                state.be_activated = True
                actions.append(f"BE activated at {pct_of_tp*100:.0f}% of TP ({price_profit/pip:.0f}p)")
                return be_sl
        return current_sl

    def _stage_trail(
        self, current_price: float, is_long: bool,
        atr_val: float, stage: int, pip: float, digit: int,
    ) -> float:
        mult = STAGE_MULTIPLIERS.get(stage, 3.0)
        trail_atr = max(atr_val * mult, pip * self.config.min_trail_distance_pips)
        if is_long:
            return round(current_price - trail_atr, digit)
        else:
            return round(current_price + trail_atr, digit)

    @staticmethod
    def _classify_immediate_stage(trail_pct: float) -> int:
        if trail_pct >= 0.80:
            return 4
        elif trail_pct >= 0.60:
            return 3
        elif trail_pct >= 0.40:
            return 2
        return 1

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
        from src.utils.helpers import atr
        return atr(df, period).iloc[-1]

    @staticmethod
    def _compute_psar(
        high: np.ndarray, low: np.ndarray, close: np.ndarray,
        acceleration: float = 0.02, max_acceleration: float = 0.2,
    ) -> np.ndarray:
        n = len(high)
        psar = np.zeros(n)
        trend = np.zeros(n)
        ep = np.zeros(n)
        af = np.zeros(n)

        psar[0] = low[0]
        trend[0] = 1
        ep[0] = high[0]
        af[0] = acceleration

        for i in range(1, n):
            if trend[i - 1] == 1:
                psar[i] = psar[i - 1] + af[i - 1] * (ep[i - 1] - psar[i - 1])
                psar[i] = min(psar[i], low[i - 1], low[i - 2]) if i >= 2 else min(psar[i], low[i - 1])
                if high[i] > ep[i - 1]:
                    ep[i] = high[i]
                    af[i] = min(af[i - 1] + acceleration, max_acceleration)
                else:
                    ep[i] = ep[i - 1]
                    af[i] = af[i - 1]
                if low[i] < psar[i]:
                    trend[i] = -1
                    psar[i] = ep[i - 1]
                    af[i] = acceleration
                    ep[i] = low[i]
                else:
                    trend[i] = 1
            else:
                psar[i] = psar[i - 1] - af[i - 1] * (psar[i - 1] - ep[i - 1])
                psar[i] = max(psar[i], high[i - 1], high[i - 2]) if i >= 2 else max(psar[i], high[i - 1])
                if low[i] < ep[i - 1]:
                    ep[i] = low[i]
                    af[i] = min(af[i - 1] + acceleration, max_acceleration)
                else:
                    ep[i] = ep[i - 1]
                    af[i] = af[i - 1]
                if high[i] > psar[i]:
                    trend[i] = 1
                    psar[i] = ep[i - 1]
                    af[i] = acceleration
                    ep[i] = high[i]
                else:
                    trend[i] = -1
        return psar

"""Simulated Order Executor for Backtesting
Replaces MT5 OrderExecutor with an in-memory position tracker
that simulates fills, SL/TP hits, and P&L calculation.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.pattern_detector import Pattern
from ..executor.order_types import OrderType


@dataclass
class SimulatedTrade:
    symbol: str
    direction: str  # "BUY" or "SELL"
    entry_time: datetime
    entry_price: float
    volume: float
    stop_loss: float
    take_profit: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""  # "sl", "tp", "signal", "trailing"
    profit: float = 0.0
    profit_pct: float = 0.0
    bars_held: int = 0
    pattern_type: Optional[str] = None
    regime: str = ""
    session: str = ""
    score: float = 0.0
    conviction: float = 0.0


class SimulatedExecutor:
    def __init__(self, initial_balance: float = 10_000.0, spread_pips: float = 0.5):
        self.balance = initial_balance
        self.equity = initial_balance
        self.spread_pips = spread_pips
        self.positions: dict[int, SimulatedTrade] = {}
        self._next_id = 1
        self._closed: List[SimulatedTrade] = []
        self.equity_curve: List[float] = [initial_balance]

    @property
    def open_positions(self) -> List[SimulatedTrade]:
        return list(self.positions.values())

    @property
    def closed_trades(self) -> List[SimulatedTrade]:
        return list(self._closed)

    def can_open(self, max_positions: int) -> bool:
        return len(self.positions) < max_positions

    def open_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        volume: float,
        stop_loss: float,
        take_profit: float,
        pattern_type: Optional[str] = None,
        regime: str = "",
        session: str = "",
        score: float = 0.0,
        conviction: float = 0.0,
    ) -> Optional[int]:
        if not self.can_open(999):
            return None
        direction_adj = 1.0 if direction.upper() == "BUY" else -1.0
        adjusted_entry = entry_price + direction_adj * (self.spread_pips * 0.0001)
        trade = SimulatedTrade(
            symbol=symbol,
            direction=direction.upper(),
            entry_time=datetime.now(),
            entry_price=adjusted_entry,
            volume=volume,
            stop_loss=stop_loss,
            take_profit=take_profit,
            pattern_type=pattern_type,
            regime=regime,
            session=session,
            score=score,
            conviction=conviction,
        )
        tid = self._next_id
        self._next_id += 1
        self.positions[tid] = trade
        risk_amt = self._calc_risk(trade)
        self.balance -= risk_amt
        self.equity = self.balance
        return tid

    def update_positions(self, high: float, low: float, close: float,
                         timestamp: datetime, trailing_config: Optional[dict] = None):
        to_close = []
        for tid, trade in self.positions.items():
            trade.bars_held += 1
            direction = 1.0 if trade.direction == "BUY" else -1.0
            if trade.direction == "BUY":
                if low <= trade.stop_loss:
                    to_close.append((tid, trade.stop_loss, "sl"))
                elif high >= trade.take_profit:
                    to_close.append((tid, trade.take_profit, "tp"))
                else:
                    trade.exit_price = close
            else:
                if high >= trade.stop_loss:
                    to_close.append((tid, trade.stop_loss, "sl"))
                elif low <= trade.take_profit:
                    to_close.append((tid, trade.take_profit, "tp"))
                else:
                    trade.exit_price = close
            if trailing_config:
                self._apply_trailing(trade, high, low, close, trailing_config)
        for tid, exit_price, reason in to_close:
            self._close_position(tid, exit_price, reason, timestamp)

    def _apply_trailing(self, trade: SimulatedTrade, high: float, low: float,
                        close: float, config: dict):
        if trade.direction == "BUY":
            new_high = max(trade.entry_price, high)
            atr_mult = config.get("trail_mult", 2.0)
            atr_val = config.get("atr", 0.0)
            if atr_val > 0:
                candidate = new_high - atr_val * atr_mult
                if candidate > trade.stop_loss:
                    trade.stop_loss = candidate
        else:
            new_low = min(trade.entry_price, low)
            atr_mult = config.get("trail_mult", 2.0)
            atr_val = config.get("atr", 0.0)
            if atr_val > 0:
                candidate = new_low + atr_val * atr_mult
                if candidate < trade.stop_loss:
                    trade.stop_loss = candidate

    def close_position(self, tid: int, reason: str = "signal", timestamp: Optional[datetime] = None):
        if tid in self.positions:
            trade = self.positions[tid]
            self._close_position(tid, trade.exit_price or trade.entry_price, reason, timestamp)

    def close_all(self, reason: str = "end", timestamp: Optional[datetime] = None):
        for tid in list(self.positions.keys()):
            self.close_position(tid, reason, timestamp)

    def get_peak_equity(self) -> float:
        return max(self.equity_curve) if self.equity_curve else self.equity

    def get_drawdown_pct(self) -> float:
        peak = self.get_peak_equity()
        return (peak - self.equity) / peak if peak > 0 else 0.0

    def _close_position(self, tid: int, exit_price: float, reason: str, timestamp: Optional[datetime] = None):
        trade = self.positions.pop(tid, None)
        if trade is None:
            return
        trade.exit_time = timestamp or datetime.now()
        trade.exit_price = exit_price
        trade.exit_reason = reason
        direction = 1.0 if trade.direction == "BUY" else -1.0
        price_diff = (exit_price - trade.entry_price) * direction
        pip_value = self._pip_value(trade.symbol)
        trade.profit = price_diff / 0.0001 * pip_value * trade.volume if pip_value > 0 else price_diff * trade.volume
        trade.profit_pct = price_diff / trade.entry_price * 100.0
        self.balance += trade.profit
        self.equity = self.balance
        self.equity_curve.append(self.equity)
        self._closed.append(trade)

    def _calc_risk(self, trade: SimulatedTrade) -> float:
        direction = 1.0 if trade.direction == "BUY" else -1.0
        sl_distance = (trade.entry_price - trade.stop_loss) * direction
        if sl_distance <= 0:
            return 0.0
        pip_value = self._pip_value(trade.symbol)
        return sl_distance / 0.0001 * pip_value * trade.volume if pip_value > 0 else 0.0

    @staticmethod
    def _pip_value(symbol: str) -> float:
        s = symbol.upper()
        if "XAU" in s or s == "GOLD":
            return 10.0
        if "XAG" in s or s == "SILVER":
            return 50.0
        if s in ("NAS100", "US100", "NDX", "DJI30", "US30", "SPX500"):
            return 1.0
        return 10.0

"""Position Tracker
Tracks open positions and their status"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum

from ..adapters.mt5_client import MT5Client


class PositionStatus(Enum):
    """Position status"""
    OPEN = "open"
    CLOSED = "closed"
    STOPPED_OUT = "stopped_out"
    TAKEN_PROFIT = "taken_profit"


@dataclass
class TrackedPosition:
    """Position being tracked"""
    ticket: int
    symbol: str
    direction: str
    volume: float
    entry_price: float
    stop_loss: float
    take_profit: float
    open_time: datetime
    status: PositionStatus = PositionStatus.OPEN
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    trailing_stop: Optional[float] = None


class PositionTracker:
    """Tracks all open positions and their state"""

    def __init__(self, mt5_client: MT5Client):
        self.mt5 = mt5_client
        self._positions: dict[int, TrackedPosition] = {}

    def sync_with_broker(self, symbol: Optional[str] = None):
        """Synchronize tracked positions with broker positions"""
        broker_positions = self.mt5.get_positions(symbol)

        # Update existing and add new
        for bp in broker_positions:
            ticket = bp["ticket"]
            if ticket not in self._positions:
                self._positions[ticket] = TrackedPosition(
                    ticket=ticket,
                    symbol=bp["symbol"],
                    direction=bp["type"],
                    volume=bp["volume"],
                    entry_price=bp["price_open"],
                    stop_loss=bp["sl"],
                    take_profit=bp["tp"],
                    open_time=bp["time"].to_pydatetime(),
                    status=PositionStatus.OPEN
                )
            else:
                # Update current state
                self._positions[ticket].current_price = bp["price_current"]
                self._positions[ticket].unrealized_pnl = bp["profit"]

        # Remove closed positions from tracking
        broker_tickets = {bp["ticket"] for bp in broker_positions}
        closed_tickets = [
            t for t, p in self._positions.items()
            if t not in broker_tickets
        ]
        for t in closed_tickets:
            del self._positions[t]

    def get_open_positions(self) -> list[TrackedPosition]:
        """Get all open positions"""
        return [p for p in self._positions.values()
                if p.status == PositionStatus.OPEN]

    def get_position(self, ticket: int) -> Optional[TrackedPosition]:
        """Get a specific position by ticket"""
        return self._positions.get(ticket)

    def update_trailing_stop(self, ticket: int, new_sl: float):
        """Update trailing stop for a position"""
        if ticket in self._positions:
            self._positions[ticket].trailing_stop = new_sl

    def close_position(self, ticket: int, reason: PositionStatus):
        """Mark a position as closed"""
        if ticket in self._positions:
            self._positions[ticket].status = reason

    def get_total_exposure(self) -> dict:
        """Get total exposure across all positions"""
        total_volume = sum(p.volume for p in self.get_open_positions())
        total_pnl = sum(p.unrealized_pnl for p in self.get_open_positions())

        return {
            "position_count": len(self.get_open_positions()),
            "total_volume": total_volume,
            "total_pnl": total_pnl
        }

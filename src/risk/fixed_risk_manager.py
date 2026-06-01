"""Fixed Risk Position Sizing and Management"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
from loguru import logger

from ..core.candle_closure_ratings import CandleRating, ClosureRating
from ..utils.helpers import pip_size


@dataclass
class RiskConfig:
    """Risk management configuration"""
    risk_per_trade: float = 0.02  # 2% per trade
    max_daily_loss: float = 0.06  # 6% daily loss limit
    max_positions: int = 1
    atr_multiplier_sl: float = 2.0
    atr_multiplier_tp: float = 4.0
    min_reward_risk_ratio: float = 2.0


@dataclass
class PositionSize:
    """Calculated position size"""
    volume: float  # Lot size
    sl_pips: float
    tp_pips: float
    risk_amount: float
    potential_reward: float
    reward_risk_ratio: float


class FixedRiskManager:
    """Fixed percentage risk position sizing"""

    def __init__(self, config: RiskConfig, balance: float):
        self.config = config
        self.balance = balance
        self.daily_loss = 0.0
        self.trades_today = 0
        self.last_reset = pd.Timestamp.now().date()

    def reset_daily(self):
        """Reset daily counters"""
        today = pd.Timestamp.now().date()
        if today > self.last_reset:
            self.daily_loss = 0.0
            self.trades_today = 0
            self.last_reset = today

    def can_trade(self) -> tuple[bool, str]:
        """Check if trading is allowed based on risk limits"""
        self.reset_daily()

        # Check daily loss limit
        daily_loss_pct = self.daily_loss / self.balance
        if daily_loss_pct >= self.config.max_daily_loss:
            return False, f"Daily loss limit reached ({daily_loss_pct:.2%})"

        # Check trade count
        if self.trades_today >= self.config.max_positions and self.config.max_positions > 0:
            return False, "Max daily trades reached"

        return True, "OK"

    def calculate_position_size(self, symbol: str, entry_price: float,
                                 stop_loss: float, take_profit: float,
                                 atr: Optional[float] = None,
                                 risk_per_trade_pct: Optional[float] = None,
                                 conviction: Optional[float] = None) -> PositionSize:
        risk_frac = risk_per_trade_pct if risk_per_trade_pct is not None else self.config.risk_per_trade

        if conviction is not None:
            risk_frac = self._conviction_adjusted_risk(conviction, risk_frac)

        risk_amount = self.balance * risk_frac

        pip_size = self._get_pip_size(symbol)
        sl_pips = abs(entry_price - stop_loss) / pip_size
        tp_pips = abs(take_profit - entry_price) / pip_size

        reward_risk_ratio = tp_pips / sl_pips if sl_pips > 0 else 0

        if reward_risk_ratio < self.config.min_reward_risk_ratio:
            logger.warning(
                f"Reward/risk ratio {reward_risk_ratio:.2f} below minimum "
                f"{self.config.min_reward_risk_ratio}"
            )

        pip_value = self._get_pip_value(symbol)

        if sl_pips > 0 and pip_value > 0:
            volume = risk_amount / (sl_pips * pip_value)
        else:
            volume = 0.01

        if atr:
            sl_atr_pips = atr * self.config.atr_multiplier_sl / pip_size
            tp_atr_pips = atr * self.config.atr_multiplier_tp / pip_size
            logger.debug(f"ATR-based levels: SL={sl_atr_pips:.1f} pips, TP={tp_atr_pips:.1f} pips")

        volume = max(volume, 0.01)

        max_volume = (self.balance / 1000.0) * 0.5
        volume = min(volume, max_volume)

        potential_reward = volume * tp_pips * pip_value

        return PositionSize(
            volume=round(volume, 2),
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            risk_amount=risk_amount,
            potential_reward=potential_reward,
            reward_risk_ratio=reward_risk_ratio
        )

    def _conviction_adjusted_risk(self, conviction: float, base_risk: float) -> float:
        CONVICTION_RISK_MAP = [
            (0.0, 0.0),
            (0.2, 0.3),
            (0.3, 0.5),
            (0.4, 0.7),
            (0.5, 1.0),
            (0.6, 1.2),
            (0.7, 1.5),
            (0.8, 2.0),
            (0.9, 2.5),
            (1.0, 3.0),
        ]
        for threshold, mult in reversed(CONVICTION_RISK_MAP):
            if conviction >= threshold:
                return base_risk * mult
        return base_risk * 0.3

    def calculate_sl_tp(self, entry_price: float, direction: str,
                         atr: float, rating: CandleRating) -> tuple[float, float]:
        """Calculate stop loss and take profit based on ATR and candle rating

        Args:
            entry_price: Entry price
            direction: "buy" or "sell"
            atr: Average True Range value
            rating: Candle rating for confirmation

        Returns:
            (stop_loss, take_profit)
        """
        pip_size = 0.01  # XAUUSDm/XAGUSDm pip size

        # Base SL on ATR multiplier
        sl_distance = atr * self.config.atr_multiplier_sl
        tp_distance = atr * self.config.atr_multiplier_tp

        # Adjust based on candle rating
        rating_multiplier = {
            ClosureRating.A_PLUS: 0.8,   # Tighter SL for strong signals
            ClosureRating.A_MINUS: 1.0,
            ClosureRating.B: 1.2,
            ClosureRating.C: 1.5,
            ClosureRating.D: 2.0,
            ClosureRating.F: 2.5,
        }.get(rating.rating, 1.0)

        sl_distance *= rating_multiplier
        tp_distance *= rating_multiplier

        if direction == "buy":
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
        else:
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance

        return sl, tp

    def record_trade(self, profit: float):
        """Record a completed trade for daily tracking"""
        self.reset_daily()
        if profit < 0:
            self.daily_loss += abs(profit)  # Accumulate absolute loss
        self.trades_today += 1

    def _get_pip_size(self, symbol: str) -> float:
        return pip_size(symbol)

    def _get_pip_value(self, symbol: str) -> float:
        s = symbol.upper()
        if "XAU" in s or s == "GOLD":
            return 10.0
        if "XAG" in s or s == "SILVER":
            return 50.0
        if s in ("NAS100", "US100", "NDX", "DJI30", "US30", "SPX500"):
            return 1.0
        return 10.0


def calculate_atr(candles: list, period: int = 14) -> float:
    """Calculate Average True Range

    Args:
        candles: List of CandleData (oldest first)
        period: ATR period

    Returns:
        ATR value
    """
    if len(candles) < period + 1:
        return 0.0

    true_ranges = []
    for i in range(1, len(candles)):
        high = candles[i].high
        low = candles[i].low
        prev_close = candles[i-1].close

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        true_ranges.append(tr)

    return np.mean(true_ranges[-period:])

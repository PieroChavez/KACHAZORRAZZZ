"""IFVG Signal Generator
Combines Candle Closure Ratings with multi-timeframe confluence
"""
from dataclasses import dataclass
from typing import Optional
from enum import Enum
from loguru import logger

from .candle_closure_ratings import (
    CandleRating,
    ClosureRating,
    calculate_confluence,
    is_strong_bullish,
    is_strong_bearish,
    detect_trend
)
from .multi_timeframe import TimeframeData


class SignalDirection(Enum):
    """Signal direction"""
    BUY = "buy"
    SELL = "sell"
    NEUTRAL = "neutral"


@dataclass
class TradingSignal:
    """Generated trading signal"""
    direction: SignalDirection
    symbol: str
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    confidence: float  # 0.0 to 1.0
    confluence_score: int
    rating_summary: str
    reason: str
    timestamp: str


class IFVGSignalGenerator:
    """IFVG Strategy Signal Generator

    Generates trading signals based on:
    1. Candle Closure Ratings across multiple timeframes
    2. Multi-timeframe confluence
    3. Trend detection
    """

    def __init__(self, min_confluence: int = 2, trend_filter: bool = True):
        self.min_confluence = min_confluence
        self.trend_filter = trend_filter

    def generate_signal(self,
                       h4_data: TimeframeData,
                       h1_data: TimeframeData,
                       m15_data: TimeframeData) -> TradingSignal:
        """Generate a trading signal from multi-timeframe data

        Args:
            h4_data: H4 timeframe data
            h1_data: H1 timeframe data
            m15_data: M15 timeframe data

        Returns:
            TradingSignal with entry levels and confidence
        """
        # Get latest ratings from each timeframe
        h4_rating = h4_data.ratings[0] if h4_data.ratings else None
        h1_rating = h1_data.ratings[0] if h1_data.ratings else None
        m15_rating = m15_data.ratings[0] if m15_data.ratings else None

        if not all([h4_rating, h1_rating, m15_rating]):
            return TradingSignal(
                direction=SignalDirection.NEUTRAL,
                symbol="XAUUSDm",
                entry_price=None,
                stop_loss=None,
                take_profit=None,
                confidence=0.0,
                confluence_score=0,
                rating_summary="",
                reason="Insufficient data",
                timestamp=""
            )

        # Calculate confluence
        confluence = calculate_confluence(
            h4_data.ratings,
            h1_data.ratings,
            m15_data.ratings,
            min_alignment=self.min_confluence
        )

        # Detect trend on H4
        trend = detect_trend(h4_data.ratings, period=20)

        # Generate rating summary
        rating_summary = (
            f"H4:{h4_rating.rating.value} "
            f"H1:{h1_rating.rating.value} "
            f"M15:{m15_rating.rating.value}"
        )

        # Determine signal direction based on confluence and trend
        direction = SignalDirection.NEUTRAL
        confidence = 0.0
        reason = ""

        if confluence["direction"] is None:
            reason = f"Weak confluence ({confluence['confluence']}/{self.min_confluence})"
        elif confluence["strength"] == "strong":
            direction = SignalDirection.BUY if confluence["direction"] == "bullish" else SignalDirection.SELL
            confidence = 0.9
            reason = f"Strong {confluence['direction']} signal"
        elif confluence["strength"] == "moderate":
            direction = SignalDirection.BUY if confluence["direction"] == "bullish" else SignalDirection.SELL
            confidence = 0.7
            reason = f"Moderate {confluence['direction']} signal"

        # Apply trend filter if enabled
        if self.trend_filter and direction != SignalDirection.NEUTRAL:
            if direction == SignalDirection.BUY and trend == "bearish":
                confidence *= 0.5
                reason += " (counter-trend, reduced confidence)"
            elif direction == SignalDirection.SELL and trend == "bullish":
                confidence *= 0.5
                reason += " (counter-trend, reduced confidence)"

        # Calculate entry, SL, TP based on latest candle
        latest = m15_rating.candle
        entry_price = latest.close
        pip_size = 0.01

        if direction == SignalDirection.BUY:
            # For bullish: entry at close, SL below low, TP above
            stop_loss = latest.low - (latest.high - latest.low) * 0.5
            take_profit = entry_price + (entry_price - stop_loss) * 2
        elif direction == SignalDirection.SELL:
            stop_loss = latest.high + (latest.high - latest.low) * 0.5
            take_profit = entry_price - (stop_loss - entry_price) * 2
        else:
            stop_loss = None
            take_profit = None

        return TradingSignal(
            direction=direction,
            symbol="XAUUSDm",
            entry_price=entry_price if direction != SignalDirection.NEUTRAL else None,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            confluence_score=confluence["confluence"],
            rating_summary=rating_summary,
            reason=reason,
            timestamp=latest.timestamp.isoformat()
        )


def filter_signals_by_rating(signal: TradingSignal,
                            min_rating: ClosureRating = ClosureRating.B) -> bool:
    """Filter signals based on minimum candle rating

    Args:
        signal: Trading signal to filter
        min_rating: Minimum acceptable rating

    Returns:
        True if signal passes filter
    """
    # This would check the individual timeframe ratings
    # For now, accept signals with confidence > 0.6
    return signal.confidence >= 0.6

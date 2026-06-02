"""Candle Closure Rating System
Based on the Candle Closure Ratings PDF - XAUUSD trading strategy
"""
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np


class ClosureRating(Enum):
    """Closure rating grades"""
    A_PLUS = "A+"
    A_MINUS = "A-"
    B = "B"
    C = "C"
    D = "D"
    F = "F"  # Reversal zone


@dataclass
class CandleData:
    """OHLCV candle data"""
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class CandleRating:
    """Candle with its closure rating"""
    candle: CandleData
    rating: ClosureRating
    closure_pct: float
    body_pct: float
    is_bullish: bool
    is_reversal: bool = False


def calculate_closure_pct(candle: CandleData) -> float:
    """Calculate closure percentage of the range.

    Returns value between 0.0 and 1.0:
    - 1.0 = closed at the very top of the range
    - 0.0 = closed at the very bottom of the range
    """
    range_size = candle.high - candle.low
    if range_size == 0:
        return 0.5

    # Distance from low to close, normalized
    closure_pct = (candle.close - candle.low) / range_size
    return closure_pct


def calculate_body_pct(candle: CandleData) -> float:
    """Calculate body size as percentage of total range"""
    range_size = candle.high - candle.low
    if range_size == 0:
        return 0.0

    body_size = abs(candle.close - candle.open)
    return body_size / range_size


def determine_rating(closure_pct: float, is_bullish: bool) -> ClosureRating:
    """Determine the closure rating based on closure percentage"""
    # Handle reversal: close near bottom on bullish candle or near top on bearish
    if is_bullish and closure_pct < 0.30:
        return ClosureRating.F
    if not is_bullish and closure_pct > 0.70:
        return ClosureRating.F

    # Standard rating thresholds
    if closure_pct >= 0.90:
        return ClosureRating.A_PLUS
    elif closure_pct >= 0.70:
        return ClosureRating.A_MINUS
    elif closure_pct >= 0.50:
        return ClosureRating.B
    elif closure_pct >= 0.30:
        return ClosureRating.C
    else:
        return ClosureRating.D


def rate_candle(candle: CandleData) -> CandleRating:
    """Rate a single candle using closure rating system"""
    closure_pct = calculate_closure_pct(candle)
    body_pct = calculate_body_pct(candle)
    is_bullish = candle.close > candle.open

    # Check for reversal (closed in shadow zone)
    is_reversal = (is_bullish and closure_pct < 0.30) or (
        not is_bullish and closure_pct > 0.70
    )

    rating = determine_rating(closure_pct, is_bullish)

    return CandleRating(
        candle=candle,
        rating=rating,
        closure_pct=closure_pct,
        body_pct=body_pct,
        is_bullish=is_bullish,
        is_reversal=is_reversal
    )


def rate_candles(candles: list[CandleData]) -> list[CandleRating]:
    """Rate multiple candles"""
    return [rate_candle(c) for c in candles]


def is_strong_bullish(rating: CandleRating) -> bool:
    """Check if rating indicates strong bullish momentum"""
    return rating.rating in (ClosureRating.A_PLUS, ClosureRating.A_MINUS) and rating.is_bullish


def is_strong_bearish(rating: CandleRating) -> bool:
    """Check if rating indicates strong bearish momentum"""
    return rating.rating in (ClosureRating.A_PLUS, ClosureRating.A_MINUS) and not rating.is_bullish


def get_rating_score(rating: ClosureRating, direction: str = "bullish") -> int:
    """Get numerical score for a rating (for confluence calculation)"""
    scores = {
        ClosureRating.A_PLUS: 4,
        ClosureRating.A_MINUS: 3,
        ClosureRating.B: 2,
        ClosureRating.C: 1,
        ClosureRating.D: 0,
        ClosureRating.F: -2,
    }
    score = scores[rating]

    # Reverse score for bearish direction
    if direction == "bearish":
        return -score
    return score


def detect_trend(ratings: list[CandleRating], period: int = 20) -> str:
    """Detect trend direction based on recent ratings

    Args:
        ratings: List of recent candle ratings (oldest first)
        period: Number of candles to consider

    Returns:
        "bullish", "bearish", or "ranging"
    """
    if len(ratings) < period:
        period = len(ratings)

    recent = ratings[-period:]

    bullish_count = sum(1 for r in recent if r.is_bullish)
    bearish_count = len(recent) - bullish_count

    bullish_ratio = bullish_count / len(recent)

    if bullish_ratio > 0.65:
        return "bullish"
    elif bullish_ratio < 0.35:
        return "bearish"
    else:
        return "ranging"


def calculate_confluence(h4_ratings: list[CandleRating],
                         h1_ratings: list[CandleRating],
                         m15_ratings: list[CandleRating],
                         min_alignment: int = 2) -> dict:
    """Calculate multi-timeframe confluence of signals

    Args:
        h4_ratings: H4 timeframe ratings (most recent first)
        h1_ratings: H1 timeframe ratings (most recent first)
        m15_ratings: M15 timeframe ratings (most recent first)
        min_alignment: Minimum timeframes that must agree

    Returns:
        Dict with confluence score and direction
    """
    # Get latest rating from each timeframe
    h4_latest = h4_ratings[0] if h4_ratings else None
    h1_latest = h1_ratings[0] if h1_ratings else None
    m15_latest = m15_ratings[0] if m15_ratings else None

    if not all([h4_latest, h1_latest, m15_latest]):
        return {"confluence": 0, "direction": None, "strength": "weak"}

    # Count bullish/bearish alignment
    alignments = {
        "bullish": 0,
        "bearish": 0
    }

    for rating in [h4_latest, h1_latest, m15_latest]:
        if is_strong_bullish(rating):
            alignments["bullish"] += 1
        elif is_strong_bearish(rating):
            alignments["bearish"] += 1

    # Determine confluence
    max_alignment = max(alignments["bullish"], alignments["bearish"])
    direction = "bullish" if alignments["bullish"] > alignments["bearish"] else "bearish"

    # Map confluence strength
    if max_alignment >= 3:
        strength = "strong"
    elif max_alignment >= 2:
        strength = "moderate"
    else:
        strength = "weak"

    return {
        "confluence": max_alignment,
        "direction": direction if max_alignment >= min_alignment else None,
        "strength": strength,
        "h4_rating": h4_latest.rating.value,
        "h1_rating": h1_latest.rating.value,
        "m15_rating": m15_latest.rating.value,
    }

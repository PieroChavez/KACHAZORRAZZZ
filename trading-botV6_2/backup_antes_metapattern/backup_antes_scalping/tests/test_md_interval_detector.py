"""Tests for IntervalDetector — MD Interval (Interactive Point) pattern."""
import numpy as np
import pandas as pd
import pytest

from src.core.md_concepts import MDConcept, MDDetection
from src.core.md_interval_detector import (
    IntervalDetector,
    IntervalResult,
    MAX_INTERVAL_RANGE_ATR,
    MIN_VOLUME_SPIKE,
    MIN_BODY_RATIO_ANATOMY,
)

N_CANDLES = 70


def _make_df(opens, closes, highs, lows, volumes=None):
    df = pd.DataFrame({
        "open": opens,
        "close": closes,
        "high": highs,
        "low": lows,
    })
    if volumes is not None:
        df["tick_volume"] = volumes
    return df


@pytest.fixture
def detector():
    return IntervalDetector(lookback=60)


@pytest.fixture
def df_short():
    n = 10
    return pd.DataFrame({
        "open": [100.0] * n,
        "close": [100.0] * n,
        "high": [100.3] * n,
        "low": [99.7] * n,
    })


def _neutral_background(n=N_CANDLES):
    opens = [100.0] * n
    closes = [100.0] * n
    highs = [102.0] * n
    lows = [100.0] * n
    vols = [100] * n
    return opens, closes, highs, lows, vols


@pytest.fixture
def df_bullish_interval():
    """70 candles with a bullish interval at index 60."""
    opens, closes, highs, lows, vols = _neutral_background()

    opens[60] = 100.2
    closes[60] = 100.5
    highs[60] = 102.5
    lows[60] = 99.8
    vols[60] = 200

    opens[61] = 103.0
    closes[61] = 105.0
    highs[61] = 105.5
    lows[61] = 102.0
    vols[61] = 250

    return _make_df(opens, closes, highs, lows, vols)


@pytest.fixture
def df_bearish_interval():
    """70 candles with a bearish interval at index 60."""
    opens, closes, highs, lows, vols = _neutral_background()

    opens[60] = 100.2
    closes[60] = 99.9
    highs[60] = 102.5
    lows[60] = 99.5
    vols[60] = 200

    opens[61] = 99.8
    closes[61] = 97.0
    highs[61] = 100.0
    lows[61] = 96.5
    vols[61] = 250

    return _make_df(opens, closes, highs, lows, vols)


@pytest.fixture
def df_bad_anatomy():
    """Interval candle where body > upper_wick — violates MD anatomy rule."""
    opens, closes, highs, lows, vols = _neutral_background()

    opens[60] = 100.0
    closes[60] = 101.5
    highs[60] = 102.0
    lows[60] = 99.0
    vols[60] = 200

    opens[61] = 103.0
    closes[61] = 105.0
    highs[61] = 105.5
    lows[61] = 102.0
    vols[61] = 250

    return _make_df(opens, closes, highs, lows, vols)


@pytest.fixture
def df_low_volume():
    """Volume spike too low to qualify as interval."""
    opens, closes, highs, lows, vols = _neutral_background()

    opens[60] = 100.2
    closes[60] = 100.5
    highs[60] = 102.5
    lows[60] = 99.8
    vols[60] = 110

    opens[61] = 103.0
    closes[61] = 105.0
    highs[61] = 105.5
    lows[61] = 102.0
    vols[61] = 110

    return _make_df(opens, closes, highs, lows, vols)


@pytest.fixture
def df_large_range():
    """Interval candle range exceeds MAX_INTERVAL_RANGE_ATR * ATR."""
    opens, closes, highs, lows, vols = _neutral_background()

    opens[60] = 100.2
    closes[60] = 100.5
    highs[60] = 106.0
    lows[60] = 99.0
    vols[60] = 200

    opens[61] = 103.0
    closes[61] = 105.0
    highs[61] = 105.5
    lows[61] = 102.0
    vols[61] = 250

    return _make_df(opens, closes, highs, lows, vols)


@pytest.fixture
def df_bullish_fvg_interval():
    """Bullish interval with a clear FVG gap left by the breakout candle."""
    opens, closes, highs, lows, vols = _neutral_background()

    opens[60] = 100.2
    closes[60] = 100.5
    highs[60] = 101.0
    lows[60] = 98.0
    vols[60] = 200

    opens[61] = 104.0
    closes[61] = 106.0
    highs[61] = 107.0
    lows[61] = 103.0
    vols[61] = 250

    return _make_df(opens, closes, highs, lows, vols)


class TestIntervalDetector:
    def test_not_found_none(self, detector):
        result = detector.detect(None)
        assert not result.found
        assert result.direction == "NEUTRAL"
        assert result.confidence == 0.0

    def test_not_found_empty_dataframe(self, detector):
        result = detector.detect(pd.DataFrame())
        assert not result.found
        assert result.direction == "NEUTRAL"
        assert result.confidence == 0.0

    def test_not_found_insufficient_data(self, detector, df_short):
        result = detector.detect(df_short)
        assert not result.found
        assert result.interval_index is None

    def test_bullish_interval_detected(self, detector, df_bullish_interval):
        result = detector.detect(df_bullish_interval)
        assert result.found
        assert result.direction == "BUY"
        assert result.confidence > 0.0
        assert result.interval_index == 60
        assert result.body_size == pytest.approx(0.3)
        assert result.volume_ratio >= MIN_VOLUME_SPIKE
        assert result.upper_wick > result.body_size
        assert result.lower_wick > result.body_size
        assert result.body_ratio < MIN_BODY_RATIO_ANATOMY

    def test_bearish_interval_detected(self, detector, df_bearish_interval):
        result = detector.detect(df_bearish_interval)
        assert result.found
        assert result.direction == "SELL"
        assert result.confidence > 0.0
        assert result.interval_index == 60
        assert result.body_size > 0

    def test_no_interval_when_body_exceeds_wick(self, detector, df_bad_anatomy):
        result = detector.detect(df_bad_anatomy)
        assert not result.found

    def test_no_interval_when_volume_too_low(self, detector, df_low_volume):
        result = detector.detect(df_low_volume)
        assert not result.found

    def test_no_interval_when_range_too_large(self, detector, df_large_range):
        result = detector.detect(df_large_range)
        assert not result.found

    def test_to_detection_returns_none_when_not_found(self, detector, df_short):
        result = detector.detect(df_short)
        assert result.to_detection() is None

    def test_to_detection_returns_md_detection_when_found(
        self, detector, df_bullish_interval
    ):
        result = detector.detect(df_bullish_interval)
        detection = result.to_detection()
        assert detection is not None
        assert isinstance(detection, MDDetection)
        assert detection.concept == MDConcept.INTERVAL
        assert detection.direction == "BUY"
        assert detection.confidence == result.confidence
        assert detection.suggested_price == result.entry_price
        assert detection.timeframe == "M15"
        assert detection.metadata["interval_high"] == result.interval_high
        assert detection.metadata["interval_low"] == result.interval_low
        assert detection.metadata["body_ratio"] == result.body_ratio
        assert detection.metadata["volume_ratio"] == result.volume_ratio
        assert detection.metadata["fvg_detected"] == result.fvg_detected

    def test_fvg_detected_on_bullish_gap(self, detector, df_bullish_fvg_interval):
        result = detector.detect(df_bullish_fvg_interval)
        assert result.found
        assert result.fvg_detected
        assert result.direction == "BUY"

    def test_volume_ratio_returns_one_with_no_volume_column(self, detector):
        df = pd.DataFrame({
            "open": [100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0],
            "high": [100.3, 100.3, 100.3],
            "low": [99.7, 99.7, 99.7],
        })
        ratio = detector._volume_ratio(df, 1)
        assert ratio == 1.0

    def test_volume_ratio_with_volume_column(self, detector):
        df = pd.DataFrame({
            "open": [100.0] * 7,
            "close": [100.0] * 7,
            "high": [100.3] * 7,
            "low": [99.7] * 7,
            "tick_volume": [100, 100, 100, 100, 100, 200, 100],
        })
        ratio = detector._volume_ratio(df, 5)
        assert ratio == pytest.approx(2.0)

    def test_entry_price_is_midpoint_of_interval_range(
        self, detector, df_bullish_interval
    ):
        result = detector.detect(df_bullish_interval)
        expected_entry = (result.interval_high + result.interval_low) / 2.0
        assert result.entry_price == expected_entry

    def test_confidence_includes_volume_and_body_components(
        self, detector, df_bullish_interval
    ):
        result = detector.detect(df_bullish_interval)
        assert result.confidence >= 0.4
        assert result.confidence <= 0.90

    def test_interval_result_dataclass_defaults(self):
        result = IntervalResult(found=False, direction="NEUTRAL", confidence=0.0)
        assert result.body_size == 0.0
        assert result.volume_ratio == 1.0
        assert result.fvg_detected is False
        assert result.notes == []

    def test_not_found_with_atr_zero(self, detector):
        df = pd.DataFrame({
            "open": [100.0] * 70,
            "close": [100.0] * 70,
            "high": [100.0] * 70,
            "low": [100.0] * 70,
        })
        result = detector.detect(df)
        assert not result.found

    def test_to_detection_without_entry_price_in_result(self):
        result = IntervalResult(found=True, direction="BUY", confidence=0.5)
        result.entry_price = None
        assert result.to_detection() is None

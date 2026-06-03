import numpy as np
import pandas as pd
import pytest

from src.core.md_concepts import MDConcept, MDDetection
from src.core.md_price_capture_detector import (
    MAX_DOJI_BODY_RATIO,
    MIN_BREAKOUT_BODY_RATIO,
    MIN_BREAKOUT_MULTIPLIER,
    MIN_VOLUME_SPIKE,
    PriceCaptureDetector,
    PriceCaptureResult,
)


def _make_base(n: int = 65) -> pd.DataFrame:
    opens = np.full(n, 100.0)
    closes = np.full(n, 101.0)
    highs = np.full(n, 101.5)
    lows = np.full(n, 99.5)
    return pd.DataFrame({"open": opens, "close": closes, "high": highs, "low": lows})


def _with_volume(df: pd.DataFrame, vol_col: str = "tick_volume") -> pd.DataFrame:
    df = df.copy()
    df[vol_col] = 100.0
    return df


def _set_candle(df: pd.DataFrame, idx: int, o: float, c: float, h: float, l: float):
    df.loc[idx, "open"] = o
    df.loc[idx, "close"] = c
    df.loc[idx, "high"] = h
    df.loc[idx, "low"] = l


class TestPriceCaptureDetector:
    """Tests for PriceCaptureDetector and PriceCaptureResult."""

    def test_not_found_none(self):
        detector = PriceCaptureDetector(lookback=60)
        result = detector.detect(None)
        assert not result.found
        assert result.direction == "NEUTRAL"
        assert result.confidence == 0.0

    def test_not_found_insufficient_data(self):
        detector = PriceCaptureDetector(lookback=60)
        df = _make_base(n=59)
        result = detector.detect(df)
        assert not result.found
        result2 = detector.detect(pd.DataFrame())
        assert not result2.found
        result3 = detector.detect(pd.DataFrame(columns=["open", "close", "high", "low"]))
        assert not result3.found

    def test_bullish_price_capture(self):
        df = _with_volume(_make_base(65))
        _set_candle(df, 63, 100.00, 100.01, 100.30, 99.70)
        _set_candle(df, 64, 100.00, 101.50, 101.60, 99.80)
        df.loc[64, "tick_volume"] = 130.0

        detector = PriceCaptureDetector(lookback=60)
        result = detector.detect(df)

        assert result.found
        assert result.direction == "BUY"
        assert result.entry_price == pytest.approx(99.70)
        assert result.doji_index == 63
        assert result.breakout_index == 64
        assert result.confidence > 0.0
        assert not result.fvg_detected

    def test_bearish_price_capture(self):
        df = _with_volume(_make_base(65))
        _set_candle(df, 63, 100.00, 99.99, 100.30, 99.70)
        _set_candle(df, 64, 101.00, 99.50, 101.20, 99.40)
        df.loc[64, "tick_volume"] = 130.0

        detector = PriceCaptureDetector(lookback=60)
        result = detector.detect(df)

        assert result.found
        assert result.direction == "SELL"
        assert result.entry_price == pytest.approx(100.30)
        assert result.doji_index == 63
        assert result.breakout_index == 64

    def test_no_capture_doji_body_ratio_too_high(self):
        df = _with_volume(_make_base(65))
        _set_candle(df, 63, 100.00, 100.20, 100.30, 99.70)
        _set_candle(df, 64, 100.00, 101.50, 101.60, 99.80)
        df.loc[64, "tick_volume"] = 130.0

        detector = PriceCaptureDetector(lookback=60)
        result = detector.detect(df)

        assert not result.found

    def test_no_capture_breakout_body_ratio_too_low(self):
        df = _with_volume(_make_base(65))
        _set_candle(df, 63, 100.00, 100.01, 100.30, 99.90)
        _set_candle(df, 64, 100.00, 100.30, 100.80, 99.80)
        df.loc[64, "tick_volume"] = 130.0

        detector = PriceCaptureDetector(lookback=60)
        result = detector.detect(df)

        assert not result.found

    def test_no_capture_breakout_multiplier_too_low(self):
        df = _with_volume(_make_base(65))
        _set_candle(df, 63, 100.00, 100.01, 100.80, 99.20)
        _set_candle(df, 64, 100.00, 101.20, 101.30, 100.00)
        df.loc[64, "tick_volume"] = 130.0

        detector = PriceCaptureDetector(lookback=60)
        result = detector.detect(df)

        assert not result.found

    def test_no_capture_volume_spike_too_low(self):
        df = _with_volume(_make_base(65))
        _set_candle(df, 63, 100.00, 100.01, 100.30, 99.70)
        _set_candle(df, 64, 100.00, 101.50, 101.60, 99.80)
        df.loc[64, "tick_volume"] = 110.0

        detector = PriceCaptureDetector(lookback=60)
        result = detector.detect(df)

        assert not result.found

    def test_to_detection_returns_none_when_not_found(self):
        result = PriceCaptureResult(found=False, direction="NEUTRAL", confidence=0.0)
        assert result.to_detection() is None

        result2 = PriceCaptureResult(found=True, direction="BUY", confidence=0.0, entry_price=None)
        assert result2.to_detection() is None

    def test_to_detection_returns_md_detection_when_found(self):
        result = PriceCaptureResult(
            found=True,
            direction="BUY",
            confidence=0.45,
            entry_price=99.70,
            doji_high=100.30,
            doji_low=99.70,
            breakout_multiplier=3.0,
            fvg_detected=False,
        )
        detection = result.to_detection()

        assert detection is not None
        assert detection.concept == MDConcept.PRICE_CAPTURE
        assert detection.direction == "BUY"
        assert detection.confidence == 0.45
        assert detection.suggested_price == 99.70
        assert detection.metadata["doji_high"] == 100.30
        assert detection.metadata["doji_low"] == 99.70
        assert detection.metadata["breakout_multiplier"] == 3.0
        assert detection.metadata["fvg_detected"] is False

    def test_fvg_detected_when_gap_before_doji(self):
        df = _with_volume(_make_base(65))
        _set_candle(df, 62, 100.00, 100.00, 99.50, 99.00)
        _set_candle(df, 63, 100.00, 100.01, 100.30, 100.00)
        _set_candle(df, 64, 100.00, 101.50, 101.60, 99.80)
        df.loc[64, "tick_volume"] = 130.0

        detector = PriceCaptureDetector(lookback=60)
        result = detector.detect(df)

        assert result.found
        assert result.fvg_detected

    def test_volume_ratio_returns_one_without_volume_column(self):
        df = _make_base(65)
        detector = PriceCaptureDetector(lookback=60)
        ratio = detector._volume_ratio(df, 10)
        assert ratio == 1.0

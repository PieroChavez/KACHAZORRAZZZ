import numpy as np
import pandas as pd
import pytest

from src.core.md_concepts import MDConcept
from src.core.md_pod_detector import PODDetector, PODResult


def _noise_df(n_rows: int = 70, base: float = 100.0,
              half_range: float = 0.025) -> pd.DataFrame:
    """Generate n_rows DOJI candles (open == close) with tiny high-low spread.
    Provides a blank canvas for injecting specific OB/TB patterns."""
    opens = [float(base)] * n_rows
    closes = [float(base)] * n_rows
    highs = [float(base + half_range)] * n_rows
    lows = [float(base - half_range)] * n_rows
    return pd.DataFrame({
        "open": opens,
        "close": closes,
        "high": highs,
        "low": lows,
    })


class TestPODDetector:
    """PODDetector.detect() and PODResult.to_detection() unit tests."""

    @pytest.fixture
    def detector(self) -> PODDetector:
        return PODDetector()

    # ---------------------------------------------------------------
    # 1. None / empty DataFrame
    # ---------------------------------------------------------------
    def test_detect_none(self, detector):
        result = detector.detect(None)
        assert not result.found
        assert result.direction == "NEUTRAL"
        assert result.confidence == 0.0

    def test_detect_empty_dataframe(self, detector):
        result = detector.detect(pd.DataFrame())
        assert not result.found
        assert result.direction == "NEUTRAL"

    # ---------------------------------------------------------------
    # 2. Insufficient data (< lookback)
    # ---------------------------------------------------------------
    def test_detect_insufficient_data(self, detector):
        df = _noise_df(n_rows=10)
        result = detector.detect(df)
        assert not result.found
        assert result.direction == "NEUTRAL"

    # ---------------------------------------------------------------
    # 3. Bullish OB -> Bearish TB -> POD gap (SELL direction)
    # ---------------------------------------------------------------
    def test_bullish_ob_bearish_tb_pod(self, detector):
        df = _noise_df(n_rows=70)
        # OB at 62: strong bullish candle
        df.at[62, "open"] = 100.0
        df.at[62, "close"] = 102.0
        df.at[62, "high"] = 102.3
        df.at[62, "low"] = 99.7
        # TB at 63: bearish candle covering >= 60 % of OB body
        df.at[63, "open"] = 101.2
        df.at[63, "close"] = 99.2
        df.at[63, "high"] = 101.5
        df.at[63, "low"] = 98.9

        result = detector.detect(df)

        assert result.found, f"POD not found: {result.notes}"
        assert result.direction == "SELL"
        assert result.ob_index == 62
        assert result.tb_index == 63
        assert result.cover_ratio >= 0.6
        assert 0.3 <= result.confidence <= 1.0
        assert result.pod_50_price is not None
        assert result.pod_low is not None
        assert result.pod_high is not None

    # ---------------------------------------------------------------
    # 4. Bearish OB -> Bullish TB -> POD gap (BUY direction)
    # ---------------------------------------------------------------
    def test_bearish_ob_bullish_tb_pod(self, detector):
        df = _noise_df(n_rows=70)
        # OB at 62: strong bearish candle
        df.at[62, "open"] = 102.0
        df.at[62, "close"] = 100.0
        df.at[62, "high"] = 102.3
        df.at[62, "low"] = 99.7
        # TB at 63: bullish candle covering >= 60 % of OB body
        df.at[63, "open"] = 100.8
        df.at[63, "close"] = 102.0
        df.at[63, "high"] = 102.3
        df.at[63, "low"] = 100.5

        result = detector.detect(df)

        assert result.found, f"POD not found: {result.notes}"
        assert result.direction == "BUY"
        assert result.ob_index == 62
        assert result.tb_index == 63
        assert result.cover_ratio >= 0.6
        assert 0.3 <= result.confidence <= 1.0
        assert result.pod_50_price is not None

    # ---------------------------------------------------------------
    # 5. OB body ratio < MIN_BODY_RATIO (0.4)
    # ---------------------------------------------------------------
    def test_no_pod_low_ob_body_ratio(self, detector):
        df = _noise_df(n_rows=70)
        # OB at 62: body = 2, range = 6.3 -> ratio = 0.32 < 0.4
        df.at[62, "open"] = 100.0
        df.at[62, "close"] = 102.0
        df.at[62, "high"] = 106.0
        df.at[62, "low"] = 99.7
        # TB at 63: would otherwise be valid
        df.at[63, "open"] = 101.2
        df.at[63, "close"] = 99.2
        df.at[63, "high"] = 101.5
        df.at[63, "low"] = 98.9

        result = detector.detect(df)
        assert not result.found

    # ---------------------------------------------------------------
    # 6. Cover ratio < min_cover_ratio (0.6)
    # ---------------------------------------------------------------
    def test_no_pod_low_cover_ratio(self, detector):
        df = _noise_df(n_rows=70)
        # OB at 62: bullish, body = 2.0
        df.at[62, "open"] = 100.0
        df.at[62, "close"] = 102.0
        df.at[62, "high"] = 102.3
        df.at[62, "low"] = 99.7
        # TB at 63: bearish, minimal overlap -> cover = 0.4 < 0.6
        df.at[63, "open"] = 101.8
        df.at[63, "close"] = 101.0
        df.at[63, "high"] = 102.1
        df.at[63, "low"] = 100.7

        result = detector.detect(df)
        assert not result.found

    # ---------------------------------------------------------------
    # 7. POD height exceeds max_pod_height_atr (2.0 * ATR)
    # ---------------------------------------------------------------
    def test_no_pod_excessive_height(self, detector):
        # OB/TB placed at indices 30/31 so they do NOT inflate the ATR
        # window (last 14 candles, indices 56-69, are all noise).
        df = _noise_df(n_rows=70)
        # OB at 30: large bullish body = 3.0
        df.at[30, "open"] = 100.0
        df.at[30, "close"] = 103.0
        df.at[30, "high"] = 103.3
        df.at[30, "low"] = 99.7
        # TB at 31: bearish with cover_ratio = 0.67
        df.at[31, "open"] = 102.4
        df.at[31, "close"] = 100.4
        df.at[31, "high"] = 102.7
        df.at[31, "low"] = 100.1
        # POD height = 0.6; noise ATR ~ 0.05 -> 2 * ATR ~ 0.10 -> 0.6 > 0.10

        result = detector.detect(df)
        assert not result.found, (
            f"Expected height rejection, got found={result.found} "
            f"notes={result.notes}"
        )

    # ---------------------------------------------------------------
    # 8. to_detection() returns None when not found / price missing
    # ---------------------------------------------------------------
    def test_to_detection_none_when_not_found(self):
        pod = PODResult(found=False, direction="NEUTRAL", confidence=0.0)
        assert pod.to_detection() is None

    def test_to_detection_none_when_pod_50_none(self):
        pod = PODResult(found=True, direction="SELL", confidence=0.7,
                        pod_50_price=None)
        assert pod.to_detection() is None

    # ---------------------------------------------------------------
    # 9. to_detection() returns MDDetection when found
    # ---------------------------------------------------------------
    def test_to_detection_returns_detection(self):
        pod = PODResult(
            found=True, direction="SELL", confidence=0.75,
            pod_50_price=100.5, pod_low=100.3, pod_high=100.7,
            cover_ratio=0.8,
        )
        det = pod.to_detection()
        assert det is not None
        assert det.concept == MDConcept.POD
        assert det.direction == "SELL"
        assert det.confidence == 0.75
        assert det.suggested_price == 100.5
        assert det.metadata["pod_low"] == 100.3
        assert det.metadata["pod_high"] == 100.7
        assert det.metadata["cover_ratio"] == 0.8

    # ---------------------------------------------------------------
    # 10. _relative_volume()
    # ---------------------------------------------------------------
    def test_relative_volume_no_volume_column(self, detector):
        df = _noise_df(n_rows=70)
        assert detector._relative_volume(df, idx=60) == 1.0

    def test_relative_volume_with_tick_volume(self):
        detector = PODDetector()
        df = _noise_df(n_rows=70)
        df["tick_volume"] = np.full(70, 100.0)
        df.loc[60, "tick_volume"] = 200.0
        rv = detector._relative_volume(df, idx=60)
        assert rv == 2.0

    # ---------------------------------------------------------------
    # 11. _calc_atr() returns 0 with insufficient data
    # ---------------------------------------------------------------
    def test_calc_atr_insufficient_data(self, detector):
        df = _noise_df(n_rows=5)
        assert detector._calc_atr(df) == 0.0

    def test_calc_atr_with_enough_data(self, detector):
        df = _noise_df(n_rows=70)
        assert detector._calc_atr(df) > 0.0

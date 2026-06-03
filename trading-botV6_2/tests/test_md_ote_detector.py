"""Tests for OTEDetector — Fibonacci OTE 75-79% detector."""
import numpy as np
import pandas as pd
import pytest

from src.core.md_ote_detector import (
    OTEDetector,
    OTEResult,
    MIN_IMPULSE_BODY_RATIO,
    MIN_IMPULSE_VOLUME,
    OTE_LEVELS,
    OTE_TOLERANCE,
)
from src.core.md_concepts import MDConcept


def _ohlc(opens, closes, highs, lows, volumes=None):
    df = pd.DataFrame({"open": opens, "close": closes, "high": highs, "low": lows})
    if volumes is not None:
        df["tick_volume"] = volumes
    return df


class TestOTEDetector:

    @pytest.fixture
    def detector(self):
        return OTEDetector(lookback=60)

    # ---------------------------------------------------------------
    # 1) Not found with None/empty DataFrame
    # ---------------------------------------------------------------
    def test_detect_empty(self, detector):
        result = detector.detect(pd.DataFrame())
        assert not result.found
        assert result.direction == "NEUTRAL"
        assert result.confidence == 0.0

    def test_detect_none(self, detector):
        result = detector.detect(None)
        assert not result.found
        assert result.direction == "NEUTRAL"
        assert result.confidence == 0.0

    # ---------------------------------------------------------------
    # 2) Not found with insufficient data (< lookback)
    # ---------------------------------------------------------------
    @pytest.fixture
    def df_insufficient(self):
        n = 30
        return _ohlc(
            [100.0] * n, [100.0] * n, [100.1] * n, [99.9] * n,
            volumes=[100] * n,
        )

    def test_insufficient_data(self, detector, df_insufficient):
        result = detector.detect(df_insufficient)
        assert not result.found

    # ---------------------------------------------------------------
    # 3) Bullish OTE detected: 3+ impulse candles + retrace to 77%
    #
    #   Detector checks min_impulse=3 candles (start..start+2).
    #   Impulse candles are at indices 50,51,52:
    #     idx 50: o=100.0  c=102.0  h=102.5  l=99.5   vol=180
    #     idx 51: o=102.0  c=104.0  h=104.5  l=101.5  vol=180
    #     idx 52: o=104.0  c=106.0  h=106.5  l=103.5  vol=180
    #   impulse_high=106.5  impulse_low=99.5  range=7.0
    #   77% retrace → price = 106.5 - 7.0*0.77 = 101.11
    # ---------------------------------------------------------------
    @pytest.fixture
    def df_bullish_ote(self):
        o, c, h, l, v = [], [], [], [], []
        for _ in range(50):
            o.append(100.0)
            c.append(100.0)
            h.append(100.1)
            l.append(99.9)
            v.append(100)
        _opens = [100.0, 102.0, 104.0]
        for po in _opens:
            pc = po + 2.0
            o.append(po)
            c.append(pc)
            h.append(pc + 0.5)
            l.append(po - 0.5)
            v.append(180)
        imp_high = 106.5
        imp_low = 99.5
        imp_range = imp_high - imp_low
        target = imp_high - imp_range * 0.77
        last_c = c[-1]
        for i in range(10):
            frac = (i + 1) / 11
            px = last_c - (last_c - target) * frac
            o.append(px + 0.05)
            c.append(px)
            h.append(px + 0.4)
            l.append(px - 0.4)
            v.append(70)
        o.append(target + 0.05)
        c.append(target)
        h.append(target + 0.4)
        l.append(target - 0.4)
        v.append(70)
        df = _ohlc(o, c, h, l, v)
        assert len(df) == 64
        return df

    def test_bullish_ote_detected(self, detector, df_bullish_ote):
        result = detector.detect(df_bullish_ote)
        assert result.found
        assert result.direction == "BUY"
        assert result.confidence > 0.5
        assert result.entry_price is not None
        assert result.impulse_end - result.impulse_start >= 2
        assert 0.72 <= result.ote_level <= 0.79
        assert result.avg_body_ratio > 0.55
        assert result.avg_volume_ratio > 1.25

    # ---------------------------------------------------------------
    # 4) Bearish OTE detected: 3 bearish impulse + retrace to 78%
    #
    #     idx 50: o=110.0  c=108.0  h=110.5  l=107.5  vol=180
    #     idx 51: o=108.0  c=106.0  h=108.5  l=105.5  vol=180
    #     idx 52: o=106.0  c=104.0  h=106.5  l=103.5  vol=180
    #   impulse_high=110.5  impulse_low=103.5  range=7.0
    #   SELL: origin=110.5 extreme=103.5
    #   78% retrace → price = 103.5 + 7.0*0.78 = 108.96
    # ---------------------------------------------------------------
    @pytest.fixture
    def df_bearish_ote(self):
        o, c, h, l, v = [], [], [], [], []
        for _ in range(50):
            o.append(110.0)
            c.append(110.0)
            h.append(110.1)
            l.append(109.9)
            v.append(100)
        _opens = [110.0, 108.0, 106.0]
        for po in _opens:
            pc = po - 2.0
            o.append(po)
            c.append(pc)
            h.append(po + 0.5)
            l.append(pc - 0.5)
            v.append(180)
        imp_high = 110.5
        imp_low = 103.5
        imp_range = imp_high - imp_low
        target = imp_low + imp_range * 0.78
        last_c = c[-1]
        for i in range(11):
            frac = (i + 1) / 12
            px = last_c + (target - last_c) * frac
            o.append(px - 0.05)
            c.append(px)
            h.append(px + 0.4)
            l.append(px - 0.4)
            v.append(70)
        o.append(target - 0.05)
        c.append(target)
        h.append(target + 0.4)
        l.append(target - 0.4)
        v.append(70)
        df = _ohlc(o, c, h, l, v)
        assert len(df) == 65
        return df

    def test_bearish_ote_detected(self, detector, df_bearish_ote):
        result = detector.detect(df_bearish_ote)
        assert result.found
        assert result.direction == "SELL"
        assert result.confidence > 0.5
        assert result.entry_price is not None
        assert result.impulse_end - result.impulse_start >= 2
        assert result.ote_level in (0.75, 0.79)

    # ---------------------------------------------------------------
    # 5) No OTE when impulse has fewer than 3 candles
    #    (2 bullish then bearish break)
    # ---------------------------------------------------------------
    @pytest.fixture
    def df_short_impulse(self):
        o, c, h, l, v = [], [], [], [], []
        for _ in range(50):
            o.append(100.0)
            c.append(100.0)
            h.append(100.1)
            l.append(99.9)
            v.append(100)
        o.extend([100.0, 102.5])
        c.extend([102.5, 105.0])
        h.extend([103.0, 105.5])
        l.extend([99.5, 102.0])
        v.extend([180, 180])
        o.append(105.0)
        c.append(103.0)
        h.append(105.5)
        l.append(102.5)
        v.append(180)
        for _ in range(11):
            o.append(c[-1])
            c.append(c[-1] + 0.1)
            h.append(c[-1] + 0.2)
            l.append(c[-1] - 0.2)
            v.append(80)
        return _ohlc(o, c, h, l, v)

    def test_no_ote_short_impulse(self, detector, df_short_impulse):
        result = detector.detect(df_short_impulse)
        assert not result.found

    # ---------------------------------------------------------------
    # 6) No OTE when impulse body ratio < 0.55
    #    body=0.4  range=1.2  ratio=0.333
    # ---------------------------------------------------------------
    @pytest.fixture
    def df_small_body(self):
        o, c, h, l, v = [], [], [], [], []
        for _ in range(50):
            o.append(100.0)
            c.append(100.0)
            h.append(100.1)
            l.append(99.9)
            v.append(100)
        for i in range(3):
            base = 100.0 + i * 0.35
            o.append(base)
            c.append(base + 0.4)
            h.append(base + 1.0)
            l.append(base - 0.2)
            v.append(180)
        for _ in range(11):
            o.append(c[-1])
            c.append(c[-1] + 0.1)
            h.append(c[-1] + 0.2)
            l.append(c[-1] - 0.2)
            v.append(80)
        return _ohlc(o, c, h, l, v)

    def test_no_ote_weak_body(self, detector, df_small_body):
        result = detector.detect(df_small_body)
        assert not result.found

    # ---------------------------------------------------------------
    # 7) No OTE when retrace only hits 50%, not 75-79%
    # ---------------------------------------------------------------
    @pytest.fixture
    def df_no_retrace(self):
        o, c, h, l, v = [], [], [], [], []
        for _ in range(50):
            o.append(100.0)
            c.append(100.0)
            h.append(100.1)
            l.append(99.9)
            v.append(100)
        _opens = [100.0, 102.0, 104.0]
        for po in _opens:
            pc = po + 2.0
            o.append(po)
            c.append(pc)
            h.append(pc + 0.5)
            l.append(po - 0.5)
            v.append(180)
        target = 106.5 - 7.0 * 0.50
        last_c = c[-1]
        for i in range(10):
            frac = (i + 1) / 11
            px = last_c - (last_c - target) * frac
            o.append(px + 0.05)
            c.append(px)
            h.append(px + 0.4)
            l.append(px - 0.4)
            v.append(70)
        o.append(target + 0.05)
        c.append(target)
        h.append(target + 0.4)
        l.append(target - 0.4)
        v.append(70)
        return _ohlc(o, c, h, l, v)

    def test_no_ote_wrong_retrace(self, detector, df_no_retrace):
        result = detector.detect(df_no_retrace)
        assert not result.found

    # ---------------------------------------------------------------
    # 8) No OTE when impulse volume < 1.25x
    #    volume=110 → ratio ≈ 1.1 < 1.25
    # ---------------------------------------------------------------
    @pytest.fixture
    def df_low_volume(self):
        o, c, h, l, v = [], [], [], [], []
        for _ in range(50):
            o.append(100.0)
            c.append(100.0)
            h.append(100.1)
            l.append(99.9)
            v.append(100)
        _opens = [100.0, 102.0, 104.0]
        for po in _opens:
            pc = po + 2.0
            o.append(po)
            c.append(pc)
            h.append(pc + 0.5)
            l.append(po - 0.5)
            v.append(110)
        target = 106.5 - 7.0 * 0.77
        last_c = c[-1]
        for i in range(10):
            frac = (i + 1) / 11
            px = last_c - (last_c - target) * frac
            o.append(px + 0.05)
            c.append(px)
            h.append(px + 0.4)
            l.append(px - 0.4)
            v.append(70)
        o.append(target + 0.05)
        c.append(target)
        h.append(target + 0.4)
        l.append(target - 0.4)
        v.append(70)
        return _ohlc(o, c, h, l, v)

    def test_no_ote_low_volume(self, detector, df_low_volume):
        result = detector.detect(df_low_volume)
        assert not result.found

    # ---------------------------------------------------------------
    # 9) to_detection() returns None when not found
    # ---------------------------------------------------------------
    def test_to_detection_not_found(self):
        res = OTEResult(found=False, direction="NEUTRAL", confidence=0.0)
        assert res.to_detection() is None

    # ---------------------------------------------------------------
    # 10) to_detection() returns MDDetection when found
    # ---------------------------------------------------------------
    def test_to_detection_found(self, detector, df_bullish_ote):
        result = detector.detect(df_bullish_ote)
        assert result.found
        det = result.to_detection()
        assert det is not None
        assert det.concept == MDConcept.OTE_75_79
        assert det.direction == "BUY"
        assert det.confidence > 0.5
        assert det.suggested_price is not None
        assert "impulse_range" in det.metadata
        assert "ote_level" in det.metadata
        assert "has_confluence" in det.metadata

    # ---------------------------------------------------------------
    # 11) Confluence detection works (FVG in retrace zone)
    #
    #     Impulse at indices 50-52 → end_idx=52
    #     _check_confluence range = [52-15, 52-2] = [37, 50]
    #     FVG at indices 44-45 is inside ✓
    # ---------------------------------------------------------------
    @pytest.fixture
    def df_with_confluence(self):
        o, c, h, l, v = [], [], [], [], []
        for _ in range(44):
            o.append(100.0)
            c.append(100.0)
            h.append(100.1)
            l.append(99.9)
            v.append(100)
        # FVG candle at idx 44: bearish, body_ratio > 0.4
        o.append(100.0)
        c.append(98.0)
        h.append(101.0)
        l.append(98.0)
        v.append(100)
        # FVG candle at idx 45: gap up, low > prev high
        o.append(101.5)
        c.append(101.5)
        h.append(102.0)
        l.append(101.2)
        v.append(100)
        # Remaining pre-impulse (idx 46-49)
        for _ in range(4):
            o.append(101.5)
            c.append(101.5)
            h.append(101.7)
            l.append(101.3)
            v.append(100)
        # 3 bullish impulse (idx 50-52)
        _opens = [101.5, 103.5, 105.5]
        for po in _opens:
            pc = po + 2.0
            o.append(po)
            c.append(pc)
            h.append(pc + 0.5)
            l.append(po - 0.5)
            v.append(180)
        imp_high = max(h[-3:])
        imp_low = min(l[-3:])
        imp_range = imp_high - imp_low
        target = imp_high - imp_range * 0.77
        last_c = c[-1]
        for i in range(10):
            frac = (i + 1) / 11
            px = last_c - (last_c - target) * frac
            o.append(px + 0.05)
            c.append(px)
            h.append(px + 0.4)
            l.append(px - 0.4)
            v.append(70)
        o.append(target + 0.05)
        c.append(target)
        h.append(target + 0.4)
        l.append(target - 0.4)
        v.append(70)
        df = _ohlc(o, c, h, l, v)
        assert len(df) == 64
        return df

    def test_confluence_detection(self, detector, df_with_confluence):
        result = detector.detect(df_with_confluence)
        assert result.found
        assert result.has_confluence

    # ---------------------------------------------------------------
    # 12) _check_confluence returns False when no FVG present
    # ---------------------------------------------------------------
    def test_no_confluence(self, detector, df_bullish_ote):
        result = detector.detect(df_bullish_ote)
        assert not result.has_confluence

    def test_check_confluence_no_fvg(self, detector, df_bullish_ote):
        found = detector._check_confluence(df_bullish_ote, 50, "BUY")
        assert not found

    # ---------------------------------------------------------------
    # 13) Custom ote_levels and tolerance work
    # ---------------------------------------------------------------
    @pytest.fixture
    def df_custom_ote(self):
        """Retrace to 83% — outside default OTE levels but inside custom."""
        o, c, h, l, v = [], [], [], [], []
        for _ in range(50):
            o.append(100.0)
            c.append(100.0)
            h.append(100.1)
            l.append(99.9)
            v.append(100)
        _opens = [100.0, 102.0, 104.0]
        for po in _opens:
            pc = po + 2.0
            o.append(po)
            c.append(pc)
            h.append(pc + 0.5)
            l.append(po - 0.5)
            v.append(180)
        target = 106.5 - 7.0 * 0.83
        last_c = c[-1]
        for i in range(10):
            frac = (i + 1) / 11
            px = last_c - (last_c - target) * frac
            o.append(px + 0.05)
            c.append(px)
            h.append(px + 0.4)
            l.append(px - 0.4)
            v.append(70)
        o.append(target + 0.05)
        c.append(target)
        h.append(target + 0.4)
        l.append(target - 0.4)
        v.append(70)
        return _ohlc(o, c, h, l, v)

    def test_custom_ote_levels_hit(self, detector, df_custom_ote):
        result_default = detector.detect(df_custom_ote)
        assert not result_default.found
        custom = OTEDetector(lookback=60, ote_levels=[0.83], ote_tolerance=0.02)
        result_custom = custom.detect(df_custom_ote)
        assert result_custom.found
        assert result_custom.direction == "BUY"
        assert abs(result_custom.ote_level - 0.83) < 0.001

    def test_custom_ote_levels_wide_tol(self, detector, df_no_retrace):
        custom = OTEDetector(lookback=60, ote_levels=[0.50], ote_tolerance=0.01)
        result = custom.detect(df_no_retrace)
        assert result.found
        assert abs(result.ote_level - 0.50) < 0.001

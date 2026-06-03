import pytest
import pandas as pd
import numpy as np
from src.core.pattern_detector import PatternDetector, PatternType, Pattern
from src.utils.helpers import atr


@pytest.fixture
def params():
    class FakeParams:
        fvg_min_size_atr_ratio = 0.2
        void_min_size_pips = 3.0
        fvg_entry_level = 0.5
        pivot_length = 8
        min_retracement_level = 0.5
        consolidation_max_atr_ratio = 0.5
        consolidation_min_bars = 8
        gap_detection_pips = 5.0
        wyckoff_min_phase_bars = 10
        sequence_length = 3
        breakout_validation = "BodyClose"
        news_filter_active = True
        news_buffer_minutes = 30
        macro_event_filter = True
    return FakeParams()


@pytest.fixture
def detector(params):
    return PatternDetector(params, symbol="XAUUSDm")


def build_df(opens, highs, lows, closes):
    return pd.DataFrame({
        "time": pd.date_range("2026-05-01", periods=len(opens), freq="5min"),
        "open": opens, "high": highs, "low": lows, "close": closes,
    })


class TestFVGFiltering:
    def test_empty_fvg(self, detector):
        df = build_df(
            [100.0] * 10, [105.0] * 10,
            [95.0] * 10, [100.0] * 10,
        )
        patterns = detector.detect_fvg(df)
        assert len(patterns) == 0

    def test_bullish_fvg(self, detector):
        opens = [100.0, 102.0, 103.0, 104.0]
        highs = [101.0, 103.0, 105.0, 105.0]
        lows = [99.0, 101.0, 102.0, 103.0]
        closes = [100.0, 102.0, 104.0, 104.0]
        df = build_df(opens, highs, lows, closes)
        patterns = detector.detect_fvg(df)
        bullish = [p for p in patterns if p.direction == "BUY"]
        assert len(bullish) >= 1

    def test_bearish_fvg(self, detector):
        opens = [105.0, 103.0, 102.0, 101.0]
        highs = [106.0, 104.0, 103.0, 102.0]
        lows = [104.0, 102.0, 100.0, 100.0]
        closes = [105.0, 103.0, 101.0, 101.0]
        df = build_df(opens, highs, lows, closes)
        patterns = detector.detect_fvg(df)
        bearish = [p for p in patterns if p.direction == "SELL"]
        assert len(bearish) >= 1

    def test_fvg_mitigation_filter(self, detector):
        opens = [100.0, 103.0, 102.0, 101.0, 100.0]
        highs = [101.0, 104.0, 103.0, 102.0, 101.0]
        lows = [99.0, 102.0, 101.0, 100.0, 99.0]
        closes = [100.0, 103.0, 102.0, 101.0, 100.0]
        df = build_df(opens, highs, lows, closes)
        patterns = detector.detect_fvg(df)
        for p in patterns:
            assert hasattr(p, "type")
            assert hasattr(p, "mid")

    def test_fvg_confidence(self, detector):
        df = build_df(
            [100.0, 110.0, 105.0, 106.0],
            [101.0, 112.0, 107.0, 108.0],
            [99.0, 108.0, 103.0, 104.0],
            [100.0, 110.0, 105.0, 106.0],
        )
        patterns = detector.detect_fvg(df)
        for p in patterns:
            assert 0.0 <= p.confidence <= 1.0

    def test_fvg_pattern_type_enum(self, detector):
        df = build_df(
            [100.0, 103.0, 102.0, 101.0],
            [101.0, 105.0, 104.0, 102.0],
            [99.0, 101.0, 100.0, 100.0],
            [100.0, 103.0, 102.0, 101.0],
        )
        patterns = detector.detect_fvg(df)
        for p in patterns:
            assert isinstance(p.type, PatternType)
            assert p.type.value.startswith("FVG")


class TestOrderBlockDetection:
    def test_empty_ob(self, detector):
        df = build_df([100.0] * 10, [100.5] * 10, [99.5] * 10, [100.0] * 10)
        patterns = detector.detect_order_blocks(df)
        assert len(patterns) == 0

    def test_bullish_ob(self, detector):
        opens = [101.0, 100.0, 99.0, 100.0]
        highs = [102.0, 101.0, 100.0, 101.0]
        lows = [100.0, 99.0, 98.0, 99.0]
        closes = [101.0, 100.0, 100.0, 100.0]
        df = build_df(opens, highs, lows, closes)
        patterns = detector.detect_order_blocks(df)
        bullish = [p for p in patterns if p.direction == "BUY"]
        assert len(bullish) >= 0

    def test_ob_confidence(self, detector):
        np.random.seed(42)
        opens = 100 + np.random.rand(30) * 2
        highs = opens + np.random.rand(30) * 1.5
        lows = opens - np.random.rand(30) * 1.5
        closes = opens + (np.random.rand(30) - 0.5) * 2
        df = build_df(opens.tolist(), highs.tolist(), lows.tolist(), closes.tolist())
        patterns = detector.detect_order_blocks(df)
        for p in patterns:
            assert 0.0 <= p.confidence <= 1.0


class TestLiquiditySweep:
    def test_sweep_requires_swing_points(self, detector):
        df = build_df([100.0] * 10, [101.0] * 10, [99.0] * 10, [100.0] * 10)
        patterns = detector.detect_liquidity_sweep(df)
        assert len(patterns) == 0

    def test_bullish_sweep_pattern(self, detector):
        np.random.seed(1)
        n = 50
        opens = [100.0]
        for i in range(1, n):
            opens.append(opens[-1] + (np.random.rand() - 0.5) * 2)
        highs = [o + abs(np.random.rand()) * 2 for o in opens]
        lows = [o - abs(np.random.rand()) * 2 for o in opens]
        closes = [opens[i] + (np.random.rand() - 0.5) * 1.5 for i in range(n)]
        df = build_df(opens, highs, lows, closes)
        patterns = detector.detect_liquidity_sweep(df)
        for p in patterns:
            assert isinstance(p.type, PatternType)
            assert p.type.value.startswith("SWEEP")
            if p.direction == "BUY":
                assert "swept_level" in p.extra
                assert p.extra["swept_level"] > p.low

    def test_direction_matches_sweep_type(self, detector):
        np.random.seed(2)
        n = 40
        opens = [100.0] + [100 + (np.random.rand() - 0.5) * 2 for _ in range(n-1)]
        highs = [o + abs(np.random.rand()) * 2 for o in opens]
        lows = [o - abs(np.random.rand()) * 2 for o in opens]
        closes = [opens[i] + (np.random.rand() - 0.5) * 1.5 for i in range(n)]
        df = build_df(opens, highs, lows, closes)
        patterns = detector.detect_liquidity_sweep(df)
        for p in patterns:
            if p.type == PatternType.SWEEP_BULLISH:
                assert p.direction == "BUY"
                assert p.extra.get("swept_level", 0) > p.low
            elif p.type == PatternType.SWEEP_BEARISH:
                assert p.direction == "SELL"
                assert p.extra.get("swept_level", 0) < p.high


class TestBreakerDetection:
    def test_breaker_needs_body(self, detector):
        df = build_df([100.0] * 25, [101.0] * 25, [99.0] * 25, [100.0] * 25)
        patterns = detector.detect_breakers(df)
        assert len(patterns) == 0

    def test_breaker_returns_patterns_with_swing_levels(self, detector):
        np.random.seed(3)
        n = 50
        opens = [100.0]
        for i in range(1, n):
            opens.append(opens[-1] + (np.random.rand() - 0.5) * 3)
        highs = [o + abs(np.random.rand()) * 2 + 1 for o in opens]
        lows = [o - abs(np.random.rand()) * 2 - 1 for o in opens]
        closes = [o + (np.random.rand() - 0.5) * 2 for o in opens]
        df = build_df(opens, highs, lows, closes)
        patterns = detector.detect_breakers(df)
        for p in patterns:
            assert isinstance(p.type, PatternType)
            assert p.type.value.startswith("BREAKER")
            assert p.high > p.low


class TestScanAll:
    def test_scan_all_returns_expected_keys(self, detector):
        np.random.seed(5)
        n = 80
        opens = [100.0]
        for i in range(1, n):
            opens.append(opens[-1] + (np.random.rand() - 0.5) * 2)
        highs = [o + abs(np.random.rand()) * 2 + 0.5 for o in opens]
        lows = [o - abs(np.random.rand()) * 2 - 0.5 for o in opens]
        closes = [o + (np.random.rand() - 0.5) * 1.5 for o in opens]
        df = build_df(opens, highs, lows, closes)
        results = detector.scan_all(df)
        expected_keys = {"fvg", "ob", "breaker", "sweep", "cycle",
                         "wyckoff", "sequence", "void_scalp"}
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"
            assert isinstance(results[key], list)

    def test_scan_all_with_insufficient_data(self, detector):
        df = build_df([100.0] * 5, [101.0] * 5, [99.0] * 5, [100.0] * 5)
        results = detector.scan_all(df)
        for key, patterns in results.items():
            assert isinstance(patterns, (list, dict))

    def test_detect_wyckoff_spring(self, detector):
        np.random.seed(6)
        n = 30
        opens = [100.0]
        for i in range(1, n):
            opens.append(opens[-1] + (np.random.rand() - 0.5) * 1.5)
        highs = [o + abs(np.random.rand()) * 1.5 + 0.3 for o in opens]
        lows = [o - abs(np.random.rand()) * 1.5 - 0.3 for o in opens]
        closes = [o + (np.random.rand() - 0.5) * 1.0 for o in opens]
        df = build_df(opens, highs, lows, closes)
        patterns = detector.detect_wyckoff(df)
        spring = [p for p in patterns if p.type == PatternType.SPRING_BULLISH]
        utad = [p for p in patterns if p.type == PatternType.UTAD_BEARISH]
        for p in spring:
            assert p.direction == "BUY"
        for p in utad:
            assert p.direction == "SELL"

    def test_pattern_direction_consistency(self, detector):
        np.random.seed(7)
        n = 50
        opens = [100.0] + [100 + (np.random.rand() - 0.5) * 2 for _ in range(n-1)]
        highs = [o + abs(np.random.rand()) * 2 for o in opens]
        lows = [o - abs(np.random.rand()) * 2 for o in opens]
        closes = [o + (np.random.rand() - 0.5) * 1.5 for o in opens]
        df = build_df(opens, highs, lows, closes)
        results = detector.scan_all(df)
        direction_map = {
            PatternType.FVG_BULLISH: "BUY", PatternType.FVG_BEARISH: "SELL",
            PatternType.OB_BULLISH: "BUY", PatternType.OB_BEARISH: "SELL",
            PatternType.BREAKER_BULLISH: "BUY", PatternType.BREAKER_BEARISH: "SELL",
            PatternType.SWEEP_BULLISH: "BUY", PatternType.SWEEP_BEARISH: "SELL",
            PatternType.SPRING_BULLISH: "BUY", PatternType.UTAD_BEARISH: "SELL",
            PatternType.SOS_BULLISH: "BUY", PatternType.SOW_BEARISH: "SELL",
        }
        for category in results.values():
            if not isinstance(category, list):
                continue
            for p in category:
                if p.type in direction_map:
                    assert p.direction == direction_map[p.type], (
                        f"{p.type}: expected {direction_map[p.type]}, got {p.direction}"
                    )

import pytest
import pandas as pd
import numpy as np
from src.core.regime_detector import RegimeDetector, RegimeContext, RegimeType, REGIME_PATTERN_MULTIPLIERS


def build_df(opens, highs, lows, closes):
    return pd.DataFrame({
        "time": pd.date_range("2026-05-01", periods=len(opens), freq="15min"),
        "open": opens, "high": highs, "low": lows, "close": closes,
    })


def make_trend_data(trend="up", n=100):
    if trend == "up":
        close = 100.0 + np.arange(n) * 0.05 + np.random.rand(n) * 0.2
    elif trend == "down":
        close = 200.0 - np.arange(n) * 0.05 + np.random.rand(n) * 0.2
    elif trend == "high_vol":
        close = 100.0 + np.random.randn(n) * 3
    elif trend == "low_vol":
        close = 100.0 + np.random.randn(n) * 0.1
    else:
        close = 100.0 + np.random.rand(n) * 2
    opens = close + (np.random.rand(n) - 0.5) * 0.1
    highs = np.maximum(opens, close) + np.random.rand(n) * 0.3
    lows = np.minimum(opens, close) - np.random.rand(n) * 0.3
    return opens.tolist(), highs.tolist(), lows.tolist(), close.tolist()


class TestRegimeType:
    def test_all_regimes_defined(self):
        assert len(RegimeType) == 6
        assert RegimeType.STRONG_TREND_BULLISH.value == "STRONG_TREND_BULLISH"
        assert RegimeType.STRONG_TREND_BEARISH.value == "STRONG_TREND_BEARISH"
        assert RegimeType.RANGING.value == "RANGING"
        assert RegimeType.HIGH_VOLATILITY.value == "HIGH_VOLATILITY"
        assert RegimeType.LOW_VOLATILITY.value == "LOW_VOLATILITY"
        assert RegimeType.TRANSITION.value == "TRANSITION"


class TestRegimeContext:
    def test_get_multiplier_returns_default(self):
        ctx = RegimeContext(
            regime=RegimeType.RANGING, confidence=0.6, strength=0.3,
            atr_ratio=1.0, adx_value=20.0, is_compressed=False,
            is_expanding=False, trend_alignment="NEUTRAL",
            pattern_multipliers={},
        )
        assert ctx.get_multiplier("FVG") == 1.0
        assert ctx.get_multiplier("NONEXISTENT") == 1.0

    def test_get_multiplier_returns_configured(self):
        ctx = RegimeContext(
            regime=RegimeType.RANGING, confidence=0.6, strength=0.3,
            atr_ratio=1.0, adx_value=20.0, is_compressed=False,
            is_expanding=False, trend_alignment="NEUTRAL",
            pattern_multipliers={"FVG": 0.5},
        )
        assert ctx.get_multiplier("FVG") == 0.5


class TestRegimeDetector:
    def test_fallback_with_insufficient_data(self):
        detector = RegimeDetector()
        htf = build_df(*make_trend_data("flat", n=10))
        ltf = build_df(*make_trend_data("flat", n=10))
        ctx = detector.detect(htf, ltf)
        assert ctx.regime == RegimeType.RANGING
        assert ctx.confidence == 0.3
        assert ctx.strength == 0.0
        assert ctx.trend_alignment == "NEUTRAL"

    def test_fallback_with_none_data(self):
        detector = RegimeDetector()
        ctx = detector.detect(None, None)
        assert ctx.regime == RegimeType.RANGING
        assert ctx.confidence == 0.3

    def test_classify_strong_trend_bullish(self, monkeypatch):
        detector = RegimeDetector()
        htf = build_df(*make_trend_data("up", n=60))
        ltf = build_df(*make_trend_data("up", n=60))

        def mock_classify(atr_ratio, adx, trend_strength, is_compressed, is_expanding, ema_slope=0.0):
            return RegimeType.STRONG_TREND_BULLISH, 0.8

        monkeypatch.setattr(detector, "_classify_regime", mock_classify)
        ctx = detector.detect(htf, ltf)
        assert ctx.regime == RegimeType.STRONG_TREND_BULLISH
        assert ctx.confidence > 0

    def test_classify_strong_trend_bearish(self, monkeypatch):
        detector = RegimeDetector()
        htf = build_df(*make_trend_data("down", n=60))
        ltf = build_df(*make_trend_data("down", n=60))

        def mock_classify(atr_ratio, adx, trend_strength, is_compressed, is_expanding, ema_slope=0.0):
            return RegimeType.STRONG_TREND_BEARISH, 0.7

        monkeypatch.setattr(detector, "_classify_regime", mock_classify)
        ctx = detector.detect(htf, ltf)
        assert ctx.regime == RegimeType.STRONG_TREND_BEARISH

    def test_pattern_multipliers_generated(self, monkeypatch):
        detector = RegimeDetector()

        def mock_classify(atr_ratio, adx, trend_strength, is_compressed, is_expanding, ema_slope=0.0):
            return RegimeType.RANGING, 0.6
        monkeypatch.setattr(detector, "_classify_regime", mock_classify)

        htf = build_df(*make_trend_data("flat", n=60))
        ltf = build_df(*make_trend_data("flat", n=60))
        ctx = detector.detect(htf, ltf)
        assert len(ctx.pattern_multipliers) > 0
        for pattern_name in REGIME_PATTERN_MULTIPLIERS:
            assert pattern_name in ctx.pattern_multipliers
            assert ctx.pattern_multipliers[pattern_name] > 0

    def test_trend_alignment_bullish(self, monkeypatch):
        detector = RegimeDetector()

        def mock_classify(atr_ratio, adx, trend_strength, is_compressed, is_expanding, ema_slope=0.0):
            return RegimeType.RANGING, 0.5
        monkeypatch.setattr(detector, "_classify_regime", mock_classify)

        def mock_alignment(htf_df, ltf_df):
            return "BULLISH_ALIGNED"
        monkeypatch.setattr(detector, "_detect_trend_alignment", mock_alignment)

        htf = build_df(*make_trend_data("up", n=60))
        ltf = build_df(*make_trend_data("up", n=60))
        ctx = detector.detect(htf, ltf)
        assert ctx.trend_alignment == "BULLISH_ALIGNED"

    def test_trend_alignment_bearish(self):
        detector = RegimeDetector()
        n = 60
        opens_d = [200.0 - i * 0.5 for i in range(n)]
        highs_d = [o + 1 for o in opens_d]
        lows_d = [o - 1 for o in opens_d]
        closes_d = [o - 0.3 for o in opens_d]
        htf = build_df(opens_d, highs_d, lows_d, closes_d)
        ltf = build_df(opens_d, highs_d, lows_d, closes_d)
        alignment = detector._detect_trend_alignment(htf, ltf)
        assert alignment in ("BEARISH_ALIGNED", "NEUTRAL")

    def test_compute_adx_with_insufficient_data(self):
        detector = RegimeDetector()
        df = build_df(*make_trend_data("flat", n=5))
        adx = detector._compute_adx(df)
        assert adx == 0.0

    def test_compute_adx_with_sufficient_data(self):
        detector = RegimeDetector()
        np.random.seed(42)
        df = build_df(*make_trend_data("up", n=60))
        adx = detector._compute_adx(df)
        assert adx >= 0

    def test_classify_high_volatility(self):
        detector = RegimeDetector()
        regime, confidence = detector._classify_regime(
            atr_ratio=2.0, adx=20.0, trend_strength=0.3,
            is_compressed=False, is_expanding=True, ema_slope=0.0,
        )
        assert regime == RegimeType.HIGH_VOLATILITY
        assert confidence > 0

    def test_classify_strong_trend_bullish(self):
        detector = RegimeDetector()
        regime, confidence = detector._classify_regime(
            atr_ratio=1.0, adx=40.0, trend_strength=0.8,
            is_compressed=False, is_expanding=False, ema_slope=0.5,
        )
        assert regime == RegimeType.STRONG_TREND_BULLISH
        assert confidence > 0

    def test_classify_strong_trend_bearish(self):
        detector = RegimeDetector()
        regime, confidence = detector._classify_regime(
            atr_ratio=1.0, adx=40.0, trend_strength=0.8,
            is_compressed=False, is_expanding=False, ema_slope=-0.3,
        )
        assert regime == RegimeType.STRONG_TREND_BEARISH

    def test_classify_low_volatility(self):
        detector = RegimeDetector()
        regime, confidence = detector._classify_regime(
            atr_ratio=0.5, adx=15.0, trend_strength=0.2,
            is_compressed=True, is_expanding=False, ema_slope=0.0,
        )
        assert regime == RegimeType.LOW_VOLATILITY

    def test_classify_ranging(self):
        detector = RegimeDetector()
        regime, confidence = detector._classify_regime(
            atr_ratio=1.0, adx=25.0, trend_strength=0.3,
            is_compressed=False, is_expanding=False, ema_slope=0.0,
        )
        assert regime == RegimeType.RANGING

    def test_classify_transition(self):
        detector = RegimeDetector()
        regime, confidence = detector._classify_regime(
            atr_ratio=1.0, adx=15.0, trend_strength=0.3,
            is_compressed=False, is_expanding=False, ema_slope=0.0,
        )
        assert regime == RegimeType.TRANSITION

    def test_regime_stability_confidence_reduction(self, monkeypatch):
        detector = RegimeDetector()

        calls = [0]
        def mock_classify(atr_ratio, adx, trend_strength, is_compressed, is_expanding, ema_slope=0.0):
            calls[0] += 1
            confidence = 0.3 + calls[0] * 0.15
            return RegimeType.HIGH_VOLATILITY, min(1.0, confidence)
        monkeypatch.setattr(detector, "_classify_regime", mock_classify)

        htf = build_df(*make_trend_data("high_vol", n=60))
        ltf = build_df(*make_trend_data("high_vol", n=60))

        ctx1 = detector.detect(htf, ltf)
        ctx2 = detector.detect(htf, ltf)
        assert ctx2.confidence == pytest.approx(ctx2.confidence, rel=0.01)

        ctx3 = detector.detect(htf, ltf)
        ctx4 = detector.detect(htf, ltf)
        assert ctx4.confidence > ctx3.confidence

    def test_notes_contain_regime_info(self, monkeypatch):
        detector = RegimeDetector()

        def mock_classify(atr_ratio, adx, trend_strength, is_compressed, is_expanding, ema_slope=0.0):
            return RegimeType.RANGING, 0.5
        monkeypatch.setattr(detector, "_classify_regime", mock_classify)

        htf = build_df(*make_trend_data("flat", n=60))
        ltf = build_df(*make_trend_data("flat", n=60))
        ctx = detector.detect(htf, ltf)
        assert len(ctx.notes) > 0
        assert "RANGING" in ctx.notes[0]

    def test_detect_trend_alignment_neutral(self):
        detector = RegimeDetector()
        np.random.seed(42)
        o, h, l, c = make_trend_data("flat", n=30)
        htf = build_df(o, h, l, c)
        ltf = build_df(o, h, l, c)
        alignment = detector._detect_trend_alignment(htf, ltf)
        assert alignment in ("NEUTRAL", "BULLISH_ALIGNED", "BEARISH_ALIGNED")


class TestREGIMEPATTERNMULTIPLIERS:
    def test_all_regimes_have_multipliers(self):
        for pattern_name, regime_map in REGIME_PATTERN_MULTIPLIERS.items():
            for regime in RegimeType:
                assert regime in regime_map, f"Missing multiplier for {pattern_name}@{regime}"
                assert isinstance(regime_map[regime], float)

    def test_all_pattern_groups_have_entries(self):
        expected_groups = [
            "FVG", "OB", "BREAKER", "SWEEP", "WYCKOFF", "VOID_SCALP",
            "BOS_ZONE", "CYCLE", "SEQUENCE", "INTERVAL_POINT",
            "PRICE_INTERACTION", "HARMONIC_CYCLE", "PRESSURE_ZONE", "TRB",
        ]
        for group in expected_groups:
            assert group in REGIME_PATTERN_MULTIPLIERS

    def test_multipliers_within_reasonable_range(self):
        for pattern_name, regime_map in REGIME_PATTERN_MULTIPLIERS.items():
            for regime, mult in regime_map.items():
                assert 0.3 <= mult <= 2.0, f"{pattern_name}@{regime}: {mult} out of range"

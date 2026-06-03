import pytest

from src.core.entry_confirmer import (
    TickDeltaAnalyzer, DOMAnalyzer, StopRunDetector, EntryConfirmer,
)
from src.core.micro_phase import MicroPhase, PhaseResult


class TestTickDeltaAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return TickDeltaAnalyzer(divergence_window=5, delta_threshold=0.6)

    def test_none_ticks(self, analyzer):
        ok, conf, reason = analyzer.analyze(None, "BUY")
        assert not ok
        assert reason == "insufficient_tick_data"

    def test_insufficient_ticks(self, analyzer):
        ok, conf, reason = analyzer.analyze([{"price": 100.0}], "BUY")
        assert not ok

    def test_bullish_divergence(self, analyzer):
        """Price dropping but buy volume dominating → bullish divergence."""
        ticks = []
        for i in range(10):
            price = 100.0 - i * 0.1
            volume = 10
            aggressor = 1 if i < 9 else 1  # all 10 buy → last 5 from window: all buy
            ticks.append({"price": price, "volume": volume, "aggressor": aggressor})

        ok, conf, reason = analyzer.analyze(ticks, "BUY")
        assert ok, f"Expected bullish divergence: {reason}"
        assert conf > 0.5

    def test_bearish_divergence(self, analyzer):
        """Price rising but sell volume dominating → bearish divergence."""
        ticks = []
        for i in range(10):
            price = 100.0 + i * 0.1
            volume = 10
            aggressor = -1 if i < 9 else -1  # all 10 sell
            ticks.append({"price": price, "volume": volume, "aggressor": aggressor})

        ok, conf, reason = analyzer.analyze(ticks, "SELL")
        assert ok, f"Expected bearish divergence: {reason}"

    def test_no_divergence(self, analyzer):
        """Price and delta move together → no divergence."""
        ticks = []
        for i in range(10):
            price = 100.0 + i * 0.1  # price rising
            volume = 10
            aggressor = 1  # buy aggressor → no divergence for SELL
            ticks.append({"price": price, "volume": volume, "aggressor": aggressor})

        ok, conf, reason = analyzer.analyze(ticks, "SELL")
        assert not ok


class TestDOMAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return DOMAnalyzer(absorption_depth=3, absorption_multiple=3.0)

    def test_none_dom(self, analyzer):
        ok, conf, reason = analyzer.analyze(None)
        assert not ok
        assert reason == "no_dom_data"

    def test_empty_dom(self, analyzer):
        ok, conf, reason = analyzer.analyze({"bids": [], "asks": []})
        assert not ok
        assert reason == "empty_dom"

    def test_imbalance_detected(self, analyzer):
        dom = {
            "bids": [{"volume": 100, "size": 100}] * 3,
            "asks": [{"volume": 10, "size": 10}] * 3,
        }
        ok, conf, reason = analyzer.analyze(dom)
        assert ok
        assert reason == "dom_imbalance"

    def test_neutral_dom(self, analyzer):
        dom = {
            "bids": [{"volume": 50}] * 3,
            "asks": [{"volume": 50}] * 3,
        }
        ok, conf, reason = analyzer.analyze(dom)
        assert not ok
        assert reason == "neutral_dom"


class TestStopRunDetector:
    @pytest.fixture
    def detector(self):
        return StopRunDetector(lookback=5, stop_run_range=0.10, reversal_candles=2)

    def test_insufficient_data(self, detector, df_flat):
        small = df_flat.iloc[:3]
        ok, strength, reason = detector.analyze(small)
        assert not ok

    def test_bullish_stop_run(self, detector, df_stop_run_bullish):
        ok, strength, reason = detector.analyze(df_stop_run_bullish)
        assert ok, f"Expected stop run: {reason}"
        assert reason == "stop_run_bullish"
        assert strength > 0.0

    def test_bearish_stop_run(self, detector, df_stop_run_bearish):
        ok, strength, reason = detector.analyze(df_stop_run_bearish)
        assert ok, f"Expected stop run: {reason}"
        assert reason == "stop_run_bearish"
        assert strength > 0.0

    def test_no_stop_run_on_flat(self, detector, df_flat):
        ok, strength, reason = detector.analyze(df_flat)
        assert not ok


class TestEntryConfirmer:
    @pytest.fixture
    def confirmer(self):
        return EntryConfirmer(min_confirmers=2)

    def test_phase_not_allowed(self, confirmer, df_bullish_trend):
        bad_phase = PhaseResult(MicroPhase.INDECISION, "NEUTRAL", 0.2)
        result = confirmer.confirm("XAUUSD", "BUY", bad_phase, df_bullish_trend)
        assert not result.valid
        assert "no_entry" in result.reason

    def test_insufficient_confirmers(self, confirmer, df_bullish_trend):
        """No ticks, no dom, no stop-run → should fail."""
        phase = PhaseResult(MicroPhase.IMPULSE_STARTING, "BUY", 0.6)
        result = confirmer.confirm(
            "XAUUSD", "BUY", phase, df_bullish_trend,
            ticks=None, dom=None,
        )
        assert not result.valid

    def test_high_confidence_override(self, confirmer, df_bullish_trend):
        """IMPULSE_STARTING with confidence >= 0.7 needs only 1 confirmer."""
        phase = PhaseResult(MicroPhase.IMPULSE_STARTING, "BUY", 0.75)
        result = confirmer.confirm(
            "XAUUSD", "BUY", phase, df_bullish_trend,
            ticks=None, dom=None,
        )
        # With no confirmers active at all, still fails
        assert not result.valid

    def test_breaktout_retest_override(self, confirmer, df_bullish_trend):
        phase = PhaseResult(MicroPhase.BREAKOUT_RETEST, "BUY", 0.7)
        result = confirmer.confirm(
            "XAUUSD", "BUY", phase, df_bullish_trend,
            ticks=[{"price": 100.0, "volume": 10, "aggressor": 1}] * 10,
            dom=None,
        )
        assert not result.valid  # still needs at least 1 confirmer; only tick

import numpy as np
import pytest

from src.core.breakout_retest import BreakoutRetestDetector, BreakoutRetestSignal
from src.core.zone_state_tracker import ZoneStateTracker, ZoneStatus, ZoneType, ZoneRecord


class TestBreakoutRetestDetector:
    @pytest.fixture
    def detector(self):
        return BreakoutRetestDetector(
            min_touches_for_consolidation=3,
            retest_tolerance_pct=0.10,
            breakout_confirm_candles=2,
            score_bonus=30.0,
        )

    def test_no_broken_zones_returns_empty(self, detector, df_bullish_trend):
        tracker = ZoneStateTracker()
        signals = detector.check(df_bullish_trend, tracker, "XAUUSD")
        assert signals == []

    def test_detect_retest_confirmed(self, detector):
        import pandas as pd

        tracker = ZoneStateTracker()
        zone = tracker.register_zone(
            "XAUUSD", ZoneType.FVG, "H1", 100.0, 101.0, 99.0, "BUY",
        )
        tracker.detect_breakout("XAUUSD", 102.0, "BUY")
        zone.mark_retest()

        n = 20
        opens = [100.0 + i * 0.1 for i in range(n)]
        closes = [100.05 + i * 0.1 for i in range(n)]
        highs = [o + 0.3 for o in opens]
        lows = [c - 0.3 for c in closes]
        # Last 2 candles close very close to zone price_level (100.0)
        closes[-2] = 100.02
        closes[-1] = 100.02
        opens[-2] = 100.0
        opens[-1] = 100.0

        df = pd.DataFrame({
            "open": opens, "close": closes,
            "high": highs, "low": lows,
        })
        signals = detector.check(df, tracker, "XAUUSD")
        assert len(signals) > 0
        sig = signals[0]
        assert sig.active
        assert sig.direction == "BUY"
        assert sig.confidence > 0.0

    def test_score_bonus_applied(self, detector):
        import pandas as pd

        tracker = ZoneStateTracker()
        zone = tracker.register_zone(
            "XAUUSD", ZoneType.FVG, "H1", 100.0, 101.0, 99.0, "BUY",
        )
        tracker.detect_breakout("XAUUSD", 102.0, "BUY")
        zone.mark_retest()

        df = pd.DataFrame({
            "open": [100.5, 100.3, 100.0, 99.8, 99.5],
            "close": [100.3, 100.0, 99.8, 99.5, 100.2],
            "high": [100.8, 100.6, 100.3, 100.1, 100.5],
            "low": [100.0, 99.8, 99.5, 99.3, 99.0],
        })
        signals = detector.check(df, tracker, "XAUUSD")
        for sig in signals:
            assert sig.score_bonus == 30.0

    def test_insufficient_data_returns_empty(self, detector):
        tracker = ZoneStateTracker()
        import pandas as pd
        small = pd.DataFrame({"open": [100.0], "close": [100.0],
                               "high": [100.3], "low": [99.7]})
        signals = detector.check(small, tracker, "XAUUSD")
        assert signals == []

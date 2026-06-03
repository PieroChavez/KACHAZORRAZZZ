import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.core.session_profiler import (
    SessionProfiler, SessionProfile, TradingSession,
    SESSION_HOURS_UTC, DEFAULT_PATTERNS_BY_SESSION,
)


class TestTradingSession:
    def test_all_sessions_defined(self):
        assert len(TradingSession) == 7
        expected = ["ASIAN", "LONDON_OPEN", "LONDON_MID", "NY_OPEN",
                     "LONDON_NY_OVERLAP", "NY_AFTERNOON", "CLOSE"]
        for name in expected:
            assert hasattr(TradingSession, name)


class TestSessionHours:
    def test_all_sessions_have_hours(self):
        for session in TradingSession:
            assert session in SESSION_HOURS_UTC
            start, end = SESSION_HOURS_UTC[session]
            assert 0 <= start < 24
            assert 0 <= end <= 24
            assert start < end


class TestSessionProfiler:
    def test_get_session_asian(self):
        profiler = SessionProfiler()
        dt = datetime(2026, 5, 25, 3, 0)
        assert profiler.get_session(dt) == TradingSession.ASIAN

    def test_get_session_london_open(self):
        profiler = SessionProfiler()
        dt = datetime(2026, 5, 25, 8, 0)
        assert profiler.get_session(dt) == TradingSession.LONDON_OPEN

    def test_get_session_london_mid(self):
        profiler = SessionProfiler()
        dt = datetime(2026, 5, 25, 10, 0)
        assert profiler.get_session(dt) == TradingSession.LONDON_MID

    def test_get_session_ny_open(self):
        profiler = SessionProfiler()
        dt = datetime(2026, 5, 25, 13, 0)
        assert profiler.get_session(dt) == TradingSession.NY_OPEN

    def test_get_session_overlap(self):
        profiler = SessionProfiler()
        dt = datetime(2026, 5, 25, 15, 0)
        assert profiler.get_session(dt) == TradingSession.LONDON_NY_OVERLAP

    def test_get_session_ny_afternoon(self):
        profiler = SessionProfiler()
        dt = datetime(2026, 5, 25, 18, 0)
        assert profiler.get_session(dt) == TradingSession.NY_AFTERNOON

    def test_get_session_close(self):
        profiler = SessionProfiler()
        dt = datetime(2026, 5, 25, 22, 0)
        assert profiler.get_session(dt) == TradingSession.CLOSE

    def test_get_session_default_uses_utcnow(self, monkeypatch):
        profiler = SessionProfiler()
        dt = datetime(2026, 5, 25, 13, 0)

        class FakeDatetime:
            @classmethod
            def utcnow(cls):
                return dt

        monkeypatch.setattr("src.core.session_profiler.datetime", FakeDatetime)
        assert profiler.get_session() == TradingSession.NY_OPEN

    def test_is_peak_time_london_open(self):
        profiler = SessionProfiler()
        dt = datetime(2026, 5, 25, 8, 30)
        session = TradingSession.LONDON_OPEN
        assert profiler.is_peak_time(session, dt)

    def test_is_not_peak_time(self):
        profiler = SessionProfiler()
        dt = datetime(2026, 5, 25, 0, 0)
        assert not profiler.is_peak_time(TradingSession.ASIAN, dt)

    def test_is_weak_time_asian(self):
        profiler = SessionProfiler()
        dt = datetime(2026, 5, 25, 1, 0)
        assert profiler.is_weak_time(TradingSession.ASIAN, dt)

    def test_is_not_weak_time(self):
        profiler = SessionProfiler()
        dt = datetime(2026, 5, 25, 4, 0)
        assert not profiler.is_weak_time(TradingSession.ASIAN, dt)

    def test_profile_asian_session(self):
        profiler = SessionProfiler()
        dt = datetime(2026, 5, 25, 3, 0)
        profile = profiler.profile("XAUUSDm", dt=dt)
        assert profile.session == TradingSession.ASIAN
        assert profile.label == "ASIAN"
        assert "OB" in profile.preferred_patterns
        assert "VOID_SCALP" in profile.avoided_patterns
        assert 0 < profile.volatility_pct

    def test_profile_london_open_peak(self):
        profiler = SessionProfiler()
        dt = datetime(2026, 5, 25, 8, 30)
        profile = profiler.profile("XAUUSDm", dt=dt)
        assert profile.is_peak
        assert profile.aggressiveness == "aggressive"
        assert profile.volume_adjustment == 1.2
        assert profile.sl_adjustment == 0.9

    def test_profile_ny_afternoon_weak(self):
        profiler = SessionProfiler()
        dt = datetime(2026, 5, 25, 19, 0)
        profile = profiler.profile("XAUUSDm", dt=dt)
        assert profile.is_weak
        assert profile.aggressiveness == "conservative"
        assert profile.volume_adjustment == 0.5
        assert profile.sl_adjustment == 1.2

    def test_profile_normal_conditions(self):
        profiler = SessionProfiler()
        dt = datetime(2026, 5, 25, 10, 0)
        profile = profiler.profile("XAUUSDm", dt=dt)
        assert not profile.is_peak
        assert not profile.is_weak
        assert profile.aggressiveness == "moderate"
        assert profile.volume_adjustment == 1.0

    def test_profile_close_session_avoids_fvg(self):
        profiler = SessionProfiler()
        dt = datetime(2026, 5, 25, 22, 0)
        profile = profiler.profile("XAUUSDm", dt=dt)
        assert "FVG" in profile.avoided_patterns
        assert profile.preferred_patterns == []

    def test_profile_with_dataframe(self):
        profiler = SessionProfiler()
        n = 30
        np.random.seed(42)
        opens = 100.0 + np.random.randn(n) * 0.5
        highs = opens + abs(np.random.randn(n)) * 0.5
        lows = opens - abs(np.random.randn(n)) * 0.5
        closes = opens + np.random.randn(n) * 0.2
        df = pd.DataFrame({
            "time": pd.date_range("2026-05-01", periods=n, freq="5min"),
            "open": opens, "high": highs, "low": lows, "close": closes,
        })
        dt = datetime(2026, 5, 25, 13, 0)
        profile = profiler.profile("XAUUSDm", ltf_df=df, dt=dt)
        assert profile.volatility_pct > 0
        assert profile.session == TradingSession.NY_OPEN

    def test_estimate_volatility_default_when_no_df(self):
        profiler = SessionProfiler()
        vol = profiler._estimate_volatility("XAUUSDm", TradingSession.ASIAN, None)
        assert vol == 0.003 * 0.7  # asian_volatility_reduction applied

    def test_estimate_volatility_default_when_insufficient_data(self):
        profiler = SessionProfiler()
        df = pd.DataFrame({"close": [100.0]}, index=[0])
        vol = profiler._estimate_volatility("XAUUSDm", TradingSession.ASIAN, df)
        assert vol == 0.003 * 0.7

    def test_adjust_decision_peak(self):
        profiler = SessionProfiler()
        dt = datetime(2026, 5, 25, 8, 30)
        profile = profiler.profile("XAUUSDm", dt=dt)
        vol, sl, tp, aggr = profiler.adjust_decision(profile, conviction=0.5)
        assert vol == 1.2 * 1.2  # peak boost × london_peak_boost
        assert sl == 0.9
        assert tp == 1.2
        assert aggr == "aggressive"

    def test_adjust_decision_weak(self):
        profiler = SessionProfiler()
        dt = datetime(2026, 5, 25, 19, 0)
        profile = profiler.profile("XAUUSDm", dt=dt)
        vol, sl, tp, aggr = profiler.adjust_decision(profile, conviction=0.5)
        assert vol == 0.5
        assert sl == 1.2
        assert tp == 0.8
        assert aggr == "conservative"

    def test_adjust_decision_normal(self):
        profiler = SessionProfiler()
        dt = datetime(2026, 5, 25, 10, 0)
        profile = profiler.profile("XAUUSDm", dt=dt)
        vol, sl, tp, aggr = profiler.adjust_decision(profile, conviction=0.5)
        assert vol == 1.0
        assert sl == 1.0
        assert tp == 1.0
        assert aggr == "moderate"

    def test_get_preferred_patterns(self):
        profiler = SessionProfiler()
        dt = datetime(2026, 5, 25, 8, 0)
        patterns = profiler.get_preferred_patterns(dt)
        assert isinstance(patterns, list)
        assert "FVG" in patterns

    def test_get_avoided_patterns(self):
        profiler = SessionProfiler()
        dt = datetime(2026, 5, 25, 22, 0)
        patterns = profiler.get_avoided_patterns(dt)
        assert isinstance(patterns, list)
        assert "FVG" in patterns

    def test_notes_contain_session_info(self):
        profiler = SessionProfiler()
        dt = datetime(2026, 5, 25, 13, 0)
        profile = profiler.profile("XAUUSDm", dt=dt)
        assert len(profile.notes) > 0
        assert any("NY_OPEN" in n for n in profile.notes)


class TestDefaultPatternsBySession:
    def test_all_sessions_have_entries(self):
        for session in TradingSession:
            assert session in DEFAULT_PATTERNS_BY_SESSION
            assert "preferred" in DEFAULT_PATTERNS_BY_SESSION[session]
            assert "avoided" in DEFAULT_PATTERNS_BY_SESSION[session]

import time
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest

from src.core.circuit_breakers import (
    CircuitBreakers,
    CircuitBreakerState,
    BreakerStatus,
)


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture
def cb():
    return CircuitBreakers()


@pytest.fixture
def df_high_volatility():
    """20 velas: primeras 15 de baja volatilidad, últimas 5 con spike."""
    opens = [100.0] * 15 + [100.0, 101.0, 99.0, 102.0, 98.0]
    closes = [100.0] * 15 + [101.0, 99.0, 102.0, 98.0, 103.0]
    highs = [100.3] * 15 + [101.5, 102.0, 103.0, 103.5, 104.0]
    lows = [99.7] * 15 + [99.5, 98.0, 98.0, 97.0, 97.0]
    return pd.DataFrame({"open": opens, "close": closes, "high": highs, "low": lows})


@pytest.fixture
def df_momentum_against_long():
    """10 velas con cierres descendentes (malo para long)."""
    closes = [100.0 - i * 0.5 for i in range(10)]
    opens = [c + 0.1 for c in closes]
    return pd.DataFrame({
        "open": opens, "close": closes,
        "high": [max(o, c) + 0.2 for o, c in zip(opens, closes)],
        "low": [min(o, c) - 0.2 for o, c in zip(opens, closes)],
    })


@pytest.fixture
def df_momentum_against_short():
    """10 velas con cierres ascendentes (malo para short)."""
    closes = [100.0 + i * 0.5 for i in range(10)]
    opens = [c - 0.1 for c in closes]
    return pd.DataFrame({
        "open": opens, "close": closes,
        "high": [max(o, c) + 0.2 for o, c in zip(opens, closes)],
        "low": [min(o, c) - 0.2 for o, c in zip(opens, closes)],
    })


@pytest.fixture
def mock_news_calendar():
    nc = MagicMock()
    nc.is_high_impact_active.return_value = False
    nc.get_upcoming_high_impact.return_value = []
    return nc


# ─────────────────────────────────────────────
# BREAKER STATUS / STATE
# ─────────────────────────────────────────────

class TestBreakerState:
    def test_any_active_false_by_default(self):
        assert CircuitBreakerState().any_active() is False

    def test_any_active_true_when_one_active(self):
        s = CircuitBreakerState()
        s.volatility_spike.active = True
        assert s.any_active() is True

    def test_highest_severity_default_is_info(self):
        assert CircuitBreakerState().highest_severity() == "info"

    def test_highest_severity_picks_critical(self):
        s = CircuitBreakerState()
        s.news_approaching = BreakerStatus(active=True, severity="critical")
        s.volatility_spike = BreakerStatus(active=True, severity="warning")
        assert s.highest_severity() == "critical"

    def test_highest_severity_ignores_inactive(self):
        s = CircuitBreakerState()
        s.news_approaching = BreakerStatus(active=False, severity="critical")
        assert s.highest_severity() == "info"


# ─────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────

class TestCircuitBreakersInit:
    def test_default_parameters(self):
        cb = CircuitBreakers()
        assert cb.volatility_threshold == 2.0
        assert cb.momentum_candles == 3
        assert cb.news_buffer_minutes == 30.0
        assert cb._atr_history == {}

    def test_custom_parameters(self):
        cb = CircuitBreakers(volatility_threshold=3.0, momentum_candles=5, news_buffer_minutes=15.0)
        assert cb.volatility_threshold == 3.0
        assert cb.momentum_candles == 5
        assert cb.news_buffer_minutes == 15.0


# ─────────────────────────────────────────────
# VOLATILITY SPIKE
# ─────────────────────────────────────────────

class TestVolatilitySpike:
    def test_not_enough_data_returns_inactive(self, cb):
        df = pd.DataFrame({"high": [100.0], "low": [99.0], "close": [99.5]})
        assert cb.check_volatility_spike("EURUSD", df).active is False

    def test_no_spike_when_flat(self, cb, df_flat):
        for _ in range(5):
            status = cb.check_volatility_spike("EURUSD", df_flat)
        assert status.active is False

    def test_spike_triggers_after_history_built(self, cb, df_flat, df_high_volatility):
        for _ in range(4):
            cb.check_volatility_spike("EURUSD", df_flat)
        with patch("src.core.circuit_breakers.time.time", return_value=1000000.0):
            status = cb.check_volatility_spike("EURUSD", df_high_volatility)
        assert status.active is True
        assert "ATR spike" in status.reason
        assert status.severity == "warning"
        assert status.expires_at == 1001800.0

    def test_spike_not_triggered_when_baseline_zero(self, cb):
        df = pd.DataFrame({
            "high": [100.0] * 20, "low": [100.0] * 20, "close": [100.0] * 20,
        })
        for _ in range(5):
            status = cb.check_volatility_spike("EURUSD", df)
        assert status.active is False

    def test_atr_history_capped_at_20(self, cb, df_flat):
        for _ in range(30):
            cb.check_volatility_spike("EURUSD", df_flat)
        assert len(cb._atr_history["EURUSD"]) <= 20


# ─────────────────────────────────────────────
# MOMENTUM AGAINST
# ─────────────────────────────────────────────

class TestMomentumAgainst:
    def test_not_enough_data_returns_inactive(self, cb):
        df = pd.DataFrame({"close": [100.0]})
        assert cb.check_momentum_against("EURUSD", df, True, 100.0).active is False

    def test_long_triggered_on_bearish_candles(self, cb, df_momentum_against_long):
        with patch("src.core.circuit_breakers.time.time", return_value=1000000.0):
            status = cb.check_momentum_against("EURUSD", df_momentum_against_long, True, 100.0)
        assert status.active is True
        assert "velas consecutivas en contra" in status.reason
        assert "PnL" in status.reason
        assert status.expires_at == 1000900.0

    def test_short_triggered_on_bullish_candles(self, cb, df_momentum_against_short):
        with patch("src.core.circuit_breakers.time.time", return_value=1000000.0):
            status = cb.check_momentum_against("EURUSD", df_momentum_against_short, False, 100.0)
        assert status.active is True

    def test_not_triggered_when_not_all_consecutive(self, cb, df_flat):
        with patch("src.core.circuit_breakers.time.time", return_value=1000000.0):
            status = cb.check_momentum_against("EURUSD", df_flat, True, 100.0)
        assert status.active is False

    def test_severity_warning_when_pnl_below_neg_1pct(self, cb, df_momentum_against_long):
        with patch("src.core.circuit_breakers.time.time", return_value=1000000.0):
            status = cb.check_momentum_against("EURUSD", df_momentum_against_long, True, 100.0)
        assert status.severity == "warning"

    def test_severity_info_when_pnl_above_neg_1pct(self, cb):
        closes = [100.0, 99.95, 99.90, 99.85]
        opens = [100.05, 100.0, 99.95, 99.90]
        df = pd.DataFrame({
            "open": opens, "close": closes,
            "high": [100.15] * 4, "low": [99.80] * 4,
        })
        with patch("src.core.circuit_breakers.time.time", return_value=1000000.0):
            status = cb.check_momentum_against("EURUSD", df, True, 100.0)
        assert status.active is True
        assert status.severity == "info"


# ─────────────────────────────────────────────
# NEWS APPROACHING
# ─────────────────────────────────────────────

class TestNewsApproaching:
    def test_no_calendar_returns_inactive(self, cb):
        status = cb.check_news_approaching(None, "EURUSD", pd.Timestamp("2025-01-01"))
        assert status.active is False

    def test_calendar_without_required_methods(self, cb):
        nc = MagicMock(spec=[])
        status = cb.check_news_approaching(nc, "EURUSD", pd.Timestamp("2025-01-01"))
        assert status.active is False

    def test_high_impact_active_triggers_critical(self, cb, mock_news_calendar):
        mock_news_calendar.is_high_impact_active.return_value = True
        with patch("src.core.circuit_breakers.time.time", return_value=1000000.0):
            status = cb.check_news_approaching(mock_news_calendar, "EURUSD", pd.Timestamp("2025-01-01"))
        assert status.active is True
        assert "alto impacto" in status.reason
        assert status.severity == "critical"
        assert status.expires_at == 1003600.0

    def test_upcoming_news_within_buffer_triggers_warning(self, cb, mock_news_calendar):
        now = pd.Timestamp("2025-01-01 12:00:00")
        future = pd.Timestamp("2025-01-01 12:15:00")
        mock_news_calendar.get_upcoming_high_impact.return_value = [
            {"time": future, "title": "NFP Release"}
        ]
        with patch("src.core.circuit_breakers.time.time", return_value=1000000.0):
            status = cb.check_news_approaching(mock_news_calendar, "EURUSD", now)
        assert status.active is True
        assert "NFP Release" in status.reason
        assert "15min" in status.reason
        assert status.severity == "warning"

    def test_upcoming_news_beyond_buffer_returns_inactive(self, cb, mock_news_calendar):
        now = pd.Timestamp("2025-01-01 12:00:00")
        future = pd.Timestamp("2025-01-01 13:00:00")
        mock_news_calendar.get_upcoming_high_impact.return_value = [
            {"time": future, "title": "NFP Release"}
        ]
        status = cb.check_news_approaching(mock_news_calendar, "EURUSD", now)
        assert status.active is False

    def test_past_news_event_ignored(self, cb, mock_news_calendar):
        now = pd.Timestamp("2025-01-01 12:00:00")
        past = pd.Timestamp("2025-01-01 11:00:00")
        mock_news_calendar.get_upcoming_high_impact.return_value = [
            {"time": past, "title": "Old News"}
        ]
        status = cb.check_news_approaching(mock_news_calendar, "EURUSD", now)
        assert status.active is False

    def test_exception_during_news_check_caught(self, cb):
        nc = MagicMock()
        nc.is_high_impact_active.side_effect = RuntimeError("connection error")
        status = cb.check_news_approaching(nc, "EURUSD", pd.Timestamp("2025-01-01"))
        assert status.active is False


# ─────────────────────────────────────────────
# CHECK ALL
# ─────────────────────────────────────────────

class TestCheckAll:
    def test_normal_no_breakers_active(self, cb, df_flat, mock_news_calendar):
        state = cb.check_all("EURUSD", df_flat, mock_news_calendar, pd.Timestamp("2025-01-01"))
        assert state.any_active() is False
        assert state.volatility_spike.active is False
        assert state.news_approaching.active is False
        assert state.momentum_against.active is False
        assert state.correlation_breach.active is False

    def test_with_positions_triggers_momentum(self, cb, df_momentum_against_long, mock_news_calendar):
        positions = [{"type": "buy", "price_open": 100.0}]
        state = cb.check_all("EURUSD", df_momentum_against_long, mock_news_calendar,
                             pd.Timestamp("2025-01-01"), positions_info=positions)
        assert state.momentum_against.active is True
        assert state.any_active() is True

    def test_without_positions_skips_momentum(self, cb, df_momentum_against_long, mock_news_calendar):
        state = cb.check_all("EURUSD", df_momentum_against_long, mock_news_calendar,
                             pd.Timestamp("2025-01-01"))
        assert state.momentum_against.active is False

    def test_multiple_positions_breaks_on_first_active(self, cb, df_momentum_against_long, mock_news_calendar):
        positions = [
            {"type": "sell", "price_open": 100.0},
            {"type": "buy", "price_open": 100.0},
        ]
        state = cb.check_all("EURUSD", df_momentum_against_long, mock_news_calendar,
                             pd.Timestamp("2025-01-01"), positions_info=positions)
        assert state.momentum_against.active is True

    def test_volatility_spike_via_check_all(self, cb, df_flat, df_high_volatility, mock_news_calendar):
        for _ in range(4):
            cb.check_all("EURUSD", df_flat, mock_news_calendar, pd.Timestamp("2025-01-01"))
        with patch("src.core.circuit_breakers.time.time", return_value=1000000.0):
            state = cb.check_all("EURUSD", df_high_volatility, mock_news_calendar,
                                 pd.Timestamp("2025-01-01"))
        assert state.volatility_spike.active is True
        assert "ATR spike" in state.volatility_spike.reason
        assert state.any_active() is True

    def test_news_via_check_all(self, cb, df_flat):
        nc = MagicMock()
        nc.is_high_impact_active.return_value = True
        with patch("src.core.circuit_breakers.time.time", return_value=1000000.0):
            state = cb.check_all("EURUSD", df_flat, nc, pd.Timestamp("2025-01-01"))
        assert state.news_approaching.active is True
        assert "alto impacto" in state.news_approaching.reason
        assert state.any_active() is True

    def test_breaker_reasons_populated_on_trigger(self, cb, df_momentum_against_long):
        nc = MagicMock()
        nc.is_high_impact_active.return_value = True
        positions = [{"type": "buy", "price_open": 100.0}]
        with patch("src.core.circuit_breakers.time.time", return_value=1000000.0):
            state = cb.check_all("EURUSD", df_momentum_against_long, nc,
                                 pd.Timestamp("2025-01-01"), positions_info=positions)
        assert state.news_approaching.reason == "Noticia de alto impacto activa o próxima"
        assert "velas consecutivas" in state.momentum_against.reason
        assert state.news_approaching.severity == "critical"
        assert state.momentum_against.severity == "warning"


# ─────────────────────────────────────────────
# EDGE CASES
# ─────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_dataframe_returns_inactive(self, cb):
        df = pd.DataFrame()
        status = cb.check_volatility_spike("EURUSD", df)
        assert status.active is False

    def test_insufficient_data_for_atr(self, cb):
        df = pd.DataFrame({"high": [100.0] * 3, "low": [99.0] * 3, "close": [99.5] * 3})
        assert cb.check_volatility_spike("EURUSD", df).active is False

    def test_insufficient_data_for_momentum(self, cb):
        df = pd.DataFrame({"close": [100.0, 100.1]})
        assert cb.check_momentum_against("EURUSD", df, True, 100.0).active is False

    def test_extreme_values_do_not_crash(self, cb):
        df = pd.DataFrame({
            "high": [1e10] * 20, "low": [1e-10] * 20, "close": [1e5] * 20,
        })
        for _ in range(5):
            status = cb.check_volatility_spike("EURUSD", df)
        assert isinstance(status.active, bool)

    def test_symbol_isolation(self, cb, df_flat):
        cb.check_volatility_spike("EURUSD", df_flat)
        assert "EURUSD" in cb._atr_history
        assert "GBPUSD" not in cb._atr_history

    def test_check_news_with_get_upcoming_none(self, cb):
        nc = MagicMock()
        nc.is_high_impact_active.return_value = False
        nc.get_upcoming_high_impact = None
        status = cb.check_news_approaching(nc, "EURUSD", pd.Timestamp("2025-01-01"))
        assert status.active is False

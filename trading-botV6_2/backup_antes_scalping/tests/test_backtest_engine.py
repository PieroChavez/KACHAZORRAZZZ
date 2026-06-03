"""Tests for backtest_engine.py"""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.backtest.backtest_engine import BacktestEngine, BacktestResult
from src.backtest.backtest_metrics import BacktestMetrics
from src.backtest.simulated_executor import SimulatedTrade


@pytest.fixture
def sample_data():
    n = 120
    np.random.seed(42)
    base = 100.0
    idx = pd.date_range("2026-01-01 00:00", periods=n, freq="5min")
    closes = base + np.cumsum(np.random.randn(n) * 0.1)
    opens = closes - np.random.rand(n) * 0.1
    highs = np.maximum(opens, closes) + np.random.rand(n) * 0.15
    lows = np.minimum(opens, closes) - np.random.rand(n) * 0.15
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=idx,
    )


@pytest.fixture
def config():
    return {
        "strategy": {"adaptive": {"min_score_to_trade": 60, "min_net_score": 5}},
        "risk_per_trade": 0.02,
    }


@pytest.fixture
def engine(config):
    return BacktestEngine(config, symbol="XAUUSD")


@pytest.fixture
def standard_mocks():
    regime = MagicMock()
    regime.regime.value = "RANGING"
    regime.trend_alignment = "NEUTRAL"
    regime.adx_value = 25.0
    regime.confidence = 0.5

    decision = MagicMock()
    decision.should_trade = True

    profile = MagicMock()
    profile.label = "LONDON_OPEN"

    signal = MagicMock()
    signal.direction = "BUY"
    signal.entry_price = 100.0
    signal.stop_loss = 99.0
    signal.take_profit = 103.0
    signal.score = 75.0
    signal.conviction = 0.8
    signal.score_breakdown = {}
    signal.distribution = None
    signal.session_profile = None
    signal.consensus = None
    primary = MagicMock()
    primary.type.name = "FVG_BULLISH"
    signal.primary_pattern = primary

    return {
        "regime": regime,
        "decision": decision,
        "profile": profile,
        "signal": signal,
    }


def _apply_mocks(engine, mocks):
    """Apply all standard mocks to the engine and return a context manager stack."""
    return patch.multiple(
        engine.regime_detector, detect=MagicMock(return_value=mocks["regime"]),
    ), patch.multiple(
        engine.strategy_engine, evaluate_adaptive=MagicMock(return_value=mocks["signal"]),
    ), patch.multiple(
        engine.continuous_decider, decide=MagicMock(return_value=mocks["decision"]),
    ), patch.multiple(
        engine.session_profiler, profile=MagicMock(return_value=mocks["profile"]),
    ), patch.object(engine.kelly_risk, "get_risk_fraction", return_value=0.5), \
        patch.object(engine.dynamic_var, "get_risk_fraction", return_value=(0.02, "normal")), \
        patch.object(engine.volatility_scaler, "adjust_sl_tp", return_value=(98.5, 103.5)), \
        patch.object(engine.market_memory, "record_interaction"), \
        patch.object(engine.market_memory, "get_level_bias", return_value=None)


def _run_with_mocks(engine, data, mocks, **run_kwargs):
    patchers = _apply_mocks(engine, mocks)
    with patchers[0], patchers[1], patchers[2], patchers[3], \
         patchers[4], patchers[5], patchers[6], patchers[7], patchers[8]:
        return engine.run(data, **run_kwargs)


class TestBacktestResult:
    def test_all_fields_present(self):
        metrics = BacktestMetrics()
        trades = [
            SimulatedTrade(
                symbol="XAUUSD", direction="BUY", entry_time=datetime.now(),
                entry_price=100.0, volume=0.1, stop_loss=99.0, take_profit=103.0,
            ),
        ]
        result = BacktestResult(
            metrics=metrics,
            trades=trades,
            equity_curve=[10000.0, 10100.0],
            fitness=1.5,
            params={"risk_per_trade": 0.02},
            total_bars=50,
            execution_time_ms=123.4,
        )
        assert result.metrics is metrics
        assert result.trades == trades
        assert result.equity_curve == [10000.0, 10100.0]
        assert result.fitness == 1.5
        assert result.params == {"risk_per_trade": 0.02}
        assert result.total_bars == 50
        assert result.execution_time_ms == 123.4
        assert isinstance(result.metrics, BacktestMetrics)

    def test_defaults(self):
        metrics = BacktestMetrics()
        result = BacktestResult(
            metrics=metrics,
            trades=[],
            equity_curve=[10000.0],
        )
        assert result.fitness == 0.0
        assert result.params == {}
        assert result.total_bars == 0
        assert result.execution_time_ms == 0.0


class TestBacktestEngineInit:
    def test_initialization(self, engine):
        assert engine.symbol == "XAUUSD"
        assert engine.config is not None
        assert hasattr(engine, "strategy_engine")
        assert hasattr(engine, "regime_detector")
        assert hasattr(engine, "session_profiler")
        assert hasattr(engine, "continuous_decider")
        assert hasattr(engine, "market_memory")
        assert hasattr(engine, "kelly_risk")
        assert hasattr(engine, "volatility_scaler")
        assert hasattr(engine, "risk_manager")
        assert hasattr(engine, "dynamic_var")
        assert engine._last_regime is None
        assert engine._last_decision is None
        assert engine._signal_id_counter == 0

    def test_symbol_propagates_to_mock_profile(self, engine):
        profile = engine.strategy_engine.profile
        assert profile.symbol == "XAUUSD"


class TestRunEdgeCases:
    def test_empty_data_dict(self, engine):
        result = engine.run({})
        assert result.total_bars == 0
        assert result.trades == []
        assert result.equity_curve == [10000.0]
        assert result.metrics.total_trades == 0
        assert result.metrics.net_profit == 0.0

    def test_insufficient_rows(self, engine):
        n = 50
        idx = pd.date_range("2026-01-01", periods=n, freq="5min")
        small = pd.DataFrame(
            {"open": [100.0] * n, "high": [100.5] * n,
             "low": [99.5] * n, "close": [100.0] * n},
            index=idx,
        )
        result = engine.run({"5min": small})
        assert result.total_bars == 0
        assert result.trades == []
        assert result.metrics.total_trades == 0

    def test_data_without_recognised_timeframe(self, engine, sample_data):
        result = engine.run({"M1": sample_data})
        assert result.total_bars == 0
        assert result.trades == []


class TestRunTrades:
    def test_run_returns_backtest_result(self, engine, sample_data, standard_mocks):
        data = {"5min": sample_data, "1H": sample_data}
        result = _run_with_mocks(engine, data, standard_mocks)

        assert isinstance(result, BacktestResult)
        assert isinstance(result.metrics, BacktestMetrics)
        assert isinstance(result.trades, list)
        assert isinstance(result.equity_curve, list)
        assert len(result.equity_curve) >= 1
        assert result.total_bars > 0
        assert isinstance(result.fitness, float)

    def test_generates_trades(self, engine, sample_data, standard_mocks):
        data = {"5min": sample_data, "1H": sample_data}
        result = _run_with_mocks(engine, data, standard_mocks)

        assert len(result.trades) > 0
        for t in result.trades:
            assert isinstance(t, SimulatedTrade)
            assert t.symbol == "XAUUSD"
            assert t.direction in ("BUY", "SELL")
            assert t.entry_price > 0
            assert t.volume > 0

    def test_trades_have_exit_reason(self, engine, sample_data, standard_mocks):
        data = {"5min": sample_data, "1H": sample_data}
        result = _run_with_mocks(engine, data, standard_mocks)

        for t in result.trades:
            assert t.exit_reason in ("sl", "tp", "end", "signal")
            assert t.exit_time is not None
            assert t.exit_price is not None

    def test_equity_curve_structure(self, engine, sample_data, standard_mocks):
        data = {"5min": sample_data, "1H": sample_data}
        result = _run_with_mocks(engine, data, standard_mocks)

        eq = result.equity_curve
        assert isinstance(eq, list)
        assert len(eq) >= 1
        assert eq[0] == 10000.0
        assert all(isinstance(v, (int, float)) for v in eq)

    def test_max_trades_respected(self, engine, sample_data, standard_mocks):
        data = {"5min": sample_data, "1H": sample_data}
        result = _run_with_mocks(engine, data, standard_mocks, max_trades=3)

        assert len(result.trades) >= 1
        # The loop breaks when closed_trades reaches max_trades, then
        # close_all may add a few more from positions still open
        assert len(result.trades) <= 10

    def test_params_triggers_reconfigure(self, engine, sample_data):
        with patch.object(engine, "_pick_ltf", return_value=sample_data):
            with patch.object(engine, "_pick_htf", return_value=None):
                with patch.object(engine, "_reconfigure") as mock_reconfigure:
                    with patch(
                        "src.backtest.backtest_engine.SimulatedExecutor",
                    ) as MockExec:
                        instance = MockExec.return_value
                        instance.closed_trades = []
                        instance.equity_curve = [10000.0]

                        params = {"risk_per_trade": 0.01}
                        engine.run({"5min": sample_data}, params=params)

                        mock_reconfigure.assert_called_once()
                        merged = mock_reconfigure.call_args[0][0]
                        assert merged["risk_per_trade"] == 0.01
                        assert merged["strategy"] == engine.config["strategy"]

    def test_run_without_params_skips_reconfigure(self, engine, sample_data):
        with patch.object(engine, "_pick_ltf", return_value=sample_data):
            with patch.object(engine, "_pick_htf", return_value=None):
                with patch.object(engine, "_reconfigure") as mock_reconfigure:
                    with patch(
                        "src.backtest.backtest_engine.SimulatedExecutor",
                    ) as MockExec:
                        instance = MockExec.return_value
                        instance.closed_trades = []
                        instance.equity_curve = [10000.0]

                        engine.run({"5min": sample_data})

                        mock_reconfigure.assert_not_called()

    def test_metrics_computed_from_trades(self, engine, sample_data, standard_mocks):
        data = {"5min": sample_data, "1H": sample_data}
        result = _run_with_mocks(engine, data, standard_mocks)

        m = result.metrics
        assert m.total_trades == len(result.trades)
        assert m.winning_trades + m.losing_trades == m.total_trades
        assert m.win_rate == (
            m.winning_trades / m.total_trades if m.total_trades > 0 else 0.0
        )
        assert m.profit_factor >= 0.0
        assert m.max_drawdown >= 0.0
        assert m.max_drawdown_pct >= 0.0
        assert m.sharpe_ratio <= 10.0
        assert isinstance(m.total_return_pct, float)

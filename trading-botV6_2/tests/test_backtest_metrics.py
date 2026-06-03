import numpy as np
import pytest
from datetime import datetime

from src.backtest.backtest_metrics import compute_metrics, BacktestMetrics, metrics_to_fitness
from src.backtest.simulated_executor import SimulatedTrade


def _trade(**kwargs) -> SimulatedTrade:
    defaults = dict(
        symbol="EURUSD",
        direction="BUY",
        entry_time=datetime(2024, 1, 1),
        entry_price=1.1000,
        volume=0.1,
        stop_loss=1.0950,
        take_profit=1.1100,
        exit_time=datetime(2024, 1, 2),
        exit_price=1.1050,
        exit_reason="tp",
        profit=50.0,
        profit_pct=0.5,
        bars_held=5,
        pattern_type="FVG",
        regime="TRENDING",
        session="LONDON",
        score=0.8,
        conviction=0.7,
    )
    defaults.update(kwargs)
    return SimulatedTrade(**defaults)


class TestCoreMetrics:
    def test_empty_trades_returns_defaults(self):
        m = compute_metrics([], [1000.0], 1000.0, 0)
        assert m.total_trades == 0
        assert m.win_rate == 0.0
        assert m.profit_factor == 0.0
        assert m.sharpe_ratio == 0.0
        assert m.sortino_ratio == 0.0
        assert m.max_drawdown == 0.0
        assert m.max_drawdown_pct == 0.0
        assert m.net_profit == 0.0
        assert m.expectancy == 0.0
        assert m.kelly_fraction == 0.0
        assert m.recovery_factor == 0.0

    def test_win_rate_and_profit_factor(self):
        trades = [
            _trade(profit=150.0, exit_reason="tp"),
            _trade(profit=250.0, exit_reason="tp"),
            _trade(profit=-50.0, exit_reason="sl"),
            _trade(profit=-50.0, exit_reason="sl"),
        ]
        m = compute_metrics(trades, [1000.0, 1300.0], 1000.0, 10)
        assert m.total_trades == 4
        assert m.winning_trades == 2
        assert m.losing_trades == 2
        assert m.win_rate == 0.5
        assert m.gross_profit == 400.0
        assert m.gross_loss == 100.0
        assert m.profit_factor == 4.0
        assert m.net_profit == 300.0
        assert m.avg_win == 200.0
        assert m.avg_loss == 50.0
        assert m.avg_bars_held == 5.0

    def test_sharpe_ratio_calculation(self):
        equity = [1000.0, 1010.0, 1020.0, 1015.0, 1025.0, 1030.0]
        returns = np.diff(equity) / np.array(equity[:-1])
        expected = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
        m = compute_metrics([_trade(profit=30.0)], equity, 1000.0, 10)
        assert m.sharpe_ratio == pytest.approx(expected, abs=1e-6)
        assert m.sharpe_ratio > 0

    def test_sortino_ratio_uses_downside_only(self):
        equity = [1000.0, 1020.0, 990.0, 1030.0, 1010.0, 1040.0]
        returns = np.diff(equity) / np.array(equity[:-1])
        expected_sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
        downside = returns[returns < 0]
        expected_sortino = float(np.mean(returns) / np.std(downside) * np.sqrt(252))
        m = compute_metrics([_trade(profit=40.0)], equity, 1000.0, 10)
        assert m.sortino_ratio == pytest.approx(expected_sortino, abs=1e-6)
        assert m.sortino_ratio != expected_sharpe

    def test_max_drawdown_peak_to_trough(self):
        equity = [1000.0, 1100.0, 1200.0, 900.0, 950.0, 1150.0]
        m = compute_metrics([_trade(profit=150.0)], equity, 1000.0, 10)
        assert m.max_drawdown == 300.0
        assert m.max_drawdown_pct == pytest.approx(0.25)

    def test_expectancy_value(self):
        trades = [
            _trade(profit=100.0),
            _trade(profit=200.0),
            _trade(profit=-50.0),
            _trade(profit=-50.0),
        ]
        m = compute_metrics(trades, [1000.0, 1200.0], 1000.0, 10)
        assert m.expectancy == 50.0

    def test_total_return_pct(self):
        m = compute_metrics([_trade(profit=50.0)], [1000.0, 1050.0], 1000.0, 10)
        assert m.total_return_pct == 5.0

    def test_kelly_fraction(self):
        trades = [
            _trade(profit=100.0),
            _trade(profit=100.0),
            _trade(profit=-50.0),
        ]
        m = compute_metrics(trades, [1000.0, 1150.0], 1000.0, 10)
        assert m.kelly_fraction == pytest.approx(0.5005, abs=0.001)
        assert m.kelly_fraction > 0

    def test_recovery_factor(self):
        equity = [1000.0, 1100.0, 1200.0, 800.0, 1000.0]
        trades = [_trade(profit=-200.0)]
        m = compute_metrics(trades, equity, 1000.0, 10)
        assert m.net_profit == -200.0
        assert m.max_drawdown == 400.0
        assert m.recovery_factor == -0.5

    def test_consecutive_wins_and_losses(self):
        trades = [
            _trade(profit=100.0),
            _trade(profit=50.0),
            _trade(profit=-30.0),
            _trade(profit=40.0),
            _trade(profit=60.0),
            _trade(profit=20.0),
            _trade(profit=-10.0),
        ]
        m = compute_metrics(trades, [1000.0, 1230.0], 1000.0, 10)
        assert m.consecutive_wins == 3
        assert m.consecutive_losses == 1

    def test_avg_risk_reward(self):
        trades = [
            _trade(direction="BUY", entry_price=100.0, stop_loss=95.0, take_profit=115.0),
            _trade(direction="SELL", entry_price=100.0, stop_loss=105.0, take_profit=90.0),
        ]
        m = compute_metrics(trades, [1000.0, 1000.0], 1000.0, 10)
        assert m.avg_risk_reward == pytest.approx(2.5)

    def test_monthly_return_not_computed_stays_zero(self):
        m = compute_metrics([_trade(profit=100.0)], [1000.0, 1100.0], 1000.0, 10)
        assert m.monthly_return_pct == 0.0


class TestEdgeCases:
    def test_only_wins_no_losses(self):
        trades = [_trade(profit=100.0), _trade(profit=50.0)]
        m = compute_metrics(trades, [1000.0, 1150.0], 1000.0, 10)
        assert m.win_rate == 1.0
        assert m.losing_trades == 0
        assert m.gross_loss == 0.0
        assert m.profit_factor == 0.0
        assert m.avg_loss == 0.0
        assert m.net_profit == 150.0

    def test_only_losses_no_wins(self):
        trades = [_trade(profit=-100.0), _trade(profit=-50.0)]
        m = compute_metrics(trades, [1000.0, 850.0], 1000.0, 10)
        assert m.win_rate == 0.0
        assert m.winning_trades == 0
        assert m.gross_profit == 0.0
        assert m.profit_factor == 0.0
        assert m.avg_win == 0.0
        assert m.net_profit == -150.0

    def test_single_winning_trade(self):
        m = compute_metrics([_trade(profit=75.0)], [1000.0, 1075.0], 1000.0, 10)
        assert m.total_trades == 1
        assert m.win_rate == 1.0
        assert m.gross_profit == 75.0
        assert m.net_profit == 75.0
        assert m.expectancy == 75.0

    def test_single_losing_trade(self):
        m = compute_metrics([_trade(profit=-75.0)], [1000.0, 925.0], 1000.0, 10)
        assert m.total_trades == 1
        assert m.win_rate == 0.0
        assert m.gross_loss == 75.0
        assert m.net_profit == -75.0
        assert m.expectancy == -75.0

    def test_zero_return_flat_equity(self):
        m = compute_metrics([_trade(profit=0.0)], [1000.0, 1000.0], 1000.0, 10)
        assert m.total_return_pct == 0.0
        assert m.sharpe_ratio == 0.0
        assert m.max_drawdown == 0.0

    def test_drawdown_without_recovery(self):
        equity = [1000.0, 900.0, 800.0, 700.0, 600.0]
        m = compute_metrics([_trade(profit=-400.0)], equity, 1000.0, 10)
        assert m.max_drawdown == 400.0
        assert m.max_drawdown_pct == pytest.approx(0.4)
        assert m.recovery_factor == -1.0

    def test_breakeven_trades(self):
        trades = [_trade(profit=50.0), _trade(profit=-50.0)]
        m = compute_metrics(trades, [1000.0, 1000.0], 1000.0, 10)
        assert m.net_profit == 0.0
        assert m.total_return_pct == 0.0
        assert m.profit_factor == 1.0

    def test_sharpe_zero_when_returns_constant(self):
        equity = [1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
        m = compute_metrics([_trade(profit=0.0)], equity, 1000.0, 10)
        assert m.sharpe_ratio == 0.0
        assert m.sortino_ratio == 0.0

    def test_max_drawdown_no_peak_below_start(self):
        equity = [1000.0, 900.0, 950.0, 850.0, 800.0]
        m = compute_metrics([_trade(profit=-200.0)], equity, 1000.0, 10)
        assert m.max_drawdown == 200.0
        assert m.max_drawdown_pct == pytest.approx(0.2, abs=1e-6)


class TestMetricsToFitness:
    def test_fitness_basic_scoring(self):
        m = BacktestMetrics(
            sharpe_ratio=1.5,
            profit_factor=2.0,
            win_rate=0.6,
            total_return_pct=15.0,
            max_drawdown_pct=0.1,
            total_trades=50,
            avg_risk_reward=1.5,
        )
        score = metrics_to_fitness(m)
        assert score > 0
        assert isinstance(score, float)

    def test_fitness_all_zero(self):
        m = BacktestMetrics()
        score = metrics_to_fitness(m)
        assert score == 0.0

    def test_fitness_custom_weights(self):
        m = BacktestMetrics(sharpe_ratio=2.0, profit_factor=3.0)
        weights = {"sharpe": 1.0, "profit_factor": 1.0}
        score = metrics_to_fitness(m, weights=weights)
        assert score == 5.0

    def test_fitness_clips_extreme_values(self):
        m = BacktestMetrics(
            sharpe_ratio=100.0,
            profit_factor=100.0,
            total_return_pct=10000.0,
            max_drawdown_pct=10.0,
            total_trades=10000,
            avg_risk_reward=100.0,
            win_rate=0.0,
        )
        score = metrics_to_fitness(m)
        assert score < 1000


class TestSegmentBreakdowns:
    def test_by_pattern(self):
        trades = [
            _trade(profit=100.0, pattern_type="FVG"),
            _trade(profit=-30.0, pattern_type="FVG"),
            _trade(profit=50.0, pattern_type="BREAKOUT"),
        ]
        m = compute_metrics(trades, [1000.0, 1120.0], 1000.0, 10)
        assert m.by_pattern["FVG"]["trades"] == 2
        assert m.by_pattern["FVG"]["win_rate"] == 0.5
        assert m.by_pattern["FVG"]["profit"] == 70.0
        assert m.by_pattern["BREAKOUT"]["profit"] == 50.0
        assert m.by_pattern["BREAKOUT"]["wins"] == 1

    def test_by_regime(self):
        trades = [
            _trade(profit=100.0, regime="TRENDING"),
            _trade(profit=-20.0, regime="TRENDING"),
            _trade(profit=30.0, regime="RANGING"),
        ]
        m = compute_metrics(trades, [1000.0, 1110.0], 1000.0, 10)
        assert m.by_regime["TRENDING"]["trades"] == 2
        assert m.by_regime["TRENDING"]["win_rate"] == 0.5
        assert m.by_regime["RANGING"]["wins"] == 1

    def test_by_session_and_exit_reason(self):
        trades = [
            _trade(profit=100.0, session="LONDON", exit_reason="tp"),
            _trade(profit=-50.0, session="NEW_YORK", exit_reason="sl"),
        ]
        m = compute_metrics(trades, [1000.0, 1050.0], 1000.0, 10)
        assert m.by_session["LONDON"]["profit"] == 100.0
        assert m.by_session["NEW_YORK"]["losses"] == 1
        assert m.by_exit_reason["tp"]["wins"] == 1
        assert m.by_exit_reason["sl"]["losses"] == 1

    def test_unknown_segment_defaults(self):
        trades = [_trade(profit=50.0, pattern_type=None, regime="", session="", exit_reason="")]
        m = compute_metrics(trades, [1000.0, 1050.0], 1000.0, 10)
        assert m.by_pattern["UNKNOWN"]["trades"] == 1
        assert m.by_regime["UNKNOWN"]["trades"] == 1
        assert m.by_session["UNKNOWN"]["trades"] == 1
        assert m.by_exit_reason["UNKNOWN"]["trades"] == 1

    def test_segment_win_rate_zero_when_no_trades(self):
        m = compute_metrics([], [1000.0], 1000.0, 0)
        assert m.by_pattern == {}
        assert m.by_regime == {}
        assert m.by_session == {}
        assert m.by_exit_reason == {}

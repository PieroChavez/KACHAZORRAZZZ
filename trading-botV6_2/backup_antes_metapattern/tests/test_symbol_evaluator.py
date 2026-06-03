from datetime import datetime
from unittest.mock import MagicMock, ANY
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.core.symbol_evaluator import SymbolEvaluator
from src.executor.order_types import OrderType


# ── Helpers ──

def _make_tf_df(n=100, start=1.0):
    """Create a simple OHLC DataFrame with n rows."""
    opens = np.linspace(start, start + n * 0.1, n)
    closes = opens + 0.2
    highs = closes + 0.3
    lows = opens - 0.3
    return pd.DataFrame({
        "open": opens, "close": closes,
        "high": highs, "low": lows,
    })


def _timeframes(n_tfs=5):
    """Dict of timeframe DataFrames for fetcher."""
    base = 100.0
    tfs = {
        "1min": _make_tf_df(100, base),
        "3min": _make_tf_df(80, base),
        "5min": _make_tf_df(60, base),
        "15min": _make_tf_df(50, base),
        "30min": _make_tf_df(40, base),
    }
    return {k: v for k, v in [*list(tfs.items())[:n_tfs]]}


def _make_engine():
    engine = MagicMock()
    engine.evaluate_adaptive.return_value = _mock_signal()
    engine._pick_tf = MagicMock(return_value=_make_tf_df(60))
    engine.tf_groups = SimpleNamespace()
    return engine


def _sym_data():
    return {
        "engine": _make_engine(),
        "profile": SimpleNamespace(
            allowed_sessions=_in_session_hours(),
            scale_entries=1,
            cooldown_bars_m1=0,
            max_concurrent_trades=1,
            max_volume=0.5,
            sl_min_pips=5.0,
            sl_max_pips=35.0,
            min_rr_ratio=1.5,
        ),
        "pip_value": 1.0,
        "last_trade_time": None,
    }


def _in_session_hours(*hours):
    """Return allowed_sessions that match the given UTC hours (default: match all)."""
    if not hours:
        return [(h, h + 1) for h in range(24)]
    return [(h, h + 1) for h in hours]


def _mock_regime():
    return SimpleNamespace(
        regime=SimpleNamespace(value="TREND"),
        atr_ratio=1.0,
        adx_value=25.0,
        confidence=0.6,
        trend_alignment="BULLISH_ALIGNED",
        session_profile=SimpleNamespace(label="LONDON"),
    )


def _mock_signal(direction="BUY", score=60.0, conviction=0.6):
    sig = MagicMock()
    sig.direction = direction
    sig.score = score
    sig.conviction = conviction
    sig.entry_price = 100.0
    sig.stop_loss = 99.0
    sig.take_profit = 102.0
    sig.notes = ["test signal"]
    sig.session_profile = SimpleNamespace(label="LONDON")
    sig.regime_context = SimpleNamespace(
        trend_alignment="BULLISH_ALIGNED",
        adx_value=25.0,
        regime=SimpleNamespace(value="TREND"),
        confidence=0.6,
    )
    sig.primary_pattern = SimpleNamespace(
        type=SimpleNamespace(name="TEST_PATTERN"),
        low=99.5,
        high=100.5,
    )
    sig.distribution = SimpleNamespace(
        mean=60.0, std=10.0, convergence=0.8,
    )
    sig.score_breakdown = {}
    sig.consensus = None
    return sig


def _mock_signal_gap(direction="BUY"):
    sig = _mock_signal(direction)
    sig.primary_pattern = SimpleNamespace(
        type=SimpleNamespace(name="FVG_BULLISH"),
        low=99.0,
        high=99.5,
    )
    return sig


def _register_symbol(mock_bot, symbol):
    """Ensure symbol exists in symbols dict."""
    if not hasattr(mock_bot, "symbols") or mock_bot.symbols is None:
        mock_bot.symbols = {}
    if symbol not in mock_bot.symbols:
        mock_bot.symbols[symbol] = SimpleNamespace()


# ── Mock bot factory ──

_UID = 0


def make_mock_bot():
    """Create a MagicMock TradingBot with all required subsystems stubbed."""
    global _UID
    _UID += 1

    engine = MagicMock()
    sig = _mock_signal()
    engine.evaluate_adaptive.return_value = sig

    cd_eval = SimpleNamespace(
        active=False, reason=SimpleNamespace(value=""),
        remaining_minutes=0, notes=[],
        recommended_volume_mult=1.0,
        recommended_min_conviction=0.0,
        recovery_stage=SimpleNamespace(value="NORMAL"),
    )

    bot = MagicMock()
    bot._cycle_cache = {}
    bot._positions_cache = []
    bot._last_regime = {}
    bot._last_md_detections = {}
    bot._last_decision = {}
    bot._signal_id_counter = 0
    bot._last_dxy_trend = ""
    bot.expert_mode = False
    bot._adaptive_cooldown_enabled = True
    bot._session_vol_mult = 1.0
    bot._session_scale_n = 1
    bot._cd_volume_mult = 1.0
    bot._cd_min_conviction = 0.0
    bot._pending_batches = {}
    bot.config = {"strategy": {"adaptive": {"min_conviction_to_trade": 0.15}, "params": {"max_spread_pips": 15.0}}}
    bot.symbols = {}
    bot.pending_orders = {}
    bot.high_confidence_score = 80.0
    bot.params = SimpleNamespace(min_rr_ratio=1.5, min_stop_distance_atr=2.0)
    bot.meta_learner = MagicMock()
    bot.market_maps = {}
    regime_mock = MagicMock()
    regime_mock.detect.return_value = _mock_regime()
    bot.regime_detectors = {"XAUUSD": regime_mock, "EURUSD": regime_mock}
    bot.market_memory = MagicMock()
    bot.market_memory.get_level_bias.return_value = None
    bot.fetcher = MagicMock()
    bot.fetcher.get_dataframes.return_value = _timeframes()
    bot.tf_optimizer = MagicMock()
    bot.tf_optimizer.analyze.return_value = SimpleNamespace(
        groups=MagicMock(), htf="15min", ltf="5min",
        volatility_regime="MEDIUM", degraded_tfs=None,
    )
    bot.order_flow = MagicMock()
    bot.order_flow.analyze.return_value = MagicMock()
    bot.order_flow.get_signal_contribution.return_value = (0.0, "")
    bot.correlation_engine = MagicMock()
    bot.correlation_engine.confirm_signal.return_value = (True, "")
    bot.correlation_engine.volume_adjustment.return_value = (1.0, "")
    bot.context_oracle = MagicMock()
    bot.context_oracle.build_context_vector.return_value = MagicMock()
    bot.context_oracle.predict_direction.return_value = ("HOLD", 0.0, 0.0)
    bot.adaptive_cooldown = MagicMock()
    bot.adaptive_cooldown.get_streak.return_value = SimpleNamespace(consecutive_losses=0)
    bot.adaptive_cooldown.evaluate.return_value = cd_eval
    bot.session_profiler = MagicMock()
    bot.session_profiler.profile.return_value = SimpleNamespace(
        label="LONDON", volume_adjustment=1.0,
        avoided_patterns=[],
    )
    bot.continuous_decider = MagicMock()
    bot.continuous_decider.decide.return_value = SimpleNamespace(
        suggested_volume_pct=1.0, conviction=0.6,
        sl_width_multiplier=1.0, tp_width_multiplier=1.0,
        should_trade=True,
    )
    engine._pick_tf = MagicMock(return_value=_make_tf_df(60))
    engine.tf_groups = SimpleNamespace()
    bot._manage_pending_orders = MagicMock()
    bot._manage_symbol_position = MagicMock()
    bot.mt5 = MagicMock()
    bot.mt5.get_positions.return_value = []
    bot.mt5.get_symbol_info.return_value = {
        "spread": 2, "digits": 5, "bid": 100.0, "ask": 100.1,
    }
    bot.circuit_breakers = MagicMock()
    bot.circuit_breakers.check_all.return_value = SimpleNamespace(
        any_active=lambda: False,
        highest_severity=lambda: "",
    )
    bot.news_calendar = MagicMock()
    bot.news_calendar.is_high_impact_active.return_value = False
    bot.risk_manager = MagicMock()
    bot.risk_manager.config = SimpleNamespace(
        atr_multiplier_sl=1.5, atr_multiplier_tp=2.0,
    )
    bot.volatility_scaler = MagicMock()
    bot.volatility_scaler.adjust_sl_tp.return_value = (99.0, 102.0, {"note": "vol scaled"})
    bot.portfolio_risk = MagicMock()
    bot.portfolio_risk.pre_check.return_value = (True, "", 1.0)
    bot.portfolio_risk.update_positions = MagicMock()
    bot.order_selector = MagicMock()
    bot.order_selector.select.return_value = SimpleNamespace(
        entry_price=100.0, confidence_adjustment=1.0,
        order_type=OrderType.MARKET,
        reason="market entry",
    )
    bot.kelly_risk = MagicMock()
    bot.kelly_risk.get_adjusted_volume_mult.return_value = 1.0
    bot._calc_volume = MagicMock(return_value=0.1)
    bot.executor = MagicMock()
    bot.executor.place_market_order.return_value = SimpleNamespace(
        success=True, message="OK", order_ticket=1000 + _UID,
    )
    bot.executor.place_pending_limit.return_value = SimpleNamespace(
        success=True, message="OK", order_ticket=2000 + _UID,
    )
    bot.executor.place_order.return_value = SimpleNamespace(
        success=True, message="OK", order_ticket=3000 + _UID,
    )

    return bot


# ── Fixtures ──

@pytest.fixture
def evaluator():
    return SymbolEvaluator()


@pytest.fixture
def bot():
    return make_mock_bot()


@pytest.fixture
def sym_data():
    return _sym_data()


# ── Tests ──

class TestEarlyReturns:
    def test_insufficient_timeframes(self, evaluator, bot, sym_data):
        bot.fetcher.get_dataframes.return_value = {"1min": pd.DataFrame()}
        evaluator.evaluate(bot, "XAUUSD", sym_data)
        bot.executor.place_market_order.assert_not_called()

    def test_no_ltf_50_bars(self, evaluator, bot, sym_data):
        tfs = {
            "1min": _make_tf_df(10),
            "5min": _make_tf_df(5),
            "15min": _make_tf_df(3),
        }
        bot.fetcher.get_dataframes.return_value = tfs
        evaluator.evaluate(bot, "XAUUSD", sym_data)
        bot.executor.place_market_order.assert_not_called()


class TestMarketMapFilter:
    def test_market_map_no_trade(self, evaluator, bot, sym_data):
        mm = MagicMock()
        mm.evaluate.return_value = SimpleNamespace(
            decision="NO_TRADE", notes=["filtered"],
            direction="BUY", md_detections=[],
            score_bonus=0.0, confidence=0.0,
            phase=None,
        )
        bot.market_maps["XAUUSD"] = mm
        evaluator.evaluate(bot, "XAUUSD", sym_data)
        bot.meta_learner.record_skipped_signal.assert_called_once()
        bot.executor.place_market_order.assert_not_called()

    def test_market_map_wait_phase_no_entry(self, evaluator, bot, sym_data):
        mm = MagicMock()
        mm.evaluate.return_value = SimpleNamespace(
            decision="WAIT", notes=["waiting"],
            direction="BUY", md_detections=[],
            score_bonus=0.0, confidence=0.0,
            phase=SimpleNamespace(allows_entry=False),
        )
        bot.market_maps["XAUUSD"] = mm
        evaluator.evaluate(bot, "XAUUSD", sym_data)
        bot.executor.place_market_order.assert_not_called()


class TestCooldown:
    def test_cooldown_active_zero_volume(self, evaluator, bot, sym_data):
        bot.adaptive_cooldown.evaluate.return_value = SimpleNamespace(
            active=True, reason=SimpleNamespace(value="RECENT_LOSS"),
            remaining_minutes=30, notes=["cooling"],
            recommended_volume_mult=0.0,
            recommended_min_conviction=0.0,
            recovery_stage=SimpleNamespace(value="RECOVERY"),
        )
        evaluator.evaluate(bot, "XAUUSD", sym_data)
        bot.meta_learner.record_skipped_signal.assert_called_once()
        bot.executor.place_market_order.assert_not_called()


class TestDirectionFilter:
    def test_hold_direction(self, evaluator, bot, sym_data):
        engine = sym_data["engine"]
        sig = _mock_signal(direction="HOLD")
        engine.evaluate_adaptive.return_value = sig
        evaluator.evaluate(bot, "XAUUSD", sym_data)
        bot.meta_learner.record_skipped_signal.assert_called_once()
        bot.executor.place_market_order.assert_not_called()


class TestConvictionFilter:
    def test_low_conviction(self, evaluator, bot, sym_data):
        engine = sym_data["engine"]
        sig = _mock_signal(direction="BUY", conviction=0.05)
        engine.evaluate_adaptive.return_value = sig
        evaluator.evaluate(bot, "XAUUSD", sym_data)
        bot.meta_learner.record_skipped_signal.assert_called_once()
        bot.executor.place_market_order.assert_not_called()


class TestDecisionFilter:
    def test_should_trade_false(self, evaluator, bot, sym_data):
        bot.continuous_decider.decide.return_value = SimpleNamespace(
            suggested_volume_pct=1.0, conviction=0.6,
            sl_width_multiplier=1.0, tp_width_multiplier=1.0,
            should_trade=False,
        )
        evaluator.evaluate(bot, "XAUUSD", sym_data)
        bot.meta_learner.record_skipped_signal.assert_called_once()
        bot.executor.place_market_order.assert_not_called()


class TestSessionFilter:
    def test_outside_trading_session(self, evaluator, bot, sym_data):
        sym_data["profile"].allowed_sessions = _in_session_hours(20)  # current hour ~4, not 20
        bot.session_profiler.profile.return_value = SimpleNamespace(
            label="TOKYO", volume_adjustment=0.5,
            avoided_patterns=[],
        )
        evaluator.evaluate(bot, "XAUUSD", sym_data)
        bot.meta_learner.record_skipped_signal.assert_called_once()
        bot.executor.place_market_order.assert_not_called()


class TestAvoidedPattern:
    def test_avoided_pattern_low_conviction(self, evaluator, bot, sym_data):
        sym_data["engine"].evaluate_adaptive.return_value = _mock_signal(
            direction="BUY", conviction=0.2, score=40.0,
        )
        bot.session_profiler.profile.return_value = SimpleNamespace(
            label="LONDON", volume_adjustment=1.0,
            avoided_patterns=["TEST_PATTERN"],
        )
        evaluator.evaluate(bot, "XAUUSD", sym_data)
        bot.meta_learner.record_skipped_signal.assert_called_once()
        bot.executor.place_market_order.assert_not_called()


class TestCircuitBreaker:
    def test_circuit_breaker_critical(self, evaluator, bot, sym_data):
        bot.circuit_breakers.check_all.return_value = SimpleNamespace(
            any_active=lambda: True,
            highest_severity=lambda: "critical",
            volatility_spike=SimpleNamespace(active=True, reason="vol spike"),
            news_approaching=SimpleNamespace(active=False, reason=""),
            momentum_against=SimpleNamespace(active=False, reason=""),
        )
        evaluator.evaluate(bot, "XAUUSD", sym_data)
        bot.meta_learner.record_skipped_signal.assert_called_once()
        bot.executor.place_market_order.assert_not_called()


class TestSpreadFilter:
    def test_spread_too_high(self, evaluator, bot, sym_data):
        bot.mt5.get_symbol_info.return_value = {
            "spread": 200, "digits": 5, "bid": 100.0, "ask": 100.1,
        }
        bot.config["strategy"]["params"]["max_spread_pips"] = 0.01
        evaluator.evaluate(bot, "XAUUSD", sym_data)
        bot.executor.place_market_order.assert_not_called()


class TestCorrelationBlock:
    def test_correlation_blocks(self, evaluator, bot, sym_data):
        # Override engine to clear sym_data profile issue
        bot.correlation_engine.confirm_signal.return_value = (False, "correlation block")
        evaluator.evaluate(bot, "XAUUSD", sym_data)
        bot.executor.place_market_order.assert_not_called()


class TestPortfolioRisk:
    def test_portfolio_risk_blocks(self, evaluator, bot, sym_data):
        bot.portfolio_risk.pre_check.return_value = (False, "too much risk", 0.0)
        evaluator.evaluate(bot, "XAUUSD", sym_data)
        bot.executor.place_market_order.assert_not_called()


class TestMaxPositions:
    def test_max_positions_no_pyramid(self, evaluator, bot, sym_data):
        bot.mt5.get_positions.return_value = [
            {"ticket": 1, "symbol": "XAUUSD", "type": "buy",
             "volume": 0.1, "price_open": 100.0},
        ]
        sym_data["profile"].max_concurrent_trades = 1
        evaluator.evaluate(bot, "XAUUSD", sym_data)
        bot.executor.place_market_order.assert_not_called()


class TestHappyPath:
    def test_market_order_placed(self, evaluator, bot, sym_data):
        evaluator.evaluate(bot, "XAUUSD", sym_data)
        bot.executor.place_market_order.assert_called_once()

    def test_market_order_volume(self, evaluator, bot, sym_data):
        evaluator.evaluate(bot, "XAUUSD", sym_data)
        args, _ = bot.executor.place_market_order.call_args
        sig, volume, price = args[0], args[1], args[2] if len(args) >= 3 else None
        assert volume == 0.1

    def test_market_order_updates_last_trade_time(self, evaluator, bot, sym_data):
        evaluator.evaluate(bot, "XAUUSD", sym_data)
        assert sym_data["last_trade_time"] is not None
        assert isinstance(sym_data["last_trade_time"], datetime)

    def test_signal_passed_to_executor(self, evaluator, bot, sym_data):
        engine = sym_data["engine"]
        sig = _mock_signal(direction="BUY", score=75.0)
        engine.evaluate_adaptive.return_value = sig
        evaluator.evaluate(bot, "XAUUSD", sym_data)
        bot.executor.place_market_order.assert_called_once()
        args, _ = bot.executor.place_market_order.call_args
        assert args[0].score == 75.0

    def test_regime_alignment_bonus(self, evaluator, bot, sym_data):
        evaluator.evaluate(bot, "XAUUSD", sym_data)
        assert True  # no crash

    def test_gap_pattern_places_limit(self, evaluator, bot, sym_data):
        sym_data["profile"].sl_max_pips = 1.0  # gap_distance > max_dist → enters LIMIT branch
        engine = sym_data["engine"]
        sig = _mock_signal_gap("BUY")
        engine.evaluate_adaptive.return_value = sig
        evaluator.evaluate(bot, "XAUUSD", sym_data)
        bot.executor.place_pending_limit.assert_called_once()

    def test_correlation_volume_adjustment(self, evaluator, bot, sym_data):
        bot.correlation_engine.volume_adjustment.return_value = (0.5, "correlated pair")
        evaluator.evaluate(bot, "XAUUSD", sym_data)
        bot.executor.place_market_order.assert_called_once()
        args, _ = bot.executor.place_market_order.call_args
        assert args[1] < 0.1  # volume reduced

    def test_portfolio_risk_volume_reduction(self, evaluator, bot, sym_data):
        bot.portfolio_risk.pre_check.return_value = (True, "reduced", 0.5)
        evaluator.evaluate(bot, "XAUUSD", sym_data)
        bot.executor.place_market_order.assert_called_once()
        args, _ = bot.executor.place_market_order.call_args
        assert args[1] <= 0.05  # 0.1 * 0.5 = 0.05


class TestScaleIn:
    def test_scale_in_places_multiple_orders(self, evaluator, bot, sym_data):
        sym_data["profile"].scale_entries = 3
        bot.order_selector.select.return_value.order_type = OrderType.LIMIT
        evaluator.evaluate(bot, "XAUUSD", sym_data)
        assert bot.executor.place_order.call_count == 3

    def test_scale_in_updates_pending_batches(self, evaluator, bot, sym_data):
        sym_data["profile"].scale_entries = 3
        bot.order_selector.select.return_value.order_type = OrderType.LIMIT
        evaluator.evaluate(bot, "XAUUSD", sym_data)
        assert len(bot._pending_batches.get("XAUUSD", {})) > 0


class TestPendingOrderDedup:
    def test_duplicate_pending_avoids(self, evaluator, bot, sym_data):
        bot.order_selector.select.return_value.order_type = OrderType.LIMIT
        bot.pending_orders[500] = {
            "symbol": "XAUUSD", "direction": "BUY",
            "poi_level": 100.0,
        }
        evaluator.evaluate(bot, "XAUUSD", sym_data)
        bot.executor.cancel_pending_order.assert_not_called()

    def test_pending_entry_price_changed(self, evaluator, bot, sym_data):
        bot.order_selector.select.return_value.order_type = OrderType.LIMIT
        bot.pending_orders[500] = {
            "symbol": "XAUUSD", "direction": "BUY",
            "poi_level": 50.0,
        }
        evaluator.evaluate(bot, "XAUUSD", sym_data)
        bot.executor.cancel_pending_order.assert_called_once()


class TestWidenRunners:
    def test_pyramid_entry_widens_tp(self, evaluator, bot, sym_data):
        bot.order_selector.select.return_value.order_type = OrderType.LIMIT
        sym_data["profile"].max_concurrent_trades = 2
        sym_data["profile"].scale_entries = 3
        bot.mt5.get_positions.return_value = [
            {"ticket": 1, "symbol": "XAUUSD", "type": "buy",
             "volume": 0.1, "price_open": 100.0},
        ]
        bot._widen_runners_tp = MagicMock()
        evaluator.evaluate(bot, "XAUUSD", sym_data)
        bot._widen_runners_tp.assert_called()

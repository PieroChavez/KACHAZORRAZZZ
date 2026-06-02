import pytest
import math
from src.risk.fixed_risk_manager import FixedRiskManager, RiskConfig, PositionSize, calculate_atr
from src.scoring.candle_closure_ratings import CandleData, ClosureRating, CandleRating


@pytest.fixture
def config():
    return RiskConfig(
        risk_per_trade=0.02,
        max_daily_loss=0.06,
        max_positions=1,
        atr_multiplier_sl=2.0,
        atr_multiplier_tp=4.0,
        min_reward_risk_ratio=2.0,
    )


@pytest.fixture
def manager(config):
    return FixedRiskManager(config, balance=10000.0)


class TestCalculateATR:
    def test_atr_insufficient_data(self):
        candles = [
            CandleData(timestamp=None, open=100.0, high=102.0, low=99.0, close=101.0, volume=100),
        ]
        assert calculate_atr(candles, period=14) == 0.0

    def test_atr_calculation(self):
        candles = []
        for i in range(20):
            candles.append(CandleData(
                timestamp=None, open=100.0 + i, high=101.0 + i,
                low=99.0 + i, close=100.5 + i, volume=100,
            ))
        atr_val = calculate_atr(candles, period=14)
        assert atr_val > 0
        assert isinstance(atr_val, float)

    def test_atr_constant_range(self):
        candles = []
        for i in range(20):
            candles.append(CandleData(
                timestamp=None, open=100.0, high=102.0, low=98.0,
                close=100.0, volume=100,
            ))
        atr_val = calculate_atr(candles, period=5)
        assert atr_val == 4.0


class TestPositionSizing:
    def test_position_size_basic(self, manager):
        result = manager.calculate_position_size(
            symbol="XAUUSDm", entry_price=2000.0,
            stop_loss=1995.0, take_profit=2010.0,
        )
        assert isinstance(result, PositionSize)
        assert result.volume > 0
        assert result.sl_pips > 0
        assert result.tp_pips > 0
        assert result.reward_risk_ratio >= 2.0

    def test_risk_amount_proportional(self):
        m1 = FixedRiskManager(RiskConfig(risk_per_trade=0.01), balance=10000.0)
        m2 = FixedRiskManager(RiskConfig(risk_per_trade=0.02), balance=10000.0)
        r1 = m1.calculate_position_size("XAUUSDm", 2000.0, 1995.0, 2020.0)
        r2 = m2.calculate_position_size("XAUUSDm", 2000.0, 1995.0, 2020.0)
        assert abs(r2.volume - r1.volume * 2) < 0.001

    def test_custom_risk_per_trade_pct(self, manager):
        default = manager.calculate_position_size("XAUUSDm", 2000.0, 1990.0, 2020.0)
        half = manager.calculate_position_size(
            "XAUUSDm", 2000.0, 1990.0, 2020.0,
            risk_per_trade_pct=0.005,
        )
        assert half.volume < default.volume
        assert abs(half.volume - default.volume * 0.25) < 0.001

    def test_minimum_volume_floor(self, manager):
        result = manager.calculate_position_size(
            "XAUUSDm", 2000.0, 1999.0, 2002.0,
        )
        assert result.volume >= 0.01

    def test_volume_capped_by_balance(self):
        small = FixedRiskManager(RiskConfig(risk_per_trade=0.5), balance=100.0)
        result = small.calculate_position_size("XAUUSDm", 2000.0, 1990.0, 2020.0)
        max_allowed = (100.0 / 1000.0) * 0.5
        assert result.volume <= max_allowed

    def test_pip_value_by_symbol(self, manager):
        xau = manager.calculate_position_size("XAUUSDm", 2000.0, 1995.0, 2010.0)
        xag = manager.calculate_position_size("XAGUSDm", 30.0, 29.5, 31.0)
        nas = manager.calculate_position_size("NAS100", 15000.0, 14990.0, 15030.0)
        assert xau.volume != xag.volume or abs(xau.sl_pips - xag.sl_pips) > 0.01
        assert nas.sl_pips > 0

    def test_position_size_properties(self, manager):
        result = manager.calculate_position_size("XAUUSDm", 2000.0, 1990.0, 2020.0)
        assert result.sl_pips > 0
        assert result.tp_pips > 0
        assert result.risk_amount > 0
        assert result.potential_reward > 0
        assert result.reward_risk_ratio >= 0


def _make_rating(rating, candle=None):
    if candle is None:
        candle = CandleData(timestamp=None, open=100.0, high=102.0, low=98.0, close=101.0, volume=100)
    return CandleRating(candle=candle, rating=rating, closure_pct=0.75, body_pct=0.5, is_bullish=True)


class TestSLTPCalculation:
    def test_buy_direction(self, manager):
        sl, tp = manager.calculate_sl_tp(
            entry_price=2000.0, direction="buy",
            atr=10.0, rating=_make_rating(ClosureRating.A_PLUS),
        )
        assert sl < 2000.0
        assert tp > 2000.0
        assert 2000.0 - sl < tp - 2000.0

    def test_sell_direction(self, manager):
        sl, tp = manager.calculate_sl_tp(
            entry_price=2000.0, direction="sell",
            atr=10.0, rating=_make_rating(ClosureRating.A_PLUS),
        )
        assert sl > 2000.0
        assert tp < 2000.0

    def test_rating_affects_distance(self, manager):
        _, tp_a = manager.calculate_sl_tp(
            2000.0, "buy", 10.0, _make_rating(ClosureRating.A_PLUS),
        )
        _, tp_c = manager.calculate_sl_tp(
            2000.0, "buy", 10.0, _make_rating(ClosureRating.C),
        )
        assert (tp_c - 2000.0) > (tp_a - 2000.0)


class TestRiskLimits:
    def test_can_trade_initially(self, manager):
        ok, reason = manager.can_trade()
        assert ok
        assert reason == "OK"

    def test_daily_loss_limit(self, manager):
        manager.daily_loss = 700.0
        ok, reason = manager.can_trade()
        assert not ok
        assert "loss limit" in reason.lower()

    def test_trade_count_limit(self, manager):
        manager.trades_today = 10
        ok, reason = manager.can_trade()
        assert not ok
        assert "trades" in reason.lower()

    def test_record_trade_loss(self, manager):
        manager.record_trade(-100.0)
        assert manager.daily_loss == 100.0  # positive accumulation
        assert manager.trades_today == 1

    def test_record_trade_profit_ignored(self, manager):
        manager.record_trade(100.0)
        assert manager.daily_loss == 0.0

    def test_reset_daily(self, manager):
        manager.daily_loss = 500.0
        manager.trades_today = 5
        manager.last_reset = __import__("pandas").Timestamp.now().date() - __import__("pandas").Timedelta(days=1)
        ok, _ = manager.can_trade()
        assert ok

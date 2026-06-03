import pytest
import pandas as pd

from src.core.candle_confirmer import CandleConfirmer, ConfirmerStatus
from src.core.liquidity_mapper import MarketMap


class TestCandleConfirmer:
    @pytest.fixture
    def confirmer(self):
        return CandleConfirmer(min_close_ratio=0.6, lookback_candles=3)

    @pytest.fixture
    def market_map(self):
        return MarketMap(
            symbol="XAUUSD",
            current_price=100.0,
            primary_trend="BULLISH",
        )

    def test_insufficient_data(self, confirmer, market_map):
        small = pd.DataFrame({"open": [100.0], "close": [100.0],
                               "high": [100.3], "low": [99.7]})
        result = confirmer._check_single(small, "BUY", "1h", market_map)
        assert result.status == ConfirmerStatus.NOT_READY

    def test_confirm_buy(self, confirmer, df_candle_confirm_buy, market_map):
        result = confirmer._check_single(df_candle_confirm_buy, "BUY", "1h", market_map)
        assert result.status == ConfirmerStatus.CONFIRMED
        assert result.validation_score >= 0.6

    def test_confirm_sell(self, confirmer, df_candle_confirm_sell, market_map):
        result = confirmer._check_single(df_candle_confirm_sell, "SELL", "1h", market_map)
        assert result.status == ConfirmerStatus.CONFIRMED
        assert result.validation_score >= 0.6

    def test_reject_opposite_direction(self, confirmer, df_candle_confirm_buy, market_map):
        """BUY candle but SELL tesis → REJECTED."""
        result = confirmer._check_single(df_candle_confirm_buy, "SELL", "1h", market_map)
        assert result.status == ConfirmerStatus.REJECTED

    def test_pending_small_body(self, confirmer, market_map):
        df = pd.DataFrame({
            "open": [100.0] * 9 + [100.0],
            "close": [100.0] * 9 + [100.3],
            "high": [100.3] * 9 + [100.5],
            "low": [99.7] * 9 + [99.7],
        })
        result = confirmer._check_single(df, "BUY", "1h", market_map)
        # ratio = 0.3/0.8 = 0.375 < 0.6 → PENDING
        assert result.status == ConfirmerStatus.PENDING

    def test_candle_body_ratio(self, confirmer):
        candle = pd.Series({"open": 100.0, "close": 102.0, "high": 102.5, "low": 99.5})
        ratio = confirmer._candle_body_ratio(candle)
        assert ratio == 2.0 / 3.0

    def test_check_includes_both_timeframes(self, confirmer, df_candle_confirm_buy, market_map):
        result = confirmer.check(df_candle_confirm_buy, df_candle_confirm_buy, "BUY", market_map)
        assert "1h" in result
        assert "4h" in result

    def test_consecutive_bullish_bonus(self, confirmer, market_map):
        """Two consecutive bullish candles add +0.15 bonus."""
        df = pd.DataFrame({
            "open": [100.0] * 8 + [100.0, 100.0],
            "close": [100.0] * 8 + [101.0, 102.0],
            "high": [100.3] * 8 + [101.3, 102.3],
            "low": [99.7] * 8 + [99.7, 99.7],
        })
        result = confirmer._check_single(df, "BUY", "1h", market_map)
        assert result.status == ConfirmerStatus.CONFIRMED
        assert result.validation_score >= 0.75

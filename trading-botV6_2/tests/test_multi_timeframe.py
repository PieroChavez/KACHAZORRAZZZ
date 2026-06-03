from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

from src.core.multi_timeframe import (
    TIMEFRAME_CODES,
    TIMEFRAME_ORDER,
    TIMEFRAME_GROUPS,
    TimeframeData,
    MultiTimeframeFetcher,
)
from src.core.candle_closure_ratings import CandleData, CandleRating


def make_candle_data(timestamp=pd.Timestamp("2024-01-01"), open_=100.0,
                     high=101.0, low=99.0, close=100.5, volume=1000):
    return CandleData(
        timestamp=timestamp,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def make_candle_rating(candle=None, rating_value="B", closure_pct=0.6,
                       body_pct=0.5, is_bullish=True, is_reversal=False):
    if candle is None:
        candle = make_candle_data()
    return CandleRating(
        candle=candle,
        rating=rating_value,
        closure_pct=closure_pct,
        body_pct=body_pct,
        is_bullish=is_bullish,
        is_reversal=is_reversal,
    )


# ─────────────────────────────────────────────
# TIMEFRAME CODES
# ─────────────────────────────────────────────

class TestTimeframeCodes:
    def test_has_expected_keys(self):
        expected = {"1min", "3min", "5min", "10min", "15min", "30min", "1h", "2h", "3h", "4h"}
        for key in expected:
            assert key.lower() in {k.lower() for k in TIMEFRAME_CODES}, f"Missing key: {key}"

    def test_mt5_constants(self):
        assert TIMEFRAME_CODES["1min"] == 1
        assert TIMEFRAME_CODES["5min"] == 5
        assert TIMEFRAME_CODES["15min"] == 15
        assert TIMEFRAME_CODES["1H"] == 16385
        assert TIMEFRAME_CODES["4H"] == 16388

    def test_all_keys_in_order(self):
        for tf in TIMEFRAME_ORDER:
            assert tf in TIMEFRAME_CODES, f"{tf} missing from TIMEFRAME_CODES"

    def test_group_membership(self):
        all_grouped = set()
        for group in TIMEFRAME_GROUPS.values():
            all_grouped.update(group)
        assert all_grouped == set(TIMEFRAME_ORDER)


# ─────────────────────────────────────────────
# TIMEFRAME DATA
# ─────────────────────────────────────────────

class TestTimeframeData:
    def test_defaults(self):
        tfd = TimeframeData(timeframe="1H")
        assert tfd.timeframe == "1H"
        assert tfd.candles == []
        assert tfd.ratings == []
        assert tfd.last_update is None


# ─────────────────────────────────────────────
# MULTI TIMEFRAME FETCHER
# ─────────────────────────────────────────────

class TestMultiTimeframeFetcher:
    @pytest.fixture
    def fetcher(self):
        client = MagicMock()
        return MultiTimeframeFetcher(client)

    def test_init_stores_client_and_initializes_data(self, fetcher):
        assert fetcher.mt5 is not None
        for tf in TIMEFRAME_ORDER:
            assert tf in fetcher._data
            assert isinstance(fetcher._data[tf], TimeframeData)
            assert fetcher._data[tf].timeframe == tf

    @patch("src.core.multi_timeframe.rate_candle")
    def test_fetch_all_returns_dict_with_all_timeframes(self, mock_rate, fetcher):
        candle = make_candle_data()
        fetcher.mt5.get_candles.return_value = [candle]
        mock_rate.return_value = make_candle_rating(candle)

        result = fetcher.fetch_all("EURUSD", count=100)

        assert len(result) == len(TIMEFRAME_ORDER)
        for tf in TIMEFRAME_ORDER:
            assert tf in result
            assert len(result[tf].candles) == 1
            assert len(result[tf].ratings) == 1
            assert result[tf].last_update is not None

    @patch("src.core.multi_timeframe.rate_candle")
    def test_get_dataframes_returns_dict_of_dataframes(self, mock_rate, fetcher):
        candle = make_candle_data()
        fetcher.mt5.get_candles.return_value = [candle]
        mock_rate.return_value = make_candle_rating(candle)

        dfs = fetcher.get_dataframes("EURUSD", count=100)

        for tf in TIMEFRAME_ORDER:
            assert tf in dfs
            assert isinstance(dfs[tf], pd.DataFrame)
            assert list(dfs[tf].columns) == ["time", "open", "high", "low", "close", "volume"]
            assert dfs[tf].iloc[0]["open"] == 100.0
            assert dfs[tf].iloc[0]["close"] == 100.5

    @patch("src.core.multi_timeframe.rate_candle")
    def test_get_dataframes_skips_empty_timeframes(self, mock_rate, fetcher):
        def side_effect(symbol, timeframe, count):
            if timeframe == 16388:
                return [make_candle_data()]
            return []
        fetcher.mt5.get_candles.side_effect = side_effect
        mock_rate.return_value = make_candle_rating()

        dfs = fetcher.get_dataframes("EURUSD", count=100)

        assert "4H" in dfs
        assert len(dfs) == 1

    @patch("src.core.multi_timeframe.rate_candle")
    def test_get_dataframes_empty_when_no_data(self, mock_rate, fetcher):
        fetcher.mt5.get_candles.return_value = []
        mock_rate.return_value = make_candle_rating()

        dfs = fetcher.get_dataframes("EURUSD", count=100)
        assert dfs == {}

    @patch("src.core.multi_timeframe.rate_candle")
    def test_fetch_all_handles_exception_gracefully(self, mock_rate, fetcher):
        fetcher.mt5.get_candles.side_effect = Exception("Connection error")
        mock_rate.return_value = make_candle_rating()

        result = fetcher.fetch_all("EURUSD", count=100)

        for tf in TIMEFRAME_ORDER:
            assert tf in result
            assert result[tf].candles == []

    @patch("src.core.multi_timeframe.rate_candle")
    def test_get_latest_rating_returns_none_when_empty(self, mock_rate, fetcher):
        assert fetcher.get_latest_rating("1H") is None

    @patch("src.core.multi_timeframe.rate_candle")
    def test_get_latest_rating_after_fetch(self, mock_rate, fetcher):
        candle = make_candle_data()
        fetcher.mt5.get_candles.return_value = [candle]
        mock_rate.return_value = make_candle_rating(candle, is_bullish=True)

        fetcher.fetch_all("EURUSD", count=100)
        rating = fetcher.get_latest_rating("1H")

        assert rating is not None
        assert rating.is_bullish is True

    @patch("src.core.multi_timeframe.rate_candle")
    def test_get_latest_candle_returns_none_when_empty(self, mock_rate, fetcher):
        assert fetcher.get_latest_candle("1H") is None

    @patch("src.core.multi_timeframe.rate_candle")
    def test_get_latest_candle_after_fetch(self, mock_rate, fetcher):
        candle = make_candle_data(timestamp=pd.Timestamp("2024-06-01"))
        fetcher.mt5.get_candles.return_value = [candle]
        mock_rate.return_value = make_candle_rating(candle)

        fetcher.fetch_all("EURUSD", count=100)
        latest = fetcher.get_latest_candle("1H")

        assert latest is not None
        assert latest.timestamp == pd.Timestamp("2024-06-01")

    @patch("src.core.multi_timeframe.rate_candle")
    def test_invalid_timeframe_returns_none(self, mock_rate, fetcher):
        assert fetcher.get_latest_rating("INVALID") is None
        assert fetcher.get_latest_candle("INVALID") is None

    def test_fetch_all_skips_missing_tf_code(self, fetcher):
        with patch.dict(TIMEFRAME_CODES, {}, clear=True):
            result = fetcher.fetch_all("EURUSD", count=100)
            for tf in TIMEFRAME_ORDER:
                assert result[tf].candles == []

    @patch("src.core.multi_timeframe.rate_candle")
    def test_dataframe_time_column_is_datetime(self, mock_rate, fetcher):
        candle = make_candle_data(timestamp=pd.Timestamp("2024-06-15 12:00"))
        fetcher.mt5.get_candles.return_value = [candle]
        mock_rate.return_value = make_candle_rating(candle)

        dfs = fetcher.get_dataframes("EURUSD", count=100)
        for tf in TIMEFRAME_ORDER:
            assert pd.api.types.is_datetime64_any_dtype(dfs[tf]["time"])

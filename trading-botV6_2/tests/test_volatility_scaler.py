import pytest
import pandas as pd
import numpy as np
from src.core.volatility_scaler import VolatilityScaler, VolatilityBacktester


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 4320
    ts = pd.date_range("2026-05-26", periods=n, freq="1min")
    close_arr = 3000 + np.cumsum(np.random.randn(n) * 0.5)
    close_s = pd.Series(close_arr)
    high = close_arr + abs(np.random.randn(n)) * 2
    low = close_arr - abs(np.random.randn(n)) * 2
    open_ = close_s.shift(1).fillna(close_s.iloc[0])
    vol = np.random.randint(100, 1000, n)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close_arr, "volume": vol,
    }, index=ts)


class TestVolatilityScaler:
    def test_compute_base(self):
        s = VolatilityScaler(expert_mode=False)
        d = s.compute("TEST", None, 1.5, 2.0)
        assert d["sl_mult"] == 1.5
        assert d["tp_mult"] == 2.0
        assert d["reason"] == "insufficient_data"

    def test_compute_expert(self, sample_df):
        s = VolatilityScaler(expert_mode=True)
        d = s.compute("XAUUSD", sample_df, 1.5, 2.0)
        assert 0.5 <= d["sl_mult"] <= 3.5
        assert 0.5 <= d["tp_mult"] <= 5.0
        assert "session" in d["reason"] or "Sesión" in d["reason"] or "intrabar" in d["reason"]

    def test_compute_expert_missing_index(self):
        df = pd.DataFrame({
            "open": [1, 2, 3], "high": [2, 3, 4], "low": [0, 1, 2],
            "close": [1, 2, 3], "volume": [100, 200, 300],
        })
        s = VolatilityScaler(expert_mode=True)
        d = s.compute("TEST", df, 1.5, 2.0)
        assert d["reason"] == "insufficient_data"

    def test_compute_expert_small_df(self):
        np.random.seed(42)
        ts = pd.date_range("2026-05-26", periods=30, freq="1min")
        df = pd.DataFrame({
            "open": np.random.randn(30) + 3000,
            "high": np.random.randn(30) + 3002,
            "low": np.random.randn(30) + 2998,
            "close": np.random.randn(30) + 3000,
            "volume": np.random.randint(100, 1000, 30),
        }, index=ts)
        s = VolatilityScaler(expert_mode=True)
        d = s.compute("TEST", df, 1.5, 2.0)
        assert "insufficient_data" in d["reason"] or d["sl_mult"] >= 0.5

    def test_bollinger_metrics(self, sample_df):
        s = VolatilityScaler(expert_mode=True)
        m = s._bollinger_metrics(sample_df)
        assert 0 < m["bb_width_ratio"] < 5
        assert 0 <= m["bb_position"] <= 1

    def test_keltner_metrics(self, sample_df):
        s = VolatilityScaler(expert_mode=True)
        m = s._keltner_metrics(sample_df)
        assert 0 < m["kc_width_ratio"] < 5

    def test_session_multiplier(self):
        s = VolatilityScaler(expert_mode=True)
        m = s._session_multiplier()
        assert 0.5 <= m["sl"] <= 1.5
        assert 0.5 <= m["tp"] <= 1.5
        assert "note" in m

    def test_adaptive_factor(self):
        s = VolatilityScaler(expert_mode=True)
        metrics = {"atr_ratio": 1.5}
        f1 = s._adaptive_factor("SYM", metrics)
        assert f1["sl"] >= 1.0
        metrics2 = {"atr_ratio": 0.5}
        f2 = s._adaptive_factor("SYM", metrics2)
        assert f2["sl"] <= 1.0

    def test_volatility_prediction(self, sample_df):
        s = VolatilityScaler(expert_mode=True)
        for i in range(10):
            s._all_metrics("XAUUSD", sample_df.iloc[:50 + i * 10])
        r = s._volatility_prediction("XAUUSD", {"current_atr": 5.0})
        assert 0.7 <= r["sl"] <= 1.4

    def test_multi_tf_factor(self, sample_df):
        s = VolatilityScaler(expert_mode=True)
        r = s._multi_tf_factor(sample_df)
        assert 0.7 <= r["sl"] <= 1.5
        assert "note" in r

    def test_adjust_sl_tp_buy(self, sample_df):
        s = VolatilityScaler(expert_mode=True)
        new_sl, new_tp, info = s.adjust_sl_tp(
            "XAUUSD", sample_df, 3000.0, "BUY",
            2990.0, 3020.0, digits=5, pip=0.0001,
        )
        assert new_sl < 3000.0
        assert new_tp > 3000.0
        assert "vol_adj" in info["note"]

    def test_adjust_sl_tp_sell(self, sample_df):
        s = VolatilityScaler(expert_mode=True)
        new_sl, new_tp, info = s.adjust_sl_tp(
            "XAUUSD", sample_df, 3000.0, "SELL",
            3010.0, 2980.0, digits=5, pip=0.0001,
        )
        assert new_sl > 3000.0
        assert new_tp < 3000.0

    def test_adjust_sl_tp_unusual_range(self, sample_df):
        s = VolatilityScaler(expert_mode=True)
        new_sl, new_tp, info = s.adjust_sl_tp(
            "XAUUSD", sample_df, 3000.0, "BUY",
            2999.0, 3001.0, digits=5, pip=0.0001,
            sl_min_pips=1.0, sl_max_pips=3.0,
        )
        assert info["note"] != "invalid_direction"

    def test_get_baseline(self, sample_df):
        s = VolatilityScaler(expert_mode=True)
        s.compute("SYM_BL", sample_df, 1.5, 2.0)
        bl = s.get_baseline("SYM_BL")
        assert bl is not None
        assert "atr_mean" in bl

    def test_reset_baseline(self, sample_df):
        s = VolatilityScaler(expert_mode=True)
        s.compute("SYM_R", sample_df, 1.5, 2.0)
        s.reset_baseline("SYM_R")
        assert s.get_baseline("SYM_R") is None


class TestVolatilityBacktester:
    def test_run(self, sample_df):
        s = VolatilityScaler(expert_mode=True)
        bt = VolatilityBacktester(s)
        stats = bt.run(sample_df, "XAUUSD", 1.5, 2.0)
        assert "samples" in stats
        assert stats["samples"] > 0
        assert 0 <= stats["sl_mult_mean"] <= 5

    def test_summary(self, sample_df):
        s = VolatilityScaler(expert_mode=True)
        bt = VolatilityBacktester(s)
        bt.run(sample_df, "XAUUSD", 1.5, 2.0)
        summary = bt.summary()
        assert "Backtest" in summary

    def test_run_insufficient_data(self):
        s = VolatilityScaler(expert_mode=True)
        bt = VolatilityBacktester(s)
        df = pd.DataFrame({"open": [1, 2], "high": [2, 3], "low": [0, 1], "close": [1, 2], "volume": [100, 200]})
        stats = bt.run(df, "TEST")
        assert "error" in stats

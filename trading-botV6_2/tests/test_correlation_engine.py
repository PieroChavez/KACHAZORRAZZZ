import pytest
import numpy as np
from datetime import datetime, timezone
from src.core.correlation_engine import CorrelationEngine, CorrelationRegime, LOOKBACKS


def make_prices(base=3000, n=500, seed=42):
    np.random.seed(seed)
    return base + np.cumsum(np.random.randn(n) * 0.5)


def ts_seq(start, n, minutes=1):
    from datetime import timedelta
    return [start - timedelta(minutes=n - i) for i in range(n)]


class TestCorrelationEngine:
    def test_update_and_self_correlation(self):
        e = CorrelationEngine()
        times = ts_seq(datetime.now(timezone.utc), 100)
        for i in range(100):
            e.update("XAUUSD", times[i], 3000 + i * 0.1)
        c = e.correlation("XAUUSD", "XAUUSD")
        assert c["correlation"] == 1.0
        assert c["regime"] == CorrelationRegime.HIGHLY_CORRELATED

    def test_highly_correlated_symbols(self):
        e = CorrelationEngine()
        np.random.seed(0)
        times = ts_seq(datetime.now(timezone.utc), 500)
        base = np.cumsum(np.random.randn(500) * 0.5)
        noise = np.random.randn(500) * 0.05
        for i in range(500):
            e.update("A", times[i], 3000 + base[i])
            e.update("B", times[i], 3000 + base[i] * 0.95 + noise[i])
        c = e.correlation("A", "B", "slow")
        assert c["correlation"] > 0.8
        assert c["n"] > 20

    def test_uncorrelated_symbols(self):
        e = CorrelationEngine()
        np.random.seed(0)
        times = ts_seq(datetime.now(timezone.utc), 500)
        for i in range(500):
            e.update("A", times[i], 3000 + np.random.randn() * 10)
            e.update("B", times[i], 3000 + np.random.randn() * 10)
        c = e.correlation("A", "B", "slow")
        assert abs(c["correlation"]) < 0.5

    def test_diverging_detection(self):
        e = CorrelationEngine()
        np.random.seed(0)
        times = ts_seq(datetime.now(timezone.utc), 600)
        base = np.cumsum(np.random.randn(600) * 0.5)
        for i in range(200):
            e.update("A", times[i], 3000 + base[i])
            e.update("B", times[i], 3000 + base[i] * 0.95)
        for i in range(200, 600):
            e.update("A", times[i], 3000 + base[i])
            e.update("B", times[i], 3000 - base[i] * 0.8)
        diverging, diff = e.are_diverging("A", "B")
        assert diverging

    def test_correlation_matrix(self):
        e = CorrelationEngine()
        np.random.seed(0)
        times = ts_seq(datetime.now(timezone.utc), 500)
        base = np.cumsum(np.random.randn(500) * 0.5)
        for i in range(500):
            e.update("XAUUSD", times[i], 3000 + base[i])
            e.update("XAGUSD", times[i], 3000 + base[i] * 0.9)
            e.update("BTCUSD", times[i], 50000 + np.random.randn() * 500)
        matrix = e.correlation_matrix(["XAUUSD", "XAGUSD", "BTCUSD"])
        assert abs(matrix["XAUUSD"]["XAGUSD"]) > 0.5
        assert abs(matrix["XAUUSD"]["BTCUSD"]) < abs(matrix["XAUUSD"]["XAGUSD"])

    def test_confirm_signal_blocks_opposite(self):
        e = CorrelationEngine()
        np.random.seed(0)
        times = ts_seq(datetime.now(timezone.utc), 500)
        base = np.cumsum(np.random.randn(500) * 0.5)
        for i in range(500):
            e.update("XAUUSD", times[i], 3000 + base[i])
            e.update("XAGUSD", times[i], 3000 + base[i] * 0.95)
        ok, reason = e.confirm_signal(
            "XAUUSD", "SELL",
            ["XAUUSD", "XAGUSD"],
            {"XAGUSD": "BUY"},
        )
        assert not ok
        assert "Correlación" in reason

    def test_confirm_signal_allows_aligned(self):
        e = CorrelationEngine()
        np.random.seed(0)
        times = ts_seq(datetime.now(timezone.utc), 500)
        base = np.cumsum(np.random.randn(500) * 0.5)
        for i in range(500):
            e.update("XAUUSD", times[i], 3000 + base[i])
            e.update("XAGUSD", times[i], 3000 + base[i] * 0.95)
        ok, reason = e.confirm_signal(
            "XAUUSD", "BUY",
            ["XAUUSD", "XAGUSD"],
            {"XAGUSD": "BUY"},
        )
        assert ok

    def test_confirm_signal_with_divergence_override(self):
        e = CorrelationEngine()
        times = ts_seq(datetime.now(timezone.utc), 600)
        base = np.cumsum(np.random.randn(600) * 0.5)
        for i in range(200):
            e.update("A", times[i], 3000 + base[i])
            e.update("B", times[i], 3000 + base[i] * 0.98)
        for i in range(200, 600):
            e.update("A", times[i], 3000 + base[i])
            e.update("B", times[i], 3000 - base[i] * 0.8)
        ok, reason = e.confirm_signal(
            "A", "BUY", ["A", "B"], {"B": "SELL"},
        )
        assert ok

    def test_volume_adjustment_high_corr(self):
        e = CorrelationEngine()
        np.random.seed(0)
        times = ts_seq(datetime.now(timezone.utc), 500)
        base = np.cumsum(np.random.randn(500) * 0.5)
        for i in range(500):
            e.update("XAUUSD", times[i], 3000 + base[i])
            e.update("XAGUSD", times[i], 3000 + base[i] * 0.98)
        adj, reason = e.volume_adjustment("XAUUSD", {"XAGUSD": 0.1})
        assert adj < 0.9
        assert "r=" in reason

    def test_volume_adjustment_no_corr(self):
        e = CorrelationEngine()
        adj, reason = e.volume_adjustment("XAUUSD", {})
        assert adj == 1.0
        assert reason == ""

    def test_get_regime_highly_correlated(self):
        e = CorrelationEngine()
        ts = datetime.now(timezone.utc)
        base = np.cumsum(np.random.randn(500) * 0.5)
        for i in range(500):
            e.update("A", ts, 3000 + base[i])
            e.update("B", ts, 3000 + base[i] * 0.98)
        regime = e.get_regime("A", "B")
        assert regime in (CorrelationRegime.HIGHLY_CORRELATED, CorrelationRegime.DIVERGING)

    def test_get_regime_uncorrelated(self):
        e = CorrelationEngine()
        ts = datetime.now(timezone.utc)
        for i in range(500):
            e.update("A", ts, 3000 + np.random.randn() * 10)
            e.update("B", ts, 3000 + np.random.randn() * 10)
        regime = e.get_regime("A", "B")
        assert regime == CorrelationRegime.UNCORRELATED

    def test_divergence_alerts(self):
        e = CorrelationEngine()
        times = ts_seq(datetime.now(timezone.utc), 600)
        base = np.cumsum(np.random.randn(600) * 0.5)
        for i in range(200):
            e.update("A", times[i], 3000 + base[i])
            e.update("B", times[i], 3000 + base[i] * 0.95)
        for i in range(200, 600):
            e.update("A", times[i], 3000 + base[i])
            e.update("B", times[i], 3000 - base[i] * 0.8)
        alerts = e.get_divergence_alerts()
        assert len(alerts) >= 1
        assert "divergiendo" in alerts[0]

    def test_reset_single_symbol(self):
        e = CorrelationEngine()
        times = ts_seq(datetime.now(timezone.utc), 100)
        for i in range(100):
            e.update("XAUUSD", times[i], 3000 + i)
        e.reset("XAUUSD")
        assert "XAUUSD" not in e._price_history

    def test_reset_all(self):
        e = CorrelationEngine()
        times = ts_seq(datetime.now(timezone.utc), 100)
        for i in range(100):
            e.update("XAUUSD", times[i], 3000 + i)
            e.update("XAGUSD", times[i], 3000 + i)
        e.reset()
        assert len(e._price_history) == 0

    def test_cache_expiry(self):
        e = CorrelationEngine()
        times = ts_seq(datetime.now(timezone.utc), 100)
        for i in range(100):
            e.update("A", times[i], 3000 + i * 0.1)
            e.update("B", times[i], 3000 + i * 0.11)
        c1 = e.correlation("A", "B", "fast")
        c2 = e.correlation("A", "B", "fast")
        assert c1["correlation"] == c2["correlation"]

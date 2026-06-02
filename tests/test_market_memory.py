import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from src.learning.market_memory import MarketMemory, LevelMemory


@pytest.fixture
def db_path(tmpdir):
    return Path(tmpdir) / "test_market_memory.db"


@pytest.fixture
def memory(db_path):
    return MarketMemory(db_path)


@pytest.fixture
def ltf_df():
    n = 30
    np.random.seed(42)
    opens = 100.0 + np.random.randn(n) * 0.5
    highs = opens + abs(np.random.randn(n)) * 0.5
    lows = opens - abs(np.random.randn(n)) * 0.5
    closes = opens + np.random.randn(n) * 0.2
    return pd.DataFrame({
        "time": pd.date_range("2026-05-01", periods=n, freq="5min"),
        "open": opens, "high": highs, "low": lows, "close": closes,
    })


class TestLevelMemory:
    def test_reliability_zero_with_no_touches(self):
        lm = LevelMemory(price=100.0, symbol="XAUUSDm")
        assert lm.reliability == 0.0

    def test_reliability_increases_with_bounces(self):
        lm = LevelMemory(price=100.0, symbol="XAUUSDm", total_touches=10, bounce_count=8, break_count=2)
        assert lm.reliability > 0.6
        assert lm.is_reliable_support
        assert lm.is_reliable_resistance

    def test_reliability_low_with_many_breaks(self):
        lm = LevelMemory(price=100.0, symbol="XAUUSDm", total_touches=10, bounce_count=2, break_count=8)
        assert lm.reliability < 0.5
        assert not lm.is_reliable_support

    def test_is_reliable_support(self):
        lm = LevelMemory(price=100.0, symbol="XAUUSDm", total_touches=10, bounce_count=8, break_count=2)
        assert lm.is_reliable_support

    def test_is_reliable_resistance(self):
        lm = LevelMemory(price=100.0, symbol="XAUUSDm", total_touches=10, bounce_count=8, break_count=2)
        assert lm.is_reliable_resistance


class TestMarketMemory:
    def test_init_creates_db(self, db_path):
        memory = MarketMemory(db_path)
        assert db_path.exists()
        conn = __import__("sqlite3").connect(str(db_path))
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = [t[0] for t in tables]
        conn.close()
        assert "level_memory" in table_names
        assert "level_interactions" in table_names

    def test_record_and_retrieve_level(self, memory):
        memory.record_interaction("XAUUSDm", 100.0, "bounce", pattern_type="FVG")
        lm = memory.get_level_reliability("XAUUSDm", 100.0)
        assert lm.price == 100.0
        assert lm.visit_count == 1
        assert lm.bounce_count == 1
        assert lm.break_count == 0
        assert "FVG" in lm.pattern_types

    def test_record_break_outcome(self, memory):
        memory.record_interaction("XAGUSDm", 30.0, "break")
        lm = memory.get_level_reliability("XAGUSDm", 30.0)
        assert lm.break_count == 1
        assert lm.bounce_count == 0

    def test_record_multiple_interactions(self, memory):
        for i in range(3):
            memory.record_interaction("XAUUSDm", 100.0, "bounce" if i < 2 else "break")
        lm = memory.get_level_reliability("XAUUSDm", 100.0)
        assert lm.visit_count == 3
        assert lm.bounce_count == 2
        assert lm.break_count == 1
        assert lm.total_touches == 3

    def test_get_level_reliability_unknown(self, memory):
        lm = memory.get_level_reliability("XAUUSDm", 999.0)
        assert lm.price == 999.0
        assert lm.visit_count == 0
        assert lm.reliability == 0.0

    def test_get_nearby_levels(self, memory):
        for price in [100.0, 101.0, 102.0, 110.0]:
            memory.record_interaction("XAUUSDm", price, "bounce")
        levels = memory.get_nearby_levels("XAUUSDm", 101.0, atr_value=1.0, max_levels=5)
        assert len(levels) >= 2
        assert all(l.price >= 100.0 - 2.0 for l in levels)
        assert all(l.price <= 100.0 + 2.0 for l in levels)

    def test_get_nearby_levels_caches(self, memory):
        memory.record_interaction("XAUUSDm", 100.0, "bounce")
        levels1 = memory.get_nearby_levels("XAUUSDm", 100.0, atr_value=1.0)
        levels2 = memory.get_nearby_levels("XAUUSDm", 100.0, atr_value=1.0)
        assert levels1 is levels2

    def test_get_consolidated_levels_with_sufficient_touches(self, memory):
        for i in range(5):
            memory.record_interaction("XAUUSDm", 100.0, "bounce")
        for i in range(3):
            memory.record_interaction("XAUUSDm", 101.0, "break")
        levels = memory.get_consolidated_levels("XAUUSDm", atr_value=1.0)
        assert len(levels) > 0
        for l in levels:
            assert l.reliability >= 0.5

    def test_get_consolidated_levels_empty_with_no_data(self, memory):
        levels = memory.get_consolidated_levels("XAUUSDm", atr_value=1.0)
        assert len(levels) == 0

    def test_get_regime_level_stats(self, memory):
        memory.record_interaction("XAUUSDm", 100.0, "bounce", regime="RANGING")
        memory.record_interaction("XAUUSDm", 101.0, "break", regime="RANGING")
        stats = memory.get_regime_level_stats("XAUUSDm", "RANGING")
        assert stats["bounce"] == 1
        assert stats["break"] == 1
        assert stats["total"] == 2
        assert stats["bounce_rate"] == 0.5

    def test_get_regime_level_stats_empty(self, memory):
        stats = memory.get_regime_level_stats("XAUUSDm", "NONEXISTENT")
        assert stats["bounce"] == 0
        assert stats["total"] == 0

    def test_get_level_bias_bullish(self, memory):
        memory.record_interaction("XAUUSDm", 99.0, "bounce")
        memory.record_interaction("XAUUSDm", 98.5, "bounce")
        bias = memory.get_level_bias("XAUUSDm", 100.0, atr_value=1.0)
        assert bias in ("BULLISH_BIAS", "RANGE_BIAS", None)

    def test_get_level_bias_none_with_no_data(self, memory):
        bias = memory.get_level_bias("XAUUSDm", 100.0, atr_value=1.0)
        assert bias is None

    def test_scan_current_levels(self, memory, ltf_df):
        for i in range(5):
            memory.record_interaction("XAUUSDm", 100.0 + i, "bounce")
        levels = memory.scan_current_levels("XAUUSDm", ltf_df)
        assert isinstance(levels, list)

    def test_scan_current_levels_with_insufficient_data(self, memory):
        df = pd.DataFrame({"high": [100.0], "low": [99.0], "close": [99.5],
                          "open": [100.0]}, index=[0])
        levels = memory.scan_current_levels("XAUUSDm", df)
        assert levels == []

    def test_merge_close_levels(self, memory):
        levels = [
            LevelMemory(price=100.0, symbol="XAUUSDm", visit_count=3, bounce_count=2, break_count=1, total_touches=3),
            LevelMemory(price=100.1, symbol="XAUUSDm", visit_count=2, bounce_count=1, break_count=1, total_touches=2),
            LevelMemory(price=101.0, symbol="XAUUSDm", visit_count=5, bounce_count=4, break_count=1, total_touches=5),
        ]
        merged = memory._merge_close_levels(levels, atr_value=1.0)
        assert len(merged) == 2

    def test_cleanup_old_entries(self, memory):
        import sqlite3
        conn = sqlite3.connect(str(memory.db_path))
        conn.execute("""
            INSERT INTO level_interactions (symbol, price, timestamp, outcome)
            VALUES ('XAUUSDm', 100.0, datetime('now', '-31 days'), 'bounce')
        """)
        conn.commit()
        conn.close()
        memory.cleanup_old_entries(days=30)
        conn = sqlite3.connect(str(memory.db_path))
        count = conn.execute("SELECT COUNT(*) FROM level_interactions").fetchone()[0]
        conn.close()
        assert count == 0

    def test_invalidate_cache_on_record(self, memory):
        memory.record_interaction("XAUUSDm", 100.0, "bounce")
        levels = memory.get_nearby_levels("XAUUSDm", 100.0, atr_value=1.0)
        assert len(memory._cache) > 0
        memory.record_interaction("XAUUSDm", 100.0, "break")
        assert len(memory._cache) == 0

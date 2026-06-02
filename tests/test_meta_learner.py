import pytest
from datetime import datetime, timedelta
from pathlib import Path
from src.learning.meta_learner import MetaLearner, TradeRecord, RegimePerformance, PatternPerformance
from src.core.regime_detector import RegimeType, REGIME_PATTERN_MULTIPLIERS
from src.core.strategy_engine import ScoringConfig


@pytest.fixture
def db_path(tmpdir):
    return Path(tmpdir) / "test_meta_learning.db"


@pytest.fixture
def meta(db_path):
    return MetaLearner(db_path)


def make_trade(symbol="XAUUSDm", direction="BUY", profit=100.0, regime="RANGING",
               session="LONDON_OPEN", conviction=0.6, score=75.0,
               primary_pattern="FVG", patterns=None, exit_reason="TP"):
    if patterns is None:
        patterns = ["FVG", "OB"]
    return TradeRecord(
        symbol=symbol, direction=direction,
        entry_price=100.0, exit_price=100.0 + profit / 1000,
        volume=0.1, profit=profit, score=score, conviction=conviction,
        regime=regime, session=session, primary_pattern=primary_pattern,
        patterns_found=patterns, regime_confidence=0.7,
        exit_reason=exit_reason, duration_minutes=30,
        timestamp=datetime.now(),
    )


class TestTradeRecord:
    def test_trade_record_creation(self):
        trade = make_trade()
        assert trade.symbol == "XAUUSDm"
        assert trade.direction == "BUY"
        assert trade.profit == 100.0
        assert trade.regime == "RANGING"
        assert trade.primary_pattern == "FVG"


class TestRegimePerformance:
    def test_regime_performance_creation(self):
        rp = RegimePerformance(
            regime="RANGING", total_trades=10, wins=7, losses=3,
            win_rate=0.7, total_profit=500.0, avg_profit=50.0,
            profit_factor=2.5, patterns_best=[("FVG", 100.0)],
            patterns_worst=[("BREAKER", -50.0)],
        )
        assert rp.regime == "RANGING"
        assert rp.win_rate == 0.7


class TestPatternPerformance:
    def test_pattern_performance_creation(self):
        pp = PatternPerformance(
            pattern_type="FVG", total_occurrences=10, as_primary_count=5,
            win_rate=0.7, avg_profit=50.0, total_profit=500.0,
            best_regime="STRONG_TREND_BULLISH", worst_regime="RANGING",
        )
        assert pp.pattern_type == "FVG"
        assert pp.win_rate == 0.7


class TestMetaLearner:
    def test_init_creates_db(self, db_path):
        meta = MetaLearner(db_path)
        assert db_path.exists()
        conn = __import__("sqlite3").connect(str(db_path))
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = [t[0] for t in tables]
        conn.close()
        assert "trade_records" in table_names
        assert "pattern_performance" in table_names
        assert "weight_adjustments" in table_names
        assert "multiplier_adjustments" in table_names

    def test_record_trade(self, meta):
        trade = make_trade()
        meta.record_trade(trade)
        conn = __import__("sqlite3").connect(str(meta.db_path))
        count = conn.execute("SELECT COUNT(*) FROM trade_records").fetchone()[0]
        conn.close()
        assert count == 1

    def test_record_trade_with_disabled_learning(self, meta):
        meta.disable_learning()
        trade = make_trade()
        meta.record_trade(trade)
        conn = __import__("sqlite3").connect(str(meta.db_path))
        count = conn.execute("SELECT COUNT(*) FROM trade_records").fetchone()[0]
        conn.close()
        assert count == 0

    def test_enable_learning(self, meta):
        meta.disable_learning()
        assert not meta._learning_enabled
        meta.enable_learning()
        assert meta._learning_enabled

    def test_analyze_performance_skips_with_insufficient_data(self, meta):
        result = meta.analyze_performance()
        assert result.get("analyzed") is False

    def test_analyze_performance_with_sufficient_data(self, meta):
        for i in range(10):
            trade = make_trade(profit=50.0 if i < 7 else -30.0, regime="RANGING")
            meta.record_trade(trade)
        result = meta.analyze_performance(force=True)
        assert "by_regime" in result
        assert "by_pattern" in result
        assert "adjustments" in result

    def test_analyze_by_regime_returns_performances(self, meta):
        for i in range(10):
            meta.record_trade(make_trade(profit=50.0 if i < 6 else -30.0, regime="RANGING"))
        perfs = meta._analyze_by_regime()
        assert len(perfs) > 0
        perf = perfs[0]
        assert perf.total_trades >= 10

    def test_analyze_by_pattern_returns_performances(self, meta):
        for i in range(5):
            meta.record_trade(make_trade(profit=50.0, primary_pattern="FVG",
                                        patterns=["FVG", "OB"]))
        perfs = meta._analyze_by_pattern()
        assert len(perfs) > 0

    def test_compute_regime_adjustments_low_win_rate(self, meta):
        for i in range(10):
            meta.record_trade(make_trade(profit=-50.0, regime="RANGING"))
        perfs = meta._analyze_by_regime()
        with_before = REGIME_PATTERN_MULTIPLIERS["FVG"].get(RegimeType.RANGING, 1.0)
        adjustments = meta._compute_regime_adjustments(perfs)
        if any("RANGING" in adj for adj in adjustments):
            with_after = REGIME_PATTERN_MULTIPLIERS["FVG"].get(RegimeType.RANGING, 1.0)
            assert with_after <= with_before

    def test_compute_regime_adjustments_high_win_rate(self, meta):
        for i in range(10):
            meta.record_trade(make_trade(profit=100.0, regime="RANGING"))
        perfs = meta._analyze_by_regime()
        with_before = REGIME_PATTERN_MULTIPLIERS["FVG"].get(RegimeType.RANGING, 1.0)
        adjustments = meta._compute_regime_adjustments(perfs)
        if any("RANGING" in adj for adj in adjustments):
            with_after = REGIME_PATTERN_MULTIPLIERS["FVG"].get(RegimeType.RANGING, 1.0)
            assert with_after >= with_before

    def test_compute_pattern_adjustments_low_win_rate(self, meta):
        for i in range(5):
            meta.record_trade(make_trade(profit=-50.0, primary_pattern="FVG",
                                        patterns=["FVG"]))
        perfs = meta._analyze_by_pattern()
        adjustments = meta._compute_pattern_adjustments(perfs)
        assert len(adjustments) > 0
        assert any("FVG" in adj for adj in adjustments)

    def test_get_best_patterns_for_regime(self, meta):
        for i in range(5):
            meta.record_trade(make_trade(profit=100.0, regime="RANGING",
                                        primary_pattern="FVG", patterns=["FVG"]))
        meta.analyze_performance(force=True)
        best = meta.get_best_patterns_for_regime("RANGING")
        assert isinstance(best, list)

    def test_get_worst_patterns_for_regime(self, meta):
        for i in range(5):
            meta.record_trade(make_trade(profit=-50.0, regime="RANGING",
                                        primary_pattern="BREAKER", patterns=["BREAKER"]))
        meta.analyze_performance(force=True)
        worst = meta.get_worst_patterns_for_regime("RANGING")
        assert isinstance(worst, list)

    def test_get_regime_summary(self, meta):
        for i in range(5):
            meta.record_trade(make_trade(profit=50.0, regime="RANGING"))
        summary = meta.get_regime_summary("RANGING")
        assert summary["regime"] == "RANGING"
        assert summary["total"] == 5

    def test_get_regime_summary_empty(self, meta):
        summary = meta.get_regime_summary("NONEXISTENT")
        assert summary["total"] == 0

    def test_get_learning_status(self, meta):
        meta.record_trade(make_trade())
        status = meta.get_learning_status()
        assert status["enabled"] is True
        assert status["total_trades_recorded"] == 1
        assert status["pattern_types_tracked"] > 0

    def test_cleanup_old_data(self, meta):
        import sqlite3
        conn = sqlite3.connect(str(meta.db_path))
        conn.execute("""
            INSERT INTO trade_records (symbol, direction, profit, score, conviction,
             regime, session, patterns_found, duration_minutes, timestamp)
            VALUES ('XAUUSDm', 'BUY', 100.0, 75.0, 0.6, 'RANGING', 'LONDON_OPEN',
             '[]', 30, datetime('now', '-91 days'))
        """)
        conn.commit()
        conn.close()
        meta.cleanup_old_data(days=90)
        conn = sqlite3.connect(str(meta.db_path))
        count = conn.execute("SELECT COUNT(*) FROM trade_records").fetchone()[0]
        conn.close()
        assert count == 0

    def test_analyze_performance_respects_interval(self, meta):
        for i in range(10):
            meta.record_trade(make_trade())
        meta.analyze_performance(force=True)
        assert meta._last_analysis is not None
        result = meta.analyze_performance(force=False)
        assert result.get("analyzed") is False

    def test_analyze_performance_force_ignores_interval(self, meta):
        for i in range(10):
            meta.record_trade(make_trade())
        meta.analyze_performance(force=True)
        meta._last_analysis = datetime.now() - timedelta(hours=5)
        result = meta.analyze_performance(force=True)
        assert "by_regime" in result

    def test_record_trade_updates_pattern_performance(self, meta):
        meta.record_trade(make_trade(profit=50.0, primary_pattern="FVG",
                                    patterns=["FVG", "OB"]))
        conn = __import__("sqlite3").connect(str(meta.db_path))
        rows = conn.execute("SELECT pattern_type, total_occurrences, as_primary FROM pattern_performance").fetchall()
        conn.close()
        pattern_types = {r[0] for r in rows}
        assert "FVG" in pattern_types
        assert "OB" in pattern_types

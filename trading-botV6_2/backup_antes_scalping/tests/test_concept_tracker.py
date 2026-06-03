import pytest

from src.core.concept_tracker import ConceptStats, ConceptPerformanceTracker
from src.core.md_concepts import MDConcept


class TestConceptStats:
    def test_initial_state(self):
        stats = ConceptStats()
        assert stats.total_signals == 0
        assert stats.winning_signals == 0
        assert stats.losing_signals == 0
        assert stats.total_pnl == 0.0
        assert stats.weight == 1.0
        assert stats.win_rate == 0.0
        assert stats.expectancy == 0.0

    def test_record_win(self):
        stats = ConceptStats()
        stats.record(won=True, pnl=50.0, confidence=0.8)
        assert stats.total_signals == 1
        assert stats.winning_signals == 1
        assert stats.total_pnl == 50.0
        assert stats.total_confidence == 0.8
        assert stats.win_rate == 1.0

    def test_record_loss(self):
        stats = ConceptStats()
        stats.record(won=False, pnl=-30.0, confidence=0.6)
        assert stats.total_signals == 1
        assert stats.losing_signals == 1
        assert stats.total_pnl == -30.0
        assert stats.win_rate == 0.0

    def test_weight_stays_default_below_min_samples(self):
        stats = ConceptStats()
        for _ in range(9):
            stats.record(won=True, pnl=10.0, confidence=0.5)
        assert stats.weight == 1.0

    def test_weight_adjusts_after_min_samples(self):
        stats = ConceptStats()
        for _ in range(10):
            stats.record(won=True, pnl=10.0, confidence=0.5)
        assert stats.weight > 1.0

    def test_weight_floor(self):
        stats = ConceptStats()
        for _ in range(10):
            stats.record(won=False, pnl=-100.0, confidence=0.5)
        assert stats.weight >= 0.0

    def test_weight_ceiling(self):
        stats = ConceptStats()
        for _ in range(10):
            stats.record(won=True, pnl=1000.0, confidence=0.5)
        assert stats.weight <= 2.0

    def test_expectancy_calculation(self):
        stats = ConceptStats()
        stats.record(won=True, pnl=100.0, confidence=0.9)
        stats.record(won=False, pnl=-20.0, confidence=0.7)
        assert stats.expectancy == 40.0
        assert stats.avg_confidence == 0.8

    def test_to_dict_roundtrip(self):
        stats = ConceptStats()
        stats.record(won=True, pnl=50.0, confidence=0.8)
        data = stats.to_dict()
        restored = ConceptStats.from_dict(data)
        assert restored.total_signals == stats.total_signals
        assert restored.winning_signals == stats.winning_signals
        assert restored.total_pnl == stats.total_pnl
        assert restored.weight == stats.weight


class TestConceptPerformanceTracker:
    @pytest.fixture
    def tracker(self, tmp_path):
        p = tmp_path / "concept_tracker.json"
        return ConceptPerformanceTracker(persistence_path=p)

    def test_get_weight_default(self, tracker):
        w = tracker.get_weight("XAUUSD", MDConcept.POD)
        assert w == 1.0

    def test_record_and_get_weight(self, tracker):
        tracker.record_result("XAUUSD", MDConcept.POD, won=True, pnl=50.0, confidence=0.8)
        w = tracker.get_weight("XAUUSD", MDConcept.POD)
        assert w == 1.0  # weight stays default until >= 10 samples

    def test_multiple_concepts_independent(self, tracker):
        tracker.record_result("XAUUSD", MDConcept.POD, won=True, pnl=50.0, confidence=0.8)
        tracker.record_result("XAUUSD", MDConcept.INTERVAL, won=False, pnl=-30.0, confidence=0.6)
        pod_w = tracker.get_weight("XAUUSD", MDConcept.POD)
        int_w = tracker.get_weight("XAUUSD", MDConcept.INTERVAL)
        assert pod_w == int_w  # both still default weight (< 10 samples)

    def test_multiple_symbols_independent(self, tracker):
        tracker.record_result("XAUUSD", MDConcept.POD, won=True, pnl=50.0, confidence=0.8)
        tracker.record_result("BTCUSD", MDConcept.POD, won=False, pnl=-30.0, confidence=0.6)
        assert tracker.get_weight("XAUUSD", MDConcept.POD) == tracker.get_weight("BTCUSD", MDConcept.POD)

    def test_get_stats(self, tracker):
        tracker.record_result("XAUUSD", MDConcept.POD, won=True, pnl=50.0, confidence=0.8)
        stats = tracker.get_stats("XAUUSD", MDConcept.POD)
        assert stats.total_signals == 1
        assert stats.winning_signals == 1

    def test_summary(self, tracker):
        tracker.record_result("XAUUSD", MDConcept.POD, won=True, pnl=50.0, confidence=0.8)
        tracker.record_result("XAUUSD", MDConcept.INTERVAL, won=False, pnl=-30.0, confidence=0.6)
        summary = tracker.summary("XAUUSD")
        assert len(summary) == 2
        assert "pod" in summary
        assert "interval" in summary

    def test_persistence(self, tmp_path):
        p = tmp_path / "concept_tracker.json"
        t1 = ConceptPerformanceTracker(persistence_path=p)
        t1.record_result("XAUUSD", MDConcept.POD, won=True, pnl=50.0, confidence=0.8)
        del t1

        t2 = ConceptPerformanceTracker(persistence_path=p)
        stats = t2.get_stats("XAUUSD", MDConcept.POD)
        assert stats.total_signals == 1
        assert stats.winning_signals == 1

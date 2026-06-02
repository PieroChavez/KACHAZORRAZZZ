import pytest
import numpy as np
from src.scoring.distributional_score import DistributionalScorer, DistributionalScore, merge_signals
from src.core.regime_detector import RegimeContext, RegimeType


def make_regime(regime_type=RegimeType.RANGING, confidence=0.6):
    return RegimeContext(
        regime=regime_type, confidence=confidence, strength=0.3,
        atr_ratio=1.0, adx_value=20.0, is_compressed=False,
        is_expanding=False, trend_alignment="NEUTRAL",
        pattern_multipliers={"FVG": 1.0},
    )


class TestDistributionalScore:
    def test_is_high_conviction(self):
        ds = DistributionalScore(mean=50.0, std=10.0, conviction=0.8)
        assert ds.is_high_conviction
        assert not ds.is_medium_conviction
        assert not ds.is_low_conviction

    def test_is_medium_conviction(self):
        ds = DistributionalScore(mean=50.0, std=10.0, conviction=0.5)
        assert ds.is_medium_conviction
        assert not ds.is_high_conviction
        assert not ds.is_low_conviction

    def test_is_low_conviction(self):
        ds = DistributionalScore(mean=50.0, std=10.0, conviction=0.2)
        assert ds.is_low_conviction
        assert not ds.is_high_conviction
        assert not ds.is_medium_conviction

    def test_risk_adjusted_score(self):
        ds = DistributionalScore(mean=50.0, std=1.0)
        assert ds.risk_adjusted_score == pytest.approx(25.0, rel=0.01)

    def test_risk_adjusted_score_with_zero_std(self):
        ds = DistributionalScore(mean=50.0, std=0.0)
        assert ds.risk_adjusted_score == 50.0


class TestDistributionalScorer:
    def test_compute_buy_direction(self):
        scorer = DistributionalScorer()
        regime = make_regime()
        ds = scorer.compute(
            buy_score=80.0, sell_score=20.0,
            tf_scores={"1H": (80.0, 20.0), "15min": (75.0, 25.0), "5min": (70.0, 30.0)},
            regime=regime,
        )
        assert ds.direction == "BUY"
        assert ds.mean > 0
        assert 0 <= ds.conviction <= 1
        assert 0 <= ds.convergence <= 1

    def test_compute_sell_direction(self):
        scorer = DistributionalScorer()
        regime = make_regime()
        ds = scorer.compute(
            buy_score=20.0, sell_score=80.0,
            tf_scores={"1H": (20.0, 80.0), "15min": (25.0, 75.0), "5min": (30.0, 70.0)},
            regime=regime,
        )
        assert ds.direction == "SELL"
        assert ds.mean < 0

    def test_compute_hold_direction(self):
        scorer = DistributionalScorer()
        regime = make_regime()
        ds = scorer.compute(
            buy_score=50.0, sell_score=50.0,
            tf_scores={"1H": (50.0, 50.0)},
            regime=regime,
        )
        assert ds.direction == "HOLD"
        assert ds.conviction == 0.0

    def test_convergence_all_aligned(self):
        scorer = DistributionalScorer()
        regime = make_regime()
        ds = scorer.compute(
            buy_score=80.0, sell_score=20.0,
            tf_scores={
                "1H": (80.0, 20.0),
                "15min": (75.0, 25.0),
                "5min": (70.0, 30.0),
            },
            regime=regime,
        )
        assert ds.convergence == 1.0

    def test_convergence_mixed(self):
        scorer = DistributionalScorer()
        regime = make_regime()
        ds = scorer.compute(
            buy_score=80.0, sell_score=20.0,
            tf_scores={
                "1H": (80.0, 20.0),
                "15min": (40.0, 60.0),
                "5min": (70.0, 30.0),
            },
            regime=regime,
        )
        assert ds.convergence < 1.0

    def test_per_timeframe_populated(self):
        scorer = DistributionalScorer()
        regime = make_regime()
        ds = scorer.compute(
            buy_score=80.0, sell_score=20.0,
            tf_scores={
                "1H": (80.0, 20.0),
                "15min": (70.0, 30.0),
            },
            regime=regime,
        )
        assert "1H" in ds.per_timeframe
        assert "15min" in ds.per_timeframe

    def test_per_group_populated(self):
        scorer = DistributionalScorer()
        regime = make_regime()
        ds = scorer.compute(
            buy_score=80.0, sell_score=20.0,
            tf_scores={
                "1H": (80.0, 20.0),
                "15min": (70.0, 30.0),
                "5min": (60.0, 40.0),
            },
            regime=regime,
        )
        for group in ("HTF", "MID", "LTF"):
            assert group in ds.per_group

    def test_ht_lt_alignment_bullish(self):
        scorer = DistributionalScorer()
        regime = make_regime()
        ds = scorer.compute(
            buy_score=80.0, sell_score=20.0,
            tf_scores={"1H": (80.0, 20.0), "5min": (70.0, 30.0)},
            regime=regime,
        )
        assert ds.ht_lt_alignment == "BULLISH_ALIGNED"

    def test_ht_lt_alignment_bearish(self):
        scorer = DistributionalScorer()
        regime = make_regime()
        ds = scorer.compute(
            buy_score=20.0, sell_score=80.0,
            tf_scores={"1H": (20.0, 80.0), "5min": (30.0, 70.0)},
            regime=regime,
        )
        assert ds.ht_lt_alignment == "BEARISH_ALIGNED"

    def test_ht_lt_alignment_mixed(self):
        scorer = DistributionalScorer()
        regime = make_regime()
        ds = scorer.compute(
            buy_score=80.0, sell_score=20.0,
            tf_scores={"1H": (80.0, 20.0), "5min": (20.0, 80.0)},
            regime=regime,
        )
        assert ds.ht_lt_alignment == "HTF_BULLISH_LTF_BEARISH"

    def test_notes_generated(self):
        scorer = DistributionalScorer()
        regime = make_regime()
        ds = scorer.compute(
            buy_score=80.0, sell_score=20.0,
            tf_scores={"M15": (80.0, 20.0)},
            regime=regime,
        )
        assert len(ds.notes) > 0
        assert "Score neto" in ds.notes[0]

    def test_conviction_boosted_by_strong_regime(self):
        scorer = DistributionalScorer()
        strong_regime = make_regime(RegimeType.STRONG_TREND_BULLISH, 0.9)
        weak_regime = make_regime(RegimeType.RANGING, 0.3)
        tf_scores = {"M15": (70.0, 30.0), "M5": (65.0, 35.0), "M1": (60.0, 40.0)}
        ds_strong = scorer.compute(70.0, 30.0, tf_scores, strong_regime)
        ds_weak = scorer.compute(70.0, 30.0, tf_scores, weak_regime)
        assert ds_strong.conviction >= ds_weak.conviction

    def test_group_std_populated(self):
        scorer = DistributionalScorer()
        regime = make_regime()
        ds = scorer.compute(
            buy_score=80.0, sell_score=20.0,
            tf_scores={
                "M15": (80.0, 20.0),
                "M5": (70.0, 30.0),
                "M1": (60.0, 40.0),
            },
            regime=regime,
        )
        for group in ("HTF", "MID", "LTF"):
            assert group in ds.group_std


class TestMergeSignals:
    def test_merge_signals_with_both_signals(self):
        scorer = DistributionalScorer()
        regime = make_regime()

        class FakeSignal:
            score = 75.0
            score_breakdown = {"htf_trend_aligned": 10.0, "tf_score_1H": 30.0}

        buy = FakeSignal()
        sell = FakeSignal()
        sell.score = 25.0
        sell.score_breakdown = {"htf_trend_aligned": 5.0, "tf_score_1H": 10.0}

        dist, buy_bd, sell_bd = merge_signals(buy, sell, regime, scorer)
        assert isinstance(dist, DistributionalScore)
        assert dist.direction == "BUY"
        assert isinstance(buy_bd, dict)
        assert isinstance(sell_bd, dict)

    def test_merge_signals_with_none_sell(self):
        scorer = DistributionalScorer()
        regime = make_regime()

        class FakeSignal:
            score = 75.0
            score_breakdown = {"htf_trend_aligned": 10.0}

        dist, buy_bd, sell_bd = merge_signals(FakeSignal(), None, regime, scorer)
        assert dist.direction == "BUY"
        assert buy_bd == {"htf_trend_aligned": 10.0}
        assert sell_bd == {}

    def test_merge_signals_without_tf_keys(self):
        scorer = DistributionalScorer()
        regime = make_regime()

        class FakeSignal:
            score = 50.0
            score_breakdown = {"htf_trend_aligned": 10.0}

        dist, _, _ = merge_signals(FakeSignal(), FakeSignal(), regime, scorer)
        assert dist.direction == "HOLD"

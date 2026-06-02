import pytest
import pandas as pd
import numpy as np
from src.core.continuous_decision import ContinuousDecider, ContinuousDecision, CONVICTION_VOLUME_MAP
from src.scoring.distributional_score import DistributionalScore
from src.core.regime_detector import RegimeContext, RegimeType


def make_regime(regime_type=RegimeType.RANGING, confidence=0.6, is_compressed=False, is_expanding=False):
    return RegimeContext(
        regime=regime_type, confidence=confidence, strength=0.3,
        atr_ratio=1.0, adx_value=20.0, is_compressed=is_compressed,
        is_expanding=is_expanding, trend_alignment="NEUTRAL",
        pattern_multipliers={},
    )


def make_dist(conviction=0.6, mean=50.0, std=10.0, direction="BUY"):
    return DistributionalScore(
        mean=mean, std=std, conviction=conviction, direction=direction,
        convergence=0.8, ht_lt_alignment="BULLISH_ALIGNED",
        per_timeframe={}, per_group={}, group_std={}, notes=[],
    )


class TestContinuousDecision:
    def test_should_trade_high_conviction(self):
        d = ContinuousDecision(
            direction="BUY", conviction=0.7, suggested_volume_pct=1.0,
            sl_width_multiplier=1.0, tp_width_multiplier=1.0,
            scale_entries_count=3, entry_aggressiveness="moderate",
            max_risk_pct=0.02,
        )
        assert d.should_trade

    def test_should_trade_low_conviction(self):
        d = ContinuousDecision(
            direction="BUY", conviction=0.2, suggested_volume_pct=0.0,
            sl_width_multiplier=1.0, tp_width_multiplier=1.0,
            scale_entries_count=0, entry_aggressiveness="none",
            max_risk_pct=0.0,
        )
        assert not d.should_trade

    def test_should_trade_hold(self):
        d = ContinuousDecision(
            direction="HOLD", conviction=0.5, suggested_volume_pct=0.6,
            sl_width_multiplier=1.0, tp_width_multiplier=1.0,
            scale_entries_count=2, entry_aggressiveness="moderate",
            max_risk_pct=0.02,
        )
        assert not d.should_trade


class TestContinuousDecider:
    def test_decide_with_none_dist(self):
        decider = ContinuousDecider()
        regime = make_regime()
        decision = decider.decide(None, regime)
        assert decision.direction == "HOLD"
        assert decision.conviction == 0.0
        assert not decision.should_trade

    def test_decide_low_conviction_hold(self):
        decider = ContinuousDecider()
        regime = make_regime()
        dist = make_dist(conviction=0.1, direction="BUY")
        decision = decider.decide(dist, regime)
        assert decision.direction == "HOLD"
        assert decision.suggested_volume_pct == 0.0

    def test_decide_returns_trade_above_threshold(self):
        decider = ContinuousDecider()
        regime = make_regime()
        dist = make_dist(conviction=0.6)
        decision = decider.decide(dist, regime)
        assert decision.direction == "BUY"
        assert decision.conviction == 0.6
        assert decision.should_trade

    def test_conviction_to_volume_mapping(self):
        decider = ContinuousDecider()
        assert decider._conviction_to_volume(0.0) == 0.0
        assert decider._conviction_to_volume(0.15) == 0.0
        assert decider._conviction_to_volume(0.3) == 0.4
        assert decider._conviction_to_volume(0.5) == 0.8
        assert decider._conviction_to_volume(0.7) == 1.2
        assert decider._conviction_to_volume(0.9) == 2.0
        assert decider._conviction_to_volume(1.0) == 3.0

    def test_sl_tp_multipliers_by_conviction(self):
        decider = ContinuousDecider()
        regime = make_regime()
        sl, tp = decider._get_sl_tp_multipliers(0.85, regime)
        assert sl == 0.8
        assert tp == 1.5
        sl, tp = decider._get_sl_tp_multipliers(0.65, regime)
        assert sl == 0.9
        assert tp == 1.2
        sl, tp = decider._get_sl_tp_multipliers(0.45, regime)
        assert sl == 1.0
        assert tp == 1.0
        sl, tp = decider._get_sl_tp_multipliers(0.2, regime)
        assert sl == 1.2
        assert tp == 0.8

    def test_scale_count_by_conviction(self):
        decider = ContinuousDecider()
        regime = make_regime()
        assert decider._get_scale_count(0.95, regime) == 5
        assert decider._get_scale_count(0.8, regime) == 3
        assert decider._get_scale_count(0.6, regime) == 2
        assert decider._get_scale_count(0.4, regime) == 1

    def test_aggressiveness_by_conviction(self):
        decider = ContinuousDecider()
        regime = make_regime()
        assert decider._get_aggressiveness(0.9, regime) == "aggressive"
        assert decider._get_aggressiveness(0.6, regime) == "moderate"
        assert decider._get_aggressiveness(0.3, regime) == "conservative"

    def test_max_risk_pct_scaling(self):
        decider = ContinuousDecider(base_risk_pct=0.02)
        regime = make_regime()
        assert decider._get_max_risk_pct(0.9, regime) == 0.04
        assert decider._get_max_risk_pct(0.7, regime) == 0.03
        assert decider._get_max_risk_pct(0.4, regime) == 0.02

    def test_high_volatility_reduces_volume(self):
        decider = ContinuousDecider()
        regime = make_regime(RegimeType.HIGH_VOLATILITY, is_expanding=True)
        dist = make_dist(conviction=0.7)
        decision = decider.decide(dist, regime)
        assert decision.suggested_volume_pct < 1.2
        assert decision.sl_width_multiplier > 1.0

    def test_low_volatility_reduces_scales(self):
        decider = ContinuousDecider()
        regime = make_regime(RegimeType.LOW_VOLATILITY, is_compressed=True)
        dist = make_dist(conviction=0.8)
        decision = decider.decide(dist, regime)
        assert decision.scale_entries_count <= 3

    def test_strong_trend_extends_tp(self):
        decider = ContinuousDecider()
        regime = make_regime(RegimeType.STRONG_TREND_BULLISH)
        dist = make_dist(conviction=0.7)
        decision = decider.decide(dist, regime)
        assert decision.tp_width_multiplier >= 1.3
        assert decision.sl_width_multiplier <= 1.0

    def test_ranging_reduces_exposure(self):
        decider = ContinuousDecider()
        regime = make_regime(RegimeType.RANGING)
        dist = make_dist(conviction=0.6)
        decision = decider.decide(dist, regime)
        assert decision.suggested_volume_pct < 1.0

    def test_high_dispersion_reduces_volume(self):
        decider = ContinuousDecider()
        regime = make_regime()
        dist = make_dist(conviction=0.7, mean=50.0, std=40.0)
        decision = decider.decide(dist, regime)
        base_vol = decider._conviction_to_volume(0.7)
        assert decision.suggested_volume_pct < base_vol

    def test_expanding_adjusts_sl_tp(self):
        decider = ContinuousDecider()
        regime = make_regime(RegimeType.TRANSITION, is_expanding=True)
        sl, tp = decider._adjust_for_regime(1.0, 1.0, regime)
        assert sl == 1.2
        assert tp == 1.3

    def test_compressed_adjusts_sl_tp(self):
        decider = ContinuousDecider()
        regime = make_regime(RegimeType.TRANSITION, is_compressed=True)
        sl, tp = decider._adjust_for_regime(1.0, 1.0, regime)
        assert sl == 0.8
        assert tp == 0.9

    def test_max_risk_reduced_in_high_vol(self):
        decider = ContinuousDecider(base_risk_pct=0.02)
        regime = make_regime(RegimeType.HIGH_VOLATILITY)
        assert decider._get_max_risk_pct(0.9, regime) == 0.012

    def test_notes_generated(self):
        decider = ContinuousDecider()
        regime = make_regime()
        dist = make_dist(conviction=0.7)
        decision = decider.decide(dist, regime)
        assert len(decision.notes) > 0
        assert any("Convicción" in n for n in decision.notes)

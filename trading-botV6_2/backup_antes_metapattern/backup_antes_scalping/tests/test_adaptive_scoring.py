import pytest
from src.core.adaptive_scoring import AdaptiveScorer, AdaptiveWeights, PATTERN_TO_WEIGHT_MAP, CONTEXT_WEIGHT_ADJUSTMENTS
from src.core.regime_detector import RegimeContext, RegimeType
from src.core.strategy_engine import ScoringConfig


def make_regime(regime_type=RegimeType.RANGING, confidence=0.6, pattern_mults=None):
    return RegimeContext(
        regime=regime_type, confidence=confidence, strength=0.3,
        atr_ratio=1.0, adx_value=20.0, is_compressed=False,
        is_expanding=False, trend_alignment="NEUTRAL",
        pattern_multipliers=pattern_mults or {},
    )


class TestAdaptiveWeights:
    def test_get_returns_adjusted_when_present(self):
        cfg = ScoringConfig()
        regime = make_regime()
        weights = AdaptiveWeights(base=cfg, regime=regime, adjusted={"htf_trend_aligned": 15.0})
        assert weights.get("htf_trend_aligned") == 15.0

    def test_get_falls_back_to_base(self):
        cfg = ScoringConfig(htf_trend_aligned=20.0)
        regime = make_regime()
        weights = AdaptiveWeights(base=cfg, regime=regime, adjusted={})
        assert weights.get("htf_trend_aligned") == 20.0

    def test_get_returns_zero_for_unknown(self):
        cfg = ScoringConfig()
        regime = make_regime()
        weights = AdaptiveWeights(base=cfg, regime=regime, adjusted={})
        assert weights.get("nonexistent") == 0.0


class TestAdaptiveScorer:
    def test_compute_weights_returns_adaptive_weights(self):
        cfg = ScoringConfig()
        scorer = AdaptiveScorer(cfg)
        regime = make_regime(RegimeType.STRONG_TREND_BULLISH, 0.8,
                             pattern_mults={"FVG": 1.8, "OB": 1.2})
        result = scorer.compute_weights(regime)
        assert isinstance(result, AdaptiveWeights)
        assert result.regime == regime
        assert len(result.adjusted) > 0

    def test_compute_weights_adjusts_pattern_weights(self):
        cfg = ScoringConfig(fvg_detected=18.0)
        scorer = AdaptiveScorer(cfg)
        regime = make_regime(RegimeType.RANGING, 0.5,
                             pattern_mults={"FVG": 0.5})
        result = scorer.compute_weights(regime)
        adjusted = result.get("fvg_detected")
        assert adjusted == pytest.approx(18.0 * 0.5, rel=0.01)

    def test_compute_weights_adjusts_context_weights(self):
        cfg = ScoringConfig(htf_trend_aligned=20.0)
        scorer = AdaptiveScorer(cfg)
        regime = make_regime(RegimeType.STRONG_TREND_BULLISH, 0.8)
        result = scorer.compute_weights(regime)
        adjusted = result.get("htf_trend_aligned")
        expected = 20.0 * CONTEXT_WEIGHT_ADJUSTMENTS["htf_trend_aligned"][RegimeType.STRONG_TREND_BULLISH]
        assert adjusted == pytest.approx(expected, rel=0.01)

    def test_compute_weights_caches_results(self):
        cfg = ScoringConfig()
        scorer = AdaptiveScorer(cfg)
        regime = make_regime()
        r1 = scorer.compute_weights(regime)
        r2 = scorer.compute_weights(regime)
        assert r1 is r2

    def test_cache_clears_after_50_entries(self):
        cfg = ScoringConfig()
        scorer = AdaptiveScorer(cfg)
        results = []
        for i in range(60):
            r = make_regime(RegimeType.RANGING, confidence=min(1.0, i / 50))
            results.append(scorer.compute_weights(r))
        assert len(scorer._cache) <= 50

    def test_get_pattern_regime_multiplier_matches(self):
        cfg = ScoringConfig()
        scorer = AdaptiveScorer(cfg)
        regime = make_regime(RegimeType.RANGING, pattern_mults={"FVG": 0.5})
        mult = scorer.get_pattern_regime_multiplier("FVG_BULLISH", regime)
        assert mult == 0.5

    def test_get_pattern_regime_multiplier_returns_default(self):
        cfg = ScoringConfig()
        scorer = AdaptiveScorer(cfg)
        regime = make_regime(RegimeType.RANGING)
        mult = scorer.get_pattern_regime_multiplier("UNKNOWN_PATTERN", regime)
        assert mult == 1.0

    def test_adjust_for_confidence(self):
        cfg = ScoringConfig()
        scorer = AdaptiveScorer(cfg)
        regime = make_regime(RegimeType.STRONG_TREND_BULLISH, 0.8)
        class FakePattern:
            confidence = 0.7
        adjusted = scorer.adjust_for_confidence(100.0, FakePattern(), regime, 0.9)
        assert 80.0 <= adjusted <= 90.0

    def test_adjust_for_confidence_without_pattern(self):
        cfg = ScoringConfig()
        scorer = AdaptiveScorer(cfg)
        regime = make_regime(RegimeType.RANGING, 0.5)
        adjusted = scorer.adjust_for_confidence(100.0, None, regime, 0.5)
        assert 30.0 <= adjusted <= 70.0

    def test_notes_contain_regime_info(self):
        cfg = ScoringConfig()
        scorer = AdaptiveScorer(cfg)
        regime = make_regime(RegimeType.HIGH_VOLATILITY, 0.7)
        result = scorer.compute_weights(regime)
        assert len(result.notes) > 0
        assert "HIGH_VOLATILITY" in result.notes[0]

    def test_multipliers_recorded(self):
        cfg = ScoringConfig(fvg_detected=18.0)
        scorer = AdaptiveScorer(cfg)
        regime = make_regime(RegimeType.RANGING, pattern_mults={"FVG": 0.5})
        result = scorer.compute_weights(regime)
        assert "fvg_detected" in result.multipliers
        assert result.multipliers["fvg_detected"] == 0.5


class TestPATTERNTOWEIGHTMAP:
    def test_all_weight_names_map(self):
        assert "fvg_detected" in PATTERN_TO_WEIGHT_MAP
        assert "order_block_valid" in PATTERN_TO_WEIGHT_MAP
        assert "breaker_retest" in PATTERN_TO_WEIGHT_MAP
        assert "liquidity_sweep_ltf" in PATTERN_TO_WEIGHT_MAP
        assert "wyckoff_phase_c_spring" in PATTERN_TO_WEIGHT_MAP
        assert "void_scalp_confirmed" in PATTERN_TO_WEIGHT_MAP

    def test_trb_weights_map(self):
        assert PATTERN_TO_WEIGHT_MAP["trb_manipulation_detected"] == "TRB"
        assert PATTERN_TO_WEIGHT_MAP["trb_displacement"] == "TRB"
        assert PATTERN_TO_WEIGHT_MAP["trb_retest"] == "TRB"

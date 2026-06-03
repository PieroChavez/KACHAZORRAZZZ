import time
from unittest.mock import patch, MagicMock, PropertyMock
import numpy as np
import pytest

from src.core.adaptive_cooldown import (
    AdaptiveCooldownConfig,
    AdaptiveCooldownEngine,
    CooldownCalculator,
    RecoveryPlanManager,
    ThompsonSampler,
    RecoveryStage,
    CooldownReason,
    CooldownDecision,
    StreakState,
    RecoveryState,
    ThompsonState,
)


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture
def cfg():
    return AdaptiveCooldownConfig(
        base_cooldown_minutes=5.0,
        min_cooldown_minutes=1.0,
        max_cooldown_minutes=60.0,
        cooldown_atr_ratio_weight=2.0,
        cooldown_consecutive_loss_weight=5.0,
        max_consecutive_losses_before_stop=5,
        thompson_min_samples=3,
        thompson_re_enable_threshold=0.15,
        thompson_max_disable_minutes=1440.0,
        recovery_target_win_rate=0.40,
        recovery_trades_to_advance=5,
        recovery_min_advance_win_rate=0.50,
    )


@pytest.fixture
def fixed_time():
    ts = 1_000_000.0
    with patch("src.core.adaptive_cooldown.time.time", return_value=ts):
        yield ts


@pytest.fixture
def thompson(cfg):
    return ThompsonSampler(cfg)


# ─────────────────────────────────────────────
# ADAPTIVE COOLDOWN CONFIG
# ─────────────────────────────────────────────

class TestAdaptiveCooldownConfig:
    def test_defaults(self):
        c = AdaptiveCooldownConfig()
        assert c.base_cooldown_minutes == 5.0
        assert c.max_cooldown_minutes == 60.0
        assert c.thompson_min_samples == 3
        assert c.loss_volume_reductions == [1.0, 0.7, 0.5, 0.3, 0.15, 0.0]
        assert c.recovery_stage_volume_mults["EMERGENCY"] == 0.0
        assert c.recovery_stage_conviction_bonuses["ALERT"] == 0.05
        assert c.recovery_stage_duration_trades["REDUCE"] == 10
        assert c.recovery_target_win_rate == 0.40


# ─────────────────────────────────────────────
# THOMPSON SAMPLER
# ─────────────────────────────────────────────

class TestThompsonSampler:
    def test_get_state_initializes_default(self, thompson):
        state = thompson.get_state("test_pattern")
        assert state.alpha == 2.0
        assert state.beta == 2.0
        assert state.total_samples == 0
        assert state.win_rate == 0.5

    def test_get_state_caches(self, thompson):
        s1 = thompson.get_state("p1")
        s2 = thompson.get_state("p1")
        assert s1 is s2

    def test_record_outcome_win(self, thompson):
        thompson.record_outcome("p", won=True)
        state = thompson.get_state("p")
        assert state.alpha == 3.0
        assert state.beta == 2.0
        assert state.total_samples == 1
        assert state.win_rate == 3.0 / 5.0

    def test_record_outcome_loss(self, thompson):
        thompson.record_outcome("p", won=False)
        state = thompson.get_state("p")
        assert state.alpha == 2.0
        assert state.beta == 3.0
        assert state.total_samples == 1
        assert state.win_rate == 2.0 / 5.0

    def test_record_outcome_accumulates(self, thompson):
        for _ in range(5):
            thompson.record_outcome("p", won=True)
        state = thompson.get_state("p")
        assert state.total_samples == 5
        assert state.alpha == 7.0
        assert state.beta == 2.0

    def test_sample_below_min_samples_returns_05(self, thompson):
        p = thompson.sample("no_samples")
        assert p == 0.5

    def test_sample_after_min_samples(self, thompson):
        for _ in range(5):
            thompson.record_outcome("p", won=True)
        with patch("numpy.random.beta", return_value=0.85):
            p = thompson.sample("p")
            np.random.beta.assert_called_once_with(7.0, 2.0)
            assert p == 0.85

    def test_should_re_enable_disabled_returns_false(self, thompson, fixed_time):
        for _ in range(5):
            thompson.record_outcome("p", won=True)
        thompson.disable("p", cooldown_minutes=60.0)
        enabled, prob = thompson.should_re_enable("p")
        assert enabled is False
        assert prob == 0.0

    def test_should_re_enable_below_min_samples(self, thompson):
        enabled, prob = thompson.should_re_enable("p")
        assert enabled is False
        assert prob == 0.0

    def test_should_re_enable_above_threshold(self, thompson, fixed_time):
        for _ in range(5):
            thompson.record_outcome("p", won=True)
        with patch("numpy.random.beta", return_value=0.85):
            enabled, prob = thompson.should_re_enable("p")
            assert enabled is True
            assert prob == 0.85

    def test_disable_sets_timer(self, thompson, fixed_time):
        thompson.disable("p", cooldown_minutes=30.0)
        state = thompson.get_state("p")
        assert state.disabled_until == 1_000_000.0 + 1800.0
        assert state.disable_count == 1

    def test_is_disabled(self, thompson, fixed_time):
        assert not thompson.is_disabled("p")
        thompson.disable("p", cooldown_minutes=10.0)
        assert thompson.is_disabled("p")

    def test_is_disabled_expired(self, thompson):
        thompson.disable("p", cooldown_minutes=0.0)
        assert not thompson.is_disabled("p")

    def test_get_remaining_cooldown(self, thompson, fixed_time):
        thompson.disable("p", cooldown_minutes=30.0)
        remaining = thompson.get_remaining_cooldown("p")
        assert remaining == 30.0

    def test_get_summary_structure(self, thompson, fixed_time):
        for _ in range(4):
            thompson.record_outcome("p", won=True)
        with patch("numpy.random.beta", return_value=0.72):
            summary = thompson.get_summary("p")
        assert summary["alpha"] == 6.0
        assert summary["beta"] == 2.0
        assert summary["samples"] == 4
        assert summary["win_rate"] == 0.75
        assert summary["disabled"] is False
        assert summary["disable_count"] == 0

    def test_get_summary_disabled(self, thompson, fixed_time):
        for _ in range(4):
            thompson.record_outcome("p", won=True)
        thompson.disable("p", cooldown_minutes=15.0)
        summary = thompson.get_summary("p")
        assert summary["disabled"] is True
        assert summary["remaining_min"] == 15.0
        assert summary["disable_count"] == 1


# ─────────────────────────────────────────────
# COOLDOWN CALCULATOR
# ─────────────────────────────────────────────

class TestCooldownCalculator:
    @pytest.fixture
    def calc(self, cfg):
        return CooldownCalculator(cfg)

    def test_calculate_normal(self, calc):
        cd = calc.calculate(1.0, "MEDIUM", 0, RecoveryStage.NORMAL)
        assert cd == 5.0

    def test_calculate_high_atr(self, calc):
        cd = calc.calculate(2.0, "MEDIUM", 0, RecoveryStage.NORMAL)
        expected = 5.0 * (1.0 + (2.0 - 1.5) * 2.0) * 1.0 * 1.0 * 1.0
        assert cd == pytest.approx(expected)

    def test_calculate_low_atr(self, calc):
        cd = calc.calculate(0.5, "MEDIUM", 0, RecoveryStage.NORMAL)
        expected = round(5.0 * max(0.5, 0.5 / 0.7) * 1.0 * 1.0 * 1.0, 1)
        assert cd == expected

    def test_calculate_high_volatility(self, calc):
        cd = calc.calculate(1.0, "HIGH", 0, RecoveryStage.NORMAL)
        assert cd == 5.0 * 1.0 * 1.5 * 1.0 * 1.0

    def test_calculate_low_volatility(self, calc):
        cd = calc.calculate(1.0, "LOW", 0, RecoveryStage.NORMAL)
        assert cd == 5.0 * 1.0 * 0.7 * 1.0 * 1.0

    def test_calculate_with_consecutive_losses(self, calc):
        cd = calc.calculate(1.0, "MEDIUM", 3, RecoveryStage.NORMAL)
        raw = 5.0 * 1.0 * 1.0 * (1.0 + 3 * 5.0) * 1.0
        expected = round(min(60.0, max(1.0, raw)), 1)
        assert cd == expected

    def test_calculate_with_recovery_stage(self, calc):
        cd = calc.calculate(1.0, "MEDIUM", 0, RecoveryStage.EMERGENCY)
        assert cd == 5.0 * 1.0 * 1.0 * 1.0 * 8.0

    def test_calculate_clamps_to_min(self, calc):
        c = AdaptiveCooldownConfig(min_cooldown_minutes=5.0, max_cooldown_minutes=60.0)
        calc2 = CooldownCalculator(c)
        cd = calc2.calculate(0.1, "LOW", 0, RecoveryStage.NORMAL)
        assert cd == 5.0

    def test_calculate_clamps_to_max(self, calc):
        cd = calc.calculate(5.0, "HIGH", 5, RecoveryStage.EMERGENCY)
        assert cd == 60.0

    @pytest.mark.parametrize("losses,expected_mult", [
        (0, 1.0),
        (1, 1.0),
        (2, 0.7),
        (3, 0.5),
        (4, 0.3),
        (5, 0.15),
        (10, 0.0),
    ])
    def test_get_volume_multiplier_by_losses(self, calc, losses, expected_mult):
        mult = calc.get_volume_multiplier(losses, RecoveryStage.NORMAL)
        assert mult == pytest.approx(expected_mult)

    def test_get_volume_multiplier_recovery_reduces_further(self, calc):
        normal = calc.get_volume_multiplier(1, RecoveryStage.NORMAL)
        emergency = calc.get_volume_multiplier(1, RecoveryStage.EMERGENCY)
        assert emergency < normal

    def test_get_conviction_requirement_increases_with_losses(self, calc):
        base = calc.get_conviction_requirement(0, RecoveryStage.NORMAL, 1.0)
        high = calc.get_conviction_requirement(4, RecoveryStage.EMERGENCY, 2.0)
        assert high > base

    def test_get_conviction_requirement_caps_at_max(self, calc):
        conv = calc.get_conviction_requirement(10, RecoveryStage.EMERGENCY, 5.0)
        assert conv == 0.60

    def test_should_hot_pause_normal(self, calc):
        should, pause = calc.should_hot_pause(0, RecoveryStage.NORMAL)
        assert not should
        assert pause == 0.0

    def test_should_hot_pause_emergency(self, calc):
        should, pause = calc.should_hot_pause(0, RecoveryStage.EMERGENCY)
        assert should
        assert pause == 240.0

    def test_should_hot_pause_max_losses(self, calc):
        should, pause = calc.should_hot_pause(5, RecoveryStage.NORMAL)
        assert should
        assert pause == 15.0

    def test_should_hot_pause_escalates(self, calc):
        should, pause = calc.should_hot_pause(7, RecoveryStage.NORMAL)
        assert should
        assert pause == 15.0 * (2.0 ** 2)


# ─────────────────────────────────────────────
# RECOVERY PLAN MANAGER
# ─────────────────────────────────────────────

class TestRecoveryPlanManager:
    @pytest.fixture
    def mgr(self, cfg):
        return RecoveryPlanManager(cfg)

    def test_initial_state_normal(self, mgr):
        state = mgr.get_state("EURUSD")
        assert state.stage == RecoveryStage.NORMAL
        assert state.trades_in_stage == 0
        assert state.wins_in_stage == 0
        assert state.losses_in_stage == 0

    def test_get_state_caches_by_symbol(self, mgr):
        s1 = mgr.get_state("EURUSD")
        s2 = mgr.get_state("EURUSD")
        assert s1 is s2

    def test_get_state_default_is_global(self, mgr):
        s1 = mgr.get_state()
        s2 = mgr.get_state("__global__")
        assert s1 is s2

    def test_record_trade_win(self, mgr):
        mgr.record_trade("EURUSD", profit=10.0, consecutive_losses=0)
        state = mgr.get_state("EURUSD")
        assert state.trades_in_stage == 1
        assert state.wins_in_stage == 1
        assert state.losses_in_stage == 0

    def test_record_trade_loss(self, mgr):
        mgr.record_trade("EURUSD", profit=-10.0, consecutive_losses=1)
        state = mgr.get_state("EURUSD")
        assert state.trades_in_stage == 1
        assert state.wins_in_stage == 0
        assert state.losses_in_stage == 1

    def test_transition_normal_to_alert_after_2_losses(self, mgr):
        mgr.record_trade("EURUSD", -10.0, consecutive_losses=2)
        state = mgr.get_state("EURUSD")
        assert state.stage == RecoveryStage.ALERT
        assert state.trades_in_stage == 0

    def test_no_transition_with_1_loss(self, mgr):
        mgr.record_trade("EURUSD", -10.0, consecutive_losses=1)
        state = mgr.get_state("EURUSD")
        assert state.stage == RecoveryStage.NORMAL

    def test_transition_alert_to_reduce_after_4_losses(self, mgr):
        mgr.get_state("EURUSD").stage = RecoveryStage.ALERT
        mgr.record_trade("EURUSD", -10.0, consecutive_losses=4)
        state = mgr.get_state("EURUSD")
        assert state.stage == RecoveryStage.REDUCE
        assert state.trades_in_stage == 0

    def test_transition_reduce_to_emergency_at_max_losses(self, mgr):
        mgr.get_state("EURUSD").stage = RecoveryStage.REDUCE
        mgr.record_trade("EURUSD", -10.0, consecutive_losses=5)
        state = mgr.get_state("EURUSD")
        assert state.stage == RecoveryStage.EMERGENCY

    def test_transition_alert_back_to_normal_after_recovery(self, mgr):
        state = mgr.get_state("EURUSD")
        state.stage = RecoveryStage.ALERT
        state.trades_in_stage = 6
        state.wins_in_stage = 4
        state.losses_in_stage = 2
        mgr._evaluate_transition(state, consecutive_losses=0)
        assert state.stage == RecoveryStage.NORMAL

    def test_transition_reduce_to_alert_after_recovery(self, mgr):
        state = mgr.get_state("EURUSD")
        state.stage = RecoveryStage.REDUCE
        state.trades_in_stage = 10
        state.wins_in_stage = 6
        state.losses_in_stage = 4
        mgr._evaluate_transition(state, consecutive_losses=1)
        assert state.stage == RecoveryStage.ALERT

    def test_transition_emergency_to_recover_after_recovery(self, mgr):
        state = mgr.get_state("EURUSD")
        state.stage = RecoveryStage.EMERGENCY
        state.trades_in_stage = 5
        state.wins_in_stage = 3
        state.losses_in_stage = 2
        mgr._evaluate_transition(state, consecutive_losses=0)
        assert state.stage == RecoveryStage.RECOVER

    def test_transition_recover_to_reduce_after_recovery(self, mgr):
        state = mgr.get_state("EURUSD")
        state.stage = RecoveryStage.RECOVER
        state.trades_in_stage = 8
        state.wins_in_stage = 4
        state.losses_in_stage = 4
        mgr._evaluate_transition(state, consecutive_losses=0)
        assert state.stage == RecoveryStage.REDUCE

    def test_get_stage_returns_max_severity(self, mgr):
        mgr.get_state("EURUSD").stage = RecoveryStage.EMERGENCY
        mgr.get_state("").stage = RecoveryStage.NORMAL
        assert mgr.get_stage("EURUSD") == RecoveryStage.EMERGENCY

    def test_get_stage_global_wins_when_symbol_lower(self, mgr):
        mgr.get_state("EURUSD").stage = RecoveryStage.NORMAL
        mgr.get_state("").stage = RecoveryStage.REDUCE
        assert mgr.get_stage("EURUSD") == RecoveryStage.REDUCE

    def test_get_summary_structure(self, mgr):
        mgr.record_trade("EURUSD", 10.0, 0)
        summary = mgr.get_summary("EURUSD")
        assert "stage" in summary
        assert "trades_in_stage" in summary
        assert "win_rate_in_stage" in summary
        assert summary["stage"] == "NORMAL"

    def test_global_and_symbol_independent(self, mgr):
        mgr.record_trade("EURUSD", -10.0, 2)
        mgr.record_trade("GBPUSD", -10.0, 1)
        assert mgr.get_state("EURUSD").stage == RecoveryStage.ALERT
        assert mgr.get_state("GBPUSD").stage == RecoveryStage.NORMAL


# ─────────────────────────────────────────────
# ADAPTIVE COOLDOWN ENGINE
# ─────────────────────────────────────────────

class TestAdaptiveCooldownEngine:
    @pytest.fixture
    def engine(self, cfg):
        return AdaptiveCooldownEngine(config=cfg)

    def test_init_sets_components(self, engine):
        assert engine.calculator is not None
        assert engine.thompson is not None
        assert engine.recovery is not None
        assert isinstance(engine._streaks, dict)

    def test_get_streak_initializes_empty(self, engine):
        streak = engine.get_streak("EURUSD")
        assert streak.consecutive_losses == 0
        assert streak.total_trades == 0

    def test_get_streak_caches(self, engine):
        s1 = engine.get_streak("EURUSD")
        s2 = engine.get_streak("EURUSD")
        assert s1 is s2

    def test_record_trade_win_resets_losses(self, engine):
        engine.get_streak("EURUSD").consecutive_losses = 3
        engine.record_trade("EURUSD", profit=10.0)
        streak = engine.get_streak("EURUSD")
        assert streak.consecutive_losses == 0
        assert streak.total_trades == 1
        assert streak.total_wins == 1
        assert streak.last_trade_result == "win"

    def test_record_trade_loss_increments(self, engine):
        engine.record_trade("EURUSD", profit=-10.0)
        streak = engine.get_streak("EURUSD")
        assert streak.consecutive_losses == 1
        assert streak.total_trades == 1
        assert streak.total_losses == 1
        assert streak.total_volume_lost == 10.0
        assert streak.last_trade_result == "loss"

    def test_record_trade_loss_triggers_hot_pause(self, engine):
        engine.record_trade("EURUSD", profit=-10.0, pattern="fvg")
        for _ in range(4):
            engine.record_trade("EURUSD", profit=-10.0)
        streak = engine.get_streak("EURUSD")
        assert streak.consecutive_losses == 5
        assert streak.hot_pause_until > 0

    def test_record_trade_updates_thompson(self, engine):
        with patch.object(engine.thompson, "record_outcome") as mock_record:
            engine.record_trade("EURUSD", profit=10.0, pattern="fvg")
            mock_record.assert_called_once_with("fvg", True)

    def test_is_hot_paused_false(self, engine):
        paused, remaining = engine.is_hot_paused("EURUSD")
        assert not paused
        assert remaining == 0.0

    def test_is_hot_paused_true(self, engine):
        streak = engine.get_streak("EURUSD")
        streak.hot_pause_until = time.time() + 300.0
        paused, remaining = engine.is_hot_paused("EURUSD")
        assert paused
        assert remaining > 0.0

    def test_evaluate_returns_hot_pause_when_active(self, engine, fixed_time):
        streak = engine.get_streak("EURUSD")
        streak.hot_pause_until = fixed_time + 600.0
        decision = engine.evaluate("EURUSD")
        assert decision.active is True
        assert decision.reason == CooldownReason.CONSECUTIVE_LOSSES
        assert decision.recommended_volume_mult == 0.0

    def test_evaluate_returns_disabled_pattern(self, engine, fixed_time):
        engine.thompson.disable("fvg", cooldown_minutes=60.0)
        decision = engine.evaluate("EURUSD", pattern="fvg")
        assert decision.active is True
        assert decision.reason == CooldownReason.PATTERN_DISABLED
        assert decision.recommended_volume_mult == 0.0

    def test_evaluate_returns_active_cooldown(self, engine, fixed_time):
        with patch.object(
            engine.calculator, "calculate", return_value=30.0
        ) as mock_calc:
            decision = engine.evaluate(
                "EURUSD", atr_ratio=2.0, volatility_regime="HIGH",
                last_trade_time=fixed_time - 60.0,
            )
            assert decision.active is True
            assert decision.remaining_minutes > 0.0
            mock_calc.assert_called_once()

    def test_evaluate_returns_no_cooldown_when_elapsed(self, engine, fixed_time):
        decision = engine.evaluate(
            "EURUSD", atr_ratio=1.0, volatility_regime="MEDIUM",
            last_trade_time=fixed_time - 600.0,
        )
        assert decision.active is False
        assert decision.remaining_minutes == 0.0
        assert decision.reason == CooldownReason.NONE

    def test_evaluate_includes_notes_in_non_normal_stage(self, engine, fixed_time):
        engine.recovery.get_state("EURUSD").stage = RecoveryStage.ALERT
        decision = engine.evaluate(
            "EURUSD", last_trade_time=fixed_time - 600.0,
        )
        assert len(decision.notes) >= 1
        assert any("Recovery stage" in n for n in decision.notes)

    def test_evaluate_includes_loss_notes(self, engine, fixed_time):
        engine.get_streak("EURUSD").consecutive_losses = 3
        decision = engine.evaluate(
            "EURUSD", last_trade_time=fixed_time - 7200.0,
        )
        assert any("Racha" in n for n in decision.notes)

    def test_evaluate_pattern_re_enable(self, engine, fixed_time):
        with patch.object(engine.thompson, "should_re_enable", return_value=(True, 0.85)):
            enabled, prob = engine.evaluate_pattern_re_enable("fvg")
            assert enabled is True
            assert prob == 0.85

    def test_disable_pattern(self, engine, fixed_time):
        engine.disable_pattern("fvg", cooldown_minutes=30.0)
        assert engine.is_pattern_disabled("fvg") is True

    def test_get_full_summary_structure(self, engine):
        engine.record_trade("EURUSD", profit=-5.0)
        engine.record_trade("EURUSD", profit=-8.0)
        summary = engine.get_full_summary("EURUSD")
        assert summary["symbol"] == "EURUSD"
        assert summary["consecutive_losses"] == 2
        assert summary["total_trades"] == 2
        assert summary["total_losses"] == 2
        assert "recovery_stage" in summary
        assert "recovery" in summary
        assert "cooldown_volume_mult" in summary
        assert "cooldown_min_conviction" in summary

    def test_get_all_pattern_thompson_summaries(self, engine):
        engine.record_trade("EURUSD", 10.0, pattern="fvg")
        engine.record_trade("EURUSD", -5.0, pattern="orderflow")
        summaries = engine.get_all_pattern_thompson_summaries()
        assert "fvg" in summaries
        assert "orderflow" in summaries
        assert len(summaries) == 2

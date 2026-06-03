"""MEJORA 14 — Cooldown y Recuperación Adaptativo (Modo Experto)
Reemplaza el cooldown fijo de 5 velas M1 + hot pause 30 min por un sistema
dinámico que:

  - Calcula cooldown óptimo según ATR ratio y volatilidad del régimen
  - Tras racha perdedora: reduce tamaño progresivamente y exige mayor convicción
  - Usa Thompson Sampling (Beta-Bernoulli) para reincorporar patrones desactivados
  - Implementa "plan de recuperación" con objetivos parciales por stages
  - Ajusta dinámicamente los thresholds de convicción según el estado de la cuenta
"""
import logging
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────

class RecoveryStage(Enum):
    NORMAL = "NORMAL"
    ALERT = "ALERT"
    REDUCE = "REDUCE"
    RECOVER = "RECOVER"
    EMERGENCY = "EMERGENCY"


class CooldownReason(Enum):
    NONE = ""
    CONSECUTIVE_LOSSES = "consecutive_losses"
    HIGH_VOLATILITY = "high_volatility"
    RECOVERY_MODE = "recovery_mode"
    PATTERN_DISABLED = "pattern_disabled"
    PATIENT_MODE = "patient_mode"


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

@dataclass
class AdaptiveCooldownConfig:
    expert_mode: bool = True

    # Cooldown base parameters
    base_cooldown_minutes: float = 5.0
    min_cooldown_minutes: float = 1.0
    max_cooldown_minutes: float = 60.0
    cooldown_atr_ratio_weight: float = 2.0
    cooldown_volatility_weight: float = 1.5
    cooldown_consecutive_loss_weight: float = 5.0
    cooldown_decay_minutes: float = 120.0

    # Consecutive loss escalation
    loss_volume_reductions: List[float] = field(
        default_factory=lambda: [1.0, 0.7, 0.5, 0.3, 0.15, 0.0]
    )
    loss_conviction_increases: List[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.10, 0.20, 0.35, 0.50]
    )
    max_consecutive_losses_before_stop: int = 5

    # Hot pause
    hot_pause_base_minutes: float = 15.0
    hot_pause_escalation_factor: float = 2.0
    hot_pause_max_minutes: float = 240.0
    hot_pause_reset_after_win: bool = True

    # Conviction thresholds
    base_min_conviction: float = 0.15
    max_min_conviction: float = 0.60
    conviction_streak_increment: float = 0.05
    conviction_streak_max_losses: int = 3

    # Recovery plan
    recovery_stage_duration_trades: Dict[str, int] = field(
        default_factory=lambda: {
            "EMERGENCY": 5,
            "RECOVER": 8,
            "REDUCE": 10,
            "ALERT": 6,
        }
    )
    recovery_stage_volume_mults: Dict[str, float] = field(
        default_factory=lambda: {
            "EMERGENCY": 0.0,
            "RECOVER": 0.3,
            "REDUCE": 0.5,
            "ALERT": 0.75,
            "NORMAL": 1.0,
        }
    )
    recovery_stage_conviction_bonuses: Dict[str, float] = field(
        default_factory=lambda: {
            "EMERGENCY": 0.40,
            "RECOVER": 0.25,
            "REDUCE": 0.15,
            "ALERT": 0.05,
            "NORMAL": 0.0,
        }
    )
    recovery_target_win_rate: float = 0.40
    recovery_trades_to_advance: int = 5
    recovery_min_advance_win_rate: float = 0.50

    # Thompson Sampling
    thompson_alpha_prior: float = 2.0
    thompson_beta_prior: float = 2.0
    thompson_min_samples: int = 3
    thompson_re_enable_threshold: float = 0.15
    thompson_max_disable_minutes: float = 1440.0

    # ATR-based adjustment
    atr_ratio_low: float = 0.7
    atr_ratio_high: float = 1.5
    atr_ratio_cooldown_mult: float = 1.5
    atr_ratio_conviction_bonus: float = 0.10


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class CooldownDecision:
    active: bool
    cooldown_minutes: float
    remaining_minutes: float
    reason: CooldownReason
    recommended_volume_mult: float
    recommended_min_conviction: float
    recovery_stage: RecoveryStage
    notes: List[str] = field(default_factory=list)


@dataclass
class ThompsonState:
    alpha: float
    beta: float
    total_samples: int
    last_sample_time: float
    disabled_until: float
    disable_count: int
    win_rate: float
    sampled_probability: float


@dataclass
class RecoveryState:
    stage: RecoveryStage
    trades_in_stage: int = 0
    wins_in_stage: int = 0
    losses_in_stage: int = 0
    entered_at: float = 0.0
    last_update: float = 0.0


@dataclass
class StreakState:
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0
    total_trades: int = 0
    total_wins: int = 0
    total_losses: int = 0
    last_trade_time: float = 0.0
    last_trade_result: str = ""
    hot_pause_until: float = 0.0
    total_volume_lost: float = 0.0


# ─────────────────────────────────────────────
# THOMPSON SAMPLER
# ─────────────────────────────────────────────

class ThompsonSampler:
    """Beta-Bernoulli Thompson Sampling for pattern re-enablement.
    
    Each pattern has a Beta(alpha, beta) posterior.
    To decide re-enablement: sample from posterior, if probability > threshold, re-enable.
    """

    def __init__(self, config: AdaptiveCooldownConfig):
        self.cfg = config
        self._states: Dict[str, ThompsonState] = {}

    def get_state(self, pattern: str) -> ThompsonState:
        if pattern not in self._states:
            self._states[pattern] = ThompsonState(
                alpha=self.cfg.thompson_alpha_prior,
                beta=self.cfg.thompson_beta_prior,
                total_samples=0,
                last_sample_time=0.0,
                disabled_until=0.0,
                disable_count=0,
                win_rate=0.5,
                sampled_probability=0.5,
            )
        return self._states[pattern]

    def record_outcome(self, pattern: str, won: bool):
        state = self.get_state(pattern)
        if won:
            state.alpha += 1.0
        else:
            state.beta += 1.0
        state.total_samples += 1
        state.last_sample_time = time.time()
        state.win_rate = state.alpha / (state.alpha + state.beta) if (state.alpha + state.beta) > 0 else 0.5

    def sample(self, pattern: str) -> float:
        """Sample from the posterior Beta distribution.
        Returns the sampled probability of winning.
        """
        state = self.get_state(pattern)
        if state.total_samples < self.cfg.thompson_min_samples:
            return 0.5
        try:
            sample = float(np.random.beta(state.alpha, state.beta))
            state.sampled_probability = sample
            return sample
        except Exception:
            return state.win_rate

    def should_re_enable(self, pattern: str) -> Tuple[bool, float]:
        """Use Thompson Sampling to decide if a disabled pattern should be re-enabled.
        Returns (should_re_enable, sampled_probability).
        """
        state = self.get_state(pattern)
        if time.time() < state.disabled_until:
            return False, 0.0
        if state.total_samples < self.cfg.thompson_min_samples:
            return False, 0.0
        sampled_p = self.sample(pattern)
        should = sampled_p > self.cfg.thompson_re_enable_threshold
        return should, sampled_p

    def disable(self, pattern: str, cooldown_minutes: float):
        state = self.get_state(pattern)
        state.disabled_until = time.time() + cooldown_minutes * 60
        state.disable_count += 1

    def is_disabled(self, pattern: str) -> bool:
        state = self.get_state(pattern)
        return time.time() < state.disabled_until

    def get_remaining_cooldown(self, pattern: str) -> float:
        state = self.get_state(pattern)
        remaining = state.disabled_until - time.time()
        return max(0.0, remaining / 60.0)

    def get_summary(self, pattern: str) -> dict:
        state = self.get_state(pattern)
        return {
            "alpha": round(state.alpha, 2),
            "beta": round(state.beta, 2),
            "samples": state.total_samples,
            "win_rate": round(state.win_rate, 4),
            "sampled_p": round(state.sampled_probability, 4),
            "disabled": self.is_disabled(pattern),
            "remaining_min": round(self.get_remaining_cooldown(pattern), 1),
            "disable_count": state.disable_count,
        }


# ─────────────────────────────────────────────
# COOLDOWN CALCULATOR
# ─────────────────────────────────────────────

class CooldownCalculator:
    """Calcula cooldown óptimo basado en ATR, volatilidad y racha."""

    def __init__(self, config: AdaptiveCooldownConfig):
        self.cfg = config

    def calculate(
        self,
        atr_ratio: float,
        volatility_regime: str,
        consecutive_losses: int,
        recovery_stage: RecoveryStage,
    ) -> float:
        base = self.cfg.base_cooldown_minutes

        atr_mult = 1.0
        if atr_ratio > self.cfg.atr_ratio_high:
            atr_mult = 1.0 + (atr_ratio - self.cfg.atr_ratio_high) * self.cfg.cooldown_atr_ratio_weight
        elif atr_ratio < self.cfg.atr_ratio_low:
            atr_mult = max(0.5, atr_ratio / self.cfg.atr_ratio_low)

        vol_mult = 1.0
        if volatility_regime == "HIGH":
            vol_mult = self.cfg.atr_ratio_cooldown_mult
        elif volatility_regime == "LOW":
            vol_mult = 0.7

        loss_mult = 1.0
        if consecutive_losses >= 1:
            loss_mult = 1.0 + consecutive_losses * self.cfg.cooldown_consecutive_loss_weight

        recovery_mult = 1.0
        recovery_mults = {
            RecoveryStage.EMERGENCY: 8.0,
            RecoveryStage.RECOVER: 4.0,
            RecoveryStage.REDUCE: 2.5,
            RecoveryStage.ALERT: 1.5,
            RecoveryStage.NORMAL: 1.0,
        }
        recovery_mult = recovery_mults.get(recovery_stage, 1.0)

        cooldown = base * atr_mult * vol_mult * loss_mult * recovery_mult
        cooldown = max(self.cfg.min_cooldown_minutes, min(self.cfg.max_cooldown_minutes, cooldown))
        return round(cooldown, 1)

    def get_conviction_requirement(
        self,
        consecutive_losses: int,
        recovery_stage: RecoveryStage,
        atr_ratio: float,
    ) -> float:
        base = self.cfg.base_min_conviction

        if consecutive_losses >= self.cfg.conviction_streak_max_losses:
            idx = min(consecutive_losses - self.cfg.conviction_streak_max_losses, len(self.cfg.loss_conviction_increases) - 1)
            base += self.cfg.loss_conviction_increases[idx]

        recovery_bonus = self.cfg.recovery_stage_conviction_bonuses.get(recovery_stage.value, 0.0)
        base += recovery_bonus

        if atr_ratio > self.cfg.atr_ratio_high:
            base += self.cfg.atr_ratio_conviction_bonus

        return min(self.cfg.max_min_conviction, round(base, 3))

    def get_volume_multiplier(
        self,
        consecutive_losses: int,
        recovery_stage: RecoveryStage,
    ) -> float:
        if consecutive_losses >= 1:
            idx = min(consecutive_losses - 1, len(self.cfg.loss_volume_reductions) - 1)
            loss_mult = self.cfg.loss_volume_reductions[idx]
        else:
            loss_mult = 1.0

        recovery_mult = self.cfg.recovery_stage_volume_mults.get(recovery_stage.value, 1.0)

        return round(min(loss_mult, recovery_mult), 3)

    def should_hot_pause(self, consecutive_losses: int, recovery_stage: RecoveryStage) -> Tuple[bool, float]:
        if recovery_stage == RecoveryStage.EMERGENCY:
            return True, self.cfg.hot_pause_max_minutes
        if consecutive_losses >= self.cfg.max_consecutive_losses_before_stop:
            pause = self.cfg.hot_pause_base_minutes * (
                self.cfg.hot_pause_escalation_factor ** (consecutive_losses - self.cfg.max_consecutive_losses_before_stop)
            )
            pause = min(pause, self.cfg.hot_pause_max_minutes)
            return True, round(pause, 1)
        return False, 0.0


# ─────────────────────────────────────────────
# RECOVERY PLAN MANAGER
# ─────────────────────────────────────────────

class RecoveryPlanManager:
    """Gestiona el plan de recuperación progresiva tras rachas malas."""

    def __init__(self, config: AdaptiveCooldownConfig):
        self.cfg = config
        self._states: Dict[str, RecoveryState] = {}
        self._global_stage: RecoveryStage = RecoveryStage.NORMAL

    def get_state(self, symbol: str = "") -> RecoveryState:
        key = symbol or "__global__"
        if key not in self._states:
            self._states[key] = RecoveryState(
                stage=RecoveryStage.NORMAL,
                entered_at=time.time(),
                last_update=time.time(),
            )
        return self._states[key]

    def record_trade(self, symbol: str, profit: float, consecutive_losses: int):
        won = profit >= 0
        state = self.get_state(symbol)
        global_state = self.get_state("")

        for s in (state, global_state):
            s.trades_in_stage += 1
            if won:
                s.wins_in_stage += 1
            else:
                s.losses_in_stage += 1
            s.last_update = time.time()

            self._evaluate_transition(s, consecutive_losses)

    def _evaluate_transition(self, state: RecoveryState, consecutive_losses: int):
        current = state.stage
        win_rate_in_stage = state.wins_in_stage / max(state.trades_in_stage, 1)
        min_trades = self.cfg.recovery_stage_duration_trades.get(current.value, 5)

        if current == RecoveryStage.NORMAL:
            if consecutive_losses >= 2:
                state.stage = RecoveryStage.ALERT
                state.trades_in_stage = 0
                state.wins_in_stage = 0
                state.losses_in_stage = 0
                state.entered_at = time.time()
                logger.info(f"[RECOVERY] NORMAL → ALERT: {consecutive_losses} pérdidas consecutivas")

        elif current == RecoveryStage.ALERT:
            if consecutive_losses >= 4:
                state.stage = RecoveryStage.REDUCE
                state.trades_in_stage = 0
                state.wins_in_stage = 0
                state.losses_in_stage = 0
                state.entered_at = time.time()
                logger.info(f"[RECOVERY] ALERT → REDUCE: escalada a {consecutive_losses} pérdidas")
            elif state.trades_in_stage >= min_trades and win_rate_in_stage >= self.cfg.recovery_target_win_rate:
                state.stage = RecoveryStage.NORMAL
                state.trades_in_stage = 0
                state.wins_in_stage = 0
                state.losses_in_stage = 0
                logger.info(f"[RECOVERY] ALERT → NORMAL: WR={win_rate_in_stage:.0%} >= {self.cfg.recovery_target_win_rate:.0%}")

        elif current == RecoveryStage.REDUCE:
            if consecutive_losses >= self.cfg.max_consecutive_losses_before_stop:
                state.stage = RecoveryStage.EMERGENCY
                state.trades_in_stage = 0
                state.wins_in_stage = 0
                state.losses_in_stage = 0
                state.entered_at = time.time()
                logger.warning(f"[RECOVERY] REDUCE → EMERGENCY: {consecutive_losses} pérdidas, parando")
            elif state.trades_in_stage >= min_trades and win_rate_in_stage >= self.cfg.recovery_min_advance_win_rate:
                state.stage = RecoveryStage.ALERT
                state.trades_in_stage = 0
                state.wins_in_stage = 0
                state.losses_in_stage = 0
                logger.info(f"[RECOVERY] REDUCE → ALERT: WR={win_rate_in_stage:.0%} >= {self.cfg.recovery_min_advance_win_rate:.0%}")

        elif current == RecoveryStage.EMERGENCY:
            if state.trades_in_stage >= min_trades and win_rate_in_stage >= self.cfg.recovery_min_advance_win_rate:
                state.stage = RecoveryStage.RECOVER
                state.trades_in_stage = 0
                state.wins_in_stage = 0
                state.losses_in_stage = 0
                logger.info(f"[RECOVERY] EMERGENCY → RECOVER: WR={win_rate_in_stage:.0%} >= {self.cfg.recovery_min_advance_win_rate:.0%}")

        elif current == RecoveryStage.RECOVER:
            if state.trades_in_stage >= min_trades and win_rate_in_stage >= self.cfg.recovery_target_win_rate:
                state.stage = RecoveryStage.REDUCE
                state.trades_in_stage = 0
                state.wins_in_stage = 0
                state.losses_in_stage = 0
                logger.info(f"[RECOVERY] RECOVER → REDUCE: WR={win_rate_in_stage:.0%} >= {self.cfg.recovery_target_win_rate:.0%}")

    def get_stage(self, symbol: str = "") -> RecoveryStage:
        sym_stage = self.get_state(symbol).stage
        global_stage = self.get_state("").stage
        severity = {RecoveryStage.NORMAL: 0, RecoveryStage.ALERT: 1, RecoveryStage.REDUCE: 2, RecoveryStage.RECOVER: 3, RecoveryStage.EMERGENCY: 4}
        if severity.get(sym_stage, 0) > severity.get(global_stage, 0):
            return sym_stage
        return global_stage

    def get_summary(self, symbol: str = "") -> dict:
        state = self.get_state(symbol)
        return {
            "stage": state.stage.value,
            "trades_in_stage": state.trades_in_stage,
            "wins_in_stage": state.wins_in_stage,
            "losses_in_stage": state.losses_in_stage,
            "win_rate_in_stage": round(state.wins_in_stage / max(state.trades_in_stage, 1), 3),
            "duration_minutes": round((time.time() - state.entered_at) / 60, 1),
        }


# ─────────────────────────────────────────────
# ADAPTIVE COOLDOWN ENGINE (Main)
# ─────────────────────────────────────────────

class AdaptiveCooldownEngine:
    """Motor principal de cooldown y recuperación adaptativo.
    
    Integra:
      - CooldownCalculator: cooldown dinámico según ATR/volatilidad/racha
      - ThompsonSampler: reincorporación probabilística de patrones
      - RecoveryPlanManager: plan de recuperación por stages
      - StreakState: estado de rachas por símbolo
    """

    def __init__(self, config: Optional[AdaptiveCooldownConfig] = None):
        self.cfg = config or AdaptiveCooldownConfig()
        self.calculator = CooldownCalculator(self.cfg)
        self.thompson = ThompsonSampler(self.cfg)
        self.recovery = RecoveryPlanManager(self.cfg)
        self._streaks: Dict[str, StreakState] = defaultdict(StreakState)
        self._last_evaluation: Dict[str, float] = {}

    # ── Streak Management ──

    def get_streak(self, symbol: str) -> StreakState:
        return self._streaks[symbol]

    def record_trade(self, symbol: str, profit: float, pattern: Optional[str] = None):
        won = profit >= 0
        streak = self._streaks[symbol]
        streak.total_trades += 1
        streak.last_trade_time = time.time()
        streak.last_trade_result = "win" if won else "loss"

        if won:
            streak.consecutive_losses = 0
            streak.total_wins += 1
            if self.cfg.hot_pause_reset_after_win:
                streak.hot_pause_until = 0.0
        else:
            streak.consecutive_losses += 1
            streak.total_losses += 1
            streak.total_volume_lost += abs(profit)
            if streak.consecutive_losses > streak.max_consecutive_losses:
                streak.max_consecutive_losses = streak.consecutive_losses

            should_pause, pause_min = self.calculator.should_hot_pause(
                streak.consecutive_losses,
                self.recovery.get_stage(symbol),
            )
            if should_pause:
                streak.hot_pause_until = time.time() + pause_min * 60
                logger.warning(
                    f"[COOLDOWN][{symbol}] HOT PAUSE {pause_min:.0f}min "
                    f"(stage={self.recovery.get_stage(symbol).value}, "
                    f"losses={streak.consecutive_losses})"
                )

        if pattern:
            self.thompson.record_outcome(pattern, won)

        self.recovery.record_trade(symbol, profit, streak.consecutive_losses)

    def is_hot_paused(self, symbol: str) -> Tuple[bool, float]:
        streak = self._streaks[symbol]
        remaining = streak.hot_pause_until - time.time()
        if remaining > 0:
            return True, remaining
        return False, 0.0

    # ── Cooldown Evaluation ──

    def evaluate(
        self,
        symbol: str,
        pattern: Optional[str] = None,
        atr_ratio: float = 1.0,
        volatility_regime: str = "MEDIUM",
        last_trade_time: Optional[float] = None,
    ) -> CooldownDecision:
        streak = self._streaks[symbol]
        stage = self.recovery.get_stage(symbol)
        notes = []

        hot_paused, hot_remaining = self.is_hot_paused(symbol)
        if hot_paused:
            return CooldownDecision(
                active=True,
                cooldown_minutes=hot_remaining / 60,
                remaining_minutes=hot_remaining / 60,
                reason=CooldownReason.CONSECUTIVE_LOSSES,
                recommended_volume_mult=0.0,
                recommended_min_conviction=self.cfg.max_min_conviction,
                recovery_stage=stage,
                notes=["Hot pause activo por racha perdedora"],
            )

        if pattern and self.thompson.is_disabled(pattern):
            remaining = self.thompson.get_remaining_cooldown(pattern)
            return CooldownDecision(
                active=True,
                cooldown_minutes=remaining,
                remaining_minutes=remaining,
                reason=CooldownReason.PATTERN_DISABLED,
                recommended_volume_mult=0.0,
                recommended_min_conviction=self.cfg.max_min_conviction,
                recovery_stage=stage,
                notes=[f"Patrón '{pattern}' desactivado temporalmente"],
            )

        elapsed_since_last = 0.0
        if last_trade_time:
            elapsed_since_last = (time.time() - last_trade_time) / 60.0

        opt_cooldown = self.calculator.calculate(
            atr_ratio, volatility_regime, streak.consecutive_losses, stage,
        )

        if elapsed_since_last < opt_cooldown:
            remaining = opt_cooldown - elapsed_since_last
            reason = CooldownReason.HIGH_VOLATILITY if atr_ratio > self.cfg.atr_ratio_high else CooldownReason.RECOVERY_MODE
            notes.append(f"Cooldown óptimo {opt_cooldown:.0f}min ({elapsed_since_last:.0f}min transcurridos)")

            return CooldownDecision(
                active=True,
                cooldown_minutes=opt_cooldown,
                remaining_minutes=round(remaining, 1),
                reason=reason,
                recommended_volume_mult=self.calculator.get_volume_multiplier(
                    streak.consecutive_losses, stage,
                ),
                recommended_min_conviction=self.calculator.get_conviction_requirement(
                    streak.consecutive_losses, stage, atr_ratio,
                ),
                recovery_stage=stage,
                notes=notes,
            )

        vol_mult = self.calculator.get_volume_multiplier(streak.consecutive_losses, stage)
        min_conv = self.calculator.get_conviction_requirement(streak.consecutive_losses, stage, atr_ratio)

        if stage != RecoveryStage.NORMAL:
            notes.append(f"Recovery stage: {stage.value} (vol×{vol_mult}, conv≥{min_conv:.0%})")
        if streak.consecutive_losses >= 2:
            notes.append(f"Racha: {streak.consecutive_losses} pérdidas consecutivas")

        return CooldownDecision(
            active=False,
            cooldown_minutes=opt_cooldown,
            remaining_minutes=0.0,
            reason=CooldownReason.NONE,
            recommended_volume_mult=vol_mult,
            recommended_min_conviction=min_conv,
            recovery_stage=stage,
            notes=notes,
        )

    # ── Thompson Sampling Integration ──

    def evaluate_pattern_re_enable(self, pattern: str) -> Tuple[bool, float]:
        return self.thompson.should_re_enable(pattern)

    def disable_pattern(self, pattern: str, cooldown_minutes: Optional[float] = None):
        if cooldown_minutes is None:
            cooldown_minutes = self.cfg.thompson_max_disable_minutes
        self.thompson.disable(pattern, cooldown_minutes)

    def is_pattern_disabled(self, pattern: str) -> bool:
        return self.thompson.is_disabled(pattern)

    def get_pattern_thompson_summary(self, pattern: str) -> dict:
        return self.thompson.get_summary(pattern)

    def get_all_pattern_thompson_summaries(self) -> Dict[str, dict]:
        summaries = {}
        for pattern, _ in self.thompson._states.items():
            summaries[pattern] = self.thompson.get_summary(pattern)
        return summaries

    # ── Recovery Plan Access ──

    def get_recovery_summary(self, symbol: str = "") -> dict:
        return self.recovery.get_summary(symbol)

    def get_recovery_stage(self, symbol: str = "") -> RecoveryStage:
        return self.recovery.get_stage(symbol)

    # ── General Summary ──

    def get_full_summary(self, symbol: str) -> dict:
        streak = self._streaks[symbol]
        stage = self.recovery.get_stage(symbol)
        return {
            "symbol": symbol,
            "recovery_stage": stage.value,
            "consecutive_losses": streak.consecutive_losses,
            "max_consecutive_losses": streak.max_consecutive_losses,
            "total_trades": streak.total_trades,
            "total_wins": streak.total_wins,
            "total_losses": streak.total_losses,
            "hot_paused": self.is_hot_paused(symbol)[0],
            "recovery": self.recovery.get_summary(symbol),
            "cooldown_volume_mult": self.calculator.get_volume_multiplier(streak.consecutive_losses, stage),
            "cooldown_min_conviction": self.calculator.get_conviction_requirement(streak.consecutive_losses, stage, 1.0),
        }

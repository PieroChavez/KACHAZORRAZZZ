"""SMC Adaptive Contextual Strategy Engine
Fuses Smart Money Concepts via a regime-aware confluence scoring system.
Pattern weights adapt dynamically to market regime, session, and conviction.
Decisions are continuous (not binary) based on distributional score analysis.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from datetime import datetime

import pandas as pd

from src.core.market_analyzer import (
    MarketAnalyzer, MarketContext, TrendDirection, MarketRegime,
)
from src.core.pattern_detector import PatternDetector, Pattern, PatternType
from src.core.vsa import VSADetector
from src.core.regime_detector import RegimeContext, RegimeType, REGIME_PATTERN_MULTIPLIERS
from src.core.adaptive_scoring import AdaptiveScorer, AdaptiveWeights
from src.core.distributional_score import DistributionalScorer, DistributionalScore, merge_signals
from src.core.session_profiler import SessionProfiler, SessionProfile, TradingSession
from src.core.bayesian_learner import BayesianWeightLearner
from src.utils.helpers import is_in_session, pip_size, atr, detect_killzone, Killzone

TIMEFRAME_GROUPS = {
    "HTF": ["4H", "3H", "2H", "1H"],
    "MID": ["30min", "15min", "10min"],
    "LTF": ["5min", "3min", "1min"],
}

logger = logging.getLogger(__name__)

# ── Imports resilientes de anticipación (con fallback) ────────────────
_ANTICIPATION_IMPORT_OK = True
try:
    from src.core.anticipation import (
        analyze_all, DivergenceDetector, OrderFlowAnalyzer,
        VolumeProfile, DXYCorrelationAnalyzer,
    )
    logger.debug("anticipation.py: módulo cargado correctamente")
except ImportError as e:
    _ANTICIPATION_IMPORT_OK = False
    logger.warning(f"anticipation.py no disponible ({e}): análisis anticipatorio desactivado")

    class _StubDivergenceResult:
        def __init__(self):
            self.strength = 0.0
            self.source = "stub"
            self.type = "none"
    class _StubDivergenceDetector:
        @staticmethod
        def detect_all(df):
            logger.debug("DivergenceDetector: stub (no disponible)")
            return []
        @staticmethod
        def best_divergence(divergences, direction):
            return None, None
    DivergenceDetector = _StubDivergenceDetector

    class _StubOFResult:
        buying_pressure = 0.0
        selling_pressure = 0.0
        delta_imbalance = 0.0
        divergence_active = False
        absorption_active = False
        exhaustion_active = False
        notes = ["stub (no disponible)"]
    class _StubOF:
        @staticmethod
        def analyze(df, lookback=30):
            logger.debug("OrderFlowAnalyzer: stub (no disponible)")
            return _StubOFResult()
    OrderFlowAnalyzer = _StubOF

    class _StubVPResult:
        poc = 0.0
        value_area_high = 0.0
        value_area_low = 0.0
        high_volume_nodes = []
        low_volume_nodes = []
        value_area_width_pips = 0.0
        is_compressed = False
        poc_shift_pips = 0.0
        notes = ["stub (no disponible)"]
    class _StubVP:
        @staticmethod
        def compute(df, num_bins=24, value_area_pct=0.70):
            logger.debug("VolumeProfile: stub (no disponible)")
            return _StubVPResult()
    VolumeProfile = _StubVP

    class _StubDXYResult:
        trend = "neutral"
        strength = 0.0
        correlation = 0.0
        divergence_active = False
        divergence_strength = 0.0
        regime = "normal"
        notes = ["stub (no disponible)"]
    class _StubDXY:
        @staticmethod
        def analyze(dxy_df, xau_df):
            logger.debug("DXYCorrelationAnalyzer: stub (no disponible)")
            return _StubDXYResult()
    DXYCorrelationAnalyzer = _StubDXY
    analyze_all = None


@dataclass
class ScoringConfig:
    htf_trend_aligned: float = 20.0
    in_discount_premium_zone: float = 15.0
    valid_market_structure: float = 10.0
    fvg_detected: float = 18.0
    order_block_valid: float = 15.0
    breaker_retest: float = 12.0
    slip_memory_present: float = 10.0
    equidad_sweep_confirmed: float = 12.0
    eslabon_breakout: float = 15.0
    wyckoff_phase_c_spring: float = 20.0
    wyckoff_phase_c_utad: float = 20.0
    wyckoff_phase_d_confirmed: float = 15.0
    ltf_bos_with_body: float = 12.0
    liquidity_sweep_ltf: float = 10.0
    in_active_session: float = 8.0
    killzone_london_open: float = 12.0
    killzone_ny_open: float = 15.0
    killzone_london_ny_overlap: float = 18.0
    killzone_asian: float = 5.0
    no_news_event: float = 5.0
    regime_expansion: float = 8.0
    regime_not_accumulation: float = 5.0
    ltf_sweep_confirmation: float = 15.0
    multiframe_alignment: float = 12.0
    void_scalp_confirmed: float = 18.0
    cycle_full: float = 15.0

    triple_confluence: float = 18.0
    wyckoff_sos_volume: float = 15.0
    wyckoff_phase_d_lps: float = 12.0
    fvg_fresh_mitigation: float = 8.0
    sweep_spring_wick: float = 10.0
    body_close_valid: float = 10.0
    price_grid_aligned: float = 8.0
    price_establishment: float = 10.0
    news_high_impact_penalty: float = -200.0

    wick_rejection_bonus: float = 5.0
    trb_manipulation_detected: float = 12.0
    trb_displacement: float = 10.0
    trb_retest: float = 8.0

    body_close_invalid: float = -50.0
    no_sweep_detected: float = -30.0
    fvg_burned_over_50: float = -20.0
    spring_body_close: float = -40.0
    bos_zone_retest: float = 20.0
    price_establishment_bonus: float = 12.0
    third_movement_ready: float = 15.0
    micro_retracement_50: float = 10.0
    psychological_price_aligned: float = 8.0

    ob_mitigated_penalty: float = -15.0
    breaker_mitigated_penalty: float = -12.0
    lps_mitigated_penalty: float = -12.0

    vsa_volume_confirmation: float = 12.0
    vsa_climax_penalty: float = -20.0
    vsa_absorption_bonus: float = 15.0
    vsa_low_volume_pullback: float = 10.0
    vsa_volume_divergence: float = -15.0
    vsa_no_demand_supply: float = -10.0

    dxy_aligned_bonus: float = 12.0
    dxy_conflict_penalty: float = -15.0

    interval_point_bonus: float = 12.0
    price_interaction_bonus: float = 14.0
    harmonic_cycle_aligned: float = 15.0
    pause_continuation_bonus: float = 10.0
    retracement_penalty: float = -15.0
    order_flow_regular_bonus: float = 8.0
    order_flow_irregular_penalty: float = -8.0
    pressure_zone_bonus: float = 14.0
    breaker_3_touch_limit: float = -25.0
    price_trend_aligned: float = 15.0
    htf_misalignment_penalty: float = -5.0

    divergence_regular_bonus: float = 18.0
    divergence_hidden_bonus: float = 10.0
    divergence_penalty: float = -12.0
    order_flow_imbalance_bonus: float = 12.0
    order_flow_divergence_bonus: float = 15.0
    order_flow_exhaustion_penalty: float = -10.0
    order_flow_absorption_bonus: float = 8.0
    volume_profile_poc_discount_bonus: float = 12.0
    volume_profile_lvn_bonus: float = 10.0
    volume_profile_compression_bonus: float = 8.0
    volume_profile_poc_shift_bonus: float = 10.0
    volume_profile_poc_shift_penalty: float = -10.0
    dxy_enhanced_breakdown_penalty: float = -15.0
    dxy_divergence_bonus: float = 14.0


@dataclass
class TimeframeVote:
    tf_name: str
    direction: str
    weight: float
    confidence: float
    reasons: List[str] = field(default_factory=list)


@dataclass
class TimeframeConsensus:
    overall_direction: str
    buy_weight: float
    sell_weight: float
    total_weight: float
    confidence: float
    tf_votes: List[TimeframeVote] = field(default_factory=list)

    def alignment_with(self, direction: str) -> float:
        if self.total_weight <= 0:
            return 0.5
        if direction == "BUY":
            return self.buy_weight / self.total_weight
        return self.sell_weight / self.total_weight


@dataclass
class TradingSignal:
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    score: float
    score_breakdown: dict
    primary_pattern: Optional[Pattern]
    supporting_patterns: List[Pattern]
    rr_ratio: float
    symbol: str
    timestamp: datetime
    notes: List[str] = field(default_factory=list)

    conviction: float = 0.0
    distribution: Optional[DistributionalScore] = None
    regime_context: Optional[RegimeContext] = None
    session_profile: Optional[SessionProfile] = None
    adaptive_weights: Optional[AdaptiveWeights] = None
    consensus: Optional[TimeframeConsensus] = None


class StrategyEngine:
    def __init__(self, profile, params, weights: ScoringConfig = None,
                 min_score: float = 70.0, high_confidence_score: float = 85.0,
                 min_net_score: float = 2.0,
                 tf_groups: Optional[Dict[str, List[str]]] = None):
        self.profile = profile
        self.params = params
        self.weights = weights or ScoringConfig()
        self.tf_groups = tf_groups or TIMEFRAME_GROUPS
        self.min_score = min_score
        self.high_confidence_score = high_confidence_score
        self.min_net_score = min_net_score
        self.min_conviction_to_trade = params.get("adaptive", {}).get("min_conviction_to_trade", 0.12) if isinstance(params, dict) else 0.12
        self._session_min_score: Optional[float] = None
        self._session_min_conviction: Optional[float] = None
        self._session_weight_mult: Dict[str, float] = {}
        self.analyzer = MarketAnalyzer(params)
        self.detector = PatternDetector(params, profile.symbol)
        self.pip = pip_size(profile.symbol)
        self.adaptive_scorer = AdaptiveScorer(self.weights)
        self.distributional_scorer = DistributionalScorer()
        self.session_profiler = SessionProfiler()
        self.bayesian_learner = BayesianWeightLearner()

    def _get_effective_weight(self, weight_name: str, breakdown_key: str = None) -> float:
        base = getattr(self.weights, weight_name, 0.0)
        feature_group = None
        for key, group in [("fvg_detected", "FVG"), ("order_block_valid", "OB"),
                           ("breaker_retest", "BREAKER"), ("liquidity_sweep_ltf", "SWEEP"),
                           ("cycle_full", "CYCLE"), ("void_scalp_confirmed", "VOID_SCALP"),
                           ("bos_zone_retest", "BOS_ZONE"), ("triple_confluence", "TRIPLE_CONF"),
                           ("ltf_sweep_confirmation", "SWEEP_CONF"),
                           ("multiframe_alignment", "MF_ALIGN"),
                           ("price_interaction_bonus", "PRICE_INTERACTION"),
                           ("interval_point_bonus", "INTERVAL_POINT"),
                           ("harmonic_cycle_aligned", "HARMONIC_CYCLE"),
                           ("pressure_zone_bonus", "PRESSURE_ZONE")]:
            if weight_name == key:
                feature_group = group
                break
        weight = base
        if feature_group:
            mult = self.bayesian_learner.get_feature_multiplier(feature_group, base)
            if mult != 1.0:
                weight = base * mult
        session_mult = self._session_weight_mult.get(feature_group) if feature_group else None
        if session_mult is not None and session_mult != 1.0:
            weight = weight * session_mult
        return weight

    def set_session_params(self, min_score: Optional[float] = None,
                           min_conviction: Optional[float] = None,
                           weight_multipliers: Optional[Dict[str, float]] = None):
        """Aplica parámetros dinámicos de sesión para el próximo ciclo.
        Si se llama sin argumentos, resetea todos los parámetros de sesión."""
        if min_score is not None:
            self._session_min_score = min_score
        if min_conviction is not None:
            self._session_min_conviction = min_conviction
        if weight_multipliers is not None:
            self._session_weight_mult.clear()
            for k, v in weight_multipliers.items():
                self._session_weight_mult[k] = v
        logger.debug(f"Session params: min_score={self._session_min_score}, "
                     f"min_conv={self._session_min_conviction}, "
                     f"weight_mults={dict(self._session_weight_mult)}")

    def clear_session_params(self):
        """Resetea todos los parámetros de sesión a valores por defecto."""
        self._session_min_score = None
        self._session_min_conviction = None
        self._session_weight_mult.clear()

    def record_breakdown_outcome(self, signal_id: str, symbol: str, direction: str,
                                  profit: float, breakdown: dict, score_net: float):
        self.bayesian_learner.record_outcome(signal_id, symbol, direction, profit, breakdown, score_net)

    def _pick_tf(self, timeframes: dict, group: str) -> pd.DataFrame:
        for tf in self.tf_groups.get(group, []):
            df = timeframes.get(tf)
            if df is not None and len(df) > 20:
                return df
        return None

    def _all_timeframes_sorted(self, timeframes: dict) -> List[pd.DataFrame]:
        all_names = []
        for group in self.tf_groups.values():
            all_names.extend(group)
        result = []
        for tf in all_names:
            df = timeframes.get(tf)
            if df is not None and len(df) > 20:
                result.append(df)
        return result

    def _all_timeframe_names(self, timeframes: dict) -> List[str]:
        all_names = []
        for group in self.tf_groups.values():
            all_names.extend(group)
        return [tf for tf in all_names if tf in timeframes and timeframes[tf] is not None and len(timeframes[tf]) > 20]

    def _compute_tf_vote(self, df: pd.DataFrame, tf_name: str) -> TimeframeVote:
        from src.utils.helpers import find_swing_points
        highs, lows = find_swing_points(df, lookback=5)
        if len(highs) < 3 or len(lows) < 3:
            return TimeframeVote(tf_name, "HOLD", 0.0, 0.0, ["swing_insufficient"])
        last_high = df["high"].iloc[highs[-1]]
        prev_high = df["high"].iloc[highs[-2]]
        last_low = df["low"].iloc[lows[-1]]
        prev_low = df["low"].iloc[lows[-2]]
        if last_high > prev_high and last_low > prev_low:
            direction = "BUY"
        elif last_high < prev_high and last_low < prev_low:
            direction = "SELL"
        else:
            direction = "HOLD"
        consistency = 0
        for i in range(1, min(4, len(highs))):
            hh = df["high"].iloc[highs[-i]] > df["high"].iloc[highs[-i-1]]
            ll = df["low"].iloc[lows[-i]] > df["low"].iloc[lows[-i-1]]
            if direction == "BUY" and hh and ll:
                consistency += 1
            elif direction == "SELL":
                lh = df["high"].iloc[highs[-i]] < df["high"].iloc[highs[-i-1]]
                hl = df["low"].iloc[lows[-i]] < df["low"].iloc[lows[-i-1]]
                if lh and hl:
                    consistency += 1
        confidence = min(1.0, consistency / 3.0)
        if tf_name in self.tf_groups.get("HTF", []):
            base_weight = 3.0
        elif tf_name in self.tf_groups.get("MID", []):
            base_weight = 2.0
        else:
            base_weight = 1.0
        weight = base_weight * (1.0 + confidence)
        reasons = [f"swing_consistency={consistency}/3"]
        return TimeframeVote(tf_name, direction, weight, confidence, reasons)

    def _timeframe_consensus(self, timeframes: dict) -> TimeframeConsensus:
        votes = []
        buy_w = 0.0
        sell_w = 0.0
        total_w = 0.0
        for tf_name in self._all_timeframe_names(timeframes):
            df = timeframes[tf_name]
            vote = self._compute_tf_vote(df, tf_name)
            votes.append(vote)
            total_w += vote.weight
            if vote.direction == "BUY":
                buy_w += vote.weight
            elif vote.direction == "SELL":
                sell_w += vote.weight
        if total_w <= 0:
            return TimeframeConsensus("HOLD", 0, 0, 0, 0, votes)
        max_dir = max(buy_w, sell_w)
        overall = "BUY" if buy_w >= sell_w else "SELL"
        if max_dir / max(total_w, 1) < 0.15:
            overall = "HOLD"
        return TimeframeConsensus(overall, buy_w, sell_w, total_w, max_dir / max(total_w, 1), votes)

    def _adjust_by_consensus(self, signal: TradingSignal, timeframes: dict) -> TradingSignal:
        consensus = self._timeframe_consensus(timeframes)
        signal.consensus = consensus
        if signal.direction not in ("BUY", "SELL"):
            return signal
        alignment = consensus.alignment_with(signal.direction)
        if alignment > 0.65:
            signal.score = min(100.0, signal.score * 1.2)
            signal.notes.append(f"Consensus TF {consensus.overall_direction} {alignment:.0%} a favor de {signal.direction}, +20%")
        elif alignment < 0.35:
            signal.score *= 0.55
            signal.notes.append(f"Consensus TF {consensus.overall_direction} vs {signal.direction} ({alignment:.0%}), -45%")
        elif alignment < 0.5:
            signal.score *= 0.75
            signal.notes.append(f"Consensus TF {consensus.overall_direction} débil vs {signal.direction} ({alignment:.0%}), -25%")
        return signal

    def evaluate(self, timeframes: Dict[str, pd.DataFrame],
                 current_time: datetime, news_active: bool = False,
                 htf_df: pd.DataFrame = None, ltf_df: pd.DataFrame = None,
                 dxy_df: pd.DataFrame = None) -> TradingSignal:
        if htf_df is None:
            htf_df = self._pick_tf(timeframes, "HTF")
        if ltf_df is None:
            ltf_df = self._pick_tf(timeframes, "LTF")

        if htf_df is None:
            all_dfs = self._all_timeframes_sorted(timeframes)
            if len(all_dfs) >= 2:
                htf_df = all_dfs[0]
                ltf_df = all_dfs[-1]
            elif len(all_dfs) == 1:
                htf_df = ltf_df = all_dfs[0]
        if htf_df is None or ltf_df is None:
            return TradingSignal(
                direction="HOLD", entry_price=0.0, stop_loss=0.0, take_profit=0.0,
                score=0.0, score_breakdown={},
                primary_pattern=None, supporting_patterns=[],
                rr_ratio=0.0, symbol=self.profile.symbol,
                timestamp=current_time, notes=["No hay datos multi-timeframe"],
            )

        ctx = self.analyzer.analyze_full_context(htf_df, ltf_df)

        ltf_patterns = self.detector.scan_all(ltf_df)
        htf_patterns = self.detector.scan_all(htf_df)

        buy_signal = self._evaluate_direction("BUY", ctx, ltf_patterns, htf_patterns, current_time, news_active, ltf_df, htf_df, dxy_df)
        sell_signal = self._evaluate_direction("SELL", ctx, ltf_patterns, htf_patterns, current_time, news_active, ltf_df, htf_df, dxy_df)

        net = buy_signal.score - sell_signal.score
        if abs(net) < self.min_net_score:
            notes = [f"Net {net:.0f} < min_net {self.min_net_score}"]
            best = buy_signal if buy_signal.score >= sell_signal.score else sell_signal
            return TradingSignal(
                direction="HOLD", entry_price=0.0, stop_loss=0.0, take_profit=0.0,
                score=best.score, score_breakdown=best.score_breakdown,
                primary_pattern=None, supporting_patterns=[],
                rr_ratio=0.0, symbol=self.profile.symbol,
                timestamp=current_time, notes=notes + best.notes,
            )

        best = buy_signal if net >= self.min_net_score else sell_signal

        total_tfs = sum(1 for v in timeframes.values() if v is not None and len(v) > 20)
        if total_tfs >= 2:
            best = self._adjust_by_consensus(best, timeframes)
        best.notes.append(f"TFs disponibles: {total_tfs}")

        if best.score < self.min_score:
            return TradingSignal(
                direction="HOLD", entry_price=0.0, stop_loss=0.0, take_profit=0.0,
                score=best.score, score_breakdown=best.score_breakdown,
                primary_pattern=None, supporting_patterns=[],
                rr_ratio=0.0, symbol=self.profile.symbol,
                timestamp=current_time, notes=[f"Score {best.score:.1f} < umbral {self.min_score}"] + best.notes,
            )
        return best

    def evaluate_adaptive(self, timeframes: Dict[str, pd.DataFrame],
                           current_time: datetime, news_active: bool = False,
                           regime: Optional[RegimeContext] = None,
                           htf_df: pd.DataFrame = None, ltf_df: pd.DataFrame = None,
                           dxy_df: pd.DataFrame = None) -> TradingSignal:
        if htf_df is None:
            htf_df = self._pick_tf(timeframes, "HTF")
        if ltf_df is None:
            ltf_df = self._pick_tf(timeframes, "LTF")

        if htf_df is None or ltf_df is None:
            return TradingSignal(
                direction="HOLD", entry_price=0.0, stop_loss=0.0, take_profit=0.0,
                score=0.0, score_breakdown={},
                primary_pattern=None, supporting_patterns=[],
                rr_ratio=0.0, symbol=self.profile.symbol,
                timestamp=current_time, notes=["No hay datos multi-timeframe"],
            )

        ctx = self.analyzer.analyze_full_context(htf_df, ltf_df)

        ltf_patterns = self.detector.scan_all(ltf_df)
        htf_patterns = self.detector.scan_all(htf_df)

        session_profile = self.session_profiler.profile(
            self.profile.symbol, ltf_df, current_time
        )

        buy_signal = self._evaluate_direction("BUY", ctx, ltf_patterns, htf_patterns,
                                               current_time, news_active, ltf_df, htf_df, dxy_df)
        sell_signal = self._evaluate_direction("SELL", ctx, ltf_patterns, htf_patterns,
                                                current_time, news_active, ltf_df, htf_df, dxy_df)

        if regime is None:
            regime = RegimeContext(
                regime=RegimeType.RANGING, confidence=0.3, strength=0.0,
                atr_ratio=1.0, adx_value=0.0, is_compressed=False, is_expanding=False,
                trend_alignment="NEUTRAL",
                pattern_multipliers={k: 1.0 for k in REGIME_PATTERN_MULTIPLIERS},
            )
        regime.set_context(session=session_profile.label)
        dist, buy_breakdown, sell_breakdown = merge_signals(
            buy_signal, sell_signal, regime,
            self.distributional_scorer,
        )

        direction = dist.direction
        conviction = dist.conviction

        min_conv = self._session_min_conviction if self._session_min_conviction is not None else getattr(self, 'min_conviction_to_trade', 0.12)
        if direction == "HOLD" or conviction < min_conv:
            best = buy_signal if buy_signal.score >= sell_signal.score else sell_signal
            return TradingSignal(
                direction="HOLD", entry_price=0.0, stop_loss=0.0, take_profit=0.0,
                score=best.score, score_breakdown=best.score_breakdown,
                primary_pattern=None, supporting_patterns=[],
                rr_ratio=0.0, symbol=self.profile.symbol,
                timestamp=current_time,
                notes=[f"Convicción {conviction:.0%} < {min_conv:.0%}"] + dist.notes + best.notes,
                conviction=conviction, distribution=dist,
                regime_context=regime, session_profile=session_profile,
            )

        best = buy_signal if direction == "BUY" else sell_signal

        total_tfs = sum(1 for v in timeframes.values() if v is not None and len(v) > 20)
        if total_tfs >= 2:
            best = self._adjust_by_consensus(best, timeframes)
            if best.consensus and direction in ("BUY", "SELL"):
                alignment = best.consensus.alignment_with(direction)
                if alignment < 0.35:
                    conviction *= 0.6
                    dist.conviction = conviction
                    dist.notes.append(f"Consensus TF {best.consensus.overall_direction} vs {direction} ({alignment:.0%}), convicción -40%")
                elif alignment > 0.65:
                    conviction = min(1.0, conviction * 1.25)
                    dist.conviction = conviction
                    dist.notes.append(f"Consensus TF {best.consensus.overall_direction} a favor ({alignment:.0%}), convicción +25%")

        effective_min_score = self._session_min_score if self._session_min_score is not None else self.min_score
        if abs(dist.mean) < effective_min_score:
            return TradingSignal(
                direction="HOLD",
                entry_price=0.0, stop_loss=0.0, take_profit=0.0,
                score=dist.mean,
                score_breakdown=best.score_breakdown,
                primary_pattern=None,
                supporting_patterns=[],
                rr_ratio=0.0,
                symbol=self.profile.symbol,
                timestamp=current_time,
                notes=[f"Score {dist.mean:.1f} < umbral {effective_min_score}"] + dist.notes + best.notes,
                conviction=conviction,
                distribution=dist,
                regime_context=regime,
                session_profile=session_profile,
            )

        entry = best.entry_price
        sl = best.stop_loss
        tp = best.take_profit
        if direction in ("BUY", "SELL") and (entry is None or entry == 0.0):
            entry = ltf_df["close"].iloc[-1] if ltf_df is not None and len(ltf_df) > 0 else 0.0
            if sl is None or sl == 0.0:
                sl = entry * 0.99 if direction == "BUY" else entry * 1.01
            if tp is None or tp == 0.0:
                tp = entry * 1.01 if direction == "BUY" else entry * 0.99
            best.notes.append(f"Entry fallback to market price {entry:.5f}")

        return TradingSignal(
            direction=direction,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            score=dist.mean,
            score_breakdown=best.score_breakdown,
            primary_pattern=best.primary_pattern,
            supporting_patterns=best.supporting_patterns,
            rr_ratio=best.rr_ratio,
            symbol=self.profile.symbol,
            timestamp=current_time,
            notes=dist.notes + best.notes,
            conviction=conviction,
            distribution=dist,
            regime_context=regime,
            session_profile=session_profile,
        )

    def _evaluate_direction(self, direction: str, ctx: MarketContext,
                            ltf_patterns: dict, htf_patterns: dict,
                            current_time: datetime, news_active: bool,
                            ltf_df: pd.DataFrame, htf_df: pd.DataFrame,
                            dxy_df: pd.DataFrame = None) -> TradingSignal:
        breakdown = {}
        score = 0.0
        supporting: List[Pattern] = []
        primary: Optional[Pattern] = None
        notes: List[str] = []

        current_price = htf_df["close"].iloc[-1] if htf_df is not None and len(htf_df) > 0 else 0

        if direction == "BUY" and ctx.htf_structure.trend == TrendDirection.BULLISH:
            score += self.weights.htf_trend_aligned
            breakdown["htf_trend_aligned"] = self.weights.htf_trend_aligned
        elif direction == "SELL" and ctx.htf_structure.trend == TrendDirection.BEARISH:
            score += self.weights.htf_trend_aligned
            breakdown["htf_trend_aligned"] = self.weights.htf_trend_aligned

        if direction == "BUY" and ctx.htf_structure.trend == TrendDirection.BEARISH:
            score += self.weights.htf_misalignment_penalty
            breakdown["htf_misalignment_penalty"] = self.weights.htf_misalignment_penalty
        elif direction == "SELL" and ctx.htf_structure.trend == TrendDirection.BULLISH:
            score += self.weights.htf_misalignment_penalty
            breakdown["htf_misalignment_penalty"] = self.weights.htf_misalignment_penalty

        if direction == "BUY" and ctx.htf_structure.in_discount_zone:
            score += self.weights.in_discount_premium_zone
            breakdown["in_discount_zone"] = self.weights.in_discount_premium_zone
        elif direction == "SELL" and ctx.htf_structure.in_premium_zone:
            score += self.weights.in_discount_premium_zone
            breakdown["in_premium_zone"] = self.weights.in_discount_premium_zone

        if ctx.htf_structure.last_bos_direction:
            if (direction == "BUY" and ctx.htf_structure.last_bos_direction == "BULLISH") or \
               (direction == "SELL" and ctx.htf_structure.last_bos_direction == "BEARISH"):
                score += self.weights.valid_market_structure
                breakdown["valid_market_structure"] = self.weights.valid_market_structure

        fvg_w = self._get_effective_weight("fvg_detected")
        fvg_match = self._best_pattern(ltf_patterns["fvg"], "FVG_BULLISH" if direction == "BUY" else "FVG_BEARISH", ltf_df)
        if fvg_match:
            score += fvg_w * fvg_match.confidence
            breakdown["fvg_detected"] = fvg_w * fvg_match.confidence
            supporting.append(fvg_match)
            if primary is None:
                primary = fvg_match

        ob_w = self._get_effective_weight("order_block_valid")
        ob_match = self._best_pattern(ltf_patterns["ob"], "OB_BULLISH" if direction == "BUY" else "OB_BEARISH", ltf_df)
        if ob_match:
            score += ob_w * ob_match.confidence
            breakdown["order_block_valid"] = ob_w * ob_match.confidence
            supporting.append(ob_match)
            if primary is None:
                primary = ob_match
            if self._check_ob_mitigated(ob_match, ltf_df, direction):
                score += self.weights.ob_mitigated_penalty
                breakdown["ob_mitigated"] = self.weights.ob_mitigated_penalty
                notes.append("OB mitigado (precio violó su límite)")

        breaker_w = self._get_effective_weight("breaker_retest")
        breaker_match = self._best_pattern(ltf_patterns["breaker"], "BREAKER_BULLISH" if direction == "BUY" else "BREAKER_BEARISH", ltf_df)
        if breaker_match:
            score += breaker_w * breaker_match.confidence
            breakdown["breaker_retest"] = breaker_w * breaker_match.confidence
            supporting.append(breaker_match)
            if self._check_breaker_mitigated(breaker_match, ltf_df, direction):
                score += self.weights.breaker_mitigated_penalty
                breakdown["breaker_mitigated"] = self.weights.breaker_mitigated_penalty
                notes.append("Breaker mitigado (precio retomó el nivel roto)")

        sweep_match = self._best_pattern(ltf_patterns["sweep"], "SWEEP_BULLISH" if direction == "BUY" else "SWEEP_BEARISH", ltf_df)
        if sweep_match:
            swept_level = sweep_match.extra.get("swept_level", sweep_match.mid)
            liq_type = self._classify_sweep_liquidity(swept_level, direction, htf_df)
            if liq_type == "EXTERNAL":
                score += self.weights.liquidity_sweep_ltf
                breakdown["liquidity_sweep_ltf"] = self.weights.liquidity_sweep_ltf
            else:
                score += self.weights.liquidity_sweep_ltf * 0.3
                breakdown["liquidity_sweep_ltf"] = self.weights.liquidity_sweep_ltf * 0.3
            supporting.append(sweep_match)

        cycle_w = self._get_effective_weight("cycle_full")
        cycle_match = self._best_pattern(ltf_patterns["cycle"], "CYCLE_BULLISH" if direction == "BUY" else "CYCLE_BEARISH", ltf_df)
        if cycle_match:
            base_cycle = (self.weights.slip_memory_present + self.weights.equidad_sweep_confirmed + self.weights.eslabon_breakout)
            score += base_cycle * cycle_w
            breakdown["cycle_full"] = base_cycle * cycle_w
            supporting.append(cycle_match)

        wyckoff_match = self._best_pattern(ltf_patterns["wyckoff"], "SPRING_BULLISH" if direction == "BUY" else "UTAD_BEARISH", ltf_df)
        if wyckoff_match:
            weight = self.weights.wyckoff_phase_c_spring if direction == "BUY" else self.weights.wyckoff_phase_c_utad
            score += weight + self.weights.wyckoff_phase_d_confirmed
            breakdown["wyckoff_full"] = weight + self.weights.wyckoff_phase_d_confirmed
            supporting.append(wyckoff_match)
            if primary is None:
                primary = wyckoff_match

        seq_match = self._best_pattern(ltf_patterns["sequence"], "SEQUENCE_BULLISH" if direction == "BUY" else "SEQUENCE_BEARISH", ltf_df)
        if seq_match:
            score += self.weights.ltf_bos_with_body * 0.6
            breakdown["sequence_123"] = self.weights.ltf_bos_with_body * 0.6
            supporting.append(seq_match)

        void_scalp_w = self._get_effective_weight("void_scalp_confirmed")
        void_scalp_match = self._best_pattern(
            ltf_patterns["void_scalp"],
            "VOID_SCALP_BULLISH" if direction == "BUY" else "VOID_SCALP_BEARISH",
            ltf_df,
        )
        if void_scalp_match:
            score += void_scalp_w * void_scalp_match.confidence
            breakdown["void_scalp"] = void_scalp_w * void_scalp_match.confidence
            supporting.append(void_scalp_match)
            if primary is None:
                primary = void_scalp_match

        bos_zone_w = self._get_effective_weight("bos_zone_retest")
        bos_zone_match = self._best_pattern(
            ltf_patterns["bos_zone_retest"],
            "BOS_ZONE_RETEST_BULLISH" if direction == "BUY" else "BOS_ZONE_RETEST_BEARISH",
            ltf_df,
        )
        if bos_zone_match:
            score += bos_zone_w * bos_zone_match.confidence
            breakdown["bos_zone_retest"] = bos_zone_w * bos_zone_match.confidence
            supporting.append(bos_zone_match)
            if primary is None:
                primary = bos_zone_match
            notes.append(f"BOS Zone Retest: indec={bos_zone_match.extra.get('indec_count', 0)}")

        pe_w = self._get_effective_weight("price_establishment_bonus")
        pe_match = self._best_pattern(
            ltf_patterns.get("price_establishment", []),
            "PRICE_ESTABLISHMENT_LONG" if direction == "BUY" else "PRICE_ESTABLISHMENT_SHORT",
            ltf_df,
        )
        if pe_match:
            score += pe_w * pe_match.confidence
            breakdown["price_est"] = pe_w * pe_match.confidence
            notes.append(f"Price est: {pe_match.extra.get('touches', 0)} toques")

        sub_fractals = ltf_patterns.get("sub_fractals", {})
        if isinstance(sub_fractals, dict) and sub_fractals.get("third_movement_ready"):
            sf_dir = sub_fractals["direction"]
            if (direction == "BUY" and sf_dir == "bullish") or \
               (direction == "SELL" and sf_dir == "bearish"):
                score += self.weights.third_movement_ready
                breakdown["3rd_movement"] = self.weights.third_movement_ready
                notes.append(f"3er movimiento subfractálico listo ({sf_dir})")

        if self._check_psychological_price(ltf_df, direction):
            score += self.weights.psychological_price_aligned
            breakdown["psych_price"] = self.weights.psychological_price_aligned
            notes.append("Precio psicológico alineado (.25/.50/.75/.00)")

        sweep_conf_w = self._get_effective_weight("ltf_sweep_confirmation")
        if self._has_ltf_sweep_confirmation(ctx, ltf_df):
            score += sweep_conf_w
            breakdown["ltf_sweep_confirmation"] = sweep_conf_w

        mf_align_w = self._get_effective_weight("multiframe_alignment")
        if self._has_multiframe_alignment(ctx):
            score += mf_align_w
            breakdown["multiframe_alignment"] = mf_align_w

        triple_conf_w = self._get_effective_weight("triple_confluence")
        triple = self._check_triple_confluence(ltf_patterns, direction)
        if triple:
            score += triple_conf_w
            breakdown["triple_confluence"] = triple_conf_w
            notes.append("Triple confluence: OB+FVG+liquidez")

        wyckoff_phase_d = self._check_wyckoff_phase_d(ltf_patterns, direction)
        if wyckoff_phase_d:
            score += self.weights.wyckoff_phase_d_lps
            breakdown["wyckoff_phase_d"] = self.weights.wyckoff_phase_d_lps
            notes.append("Wyckoff Phase D: LPS/LPSY detectado")
            if self._check_lps_mitigated(wyckoff_phase_d, ltf_df, direction):
                score += self.weights.lps_mitigated_penalty
                breakdown["lps_mitigated"] = self.weights.lps_mitigated_penalty
                notes.append("LPS/LPSY mitigado (precio lo violó)")

        if self._check_wyckoff_sos_volume(ltf_patterns, direction):
            score += self.weights.wyckoff_sos_volume
            breakdown["wyckoff_sos_volume"] = self.weights.wyckoff_sos_volume

        trb_result = self.analyzer.detect_trb_manipulation(ltf_df, lookback_bars=30)
        if trb_result["fake_breakout"]:
            trb_dir_ok = (direction == "SELL" and trb_result["direction"] == "SELL") or \
                         (direction == "BUY" and trb_result["direction"] == "BUY")
            if trb_dir_ok:
                score += self.weights.trb_manipulation_detected
                breakdown["trb_manipulation"] = self.weights.trb_manipulation_detected
                notes.append("TRB: fake breakout + manipulation detectado")
                if trb_result["displacement"]:
                    score += self.weights.trb_displacement
                    breakdown["trb_displacement"] = self.weights.trb_displacement
                    notes.append(f"TRB: displacement {trb_result['displacement_size']:.1f}")
                if self._check_trb_retest(ltf_df, trb_result, direction):
                    score += self.weights.trb_retest
                    breakdown["trb_retest"] = self.weights.trb_retest
                    notes.append("TRB: retest de rango roto")

        fvg_fresh = self._check_fvg_fresh(ltf_patterns, ltf_df, direction)
        if fvg_fresh:
            score += self.weights.fvg_fresh_mitigation
            breakdown["fvg_fresh"] = self.weights.fvg_fresh_mitigation

        if self._check_sweep_wick(ltf_patterns, direction):
            score += self.weights.sweep_spring_wick
            breakdown["sweep_wick"] = self.weights.sweep_spring_wick

        if self._check_body_close_valid(ctx, direction):
            score += self.weights.body_close_valid
            breakdown["body_close_valid"] = self.weights.body_close_valid

        if primary is not None:
            pattern_level = primary.high if direction == "SELL" else primary.low
            pe_result = self._check_price_establishment(ltf_df, pattern_level, direction)
            if pe_result["valid"]:
                base = self.weights.price_establishment
                bonus = self.weights.wick_rejection_bonus * (pe_result["score_mult"] - 1.0)
                total_pe = base + bonus
                score += total_pe
                breakdown["price_establishment"] = total_pe
                breakdown["wick_rejections"] = pe_result["count"]
                notes.append(f"Wick rejections: {pe_result['count']} ({pe_result['intensity']})")

        body_invalid = self._check_body_close_invalid(ltf_df, ltf_patterns, direction)
        if body_invalid:
            score += self.weights.body_close_invalid
            breakdown["body_close_invalid"] = self.weights.body_close_invalid
            notes.append("INVALID: body close fuera de rango (real breakout)")

        if not self._has_sweep_for_direction(ltf_patterns, direction):
            score += self.weights.no_sweep_detected
            breakdown["no_sweep"] = self.weights.no_sweep_detected
            notes.append("INVALID: sin sweep de liquidez")

        fvg_burned = self._check_fvg_burned(ltf_patterns, ltf_df, direction)
        if fvg_burned:
            score += self.weights.fvg_burned_over_50
            breakdown["fvg_burned"] = self.weights.fvg_burned_over_50
            notes.append("FVG >50% mitigado (quemado)")

        spring_body = self._check_spring_body_close(ltf_patterns, direction)
        if spring_body:
            score += self.weights.spring_body_close
            breakdown["spring_body_close"] = self.weights.spring_body_close
            notes.append("Spring/UTAD cerró con cuerpo: fallo real")

        vsa = VSADetector.analyze(ltf_df, direction)
        vq = vsa.get("volume_quality", 0.5)
        if vq < 0.5:
            notes.append(f"VSA: volumen poco confiable (calidad={vq:.2f}) — señales atenuadas")
        if vsa.get("volume_confirmation"):
            contrib = self.weights.vsa_volume_confirmation * (0.3 + 0.7 * vq)
            score += contrib
            breakdown["vsa_volume_confirmation"] = contrib
            notes.append(f"VSA: volumen confirma breakout (x{vq:.2f})")
        if vsa.get("climax"):
            contrib = self.weights.vsa_climax_penalty * (0.3 + 0.7 * vq)
            score += contrib
            breakdown["vsa_climax"] = contrib
            notes.append(f"VSA: CLIMAX detectado (x{vq:.2f})")
        if vsa.get("absorption"):
            contrib = self.weights.vsa_absorption_bonus * (0.3 + 0.7 * vq)
            score += contrib
            breakdown["vsa_absorption"] = contrib
            notes.append(f"VSA: absorción (x{vq:.2f})")
        if vsa.get("low_volume_pullback"):
            contrib = self.weights.vsa_low_volume_pullback * (0.3 + 0.7 * vq)
            score += contrib
            breakdown["vsa_low_volume_pullback"] = contrib
            notes.append(f"VSA: pullback bajo volumen (x{vq:.2f})")
        if vsa.get("volume_divergence"):
            contrib = self.weights.vsa_volume_divergence * (0.3 + 0.7 * vq)
            score += contrib
            breakdown["vsa_volume_divergence"] = contrib
            notes.append(f"VSA: divergencia volumen (x{vq:.2f})")
        if vsa.get("no_demand_supply"):
            contrib = self.weights.vsa_no_demand_supply * (0.3 + 0.7 * vq)
            score += contrib
            breakdown["vsa_no_demand_supply"] = contrib
            notes.append(f"VSA: sin demanda/oferta (x{vq:.2f})")

        dxy_result = self._check_dxy_correlation(dxy_df, direction)
        if dxy_result["score"] != 0:
            score += dxy_result["score"]
            key = "dxy_aligned" if dxy_result["aligned"] else "dxy_conflict"
            breakdown[key] = dxy_result["score"]
            notes.append(f"DXY {dxy_result['trend']}: {key}")

        # ── Anticipación avanzada (divergencias, order flow, volume profile) ──

        divergences = DivergenceDetector.detect_all(htf_df)
        best_bull, bull_type = DivergenceDetector.best_divergence(divergences, "BUY")
        best_bear, bear_type = DivergenceDetector.best_divergence(divergences, "SELL")

        if direction == "BUY" and best_bull:
            w_base = (self.weights.divergence_regular_bonus if bull_type == "regular"
                      else self.weights.divergence_hidden_bonus)
            contrib = w_base * best_bull.strength
            score += contrib
            breakdown["divergence_bull"] = contrib
            notes.append(f"Div {bull_type.upper()}: {best_bull.source} (str={best_bull.strength:.0%})")
        elif direction == "SELL" and best_bear:
            w_base = (self.weights.divergence_regular_bonus if bear_type == "regular"
                      else self.weights.divergence_hidden_bonus)
            contrib = w_base * best_bear.strength
            score += contrib
            breakdown["divergence_bear"] = contrib
            notes.append(f"Div {bear_type.upper()}: {best_bear.source} (str={best_bear.strength:.0%})")
        elif (direction == "BUY" and best_bear) or (direction == "SELL" and best_bull):
            contrib = self.weights.divergence_penalty * 0.5
            score += contrib
            breakdown["divergence_misaligned"] = contrib
            notes.append("Divergencia en contra del trade")

        # Order flow
        of = OrderFlowAnalyzer.analyze(ltf_df)
        if direction == "BUY" and of.buying_pressure > of.selling_pressure * 1.3:
            contrib = self.weights.order_flow_imbalance_bonus * of.buying_pressure
            score += contrib
            breakdown["of_buying_pressure"] = contrib
        elif direction == "SELL" and of.selling_pressure > of.buying_pressure * 1.3:
            contrib = self.weights.order_flow_imbalance_bonus * of.selling_pressure
            score += contrib
            breakdown["of_selling_pressure"] = contrib

        if of.divergence_active:
            aligned = (direction == "BUY" and of.delta_imbalance > 0) or \
                      (direction == "SELL" and of.delta_imbalance < 0)
            if aligned:
                contrib = self.weights.order_flow_divergence_bonus
                score += contrib
                breakdown["of_divergence_aligned"] = contrib
                notes.append("OF divergencia alineada con dirección")
        if of.exhaustion_active:
            contrib = self.weights.order_flow_exhaustion_penalty
            score += contrib
            breakdown["of_exhaustion"] = contrib
            notes.append("OF: agotamiento de delta")
        if of.absorption_active:
            contrib = self.weights.order_flow_absorption_bonus
            score += contrib
            breakdown["of_absorption"] = contrib
            notes.append("OF: absorción (price no sigue volumen)")

        # Volume Profile
        vp = VolumeProfile.compute(htf_df)
        if vp.poc != 0:
            is_premium = direction == "BUY" and current_price >= vp.poc
            is_discount = direction == "SELL" and current_price <= vp.poc
            if is_premium or is_discount:
                contrib = self.weights.volume_profile_poc_discount_bonus
                score += contrib
                zone = "premium" if is_premium else "discount"
                breakdown["vp_poc_discount"] = contrib
                notes.append(f"VP: precio en zona {zone} vs POC")
        if vp.low_volume_nodes:
            contrib = self.weights.volume_profile_lvn_bonus
            score += contrib
            breakdown["vp_lvn"] = contrib
            notes.append("VP: LVN detectado (bajo volumen)")
        if vp.is_compressed:
            contrib = self.weights.volume_profile_compression_bonus
            score += contrib
            breakdown["vp_compression"] = contrib
            notes.append("VP: compresión (rango estrecho)")
        if abs(vp.poc_shift_pips) > 5:
            if (direction == "BUY" and vp.poc_shift_pips > 0) or \
               (direction == "SELL" and vp.poc_shift_pips < 0):
                contrib = self.weights.volume_profile_poc_shift_bonus
                score += contrib
                breakdown["vp_poc_shift_aligned"] = contrib
                notes.append(f"VP: POC shift {vp.poc_shift_pips:+.0f}p alineado")
            else:
                contrib = self.weights.volume_profile_poc_shift_penalty
                score += contrib
                breakdown["vp_poc_shift_misaligned"] = contrib
                notes.append(f"VP: POC shift {vp.poc_shift_pips:+.0f}p en contra")

        # DXY augmented
        if dxy_df is not None and len(dxy_df) >= 20:
            dxy_regime = DXYCorrelationAnalyzer.analyze(dxy_df, htf_df)
            if dxy_regime.divergence_active:
                dxy_score = dxy_regime.strength * (1 if dxy_regime.trend == "bullish" else -1)
                if (direction == "BUY" and dxy_score < -0.3) or \
                   (direction == "SELL" and dxy_score > 0.3):
                    contrib = self.weights.dxy_divergence_bonus
                    score += contrib
                    breakdown["dxy_divergence"] = contrib
                    notes.append("DXY diverge de pair — confirma contra-tendencia")

        # ── Nuevos conceptos del curso ─────────────────────────────

        ip_w = self._get_effective_weight("interval_point_bonus")
        ip_match = self._best_pattern(
            ltf_patterns.get("interval_points", []),
            "INTERVAL_POINT_BULLISH" if direction == "BUY" else "INTERVAL_POINT_BEARISH",
            ltf_df,
        )
        if ip_match:
            score += ip_w * ip_match.confidence
            breakdown["interval_point"] = ip_w * ip_match.confidence
            notes.append(f"Punto de Intervalo: mechas {ip_match.extra.get('wick_ratio', 0):.1f}× cuerpo")
            if primary is None:
                primary = ip_match

        pi_w = self._get_effective_weight("price_interaction_bonus")
        pi_match = self._best_pattern(
            ltf_patterns.get("price_interaction", []),
            "PRICE_INTERACTION_BULLISH" if direction == "BUY" else "PRICE_INTERACTION_BEARISH",
            ltf_df,
        )
        if pi_match:
            score += pi_w * pi_match.confidence
            breakdown["price_interaction"] = pi_w * pi_match.confidence
            notes.append(f"Interacción precio: nivel {pi_match.extra.get('interacted_level', 0):.2f}")
            if primary is None:
                primary = pi_match

        hc_w = self._get_effective_weight("harmonic_cycle_aligned")
        hc_match = self._best_pattern(
            ltf_patterns.get("harmonic_cycle", []),
            "HARMONIC_CYCLE_BULLISH" if direction == "BUY" else "HARMONIC_CYCLE_BEARISH",
            ltf_df,
        )
        if hc_match:
            score += hc_w * hc_match.confidence
            breakdown["harmonic_cycle"] = hc_w * hc_match.confidence
            notes.append(f"Ciclo Armónico: 50% en {hc_match.mid:.2f}")

        pz_w = self._get_effective_weight("pressure_zone_bonus")
        pz_match = self._best_pattern(
            ltf_patterns.get("pressure_zones", []),
            "PRESSURE_ZONE_BULLISH" if direction == "BUY" else "PRESSURE_ZONE_BEARISH",
            ltf_df,
        )
        if pz_match:
            score += pz_w * pz_match.confidence
            breakdown["pressure_zone"] = pz_w * pz_match.confidence
            notes.append(f"Zona POD: {pz_match.extra.get('touches', 0)} toques, rango {pz_match.extra.get('zone_range', 0):.1f}")

        # Pause vs Retracement (Clase 12)
        from src.utils.helpers import classify_retracement
        retrace_type = classify_retracement(ltf_df, direction)
        if retrace_type == "pause":
            score += self.weights.pause_continuation_bonus
            breakdown["pause_continuation"] = self.weights.pause_continuation_bonus
            notes.append("Pausa (<4 velas): continuación de impulso esperada")
        elif retrace_type == "retracement":
            score += self.weights.retracement_penalty
            breakdown["retracement_penalty"] = self.weights.retracement_penalty
            notes.append("Retroceso (>4 velas): estructura profundizando, penalizado")

        # Order Flow (Clase 6, 12)
        from src.utils.helpers import classify_order_flow
        of_type = classify_order_flow(ltf_df)
        if of_type == "regular":
            score += self.weights.order_flow_regular_bonus
            breakdown["order_flow_regular"] = self.weights.order_flow_regular_bonus
            notes.append("Order Flow Regular: fractal limpio")
        elif of_type == "irregular":
            score += self.weights.order_flow_irregular_penalty
            breakdown["order_flow_irregular"] = self.weights.order_flow_irregular_penalty
            notes.append("Order Flow Irregular: price zone-hopping (menos confiable)")

        # Breaker 3-touch limit (Clase 10)
        breaker_key = "BREAKER_BULLISH" if direction == "BUY" else "BREAKER_BEARISH"
        for bp in ltf_patterns.get("breaker", []):
            if bp.type.name == breaker_key:
                from src.utils.helpers import breaker_touch_count
                touches = breaker_touch_count(bp, ltf_df)
                if touches >= 4:
                    notes.append(f"BREAKER {bp.mid:.2f} alcanzó {touches} toques → inválido")
                    score += self.weights.breaker_3_touch_limit
                    breakdown["breaker_3_touch"] = self.weights.breaker_3_touch_limit

        if is_in_session(current_time, self.profile.allowed_sessions):
            score += self.weights.in_active_session
            breakdown["in_active_session"] = self.weights.in_active_session
        else:
            notes.append("Fuera de sesión activa")

        zone = detect_killzone(current_time)
        if zone == Killzone.LONDON_OPEN:
            score += self.weights.killzone_london_open
            breakdown["killzone"] = self.weights.killzone_london_open
            notes.append("Killzone: London Open")
        elif zone == Killzone.NY_OPEN:
            score += self.weights.killzone_ny_open
            breakdown["killzone"] = self.weights.killzone_ny_open
            notes.append("Killzone: NY Open")
        elif zone == Killzone.LONDON_NY_OVERLAP:
            score += self.weights.killzone_london_ny_overlap
            breakdown["killzone"] = self.weights.killzone_london_ny_overlap
            notes.append("Killzone: London/NY Overlap")
        elif zone == Killzone.ASIAN:
            score += self.weights.killzone_asian
            breakdown["killzone"] = self.weights.killzone_asian
            notes.append("Killzone: Asian")

        if news_active:
            score += self.weights.news_high_impact_penalty
            breakdown["news_penalty"] = self.weights.news_high_impact_penalty
            notes.append("Noticia de alto impacto activa")
        if not news_active:
            score += self.weights.no_news_event
            breakdown["no_news"] = self.weights.no_news_event

        if ctx.regime == MarketRegime.EXPANSION:
            score += self.weights.regime_expansion
            breakdown["regime_expansion"] = self.weights.regime_expansion
        if ctx.regime != MarketRegime.ACCUMULATION:
            score += self.weights.regime_not_accumulation
            breakdown["regime_not_accumulation"] = self.weights.regime_not_accumulation

        price_aligned, best_level, pip_dist = self._price_grid_alignment(ltf_df["close"].iloc[-1])
        if price_aligned:
            score += self.weights.price_grid_aligned
            breakdown["price_grid_aligned"] = self.weights.price_grid_aligned

        # Price-action trend bonus/penalty (con fallback a EMA slope)
        lookback = min(len(ltf_df), 15)
        if lookback >= 5:
            from src.utils.helpers import find_swing_points
            highs, lows = find_swing_points(ltf_df, lookback=3)
            trend_up = False
            trend_dn = False
            if len(highs) >= 2 and len(lows) >= 2:
                last_hh = ltf_df["high"].iloc[highs[-1]] > ltf_df["high"].iloc[highs[-2]]
                last_hl = ltf_df["low"].iloc[lows[-1]] > ltf_df["low"].iloc[lows[-2]]
                if last_hh and last_hl:
                    trend_up = True
                elif not last_hh and not last_hl:
                    trend_dn = True
            if not trend_up and not trend_dn:
                ema8 = ltf_df["close"].ewm(span=8).mean()
                if len(ema8) >= 5:
                    trend_up = ema8.iloc[-1] > ema8.iloc[-5]
                    trend_dn = ema8.iloc[-1] < ema8.iloc[-5]
            if trend_up:
                if direction == "BUY":
                    score += self.weights.price_trend_aligned
                    breakdown["price_trend"] = self.weights.price_trend_aligned
                else:
                    score -= self.weights.price_trend_aligned * 2.0
                    breakdown["price_trend_penalty"] = -self.weights.price_trend_aligned * 2.0
            elif trend_dn:
                if direction == "SELL":
                    score += self.weights.price_trend_aligned
                    breakdown["price_trend"] = self.weights.price_trend_aligned
                else:
                    score -= self.weights.price_trend_aligned * 2.0
                    breakdown["price_trend_penalty"] = -self.weights.price_trend_aligned * 2.0

        pattern_confluence = self._compute_pattern_confluence(ltf_patterns, direction, ltf_df)
        if pattern_confluence < 1.0:
            old_score = score
            score *= pattern_confluence
            breakdown["pattern_confluence_penalty"] = score - old_score
            notes.append(f"Patrones solapados: mult={pattern_confluence:.2f}")

        DIRECTION_PATTERN_CAP = 45.0
        pattern_keys = [k for k in breakdown if k not in (
            "htf_trend_aligned", "in_discount_zone", "in_premium_zone",
            "valid_market_structure", "killzone", "news_penalty", "no_news",
            "regime_expansion", "regime_not_accumulation", "in_active_session",
            "price_grid_aligned", "price_trend", "price_trend_penalty",
            "dxy_aligned", "dxy_conflict",
        )]
        pattern_total = sum(v for k, v in breakdown.items() if k in pattern_keys and v > 0)
        if pattern_total > DIRECTION_PATTERN_CAP:
            scale = DIRECTION_PATTERN_CAP / max(1.0, pattern_total)
            for k in pattern_keys:
                if breakdown.get(k, 0) > 0:
                    reduction = breakdown[k] * (1 - scale)
                    breakdown[k] *= scale
                    score -= reduction
            notes.append(f"Patrones capeados: {pattern_total:.0f}→{DIRECTION_PATTERN_CAP:.0f}")

        if primary is None:
            return TradingSignal(
                direction="HOLD", entry_price=0.0, stop_loss=0.0, take_profit=0.0,
                score=score, score_breakdown=breakdown,
                primary_pattern=None, supporting_patterns=supporting,
                rr_ratio=0.0, symbol=self.profile.symbol,
                timestamp=current_time, notes=notes + ["Sin patrón primario detectado"],
            )

        entry, sl, tp, rr = self._build_order(direction, primary, ctx, ltf_df, supporting, htf_patterns)
        if rr < self.profile.min_rr_ratio:
            score *= 0.5
            notes.append(f"RR={rr:.2f} bajo el mínimo {self.profile.min_rr_ratio}")

        return TradingSignal(
            direction=direction, entry_price=entry, stop_loss=sl, take_profit=tp,
            score=score, score_breakdown=breakdown,
            primary_pattern=primary, supporting_patterns=supporting,
            rr_ratio=rr, symbol=self.profile.symbol,
            timestamp=current_time, notes=notes,
        )

    def _find_tp_target(self, direction: str, entry: float, sl: float, ltf_df: pd.DataFrame, ctx: MarketContext,
                        htf_patterns: Optional[Dict] = None) -> float:
        risk = abs(entry - sl)
        pip = self.pip
        atr_val = atr(ltf_df, 14).iloc[-1]

        htf = ctx.htf_structure
        candidates = []

        # --- 1) HTF structural targets (professional levels) ---
        if direction == "BUY":
            if htf.last_swing_high and htf.last_swing_high.price > entry:
                rr = (htf.last_swing_high.price - entry) / risk if risk > 0 else 0
                candidates.append((htf.last_swing_high.price, rr, "HTF_SWING_HIGH"))
            if htf.valid_range_high > entry:
                rr = (htf.valid_range_high - entry) / risk if risk > 0 else 0
                candidates.append((htf.valid_range_high, rr, "HTF_VALID_RANGE_HIGH"))
        else:
            if htf.last_swing_low and htf.last_swing_low.price < entry:
                rr = (entry - htf.last_swing_low.price) / risk if risk > 0 else 0
                candidates.append((htf.last_swing_low.price, rr, "HTF_SWING_LOW"))
            if htf.valid_range_low > 0 and htf.valid_range_low < entry:
                rr = (entry - htf.valid_range_low) / risk if risk > 0 else 0
                candidates.append((htf.valid_range_low, rr, "HTF_VALID_RANGE_LOW"))

        # --- 2) HTF Order Blocks as TP targets ---
        if htf_patterns:
            obs = htf_patterns.get("order_block", [])
            for ob in obs:
                if direction == "BUY" and ob.high > entry:
                    rr = (ob.high - entry) / risk if risk > 0 else 0
                    candidates.append((ob.high, rr, "HTF_OB"))
                elif direction == "SELL" and ob.low < entry:
                    rr = (entry - ob.low) / risk if risk > 0 else 0
                    candidates.append((ob.low, rr, "HTF_OB"))

        # --- 3) HTF equilibrium as secondary target ---
        if htf.equilibrium_level > 0:
            if direction == "BUY" and htf.equilibrium_level > entry:
                rr = (htf.equilibrium_level - entry) / risk if risk > 0 else 0
                candidates.append((htf.equilibrium_level, rr, "HTF_EQ"))
            elif direction == "SELL" and htf.equilibrium_level < entry:
                rr = (entry - htf.equilibrium_level) / risk if risk > 0 else 0
                candidates.append((htf.equilibrium_level, rr, "HTF_EQ"))

        # --- 4) LTF swing points as micro-targets ---
        from src.utils.helpers import find_swing_points
        highs_idx, lows_idx = find_swing_points(ltf_df, lookback=3)
        if direction == "BUY":
            for idx in highs_idx:
                if idx >= 0 and ltf_df["high"].iloc[idx] > entry:
                    tp_candidate = ltf_df["high"].iloc[idx]
                    rr = (tp_candidate - entry) / risk if risk > 0 else 0
                    candidates.append((tp_candidate, rr, "LTF_SWING"))
            for idx in lows_idx:
                if idx >= 0 and ltf_df["low"].iloc[idx] > entry:
                    tp_candidate = ltf_df["low"].iloc[idx]
                    rr = (tp_candidate - entry) / risk if risk > 0 else 0
                    candidates.append((tp_candidate, rr, "LTF_SWING"))
        else:
            for idx in lows_idx:
                if idx >= 0 and ltf_df["low"].iloc[idx] < entry:
                    tp_candidate = ltf_df["low"].iloc[idx]
                    rr = (entry - tp_candidate) / risk if risk > 0 else 0
                    candidates.append((tp_candidate, rr, "LTF_SWING"))
            for idx in highs_idx:
                if idx >= 0 and ltf_df["high"].iloc[idx] < entry:
                    tp_candidate = ltf_df["high"].iloc[idx]
                    rr = (entry - tp_candidate) / risk if risk > 0 else 0
                    candidates.append((tp_candidate, rr, "LTF_SWING"))

        if not candidates:
            return self._tp_fallback(direction, entry, risk, atr_val)

        # Dynamic min RR: scales with volatility
        atr_ratio = atr_val / (ltf_df["close"].iloc[-1] * pip)
        if atr_ratio > 0.002:
            rr_threshold = max(2.5, self.profile.min_rr_ratio)
        elif atr_ratio < 0.001:
            rr_threshold = max(1.5, self.profile.min_rr_ratio)
        else:
            rr_threshold = max(2.0, self.profile.min_rr_ratio)

        # Cap max TP distance to avoid exaggerated targets
        max_tp_atr_mult = 12.0
        max_tp_distance = atr_val * max_tp_atr_mult

        best = None
        candidates.sort(key=lambda x: abs(x[0] - entry))
        for tp, rr, src in candidates:
            tp_distance = abs(tp - entry)
            if rr >= rr_threshold and tp_distance <= max_tp_distance:
                best = (tp, rr, src)
                break

        if best is None:
            candidates.sort(key=lambda x: abs(x[0] - entry))
            for tp, rr, src in candidates:
                tp_distance = abs(tp - entry)
                if tp_distance <= max_tp_distance:
                    best = (tp, rr, src)
                    break

        if best is None:
            candidates.sort(key=lambda x: abs(x[0] - entry))
            best = candidates[0]
            if best[1] < 1.0:
                return self._tp_fallback(direction, entry, risk, atr_val)

        return best[0]

    def _tp_fallback(self, direction: str, entry: float, risk: float, atr_val: float) -> float:
        min_rr = self.profile.min_rr_ratio
        rr_mult = max(min_rr, 2.0)
        if direction == "BUY":
            return entry + atr_val * rr_mult
        else:
            return entry - atr_val * rr_mult

    def _build_order(self, direction: str, primary: Optional[Pattern], ctx: MarketContext,
                     ltf_df: pd.DataFrame, supporting: Optional[List[Pattern]] = None,
                     htf_patterns: Optional[Dict] = None) -> Tuple[float, float, float, float]:
        atr_val = atr(ltf_df, 14).iloc[-1]
        pip = self.pip
        sl_buffer = max(atr_val * 0.5, atr_val * 0.0)
        current_price = ltf_df["close"].iloc[-1]
        supporting = supporting or []

        # --- Coupled Entry + SL from structural levels ---
        entry = current_price
        sl = None
        sl_is_tight = False

        if primary is not None:
            pattern_type = primary.type.name
            if pattern_type.startswith("FVG") or pattern_type.startswith("VOID_SCALP"):
                gap_entry = primary.low + (primary.high - primary.low) * 0.5
                entry_dist = abs(current_price - gap_entry)
                if entry_dist <= atr_val * 9.0:
                    entry = gap_entry
                    sl = primary.low - sl_buffer if direction == "BUY" else primary.high + sl_buffer
                    sl_is_tight = True
            elif pattern_type.startswith("SOS") or pattern_type.startswith("SOW"):
                gap_entry = primary.mid
                entry_dist = abs(current_price - gap_entry)
                if entry_dist <= atr_val * 9.0:
                    entry = gap_entry
                    sl = primary.low - sl_buffer if direction == "BUY" else primary.high + sl_buffer
                    sl_is_tight = True
            elif pattern_type.startswith("OB"):
                ob_entry = primary.high if direction == "SELL" else primary.low
                entry_dist = abs(current_price - ob_entry)
                if entry_dist <= atr_val * 12.0:
                    entry = ob_entry
                    sl = primary.low - sl_buffer if direction == "BUY" else primary.high + sl_buffer
                    sl_is_tight = True
            elif pattern_type.startswith("BOS_ZONE_RETEST"):
                zone_boundary = primary.extra.get("zone_min") if direction == "BUY" else primary.extra.get("zone_max")
                if zone_boundary is not None:
                    entry = zone_boundary
                    sl_candidate = primary.extra.get("sl_price")
                    if sl_candidate is not None:
                        sl = sl_candidate - sl_buffer if direction == "BUY" else sl_candidate + sl_buffer
                        sl_is_tight = True
            elif pattern_type.startswith("INTERVAL_POINT"):
                ip_entry = primary.high if direction == "SELL" else primary.low
                entry_dist = abs(current_price - ip_entry)
                if entry_dist <= atr_val * 12.0:
                    entry = ip_entry
                    sl = primary.low - sl_buffer if direction == "BUY" else primary.high + sl_buffer
                    sl_is_tight = True
            elif pattern_type.startswith("PRICE_INTERACTION"):
                pi_entry = primary.mid
                entry_dist = abs(current_price - pi_entry)
                if entry_dist <= atr_val * 12.0:
                    entry = pi_entry
                    sl = primary.low - sl_buffer if direction == "BUY" else primary.high + sl_buffer
                    sl_is_tight = True
            elif pattern_type.startswith("HARMONIC_CYCLE"):
                hc_entry = primary.mid
                entry_dist = abs(current_price - hc_entry)
                if entry_dist <= atr_val * 9.0:
                    entry = hc_entry
                    sl = primary.low - sl_buffer if direction == "BUY" else primary.high + sl_buffer
                    sl_is_tight = True
            elif pattern_type.startswith("PRESSURE_ZONE"):
                pz_entry = primary.mid
                entry_dist = abs(current_price - pz_entry)
                if entry_dist <= atr_val * 9.0:
                    entry = pz_entry
                    sl = primary.low - sl_buffer if direction == "BUY" else primary.high + sl_buffer
                    sl_is_tight = True

        if sl is None:
            from src.utils.helpers import find_swing_points
            highs_idx, lows_idx = find_swing_points(ltf_df, lookback=3)

            if direction == "BUY":
                best_low = None
                for idx in reversed(lows_idx):
                    sp = ltf_df["low"].iloc[idx]
                    if sp < current_price:
                        best_low = sp
                        break
                if best_low is not None:
                    entry = best_low + atr_val * 0.1
                    sl = best_low - atr_val * 0.3
                    sl_is_tight = True
                else:
                    entry = current_price - atr_val * 0.15
            else:
                best_high = None
                for idx in reversed(highs_idx):
                    sp = ltf_df["high"].iloc[idx]
                    if sp > current_price:
                        best_high = sp
                        break
                if best_high is not None:
                    entry = best_high - atr_val * 0.1
                    sl = best_high + atr_val * 0.3
                    sl_is_tight = True
                else:
                    entry = current_price + atr_val * 0.15

        # --- SL: structural search if tight SL wasn't set ---
        if not sl_is_tight:
            sl = None
            sweep_patterns = [p for p in supporting if p.type.name.startswith("SWEEP")]
            matching_sweeps = [p for p in sweep_patterns if p.direction == direction]
            if matching_sweeps:
                best = matching_sweeps[0]
                swept_level = best.extra.get("swept_level", best.mid)
                if direction == "BUY":
                    sl = swept_level - sl_buffer
                else:
                    sl = swept_level + sl_buffer

            if sl is None:
                for p in supporting:
                    if p.type.name.startswith("INTERVAL_POINT") and p.direction == direction:
                        sl = p.low - sl_buffer if direction == "BUY" else p.high + sl_buffer
                        break

            if sl is None:
                for p in supporting:
                    if p.type.name.startswith("PRICE_INTERACTION") and p.direction == direction:
                        level = p.extra.get("interacted_level", p.mid)
                        sl = level - sl_buffer if direction == "BUY" else level + sl_buffer
                        break

            if sl is None:
                for p in supporting:
                    if p.type.name.startswith("HARMONIC_CYCLE") and p.direction == direction:
                        sl = p.low - sl_buffer if direction == "BUY" else p.high + sl_buffer
                        break

            if sl is None:
                for p in supporting:
                    if p.type == PatternType.SPRING_BULLISH and direction == "BUY":
                        spring_low = p.extra.get("spring_low", p.low)
                        sl = spring_low - sl_buffer
                        break
                    elif p.type == PatternType.UTAD_BEARISH and direction == "SELL":
                        utad_high = p.extra.get("utad_high", p.high)
                        sl = utad_high + sl_buffer
                        break

            if sl is None:
                from src.utils.helpers import find_swing_points
                highs_idx, lows_idx = find_swing_points(ltf_df, lookback=3)
                if direction == "BUY":
                    for idx in reversed(lows_idx):
                        if ltf_df["low"].iloc[idx] < entry:
                            sl = ltf_df["low"].iloc[idx] - sl_buffer
                            break
                    if sl is None:
                        sl = entry - atr_val * 0.8
                else:
                    for idx in reversed(highs_idx):
                        if ltf_df["high"].iloc[idx] > entry:
                            sl = ltf_df["high"].iloc[idx] + sl_buffer
                            break
                    if sl is None:
                        sl = entry + atr_val * 0.8

        risk = abs(entry - sl)

        # --- TP: use structural target ---
        if getattr(self.profile, 'tp_fixed_pips', 0) > 0:
            tp_distance = self.profile.tp_fixed_pips * pip
            tp = entry + tp_distance if direction == "BUY" else entry - tp_distance
        else:
            tp = self._find_tp_target(direction, entry, sl, ltf_df, ctx, htf_patterns=htf_patterns)

        reward = abs(tp - entry)
        rr = reward / risk if risk > 0 else 0.0
        return entry, sl, tp, rr

    def _has_ltf_sweep_confirmation(self, ctx: MarketContext, ltf_df: pd.DataFrame) -> bool:
        if ltf_df is None or len(ltf_df) < 10:
            return False
        current_close = ltf_df["close"].iloc[-1]
        if ctx.htf_structure.trend == TrendDirection.BULLISH:
            recent_low = ltf_df["low"].iloc[-10:].min()
            recent_high = ltf_df["high"].iloc[-5:].max()
            sweep_occurred = recent_low < ltf_df["low"].iloc[-15:-10].min() if len(ltf_df) > 15 else False
            recovery = current_close > recent_high * 0.995
            return sweep_occurred and recovery
        elif ctx.htf_structure.trend == TrendDirection.BEARISH:
            recent_high = ltf_df["high"].iloc[-10:].max()
            recent_low = ltf_df["low"].iloc[-5:].min()
            sweep_occurred = recent_high > ltf_df["high"].iloc[-15:-10].max() if len(ltf_df) > 15 else False
            rejection = current_close < recent_low * 1.005
            return sweep_occurred and rejection
        return False

    def _has_multiframe_alignment(self, ctx: MarketContext) -> bool:
        htf_trend = ctx.htf_structure.trend
        ltf_trend = ctx.ltf_structure.trend
        if htf_trend == TrendDirection.RANGING:
            return False
        return htf_trend == ltf_trend

    def _best_pattern(self, patterns: List[Pattern], type_name: str, ltf_df: pd.DataFrame) -> Optional[Pattern]:
        if not patterns:
            return None
        filtered = [p for p in patterns if p.type.name == type_name]
        if not filtered:
            return None
        current_price = ltf_df["close"].iloc[-1]
        atr_now = (ltf_df["high"].iloc[-14:] - ltf_df["low"].iloc[-14:]).mean()
        viable = [p for p in filtered if abs(p.mid - current_price) <= atr_now * 4]
        if not viable:
            return None
        recent = [p for p in viable if abs(p.index - len(ltf_df) + 1) <= 10]
        if not recent:
            return None
        return max(recent, key=lambda p: p.index)

    def _check_triple_confluence(self, ltf_patterns: dict, direction: str) -> bool:
        """Triple Confluence: OB + FVG + Liquidity all present within proximity."""
        ob_key = "OB_BULLISH" if direction == "BUY" else "OB_BEARISH"
        fvg_key = "FVG_BULLISH" if direction == "BUY" else "FVG_BEARISH"
        sweep_key = "SWEEP_BULLISH" if direction == "BUY" else "SWEEP_BEARISH"
        has_ob = any(p.type.name == ob_key for p in ltf_patterns.get("ob", []))
        has_fvg = any(p.type.name == fvg_key for p in ltf_patterns.get("fvg", []))
        has_sweep = any(p.type.name == sweep_key for p in ltf_patterns.get("sweep", []))
        return has_ob and has_fvg and has_sweep

    def _check_wyckoff_phase_d(self, ltf_patterns: dict, direction: str) -> Optional[Pattern]:
        """Check if we're at LPS/LPSY in Wyckoff Phase D. Returns the pattern if found."""
        target = "SOS_BULLISH" if direction == "BUY" else "SOW_BEARISH"
        for p in ltf_patterns.get("wyckoff_phase_d", []):
            if p.type.name == target:
                return p
        return None

    def _check_wyckoff_sos_volume(self, ltf_patterns: dict, direction: str) -> bool:
        """SOS/SOW confirmed with volume (already in detector output)."""
        target = "SOS_BULLISH" if direction == "BUY" else "SOW_BEARISH"
        return any(p.type.name == target for p in ltf_patterns.get("wyckoff_phase_d", []))

    def _check_fvg_fresh(self, ltf_patterns: dict, ltf_df: pd.DataFrame, direction: str) -> bool:
        """FVG < 50% mitigated."""
        fvg_key = "FVG_BULLISH" if direction == "BUY" else "FVG_BEARISH"
        for p in ltf_patterns.get("fvg", []):
            if p.type.name == fvg_key:
                pct = self.detector._fvg_mitigation_pct(ltf_df, p)
                if pct < 0.5:
                    return True
        return False

    def _check_sweep_wick(self, ltf_patterns: dict, direction: str) -> bool:
        """Sweep or Spring/UTAD detected (confirms wick manipulation)."""
        sweep_key = "SWEEP_BULLISH" if direction == "BUY" else "SWEEP_BEARISH"
        has_sweep = any(p.type.name == sweep_key for p in ltf_patterns.get("sweep", []))
        wyckoff_key = PatternType.SPRING_BULLISH if direction == "BUY" else PatternType.UTAD_BEARISH
        has_wyckoff = any(p.type == wyckoff_key for p in ltf_patterns.get("wyckoff", []))
        return has_sweep or has_wyckoff

    def _check_body_close_valid(self, ctx: MarketContext, direction: str) -> bool:
        """Body close validity: HTF trend aligned and no recent body outside range."""
        if direction == "BUY" and ctx.htf_structure.trend == TrendDirection.BEARISH:
            return False
        if direction == "SELL" and ctx.htf_structure.trend == TrendDirection.BULLISH:
            return False
        return True

    def _check_body_close_invalid(self, ltf_df: pd.DataFrame, ltf_patterns: dict, direction: str) -> bool:
        """Check if any Spring/UTAD in patterns closed with body outside range."""
        for p in ltf_patterns.get("wyckoff", []):
            if self.detector.check_body_close_invalidation(ltf_df, [p]):
                return True
        return False

    def _has_sweep_for_direction(self, ltf_patterns: dict, direction: str) -> bool:
        """Check if there's a sweep in the direction of the trade.
        Also checks Wyckoff Spring/UTAD as sweep confirmation."""
        sweep_key = "SWEEP_BULLISH" if direction == "BUY" else "SWEEP_BEARISH"
        has_sweep = any(p.type.name == sweep_key for p in ltf_patterns.get("sweep", []))
        has_wyckoff = any(
            p.type == (PatternType.SPRING_BULLISH if direction == "BUY" else PatternType.UTAD_BEARISH)
            for p in ltf_patterns.get("wyckoff", [])
        )
        return has_sweep or has_wyckoff

    def _check_fvg_burned(self, ltf_patterns: dict, ltf_df: pd.DataFrame, direction: str) -> bool:
        """Check if FVG is > 50% mitigated (burned)."""
        fvg_key = "FVG_BULLISH" if direction == "BUY" else "FVG_BEARISH"
        for p in ltf_patterns.get("fvg", []):
            if p.type.name == fvg_key:
                pct = self.detector._fvg_mitigation_pct(ltf_df, p)
                if pct > 0.5:
                    return True
        return False

    def _check_ob_mitigated(self, ob: Optional[Pattern], ltf_df: pd.DataFrame, direction: str) -> bool:
        """Order Block is mitigated if price broke past its outer boundary since detection."""
        if ob is None:
            return False
        after = ltf_df.iloc[ob.index + 2:]
        if after.empty:
            return False
        if direction == "BUY":
            return after["low"].min() <= ob.high
        else:
            return after["high"].max() >= ob.low

    def _check_breaker_mitigated(self, breaker: Optional[Pattern], ltf_df: pd.DataFrame, direction: str) -> bool:
        """Breaker is mitigated if price retook the broken level (closed back past mid)."""
        if breaker is None:
            return False
        after = ltf_df.iloc[breaker.index + 1:]
        if after.empty:
            return False
        level = breaker.mid
        if direction == "BUY":
            return after["close"].iloc[-1] < level
        else:
            return after["close"].iloc[-1] > level

    def _check_lps_mitigated(self, pattern: Optional[Pattern], ltf_df: pd.DataFrame, direction: str) -> bool:
        """Wyckoff Phase D LPS/LPSY is mitigated if price broke past it."""
        if pattern is None:
            return False
        lps_level = pattern.mid
        current = ltf_df["close"].iloc[-1]
        if direction == "BUY":
            return current < lps_level
        else:
            return current > lps_level

    def _check_spring_body_close(self, ltf_patterns: dict, direction: str) -> bool:
        """Check if Spring/UTAD has body close invalidation."""
        for p in ltf_patterns.get("wyckoff", []):
            if p.type == PatternType.SPRING_BULLISH and direction == "BUY":
                return True
            if p.type == PatternType.UTAD_BEARISH and direction == "SELL":
                return True
        return False

    def _price_grid_alignment(self, price: float) -> tuple:
        """Verifica alineación con parrilla institucional (sentimales para XAUUSD/XAGUSD).
        Returns: (aligned: bool, best_level: float, distance_pips: float)"""
        symbol = self.profile.symbol
        if "XAU" not in symbol.upper() and "XAG" not in symbol.upper():
            return True, price, 0.0
        pip = self.pip
        cents = round((price - int(price)) * 100)
        if "XAU" in symbol.upper():
            sentimales = [0, 25, 50, 75]
            distances = [min(abs(cents - s), abs(cents - s + 100), abs(cents - s - 100)) for s in sentimales]
        else:
            sentimales = [0, 25, 50, 75]
            distances = [abs(cents - s) for s in sentimales]
        best_dist = min(distances)
        best_idx = distances.index(best_dist)
        best_level = int(price) + sentimales[best_idx] / 100
        return best_dist <= 3, best_level, best_dist * pip

    def _check_price_establishment(self, ltf_df: pd.DataFrame, level: float, direction: str) -> dict:
        """Price Establishment: mechas barren el nivel pero cuerpos respetan (manual §20.3).
        Usa count_wick_rejections() del analyzer para bonus escalable."""
        result = self.analyzer.count_wick_rejections(ltf_df, level, direction, lookback=8, min_rejections=2)
        return result

    def _check_trb_retest(self, ltf_df: pd.DataFrame, trb: dict, direction: str) -> bool:
        """Verifica si el precio está haciendo retest del rango TRB roto (Fase 4 del modelo)."""
        if not trb.get("fake_breakout"):
            return False
        current_price = ltf_df["close"].iloc[-1]
        range_high = trb.get("range_high")
        range_low = trb.get("range_low")
        if range_high is None or range_low is None:
            return False
        pip = self.pip
        tolerance = pip * 8
        if direction == "SELL":
            if abs(current_price - range_low) <= tolerance and current_price > range_low:
                return True
        else:
            if abs(current_price - range_high) <= tolerance and current_price < range_high:
                return True
        return False

    def _classify_sweep_liquidity(self, sweep_level: float, direction: str, htf_df: pd.DataFrame) -> str:
        if htf_df is None or len(htf_df) < 20:
            return "INTERNAL"
        from src.utils.helpers import pip_size, find_swing_points
        tolerance = pip_size(self.profile.symbol) * 3
        htf_highs, htf_lows = find_swing_points(htf_df, lookback=5)
        if direction == "BUY":
            for li in htf_lows:
                if abs(sweep_level - htf_df["low"].iloc[li]) <= tolerance:
                    return "EXTERNAL"
        else:
            for hi in htf_highs:
                if abs(sweep_level - htf_df["high"].iloc[hi]) <= tolerance:
                    return "EXTERNAL"
        return "INTERNAL"

    def _check_psychological_price(self, ltf_df: pd.DataFrame, direction: str) -> bool:
        """Check if current price aligns with institutional psychological prices.
        XAUUSD scalping: .25, .50, .75, .00 (sentimales)
        Forex: .25, .50, .75, .00 (cuartetos y medios)
        Indices: 250, 500, 750, 000"""
        if ltf_df is None or len(ltf_df) < 1:
            return False
        price = ltf_df["close"].iloc[-1]
        symbol = self.profile.symbol
        if "XAU" in symbol or "XAG" in symbol:
            cents = round((price - int(price)) * 100)
            sentimales = [0, 25, 50, 75]
            distances = [min(abs(cents - s), abs(cents - s + 100), abs(cents - s - 100)) for s in sentimales]
        else:
            cents = round((price - int(price)) * 100)
            sentimales = [0, 25, 50, 75]
            distances = [abs(cents - s) for s in sentimales]
        return min(distances) <= 3

    def _compute_pattern_confluence(self, ltf_patterns: dict, direction: str, ltf_df: pd.DataFrame) -> float:
        """Group all detected patterns by price zone (0.5 ATR clusters) and compute
        a net directional contribution. Only one pattern per zone per direction counts.
        Returns a multiplier (0.5-1.0) to penalize overlapping/conflicting patterns.
        """
        atr_val = (ltf_df["high"].iloc[-14:] - ltf_df["low"].iloc[-14:]).mean()
        if atr_val == 0:
            return 1.0
        cluster_radius = atr_val * 0.5
        direction_tag = "BULLISH" if direction == "BUY" else "BEARISH"
        opp_tag = "BEARISH" if direction == "BUY" else "BULLISH"
        own_prices = []
        opp_prices = []
        for key in ("fvg", "ob", "breaker", "sweep", "cycle", "wyckoff", "void_scalp",
                     "bos_zone_retest", "interval_points", "price_interaction",
                     "harmonic_cycle", "pressure_zones"):
            for p in ltf_patterns.get(key, []):
                if direction_tag in p.type.name:
                    own_prices.append(p.mid)
                elif opp_tag in p.type.name:
                    opp_prices.append(p.mid)

        if not own_prices:
            return 1.0

        def count_clusters(prices):
            if not prices:
                return 0
            sorted_p = sorted(prices)
            clusters = 1
            anchor = sorted_p[0]
            for p in sorted_p[1:]:
                if p - anchor > cluster_radius:
                    clusters += 1
                    anchor = p
            return clusters

        own_clusters = count_clusters(own_prices)
        own_count = len(own_prices)

        overlap_count = 0
        if opp_prices:
            for op in opp_prices:
                for dp in own_prices:
                    if abs(op - dp) <= cluster_radius:
                        overlap_count += 1
                        break

        overlap_ratio = min(1.0, overlap_count / max(1, own_count))
        density = min(1.0, own_count / max(1, own_clusters) / 3.0) if own_clusters > 0 else 1.0

        multiplier = 1.0 - (density * 0.15) - (overlap_ratio * 0.15)
        return max(0.70, multiplier)

    def _check_dxy_correlation(self, dxy_df: pd.DataFrame, direction: str) -> dict:
        """DXY correlation: XAUUSD/XAGUSD are inversely correlated with USD index.
        DXY bullish = USD strong → SELL bonus / BUY penalty.
        DXY bearish = USD weak → BUY bonus / SELL penalty."""
        if dxy_df is None or len(dxy_df) < 25:
            return {"score": 0, "aligned": False, "trend": "nodata"}
        sma20 = dxy_df["close"].iloc[-20:].mean()
        current = dxy_df["close"].iloc[-1]
        pct_diff = (current - sma20) / sma20
        if abs(pct_diff) < 0.0005:
            return {"score": 0, "aligned": False, "trend": "neutral"}
        dxy_bullish = pct_diff > 0
        if direction == "BUY":
            if not dxy_bullish:
                return {"score": self.weights.dxy_aligned_bonus, "aligned": True, "trend": "bearish"}
            return {"score": self.weights.dxy_conflict_penalty, "aligned": False, "trend": "bullish"}
        else:
            if dxy_bullish:
                return {"score": self.weights.dxy_aligned_bonus, "aligned": True, "trend": "bullish"}
            return {"score": self.weights.dxy_conflict_penalty, "aligned": False, "trend": "bearish"}

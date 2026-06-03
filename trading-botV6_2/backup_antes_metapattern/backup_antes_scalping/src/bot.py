"""SMC Trading Bot - Main Entry Point
Automated trading bot for XAUUSD.. using Smart Money Concepts scoring strategy
"""
import json
import signal
import sys
import asyncio
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from uuid import uuid4
from loguru import logger

_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))
del _proj_root

from src.adapters.mt5_client import MT5Client
from src.core.multi_timeframe import MultiTimeframeFetcher
from src.core.strategy_engine import StrategyEngine, TradingSignal, ScoringConfig
from src.core.state_persistence import StatePersistence
from src.core.news_calendar import NewsCalendar
from src.core.regime_detector import RegimeDetector, RegimeContext
from src.core.regime_trainer import RegimeTrainer
from src.core.continuous_decision import ContinuousDecider, ContinuousDecision
from src.core.market_memory import MarketMemory
from src.core.session_profiler import SessionProfiler
from src.core.meta_learner import MetaLearner, TradeRecord
from src.core.context_oracle import ContextOracle
from src.core.kelly_risk import KellyRiskManager
from src.risk.fixed_risk_manager import FixedRiskManager, RiskConfig, calculate_atr
from src.risk.dynamic_var_risk import DynamicVaRRiskManager
from src.risk.portfolio_risk import PortfolioRiskManager
from src.executor.order_executor import OrderExecutor
from src.executor.order_type_selector import OrderTypeSelector
from src.executor.order_types import OrderType
from src.executor.trailing_stop import TrailingStopManager, TrailingConfig
from src.core.timeframe_optimizer import TimeframeOptimizer
from src.scheduler.timeframe_scheduler import TimeframeScheduler
from src.utils.helpers import pip_size, is_in_session, atr as _atr_fn
from src.core.regime_detector import RegimeType
from src.core.volatility_scaler import VolatilityScaler
from src.core.correlation_engine import CorrelationEngine
from src.core.order_flow import OrderFlowEngine, OrderFlowConfig
from src.core.adaptive_cooldown import AdaptiveCooldownEngine, AdaptiveCooldownConfig
from src.core.market_map import MarketMap, MarketMapResult
from src.core.liquidity_mapper import LiquidityMapper
from src.core.concept_tracker import ConceptPerformanceTracker
from src.core.md_concepts import MDConcept, MDDetection
from src.optimization.genetic_optimizer import GeneticOptimizer, WalkForwardOptimizer
from src.optimization.fitness import evaluate_fitness
from src.optimization.parameter_space import EXPERT_SPACE
import pandas as pd
import os
from src.core.failure_analyzer import FailureAnalyzer
from src.core.adaptive_thresholds import AdaptiveThresholdEngine
from src.core.circuit_breakers import CircuitBreakers
from src.core.symbol_evaluator import SymbolEvaluator
from src.core.market_hours import MarketHoursController


@dataclass
class SymbolProfile:
    symbol: str
    htf_context: str = "H4"
    htf_secondary: str = "H1"
    ltf_trigger: str = "M15"
    ltf_refine: str = "M5"
    min_retracement: float = 0.50
    fvg_entry_level: float = 0.50
    sl_buffer_pips: float = 0.5
    sl_fixed_pips: float = 0.0
    sl_min_pips: float = 10.0
    sl_max_pips: float = 15.0
    min_rr_ratio: float = 3.0
    tp_fixed_pips: float = 0.0
    risk_per_trade_pct: float = 2.0
    max_concurrent_trades: int = 1
    max_volume: float = 0
    max_daily_loss_pct: float = 6.0
    max_daily_trades: int = 10
    allowed_sessions: List[Tuple[int, int]] = field(default_factory=list)
    cooldown_bars_m1: int = 5
    scale_entries: int = 5
    scale_close_at_tp_pct: float = 0.5
    scale_close_ratio: float = 0.5


@dataclass
class StrategyParams:
    consolidation_max_atr_ratio: float = 0.6
    consolidation_min_bars: int = 8
    expansion_tick_acceleration: float = 1.8
    gap_detection_pips: float = 8.0
    macro_event_filter: bool = True
    min_retracement_level: float = 0.50
    breakout_validation: str = "BodyClose"
    fvg_min_size_atr_ratio: float = 0.3
    fvg_entry_level: float = 0.50
    void_min_size_pips: float = 5.0
    pivot_length: int = 10
    wyckoff_min_phase_bars: int = 12
    sequence_length: int = 3
    news_filter_active: bool = True
    news_buffer_minutes: int = 30
    min_reversal_score: float = 65.0
    min_net_score: float = 2.0
    min_stop_distance_atr: float = 0.3
    min_rr_ratio: float = 3.0
    dxy_symbol: str = "DXY"



def setup_logging():
    logger.remove()
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.add(log_dir / "trading_bot_{time}.log", rotation="00:00", retention="7 days",
              level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    logger.add(sys.stderr, level="INFO")


def load_config(config_dir: Path) -> dict:
    with open(config_dir / "broker.json") as f:
        broker = json.load(f)
    with open(config_dir / "strategy.json") as f:
        strategy = json.load(f)
    with open(config_dir / "risk.json") as f:
        risk = json.load(f)
    return {"broker": broker, "strategy": strategy, "risk": risk}


def candles_to_dataframe(candles: list) -> pd.DataFrame:
    records = []
    for c in candles:
        records.append({
            "time": c.timestamp,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        })
    df = pd.DataFrame(records)
    df["time"] = pd.to_datetime(df["time"])
    return df


def build_symbol_profile(cfg: dict) -> SymbolProfile:
    return SymbolProfile(
        symbol=cfg.get("symbol", "XAUUSD.."),
        htf_context=cfg.get("htf_context", "H4"),
        htf_secondary=cfg.get("htf_secondary", "H1"),
        ltf_trigger=cfg.get("ltf_trigger", "M15"),
        ltf_refine=cfg.get("ltf_refine", "M5"),
        min_retracement=cfg.get("min_retracement", 0.5),
        fvg_entry_level=cfg.get("fvg_entry_level", 0.5),
        sl_buffer_pips=cfg.get("sl_buffer_pips", 3.0),
        sl_fixed_pips=cfg.get("sl_fixed_pips", 0.0),
        min_rr_ratio=cfg.get("min_rr_ratio", 3.0),
        tp_fixed_pips=cfg.get("tp_fixed_pips", 0.0),
        risk_per_trade_pct=cfg.get("risk_per_trade_pct", 2.0),
        max_concurrent_trades=cfg.get("max_concurrent_trades", 1),
        max_volume=cfg.get("max_volume", 0),
        max_daily_loss_pct=cfg.get("max_daily_loss_pct", 6.0),
        max_daily_trades=cfg.get("max_daily_trades", 10),
        allowed_sessions=[tuple(s) for s in cfg.get("allowed_sessions", [])],
        sl_min_pips=cfg.get("sl_min_pips", 10.0),
        sl_max_pips=cfg.get("sl_max_pips", 15.0),
        cooldown_bars_m1=cfg.get("cooldown_bars_m1", 5),
        scale_entries=cfg.get("scale_entries", 5),
        scale_close_at_tp_pct=cfg.get("scale_close_at_tp_pct", 0.5),
        scale_close_ratio=cfg.get("scale_close_ratio", 0.5),
    )


def build_strategy_params(cfg: dict) -> StrategyParams:
    return StrategyParams(
        consolidation_max_atr_ratio=cfg.get("consolidation_max_atr_ratio", 0.6),
        consolidation_min_bars=cfg.get("consolidation_min_bars", 8),
        expansion_tick_acceleration=cfg.get("expansion_tick_acceleration", 1.8),
        gap_detection_pips=cfg.get("gap_detection_pips", 8.0),
        macro_event_filter=cfg.get("macro_event_filter", True),
        min_retracement_level=cfg.get("min_retracement_level", 0.5),
        breakout_validation=cfg.get("breakout_validation", "BodyClose"),
        fvg_min_size_atr_ratio=cfg.get("fvg_min_size_atr_ratio", 0.3),
        fvg_entry_level=cfg.get("fvg_entry_level", 0.5),
        void_min_size_pips=cfg.get("void_min_size_pips", 5.0),
        pivot_length=cfg.get("pivot_length", 10),
        wyckoff_min_phase_bars=cfg.get("wyckoff_min_phase_bars", 12),
        sequence_length=cfg.get("sequence_length", 3),
        news_filter_active=cfg.get("news_filter_active", True),
        news_buffer_minutes=cfg.get("news_buffer_minutes", 30),
        min_reversal_score=cfg.get("min_reversal_score", 75.0),
        min_net_score=cfg.get("min_net_score", 2.0),
        min_stop_distance_atr=cfg.get("min_stop_distance_atr", 0.3),
        min_rr_ratio=cfg.get("min_rr_ratio", 3.0),
        dxy_symbol=cfg.get("dxy_symbol", "DXY"),
    )


def build_scoring_config(cfg: dict) -> ScoringConfig:
    return ScoringConfig(
        htf_trend_aligned=cfg.get("htf_trend_aligned", 20.0),
        in_discount_premium_zone=cfg.get("in_discount_premium_zone", 15.0),
        valid_market_structure=cfg.get("valid_market_structure", 10.0),
        fvg_detected=cfg.get("fvg_detected", 18.0),
        order_block_valid=cfg.get("order_block_valid", 15.0),
        breaker_retest=cfg.get("breaker_retest", 12.0),
        slip_memory_present=cfg.get("slip_memory_present", 10.0),
        equidad_sweep_confirmed=cfg.get("equidad_sweep_confirmed", 12.0),
        eslabon_breakout=cfg.get("eslabon_breakout", 15.0),
        wyckoff_phase_c_spring=cfg.get("wyckoff_phase_c_spring", 20.0),
        wyckoff_phase_c_utad=cfg.get("wyckoff_phase_c_utad", 20.0),
        wyckoff_phase_d_confirmed=cfg.get("wyckoff_phase_d_confirmed", 15.0),
        ltf_bos_with_body=cfg.get("ltf_bos_with_body", 12.0),
        liquidity_sweep_ltf=cfg.get("liquidity_sweep_ltf", 10.0),
        in_active_session=cfg.get("in_active_session", 8.0),
        killzone_london_open=cfg.get("killzone_london_open", 12.0),
        killzone_ny_open=cfg.get("killzone_ny_open", 15.0),
        killzone_london_ny_overlap=cfg.get("killzone_london_ny_overlap", 18.0),
        killzone_asian=cfg.get("killzone_asian", 5.0),
        no_news_event=cfg.get("no_news_event", 5.0),
        regime_expansion=cfg.get("regime_expansion", 8.0),
        regime_not_accumulation=cfg.get("regime_not_accumulation", 5.0),
        ltf_sweep_confirmation=cfg.get("ltf_sweep_confirmation", 15.0),
        multiframe_alignment=cfg.get("multiframe_alignment", 12.0),
        void_scalp_confirmed=cfg.get("void_scalp_confirmed", 18.0),

        triple_confluence=cfg.get("triple_confluence", 18.0),
        wyckoff_sos_volume=cfg.get("wyckoff_sos_volume", 15.0),
        wyckoff_phase_d_lps=cfg.get("wyckoff_phase_d_lps", 12.0),
        fvg_fresh_mitigation=cfg.get("fvg_fresh_mitigation", 8.0),
        sweep_spring_wick=cfg.get("sweep_spring_wick", 10.0),
        body_close_valid=cfg.get("body_close_valid", 10.0),
        price_grid_aligned=cfg.get("price_grid_aligned", 8.0),
        price_establishment=cfg.get("price_establishment", 10.0),
        news_high_impact_penalty=cfg.get("news_high_impact_penalty", -200.0),

        wick_rejection_bonus=cfg.get("wick_rejection_bonus", 5.0),
        trb_manipulation_detected=cfg.get("trb_manipulation_detected", 12.0),
        trb_displacement=cfg.get("trb_displacement", 10.0),
        trb_retest=cfg.get("trb_retest", 8.0),

        body_close_invalid=cfg.get("body_close_invalid", -50.0),
        no_sweep_detected=cfg.get("no_sweep_detected", -30.0),
        fvg_burned_over_50=cfg.get("fvg_burned_over_50", -20.0),
        spring_body_close=cfg.get("spring_body_close", -40.0),

        bos_zone_retest=cfg.get("bos_zone_retest", 20.0),
        price_establishment_bonus=cfg.get("price_establishment_bonus", 12.0),
        third_movement_ready=cfg.get("third_movement_ready", 15.0),
        micro_retracement_50=cfg.get("micro_retracement_50", 10.0),
        psychological_price_aligned=cfg.get("psychological_price_aligned", 8.0),

        ob_mitigated_penalty=cfg.get("ob_mitigated_penalty", -15.0),
        breaker_mitigated_penalty=cfg.get("breaker_mitigated_penalty", -12.0),
        lps_mitigated_penalty=cfg.get("lps_mitigated_penalty", -12.0),

        vsa_volume_confirmation=cfg.get("vsa_volume_confirmation", 12.0),
        vsa_climax_penalty=cfg.get("vsa_climax_penalty", -20.0),
        vsa_absorption_bonus=cfg.get("vsa_absorption_bonus", 15.0),
        vsa_low_volume_pullback=cfg.get("vsa_low_volume_pullback", 10.0),
        vsa_volume_divergence=cfg.get("vsa_volume_divergence", -15.0),
        vsa_no_demand_supply=cfg.get("vsa_no_demand_supply", -10.0),

        dxy_aligned_bonus=cfg.get("dxy_aligned_bonus", 12.0),
        dxy_conflict_penalty=cfg.get("dxy_conflict_penalty", -15.0),

        interval_point_bonus=cfg.get("interval_point_bonus", 12.0),
        price_interaction_bonus=cfg.get("price_interaction_bonus", 14.0),
        harmonic_cycle_aligned=cfg.get("harmonic_cycle_aligned", 15.0),
        pause_continuation_bonus=cfg.get("pause_continuation_bonus", 10.0),
        retracement_penalty=cfg.get("retracement_penalty", -15.0),
        order_flow_regular_bonus=cfg.get("order_flow_regular_bonus", 8.0),
        order_flow_irregular_penalty=cfg.get("order_flow_irregular_penalty", -8.0),
        pressure_zone_bonus=cfg.get("pressure_zone_bonus", 14.0),
        breaker_3_touch_limit=cfg.get("breaker_3_touch_limit", -25.0),
        price_trend_aligned=cfg.get("price_trend_aligned", 15.0),
        htf_misalignment_penalty=cfg.get("htf_misalignment_penalty", -5.0),

        divergence_regular_bonus=cfg.get("divergence_regular_bonus", 18.0),
        divergence_hidden_bonus=cfg.get("divergence_hidden_bonus", 10.0),
        divergence_penalty=cfg.get("divergence_penalty", -12.0),
        order_flow_imbalance_bonus=cfg.get("order_flow_imbalance_bonus", 12.0),
        order_flow_divergence_bonus=cfg.get("order_flow_divergence_bonus", 15.0),
        order_flow_exhaustion_penalty=cfg.get("order_flow_exhaustion_penalty", -10.0),
        order_flow_absorption_bonus=cfg.get("order_flow_absorption_bonus", 8.0),
        volume_profile_poc_discount_bonus=cfg.get("volume_profile_poc_discount_bonus", 12.0),
        volume_profile_lvn_bonus=cfg.get("volume_profile_lvn_bonus", 10.0),
        volume_profile_compression_bonus=cfg.get("volume_profile_compression_bonus", 8.0),
        volume_profile_poc_shift_bonus=cfg.get("volume_profile_poc_shift_bonus", 10.0),
        volume_profile_poc_shift_penalty=cfg.get("volume_profile_poc_shift_penalty", -10.0),
        dxy_enhanced_breakdown_penalty=cfg.get("dxy_enhanced_breakdown_penalty", -15.0),
        dxy_divergence_bonus=cfg.get("dxy_divergence_bonus", 14.0),
    )


class TradingBot:
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.config = load_config(config_dir)
        setup_logging()

        broker_config = self.config["broker"]["mt5"]
        self.mt5 = MT5Client(
            login=os.environ.get("MT5_LOGIN") or broker_config.get("login"),
            password=os.environ.get("MT5_PASSWORD") or broker_config.get("password"),
            server=os.environ.get("MT5_SERVER") or broker_config.get("server"),
            path=broker_config.get("path"),
        )

        self.fetcher = MultiTimeframeFetcher(self.mt5)
        str_cfg = self.config["strategy"]
        raw_symbols = str_cfg.get("active_symbols", ["XAUUSD.."])
        self.active_symbols = []
        self.params = build_strategy_params(str_cfg.get("params", {}))
        scoring_cfg = build_scoring_config(str_cfg.get("scoring", {}))
        self.min_score = str_cfg.get("min_score_to_trade", 65.0)
        self.high_confidence_score = str_cfg.get("high_confidence_score", 85.0)

        news_buffer = self.params.news_buffer_minutes if self.params.news_filter_active else 0
        self.news_calendar = NewsCalendar(config_dir=config_dir, buffer_minutes=news_buffer)

        self.min_reentry_score = str_cfg.get("min_reentry_score", 70.0)

        self.mt5.connect()
        self._sym_map = {}
        resolved_active = []
        for sym in raw_symbols:
            base = sym.rstrip("m.").split("_")[0]
            resolved = self.mt5.resolve_symbol(base)
            if resolved:
                self._sym_map[sym] = resolved
                resolved_active.append(resolved)
                if resolved != sym:
                    logger.info(f"Symbol {sym} → {resolved}")
            else:
                logger.error(f"Symbol {sym} no disponible, omitiendo")
        if not resolved_active:
            logger.error("No hay símbolos disponibles. Saliendo.")
            return
        self.active_symbols = resolved_active

        self.symbols = {}
        for cfg_name in raw_symbols:
            resolved = self._sym_map.get(cfg_name)
            if not resolved:
                continue
            sym_cfg = str_cfg["symbols"].get(cfg_name, {})
            profile = build_symbol_profile(sym_cfg)
            profile.symbol = resolved
            engine = StrategyEngine(
                profile=profile, params=self.params,
                weights=scoring_cfg, min_score=self.min_score,
                high_confidence_score=self.high_confidence_score,
                min_net_score=self.params.min_net_score,
            )
            self.symbols[resolved] = {
                "profile": profile,
                "engine": engine,
                "last_trade_time": None,
                "pip_value": pip_size(resolved),
            }

        risk_config = RiskConfig(
            risk_per_trade=self.config["risk"]["risk_per_trade"],
            max_daily_loss=self.config["risk"]["max_daily_loss"],
            max_positions=self.config["risk"]["max_positions"],
            atr_multiplier_sl=self.config["risk"]["atr_multiplier_sl"],
            atr_multiplier_tp=self.config["risk"]["atr_multiplier_tp"],
            min_reward_risk_ratio=self.config["risk"]["min_reward_risk_ratio"],
        )

        account_info = self.mt5.get_account_info()
        balance = account_info["balance"] if account_info else 1000.0

        self.risk_manager = FixedRiskManager(risk_config, balance)
        self.volatility_scaler = VolatilityScaler(expert_mode=True)
        self.correlation_engine = CorrelationEngine()
        self.portfolio_risk = PortfolioRiskManager(
            correlation_engine=self.correlation_engine,
            max_portfolio_risk_pct=self.config.get("risk", {}).get("max_portfolio_risk_pct", 0.06),
            max_correlation_for_full_size=self.config.get("risk", {}).get("max_correlation_for_full_size", 0.7),
            max_net_directional_exposure_pct=self.config.get("risk", {}).get("max_net_directional_exposure_pct", 0.10),
            max_gross_exposure_pct=self.config.get("risk", {}).get("max_gross_exposure_pct", 0.20),
            portfolio_drawdown_limit_pct=self.config.get("risk", {}).get("portfolio_drawdown_limit_pct", 0.15),
            max_concurrent_setups=self.config.get("risk", {}).get("max_concurrent_setups", 3),
            balance=balance,
        )
        self.genetic_optimizer = GeneticOptimizer(expert_mode=True)
        self._optimization_result = None
        self.executor = OrderExecutor(self.mt5)
        scheduler_sym = self.active_symbols[0]
        for s in self.active_symbols:
            if "BTC" in s.upper():
                scheduler_sym = s
                break
        self.scheduler = TimeframeScheduler(self.fetcher, scheduler_sym)
        logger.info(f"Scheduler usando {scheduler_sym} como referencia")

        project_root = Path(__file__).parent.parent
        db_path = project_root / "data" / "trading_state.db"
        db_path.parent.mkdir(exist_ok=True)
        self.state_persistence = StatePersistence(db_path)

        self.running = False
        self.position_states = {}
        self.pending_orders = {}
        self.batches: Dict[str, Dict[str, dict]] = {}
        self._pending_batches: Dict[str, Dict[str, dict]] = {}
        self._dxy_unavailable = False

        regime_params = str_cfg.get("regime_params", {})
        model_dir = Path(__file__).resolve().parent.parent / "data" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        self.regime_detectors: Dict[str, RegimeDetector] = {}
        for sym in self.active_symbols:
            self.regime_detectors[sym] = RegimeDetector(
                atr_period=regime_params.get("atr_period", 14),
                adx_period=regime_params.get("adx_period", 14),
                symbol=sym,
                model_dir=model_dir,
            )
        self._model_dir = model_dir
        self._model_mtimes: Dict[str, float] = {}
        self.continuous_decider = ContinuousDecider(base_risk_pct=risk_config.risk_per_trade)
        self.market_memory = MarketMemory(db_path=project_root / "data" / "market_memory.db")
        session_params = str_cfg.get("session_params", {})
        self.session_profiler = SessionProfiler(
            asian_volatility_reduction=session_params.get("asian_volatility_reduction", 0.7),
            london_peak_boost=session_params.get("london_peak_boost", 1.2),
            ny_overlap_boost=session_params.get("ny_overlap_boost", 1.3),
            close_session_penalty=session_params.get("close_session_penalty", 0.3),
        )
        self.meta_learner = MetaLearner(
            db_path=project_root / "data" / "meta_learning.db",
            base_weights=scoring_cfg,
        )
        self.failure_analyzer = FailureAnalyzer(project_root / "data" / "failure_analysis.db")
        self.adaptive_thresholds = AdaptiveThresholdEngine(
            project_root / "data" / "adaptive_thresholds.db"
        )
        self.circuit_breakers = CircuitBreakers(
            volatility_threshold=str_cfg.get("circuit_breakers", {}).get("volatility_threshold", 2.0),
            momentum_candles=str_cfg.get("circuit_breakers", {}).get("momentum_candles", 3),
            news_buffer_minutes=str_cfg.get("circuit_breakers", {}).get("news_buffer_minutes", 30.0),
        )
        project_root = Path(__file__).parent.parent
        self.context_oracle = ContextOracle(project_root / "data" / "context_memory.db")
        self.kelly_risk = KellyRiskManager(
            db_path=project_root / "data" / "kelly_state.db",
            initial_risk=self.config["risk"]["risk_per_trade"],
            max_risk=self.config["risk"].get("max_risk_per_trade", 0.04),
        )
        self.dynamic_var_risk = DynamicVaRRiskManager(
            initial_risk=self.config["risk"]["risk_per_trade"],
            max_risk=self.config["risk"].get("max_risk_per_trade", 0.04),
            min_risk=self.config["risk"].get("min_risk_per_trade", 0.005),
        )
        self.dynamic_var_risk.update_balance(balance)
        self._last_meta_analysis = time.time()
        self._meta_analysis_interval = 14400  # cada 4 horas
        self._signal_id_counter = 0
        self.order_selector = OrderTypeSelector(params=self.params)
        self.trailing_mgr = TrailingStopManager()
        self.tf_optimizer = TimeframeOptimizer()
        self.expert_mode = self.config["strategy"].get("mode", "DEMO") == "EXPERT"
        self.order_flow = OrderFlowEngine(OrderFlowConfig(expert_mode=self.expert_mode))
        if self.expert_mode and hasattr(self, 'mt5'):
            self.order_flow.set_mt5_client(self.mt5)

        self.concept_tracker = ConceptPerformanceTracker()
        self.market_maps: Dict[str, MarketMap] = {
            sym: MarketMap(sym, concept_tracker=self.concept_tracker)
            for sym in self.active_symbols
        }
        self._last_md_detections: Dict[str, List[MDDetection]] = {}

        self._last_regime: Dict[str, RegimeContext] = {}
        self._last_decision: Dict[str, ContinuousDecision] = {}

        self._last_atr: Dict[str, float] = {}
        self._atr_spike_threshold = 0.50
        self._last_dxy_trend: str = ""

        self._positions_cache: List[dict] = []
        self._cycle_cache: dict = {}

        self._adaptive_cooldown_enabled = False  # Desactivado para scalping
        self.adaptive_cooldown = AdaptiveCooldownEngine(
            AdaptiveCooldownConfig(expert_mode=self.expert_mode)
        )
        self._symbol_evaluator = SymbolEvaluator()

        self.hours_controller = MarketHoursController()
        if self.active_symbols:
            self.hours_controller.classify_all(set(self.symbols.keys()))
        self._last_market_hours_persist: float = 0.0
        self._market_hours_persist_interval: float = 300.0  # 5 min

    async def _initialize_state(self):
        await self.state_persistence.initialize()
        self.failure_analyzer.initialize()
        self.adaptive_thresholds.initialize()
        saved_state = await self.state_persistence.load_daily_state()
        self.risk_manager.daily_loss = saved_state.get("daily_loss", 0.0)
        open_positions = await self.state_persistence.load_open_positions()
        for pos in open_positions:
            entry_price = pos.get("entry_price", 0)
            tp_original = pos.get("tp_original", 0)
            is_long = (pos.get("direction", "").upper() == "BUY")
            self.position_states[pos["ticket"]] = {
                "symbol": pos.get("symbol", ""),
                "be_activated": pos.get("be_activated", False),
                "trail_activated": pos.get("trailing_activated", False),
                "original_sl": pos.get("sl_original", 0),
                "entry_price": entry_price,
                "is_long": is_long,
                "volume": pos.get("volume", 0),
                "managed_tp": tp_original,
                "tp_expanded": 0,
                "trail_stage": 0,
                "original_tp_distance": abs(tp_original - entry_price) if tp_original else 0,
                "last_profit": 0,
                "last_price": 0,
            }
            self.trailing_mgr.initialize_state(
                ticket=pos["ticket"], symbol=pos.get("symbol", ""),
                is_long=is_long, entry_price=entry_price,
                current_sl=pos.get("sl_original", 0),
                current_tp=tp_original,
                current_price=pos.get("price_current", entry_price),
                profit=0, volume=pos.get("volume", 0),
            )
            logger.info(f"Loaded position: ticket={pos['ticket']}, "
                        f"BE={pos['be_activated']}, Trail={pos['trailing_activated']}")

    async def _restore_market_hours(self):
        try:
            saved = await self.state_persistence.load_market_hours()
            if saved:
                self.hours_controller.restore_states(saved)
                logger.info(f"[MarketHours] Restored state for {len(saved)} symbols")
        except Exception as e:
            logger.warning(f"[MarketHours] Could not restore state: {e}")

    async def _save_state_periodic(self):
        await self.state_persistence.save_daily_state(
            daily_loss=self.risk_manager.daily_loss,
            trades_count=len(self.position_states),
        )

    def _train_regime_models(self):
        for sym in self.active_symbols:
            model_path = self._model_dir / f"regime_rf_{sym}.pkl"
            if model_path.exists():
                logger.info(f"[{sym}] ML regime model already exists, skipping training")
                continue
            logger.info(f"[{sym}] No ML regime model found, training from historical data...")
            try:
                import time as _time
                trainer = RegimeTrainer(self._model_dir)
                num_candles = 5000
                max_retries = 3
                df = None
                for attempt in range(max_retries):
                    try:
                        raw = self.mt5.get_candles(sym, timeframe=5, count=num_candles)
                        if raw and len(raw) >= 2000:
                            records = []
                            for c in raw:
                                records.append({
                                    "time": c.timestamp,
                                    "open": c.open,
                                    "high": c.high,
                                    "low": c.low,
                                    "close": c.close,
                                    "volume": c.volume,
                                })
                            df = pd.DataFrame(records)
                            df["time"] = pd.to_datetime(df["time"])
                            break
                        else:
                            logger.warning(f"[{sym}] Attempt {attempt + 1}: obtained {len(raw) if raw else 0}/{num_candles} candles, retrying...")
                            _time.sleep(2)
                    except Exception as e:
                        logger.warning(f"[{sym}] Attempt {attempt + 1} failed: {e}")
                        _time.sleep(2)
                if df is None or len(df) < 2000:
                    logger.warning(f"[{sym}] Insufficient data ({len(df) if df is not None else 0} candles), skipping ML training")
                    continue
                logger.info(f"[{sym}] Fetched {len(df)} M5 candles for training")
                result = trainer.train(df)
                if result.get("status") == "success":
                    trainer.save_model(sym)
                    acc = result.get("test_accuracy", 0)
                    logger.info(f"[{sym}] ML regime model trained: test_acc={acc:.1%}, "
                                f"samples={result.get('samples', 0)}")
                    self.regime_detectors[sym]._load_ml_model(self._model_dir)
                else:
                    logger.warning(f"[{sym}] ML training {result.get('status')}: {result.get('samples', 0)} samples")
            except Exception as e:
                logger.exception(f"[{sym}] Error training ML regime model: {e}")

    def _hot_reload_models(self):
        for sym in list(self.regime_detectors.keys()):
            path = self._model_dir / f"regime_rf_{sym}.pkl"
            if not path.exists():
                continue
            try:
                mtime = path.stat().st_mtime
                last = self._model_mtimes.get(sym, 0)
                if mtime > last:
                    if self.regime_detectors[sym]._load_ml_model(self._model_dir):
                        self._model_mtimes[sym] = mtime
                        logger.info(f"[{sym}] ML regime model hot-reloaded")
            except Exception as e:
                logger.debug(f"[{sym}] Model reload check failed: {e}")

    def _run_genetic_optimization(self):
        logger.info("Iniciando optimización genética en background...")
        t = threading.Thread(target=self._optimization_worker, daemon=True)
        t.start()

    def _optimization_worker(self):
        try:
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as _TO
            from src.optimization.fitness import evaluate_fitness as _eval_fit
            import pandas as _pd
            from src.core.multi_timeframe import TIMEFRAME_CODES as _TF_CODES

            df = None
            data_timeout = 45

            def _fetch_tf(tf_name: str):
                code = _TF_CODES.get(tf_name)
                if code is None:
                    return None
                rates = mt5.copy_rates_from_pos(
                    self.active_symbols[0], code, 0, 500
                )
                if rates is None or len(rates) == 0:
                    return None
                records = []
                for r in rates:
                    records.append({
                        "time": _pd.Timestamp.fromtimestamp(r["time"], tz="UTC"),
                        "open": float(r["open"]),
                        "high": float(r["high"]),
                        "low": float(r["low"]),
                        "close": float(r["close"]),
                        "volume": float(r["tick_volume"]),
                    })
                d = _pd.DataFrame(records)
                d["time"] = _pd.to_datetime(d["time"])
                return d

            import MetaTrader5 as mt5
            for tf_name in ["5min", "15min", "1min"]:
                with ThreadPoolExecutor(max_workers=1) as _ex:
                    _f = _ex.submit(_fetch_tf, tf_name)
                    try:
                        d = _f.result(timeout=data_timeout)
                        if d is not None and len(d) >= 500:
                            df = d
                            logger.info(f"Opt data {tf_name}: {len(d)} velas")
                            break
                    except _TO:
                        logger.warning(f"Opt timeout ({data_timeout}s) {tf_name}")
                    except Exception as _e:
                        logger.warning(f"Opt error {tf_name}: {_e}")

            if df is None or len(df) < 500:
                logger.warning("Datos insuficientes para optimización, saltando")
                return

            def _fit(params):
                return _eval_fit(params, df)

            self.genetic_optimizer.set_fitness_fn(_fit)
            result = self.genetic_optimizer.run()
            self._optimization_result = result
            best = result["best_params"]
            logger.info(f"Optimización completada: fitness={result['best_fitness']:.4f}")
            logger.info(f"Mejores params: risk={best.get('risk_per_trade', 'N/A'):.4f} "
                        f"SL={best.get('atr_multiplier_sl', 'N/A'):.2f} "
                        f"TP={best.get('atr_multiplier_tp', 'N/A'):.2f} "
                        f"min_score={best.get('min_score_to_trade', 'N/A')}")

            if self.expert_mode:
                t = threading.Thread(target=self._background_optimization, daemon=True)
                t.start()
        except Exception as e:
            logger.exception(f"Error en optimización genética: {e}")

    def _background_optimization(self):
        import time as _time
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as _TO
        import pandas as _pd
        import MetaTrader5 as _mt5
        from src.optimization.fitness import evaluate_fitness as _eval_fit
        from src.optimization.genetic_optimizer import WalkForwardOptimizer
        from src.core.multi_timeframe import TIMEFRAME_CODES as _TF_CODES
        _time.sleep(3600)
        while self.running:
            logger.info("Ejecutando re-optimización en background...")
            try:
                df = None
                for _tf in ["5min", "15min", "1min"]:
                    _code = _TF_CODES.get(_tf)
                    if _code is None:
                        continue
                    with ThreadPoolExecutor(max_workers=1) as _ex:
                        _f = _ex.submit(
                            _mt5.copy_rates_from_pos,
                            self.active_symbols[0], _code, 0, 500,
                        )
                        try:
                            rates = _f.result(timeout=45)
                        except _TO:
                            logger.warning(f"Bg timeout {_tf}")
                            continue
                    if rates is None or len(rates) == 0:
                        continue
                    records = []
                    for r in rates:
                        records.append({
                            "time": _pd.Timestamp.fromtimestamp(r["time"], tz="UTC"),
                            "open": float(r["open"]), "high": float(r["high"]),
                            "low": float(r["low"]), "close": float(r["close"]),
                            "volume": float(r["tick_volume"]),
                        })
                    d = _pd.DataFrame(records)
                    d["time"] = _pd.to_datetime(d["time"])
                    if len(d) >= 500:
                        df = d
                        logger.info(f"Bg datos {_tf}: {len(d)} velas")
                        break
                if df is None or len(df) < 500:
                    _time.sleep(600)
                    continue

                def _wf_fit(params, d=df):
                    return _eval_fit(params, d)

                wf = WalkForwardOptimizer(self.genetic_optimizer, n_windows=3, window_size=800, step_size=200)
                wf_result = wf.run(df, _wf_fit)
                if "avg_params" in wf_result:
                    logger.info(f"Walk-forward avg fitness: {wf_result['avg_test_fitness']:.4f}")
                    self._optimization_result = wf_result
            except Exception as e:
                logger.warning(f"Background optimization error: {e}")
            _time.sleep(7200)

    def start(self, max_duration: int = 0):
        logger.info("=" * 50)
        logger.info("SMC Scoring Trading Bot starting...")
        logger.info(f"Symbols: {', '.join(self.active_symbols)} | Min Score: {self.min_score}")
        logger.info(f"ContextOracle activo | KellyRiskManager activo")
        if max_duration:
            logger.info(f"Max duration: {max_duration // 60} min")
        logger.info("=" * 50)

        if not self.mt5.connect():
            logger.error("Failed to connect to MT5. Exiting.")
            return

        for sym in self.active_symbols:
            stale = self.mt5.get_pending_orders(sym)
            if stale:
                logger.info(f"[{sym}] Limpiando {len(stale)} órdenes pendientes del inicio anterior")
                for o in stale:
                    self.executor.cancel_pending_order(o["ticket"])

        self._train_regime_models()
        self._run_genetic_optimization()

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.running = True
        self.start_time = time.time()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._initialize_state())

        self.loop.run_until_complete(self._restore_market_hours())

        self.scheduler.add_callback(self._on_new_candle)
        self.scheduler.start()
        self._evaluate()

        logger.info("Bot running. Press Ctrl+C to stop.")

        while self.running:
            self.loop.run_until_complete(asyncio.sleep(self.config["strategy"].get("loop_sleep_seconds", 5)))
            self.loop.run_until_complete(self._save_state_periodic())
            self._hot_reload_models()

            now_ts = time.time()
            if now_ts - self._last_market_hours_persist > self._market_hours_persist_interval:
                self.loop.run_until_complete(
                    self.state_persistence.save_market_hours(
                        self.hours_controller.get_all_states()
                    )
                )
                self.hours_controller.mark_persisted()
                self._last_market_hours_persist = now_ts

            if now_ts - self._last_meta_analysis > self._meta_analysis_interval:
                try:
                    meta_result = self.meta_learner.analyze_performance()
                    if meta_result.get("analyzed"):
                        logger.info(f"Meta-Learning: {len(meta_result.get('adjustments', []))} ajustes")
                        for adj in meta_result["adjustments"]:
                            logger.info(f"  Ajuste: {adj}")
                except Exception as e:
                    logger.warning(f"Meta-Learning analysis error: {e}")

            if max_duration and (time.time() - self.start_time) >= max_duration:
                logger.info(f"Max duration ({max_duration // 60} min) reached, stopping.")
                break

    def stop(self):
        logger.info("Stopping bot...")
        self.running = False
        self.scheduler.stop()
        if hasattr(self, 'loop') and self.loop.is_running():
            self.loop.run_until_complete(self._save_state_periodic())
            self.loop.run_until_complete(self.state_persistence.close())
        self.mt5.disconnect()
        if hasattr(self, 'failure_analyzer'):
            self.failure_analyzer.close()
        if hasattr(self, 'adaptive_thresholds'):
            self.adaptive_thresholds.close()
        logger.info("Bot stopped.")

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}")
        self.stop()
        sys.exit(0)

    def _on_new_candle(self, timeframe: str, candle_time: datetime):
        logger.info(f"New {timeframe} candle at {candle_time}")
        if timeframe == "1min":
            for symbol in self.symbols:
                self.hours_controller.on_new_candle(symbol, "1min", candle_time)
                self._manage_positions_light(symbol)
            if any(s["profile"].ltf_trigger == "M1" for s in self.symbols.values()):
                self._evaluate()
                return
        if timeframe != "5min":
            return
        self._evaluate()

    def _fetch_dxy_data(self) -> Optional[pd.DataFrame]:
        if self._dxy_unavailable:
            return None
        dxy_symbol = getattr(self.params, "dxy_symbol", "DXY")
        if not dxy_symbol:
            return None
        try:
            candles = self.mt5.get_candles(dxy_symbol, timeframe=15, count=60)
            if not candles:
                self._dxy_unavailable = True
                return None
            records = []
            for c in candles:
                records.append({
                    "time": c.timestamp, "open": c.open, "high": c.high,
                    "low": c.low, "close": c.close, "volume": c.volume,
                })
            df = pd.DataFrame(records)
            df["time"] = pd.to_datetime(df["time"])
            closes = df["close"].values
            if len(closes) >= 10:
                recent = closes[-5:].mean()
                prior = closes[-10:-5].mean()
                self._last_dxy_trend = "BULLISH" if recent > prior else "BEARISH" if recent < prior else "NEUTRAL"
            logger.info(f"DXY data fetched: {len(candles)} candles @ M15, trend={self._last_dxy_trend}")
            return df
        except Exception as e:
            self._dxy_unavailable = True
            logger.warning(f"DXY fetch failed, disabling: {e}")
            return None

    def _evaluate(self):
        logger.info("=" * 40)
        logger.info(f"SMC Evaluation started at {datetime.now()}")

        try:
            can_trade, reason = self.risk_manager.can_trade()
            if not can_trade:
                logger.info(f"Cannot trade: {reason}")
                return

            self.hours_controller.update_states()

            changed = self.hours_controller.get_changed_states()
            if changed:
                if threading.current_thread() is threading.main_thread():
                    self.loop.run_until_complete(
                        self.state_persistence.save_market_hours(changed)
                    )
                else:
                    asyncio.run_coroutine_threadsafe(
                        self.state_persistence.save_market_hours(changed),
                        self.loop
                    )
                self.hours_controller.mark_persisted()

            dxy_df = None
            if self.hours_controller.is_traditional_market_open():
                dxy_df = self._fetch_dxy_data()
            else:
                if not self._dxy_unavailable:
                    logger.info("[MarketHours] No traditional markets open, skipping DXY")

            for sym, sym_data in self.symbols.items():
                try:
                    if not self.hours_controller.should_evaluate(sym):
                        state = self.hours_controller.get_state(sym)
                        logger.debug(f"[{sym}] MarketHours state={state.value}, skipping evaluate")
                        continue
                    self._symbol_evaluator.evaluate(self, sym, sym_data, dxy_df=dxy_df)
                except Exception as e:
                    logger.exception(f"Error evaluating {sym}: {e}")

        except Exception as e:
            logger.error(f"Evaluation error: {e}")

        logger.info(f"Evaluation completed at {datetime.now()}")
        logger.info("=" * 40)

    def _all_positions_at_be(self, symbol: str, open_positions: list) -> bool:
        for p in open_positions:
            is_long = p["type"] == "buy"
            sl = p.get("sl") or 0
            entry = p["price_open"]
            if sl == 0:
                return False
            if is_long and sl < entry:
                return False
            if not is_long and sl > entry:
                return False
        return True

    def _widen_runners_tp(self, symbol: str, signal, new_entry: float, pip_value: float):
        sym_batches = self.batches.get(symbol, {})
        new_tp_dist = abs(signal.take_profit - new_entry)
        if new_tp_dist <= 0:
            return
        open_positions = self.mt5.get_positions(symbol)
        open_tickets = {p["ticket"] for p in open_positions}
        runner_tickets = set()
        for batch in sym_batches.values():
            if batch.get("scaled_out"):
                for t in batch["tickets"]:
                    if t in open_tickets:
                        runner_tickets.add(t)
        if not runner_tickets:
            return
        symbol_info = self.mt5.get_symbol_info(symbol)
        digit = symbol_info.get("digits", 5) if symbol_info else 5
        widened = 0
        for ticket in runner_tickets:
            pos = next((p for p in open_positions if p["ticket"] == ticket), None)
            if not pos:
                continue
            pos_entry = pos["price_open"]
            is_long = pos["type"] == "buy"
            old_tp = pos.get("tp")
            if is_long:
                new_tp = round(pos_entry + new_tp_dist, digit)
            else:
                new_tp = round(pos_entry - new_tp_dist, digit)
            if old_tp and ((is_long and new_tp <= old_tp) or (not is_long and new_tp >= old_tp)):
                continue
            result = self.executor.modify_position(ticket, pos["sl"], new_tp)
            if result.success:
                widened += 1
                logger.info(f"[{symbol}] Runner {ticket} TP widened: {old_tp} -> {new_tp}")
        if widened:
            logger.info(f"[{symbol}] Widened TP for {widened}/{len(runner_tickets)} runners")

    def _manage_pending_orders(self, symbol: str, ltf_df):
        """Check pending orders: cancel if invalidated, detect fills, promote to batches"""
        if not self.pending_orders:
            return

        open_positions = self.mt5.get_positions(symbol)
        open_tickets = {p["ticket"] for p in open_positions}
        current_price = ltf_df["close"].iloc[-1]
        from src.utils.helpers import atr
        atr_val = atr(ltf_df, 14).iloc[-1]
        invalidation_dist = max(atr_val * 2.0, 0.001)

        to_remove = []
        for ticket, info in list(self.pending_orders.items()):
            if info["symbol"] != symbol:
                continue

            if ticket in open_tickets:
                self._promote_pending_to_batch(symbol, ticket, info, open_positions)
                to_remove.append(ticket)
                continue

            poi = info["poi_level"]
            direction = info["direction"]
            invalidated = False
            if direction == "BUY" and current_price < poi - invalidation_dist:
                invalidated = True
            elif direction == "SELL" and current_price > poi + invalidation_dist:
                invalidated = True
            if invalidated:
                logger.info(f"[{symbol}] POI {poi} invalidated (price {current_price}), cancelling pending {ticket}")
                self.executor.cancel_pending_order(ticket)
                to_remove.append(ticket)
                continue

            profile_obj = self.symbols.get(symbol, {}).get("profile")
            cooldown_min = profile_obj.cooldown_bars_m1 if profile_obj is not None else 0
            placed = info.get("placed_at")
            if cooldown_min > 0 and placed:
                elapsed = (datetime.now(timezone.utc).replace(tzinfo=None) - placed).total_seconds() / 60
                if elapsed >= cooldown_min:
                    logger.info(f"[{symbol}] Pending {ticket} stale ({elapsed:.0f}min ≥ {cooldown_min}min cooldown), cancelling")
                    self.executor.cancel_pending_order(ticket)
                    to_remove.append(ticket)

        for t in to_remove:
            self.pending_orders.pop(t, None)

        self._cleanup_pending_batches(symbol)

    def _promote_pending_to_batch(self, symbol, ticket, info, open_positions):
        zone_id = info.get("zone_id")
        if not zone_id:
            pos = next((p for p in open_positions if p["ticket"] == ticket), None)
            if pos:
                self.position_states[ticket] = {
                    "be_activated": False, "trail_activated": False,
                    "original_sl": info.get("sl", 0), "entry_price": pos["price_open"],
                    "hit_50pct": False, "managed_tp": info.get("tp", 0),
                    "is_long": pos["type"] == "buy",
                }
            if ticket in self.pending_orders:
                del self.pending_orders[ticket]
            return

        pend_batch = self._pending_batches.get(symbol, {}).get(zone_id)
        if not pend_batch:
            return

        if ticket not in pend_batch["filled_tickets"]:
            pend_batch["filled_tickets"].append(ticket)

        filled = pend_batch["filled_tickets"]
        all_pend = pend_batch["tickets"]

        if len(filled) >= len(all_pend) * 0.5:
            self.batches.setdefault(symbol, {})[zone_id] = {
                "tickets": filled[:],
                "direction": pend_batch["direction"],
                "scaled_out": False,
                "entry_price": pend_batch["entry_price"],
                "sl": pend_batch["sl"],
                "tp": pend_batch["tp"],
            }
            for ft in filled:
                pos = next((p for p in open_positions if p["ticket"] == ft), None)
                if pos:
                    self.position_states[ft] = {
                        "be_activated": False, "trail_activated": False,
                        "original_sl": pend_batch["sl"], "entry_price": pos["price_open"],
                        "hit_50pct": False, "managed_tp": pend_batch["tp"],
                        "zone_id": zone_id,
                    }
            logger.info(f"[{symbol}] Batch {zone_id[:8]} promoted: {len(filled)} filled")

    def _cleanup_pending_batches(self, symbol):
        sym_pend = self._pending_batches.get(symbol, {})
        active_batches = self.batches.get(symbol, {})
        for zone_id, pend_batch in list(sym_pend.items()):
            filled = pend_batch["filled_tickets"]
            all_pend = pend_batch["tickets"]
            if not filled and zone_id not in active_batches:
                continue
            unfilled = [t for t in all_pend if t not in filled]
            for t in unfilled:
                if t not in self.pending_orders:
                    continue
                self.executor.cancel_pending_order(t)
                del self.pending_orders[t]
            if all(t not in self.pending_orders for t in all_pend):
                del sym_pend[zone_id]

    def _calc_volume(self, signal, profile, ltf_df, pip_value, symbol, conviction=None) -> float:
        from src.utils.helpers import atr
        atr_val = atr(ltf_df, 14).iloc[-1]

        prev_atr = self._last_atr.get(symbol)
        vol_mult = 1.0
        if prev_atr and prev_atr > 0:
            atr_change = (atr_val - prev_atr) / prev_atr
            if atr_change > self._atr_spike_threshold:
                vol_mult = max(0.3, 1.0 - atr_change)
                logger.info(f"[{symbol}] ATR spike {atr_change:.0%} → volumen ×{vol_mult:.2f}")
        self._last_atr[symbol] = atr_val

        if getattr(self, '_adaptive_cooldown_enabled', True):
            cd_mult = getattr(self, '_cd_volume_mult', None)
            if cd_mult is not None and cd_mult < 1.0:
                vol_mult *= cd_mult
                logger.info(f"[{symbol}] Cooldown volumen ×{cd_mult:.2f}")
            streak = self.adaptive_cooldown.get_streak(symbol)
            if streak.consecutive_losses >= 3:
                logger.info(f"[{symbol}] Racha: {streak.consecutive_losses} pérdidas consecutivas")

        symbol_info = self.mt5.get_symbol_info(symbol)
        if symbol_info:
            spread_points = symbol_info.get("spread", 0)
            spread_pips = spread_points / (10 ** (symbol_info.get("digits", 5) - 1))
            max_spread = self.config["strategy"].get("params", {}).get("max_spread_pips", 15.0)
            if spread_pips > max_spread:
                logger.warning(f"[{symbol}] Spread {spread_pips:.1f} pips exceeds max")
                return 0.0

        risk_pct = profile.risk_per_trade_pct / 100.0

        decision = self._last_decision.get(symbol)
        if conviction is None:
            conviction = signal.conviction
        if decision:
            risk_pct *= min(decision.suggested_volume_pct, 3.0)

        # ── Dynamic VaR: ecuación unificada ──
        kelly_fraction = self.kelly_risk.get_risk_fraction(conviction)

        regime = self._last_regime.get(symbol)
        atr_ratio = regime.atr_ratio if regime else 1.0

        session_weight = getattr(self, '_session_vol_mult', 1.0)

        if not self._positions_cache:
            self._positions_cache = self.mt5.get_positions()
        all_positions = self._positions_cache
        correlation_factor = 1.0
        if all_positions:
            corr_vol_adj, _ = self.correlation_engine.volume_adjustment(
                symbol, {p["symbol"]: p["volume"] for p in all_positions},
            )
            correlation_factor = corr_vol_adj

        dynamic_risk, var_components = self.dynamic_var_risk.get_risk_fraction(
            kelly_fraction=kelly_fraction,
            conviction=conviction,
            atr_ratio=atr_ratio,
            session_weight=session_weight,
            correlation_factor=correlation_factor,
        )

        risk_pct = dynamic_risk
        var_notes = "; ".join(var_components.notes) if var_components.notes else "base"
        logger.info(
            f"[{symbol}] VaR dinámico: Kelly×{var_components.kelly_base:.4f} "
            f"vol×{var_components.volatility_weight:.3f} "
            f"time×{var_components.time_weight:.3f} "
            f"dd×{var_components.drawdown_weight:.3f} "
            f"corr×{var_components.correlation_weight:.3f} "
            f"→ {dynamic_risk*100:.4f}% ({var_notes})"
        )

        position = self.risk_manager.calculate_position_size(
            symbol, signal.entry_price, signal.stop_loss, signal.take_profit, atr_val,
            risk_per_trade_pct=risk_pct, conviction=conviction,
        )
        volume = position.volume * vol_mult
        max_vol = getattr(profile, 'max_volume', 0)
        if max_vol > 0:
            volume = min(volume, max_vol)

        volume = max(0.01, round(volume, 2))

        logger.info(f"[{symbol}] Position: {volume} lots (conv={conviction:.0%}, "
                    f"score={signal.score:.1f}), SL: {position.sl_pips:.1f} pips, "
                    f"TP: {position.tp_pips:.1f} pips, RR: {position.reward_risk_ratio:.2f}")
        return volume

    def _record_trade(self, symbol: str, position: dict, profit: float,
                      fresh_signal: TradingSignal, exit_reason: str = "reversal"):
        self.risk_manager.record_trade(profit)

        if not hasattr(self, 'meta_learner') or not self.meta_learner:
            return

        if getattr(self, '_adaptive_cooldown_enabled', True):
            pattern_name = fresh_signal.primary_pattern.type.name if fresh_signal and fresh_signal.primary_pattern else None
            self.adaptive_cooldown.record_trade(symbol, profit, pattern=pattern_name)

        direction = "BUY" if position.get("type") == "buy" else "SELL"
        regime_val = self._last_regime.get(symbol).regime.value if self._last_regime.get(symbol) else "UNKNOWN"
        alignment = self._last_regime.get(symbol).trend_alignment if self._last_regime.get(symbol) else "NEUTRAL"

        try:
            if fresh_signal and fresh_signal.score_breakdown:
                engine = self.symbols.get(symbol, {}).get("engine")
                if engine and hasattr(engine, 'record_breakdown_outcome'):
                    engine.record_breakdown_outcome(
                        signal_id=f"{symbol}_{position.get('ticket', 0)}",
                        symbol=symbol, direction=direction,
                        profit=profit,
                        breakdown=fresh_signal.score_breakdown,
                        score_net=fresh_signal.score,
                    )

            last_reg = self._last_regime.get(symbol)
            atr_r = last_reg.atr_ratio if last_reg else 1.0
            vol_reg = "HIGH" if atr_r > 1.5 else "LOW" if atr_r < 0.7 else "MEDIUM"
            ses_lbl = fresh_signal.session_profile.label if fresh_signal and fresh_signal.session_profile else ""
            streak = -self.adaptive_cooldown.get_streak(symbol).consecutive_losses

            ctx_vector = self.context_oracle.build_context_vector(
                adx=last_reg.adx_value if last_reg else 0,
                atr_ratio=atr_r,
                regime_type=regime_val,
                alignment=alignment,
                hour=datetime.now().hour,
                conviction=fresh_signal.conviction if fresh_signal else 0,
                score_net=fresh_signal.score if fresh_signal else 0,
                session=ses_lbl,
                day_of_week=datetime.now().weekday(),
                volatility_regime=vol_reg,
                dxy_trend=self._last_dxy_trend if hasattr(self, '_last_dxy_trend') else "",
                streak=streak,
            )
            self.context_oracle.record_outcome(
                vector=ctx_vector, direction=direction,
                profit=profit,
                conviction=fresh_signal.conviction if fresh_signal else 0,
                score=fresh_signal.score if fresh_signal else 0,
                regime=regime_val, alignment=alignment,
                session=ses_lbl,
                streak=streak,
                volatility_regime=vol_reg,
            )

            self.kelly_risk.record_trade(
                symbol=symbol, direction=direction,
                profit=profit,
                conviction=fresh_signal.conviction if fresh_signal else 0,
                regime=regime_val,
            )

            self.dynamic_var_risk.record_trade_result(profit)
            self.portfolio_risk.record_trade_result(symbol, profit)
            account_info = self.mt5.get_account_info()
            if account_info:
                self.dynamic_var_risk.update_balance(account_info["balance"])

            record = TradeRecord(
                symbol=symbol,
                direction=direction,
                entry_price=position.get("price_open", 0),
                exit_price=position.get("price_current", 0),
                volume=position.get("volume", 0),
                profit=profit,
                score=fresh_signal.score if fresh_signal else 0,
                conviction=fresh_signal.conviction if fresh_signal else 0,
                regime=regime_val,
                session=fresh_signal.session_profile.label if fresh_signal and fresh_signal.session_profile else "UNKNOWN",
                primary_pattern=fresh_signal.primary_pattern.type.name if fresh_signal and fresh_signal.primary_pattern else None,
                patterns_found=list(fresh_signal.score_breakdown.keys()) if fresh_signal and fresh_signal.score_breakdown else [],
                regime_confidence=self._last_regime.get(symbol).confidence if self._last_regime.get(symbol) else 0,
                exit_reason=exit_reason,
                duration_minutes=0,
                timestamp=datetime.now(),
            )
            self.meta_learner.record_trade(record)

            # ── MD Concept Feedback Loop ──
            self._feedback_md_concepts(symbol, profit, direction)

            from src.core.failure_analyzer import TradePostmortem
            try:
                pm = TradePostmortem(
                    symbol=symbol,
                    direction=record.direction,
                    entry_price=record.entry_price,
                    exit_price=record.exit_price,
                    profit=profit,
                    score=record.score,
                    conviction=record.conviction,
                    regime=record.regime,
                    session=record.session,
                    primary_pattern=record.primary_pattern,
                    patterns_found=record.patterns_found,
                    exit_reason=exit_reason,
                    timestamp=time.time(),
                    duration_minutes=0,
                )
                self.failure_analyzer.record_trade(pm)
            except Exception as e2:
                logger.warning(f"[{symbol}] Error in failure analysis: {e2}")
        except Exception as e:
            logger.warning(f"[{symbol}] Error recording trade: {e}")

    def _feedback_md_concepts(self, symbol: str, profit: float, direction: str):
        """Feed back trade results to MD concept tracker.
        Cada concepto MD activo durante la evaluación recibe feedback
        de si el trade ganó o perdió. El tracker ajusta pesos por símbolo.
        """
        if not hasattr(self, 'concept_tracker') or not self.concept_tracker:
            return
        detections = self._last_md_detections.get(symbol, [])
        if not detections:
            return
        won = profit > 0
        for det in detections:
            if det.direction not in (direction, "NEUTRAL", ""):
                continue
            try:
                self.concept_tracker.record_result(
                    symbol=symbol,
                    concept=det.concept,
                    won=won,
                    pnl=profit,
                    confidence=det.confidence,
                )
            except Exception as e:
                logger.warning(f"[{symbol}] Error feeding back {det.concept.value}: {e}")

    def _get_regime_config(self, symbol: str) -> dict:
        regime = self._last_regime.get(symbol)
        rt = regime.regime if regime else None
        configs = {
            RegimeType.STRONG_TREND_BULLISH: {
                "trail_mults": {1: 2.5, 2: 2.0, 3: 1.5, 4: 1.2},
                "stage_names": {1: "medium", 2: "tight", 3: "lock", 4: "lock+"},
                "tp_thresholds": [0.40, 0.55, 0.70, 0.90, 1.10],
                "tp_mults": [1.3, 1.6, 2.0, 2.5, 3.0],
                "reversal_strong": 85, "reversal_loss": 60,
                "be_at": 0.12,
            },
            RegimeType.STRONG_TREND_BEARISH: {
                "trail_mults": {1: 2.5, 2: 2.0, 3: 1.5, 4: 1.2},
                "stage_names": {1: "medium", 2: "tight", 3: "lock", 4: "lock+"},
                "tp_thresholds": [0.40, 0.55, 0.70, 0.90, 1.10],
                "tp_mults": [1.3, 1.6, 2.0, 2.5, 3.0],
                "reversal_strong": 85, "reversal_loss": 60,
                "be_at": 0.12,
            },
            RegimeType.RANGING: {
                "trail_mults": {1: 3.0, 2: 2.5, 3: 2.0, 4: 1.5},
                "stage_names": {1: "wide", 2: "medium", 3: "tight", 4: "lock"},
                "tp_thresholds": [0.30, 0.45, 0.60],
                "tp_mults": [1.2, 1.4, 1.6],
                "reversal_strong": 70, "reversal_loss": 45,
                "be_at": 0.10,
            },
            RegimeType.TRANSITION: {
                "trail_mults": {1: 3.0, 2: 2.5, 3: 2.0, 4: 1.5},
                "stage_names": {1: "wide", 2: "medium", 3: "tight", 4: "lock"},
                "tp_thresholds": [0.25, 0.40, 0.55],
                "tp_mults": [1.1, 1.3, 1.5],
                "reversal_strong": 65, "reversal_loss": 40,
                "be_at": 0.10,
            },
            RegimeType.HIGH_VOLATILITY: {
                "trail_mults": {1: 4.0, 2: 3.5, 3: 3.0, 4: 2.5},
                "stage_names": {1: "very_wide", 2: "wide", 3: "medium", 4: "tight"},
                "tp_thresholds": [0.50, 0.70],
                "tp_mults": [1.2, 1.4],
                "reversal_strong": 90, "reversal_loss": 70,
                "be_at": 0.15,
            },
        }
        result = configs.get(rt, {}).copy()
        try:
            rev_adaptive, loss_adaptive = self.adaptive_thresholds.get_optimal_thresholds()
            if result:
                result["reversal_strong"] = min(result.get("reversal_strong", 80), rev_adaptive)
                result["reversal_loss"] = min(result.get("reversal_loss", 50), loss_adaptive)
            else:
                result["reversal_strong"] = rev_adaptive
                result["reversal_loss"] = loss_adaptive
        except Exception:
            pass
        if not result:
            result = {}
        try:
            bs = getattr(self, 'circuit_breakers', None)
            if bs and hasattr(bs, '_atr_history'):
                atr_hist = bs._atr_history.get(symbol, [])
                if len(atr_hist) >= 5:
                    baseline = sum(atr_hist[:-1]) / len(atr_hist[:-1])
                    current = atr_hist[-1]
                    if baseline > 0 and current > baseline * 1.5:
                        factor = current / baseline
                        trail_mults = result.get("trail_mults", {})
                        if trail_mults:
                            result["trail_mults"] = {k: v * factor for k, v in trail_mults.items()}
        except Exception:
            pass
        return result

    def _manage_symbol_position(self, symbol: str, current_signal: TradingSignal = None):
        if not self._positions_cache:
            self._positions_cache = self.mt5.get_positions()
        all_positions = self._positions_cache
        if not all_positions:
            self.position_states.clear()
            return

        sym_positions = [p for p in all_positions if p["symbol"] == symbol]
        if not sym_positions:
            for ticket, state in list(self.position_states.items()):
                if state.get("symbol") == symbol and state.get("entry_price"):
                    logger.info(f"[{symbol}] Position {ticket} vanished (SL/TP hit): recording trade (no open positions)")
                    self._record_trade(
                        symbol=symbol,
                        position={
                            "type": "buy" if state.get("is_long", True) else "sell",
                            "price_open": state["entry_price"],
                            "price_current": state.get("last_price", state["entry_price"]),
                            "volume": state.get("volume", 0),
                            "profit": state.get("last_profit", 0),
                        },
                        profit=state.get("last_profit", 0),
                        fresh_signal=current_signal,
                        exit_reason="sl_tp",
                    )
                self.trailing_mgr.remove_state(ticket)
                self.position_states.pop(ticket, None)
            return

        sym_data = self.symbols.get(symbol)
        if not sym_data:
            return

        profile = sym_data["profile"]

        timeframes = self.fetcher.get_dataframes(symbol, count=300)
        ltf_df = None
        for tf in ["1min", "3min", "5min"]:
            df = timeframes.get(tf)
            if df is not None and len(df) >= 50:
                ltf_df = df
                break
        if ltf_df is None:
            return

        pip = sym_data["pip_value"]
        symbol_info = self.mt5.get_symbol_info(symbol)
        if not symbol_info:
            return

        digit = symbol_info.get("digits", 5)
        fresh_signal = current_signal

        open_by_ticket = {p["ticket"]: p for p in sym_positions}

        self._check_scale_out(symbol, open_by_ticket, profile)

        fresh_positions = [p for p in sym_positions if p["ticket"] in open_by_ticket]

        for position in fresh_positions:
            ticket = position["ticket"]
            pos_type = position["type"]
            is_long = (pos_type == "buy")
            entry_price = position["price_open"]
            current_price = position["price_current"]
            profit = position["profit"]
            current_sl = position["sl"]
            current_tp = position["tp"]

            ts = self.trailing_mgr.get_state(ticket)
            if ts is None:
                ts = self.trailing_mgr.initialize_state(
                    ticket=ticket, symbol=symbol, is_long=is_long,
                    entry_price=entry_price, current_sl=current_sl,
                    current_tp=current_tp, current_price=current_price,
                    profit=profit, volume=position["volume"],
                )
                self.position_states[ticket] = ts.to_dict()
            else:
                self.trailing_mgr.update_state(ticket, profit, current_price)

            state = self.position_states[ticket]
            state["last_profit"] = profit
            state["last_price"] = current_price
            be_activated = ts.be_activated
            be_was_active = be_activated
            trail_activated = ts.trail_activated
            original_tp = ts.managed_tp or current_tp

            logger.info(f"[{symbol}] Position {ticket}: P/L=${profit:.2f}, Entry={entry_price}, "
                        f"Current={current_price}, SL={current_sl}, BE={be_activated}, "
                        f"Trail={trail_activated}, FreshSignal={fresh_signal.direction}")

            new_sl = current_sl
            new_tp = current_tp

            # --- Recalculate SL/TP from fresh signal if same direction (solo una vez) ---
            signal_dir = fresh_signal.direction.upper()
            pos_dir = "BUY" if is_long else "SELL"
            if signal_dir in ("BUY", "SELL") and signal_dir == pos_dir and not ts.recalculated:
                fresh_tp = fresh_signal.take_profit
                if fresh_tp and fresh_tp > 0:
                    base_sl_mult = self.risk_manager.config.atr_multiplier_sl
                    vol_adj = self.volatility_scaler.compute(
                        symbol, ltf_df, base_sl_mult, base_sl_mult * 1.33
                    )
                    sl_mult = vol_adj["sl_mult"]
                    tp_mult = vol_adj["tp_mult"]

                    result = self.trailing_mgr.recalculate_sl_tp(
                        state=ts, ltf_df=ltf_df, fresh_tp=fresh_tp,
                        profile=profile, pip=pip, digit=digit,
                        sl_mult=sl_mult, tp_mult=tp_mult,
                        base_sl_pips=profile.sl_min_pips,
                        base_tp_pips=profile.sl_max_pips,
                    )
                    new_sl = result.new_sl
                    new_tp = result.new_tp
                    ts = result.updated_state
                    state.update(ts.to_dict())
                    original_tp = new_tp
                    be_activated = False
                    be_was_active = False
                    logger.info(f"[{symbol}] SL/TP recalculado {ticket}: "
                                f"SL {current_sl} -> {new_sl}, TP {current_tp} -> {new_tp}")
                    logger.info(f"[{symbol}] Vol SL/TP recalc: SL×{sl_mult:.2f} TP×{tp_mult:.2f} ({vol_adj['reason']})")
                    for action in result.actions:
                        logger.info(f"[{symbol}] {action}")

            # --- Compute trailing via TrailingStopManager (handles BE + stage + algorithm) ---
            result = self.trailing_mgr.compute(
                ticket=ticket, state=ts,
                ltf_df=ltf_df, current_price=current_price,
                current_sl=new_sl, current_tp=new_tp,
                pip=pip, digit=digit,
            )
            ts = result.updated_state
            state.update(ts.to_dict())
            new_sl = result.new_sl
            new_tp = result.new_tp
            be_activated = ts.be_activated
            for action in result.actions:
                logger.info(f"[{symbol}] {action}")

            price_profit = current_price - entry_price if is_long else entry_price - current_price
            tp_distance = abs(original_tp - entry_price) if original_tp else 999

            # --- Close on reversal signal (regime-aware) ---
            close_trade = False
            rconf_rev = self._get_regime_config(symbol)
            min_rev = rconf_rev.get("reversal_strong", self.params.min_reversal_score)
            min_loss_rev = rconf_rev.get("reversal_loss", self.min_score)
            if fresh_signal.direction.upper() == "BUY" and not is_long:
                if fresh_signal.score >= min_rev:
                    close_trade = True
                    logger.warning(f"[{symbol}] Closing SELL {ticket}: bullish signal (score={fresh_signal.score:.0f})")
                elif profit < 0 and fresh_signal.score >= min_loss_rev:
                    close_trade = True
                    logger.warning(f"[{symbol}] Closing losing SELL {ticket}: trend reversal (score={fresh_signal.score:.0f})")
            elif fresh_signal.direction.upper() == "SELL" and is_long:
                if fresh_signal.score >= min_rev:
                    close_trade = True
                    logger.warning(f"[{symbol}] Closing BUY {ticket}: bearish signal (score={fresh_signal.score:.0f})")
                elif profit < 0 and fresh_signal.score >= min_loss_rev:
                    close_trade = True
                    logger.warning(f"[{symbol}] Closing losing BUY {ticket}: trend reversal (score={fresh_signal.score:.0f})")

            if close_trade:
                close_result = self.executor.close_position(ticket)
                if close_result.success:
                    logger.info(f"[{symbol}] Closed {ticket}: structure reversed, P/L=${profit:.2f}")
                    self.adaptive_thresholds.record_reversal(
                        score=fresh_signal.score if fresh_signal else 0,
                        conviction=fresh_signal.conviction if fresh_signal else 0,
                        regime=self._last_regime.get(symbol).regime.value if self._last_regime.get(symbol) else "UNKNOWN",
                        direction="BUY" if is_long else "SELL",
                        profit=profit,
                    )
                    self._record_trade(
                        symbol=symbol, position=position, profit=profit,
                        fresh_signal=fresh_signal, exit_reason="reversal",
                    )
                    self.trailing_mgr.remove_state(ticket)
                    state.clear()
                    open_by_ticket.pop(ticket, None)
                continue

            # --- Progressive TP expansion for runners + SL lock profit ---
            if be_activated and original_tp and tp_distance > 0:
                current_stage = state.get("tp_expanded", 0)
                rconf_tp = self._get_regime_config(symbol)
                thresholds = rconf_tp.get("tp_thresholds", [0.50, 0.65, 0.80, 1.00, 1.25])
                multipliers = rconf_tp.get("tp_mults", [1.5, 2.0, 2.5, 3.0, 3.5])
                base_distance = ts.original_tp_distance
                if current_stage < len(thresholds):
                    pct_of_tp = price_profit / base_distance if base_distance > 0 else 0
                    threshold = thresholds[current_stage]
                    if pct_of_tp >= threshold:
                        mult = multipliers[current_stage]
                        add_distance = base_distance * mult
                        if is_long:
                            new_tp = round(entry_price + add_distance, digit)
                        else:
                            new_tp = round(entry_price - add_distance, digit)
                        state["managed_tp"] = new_tp
                        state["tp_expanded"] = current_stage + 1
                        logger.info(f"[{symbol}] TP expandido {ticket} stage {current_stage+1}: "
                                    f"{current_tp} -> {new_tp} ({mult:.1f}x base)")
                        lock_pct = max(0.05, min(0.50, 0.10 + current_stage * 0.10))
                        locked_sl = round(entry_price + price_profit * lock_pct, digit) if is_long else round(entry_price - price_profit * lock_pct, digit)
                        if is_long and locked_sl > new_sl:
                            new_sl = locked_sl
                            logger.info(f"[{symbol}] SL bloqueado {ticket}: {current_sl} -> {new_sl} ({lock_pct*100:.0f}% ganancia)")
                        elif not is_long and locked_sl < new_sl:
                            new_sl = locked_sl
                            logger.info(f"[{symbol}] SL bloqueado {ticket}: {current_sl} -> {new_sl} ({lock_pct*100:.0f}% ganancia)")

            # --- Modify SL/TP por posición ---
            if new_sl != current_sl or new_tp != current_tp:
                mod_result = self.executor.modify_position(ticket, new_sl, new_tp)
                if mod_result.success:
                    state["current_sl"] = new_sl

        # --- Enforce SL ordering: best-entry (older) positions must have safer SL than worse-entry (newer) ---
        if len(fresh_positions) > 1:
            buy_group = sorted(
                [p for p in fresh_positions if p["type"] == "buy"],
                key=lambda p: p["price_open"]
            )
            sell_group = sorted(
                [p for p in fresh_positions if p["type"] == "sell"],
                key=lambda p: p["price_open"], reverse=True
            )
            for group, is_long in [(buy_group, True), (sell_group, False)]:
                if len(group) < 2:
                    continue
                for i in range(len(group) - 2, -1, -1):
                    p_old, p_new = group[i], group[i + 1]
                    t_old = p_old["ticket"]
                    s_old = self.position_states.get(t_old, {}).get("current_sl") or p_old["sl"]
                    t_new = p_new["ticket"]
                    s_new = self.position_states.get(t_new, {}).get("current_sl") or p_new["sl"]
                    if is_long and s_old > s_new:
                        self.executor.modify_position(t_old, s_new, p_old["tp"])
                        if t_old in self.position_states:
                            self.position_states[t_old]["current_sl"] = s_new
                        logger.info(f"[{symbol}] SL ordering BUY {t_old}: {s_old} -> {s_new} (capped by {t_new})")
                    elif not is_long and s_old < s_new:
                        self.executor.modify_position(t_old, s_new, p_old["tp"])
                        if t_old in self.position_states:
                            self.position_states[t_old]["current_sl"] = s_new
                        logger.info(f"[{symbol}] SL ordering SELL {t_old}: {s_old} -> {s_new} (capped by {t_new})")

        # --- Cancel opposite pending orders (symbol-level, outside position loop) ---
        if fresh_signal.direction.upper() in ("BUY", "SELL"):
            rev_dir = "BUY" if fresh_signal.direction.upper() == "SELL" else "SELL"
            to_cancel = [t for t, info in list(self.pending_orders.items())
                         if info.get("symbol") == symbol and info.get("direction") == rev_dir]
            for t in to_cancel:
                self.executor.cancel_pending_order(t)
                self.pending_orders.pop(t, None)
                logger.info(f"[{symbol}] Reversal: cancelled pending {rev_dir} order {t}")

        # --- Clean up positions closed by SL/TP at broker ---
        for ticket, state in list(self.position_states.items()):
            if state.get("symbol") != symbol:
                continue
            if ticket not in open_by_ticket:
                if state.get("entry_price"):
                    logger.info(f"[{symbol}] Position {ticket} vanished (SL/TP hit): recording trade")
                    self._record_trade(
                        symbol=symbol,
                        position={
                            "type": "buy" if state.get("is_long", True) else "sell",
                            "price_open": state["entry_price"],
                            "price_current": state.get("last_price", state["entry_price"]),
                            "volume": state.get("volume", 0),
                            "profit": state.get("last_profit", 0),
                        },
                        profit=state.get("last_profit", 0),
                        fresh_signal=fresh_signal,
                        exit_reason="sl_tp",
                    )
                self.trailing_mgr.remove_state(ticket)
                self.position_states.pop(ticket, None)

    def _manage_positions_light(self, symbol: str):
        """Lightweight position management on every 1-min candle.
        Delegates BE + trailing to TrailingStopManager. Handles TP expansion."""
        sym_positions = self.mt5.get_positions(symbol)
        if not sym_positions:
            for ticket, state in list(self.position_states.items()):
                if state.get("symbol") != symbol:
                    continue
                if state.get("entry_price"):
                    logger.info(f"[{symbol}] Position {ticket} vanished (SL/TP hit): recording trade")
                    self._record_trade(
                        symbol=symbol,
                        position={
                            "type": "buy" if state.get("is_long", True) else "sell",
                            "price_open": state["entry_price"],
                            "price_current": state.get("last_price", state["entry_price"]),
                            "volume": state.get("volume", 0),
                            "profit": state.get("last_profit", 0),
                        },
                        profit=state.get("last_profit", 0),
                        fresh_signal=None,
                        exit_reason="sl_tp",
                    )
                self.trailing_mgr.remove_state(ticket)
                self.position_states.pop(ticket, None)
            return

        sym_data = self.symbols.get(symbol)
        if not sym_data:
            return

        timeframes = self.fetcher.get_dataframes(symbol, count=300)
        trail_df = None
        for tf in ["5min", "15min", "1min"]:
            df = timeframes.get(tf)
            if df is not None and len(df) >= 50:
                trail_df = df
                break

        pip = sym_data["pip_value"]
        symbol_info = self.mt5.get_symbol_info(symbol)
        if not symbol_info:
            return
        digit = symbol_info.get("digits", 5)

        open_by_ticket = {p["ticket"]: p for p in sym_positions}

        for ticket, pos in open_by_ticket.items():
            if ticket not in self.position_states:
                continue

            state = self.position_states[ticket]
            if state.get("is_long") is None:
                continue
            if state.get("be_activated") and state.get("tp_expanded", 0) >= len(self._get_regime_config(symbol).get("tp_thresholds", [0.50, 0.65, 0.80, 1.00, 1.25])):
                continue

            ts = self.trailing_mgr.get_state(ticket)
            if ts is None:
                continue
            self.trailing_mgr.update_state(ticket, pos["profit"], pos["price_current"])

            is_long = state["is_long"]
            entry_price = state["entry_price"]
            current_price = pos["price_current"]
            current_sl = pos["sl"]
            current_tp = pos["tp"]
            original_tp = state.get("managed_tp", current_tp)

            price_profit = current_price - entry_price if is_long else entry_price - current_price
            tp_distance = abs(original_tp - entry_price) if original_tp else 999

            new_sl = current_sl
            new_tp = current_tp
            modified = False

            # Compute trailing via TrailingStopManager (handles BE + stage + algorithm)
            result = self.trailing_mgr.compute(
                ticket=ticket, state=ts,
                ltf_df=trail_df, current_price=current_price,
                current_sl=new_sl, current_tp=new_tp,
                pip=pip, digit=digit,
            )
            ts = result.updated_state
            state.update(ts.to_dict())
            new_sl = result.new_sl
            new_tp = result.new_tp
            for action in result.actions:
                logger.info(f"[{symbol}] (1min) {action}")
            if new_sl != current_sl or new_tp != current_tp:
                modified = True

            # Progressive TP expansion (1min) + SL lock
            if ts.be_activated and original_tp and tp_distance > 0:
                current_stage = state.get("tp_expanded", 0)
                rconf_tp = self._get_regime_config(symbol)
                thresholds = rconf_tp.get("tp_thresholds", [0.50, 0.65, 0.80, 1.00, 1.25])
                multipliers = rconf_tp.get("tp_mults", [1.5, 2.0, 2.5, 3.0, 3.5])
                base_distance = ts.original_tp_distance
                if current_stage < len(thresholds):
                    pct_of_tp = price_profit / base_distance if base_distance > 0 else 0
                    threshold = thresholds[current_stage]
                    if pct_of_tp >= threshold:
                        mult = multipliers[current_stage]
                        add_distance = base_distance * mult
                        new_tp = round(entry_price + add_distance, digit) if is_long else round(entry_price - add_distance, digit)
                        state["managed_tp"] = new_tp
                        state["tp_expanded"] = current_stage + 1
                        lock_pct = max(0.05, min(0.50, 0.10 + current_stage * 0.10))
                        locked_sl = round(entry_price + price_profit * lock_pct, digit) if is_long else round(entry_price - price_profit * lock_pct, digit)
                        if is_long and locked_sl > new_sl:
                            new_sl = locked_sl
                            modified = True
                            logger.info(f"[{symbol}] SL bloqueado (1min) {ticket}: {current_sl} -> {new_sl} ({lock_pct*100:.0f}% ganancia)")
                        elif not is_long and locked_sl < new_sl:
                            new_sl = locked_sl
                            modified = True
                            logger.info(f"[{symbol}] SL bloqueado (1min) {ticket}: {current_sl} -> {new_sl} ({lock_pct*100:.0f}% ganancia)")
                        modified = True
                        logger.info(f"[{symbol}] TP expandido (1min) {ticket} stage {current_stage+1}: "
                                    f"{current_tp} -> {new_tp}")

            if modified:
                self.executor.modify_position(ticket, new_sl, new_tp)
                state["current_sl"] = new_sl

            state["last_profit"] = pos["profit"]
            state["last_price"] = current_price

        # --- Enforce SL ordering across same-direction positions (1min) ---
        if len(sym_positions) > 1:
            buy_group = sorted(
                [p for p in sym_positions if p["type"] == "buy"],
                key=lambda p: p["price_open"]
            )
            sell_group = sorted(
                [p for p in sym_positions if p["type"] == "sell"],
                key=lambda p: p["price_open"], reverse=True
            )
            for group, is_long in [(buy_group, True), (sell_group, False)]:
                if len(group) < 2:
                    continue
                for i in range(len(group) - 2, -1, -1):
                    p_old, p_new = group[i], group[i + 1]
                    t_old = p_old["ticket"]
                    s_old = self.position_states.get(t_old, {}).get("current_sl") or p_old["sl"]
                    t_new = p_new["ticket"]
                    s_new = self.position_states.get(t_new, {}).get("current_sl") or p_new["sl"]
                    if is_long and s_old > s_new:
                        self.executor.modify_position(t_old, s_new, p_old["tp"])
                        if t_old in self.position_states:
                            self.position_states[t_old]["current_sl"] = s_new
                        logger.info(f"[{symbol}] SL ordering BUY (1min) {t_old}: {s_old} -> {s_new} (capped by {t_new})")
                    elif not is_long and s_old < s_new:
                        self.executor.modify_position(t_old, s_new, p_old["tp"])
                        if t_old in self.position_states:
                            self.position_states[t_old]["current_sl"] = s_new
                        logger.info(f"[{symbol}] SL ordering SELL (1min) {t_old}: {s_old} -> {s_new} (capped by {t_new})")

    def _check_scale_out(self, symbol: str, open_by_ticket: dict, profile):
        symbol_batches = self.batches.get(symbol, {})
        if not symbol_batches and len(open_by_ticket) >= 4:
            first_p = list(open_by_ticket.values())[0]
            direction = "BUY" if first_p["type"] == "buy" else "SELL"
            tickets = list(open_by_ticket.keys())
            entry = first_p["price_open"]
            tp = first_p.get("tp") or 0
            sl = first_p.get("sl") or 0
            zone_id = f"rec_{symbol}_{direction}_{entry}"
            self.batches.setdefault(symbol, {})[zone_id] = {
                "tickets": tickets, "direction": direction,
                "scaled_out": False, "entry_price": entry,
                "sl": sl, "tp": tp,
            }
            symbol_batches = self.batches[symbol]
            logger.info(f"[{symbol}] Batch recuperado: {len(tickets)} tickets dir={direction} entry={entry} tp={tp}")
        if not symbol_batches:
            return

        for zone_id, batch in list(symbol_batches.items()):
            if batch.get("scaled_out"):
                continue

            open_tickets = [t for t in batch["tickets"] if t in open_by_ticket]
            if not open_tickets:
                if all(t not in open_by_ticket for t in batch["tickets"]):
                    del symbol_batches[zone_id]
                continue

            positions_data = [open_by_ticket[t] for t in open_tickets]
            entry_price = batch["entry_price"]
            tp_price = batch["tp"]
            is_long = batch["direction"] == "BUY"

            first_pos = positions_data[0]
            pos_tp = first_pos.get("tp")
            if pos_tp and tp_price:
                batch_tp_dist = abs(tp_price - entry_price)
                pos_tp_dist = abs(pos_tp - (first_pos["price_open"] or entry_price))
                if pos_tp_dist > 0 and abs(batch_tp_dist - pos_tp_dist) / pos_tp_dist > 0.2:
                    tp_price = pos_tp
                    entry_price = first_pos["price_open"] or entry_price
                    batch["tp"] = tp_price
                    batch["entry_price"] = entry_price
            if tp_price == 0 and pos_tp:
                tp_price = pos_tp
                batch["tp"] = tp_price
            if entry_price == 0 and first_pos.get("price_open"):
                entry_price = first_pos["price_open"]
                batch["entry_price"] = entry_price

            tp_distance = abs(tp_price - entry_price) if tp_price else 999

            if tp_distance <= 0 or tp_distance >= 999:
                continue

            avg_pnl_pct = 0
            for p in positions_data:
                pos_entry = p["price_open"]
                pos_tp_val = p.get("tp") or tp_price
                pos_tp_dist = abs(pos_tp_val - pos_entry) if pos_tp_val else 1
                if pos_tp_dist == 0:
                    pos_tp_dist = 1
                if is_long:
                    pct = (p["price_current"] - pos_entry) / pos_tp_dist
                else:
                    pct = (pos_entry - p["price_current"]) / pos_tp_dist
                avg_pnl_pct += pct
            avg_pnl_pct /= len(positions_data) if positions_data else 1

            close_at = profile.scale_close_at_tp_pct
            if avg_pnl_pct >= close_at:
                ratio = profile.scale_close_ratio
                to_close = max(int(len(open_tickets) * ratio), 1)
                close_tickets = open_tickets[:to_close]
                logger.info(f"[{symbol}] Scale-out zone={zone_id[:8]}: "
                            f"{avg_pnl_pct*100:.0f}% TP, closing {to_close}/{len(open_tickets)} tickets")
                for t in close_tickets:
                    close_result = self.executor.close_position(t)
                    if close_result.success:
                        logger.info(f"[{symbol}] Scale-out closed {t}: P/L=${open_by_ticket[t]['profit']:.2f}")
                        open_by_ticket.pop(t, None)
                        self.trailing_mgr.remove_state(t)
                        if t in self.position_states:
                            self.position_states[t].clear()
                    else:
                        logger.warning(f"[{symbol}] Scale-out close failed for {t}: {close_result.message}")
                batch["scaled_out"] = True

                remaining = [t for t in open_tickets if t not in close_tickets]
                if remaining:
                    logger.info(f"[{symbol}] Runners remaining: {len(remaining)} tickets after scale-out")


def main():
    project_root = Path(__file__).parent.parent
    from dotenv import load_dotenv
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    bot = TradingBot(project_root / "config")
    duration = 0
    for i, arg in enumerate(sys.argv):
        if arg == "--duration" and i + 1 < len(sys.argv):
            duration = int(sys.argv[i + 1])
    bot.start(max_duration=duration)


if __name__ == "__main__":
    main()

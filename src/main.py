"""SMC Trading Bot - Main Entry Point
Automated trading bot for XAUUSDm using Smart Money Concepts scoring strategy
"""
import json
import signal
import sys
import asyncio
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
from src.utils.state_persistence import StatePersistence
from src.core.news_calendar import NewsCalendar
from src.core.regime_detector import RegimeDetector, RegimeContext
from src.core.continuous_decision import ContinuousDecider, ContinuousDecision
from src.learning.market_memory import MarketMemory
from src.core.session_profiler import SessionProfiler
from src.learning.meta_learner import MetaLearner, TradeRecord
from src.risk.fixed_risk_manager import FixedRiskManager, RiskConfig, calculate_atr
from src.executor.order_executor import OrderExecutor
from src.scheduler.timeframe_scheduler import TimeframeScheduler
from src.utils.helpers import pip_size, is_in_session
import pandas as pd


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
    min_rr_ratio: float = 4.0
    tp_fixed_pips: float = 0.0
    risk_per_trade_pct: float = 2.0
    max_concurrent_trades: int = 1
    max_volume: float = 0
    max_daily_loss_pct: float = 6.0
    max_daily_trades: int = 10
    allowed_sessions: List[Tuple[int, int]] = field(default_factory=list)
    cooldown_bars_m1: int = 5
    scale_entries: int = 10
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
        symbol=cfg.get("symbol", "XAUUSDm"),
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
        scale_entries=cfg.get("scale_entries", 10),
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
    )


class TradingBot:
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.config = load_config(config_dir)
        setup_logging()

        import os
        broker_config = self.config["broker"]["mt5"]
        self.mt5 = MT5Client(
            login=os.environ.get("MT5_LOGIN") or broker_config.get("login"),
            password=os.environ.get("MT5_PASSWORD") or broker_config.get("password"),
            server=os.environ.get("MT5_SERVER") or broker_config.get("server"),
        )

        self.fetcher = MultiTimeframeFetcher(self.mt5)
        str_cfg = self.config["strategy"]
        self.active_symbols = str_cfg.get("active_symbols", ["XAUUSDm"])
        self.params = build_strategy_params(str_cfg.get("params", {}))
        scoring_cfg = build_scoring_config(str_cfg.get("scoring", {}))
        self.min_score = str_cfg.get("min_score_to_trade", 65.0)
        self.high_confidence_score = str_cfg.get("high_confidence_score", 85.0)

        news_buffer = self.params.news_buffer_minutes if self.params.news_filter_active else 0
        self.news_calendar = NewsCalendar(config_dir=config_dir, buffer_minutes=news_buffer)

        self.min_reentry_score = str_cfg.get("min_reentry_score", 70.0)

        self.symbols = {}
        for sym in self.active_symbols:
            sym_cfg = str_cfg["symbols"].get(sym, {})
            profile = build_symbol_profile(sym_cfg)
            engine = StrategyEngine(
                profile=profile, params=self.params,
                weights=scoring_cfg, min_score=self.min_score,
                high_confidence_score=self.high_confidence_score,
                min_net_score=self.params.min_net_score,
            )
            self.symbols[sym] = {
                "profile": profile,
                "engine": engine,
                "last_trade_time": None,
                "pip_value": pip_size(sym),
            }

        risk_config = RiskConfig(
            risk_per_trade=self.config["risk"]["risk_per_trade"],
            max_daily_loss=self.config["risk"]["max_daily_loss"],
            max_positions=self.config["risk"]["max_positions"],
            atr_multiplier_sl=self.config["risk"]["atr_multiplier_sl"],
            atr_multiplier_tp=self.config["risk"]["atr_multiplier_tp"],
            min_reward_risk_ratio=self.config["risk"]["min_reward_risk_ratio"],
        )

        self.mt5.connect()
        account_info = self.mt5.get_account_info()
        balance = account_info["balance"] if account_info else 1000.0

        self.risk_manager = FixedRiskManager(risk_config, balance)
        self.executor = OrderExecutor(self.mt5)
        self.scheduler = TimeframeScheduler(self.fetcher, self.active_symbols[0])

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
        self.regime_detector = RegimeDetector(
            atr_period=regime_params.get("atr_period", 14),
            adx_period=regime_params.get("adx_period", 14),
        )
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
        self._last_meta_analysis = time.time()
        self._meta_analysis_interval = 14400  # cada 4 horas

        self._last_regime: Dict[str, RegimeContext] = {}
        self._last_decision: Dict[str, ContinuousDecision] = {}

        self._consecutive_losses: Dict[str, int] = {}
        self._hot_pause_until: Dict[str, Optional[float]] = {}
        self._last_atr: Dict[str, float] = {}
        self._total_daily_losses: int = 0
        self._max_consecutive_losses = 5
        self._hot_pause_duration = 1800  # 30 minutos de pausa tras racha
        self._atr_spike_threshold = 0.50  # 50% de incremento en ATR reduce tamaño

    async def _initialize_state(self):
        await self.state_persistence.initialize()
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
                "original_tp_distance": abs(tp_original - entry_price) if tp_original else 0,
                "last_profit": 0,
                "last_price": 0,
            }
            logger.info(f"Loaded position: ticket={pos['ticket']}, "
                        f"BE={pos['be_activated']}, Trail={pos['trailing_activated']}")

    async def _save_state_periodic(self):
        await self.state_persistence.save_daily_state(
            daily_loss=self.risk_manager.daily_loss,
            trades_count=len(self.position_states),
        )

    def start(self, max_duration: int = 0):
        logger.info("=" * 50)
        logger.info("SMC Scoring Trading Bot starting...")
        logger.info(f"Symbols: {', '.join(self.active_symbols)} | Min Score: {self.min_score}")
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

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.running = True
        self.start_time = time.time()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._initialize_state())

        self.scheduler.add_callback(self._on_new_candle)
        self.scheduler.start()
        self._evaluate()

        logger.info("Bot running. Press Ctrl+C to stop.")

        while self.running:
            self.loop.run_until_complete(asyncio.sleep(self.config["strategy"].get("loop_sleep_seconds", 5)))
            self.loop.run_until_complete(self._save_state_periodic())

            if time.time() - self._last_meta_analysis > self._meta_analysis_interval:
                self._last_meta_analysis = time.time()
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
        logger.info("Bot stopped.")

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}")
        self.stop()
        sys.exit(0)

    def _on_new_candle(self, timeframe: str, candle_time: datetime):
        logger.info(f"New {timeframe} candle at {candle_time}")
        if timeframe == "1min":
            for symbol in self.symbols:
                self._manage_positions_light(symbol)
        if timeframe != "5min":
            return
        self._evaluate()

    def _fetch_dxy_data(self) -> Optional[pd.DataFrame]:
        if self._dxy_unavailable:
            return None
        dxy_symbol = getattr(self.params, "dxy_symbol", "DX")
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
            logger.info(f"DXY data fetched: {len(candles)} candles @ M15")
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

            dxy_df = self._fetch_dxy_data()

            for sym, sym_data in self.symbols.items():
                try:
                    self._evaluate_symbol(sym, sym_data, dxy_df=dxy_df)
                except Exception as e:
                    logger.exception(f"Error evaluating {sym}: {e}")

        except Exception as e:
            logger.error(f"Evaluation error: {e}")

        logger.info(f"Evaluation completed at {datetime.now()}")
        logger.info("=" * 40)

    def _evaluate_symbol(self, symbol: str, sym_data: dict, dxy_df: pd.DataFrame = None):
        profile = sym_data["profile"]
        engine = sym_data["engine"]
        pip_value = sym_data["pip_value"]

        timeframes = self.fetcher.get_dataframes(symbol, count=300)
        if len(timeframes) < 3:
            return

        ltf_df = None
        for tf in ["1min", "3min", "5min"]:
            df = timeframes.get(tf)
            if df is not None and len(df) >= 50:
                ltf_df = df
                break
        if ltf_df is None:
            return

        self._manage_pending_orders(symbol, ltf_df)

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        news_active = self.news_calendar.is_high_impact_active(now, symbol)

        htf_df = engine._pick_tf(timeframes, "HTF")

        regime = self.regime_detector.detect(htf_df, ltf_df)
        self._last_regime[symbol] = regime

        signal = engine.evaluate_adaptive(
            timeframes, now, news_active=news_active,
            regime=regime, dxy_df=dxy_df,
        )

        conv = signal.conviction

        logger.info(f"[{symbol}] {signal.direction} | Score: {signal.score:.1f} | "
                    f"Convicción: {conv:.0%} | "
                    f"Primary: {signal.primary_pattern.type.name if signal.primary_pattern else 'none'}"
                    f"{' | Notes: '+'; '.join(signal.notes) if signal.notes else ''}")
        if signal.distribution:
            logger.info(f"[{symbol}] Distribución: μ={signal.distribution.mean:.1f} σ={signal.distribution.std:.1f} "
                        f"convergencia={signal.distribution.convergence:.0%}")
        if signal.regime_context:
            logger.info(f"[{symbol}] Régimen: {signal.regime_context.regime.value} "
                        f"(conf={signal.regime_context.confidence:.0%}, ADX={signal.regime_context.adx_value:.0f})")

        # --- Alimentar market_memory con swing points detectados ---
        from src.utils.helpers import atr, find_swing_points
        atr_val = atr(ltf_df, 14).iloc[-1]
        highs_idx, lows_idx = find_swing_points(ltf_df, lookback=3)
        current_close = ltf_df["close"].iloc[-1]
        for idx in highs_idx[-10:]:
            price = round(ltf_df["high"].iloc[idx], 5)
            outcome = "break" if current_close > price else "bounce"
            self.market_memory.record_interaction(symbol, price, outcome, ltf_df=ltf_df)
        for idx in lows_idx[-10:]:
            price = round(ltf_df["low"].iloc[idx], 5)
            outcome = "break" if current_close < price else "bounce"
            self.market_memory.record_interaction(symbol, price, outcome, ltf_df=ltf_df)

        # --- Ajustar convicción según niveles clave del market_memory ---
        level_bias = self.market_memory.get_level_bias(symbol, current_close, atr_val)
        if level_bias and signal.direction in ("BUY", "SELL"):
            old_conv = conv
            if level_bias == "BULLISH_BIAS" and signal.direction == "SELL":
                conv *= 0.5
                logger.info(f"[{symbol}] Soporte cerca (bias BULLISH): SELL penalizado {old_conv:.0%} → {conv:.0%}")
            elif level_bias == "BULLISH_BIAS" and signal.direction == "BUY":
                conv = min(1.0, conv * 1.2)
                logger.info(f"[{symbol}] Soporte cerca (bias BULLISH): BUY potenciado {old_conv:.0%} → {conv:.0%}")
            elif level_bias == "BEARISH_BIAS" and signal.direction == "BUY":
                conv *= 0.5
                logger.info(f"[{symbol}] Resistencia cerca (bias BEARISH): BUY penalizado {old_conv:.0%} → {conv:.0%}")
            elif level_bias == "BEARISH_BIAS" and signal.direction == "SELL":
                conv = min(1.0, conv * 1.2)
                logger.info(f"[{symbol}] Resistencia cerca (bias BEARISH): SELL potenciado {old_conv:.0%} → {conv:.0%}")
            elif level_bias == "RANGE_BIAS":
                conv *= 0.8
                logger.info(f"[{symbol}] Rango detectado: convicción reducida {old_conv:.0%} → {conv:.0%}")

        session_profile = self.session_profiler.profile(symbol, ltf_df, now)
        decision = self.continuous_decider.decide(
            signal.distribution if signal.distribution else None,
            regime, profile, ltf_df,
        )
        self._last_decision[symbol] = decision
        logger.info(f"[{symbol}] Decisión: vol={decision.suggested_volume_pct:.1f}x conv={decision.conviction:.0%} "
                    f"SLx={decision.sl_width_multiplier:.1f} TPx={decision.tp_width_multiplier:.1f}")

        self._manage_symbol_position(symbol, current_signal=signal)

        hot_pause_until = self._hot_pause_until.get(symbol)
        if hot_pause_until and time.time() < hot_pause_until:
            remaining = int(hot_pause_until - time.time())
            logger.info(f"[{symbol}] HOT PAUSE activo ({remaining}s restantes), saltando entrada")
            return

        if signal.direction not in ("BUY", "SELL"):
            if hasattr(self, 'meta_learner') and self.meta_learner:
                self.meta_learner.record_skipped_signal(
                    symbol=symbol, direction=signal.direction,
                    score=signal.score, conviction=conv,
                    regime=regime.regime.value if regime else "UNKNOWN",
                    session=session_profile.label if session_profile else "UNKNOWN",
                    reason="direccion_hold",
                    pattern_type=signal.primary_pattern.type.name if signal.primary_pattern else None,
                    price=ltf_df["close"].iloc[-1] if ltf_df is not None else None,
                )
            return
        if conv < 0.3:
            if hasattr(self, 'meta_learner') and self.meta_learner:
                self.meta_learner.record_skipped_signal(
                    symbol=symbol, direction=signal.direction,
                    score=signal.score, conviction=conv,
                    regime=regime.regime.value if regime else "UNKNOWN",
                    session=session_profile.label if session_profile else "UNKNOWN",
                    reason="conviccion_baja",
                    pattern_type=signal.primary_pattern.type.name if signal.primary_pattern else None,
                    price=ltf_df["close"].iloc[-1] if ltf_df is not None else None,
                )
            logger.info(f"[{symbol}] Convicción {conv:.0%} < 30%, saltando entrada")
            return
        if not decision.should_trade:
            if hasattr(self, 'meta_learner') and self.meta_learner:
                self.meta_learner.record_skipped_signal(
                    symbol=symbol, direction=signal.direction,
                    score=signal.score, conviction=conv,
                    regime=regime.regime.value if regime else "UNKNOWN",
                    session=session_profile.label if session_profile else "UNKNOWN",
                    reason="decision_no_operar",
                    pattern_type=signal.primary_pattern.type.name if signal.primary_pattern else None,
                    price=ltf_df["close"].iloc[-1] if ltf_df is not None else None,
                )
            logger.info(f"[{symbol}] Decisión indica no operar, saltando")
            return

        if not is_in_session(now, profile.allowed_sessions):
            if hasattr(self, 'meta_learner') and self.meta_learner:
                self.meta_learner.record_skipped_signal(
                    symbol=symbol, direction=signal.direction,
                    score=signal.score, conviction=conv,
                    regime=regime.regime.value if regime else "UNKNOWN",
                    session=session_profile.label if session_profile else "UNKNOWN",
                    reason="fuera_de_sesion",
                    pattern_type=signal.primary_pattern.type.name if signal.primary_pattern else None,
                    price=ltf_df["close"].iloc[-1] if ltf_df is not None else None,
                )
            logger.info(f"[{symbol}] Fuera de sesión activa {profile.allowed_sessions}, saltando")
            return

        for avoided in session_profile.avoided_patterns:
            if signal.primary_pattern and avoided in signal.primary_pattern.type.name.upper():
                if conv < 0.5:
                    if hasattr(self, 'meta_learner') and self.meta_learner:
                        self.meta_learner.record_skipped_signal(
                            symbol=symbol, direction=signal.direction,
                            score=signal.score, conviction=conv,
                            regime=regime.regime.value if regime else "UNKNOWN",
                            session=session_profile.label if session_profile else "UNKNOWN",
                            reason="patron_evitado_en_sesion",
                            pattern_type=signal.primary_pattern.type.name if signal.primary_pattern else None,
                            price=ltf_df["close"].iloc[-1] if ltf_df is not None else None,
                        )
                    logger.info(f"[{symbol}] Patrón {signal.primary_pattern.type.name} evitado en sesión {session_profile.label}, saltando")
                    return
                logger.info(f"[{symbol}] Patrón {signal.primary_pattern.type.name} evitado en sesión {session_profile.label}, "
                            f"pero convicción {conv:.0%} ≥ 50%, permitiendo")



        open_positions = self.mt5.get_positions(symbol)
        pyramid_entry = False
        if open_positions:
            sym_batches = self.batches.get(symbol, {})
            has_active = any(not b.get("scaled_out") for b in sym_batches.values())
            if has_active:
                if signal.score >= self.high_confidence_score:
                    logger.info(f"[{symbol}] Active batch exists but score {signal.score:.0f} >= high conf, allowing pyramid")
                    pyramid_entry = True
                else:
                    if hasattr(self, 'meta_learner') and self.meta_learner:
                        self.meta_learner.record_skipped_signal(
                            symbol=symbol, direction=signal.direction,
                            score=signal.score, conviction=conv,
                            regime=regime.regime.value if regime else "UNKNOWN",
                            session=session_profile.label if session_profile else "UNKNOWN",
                            reason="batch_activo_sin_piramide",
                            pattern_type=signal.primary_pattern.type.name if signal.primary_pattern else None,
                            price=ltf_df["close"].iloc[-1] if ltf_df is not None else None,
                        )
                    logger.info(f"[{symbol}] Active batch exists, score {signal.score:.0f} < {self.high_confidence_score}, skipping")
                    return
            else:
                pyramid_entry = True

        primary = signal.primary_pattern
        is_gap_pattern = primary is not None and (
            primary.type.name.startswith("FVG") or primary.type.name.startswith("VOID_SCALP")
        )
        if is_gap_pattern:
            gap_mid = primary.low + (primary.high - primary.low) * 0.5
            gap_distance = abs(ltf_df["close"].iloc[-1] - gap_mid)
            sl_max_pips = getattr(profile, 'sl_max_pips', None)
            if sl_max_pips is None:
                sl_max_pips = 20.0
            max_dist = sl_max_pips * pip_value * 3
            if gap_distance > max_dist:
                logger.info(f"[{symbol}] Gap distance {gap_distance/pip_value:.0f}p > {sl_max_pips*3:.0f}p, placing LIMIT @ {gap_mid}")
                volume = self._calc_volume(signal, profile, ltf_df, pip_value, symbol)
                if volume > 0:
                    result = self.executor.place_pending_limit(signal, volume, gap_mid)
                    if result.success:
                        self.pending_orders[result.order_ticket] = {
                            "symbol": symbol, "direction": signal.direction,
                            "poi_level": gap_mid, "signal_score": signal.score,
                            "placed_at": datetime.now(timezone.utc).replace(tzinfo=None),
                        }
                        sym_data["last_trade_time"] = datetime.now(timezone.utc).replace(tzinfo=None)
                return

        from src.utils.helpers import atr
        atr_val = atr(ltf_df, 14).iloc[-1]

        sl_min_pips = getattr(profile, 'sl_min_pips', 5.0)
        symbol_info = self.mt5.get_symbol_info(symbol)
        if symbol_info:
            spread_points = symbol_info.get("spread", 0)
            spread_pips = spread_points / (10 ** (symbol_info.get("digits", 5) - 1))
            max_spread = self.config["strategy"].get("params", {}).get("max_spread_pips", 15.0)
            if spread_pips > max_spread:
                logger.warning(f"[{symbol}] Spread {spread_pips:.1f} pips exceeds max, skipping")
                return
            if spread_pips * 2 > sl_min_pips:
                sl_min_pips = spread_pips * 2
            if isinstance(signal.stop_loss, (int, float)) and signal.stop_loss > 0:
                sl_distance = abs(signal.entry_price - signal.stop_loss) / pip_value
                if sl_distance < sl_min_pips:
                    new_sl_dist = sl_min_pips * pip_value
                    if signal.direction == "BUY":
                        adjusted_sl = signal.entry_price - new_sl_dist
                    else:
                        adjusted_sl = signal.entry_price + new_sl_dist
                    signal.stop_loss = adjusted_sl
                    logger.info(f"[{symbol}] SL ajustado de {sl_distance:.1f} a {sl_min_pips:.1f} pips")

        volume = self._calc_volume(signal, profile, ltf_df, pip_value, symbol)
        if volume <= 0:
            return

        scale_n = getattr(profile, 'scale_entries', 1)
        min_lot = 0.01
        per_unit = max(round(volume / scale_n / min_lot) * min_lot, min_lot) if scale_n > 1 else volume
        actual_volume = per_unit * scale_n if scale_n > 1 else volume
        if actual_volume > getattr(profile, 'max_volume', 0) > 0:
            actual_volume = min(actual_volume, profile.max_volume)
            per_unit = max(round(actual_volume / scale_n / min_lot) * min_lot, min_lot)

        entry_price = signal.entry_price

        symbol_info_2 = self.mt5.get_symbol_info(symbol)
        if symbol_info_2:
            if signal.direction == "SELL" and entry_price <= symbol_info_2["bid"]:
                entry_price = symbol_info_2["bid"] + atr_val * 0.5
                logger.info(f"[{symbol}] SELL entry ajustado a {entry_price:.5f} (bid={symbol_info_2['bid']:.5f})")
            elif signal.direction == "BUY" and entry_price >= symbol_info_2["ask"]:
                entry_price = symbol_info_2["ask"] - atr_val * 0.5
                logger.info(f"[{symbol}] BUY entry ajustado a {entry_price:.5f} (ask={symbol_info_2['ask']:.5f})")

        opposite_dir = "BUY" if signal.direction == "SELL" else "SELL"
        opposite_pending = [(t, info) for t, info in self.pending_orders.items()
                            if info.get("symbol") == symbol and info.get("direction") == opposite_dir]
        if opposite_pending:
            logger.info(f"[{symbol}] Cancelando {len(opposite_pending)} órdenes {opposite_dir} opuestas antes de {signal.direction}")
            for t, _ in opposite_pending:
                self.executor.cancel_pending_order(t)
                self.pending_orders.pop(t, None)

        if scale_n > 1 and per_unit >= min_lot:
            existing = [(t, info) for t, info in self.pending_orders.items()
                        if info.get("symbol") == symbol and info.get("direction") == signal.direction]
            if existing:
                old_entry = existing[0][1].get("poi_level")
                entry_diff = abs(old_entry - entry_price) if old_entry else 0
                if entry_diff > atr_val * 0.5:
                    logger.info(f"[{symbol}] Entry price cambió de {old_entry:.5f} a {entry_price:.5f}, cancelando {len(existing)} viejas")
                    for t, _ in existing:
                        self.executor.cancel_pending_order(t)
                        self.pending_orders.pop(t, None)
                else:
                    logger.info(f"[{symbol}] Ya hay {len(existing)} órdenes pendientes {signal.direction} activas @ {old_entry:.5f}, saltando nuevo batch")
                    return
            zone_id = str(uuid4())
            pend_tickets = []
            logger.info(f"[{symbol}] Scale pending: {scale_n}x{per_unit} lots @ {entry_price}, zone={zone_id[:8]}")
            for i in range(scale_n):
                result = self.executor.place_pending_entry(signal, per_unit, entry_price)
                if result.success:
                    pend_tickets.append(result.order_ticket)
                    self.pending_orders[result.order_ticket] = {
                        "symbol": symbol, "direction": signal.direction,
                        "poi_level": entry_price, "placed_at": datetime.now(timezone.utc).replace(tzinfo=None),
                        "zone_id": zone_id, "volume": per_unit,
                        "sl": signal.stop_loss, "tp": signal.take_profit,
                    }
                else:
                    logger.warning(f"[{symbol}] Pending order {i+1}/{scale_n} failed: {result.message}")
            if pend_tickets:
                self._pending_batches.setdefault(symbol, {})[zone_id] = {
                    "tickets": pend_tickets,
                    "direction": signal.direction,
                    "entry_price": entry_price,
                    "sl": signal.stop_loss,
                    "tp": signal.take_profit,
                    "scale_n": scale_n,
                    "filled_tickets": [],
                }
                sym_data["last_trade_time"] = datetime.now(timezone.utc).replace(tzinfo=None)
                logger.info(f"[{symbol}] Pending batch {zone_id[:8]}: {len(pend_tickets)}/{scale_n} placed")
                if pyramid_entry:
                    self._widen_runners_tp(symbol, signal, entry_price, pip_value)
        elif scale_n <= 1 or per_unit < min_lot:
            result = self.executor.place_pending_entry(signal, volume, entry_price)
            if result.success:
                sym_data["last_trade_time"] = datetime.now(timezone.utc).replace(tzinfo=None)
                self.pending_orders[result.order_ticket] = {
                    "symbol": symbol, "direction": signal.direction,
                    "poi_level": entry_price, "placed_at": datetime.now(timezone.utc).replace(tzinfo=None),
                    "zone_id": None, "volume": volume,
                    "sl": signal.stop_loss, "tp": signal.take_profit,
                }
                logger.info(f"[{symbol}] Pending order placed: {result.message}")
            else:
                logger.warning(f"[{symbol}] Pending order failed: {result.message}")

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

    def _calc_volume(self, signal, profile, ltf_df, pip_value, symbol) -> float:
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

        consecutive_losses = self._consecutive_losses.get(symbol, 0)
        if consecutive_losses >= 3:
            loss_mult = max(0.3, 1.0 - consecutive_losses * 0.15)
            vol_mult *= loss_mult
            logger.info(f"[{symbol}] {consecutive_losses} pérdidas consecutivas → volumen ×{loss_mult:.2f}")

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
        conviction = signal.conviction
        if decision:
            risk_pct *= min(decision.suggested_volume_pct, 3.0)

        position = self.risk_manager.calculate_position_size(
            symbol, signal.entry_price, signal.stop_loss, signal.take_profit, atr_val,
            risk_per_trade_pct=risk_pct, conviction=conviction,
        )
        volume = position.volume * vol_mult
        max_vol = getattr(profile, 'max_volume', 0)
        if max_vol > 0:
            volume = min(volume, max_vol)

        if signal.session_profile:
            volume *= signal.session_profile.volume_adjustment
            if signal.session_profile.is_weak:
                logger.info(f"[{symbol}] Sesión débil ({signal.session_profile.label}): volumen -50%")
            elif signal.session_profile.is_peak:
                logger.info(f"[{symbol}] Hora pico ({signal.session_profile.label}): volumen +20%")

        volume = max(0.01, round(volume, 2))

        logger.info(f"[{symbol}] Position: {volume} lots (conv={conviction:.0%}, "
                    f"score={signal.score:.1f}), SL: {position.sl_pips:.1f} pips, "
                    f"TP: {position.tp_pips:.1f} pips, RR: {position.reward_risk_ratio:.2f}")
        return volume

    def _record_trade(self, symbol: str, position: dict, profit: float,
                      fresh_signal: TradingSignal, exit_reason: str = "reversal"):
        if not hasattr(self, 'meta_learner') or not self.meta_learner:
            return

        if profit < 0:
            losses = self._consecutive_losses.get(symbol, 0) + 1
            self._consecutive_losses[symbol] = losses
            if losses >= self._max_consecutive_losses:
                self._hot_pause_until[symbol] = time.time() + self._hot_pause_duration
                logger.warning(
                    f"[{symbol}] {losses} pérdidas consecutivas → HOT PAUSE "
                    f"{self._hot_pause_duration // 60} min"
                )
        else:
            self._consecutive_losses[symbol] = 0

        try:
            record = TradeRecord(
                symbol=symbol,
                direction="BUY" if position.get("type") == "buy" else "SELL",
                entry_price=position.get("price_open", 0),
                exit_price=position.get("price_current", 0),
                volume=position.get("volume", 0),
                profit=profit,
                score=fresh_signal.score if fresh_signal else 0,
                conviction=fresh_signal.conviction if fresh_signal else 0,
                regime=self._last_regime.get(symbol).regime.value if self._last_regime.get(symbol) else "UNKNOWN",
                session=fresh_signal.session_profile.label if fresh_signal and fresh_signal.session_profile else "UNKNOWN",
                primary_pattern=fresh_signal.primary_pattern.type.name if fresh_signal and fresh_signal.primary_pattern else None,
                patterns_found=list(fresh_signal.score_breakdown.keys()) if fresh_signal and fresh_signal.score_breakdown else [],
                regime_confidence=self._last_regime.get(symbol).confidence if self._last_regime.get(symbol) else 0,
                exit_reason=exit_reason,
                duration_minutes=0,
                timestamp=datetime.now(),
            )
            self.meta_learner.record_trade(record)
        except Exception as e:
            logger.warning(f"[{symbol}] Error recording trade: {e}")

    def _manage_symbol_position(self, symbol: str, current_signal: TradingSignal = None):
        from src.utils.helpers import atr
        all_positions = self.mt5.get_positions()
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

            if ticket not in self.position_states:
                self.position_states[ticket] = {
                    "symbol": symbol,
                    "be_activated": False, "trail_activated": False,
                    "original_sl": current_sl, "entry_price": entry_price,
                    "tp_expanded": 0, "managed_tp": current_tp,
                    "original_tp_distance": abs(current_tp - entry_price) if current_tp else 0,
                    "is_long": is_long, "volume": position["volume"],
                    "last_profit": profit, "last_price": current_price,
                }

            state = self.position_states[ticket]
            state["last_profit"] = profit
            state["last_price"] = current_price
            be_activated = state["be_activated"]
            be_was_active = be_activated
            trail_activated = state["trail_activated"]
            original_tp = state.get("managed_tp", current_tp)

            logger.info(f"[{symbol}] Position {ticket}: P/L=${profit:.2f}, Entry={entry_price}, "
                        f"Current={current_price}, SL={current_sl}, BE={be_activated}, "
                        f"Trail={trail_activated}, FreshSignal={fresh_signal.direction}")

            new_sl = current_sl
            new_tp = current_tp

            # --- Recalculate SL/TP from fresh signal if same direction (solo una vez) ---
            signal_dir = fresh_signal.direction.upper()
            pos_dir = "BUY" if is_long else "SELL"
            already_recalculated = state.get("recalculated", False)
            if signal_dir in ("BUY", "SELL") and signal_dir == pos_dir and not already_recalculated:
                fresh_tp = fresh_signal.take_profit
                if fresh_tp and fresh_tp > 0:
                    atr_val = atr(ltf_df, 14).iloc[-1]
                    sl_min_dist = getattr(profile, 'sl_min_pips', 5.0) * pip
                    sl_max_dist = getattr(profile, 'sl_max_pips', 20.0) * pip
                    sl_distance = atr_val * self.risk_manager.config.atr_multiplier_sl
                    sl_distance = max(min(sl_distance, sl_max_dist), sl_min_dist)
                    if is_long:
                        new_sl = round(entry_price - sl_distance, digit)
                        tp_raw = fresh_tp
                        tp_min = max(current_price + atr_val * 2, entry_price + sl_distance * 2)
                        new_tp = max(tp_raw, tp_min)
                    else:
                        new_sl = round(entry_price + sl_distance, digit)
                        tp_raw = fresh_tp
                        tp_min = min(current_price - atr_val * 2, entry_price - sl_distance * 2)
                        new_tp = min(tp_raw, tp_min)
                    new_tp_dist = abs(new_tp - entry_price)
                    state["be_activated"] = False
                    state["trail_activated"] = False
                    state["tp_expanded"] = 0
                    state["original_tp_distance"] = new_tp_dist
                    state["managed_tp"] = new_tp
                    state["original_sl"] = new_sl
                    state["recalculated"] = True
                    original_tp = new_tp
                    be_activated = False
                    be_was_active = False
                    logger.info(f"[{symbol}] SL/TP recalculado {ticket}: "
                                f"SL {current_sl} -> {new_sl}, TP {current_tp} -> {new_tp}")
                    # If already deep in profit, apply trailing SL immediately
                    price_profit_check = current_price - entry_price if is_long else entry_price - current_price
                    if price_profit_check > atr_val * 3:
                        trail_atr = max(atr_val * 2.5, pip * 15)
                        if is_long:
                            trail_sl = round(current_price - trail_atr, digit)
                            if trail_sl > new_sl:
                                new_sl = trail_sl
                                state["trail_activated"] = True
                                logger.info(f"[{symbol}] Trail inmediato {ticket}: SL {new_sl} (ATR*2.5)")
                        else:
                            trail_sl = round(current_price + trail_atr, digit)
                            if trail_sl < new_sl:
                                new_sl = trail_sl
                                state["trail_activated"] = True
                                logger.info(f"[{symbol}] Trail inmediato {ticket}: SL {new_sl} (ATR*2.5)")
            price_profit = current_price - entry_price if is_long else entry_price - current_price
            tp_distance = abs(original_tp - entry_price) if original_tp else 999
            original_sl_price = state.get("original_sl", current_sl)
            sl_distance = abs(entry_price - original_sl_price) if original_sl_price else 0

            if not be_activated and tp_distance > 0:
                pct_of_tp = price_profit / tp_distance
                if pct_of_tp >= 0.2:
                    min_pip_pips = 5.0
                    buffer = max(pip * min_pip_pips, pip * 0.05)
                    be_sl = round(entry_price + buffer, digit) if is_long else round(entry_price - buffer, digit)
                    if is_long:
                        new_sl = max(new_sl, be_sl)
                    else:
                        new_sl = min(new_sl, be_sl)
                    be_activated = True
                    state["be_activated"] = True
                    logger.info(f"[{symbol}] BE activado a {pct_of_tp*100:.0f}% de TP ({price_profit/pip:.0f}p)")

            close_trade = False
            min_rev = self.params.min_reversal_score
            loss_pct = (current_price - entry_price) / entry_price if is_long else (entry_price - current_price) / entry_price
            if fresh_signal.direction.upper() == "BUY" and not is_long:
                if fresh_signal.score >= min_rev:
                    close_trade = True
                    logger.warning(f"[{symbol}] Closing SELL {ticket}: bullish signal (score={fresh_signal.score:.0f})")
                elif profit < 0 and fresh_signal.score >= self.min_score:
                    close_trade = True
                    logger.warning(f"[{symbol}] Closing losing SELL {ticket}: trend reversal (score={fresh_signal.score:.0f})")
            elif fresh_signal.direction.upper() == "SELL" and is_long:
                if fresh_signal.score >= min_rev:
                    close_trade = True
                    logger.warning(f"[{symbol}] Closing BUY {ticket}: bearish signal (score={fresh_signal.score:.0f})")
                elif profit < 0 and fresh_signal.score >= self.min_score:
                    close_trade = True
                    logger.warning(f"[{symbol}] Closing losing BUY {ticket}: trend reversal (score={fresh_signal.score:.0f})")

            if close_trade:
                close_result = self.executor.close_position(ticket)
                if close_result.success:
                    logger.info(f"[{symbol}] Closed {ticket}: structure reversed, P/L=${profit:.2f}")
                    self._record_trade(
                        symbol=symbol, position=position, profit=profit,
                        fresh_signal=fresh_signal, exit_reason="reversal",
                    )
                    state.clear()
                    open_by_ticket.pop(ticket, None)
                continue

            # --- Trailing stop: ATR en M5/M15 para evitar ruido de M1 ---
            if be_was_active:
                trail_tf = timeframes.get("5min") if timeframes.get("5min") is not None else (timeframes.get("15min") if timeframes.get("15min") is not None else ltf_df)
                trail_atr_base = atr(trail_tf, 14).iloc[-1] * 2.5
                trail_atr = max(trail_atr_base, pip * 15)
                if is_long:
                    candidate_sl = round(current_price - trail_atr, digit)
                    if candidate_sl > new_sl:
                        new_sl = candidate_sl
                        state["trail_activated"] = True
                        logger.info(f"[{symbol}] Trail SL {ticket}: {current_sl} -> {new_sl} (ATR*2.5 M5)")
                else:
                    candidate_sl = round(current_price + trail_atr, digit)
                    if candidate_sl < new_sl:
                        new_sl = candidate_sl
                        state["trail_activated"] = True
                        logger.info(f"[{symbol}] Trail SL {ticket}: {current_sl} -> {new_sl} (ATR*2.5 M5)")

            # --- Progressive TP expansion for runners ---
            if be_activated and original_tp and tp_distance > 0:
                current_stage = state.get("tp_expanded", 0)
                thresholds = [0.50, 0.65, 0.80]
                multipliers = [1.5, 2.0, 2.5]
                base_distance = state.get("original_tp_distance", tp_distance)
                if current_stage < len(thresholds):
                    pct_of_tp = price_profit / tp_distance
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
                self.position_states.pop(ticket, None)

    def _manage_positions_light(self, symbol: str):
        """Lightweight position management on every 1-min candle.
        Handles BE activation and TP expansion without full pattern detection.
        Runs between 5-min evaluations to catch intra-candle price moves."""
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
                self.position_states.pop(ticket, None)
            return

        sym_data = self.symbols.get(symbol)
        if not sym_data:
            return

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
            if state.get("be_activated") and state.get("tp_expanded", 0) >= 3:
                continue

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

            # BE activation at 20% of TP
            if not state.get("be_activated") and tp_distance > 0:
                pct_of_tp = price_profit / tp_distance
                if pct_of_tp >= 0.2:
                    min_buffer_pips = 5.0
                    buffer = max(pip * min_buffer_pips, pip * 0.05)
                    be_sl = round(entry_price + buffer, digit) if is_long else round(entry_price - buffer, digit)
                    if is_long:
                        new_sl = max(new_sl, be_sl)
                    else:
                        new_sl = min(new_sl, be_sl)
                    state["be_activated"] = True
                    modified = True
                    logger.info(f"[{symbol}] BE activado (1min) {ticket}: {pct_of_tp*100:.0f}% TP ({price_profit/pip:.0f}p)")

            # Progressive TP expansion (1min)
            if state.get("be_activated") and original_tp and tp_distance > 0:
                current_stage = state.get("tp_expanded", 0)
                thresholds = [0.50, 0.65, 0.80]
                multipliers = [1.5, 2.0, 2.5]
                base_distance = state.get("original_tp_distance", tp_distance)
                if current_stage < len(thresholds):
                    pct_of_tp = price_profit / tp_distance
                    threshold = thresholds[current_stage]
                    if pct_of_tp >= threshold:
                        mult = multipliers[current_stage]
                        add_distance = base_distance * mult
                        new_tp = round(entry_price + add_distance, digit) if is_long else round(entry_price - add_distance, digit)
                        state["managed_tp"] = new_tp
                        state["tp_expanded"] = current_stage + 1
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

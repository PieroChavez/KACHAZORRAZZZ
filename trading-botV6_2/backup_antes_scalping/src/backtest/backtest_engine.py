"""SMC Backtesting Engine
Replicates the full _evaluate_symbol() pipeline using historical data,
real SMC components, and a simulated executor.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from src.core.strategy_engine import StrategyEngine, TradingSignal
from src.core.regime_detector import RegimeDetector, RegimeContext
from src.core.session_profiler import SessionProfiler
from src.core.continuous_decision import ContinuousDecider
from src.core.market_memory import MarketMemory
from src.core.context_oracle import ContextOracle
from src.core.kelly_risk import KellyRiskManager
from src.core.volatility_scaler import VolatilityScaler
from src.core.correlation_engine import CorrelationEngine
from src.risk.fixed_risk_manager import FixedRiskManager
from src.risk.dynamic_var_risk import DynamicVaRRiskManager
from src.utils.helpers import atr, find_swing_points, pip_size

from .simulated_executor import SimulatedExecutor, SimulatedTrade
from .backtest_metrics import BacktestMetrics, compute_metrics, metrics_to_fitness

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    metrics: BacktestMetrics
    trades: List[SimulatedTrade]
    equity_curve: List[float]
    fitness: float = 0.0
    params: Dict = field(default_factory=dict)
    total_bars: int = 0
    execution_time_ms: float = 0.0


class BacktestEngine:
    """Offline backtest engine that replicates the SMC evaluation pipeline."""

    def __init__(self, config: dict, symbol: str = "XAUUSD"):
        self.config = config
        self.symbol = symbol

        strategy_cfg = config.get("strategy", {})
        adaptive = strategy_cfg.get("adaptive", {})
        mock_profile = type("MockProfile", (), {
            "symbol": symbol,
            "allowed_sessions": [],
            "max_concurrent_trades": 999,
            "cooldown_bars_m1": 0,
            "min_retracement": 0.5,
            "fvg_entry_level": 0.5,
            "sl_buffer_pips": 0.5,
            "sl_fixed_pips": 0.0,
            "sl_min_pips": 5.0,
            "sl_max_pips": 35.0,
            "min_rr_ratio": 4.0,
            "tp_fixed_pips": 0.0,
            "risk_per_trade_pct": 2.0,
            "max_volume": 0,
            "max_daily_loss_pct": 6.0,
            "max_daily_trades": 10,
            "scale_entries": 1,
            "scale_close_at_tp_pct": 0.5,
            "scale_close_ratio": 0.5,
        })()
        bt_params = type("BTParams", (), {
            "consolidation_max_atr_ratio": 0.6,
            "expansion_tick_acceleration": 1.8,
            "gap_detection_pips": 8.0,
            "macro_event_filter": False,
            "min_retracement_level": 0.50,
            "breakout_validation": "BodyClose",
            "fvg_min_size_atr_ratio": 0.3,
            "fvg_entry_level": 0.50,
            "void_min_size_pips": 5.0,
            "pivot_length": 10,
            "wyckoff_min_phase_bars": 12,
            "sequence_length": 3,
            "news_filter_active": False,
            "news_buffer_minutes": 30,
            "min_stop_distance_atr": 0.5,
            "min_reversal_score": 65.0,
            "min_net_score": adaptive.get("min_net_score", 5.0),
        })()
        self.strategy_engine = StrategyEngine(
            profile=mock_profile,
            params=bt_params,
            weights=None,
            min_score=adaptive.get("min_score_to_trade", 60),
            high_confidence_score=adaptive.get("high_confidence_score", 80),
            min_net_score=adaptive.get("min_net_score", 5),
        )
        self.regime_detector = RegimeDetector()
        self.session_profiler = SessionProfiler()
        self.continuous_decider = ContinuousDecider()
        self.market_memory = MarketMemory()
        self.context_oracle = ContextOracle()
        self.kelly_risk = KellyRiskManager()
        self.volatility_scaler = VolatilityScaler()
        self.correlation_engine = CorrelationEngine()

        from src.risk.fixed_risk_manager import RiskConfig
        rc = RiskConfig(
            risk_per_trade=config.get("risk_per_trade", 0.02),
            max_daily_loss=0.06,
            max_positions=5,
            atr_multiplier_sl=config.get("atr_multiplier_sl", 1.5),
            atr_multiplier_tp=config.get("atr_multiplier_tp", 2.0),
            min_reward_risk_ratio=config.get("min_reward_risk_ratio", 1.5),
        )
        self.risk_manager = FixedRiskManager(rc, balance=10_000.0)
        self.dynamic_var = DynamicVaRRiskManager()

        self._last_regime: Optional[RegimeContext] = None
        self._last_decision = None
        self._signal_id_counter = 0

    def run(self, data: Dict[str, pd.DataFrame], params: Optional[Dict] = None,
            step: int = 5, initial_balance: float = 10_000.0,
            max_trades: int = 500) -> BacktestResult:
        """Run backtest over historical data.

        Args:
            data: dict mapping timeframe name -> OHLCV DataFrame
            params: parameter overrides (merged into config)
            step: evaluate every N LTF bars for performance
            initial_balance: starting balance
            max_trades: cut off after this many trades

        Returns:
            BacktestResult with metrics, trades, equity curve
        """
        if params:
            merged = dict(self.config)
            merged.update(params)
            self._reconfigure(merged)

        ltf_df = self._pick_ltf(data)
        if ltf_df is None or len(ltf_df) < 100:
            return BacktestResult(
                metrics=BacktestMetrics(),
                trades=[], equity_curve=[initial_balance],
                total_bars=0,
            )

        htf_df = self._pick_htf(data)
        ltf_df = self._ensure_time_column(ltf_df)
        if htf_df is not None:
            htf_df = self._ensure_time_column(htf_df)

        n = len(ltf_df)
        executor = SimulatedExecutor(initial_balance=initial_balance)
        steps = list(range(60, n, step))
        total = len(steps)
        pip_value = pip_size(self.symbol)

        for idx, i in enumerate(steps):
            if len(executor.closed_trades) >= max_trades:
                break
            bar = ltf_df.iloc[:i + 1].copy()
            htf_bar = htf_df.iloc[:self._align_index(htf_df, ltf_df.index[i]) + 1].copy() if htf_df is not None else bar
            now = ltf_df.index[i]

            timestamp = now.to_pydatetime() if hasattr(now, 'to_pydatetime') else now
            high = bar["high"].iloc[-1]
            low = bar["low"].iloc[-1]
            close = bar["close"].iloc[-1]

            executor.update_positions(high, low, close, timestamp)

            timeframes = self._build_timeframes(data, now)

            regime = self.regime_detector.detect(htf_bar, bar)
            self._last_regime = regime

            signal = self.strategy_engine.evaluate_adaptive(
                timeframes, now, news_active=False, regime=regime, dxy_df=None,
            )

            self._signal_id_counter += 1
            conv = signal.conviction

            decision = self.continuous_decider.decide(
                signal.distribution if signal.distribution else None,
                regime, None, bar,
            )
            self._last_decision = decision
            session_profile = signal.session_profile or self.session_profiler.profile(self.symbol, bar, now)

            self._apply_level_bias(bar, close)

            conv = self._adjust_conviction_for_regime(conv, regime, signal)
            conv = self._adjust_conviction_for_market_memory(conv, signal, close)
            conv = self._adjust_conviction_for_consensus(conv, signal)

            if signal.direction not in ("BUY", "SELL"):
                continue
            if conv < 0.15:
                continue
            if not decision.should_trade:
                continue

            atr_val = atr(bar, 14).iloc[-1] if len(bar) >= 15 else 0.0
            if atr_val <= 0:
                continue

            sl_min_pips = 5.0
            entry_price = signal.entry_price
            stop_loss = signal.stop_loss or (entry_price - atr_val * 1.5 if signal.direction == "BUY" else entry_price + atr_val * 1.5)
            take_profit = signal.take_profit or (entry_price + atr_val * 2.0 if signal.direction == "BUY" else entry_price - atr_val * 2.0)

            vol_result = self.volatility_scaler.adjust_sl_tp(
                self.symbol, bar, entry_price, signal.direction,
                stop_loss, take_profit,
                digits=5, pip=pip_value,
                base_sl_mult=1.5, base_tp_mult=2.0,
                sl_min_pips=sl_min_pips,
                sl_max_pips=35.0,
            )
            stop_loss, take_profit = vol_result[0], vol_result[1]

            if not executor.can_open(5):
                continue

            volume = self._calc_sim_volume(signal, conv, atr_val, entry_price, stop_loss, take_profit)
            if volume <= 0:
                continue

            pattern_type = signal.primary_pattern.type.name if signal.primary_pattern else None
            executor.open_position(
                symbol=self.symbol,
                direction=signal.direction,
                entry_price=entry_price,
                volume=min(volume, 1.0),
                stop_loss=stop_loss,
                take_profit=take_profit,
                pattern_type=pattern_type,
                regime=regime.regime.value if regime else "",
                session=session_profile.label if session_profile else "",
                score=signal.score,
                conviction=conv,
            )

        executor.close_all(reason="end")
        metrics = compute_metrics(
            executor.closed_trades, executor.equity_curve,
            initial_balance, total,
        )
        fitness = metrics_to_fitness(metrics)
        return BacktestResult(
            metrics=metrics,
            trades=executor.closed_trades,
            equity_curve=executor.equity_curve,
            fitness=fitness,
            params=params or {},
            total_bars=total,
        )

    def _pick_ltf(self, data: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
        for tf in ("1min", "3min", "5min", "M5", "M15"):
            if tf in data and data[tf] is not None and len(data[tf]) >= 50:
                return data[tf]
        return None

    def _pick_htf(self, data: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
        for tf in ("4H", "H4", "1H", "H1"):
            if tf in data and data[tf] is not None and len(data[tf]) >= 20:
                return data[tf]
        return None

    @staticmethod
    def _align_index(ref_df: pd.DataFrame, target_ts) -> int:
        try:
            mask = ref_df.index <= target_ts
            if not mask.any():
                return len(ref_df) - 1
            return mask.sum() - 1
        except Exception:
            return len(ref_df) - 1

    def _build_timeframes(self, data: Dict[str, pd.DataFrame], now) -> Dict[str, pd.DataFrame]:
        result = {}
        for tf_name, df in data.items():
            if df is not None and len(df) > 0:
                df_use = self._ensure_time_column(df)
                idx = self._align_index(df_use, now)
                result[tf_name] = df_use.iloc[:idx + 1].copy()
        return result

    @staticmethod
    def _ensure_time_column(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "time" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df["time"] = df.index
            else:
                df["time"] = pd.Timestamp.now()
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df = df.set_index("time", drop=False).sort_index()
            except Exception:
                df.index = pd.to_datetime(df.index)
        return df

    def _apply_level_bias(self, df: pd.DataFrame, close: float):
        atr_val = atr(df, 14).iloc[-1] if len(df) >= 15 else 0.0
        highs_idx, lows_idx = find_swing_points(df, lookback=3)
        for idx in highs_idx[-10:]:
            price = round(df["high"].iloc[idx], 5)
            outcome = "break" if close > price else "bounce"
            self.market_memory.record_interaction(self.symbol, price, outcome, ltf_df=df)
        for idx in lows_idx[-10:]:
            price = round(df["low"].iloc[idx], 5)
            outcome = "break" if close < price else "bounce"
            self.market_memory.record_interaction(self.symbol, price, outcome, ltf_df=df)

    def _adjust_conviction_for_market_memory(self, conv: float, signal: TradingSignal, close: float) -> float:
        atr_val = 0.0
        level_bias = self.market_memory.get_level_bias(self.symbol, close, atr_val)
        if level_bias and signal.direction in ("BUY", "SELL"):
            if level_bias == "BULLISH_BIAS" and signal.direction == "SELL":
                conv *= 0.5
            elif level_bias == "BULLISH_BIAS" and signal.direction == "BUY":
                conv = min(1.0, conv * 1.2)
            elif level_bias == "BEARISH_BIAS" and signal.direction == "BUY":
                conv *= 0.5
            elif level_bias == "BEARISH_BIAS" and signal.direction == "SELL":
                conv = min(1.0, conv * 1.2)
            elif level_bias == "RANGE_BIAS":
                conv *= 0.8
        return conv

    def _adjust_conviction_for_regime(self, conv: float, regime: RegimeContext, signal: TradingSignal) -> float:
        htf_alignment = regime.trend_alignment if regime else "NEUTRAL"
        adx_value = regime.adx_value if regime else 0
        regime_confidence = regime.confidence if regime else 0

        if htf_alignment in ("BULLISH_ALIGNED", "BEARISH_ALIGNED") and signal.direction in ("BUY", "SELL"):
            is_aligned = (htf_alignment == "BULLISH_ALIGNED" and signal.direction == "BUY") or \
                         (htf_alignment == "BEARISH_ALIGNED" and signal.direction == "SELL")
            adx_factor = min(adx_value / 50.0, 1.0)
            conf_factor = 0.3 + (regime_confidence * 0.7)
            strength = min(1.0, adx_factor * conf_factor)
            if is_aligned:
                conv = min(1.0, conv * (1.0 + strength * 1.5))
            else:
                conv *= max(0.05, 1.0 - strength * 0.95)
        elif htf_alignment in ("HTF_BULLISH_LTF_BEARISH", "HTF_BEARISH_LTF_BULLISH") and signal.direction in ("BUY", "SELL"):
            htf_dir = "BUY" if htf_alignment == "HTF_BULLISH_LTF_BEARISH" else "SELL"
            is_aligned = signal.direction == htf_dir
            adx_factor = min(adx_value / 40.0, 1.0)
            conf_factor = 0.5 + (regime_confidence * 0.5)
            strength = min(1.0, adx_factor * conf_factor)
            if is_aligned:
                conv = min(1.0, conv * (1.0 + strength * 0.8))
            else:
                conv *= max(0.15, 1.0 - strength * 0.85)
        return conv

    def _adjust_conviction_for_consensus(self, conv: float, signal: TradingSignal) -> float:
        if signal.consensus and signal.direction in ("BUY", "SELL"):
            alignment = signal.consensus.alignment_with(signal.direction)
            if alignment < 0.3:
                conv *= 0.5
            elif alignment > 0.7:
                conv = min(1.0, conv * 1.3)
        return conv

    def _calc_sim_volume(self, signal: TradingSignal, conviction: float,
                         atr_val: float, entry: float, sl: float, tp: float) -> float:
        risk_pct = self.config.get("risk_per_trade", 0.01)
        kelly_fraction = self.kelly_risk.get_risk_fraction(conviction)
        dynamic_risk, _ = self.dynamic_var.get_risk_fraction(
            kelly_fraction=kelly_fraction,
            conviction=conviction,
            atr_ratio=max(atr_val / 20.0, 0.3),
            session_weight=1.0,
            correlation_factor=1.0,
        )
        risk_pct = min(dynamic_risk, 0.03)
        atr_val = max(atr_val, 0.0001)
        sl_distance = abs(entry - sl)
        if sl_distance <= 0:
            return 0.01
        vol = (risk_pct * 10000) / (sl_distance / atr_val)
        return max(0.01, round(min(vol, 1.0), 2))

    def _reconfigure(self, config: Dict):
        self.config = config
        strategy_cfg = config.get("strategy", {})
        adaptive = strategy_cfg.get("adaptive", {})
        from src.risk.fixed_risk_manager import RiskConfig
        mock_profile = type("MockProfile", (), {
            "symbol": self.symbol, "allowed_sessions": [],
            "max_concurrent_trades": 999, "cooldown_bars_m1": 0,
            "min_retracement": 0.5, "fvg_entry_level": 0.5,
            "sl_buffer_pips": 0.5, "sl_fixed_pips": 0.0,
            "sl_min_pips": 5.0, "sl_max_pips": 35.0,
            "min_rr_ratio": 4.0, "tp_fixed_pips": 0.0,
            "risk_per_trade_pct": 2.0, "max_volume": 0,
            "max_daily_loss_pct": 6.0, "max_daily_trades": 10,
            "scale_entries": 1, "scale_close_at_tp_pct": 0.5,
            "scale_close_ratio": 0.5,
        })()
        bt_params = type("BTParams", (), {
            "consolidation_max_atr_ratio": 0.6,
            "expansion_tick_acceleration": 1.8,
            "gap_detection_pips": 8.0, "macro_event_filter": False,
            "min_retracement_level": 0.50, "breakout_validation": "BodyClose",
            "fvg_min_size_atr_ratio": 0.3, "fvg_entry_level": 0.50,
            "void_min_size_pips": 5.0, "pivot_length": 10,
            "wyckoff_min_phase_bars": 12, "sequence_length": 3,
            "news_filter_active": False, "news_buffer_minutes": 30,
            "min_stop_distance_atr": 0.5, "min_reversal_score": 65.0,
            "min_net_score": adaptive.get("min_net_score", 5.0),
        })()
        self.strategy_engine = StrategyEngine(
            profile=mock_profile,
            params=bt_params,
            weights=None,
            min_score=adaptive.get("min_score_to_trade", 60),
            high_confidence_score=adaptive.get("high_confidence_score", 80),
            min_net_score=adaptive.get("min_net_score", 5),
        )
        rc = RiskConfig(
            risk_per_trade=config.get("risk_per_trade", 0.02),
            max_daily_loss=0.06,
            max_positions=5,
            atr_multiplier_sl=config.get("atr_multiplier_sl", 1.5),
            atr_multiplier_tp=config.get("atr_multiplier_tp", 2.0),
            min_reward_risk_ratio=config.get("min_reward_risk_ratio", 1.5),
        )
        self.risk_manager = FixedRiskManager(rc, balance=10_000.0)

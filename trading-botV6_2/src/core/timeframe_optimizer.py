"""Auto-Selección de Timeframes Óptimos — MEJORA 12
Analiza volatilidad y ruido para seleccionar TFs dinámicamente,
adaptando períodos de indicadores al régimen actual.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.helpers import atr

HTF_POOL = ["4H", "3H", "2H", "1H"]
MID_POOL = ["30min", "15min", "10min"]
LTF_POOL = ["5min", "3min", "1min"]


@dataclass
class IndicatorPeriods:
    atr_period: int = 14
    adx_period: int = 14
    ema_fast: int = 9
    ema_slow: int = 21
    swing_lookback: int = 5
    lookback_bars: int = 300

    def to_dict(self) -> dict:
        return {
            "atr_period": self.atr_period,
            "adx_period": self.adx_period,
            "ema_fast": self.ema_fast,
            "ema_slow": self.ema_slow,
            "swing_lookback": self.swing_lookback,
            "lookback_bars": self.lookback_bars,
        }


@dataclass
class TFVolatilityInfo:
    tf: str
    atr_value: float
    atr_pct: float
    noise_score: float = 0.0
    trend_consistency: float = 0.0
    weight: float = 1.0
    degraded: bool = False


@dataclass
class TimeframeSelection:
    htf: str
    mid: str
    ltf: str
    htf_pool: List[str] = field(default_factory=lambda: list(HTF_POOL))
    mid_pool: List[str] = field(default_factory=lambda: list(MID_POOL))
    ltf_pool: List[str] = field(default_factory=lambda: list(LTF_POOL))
    indicators: IndicatorPeriods = field(default_factory=IndicatorPeriods)
    volatility_regime: str = "MEDIUM"
    tf_info: Dict[str, TFVolatilityInfo] = field(default_factory=dict)
    degraded_tfs: List[str] = field(default_factory=list)

    @property
    def groups(self) -> Dict[str, List[str]]:
        return {
            "HTF": self.htf_pool,
            "MID": self.mid_pool,
            "LTF": self.ltf_pool,
        }


DEFAULT_CONFIG = {
    "atr_pct_high": 0.015,
    "atr_pct_low": 0.005,
    "noise_threshold": 0.65,
    "min_consistency": 0.30,
    "lookback_volatility": 100,
    "lookback_noise": 60,
    "enable_volatility_adaptation": True,
    "enable_noise_detection": True,
    "enable_indicator_adaptation": True,
    "noise_penalty_weight": 0.5,
}


class TimeframeOptimizer:
    """Selecciona timeframes óptimos según volatilidad y ruido."""

    def __init__(self, config: Optional[dict] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self._history: Dict[str, List[TFVolatilityInfo]] = {}
        self._last_selection: Dict[str, Optional[TimeframeSelection]] = {}

    def analyze(
        self, symbol: str, timeframes: Dict[str, pd.DataFrame],
    ) -> TimeframeSelection:
        available = {tf: df for tf, df in timeframes.items()
                     if df is not None and len(df) > 50}

        if not available:
            return self._default_selection()

        tf_info = self._compute_tf_info(available)
        degraded = self._detect_noisy_tfs(tf_info) if self.config["enable_noise_detection"] else []
        regime = self._classify_volatility(tf_info)

        selection = self._select_timeframes(tf_info, regime, degraded)
        selection.degraded_tfs = degraded
        selection.volatility_regime = regime
        selection.tf_info = tf_info
        selection.indicators = self._adapt_indicators(regime)

        self._last_selection[symbol] = selection
        return selection

    def get_last_selection(self, symbol: str) -> Optional[TimeframeSelection]:
        return self._last_selection.get(symbol)

    def _compute_tf_info(
        self, timeframes: Dict[str, pd.DataFrame],
    ) -> Dict[str, TFVolatilityInfo]:
        result = {}
        for tf, df in timeframes.items():
            try:
                atr_vals = atr(df, min(14, len(df) // 3))
                atr_val = atr_vals.iloc[-1]
                close = df["close"].iloc[-1]
                atr_pct = atr_val / close if close > 0 else 0

                noise, consistency = self._estimate_noise(df)

                result[tf] = TFVolatilityInfo(
                    tf=tf, atr_value=atr_val, atr_pct=atr_pct,
                    noise_score=noise, trend_consistency=consistency,
                )
            except Exception as e:
                logger.debug(f"[{symbol}] TF {tf}: skip ({e})")
                continue
        return result

    def _estimate_noise(self, df: pd.DataFrame) -> Tuple[float, float]:
        from src.utils.helpers import find_swing_points
        lookback = min(self.config["lookback_noise"], len(df) // 2)
        segment = df.iloc[-lookback:]

        highs_idx, lows_idx = find_swing_points(segment, lookback=5)
        total_swings = len(highs_idx) + len(lows_idx)

        if total_swings < 4:
            return 0.5, 0.3

        consistent_moves = 0
        dir_changes = 0
        for i in range(1, min(total_swings, 8)):
            prev_idx = min(i - 1, len(highs_idx) - 1, len(lows_idx) - 1)
            if i < len(highs_idx) and prev_idx < len(highs_idx):
                h_up = df["high"].iloc[highs_idx[i]] > df["high"].iloc[highs_idx[prev_idx]]
            else:
                h_up = True
            if i < len(lows_idx) and prev_idx < len(lows_idx):
                l_up = df["low"].iloc[lows_idx[i]] > df["low"].iloc[lows_idx[prev_idx]]
            else:
                l_up = True
            if h_up == l_up:
                consistent_moves += 1
            else:
                dir_changes += 1

        consistency = consistent_moves / max(consistent_moves + dir_changes, 1)
        noise = 1.0 - consistency if total_swings > 2 else 0.5
        return min(max(noise, 0), 1), min(max(consistency, 0), 1)

    def _detect_noisy_tfs(
        self, tf_info: Dict[str, TFVolatilityInfo],
    ) -> List[str]:
        degraded = []
        for tf, info in tf_info.items():
            if info.noise_score >= self.config["noise_threshold"]:
                info.degraded = True
                info.weight = self.config["noise_penalty_weight"]
                degraded.append(tf)
                logger.debug(f"TF {tf} degradado: noise={info.noise_score:.2f}")
        return degraded

    def _classify_volatility(
        self, tf_info: Dict[str, TFVolatilityInfo],
    ) -> str:
        ltfs = [v for k, v in tf_info.items() if k in LTF_POOL]
        if not ltfs:
            ltfs = list(tf_info.values())
        if not ltfs:
            return "MEDIUM"
        avg_atr_pct = np.mean([v.atr_pct for v in ltfs])
        if avg_atr_pct >= self.config["atr_pct_high"]:
            return "HIGH"
        elif avg_atr_pct <= self.config["atr_pct_low"]:
            return "LOW"
        return "MEDIUM"

    def _select_timeframes(
        self, tf_info: Dict[str, TFVolatilityInfo],
        regime: str, degraded: List[str],
    ) -> TimeframeSelection:
        if not self.config["enable_volatility_adaptation"]:
            return self._default_selection()

        htf_pool = list(HTF_POOL)
        mid_pool = list(MID_POOL)
        ltf_pool = list(LTF_POOL)

        if regime == "HIGH":
            htf_pool = [tf for tf in htf_pool if tf in ("2H", "1H")]
            if not htf_pool:
                htf_pool = ["1H"]
            mid_pool = [tf for tf in mid_pool if tf in ("15min", "10min")]
            if not mid_pool:
                mid_pool = ["15min"]
            ltf_pool = [tf for tf in ltf_pool if tf in ("3min", "1min")]
            if not ltf_pool:
                ltf_pool = ["1min"]
            logger.info(f"Vol HIGH: TFs rápidos {htf_pool[0]}/{mid_pool[0]}/{ltf_pool[0]}")
        elif regime == "LOW":
            htf_pool = [tf for tf in htf_pool if tf in ("4H", "3H")]
            if not htf_pool:
                htf_pool = ["4H"]
            mid_pool = [tf for tf in mid_pool if tf in ("30min", "15min")]
            if not mid_pool:
                mid_pool = ["30min"]
            ltf_pool = [tf for tf in ltf_pool if tf in ("5min", "3min")]
            if not ltf_pool:
                ltf_pool = ["5min"]
            logger.info(f"Vol LOW: TFs lentos {htf_pool[0]}/{mid_pool[0]}/{ltf_pool[0]}")
        else:
            logger.debug("Vol MEDIUM: TFs estándar")

        degraded_htf = [tf for tf in degraded if tf in htf_pool]
        degraded_mid = [tf for tf in degraded if tf in mid_pool]
        degraded_ltf = [tf for tf in degraded if tf in ltf_pool]

        if degraded_htf and len(htf_pool) > 1:
            htf_pool = [tf for tf in htf_pool if tf not in degraded_htf]
        if degraded_mid and len(mid_pool) > 1:
            mid_pool = [tf for tf in mid_pool if tf not in degraded_mid]
        if degraded_ltf and len(ltf_pool) > 1:
            ltf_pool = [tf for tf in ltf_pool if tf not in degraded_ltf]

        htf = self._pick_best(htf_pool, tf_info)
        mid = self._pick_best(mid_pool, tf_info)
        ltf = self._pick_best(ltf_pool, tf_info)

        return TimeframeSelection(
            htf=htf or htf_pool[0], mid=mid or mid_pool[0], ltf=ltf or ltf_pool[0],
            htf_pool=htf_pool, mid_pool=mid_pool, ltf_pool=ltf_pool,
        )

    def _pick_best(
        self, pool: List[str], tf_info: Dict[str, TFVolatilityInfo],
    ) -> Optional[str]:
        candidates = [(tf, tf_info.get(tf)) for tf in pool if tf in tf_info]
        if not candidates:
            return pool[0] if pool else None
        scored = [(tf, info.weight * (1.0 - info.noise_score) * (1.0 + info.trend_consistency))
                  for tf, info in candidates]
        scored.sort(key=lambda x: -x[1])
        return scored[0][0]

    def _adapt_indicators(self, regime: str) -> IndicatorPeriods:
        if not self.config["enable_indicator_adaptation"]:
            return IndicatorPeriods()
        if regime == "HIGH":
            return IndicatorPeriods(
                atr_period=10, adx_period=10,
                ema_fast=7, ema_slow=17,
                swing_lookback=3, lookback_bars=200,
            )
        elif regime == "LOW":
            return IndicatorPeriods(
                atr_period=24, adx_period=20,
                ema_fast=13, ema_slow=34,
                swing_lookback=7, lookback_bars=400,
            )
        return IndicatorPeriods(
            atr_period=14, adx_period=14,
            ema_fast=9, ema_slow=21,
            swing_lookback=5, lookback_bars=300,
        )

    def _default_selection(self) -> TimeframeSelection:
        return TimeframeSelection(
            htf="1H", mid="15min", ltf="5min",
            htf_pool=list(HTF_POOL), mid_pool=list(MID_POOL), ltf_pool=list(LTF_POOL),
        )

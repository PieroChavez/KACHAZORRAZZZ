"""Fractal Cascade Strategy — pure price-action structural trading
Multi-TF hunter (4H, 2H, 30min, 15min) with proximity alerts + independent 5M
sub-fractal hunting. Triple order packs with dynamic trailing.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timedelta
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from loguru import logger

import pandas as pd
import numpy as np

from ..core.multi_timeframe import (
    MultiTimeframeFetcher, TIMEFRAME_ORDER, TIMEFRAME_GROUPS, HISTORICAL_COUNT
)
from ..core.session_profiler import SessionProfiler, TradingSession
from ..core.regime_detector import RegimeDetector
from ..utils.helpers import pip_size, atr, find_swing_points
from ..adapters.mt5_client import MT5Client
from ..neural.integrator import NeuralAdvisor
from ..learning.meta_learner import MetaLearner, TradeRecord
from .fractal_db import FractalDB, Fractal
from .order_pack import OrderPackManager
from .fractal_learner import FractalLearner


FIB_LEVEL = 0.72
PROXIMITY_PIPS = 5.0
MACRO_TFS = ["4H", "2H", "30min", "15min"]


class FractalCascadeStrategy:
    def __init__(self, symbol: str, mt5_client: MT5Client,
                 fetcher: MultiTimeframeFetcher,
                 meta_learner: Optional[MetaLearner] = None):
        self.symbol = symbol
        self.mt5 = mt5_client
        self.fetcher = fetcher
        self.pip = pip_size(symbol)
        self.db = FractalDB(symbol)
        self.orders = OrderPackManager(mt5_client, symbol, copy_enabled=False)
        self.learner = FractalLearner(symbol)
        self.meta_learner = meta_learner
        self.neural_advisor = NeuralAdvisor()
        self.regime_detector = RegimeDetector()
        self.session_profiler = SessionProfiler()
        self._prev_swings: Dict[str, dict] = {}
        self._alerts: Dict[int, datetime] = {}
        self._last_analysis: Optional[datetime] = None
        self._last_scanned_candle: Dict[str, pd.Timestamp] = {}
        self._pack_contexts: Dict[int, dict] = {}
        self._current_regime = None
        self._current_session = None

    # ── Main Entry ─────────────────────────────────────────────────────

    def manage_orders(self):
        self.orders.sync_manually_closed()
        self.orders.manage_all(datetime.utcnow(), None)
        for pid in self.orders.pop_closed_pack_ids():
            self._record_pack_outcome(pid)

    def evaluate(self, timeframes: Dict[str, pd.DataFrame], current_time: datetime,
                 skip_entries: bool = False):

        df_5m = timeframes.get("5min")
        self.orders.manage_all(current_time, df_5m)
        for pid in self.orders.pop_closed_pack_ids():
            self._record_pack_outcome(pid)

        htf_df = timeframes.get("4H")
        if htf_df is None:
            htf_df = df_5m
        ltf_df = df_5m
        if ltf_df is None:
            ltf_df = htf_df
        regime_ctx = self.regime_detector.detect(htf_df, ltf_df)
        session_ctx = self.session_profiler.get_session()
        self._current_regime = regime_ctx
        self._current_session = session_ctx

        if skip_entries:
            return

        self._scan_macro_fractals(timeframes)
        self._scan_subfractals_5m(timeframes.get("5min"))
        self._check_entry_conditions(timeframes, current_time)
        self._cleanse_fractals()

        analysis_interval = timedelta(hours=4)
        if (self._last_analysis is None
                or current_time - self._last_analysis > analysis_interval):
            result = self.learner.analyze()
            if result.get("analyzed"):
                for adj in result.get("adjustments", []):
                    logger.info(f"[FractalLearner] {adj}")
            self._last_analysis = current_time

    # ── Macro Fractal Scanning (4H, 2H, 30min, 15min) ─────────────────

    def _scan_macro_fractals(self, timeframes: Dict[str, pd.DataFrame]):
        for tf in MACRO_TFS:
            df = timeframes.get(tf)
            if df is None or len(df) < 30:
                continue
            last_time = df["time"].iloc[-1]
            prev_time = self._last_scanned_candle.get(tf)
            if prev_time is not None and prev_time == last_time:
                continue
            self._last_scanned_candle[tf] = last_time
            self._detect_bullish(tf, df, is_subfractal=False)
            self._detect_bearish(tf, df, is_subfractal=False)

    # ── Independent 5M Sub‑fractal Hunting ────────────────────────────

    def _scan_subfractals_5m(self, df_5m: Optional[pd.DataFrame]):
        if df_5m is None or len(df_5m) < 30:
            return
        self._detect_bullish("5min", df_5m, is_subfractal=True)
        self._detect_bearish("5min", df_5m, is_subfractal=True)

    # ── CHoCH / BOS Detection ─────────────────────────────────────────

    def _detect_bullish(self, tf: str, df: pd.DataFrame, is_subfractal: bool = False):
        highs, lows = find_swing_points(df, lookback=3)
        if len(highs) < 3 or len(lows) < 3:
            return

        prev_key = f"{tf}_bullish_{int(is_subfractal)}"
        last_low_idx = lows[-1]
        last_low_price = df["low"].iloc[last_low_idx]

        cached = self._prev_swings.get(prev_key)
        if cached and cached["low_idx"] == last_low_idx:
            return

        for i in range(1, len(highs)):
            curr_high_idx = highs[i]
            for j in range(len(lows)):
                low_idx = lows[j]
                if low_idx > curr_high_idx:
                    low_price = df["low"].iloc[low_idx]
                    if low_price > df["low"].iloc[lows[j - 1]]:
                        self._register_bullish(tf, df, low_idx, curr_high_idx, is_subfractal)
                        break

        self._prev_swings[prev_key] = {"low_idx": last_low_idx,
                                        "low_price": last_low_price}

    def _detect_bearish(self, tf: str, df: pd.DataFrame, is_subfractal: bool = False):
        highs, lows = find_swing_points(df, lookback=3)
        if len(highs) < 3 or len(lows) < 3:
            return

        prev_key = f"{tf}_bearish_{int(is_subfractal)}"
        last_high_idx = highs[-1]
        last_high_price = df["high"].iloc[last_high_idx]

        cached = self._prev_swings.get(prev_key)
        if cached and cached["high_idx"] == last_high_idx:
            return

        for i in range(1, len(lows)):
            curr_low_idx = lows[i]
            for j in range(len(highs)):
                high_idx = highs[j]
                if high_idx > curr_low_idx:
                    high_price = df["high"].iloc[high_idx]
                    if high_price < df["high"].iloc[highs[j - 1]]:
                        self._register_bearish(tf, df, high_idx, curr_low_idx, is_subfractal)
                        break

        self._prev_swings[prev_key] = {"high_idx": last_high_idx,
                                        "high_price": last_high_price}

    # ── Fractal Registration ───────────────────────────────────────────

    def _register_bullish(self, tf: str, df: pd.DataFrame,
                           low_idx: int, high_idx: int, is_subfractal: bool):
        level1 = df["low"].iloc[low_idx]
        level0 = df["high"].iloc[high_idx]
        if level0 <= level1:
            return
        fib_range = level0 - level1
        fib_072 = level1 + FIB_LEVEL * fib_range
        if self._has_duplicate(tf, "bullish", level1, is_subfractal):
            return

        f = Fractal(
            symbol=self.symbol, timeframe=tf, direction="bullish",
            level0=level0, level1=level1, fib_072=fib_072,
            swing_high=level0, swing_low=level1,
            bos_index=high_idx, is_subfractal=is_subfractal,
            bos_time=df["time"].iloc[high_idx].to_pydatetime(),
            note=f"BULL {tf} {'SUB' if is_subfractal else 'MACRO'}"
                 f" L1={level1:.2f} L0={level0:.2f}"
        )
        self.db.add_fractal(f)

    def _register_bearish(self, tf: str, df: pd.DataFrame,
                           high_idx: int, low_idx: int, is_subfractal: bool):
        level1 = df["high"].iloc[high_idx]
        level0 = df["low"].iloc[low_idx]
        if level1 <= level0:
            return
        fib_range = level1 - level0
        fib_072 = level0 + FIB_LEVEL * fib_range
        if self._has_duplicate(tf, "bearish", level1, is_subfractal):
            return

        f = Fractal(
            symbol=self.symbol, timeframe=tf, direction="bearish",
            level0=level0, level1=level1, fib_072=fib_072,
            swing_high=level1, swing_low=level0,
            bos_index=low_idx, is_subfractal=is_subfractal,
            bos_time=df["time"].iloc[low_idx].to_pydatetime(),
            note=f"BEAR {tf} {'SUB' if is_subfractal else 'MACRO'}"
                 f" L1={level1:.2f} L0={level0:.2f}"
        )
        self.db.add_fractal(f)

    def _has_duplicate(self, tf: str, direction: str, level1: float,
                        is_subfractal: bool = False) -> bool:
        for ef in self.db.get_active_fractals():
            if (ef.timeframe == tf and ef.direction == direction
                    and ef.is_subfractal == is_subfractal):
                if abs(ef.level1 - level1) / (abs(ef.level0 - ef.level1) + 1e-10) < 0.1:
                    return True
        return False

    # ── Entry Logic ───────────────────────────────────────────────────

    def _check_entry_conditions(self, timeframes: Dict[str, pd.DataFrame],
                                 current_time: datetime):
        candidates = self.db.get_active_not_hit()
        if not candidates:
            return
        price = self._current_price()
        if price is None:
            return
        df_5m = timeframes.get("5min")
        info = self.mt5.get_symbol_info(self.symbol)
        bid = info["bid"] if info else price - 0.5
        ask = info["ask"] if info else price + 0.5

        for f in candidates:
            if not self._fractal_valid(f, price):
                self.db.invalidate(f.id)
                self._alerts.pop(f.id, None)
                tag = "SUB" if f.is_subfractal else "MACRO"
                logger.info(f"[{self.symbol}] {tag} #{f.id} invalidado (nivel 1 violado)")
                continue

            if self._fractal_stale(f, price):
                self._cancel_pending_for(f)
                self.db.invalidate(f.id)
                self._alerts.pop(f.id, None)
                tag = "SUB" if f.is_subfractal else "MACRO"
                logger.info(f"[{self.symbol}] {tag} #{f.id} invalidado (estructura superada)")
                continue

            is_buy = f.direction == "bullish"
            wrong_side = (is_buy and f.fib_072 >= ask) or (not is_buy and f.fib_072 <= bid)
            if wrong_side:
                self._cancel_pending_for(f)
                self.db.invalidate(f.id)
                self._alerts.pop(f.id, None)
                tag = "SUB" if f.is_subfractal else "MACRO"
                logger.info(f"[{self.symbol}] {tag} #{f.id} invalidado (entry en lado incorrecto)")
                continue

            sl_dist = abs(f.fib_072 - f.level1)
            price_dist = abs(f.fib_072 - price)
            if price_dist > sl_dist:
                self._cancel_pending_for(f)
                self.db.invalidate(f.id)
                self._alerts.pop(f.id, None)
                tag = "SUB" if f.is_subfractal else "MACRO"
                logger.info(f"[{self.symbol}] {tag} #{f.id} invalidado (limit muy lejos: "
                           f"price_dist={price_dist:.2f} > sl_dist={sl_dist:.2f})")
                continue

            if not self._macro_bias_allows(f, df_5m):
                continue

            if not self._hh_ll_confirms(f, df_5m):
                continue

            if f.is_subfractal or f.timeframe == "15min":
                self._execute_entry(f, price, df_5m)
            else:
                if self._get_confirmation(f, df_5m):
                    self._execute_entry(f, price, df_5m)

    def _execute_entry(self, f: Fractal, price: float, df_5m: pd.DataFrame):
        session = self._current_session or self.session_profiler.get_session()
        vol_total = self._calc_volume(f, session)
        direction = "BUY" if f.direction == "bullish" else "SELL"
        try:
            pack = self.orders.place_pack(
                f.id, direction, f.fib_072, f.level1, f.timeframe, vol_total
            )
            if pack:
                self.db.mark_entry_hit(f.id, f.fib_072, f.level1)
                self._alerts.pop(f.id, None)
                range_size = abs(f.level0 - f.level1)
                sess_name = session.value if session else ""
                self.learner.record_entry(
                    pack.id, f.id, f.timeframe, f.direction,
                    f.is_subfractal, pack.entry_price, pack.sl_initial,
                    pack.tp1, vol_total, range_size,
                    f.fib_072, sess_name
                )
                regime = self._current_regime
                self._pack_contexts[pack.id] = {
                    "direction": direction,
                    "entry_price": pack.entry_price,
                    "volume": vol_total,
                    "session": sess_name,
                    "regime": regime.regime.value if regime else "NEUTRAL",
                    "regime_confidence": regime.confidence if regime else 0.5,
                    "entry_time": datetime.utcnow(),
                }
        except Exception:
            logger.exception(f"[{self.symbol}] Error al ejecutar entrada fractal #{f.id}, invalidando")
            self.db.invalidate(f.id)
            self._alerts.pop(f.id, None)
            return

    def _current_price(self) -> Optional[float]:
        info = self.mt5.get_symbol_info(self.symbol)
        if info:
            return (info["ask"] + info["bid"]) / 2
        return None

    def _fractal_valid(self, f: Fractal, price: float) -> bool:
        buf = 0.02
        if f.direction == "bullish":
            return price > f.level1 * (1 - buf)
        else:
            return price < f.level1 * (1 + buf)

    def _fractal_stale(self, f: Fractal, price: float) -> bool:
        buf = 0.01
        if f.direction == "bullish":
            return price > f.level0 * (1 + buf)
        else:
            return price < f.level0 * (1 - buf)

    def _cancel_pending_for(self, f: Fractal):
        active_packs = self.orders.get_active_packs()
        for pack in active_packs:
            if pack.fractal_id == f.id:
                self.orders.cancel_pack(pack.id)
                break

    def _macro_bias_allows(self, f: Fractal, df_5m: pd.DataFrame) -> bool:
        active = self.db.get_active_fractals()
        macro_bearish = any(
            not ef.is_subfractal and ef.direction == "bearish" and ef.active
            for ef in active
        )
        macro_bullish = any(
            not ef.is_subfractal and ef.direction == "bullish" and ef.active
            for ef in active
        )

        if macro_bearish and f.direction == "bullish":
            if df_5m is None or len(df_5m) < 20:
                return False
            _, lows = find_swing_points(df_5m, lookback=3)
            if len(lows) < 2:
                logger.info(f"[{self.symbol}] Fractal #{f.id} BUY bloqueado por macro bajista (sin HL en 5M)")
                return False
            hl_detected = df_5m["low"].iloc[lows[-1]] > df_5m["low"].iloc[lows[-2]]
            if not hl_detected:
                logger.info(f"[{self.symbol}] Fractal #{f.id} BUY bloqueado por macro bajista (sin HL en 5M)")
                return False
            return True

        if macro_bullish and f.direction == "bearish":
            if df_5m is None or len(df_5m) < 20:
                return False
            highs, _ = find_swing_points(df_5m, lookback=3)
            if len(highs) < 2:
                logger.info(f"[{self.symbol}] Fractal #{f.id} SELL bloqueado por macro alcista (sin LH en 5M)")
                return False
            lh_detected = df_5m["high"].iloc[highs[-1]] < df_5m["high"].iloc[highs[-2]]
            if not lh_detected:
                logger.info(f"[{self.symbol}] Fractal #{f.id} SELL bloqueado por macro alcista (sin LH en 5M)")
                return False
            return True

        return True

    def _hh_ll_confirms(self, f: Fractal, df_5m: pd.DataFrame) -> bool:
        if df_5m is None or len(df_5m) < 20:
            return False
        highs, lows = find_swing_points(df_5m, lookback=3)
        if f.direction == "bullish":
            if len(highs) < 2:
                logger.info(f"[{self.symbol}] Fractal #{f.id} BUY saltado: sin estructura HH/LH en 5M")
                return False
            return True
        else:
            if len(lows) < 2:
                logger.info(f"[{self.symbol}] Fractal #{f.id} SELL saltado: sin estructura LL/HL en 5M")
                return False
            return True

    def _get_confirmation(self, f: Fractal, df_5m: pd.DataFrame) -> bool:
        if df_5m is None or len(df_5m) < 10:
            return False
        body = abs(df_5m["close"].iloc[-1] - df_5m["open"].iloc[-1])
        total_range = df_5m["high"].iloc[-1] - df_5m["low"].iloc[-1]
        if total_range == 0:
            return False
        body_ratio = body / total_range
        close = df_5m["close"].iloc[-1]
        if f.direction == "bullish" and close > df_5m["open"].iloc[-1]:
            return True
        if f.direction == "bearish" and close < df_5m["open"].iloc[-1]:
            return True
        return body_ratio > 0.5

    def _record_pack_outcome(self, pack_id: int):
        pack = self.orders.get_pack_by_id(pack_id)
        if not pack or pack.status == "active":
            return
        profit = self.orders.get_pack_total_profit(pack_id)
        outcome = "win" if profit > 0 else "loss"
        subs = self.orders.get_all_subs(pack_id)
        exit_price = 0.0
        for s in subs:
            if s.closed_at:
                exit_price = s.sl_current
                break
        self.learner.record_exit(pack_id, outcome, exit_price, profit)

        if self.meta_learner is None:
            return
        ctx = self._pack_contexts.pop(pack_id, {})
        if not ctx:
            return
        duration = 0.0
        entry_time = ctx.get("entry_time")
        if entry_time:
            duration = (datetime.utcnow() - entry_time).total_seconds() / 60.0
        trade = TradeRecord(
            symbol=self.symbol,
            direction=ctx.get("direction", pack.direction),
            entry_price=ctx.get("entry_price", pack.entry_price),
            exit_price=exit_price,
            volume=ctx.get("volume", pack.volume_total),
            profit=profit,
            score=50.0,
            conviction=0.5,
            regime=ctx.get("regime", "NEUTRAL"),
            session=ctx.get("session", ""),
            primary_pattern="FRACTAL",
            patterns_found=["FRACTAL"],
            regime_confidence=ctx.get("regime_confidence", 0.5),
            exit_reason=outcome,
            duration_minutes=int(duration),
            timestamp=datetime.utcnow(),
        )
        self.meta_learner.record_trade(trade)

    def _calc_volume(self, f: Fractal,
                     session: Optional[TradingSession] = None) -> float:
        return 0.1

    # ── Lifecycle Cleanup ─────────────────────────────────────────────

    def _cleanse_fractals(self):
        active_packs = self.orders.get_active_packs()
        packed_fids = {p.fractal_id for p in active_packs}

        for f in self.db.get_active_fractals():
            if f.hit_entry and f.id not in packed_fids:
                self.db.invalidate(f.id)
                logger.info(f"[{self.symbol}] Fractal #{f.id} limpiado (pack cerrado)")

    # ── Status / Dashboard ────────────────────────────────────────────

    def get_status(self) -> dict:
        active = self.db.get_active_fractals()
        not_hit = [f for f in active if not f.hit_entry]
        packs = self.orders.get_pack_summary()

        macro_fractals = [f for f in active if not f.is_subfractal]
        sub_fractals = [f for f in active if f.is_subfractal]

        alerts = [
            {"id": fid, "tf": self.db.get_by_id(fid).timeframe,
             "dir": self.db.get_by_id(fid).direction,
             "072": self.db.get_by_id(fid).fib_072}
            for fid in self._alerts
            if self.db.get_by_id(fid) and self.db.get_by_id(fid).active
        ]

        learner_summary = self.learner.get_summary()

        return {
            "symbol": self.symbol,
            "active_fractals": len(active),
            "macro_fractals": len(macro_fractals),
            "subfractals_5m": len(sub_fractals),
            "alerts": len(alerts),
            "alert_list": alerts,
            "awaiting_entry": len(not_hit),
            "active_packs": len(packs),
            "learner": {
                "trades": learner_summary["closed_trades"],
                "win_rate": f"{learner_summary['win_rate']:.0%}",
                "profit": learner_summary["total_profit"],
                "volume_mults": learner_summary["volume_multipliers"],
            },
            "fractals": [
                {"id": f.id, "tf": f.timeframe, "dir": f.direction,
                 "072": f.fib_072, "L1": f.level1, "L0": f.level0,
                 "hit": f.hit_entry, "sub": f.is_subfractal}
                for f in active
            ],
            "packs": packs,
        }

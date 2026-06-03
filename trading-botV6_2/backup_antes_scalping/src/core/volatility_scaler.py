"""Volatility Scaler — Dynamic SL/TP Multipliers (Modo Experto)
Adjusts SL/TP distances based on intrabar volatility metrics,
session profiles, multi-timeframe analysis, adaptive baselines,
volatility prediction, and Bollinger/Keltner channel width.
"""
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.helpers import atr

logger = logging.getLogger(__name__)

# Session time ranges in UTC hour
SESSION_RANGES = {
    "asia": (0, 8),
    "london_open": (8, 9),
    "london": (9, 13),
    "ny_open": (13, 14),
    "overlap": (13, 16),
    "ny": (14, 17),
    "ny_afternoon": (17, 22),
}

SESSION_MULTIPLIERS = {
    "asia": {"sl": 0.85, "tp": 0.80, "note": "Sesión Asia — baja volatilidad"},
    "london_open": {"sl": 1.15, "tp": 1.20, "note": "Apertura Londres — volatilidad creciente"},
    "london": {"sl": 1.05, "tp": 1.10, "note": "Sesión Londres"},
    "ny_open": {"sl": 1.25, "tp": 1.30, "note": "Apertura NY — alta volatilidad"},
    "overlap": {"sl": 1.20, "tp": 1.25, "note": "Solapamiento Londres+NY — máxima volatilidad"},
    "ny": {"sl": 1.15, "tp": 1.20, "note": "Sesión NY"},
    "ny_afternoon": {"sl": 0.90, "tp": 0.85, "note": "Tarde NY — volatilidad decreciente"},
}

DEFAULT_SESSION = {"sl": 1.0, "tp": 1.0, "note": "Fuera de sesiones principales"}


class VolatilityScaler:
    def __init__(self, atr_period: int = 14, expert_mode: bool = True):
        self.atr_period = atr_period
        self.expert_mode = expert_mode
        self._cache: Dict[str, dict] = {}
        self._baselines: Dict[str, dict] = {}
        self._prediction_cache: Dict[str, list] = {}

    def compute(self, symbol: str, ltf_df: pd.DataFrame,
                base_sl_mult: float = 1.5, base_tp_mult: float = 2.0) -> Dict:
        metrics = self._all_metrics(symbol, ltf_df)
        if metrics is None:
            return {"sl_mult": base_sl_mult, "tp_mult": base_tp_mult,
                    "reason": "insufficient_data"}

        self._cache[symbol] = metrics

        raw_sl_mult, raw_tp_mult = self._dynamic_mults(metrics, base_sl_mult, base_tp_mult)

        if self.expert_mode:
            sess_mult = self._session_multiplier()
            raw_sl_mult *= sess_mult["sl"]
            raw_tp_mult *= sess_mult["tp"]

            adp = self._adaptive_factor(symbol, metrics)
            raw_sl_mult *= adp["sl"]
            raw_tp_mult *= adp["tp"]

            pred = self._volatility_prediction(symbol, metrics)
            raw_sl_mult *= pred["sl"]
            raw_tp_mult *= pred["tp"]

            mtf = self._multi_tf_factor(ltf_df)
            raw_sl_mult *= mtf["sl"]
            raw_tp_mult *= mtf["tp"]

        sl_mult = max(0.5, min(3.5, raw_sl_mult))
        tp_mult = max(0.5, min(5.0, raw_tp_mult))

        parts = []
        if metrics.get("intrabar_ratio"):
            parts.append(f"intrabar={metrics['intrabar_ratio']:.2f}")
        if metrics.get("atr_roc"):
            parts.append(f"atr_roc={metrics['atr_roc']:.2f}")
        if metrics.get("volume_ratio"):
            parts.append(f"vol={metrics['volume_ratio']:.2f}")
        if metrics.get("wick_ratio"):
            parts.append(f"wick={metrics['wick_ratio']:.2f}")
        if self.expert_mode:
            parts.append(sess_mult["note"])
            if adp["note"]:
                parts.append(adp["note"])
            if pred["note"]:
                parts.append(pred["note"])
            if mtf["note"]:
                parts.append(mtf["note"])
            if metrics.get("bb_width_ratio"):
                parts.append(f"bb={metrics['bb_width_ratio']:.2f}")
            if metrics.get("kc_width_ratio"):
                parts.append(f"kc={metrics['kc_width_ratio']:.2f}")

        result = {
            "sl_mult": round(sl_mult, 2),
            "tp_mult": round(tp_mult, 2),
            "reason": "; ".join(parts) if parts else "base",
            **metrics,
        }
        return result

    def _all_metrics(self, symbol: str, df: pd.DataFrame) -> Optional[dict]:
        base = self._base_metrics(df)
        if base is None:
            return None
        if self.expert_mode:
            bb = self._bollinger_metrics(df)
            kc = self._keltner_metrics(df)
            base.update(bb)
            base.update(kc)
        return base

    def _base_metrics(self, df: pd.DataFrame) -> Optional[dict]:
        if df is None or len(df) < self.atr_period + 5:
            return None
        atr_vals = atr(df, self.atr_period)
        if len(atr_vals) < 2:
            return None
        current_atr = float(atr_vals.iloc[-1])
        atr_sma = atr_vals.iloc[-50:].mean() if len(atr_vals) >= 50 else atr_vals.mean()
        atr_ratio = current_atr / atr_sma if atr_sma > 0 else 1.0
        atr_roc = current_atr / atr_vals.iloc[-10] if len(atr_vals) >= 10 and atr_vals.iloc[-10] > 0 else 1.0
        last3 = df.iloc[-3:]
        candle_ranges = (last3["high"] - last3["low"]).values
        avg_range = float(np.mean(candle_ranges))
        intrabar_ratio = avg_range / current_atr if current_atr > 0 else 1.0
        vol_series = df["volume"].iloc[-20:] if len(df) >= 20 else df["volume"]
        vol_mean = float(vol_series.mean())
        vol_current = float(df["volume"].iloc[-1])
        volume_ratio = vol_current / vol_mean if vol_mean > 0 else 1.0
        body = abs(last3["close"] - last3["open"]).values
        wick_total = candle_ranges - body
        wick_avg = float(np.mean(wick_total))
        wick_ratio = wick_avg / avg_range if avg_range > 0 else 0.5
        wide_count = int(sum(1 for r in candle_ranges if r > current_atr * 1.2))
        return {
            "current_atr": current_atr, "atr_ratio": atr_ratio, "atr_roc": atr_roc,
            "intrabar_ratio": intrabar_ratio, "volume_ratio": volume_ratio,
            "wick_ratio": wick_ratio, "wide_count": wide_count,
        }

    def _bollinger_metrics(self, df: pd.DataFrame) -> dict:
        if len(df) < 22:
            return {"bb_width_ratio": 1.0, "bb_position": 0.5}
        close = df["close"]
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        bb_width = (bb_upper - bb_lower) / sma20
        bb_width_hist = bb_width.iloc[-50:].mean() if len(bb_width) >= 50 else bb_width.mean()
        bb_width_current = float(bb_width.iloc[-1])
        bb_width_ratio = bb_width_current / bb_width_hist if bb_width_hist > 0 else 1.0
        bb_pos = (close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) if (bb_upper.iloc[-1] - bb_lower.iloc[-1]) > 0 else 0.5
        bb_pos = float(max(0, min(1, bb_pos)))
        return {"bb_width_ratio": round(bb_width_ratio, 4), "bb_position": round(bb_pos, 3)}

    def _keltner_metrics(self, df: pd.DataFrame) -> dict:
        if len(df) < 22:
            return {"kc_width_ratio": 1.0}
        typical = (df["high"] + df["low"] + df["close"]) / 3
        ema20 = typical.ewm(span=20).mean()
        atr14 = atr(df, 14)
        kc_upper = ema20 + 2 * atr14
        kc_lower = ema20 - 2 * atr14
        kc_width = (kc_upper - kc_lower) / ema20
        kc_hist = kc_width.iloc[-50:].mean() if len(kc_width) >= 50 else kc_width.mean()
        kc_current = float(kc_width.iloc[-1])
        kc_ratio = kc_current / kc_hist if kc_hist > 0 else 1.0
        return {"kc_width_ratio": round(kc_ratio, 4)}

    def _session_multiplier(self) -> dict:
        now = datetime.now(timezone.utc)
        hour = now.hour
        for name, (start, end) in SESSION_RANGES.items():
            if start <= hour < end:
                return SESSION_MULTIPLIERS[name]
        return DEFAULT_SESSION

    def _adaptive_factor(self, symbol: str, metrics: dict) -> dict:
        atr_ratio = metrics.get("atr_ratio", 1.0)
        bl = self._baselines.get(symbol)
        if bl is None:
            self._baselines[symbol] = {"atr_mean": atr_ratio, "count": 1}
            return {"sl": 1.0, "tp": 1.0, "note": ""}
        alpha = min(0.1, 1.0 / (bl["count"] + 1))
        bl["atr_mean"] = bl["atr_mean"] * (1 - alpha) + atr_ratio * alpha
        bl["count"] += 1
        dev = atr_ratio / bl["atr_mean"] if bl["atr_mean"] > 0 else 1.0
        if dev > 1.3:
            adj = 1.0 + (dev - 1.3) * 0.3
            return {"sl": adj, "tp": adj, "note": f"atr+{dev:.1f}x vs baseline"}
        elif dev < 0.7:
            adj = 0.85 + (dev / 0.7) * 0.15
            return {"sl": adj, "tp": adj, "note": f"atr-{dev:.1f}x vs baseline"}
        return {"sl": 1.0, "tp": 1.0, "note": ""}

    def _volatility_prediction(self, symbol: str, metrics: dict) -> dict:
        current_atr = metrics.get("current_atr", 0)
        if current_atr <= 0:
            return {"sl": 1.0, "tp": 1.0, "note": ""}
        cache = self._prediction_cache.setdefault(symbol, [])
        cache.append(current_atr)
        if len(cache) > 10:
            cache.pop(0)
        if len(cache) < 4:
            return {"sl": 1.0, "tp": 1.0, "note": ""}
        recent = cache[-3:]
        slope = (recent[-1] - recent[0]) / max(recent[0], 1e-10)
        if slope > 0.08:
            widen = 1.0 + slope * 0.5
            return {"sl": min(widen, 1.3), "tp": min(widen * 1.1, 1.4), "note": f"atr↑{slope*100:.0f}%"}
        elif slope < -0.08:
            shrink = 1.0 + slope * 0.3
            return {"sl": max(shrink, 0.8), "tp": max(shrink, 0.8), "note": f"atr↓{abs(slope)*100:.0f}%"}
        return {"sl": 1.0, "tp": 1.0, "note": ""}

    def _multi_tf_factor(self, ltf_df: pd.DataFrame) -> dict:
        if ltf_df is None or len(ltf_df) < 200:
            return {"sl": 1.0, "tp": 1.0, "note": ""}
        try:
            idx = pd.DatetimeIndex(ltf_df.index) if not isinstance(ltf_df.index, pd.DatetimeIndex) else ltf_df.index
            df_copy = ltf_df.copy()
            df_copy.index = idx
            df5 = df_copy.resample("5min").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna()
            df15 = df_copy.resample("15min").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna()
        except Exception:
            return {"sl": 1.0, "tp": 1.0, "note": ""}
        if len(df5) < 10 or len(df15) < 5:
            return {"sl": 1.0, "tp": 1.0, "note": ""}
        atr1 = atr(ltf_df, self.atr_period).iloc[-1] if len(ltf_df) > self.atr_period else 0
        atr5 = atr(df5, self.atr_period).iloc[-1] if len(df5) > self.atr_period else 0
        atr15 = atr(df15, self.atr_period).iloc[-1] if len(df15) > self.atr_period else 0
        if atr1 <= 0:
            return {"sl": 1.0, "tp": 1.0, "note": ""}
        norm5 = atr5 / atr1 if atr1 > 0 else 1.0
        norm15 = atr15 / atr1 if atr1 > 0 else 1.0
        combined = (norm5 + norm15) / 2
        if combined > 1.3:
            adj = 1.0 + (combined - 1.3) * 0.2
            return {"sl": adj, "tp": adj, "note": f"mtf={combined:.2f}x"}
        elif combined < 0.7:
            adj = 0.85 + (combined / 0.7) * 0.15
            return {"sl": adj, "tp": adj, "note": f"mtf={combined:.2f}x"}
        return {"sl": 1.0, "tp": 1.0, "note": ""}

    def _dynamic_mults(self, m: dict, base_sl: float, base_tp: float) -> Tuple[float, float]:
        sl_mult = base_sl
        tp_mult = base_tp
        ir = m.get("intrabar_ratio", 1.0)
        if ir > 1.5:
            excess = ir - 1.5
            sl_mult *= 1.0 + excess * 0.6
            tp_mult *= 1.0 + excess * 0.66
        elif ir < 0.6:
            shrink = 0.75 + (ir / 0.6) * 0.25
            sl_mult *= shrink
            tp_mult *= shrink
        ar = m.get("atr_ratio", 1.0)
        if ar > 1.3:
            sl_mult *= 1.0 + (ar - 1.3) * 0.5
            tp_mult *= 1.0 + (ar - 1.3) * 0.6
        elif ar < 0.7:
            sl_mult *= 0.8 + (ar / 0.7) * 0.2
            tp_mult *= 0.8 + (ar / 0.7) * 0.2
        aroc = m.get("atr_roc", 1.0)
        if aroc > 1.2:
            sl_mult *= 1.0 + (aroc - 1.2) * 0.4
            tp_mult *= 1.0 + (aroc - 1.2) * 0.5
        elif aroc < 0.8:
            sl_mult *= 0.85 + (aroc / 0.8) * 0.15
            tp_mult *= 0.85 + (aroc / 0.8) * 0.15
        vr = m.get("volume_ratio", 1.0)
        if vr > 1.8 and ir > 1.3:
            confirm = 1.0 + (vr - 1.8) * 0.15
            sl_mult *= confirm
            tp_mult *= confirm
        wr = m.get("wick_ratio", 0.5)
        if wr > 0.6:
            sl_mult *= 1.0 + (wr - 0.6) * 0.5
        elif wr < 0.2 and ir < 0.7:
            sl_mult *= 0.85
        wc = m.get("wide_count", 0)
        if wc >= 2:
            cluster = 1.0 + wc * 0.1
            sl_mult *= cluster
            tp_mult *= cluster
        if self.expert_mode:
            bb_r = m.get("bb_width_ratio", 1.0)
            if bb_r > 1.3:
                sl_mult *= 1.0 + (bb_r - 1.3) * 0.25
                tp_mult *= 1.0 + (bb_r - 1.3) * 0.3
            elif bb_r < 0.7:
                sl_mult *= 0.85
                tp_mult *= 0.85
            kc_r = m.get("kc_width_ratio", 1.0)
            if kc_r > 1.4:
                sl_mult *= 1.0 + (kc_r - 1.4) * 0.2
                tp_mult *= 1.0 + (kc_r - 1.4) * 0.25
        return sl_mult, tp_mult

    def adjust_sl_tp(self, symbol: str, ltf_df: pd.DataFrame,
                     entry: float, direction: str,
                     raw_sl: float, raw_tp: float,
                     digits: int, pip: float,
                     base_sl_mult: float = 1.5,
                     base_tp_mult: float = 2.0,
                     sl_min_pips: float = 5.0,
                     sl_max_pips: float = 35.0) -> Tuple[float, float, Dict]:
        scaler = self.compute(symbol, ltf_df, base_sl_mult, base_tp_mult)
        sl_mult = scaler["sl_mult"]
        tp_mult = scaler["tp_mult"]
        atr_val = scaler.get("current_atr", 0)
        if atr_val <= 0:
            return raw_sl, raw_tp, {"note": "no_atr"}
        sl_min_dist = sl_min_pips * pip
        sl_max_dist = sl_max_pips * pip
        sl_distance = atr_val * sl_mult
        sl_distance = max(min(sl_distance, sl_max_dist), sl_min_dist)
        tp_distance = atr_val * tp_mult
        tp_distance = max(tp_distance, sl_distance * 1.2)
        if direction.upper() == "BUY":
            new_sl = round(entry - sl_distance, digits)
            new_tp = round(entry + tp_distance, digits)
        elif direction.upper() == "SELL":
            new_sl = round(entry + sl_distance, digits)
            new_tp = round(entry - tp_distance, digits)
        else:
            return raw_sl, raw_tp, {"note": "invalid_direction"}
        note = (
            f"vol_adj: SL×{sl_mult:.2f} TP×{tp_mult:.2f} "
            f"({scaler['reason']})"
        )
        return new_sl, new_tp, {"sl_mult": sl_mult, "tp_mult": tp_mult, "note": note}

    def get_baseline(self, symbol: str) -> Optional[dict]:
        return self._baselines.get(symbol)

    def reset_baseline(self, symbol: str = None):
        if symbol:
            self._baselines.pop(symbol, None)
            self._prediction_cache.pop(symbol, None)
        else:
            self._baselines.clear()
            self._prediction_cache.clear()
        logger.info(f"Baselines reseteados {'para ' + symbol if symbol else 'todos'}")


class VolatilityBacktester:
    """Backtesting framework for VolatilityScaler"""
    def __init__(self, scaler: VolatilityScaler):
        self.scaler = scaler
        self.results = []

    def run(self, df: pd.DataFrame, symbol: str,
            base_sl_mult: float = 1.5, base_tp_mult: float = 2.0) -> Dict:
        self.scaler.reset_baseline(symbol)
        results = []
        n = len(df)
        if n < 100:
            return {"error": "insufficient_data"}
        step = max(1, n // 200)
        for i in range(50, n, step):
            window = df.iloc[:i]
            if len(window) < 50:
                continue
            m = self.scaler.compute(symbol, window, base_sl_mult, base_tp_mult)
            if m.get("current_atr", 0) <= 0:
                continue
            results.append({
                "index": i,
                "sl_mult": m["sl_mult"],
                "tp_mult": m["tp_mult"],
                "atr": m["current_atr"],
                "atr_ratio": m.get("atr_ratio", 1),
                "intrabar_ratio": m.get("intrabar_ratio", 1),
                "bb_width_ratio": m.get("bb_width_ratio", 1),
                "reason": m["reason"],
            })
        self.results = results
        if not results:
            return {"error": "no_results"}
        sl_mults = [r["sl_mult"] for r in results]
        tp_mults = [r["tp_mult"] for r in results]
        return {
            "symbol": symbol,
            "samples": len(results),
            "sl_mult_mean": float(np.mean(sl_mults)),
            "sl_mult_std": float(np.std(sl_mults)),
            "sl_mult_min": float(min(sl_mults)),
            "sl_mult_max": float(max(sl_mults)),
            "tp_mult_mean": float(np.mean(tp_mults)),
            "tp_mult_std": float(np.std(tp_mults)),
            "pct_above_base_sl": float(np.mean([1 for s in sl_mults if s > base_sl_mult]) / len(sl_mults) * 100),
            "pct_below_base_sl": float(np.mean([1 for s in sl_mults if s < base_sl_mult]) / len(sl_mults) * 100),
            "reasons": list(set(r["reason"] for r in results)),
        }

    def summary(self) -> str:
        if not self.results:
            return "Sin resultados"
        sl = [r["sl_mult"] for r in self.results]
        tp = [r["tp_mult"] for r in self.results]
        return (
            f"Backtest: {len(self.results)} muestras\n"
            f"  SL mult: μ={np.mean(sl):.3f} σ={np.std(sl):.3f} "
            f"[{min(sl):.2f}–{max(sl):.2f}]\n"
            f"  TP mult: μ={np.mean(tp):.3f} σ={np.std(tp):.3f} "
            f"[{min(tp):.2f}–{max(tp):.2f}]"
        )

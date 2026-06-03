"""Correlation Engine — Análisis de Correlación entre Símbolos (Mejora 4, Modo Experto)
Evita tomar direcciones opuestas en pares correlacionados y usa
la correlación como filtro de confirmación de señales.
"""
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

LOOKBACKS = {
    "fast": 120,     # ~2h de datos 1m
    "medium": 480,   # ~8h
    "slow": 1440,    # ~24h
}

CORR_THRESHOLD_HIGH = 0.7
CORR_THRESHOLD_LOW = 0.3


class CorrelationRegime:
    HIGHLY_CORRELATED = "highly_correlated"
    UNCORRELATED = "uncorrelated"
    DIVERGING = "diverging"
    STRONG_DIVERGING = "strong_diverging"


class CorrelationEngine:
    def __init__(self, max_history: int = 2000):
        self.max_history = max_history
        self._price_history: Dict[str, np.ndarray] = {}
        self._timestamps: Dict[str, np.ndarray] = {}
        self._cache: Dict[Tuple[str, str, str], dict] = {}
        self._divergence_alerts: Dict[Tuple[str, str], float] = {}

    def update(self, symbol: str, timestamp, close_price: float):
        ts = timestamp.timestamp() if hasattr(timestamp, "timestamp") else timestamp
        if symbol not in self._price_history:
            self._price_history[symbol] = np.array([close_price])
            self._timestamps[symbol] = np.array([ts])
        else:
            self._price_history[symbol] = np.append(self._price_history[symbol], close_price)
            self._timestamps[symbol] = np.append(self._timestamps[symbol], ts)
            n = len(self._price_history[symbol])
            if n > self.max_history:
                self._price_history[symbol] = self._price_history[symbol][-self.max_history:]
                self._timestamps[symbol] = self._timestamps[symbol][-self.max_history:]

    def _prices_since(self, symbol: str, minutes: int) -> Optional[np.ndarray]:
        if symbol not in self._price_history:
            return None
        now = datetime.now(timezone.utc).timestamp()
        cutoff = now - minutes * 60
        mask = self._timestamps[symbol] >= cutoff
        if mask.sum() < 30:
            return None
        return self._price_history[symbol][mask]

    def _compute(self, a: np.ndarray, b: np.ndarray) -> dict:
        if len(a) < 30 or len(b) < 30:
            return {"correlation": 0.0, "p_value": 1.0, "n": 0}
        n = min(len(a), len(b))
        a, b = a[-n:], b[-n:]
        returns_a = np.diff(np.log(a + 1e-10))
        returns_b = np.diff(np.log(b + 1e-10))
        if len(returns_a) < 20:
            return {"correlation": 0.0, "p_value": 1.0, "n": 0}
        r, p = sp_stats.pearsonr(returns_a, returns_b)
        return {"correlation": float(r), "p_value": float(p), "n": len(returns_a)}

    def correlation(self, sym1: str, sym2: str,
                    lookback: str = "medium") -> dict:
        if sym1 == sym2:
            return {"correlation": 1.0, "p_value": 0.0, "n": 0, "regime": CorrelationRegime.HIGHLY_CORRELATED}
        cache_key = (sym1, sym2, lookback)
        cached = self._cache.get(cache_key)
        if cached and (datetime.now(timezone.utc).timestamp() - cached["_ts"]) < 120:
            return cached
        minutes = LOOKBACKS.get(lookback, 480)
        a = self._prices_since(sym1, minutes)
        b = self._prices_since(sym2, minutes)
        if a is None or b is None:
            return {"correlation": 0.0, "p_value": 1.0, "n": 0, "regime": CorrelationRegime.UNCORRELATED}
        result = self._compute(a, b)
        result["lookback"] = lookback
        result["_ts"] = datetime.now(timezone.utc).timestamp()
        corr = result["correlation"]
        if abs(corr) >= CORR_THRESHOLD_HIGH:
            result["regime"] = CorrelationRegime.HIGHLY_CORRELATED
        elif abs(corr) <= CORR_THRESHOLD_LOW:
            result["regime"] = CorrelationRegime.UNCORRELATED
        else:
            result["regime"] = CorrelationRegime.DIVERGING
        self._cache[cache_key] = result
        return result

    def _get_short_long_corr(self, sym1: str, sym2: str) -> Tuple[float, float]:
        fast = self.correlation(sym1, sym2, "fast").get("correlation", 0)
        slow = self.correlation(sym1, sym2, "slow").get("correlation", 0)
        return fast, slow

    def are_diverging(self, sym1: str, sym2: str) -> Tuple[bool, float]:
        fast, slow = self._get_short_long_corr(sym1, sym2)
        diff = abs(fast - slow)
        if diff < 0.25:
            return False, diff
        abs_fast, abs_slow = abs(fast), abs(slow)
        if abs_slow > CORR_THRESHOLD_LOW and abs_fast < abs_slow * 0.5:
            return True, diff
        if fast * slow < 0 and diff > 0.4:
            return True, diff
        if abs_fast > 0.8 and abs_slow < 0.3 and diff > 0.5:
            return True, diff
        return False, diff

    def correlation_matrix(self, symbols: List[str],
                           lookback: str = "medium") -> Dict[str, Dict[str, float]]:
        matrix = {}
        for s1 in symbols:
            matrix[s1] = {}
            for s2 in symbols:
                if s1 == s2:
                    matrix[s1][s2] = 1.0
                else:
                    matrix[s1][s2] = self.correlation(s1, s2, lookback).get("correlation", 0)
        return matrix

    def confirm_signal(self, symbol: str, direction: str,
                       all_symbols: List[str],
                       active_positions: Dict[str, str] = None) -> Tuple[bool, str]:
        if active_positions is None:
            active_positions = {}
        conflicts = []
        for other_sym, other_dir in active_positions.items():
            if other_sym == symbol:
                continue
            corr_data = self.correlation(symbol, other_sym)
            corr = corr_data.get("correlation", 0)
            if corr > CORR_THRESHOLD_HIGH and direction != other_dir:
                conflicts.append(f"{other_sym}({other_dir} r={corr:.2f})")
        if conflicts:
            diverging, diff = self.are_diverging(symbol, list(active_positions.keys())[0]) if active_positions else (False, 0)
            if diverging:
                return True, f"Divergencia detectada (diff={diff:.2f}) — permitiendo pese a conflicto: {', '.join(conflicts)}"
            msg = f"Correlación alta con posiciones opuestas: {', '.join(conflicts)}"
            return False, msg
        for other_sym in all_symbols:
            if other_sym == symbol or other_sym in active_positions:
                continue
            if other_sym not in self._price_history:
                continue
            corr_data = self.correlation(symbol, other_sym)
            corr = corr_data.get("correlation", 0)
            if corr > CORR_THRESHOLD_HIGH:
                other_prices = self._prices_since(other_sym, LOOKBACKS["fast"])
                if other_prices is not None and len(other_prices) > 20:
                    change = (other_prices[-1] / other_prices[-20] - 1) * 100
                    if direction == "BUY" and change < -1.0:
                        return False, f"{other_sym}(r={corr:.2f}) bajando {change:.1f}% — contradice señal BUY"
                    elif direction == "SELL" and change > 1.0:
                        return False, f"{other_sym}(r={corr:.2f}) subiendo {change:.1f}% — contradice señal SELL"
        return True, ""

    def volume_adjustment(self, symbol: str,
                          existing_positions: Dict[str, float]) -> Tuple[float, str]:
        if not existing_positions:
            return 1.0, ""
        total_reduction = 1.0
        notes = []
        for other_sym, other_vol in existing_positions.items():
            if other_sym == symbol:
                continue
            corr_data = self.correlation(symbol, other_sym)
            corr = abs(corr_data.get("correlation", 0))
            if corr > 0.8:
                reduction = 1.0 - corr * 0.3
                total_reduction *= reduction
                notes.append(f"{other_sym}(r={corr:.2f}) → ×{reduction:.2f}")
        if notes:
            return round(total_reduction, 3), "; ".join(notes)
        return 1.0, ""

    def get_regime(self, sym1: str, sym2: str) -> str:
        data = self.correlation(sym1, sym2)
        regime = data.get("regime", CorrelationRegime.UNCORRELATED)
        if regime == CorrelationRegime.HIGHLY_CORRELATED:
            diverging, diff = self.are_diverging(sym1, sym2)
            if diverging:
                return CorrelationRegime.STRONG_DIVERGING
        return regime

    def get_divergence_alerts(self) -> List[str]:
        alerts = []
        checked = set()
        symbols = list(self._price_history.keys())
        for i, s1 in enumerate(symbols):
            for s2 in symbols[i + 1:]:
                pair = tuple(sorted([s1, s2]))
                if pair in checked:
                    continue
                checked.add(pair)
                diverging, diff = self.are_diverging(s1, s2)
                if diverging:
                    fast, slow = self._get_short_long_corr(s1, s2)
                    alerts.append(f"{s1}<->{s2}: divergiendo (fast={fast:.2f} slow={slow:.2f} diff={diff:.2f})")
                    last_alert = self._divergence_alerts.get(pair, 0)
                    now = datetime.now(timezone.utc).timestamp()
                    if now - last_alert > 3600:
                        logger.warning(f"[CORR] Divergencia: {s1}<->{s2} fast={fast:.2f} slow={slow:.2f}")
                        self._divergence_alerts[pair] = now
        return alerts

    def reset(self, symbol: str = None):
        if symbol:
            self._price_history.pop(symbol, None)
            self._timestamps.pop(symbol, None)
        else:
            self._price_history.clear()
            self._timestamps.clear()
        self._cache.clear()

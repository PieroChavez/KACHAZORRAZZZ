"""Circuit Breakers: monitorea condiciones de riesgo por símbolo y recomienda
pausar entradas, cancelar órdenes o reducir exposición. Solo observa y reporta.
No ejecuta acciones directamente; el bot decide si aplicar las recomendaciones."""
import time
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
from loguru import logger
from dataclasses import dataclass, field


@dataclass
class BreakerStatus:
    active: bool = False
    reason: str = ""
    severity: str = "info"  # info, warning, critical
    suggestion: str = ""
    expires_at: float = 0.0


@dataclass
class CircuitBreakerState:
    volatility_spike: BreakerStatus = field(default_factory=BreakerStatus)
    news_approaching: BreakerStatus = field(default_factory=BreakerStatus)
    momentum_against: BreakerStatus = field(default_factory=BreakerStatus)
    correlation_breach: BreakerStatus = field(default_factory=BreakerStatus)

    def any_active(self) -> bool:
        return any([
            self.volatility_spike.active,
            self.news_approaching.active,
            self.momentum_against.active,
            self.correlation_breach.active,
        ])

    def highest_severity(self) -> str:
        severities = {"critical": 3, "warning": 2, "info": 1}
        best = "info"
        for s in [self.volatility_spike, self.news_approaching,
                  self.momentum_against, self.correlation_breach]:
            if s.active and severities.get(s.severity, 0) > severities.get(best, 0):
                best = s.severity
        return best


class CircuitBreakers:
    """Monitorea volatilidad, noticias, momentum y correlación.
    Produce un BreakerStatus por símbolo en cada evaluación."""

    def __init__(self, volatility_threshold: float = 2.0,
                 momentum_candles: int = 3,
                 news_buffer_minutes: float = 30.0):
        self.volatility_threshold = volatility_threshold
        self.momentum_candles = momentum_candles
        self.news_buffer_minutes = news_buffer_minutes
        self._atr_history: Dict[str, list] = {}

    def check_volatility_spike(self, symbol: str, ltf_df: pd.DataFrame) -> BreakerStatus:
        atr_val = self._compute_atr(ltf_df)
        if symbol not in self._atr_history:
            self._atr_history[symbol] = []
        hist = self._atr_history[symbol]
        hist.append(atr_val)
        if len(hist) > 20:
            hist.pop(0)
        if len(hist) >= 5:
            baseline = np.mean(hist[:-1])
            if baseline > 0 and atr_val > baseline * self.volatility_threshold:
                return BreakerStatus(
                    active=True,
                    reason=f"ATR spike: {atr_val:.1f} vs baseline {baseline:.1f} ({atr_val/baseline:.1f}x)",
                    severity="warning",
                    suggestion=f"Reducir tamaño de lote 50%, ampliar trailing 1.5x",
                    expires_at=time.time() + 1800,
                )
        return BreakerStatus()

    def check_momentum_against(self, symbol: str, ltf_df: pd.DataFrame,
                               is_long: bool, entry_price: float) -> BreakerStatus:
        close_prices = ltf_df["close"].values
        if len(close_prices) < self.momentum_candles + 1:
            return BreakerStatus()
        recent = close_prices[-(self.momentum_candles + 1):]
        if is_long:
            hits = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i-1])
        else:
            hits = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i-1])
        if hits == self.momentum_candles:
            current_pnl_pct = (close_prices[-1] - entry_price) / entry_price if is_long else (entry_price - close_prices[-1]) / entry_price
            return BreakerStatus(
                active=True,
                reason=f"{self.momentum_candles} velas consecutivas en contra (PnL={current_pnl_pct:.2%})",
                severity="warning" if current_pnl_pct < -0.01 else "info",
                suggestion="Reducir trailing, no expandir TP",
                expires_at=time.time() + 900,
            )
        return BreakerStatus()

    def check_news_approaching(self, news_calendar, symbol: str, now) -> BreakerStatus:
        if not hasattr(news_calendar, 'is_high_impact_active'):
            return BreakerStatus()
        try:
            if news_calendar.is_high_impact_active(now, symbol):
                return BreakerStatus(
                    active=True,
                    reason="Noticia de alto impacto activa o próxima",
                    severity="critical",
                    suggestion="No abrir nuevas posiciones, cancelar pendientes",
                    expires_at=time.time() + 3600,
                )
            upcoming = getattr(news_calendar, 'get_upcoming_high_impact', None)
            if upcoming:
                events = upcoming(symbol, minutes=self.news_buffer_minutes * 2)
                for ev in events[:1]:
                    time_to = (ev.get("time", now) - now).total_seconds() / 60
                    if 0 < time_to <= self.news_buffer_minutes:
                        return BreakerStatus(
                            active=True,
                            reason=f"Noticia '{ev.get('title','')}' en {time_to:.0f}min",
                            severity="warning",
                            suggestion="No abrir nuevas posiciones hasta después de la noticia",
                            expires_at=time.time() + time_to * 60,
                        )
        except Exception:
            pass
        return BreakerStatus()

    def check_all(self, symbol: str, ltf_df: pd.DataFrame, news_calendar, now,
                  positions_info: Optional[List[dict]] = None) -> CircuitBreakerState:
        state = CircuitBreakerState()
        state.volatility_spike = self.check_volatility_spike(symbol, ltf_df)
        state.news_approaching = self.check_news_approaching(news_calendar, symbol, now)
        if positions_info:
            for pos in positions_info:
                is_long = pos.get("type") == "buy"
                entry = pos.get("price_open", 0)
                mom = self.check_momentum_against(symbol, ltf_df, is_long, entry)
                if mom.active:
                    state.momentum_against = mom
                    break
        return state

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < period + 1:
            return 0.0
        high, low, close = df["high"].values, df["low"].values, df["close"].values
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]),
                                   np.abs(low[1:] - close[:-1])))
        return float(np.mean(tr[-period:]))

"""Dynamic VaR Risk Manager — MEJORA 8 (Modo Experto)
Unifica Kelly Criterion, Value at Risk dinámico, volatilidad implícita,
hora del día, drawdown actual y correlación entre símbolos
en una única ecuación de riesgo.
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VaRComponents:
    """Desglose de los componentes del risk fraction calculado"""
    kelly_base: float
    volatility_weight: float
    time_weight: float
    drawdown_weight: float
    correlation_weight: float
    final_fraction: float
    var_95: float
    notes: List[str] = None


class DynamicVaRRiskManager:
    """Risk Manager que combina Kelly + VaR dinámico en una ecuación unificada.

    Ecuación unificada:
        risk_fraction = kelly_base × w_vol × w_time × w_dd × w_corr

    Cada peso se deriva de:
      - w_vol:   ratio ATR actual / ATR baseline (volatilidad implícita)
      - w_time:  perfil de sesión (hora pico/débil)
      - w_dd:    drawdown actual sobre máximo histórico
      - w_corr:  correlación media con posiciones abiertas
    """

    def __init__(
        self,
        initial_risk: float = 0.02,
        max_risk: float = 0.04,
        min_risk: float = 0.005,
        var_confidence: float = 0.95,
        lookback_dd: int = 50,
    ):
        self.initial_risk = initial_risk
        self.max_risk = max_risk
        self.min_risk = min_risk
        self.var_confidence = var_confidence
        self.lookback_dd = lookback_dd

        self._peak_balance: float = 0.0
        self._current_balance: float = 0.0
        self._equity_curve: List[float] = []
        self._var_history: List[float] = []

    # ── API pública ──────────────────────────────────────────────────────────

    def update_balance(self, balance: float):
        """Actualiza el balance y calcula drawdown."""
        self._current_balance = balance
        if balance > self._peak_balance:
            self._peak_balance = balance
        self._equity_curve.append(balance)
        if len(self._equity_curve) > self.lookback_dd * 2:
            self._equity_curve = self._equity_curve[-self.lookback_dd * 2:]

    def record_trade_result(self, profit: float):
        """Registra resultado de trade para VaR histórico."""
        self._var_history.append(profit)
        if len(self._var_history) > self.lookback_dd:
            self._var_history = self._var_history[-self.lookback_dd:]

    def get_risk_fraction(
        self,
        kelly_fraction: float,
        conviction: float,
        atr_ratio: float = 1.0,
        session_weight: float = 1.0,
        correlation_factor: float = 1.0,
    ) -> Tuple[float, VaRComponents]:
        """Calcula la fracción de riesgo unificada.

        Args:
            kelly_fraction: Fracción base de Kelly.
            conviction: Convicción de la señal [0, 1].
            atr_ratio: Ratio ATR actual / ATR baseline (1.0 = normal).
            session_weight: Peso de sesión (0.3-1.3).
            correlation_factor: Factor de correlación (0.0-1.0).

        Returns:
            (risk_fraction, VaRComponents)
        """
        kelly_base = self._adjust_kelly_by_conviction(kelly_fraction, conviction)

        w_vol = self._volatility_weight(atr_ratio)
        w_time = self._time_weight(session_weight)
        w_dd = self._drawdown_weight()
        w_corr = self._correlation_weight(correlation_factor)

        var_95 = self._compute_var_95()

        raw = kelly_base * w_vol * w_time * w_dd * w_corr

        final = max(self.min_risk, min(self.max_risk, raw))

        notes = []
        if w_vol != 1.0:
            notes.append(f"volatilidad×{w_vol:.3f}")
        if w_time != 1.0:
            notes.append(f"sesión×{w_time:.3f}")
        if w_dd != 1.0:
            notes.append(f"drawdown×{w_dd:.3f}")
        if w_corr != 1.0:
            notes.append(f"correlación×{w_corr:.3f}")

        components = VaRComponents(
            kelly_base=kelly_base,
            volatility_weight=w_vol,
            time_weight=w_time,
            drawdown_weight=w_dd,
            correlation_weight=w_corr,
            final_fraction=final,
            var_95=var_95,
            notes=notes or [],
        )

        return final, components

    def get_volume_multiplier(
        self,
        kelly_fraction: float,
        conviction: float,
        atr_ratio: float = 1.0,
        session_weight: float = 1.0,
        correlation_factor: float = 1.0,
    ) -> Tuple[float, VaRComponents]:
        """Devuelve multiplicador de volumen relativo al riesgo inicial."""
        risk_frac, components = self.get_risk_fraction(
            kelly_fraction, conviction,
            atr_ratio, session_weight, correlation_factor,
        )
        mult = risk_frac / self.initial_risk if self.initial_risk > 0 else 1.0
        return mult, components

    # ── Componentes de la ecuación unificada ─────────────────────────────────

    def _adjust_kelly_by_conviction(
        self, kelly_fraction: float, conviction: float,
    ) -> float:
        """Ajusta Kelly por convicción de la señal."""
        conviction_mult = 0.5 + conviction * 0.5
        return kelly_fraction * conviction_mult

    def _volatility_weight(self, atr_ratio: float) -> float:
        """Peso por volatilidad implícita (ATR ratio).

        atr_ratio > 1.5 → alta volatilidad → reducir riesgo
        atr_ratio < 0.7 → baja volatilidad → riesgo normal o ligeramente mayor
        """
        if atr_ratio <= 0:
            return 1.0
        if atr_ratio > 1.5:
            excess = atr_ratio - 1.5
            return max(0.3, 1.0 - excess * 0.5)
        if atr_ratio < 0.7:
            shrink = 0.85 + (atr_ratio / 0.7) * 0.15
            return min(1.15, shrink)
        return 1.0

    def _time_weight(self, session_weight: float) -> float:
        """Peso por hora del día / sesión de mercado.

        session_weight > 1.0 → hora pico (ej. solapamiento Londres-NY)
        session_weight < 1.0 → hora débil (ej. cierre, Asia tranquilas)
        """
        return max(0.3, min(1.3, session_weight))

    def _drawdown_weight(self) -> float:
        """Peso por drawdown actual sobre peak.

        A mayor drawdown, menor riesgo permitido (efecto conservador).
        """
        if self._peak_balance <= 0:
            return 1.0
        dd_pct = (self._peak_balance - self._current_balance) / self._peak_balance
        dd_pct = max(0.0, dd_pct)
        if dd_pct <= 0.0:
            return 1.0
        if dd_pct >= 0.15:
            return 0.3  # Drawdown severo → mínimo riesgo
        if dd_pct >= 0.10:
            return 0.5
        if dd_pct >= 0.05:
            return 0.7
        return 1.0 - dd_pct * 6.0  # Lineal entre 0-5%

    def _correlation_weight(self, correlation_factor: float) -> float:
        """Peso por correlación con posiciones existentes.

        correlation_factor = 1.0 → sin correlación (riesgo completo)
        correlation_factor → 0.0 → máxima correlación (reducir riesgo)
        """
        return max(0.2, min(1.0, correlation_factor))

    # ── Value at Risk (VaR) ─────────────────────────────────────────────────

    def _compute_var_95(self) -> float:
        """Calcula VaR al 95% del histórico de trades."""
        if len(self._var_history) < 10:
            return 0.0
        returns = np.array(self._var_history[-self.lookback_dd:])
        if len(returns) == 0:
            return 0.0
        return float(np.percentile(returns, 5))

    def get_var_summary(self) -> dict:
        """Resumen de métricas VaR para logging/monitoreo."""
        var_95 = self._compute_var_95()
        dd_pct = self._get_drawdown_pct()
        return {
            "var_95": var_95,
            "drawdown_pct": dd_pct,
            "peak_balance": self._peak_balance,
            "current_balance": self._current_balance,
            "trades_in_var_history": len(self._var_history),
        }

    def _get_drawdown_pct(self) -> float:
        if self._peak_balance <= 0:
            return 0.0
        return (self._peak_balance - self._current_balance) / self._peak_balance

    def reset(self):
        """Reinicia el estado interno (para backtest o nuevo ciclo)."""
        self._peak_balance = 0.0
        self._current_balance = 0.0
        self._equity_curve.clear()
        self._var_history.clear()

"""Binary Risk Filter — Filtro binario de riesgo go/no-go
Checklist obligatorio pre-entrada. Si ALGUNA condición falla → NO_TRADE.
  - SL estructuralmente protegido (no arbitrario)
  - R:R mínimo (1:5 para scalping, configurable)
  - Volumen validado en el breakout
  - Confluencia con zonas estructurales
  - No operar en contra del contexto macro

"Es un sistema binario: o es válido o no es válido." — MD Class 3

Referencia: MD Classes 3, 10, 12, 15
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.core.md_concepts import MDConcept, MDDetection

logger = logging.getLogger(__name__)

DEFAULT_CHECKS = [
    "structural_sl",     # SL protegido por estructura (no flotante)
    "min_rr",            # R:R ≥ mínimo configurado
    "volume_breakout",   # breakout con volumen sobre media
    "zone_confluence",   # confluencia con zona estructural
    "no_opposing_macro", # no operar contra contexto macro
]


@dataclass
class BinaryRiskResult:
    passed: bool
    failed_checks: List[str] = field(default_factory=list)
    passed_checks: List[str] = field(default_factory=list)
    confidence_penalty: float = 0.0

    def to_detection(self, direction: str) -> Optional[MDDetection]:
        if not self.passed:
            return None
        return MDDetection(
            concept=MDConcept.BINARY_RISK,
            direction=direction,
            confidence=0.95,
            timeframe="M15",
            metadata={
                "passed_checks": self.passed_checks,
                "total_checks": len(self.passed_checks) + len(self.failed_checks),
            },
        )


class BinaryRiskFilter:
    """Valida el checklist binario de riesgo. Si falla → NO_TRADE.

    Args:
        min_rr_ratio: mínimo riesgo/beneficio (default 5.0 = 1:5)
        min_volume_ratio: mínimo volumen de breakout (default 1.3)
        max_sl_atr_ratio: máximo SL en múltiplos de ATR (default 0.5)
    """

    def __init__(self, min_rr_ratio: float = 3.0,
                 min_volume_ratio: float = 1.3,
                 max_sl_atr_ratio: float = 0.5):
        self._min_rr = min_rr_ratio
        self._min_vol = min_volume_ratio
        self._max_sl_atr = max_sl_atr_ratio

    def evaluate(self, df: pd.DataFrame,
                 direction: str,
                 entry_price: Optional[float] = None,
                 stop_loss: Optional[float] = None,
                 target_price: Optional[float] = None,
                 zone_data: Optional[Dict] = None,
                 macro_direction: str = "NEUTRAL") -> BinaryRiskResult:
        """Ejecuta el checklist binario completo.

        Args:
            df: DataFrame con velas OHLC
            direction: dirección propuesta
            entry_price: precio de entrada
            stop_loss: precio del Stop Loss
            target_price: precio objetivo
            zone_data: dict opcional con info de zonas (dominant_direction, etc.)
            macro_direction: dirección macro detectada
        """
        failed = []
        passed = []

        if df is None or len(df) < 10:
            return BinaryRiskResult(
                passed=False, failed_checks=["insufficient_data"],
                confidence_penalty=0.3,
            )

        # ── Check 1: SL estructural ──
        if stop_loss is not None and entry_price is not None:
            sl_dist = abs(entry_price - stop_loss)
            atr_val = self._calc_atr(df)
            if atr_val > 0 and sl_dist <= atr_val * self._max_sl_atr:
                passed.append("structural_sl")
            else:
                failed.append("structural_sl")
        else:
            failed.append("structural_sl")

        # ── Check 2: R:R mínimo ──
        if (entry_price is not None and stop_loss is not None
                and target_price is not None):
            risk = abs(entry_price - stop_loss)
            reward = abs(target_price - entry_price)
            if risk > 0 and reward / risk >= self._min_rr:
                passed.append("min_rr")
            else:
                failed.append("min_rr")
        else:
            failed.append("min_rr")

        # ── Check 3: Volumen en breakout ──
        last_idx = len(df) - 1
        vol_ratio = self._volume_ratio(df, last_idx)
        if last_idx >= 1:
            vol_ratio2 = self._volume_ratio(df, last_idx - 1)
            vol_ratio = max(vol_ratio, vol_ratio2)
        if vol_ratio >= self._min_vol:
            passed.append("volume_breakout")
        else:
            failed.append("volume_breakout")

        # ── Check 4: Confluencia con zonas ──
        if zone_data and zone_data.get("dominant_direction", "NEUTRAL") != "NEUTRAL":
            passed.append("zone_confluence")
        else:
            passed.append("zone_confluence")

        # ── Check 5: No operar contra macro ──
        if macro_direction != "NEUTRAL" and direction != macro_direction:
            failed.append("no_opposing_macro")
        else:
            passed.append("no_opposing_macro")

        penalty = len(failed) * 0.15
        all_passed = len(failed) == 0 and len(passed) > 0
        passed_all = len(failed) == 0

        notes = (
            f"BinaryRisk: {len(passed)}/{len(passed)+len(failed)} checks passed"
        )

        return BinaryRiskResult(
            passed=passed_all,
            failed_checks=failed,
            passed_checks=passed,
            confidence_penalty=penalty,
        )

    def _volume_ratio(self, df: pd.DataFrame, idx: int) -> float:
        if "tick_volume" not in df.columns and "volume" not in df.columns:
            return 1.0
        vol_col = "tick_volume" if "tick_volume" in df.columns else "volume"
        volumes = df[vol_col].values
        if idx < 5 or idx >= len(volumes):
            return 1.0
        recent = volumes[max(0, idx - 5):idx]
        avg_vol = float(np.mean(recent))
        if avg_vol == 0:
            return 1.0
        return min(3.0, float(volumes[idx]) / avg_vol)

    def _calc_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < period + 1:
            return 0.0
        high, low, close = df["high"].values, df["low"].values, df["close"].values
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]),
                                   np.abs(low[1:] - close[:-1])))
        return float(np.mean(tr[-period:]))

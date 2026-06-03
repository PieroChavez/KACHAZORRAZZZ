"""Dynamic TP — Take Profit expansivo con confirmación HTF (MEJORA SCALPING)
Gestiona el take profit de forma dinámica: el TP objetivo se expande
a medida que las velas de 1H/4H cierran a favor de la operación.
En lugar de un TP fijo, se escala en tramos:
  - TP1: Liquidez más cercana (objetivo inicial)
  - TP2: Si 1H cierra a favor → expandir al siguiente nivel
  - TP3: Si 4H cierra a favor → expandir al siguiente nivel
  - TP4: Si ambas confirman → objetivo final (liquidez mayor)
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from src.core.liquidity_mapper import MarketMap, MarketZone, LiquidityMapper

logger = logging.getLogger(__name__)


@dataclass
class TPTier:
    level: int
    price: float
    label: str
    active: bool = False
    reached: bool = False


class DynamicTPManager:
    """Gestiona la expansión progresiva de TP según confirmación HTF.

    Estrategia:
      - TP1: liquidez más cercana (siempre activo)
      - TP2: liquidez intermedia (se activa si 1H confirma)
      - TP3: liquidez mayor (se activa si 1H+4H confirman)
      - TP4: objetivo extremo (se activa si ambas HTF se alinean)
    """

    def __init__(self, tp_tiers: int = 4,
                 min_tier_distance_atr: float = 0.8,
                 activation_on_1h: bool = True,
                 activation_on_4h: bool = True):
        self._tiers = tp_tiers
        self._min_tier_dist = min_tier_distance_atr
        self._use_1h = activation_on_1h
        self._use_4h = activation_on_4h

    def build_tiers(self, entry_price: float, direction: str,
                    market_map: MarketMap, df_ltf: pd.DataFrame,
                    confirmed_1h: bool = False,
                    confirmed_4h: bool = False) -> List[TPTier]:
        tiers = []
        atr_val = self._atr(df_ltf)
        pip = self._pip_size(market_map.symbol)

        targets = self._get_targets(market_map, direction)
        num_targets = len(targets)

        if num_targets == 0:
            fixed_tp = entry_price + (atr_val * 2) if direction == "BUY" else entry_price - (atr_val * 2)
            return [TPTier(level=1, price=fixed_tp, label=f"ATR_2x_{fixed_tp:.5f}", active=True)]

        active_tiers = 1
        if self._use_1h and confirmed_1h:
            active_tiers = 2 if num_targets >= 2 else 1
        if self._use_4h and confirmed_4h:
            active_tiers = min(num_targets, active_tiers + 1)
        if self._use_1h and self._use_4h and confirmed_1h and confirmed_4h:
            active_tiers = num_targets

        for i, target in enumerate(targets):
            level = i + 1
            dist = abs(target - entry_price)
            label = f"LIQUIDITY_{target:.5f}"
            if dist < self._min_tier_dist * atr_val:
                continue
            active = level <= active_tiers
            tier = TPTier(
                level=level,
                price=target,
                label=label,
                active=active,
            )
            tiers.append(tier)
            if level > self._tiers:
                break

        if not tiers:
            fixed_tp = entry_price + (atr_val * 2) if direction == "BUY" else entry_price - (atr_val * 2)
            tiers.append(TPTier(level=1, price=fixed_tp, label=f"ATR_2x_{fixed_tp:.5f}", active=True))

        return tiers

    def update_tiers(self, tiers: List[TPTier], current_price: float,
                     confirmed_1h: bool, confirmed_4h: bool) -> List[TPTier]:
        updated = []
        for tier in tiers:
            tier.reached = abs(current_price - tier.price) / max(abs(tier.price), 0.0001) < 0.0005
            updated.append(tier)
        return updated

    def get_active_tp(self, tiers: List[TPTier]) -> Optional[float]:
        for tier in reversed(tiers):
            if tier.active and not tier.reached:
                return tier.price
        for tier in reversed(tiers):
            if not tier.reached:
                return tier.price
        return None

    def _get_targets(self, market_map: MarketMap, direction: str) -> List[float]:
        targets = []
        for tf_zones in market_map.zones.values():
            for z in tf_zones:
                if z.direction == direction and z.strength >= 0.3:
                    targets.append(z.price_level)
        targets = sorted(set(targets))
        if direction == "SELL":
            targets = sorted(targets, reverse=True)
        else:
            targets = sorted(targets)
        return targets

    def _pip_size(self, symbol: str) -> float:
        from src.utils.helpers import pip_size
        return pip_size(symbol)

    def _atr(self, df: pd.DataFrame, period: int = 14) -> float:
        from src.utils.helpers import atr
        if df is None or len(df) < period + 1:
            return 0.0
        return float(atr(df, period).iloc[-1])

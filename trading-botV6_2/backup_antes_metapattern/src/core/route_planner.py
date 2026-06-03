"""Route Planner — Ruta desde precio actual hasta liquidez objetivo (MEJORA SCALPING)
Traza el camino desde el precio actual hasta el siguiente objetivo de liquidez,
identificando las zonas intermedias (FVGs, OBs) que el precio deberá respetar
o rellenar en el camino.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

from src.core.liquidity_mapper import MarketMap, MarketZone, LiquidityMapper

logger = logging.getLogger(__name__)


@dataclass
class RouteNode:
    price: float
    node_type: str
    timeframe: str
    direction: str
    strength: float
    is_required: bool
    distance_from_entry: float
    notes: List[str] = field(default_factory=list)


@dataclass
class Route:
    symbol: str
    direction: str
    entry_price: float
    target_price: float
    target_type: str
    nodes: List[RouteNode] = field(default_factory=list)
    total_distance: float = 0.0
    confidence: float = 0.0
    has_valid_sl_zone: bool = False
    sl_zone_price: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.nodes) > 0 and self.has_valid_sl_zone and self.confidence >= 0.3

    @property
    def distance_to_target(self) -> float:
        if self.direction == "BUY":
            return self.target_price - self.entry_price
        return self.entry_price - self.target_price

    def get_nearest_zone_ahead(self) -> Optional[RouteNode]:
        for node in self.nodes:
            if node.direction == self.direction and node.distance_from_entry > 0:
                return node
        return None


class RoutePlanner:
    """Planea la ruta desde el precio actual hasta la liquidez objetivo.

    Usa el MarketMap para identificar:
      - Siguiente liquidez mayor (target final)
      - FVGs/OBs intermedios que el precio deberá rellenar/respetar
      - Zona de SL viable (detrás del FVG/OB más cercano)
    """

    def __init__(self, max_route_distance_pips: float = 100.0,
                 min_target_distance_atr: float = 1.5):
        self._max_distance = max_route_distance_pips
        self._min_target_atr = min_target_distance_atr

    def plan(self, market_map: MarketMap, df_ltf: pd.DataFrame,
             entry_price: float, direction: str) -> Route:
        if direction == "NEUTRAL":
            return Route(market_map.symbol, "NEUTRAL", entry_price, entry_price, "NONE")

        pip = self._pip_size(market_map.symbol)
        atr_val = self._atr(df_ltf)

        target = None
        target_type = "NONE"
        if direction == "BUY":
            target = market_map.nearest_liquidity_above
            target_type = "LIQUIDITY_ABOVE"
            fallback_target = market_map.nearest_fvg_above
            if target is None and fallback_target is not None:
                target = fallback_target
                target_type = "FVG_ABOVE"
        else:
            target = market_map.nearest_liquidity_below
            target_type = "LIQUIDITY_BELOW"
            fallback_target = market_map.nearest_fvg_below
            if target is None and fallback_target is not None:
                target = fallback_target
                target_type = "FVG_BELOW"

        if target is None:
            return Route(market_map.symbol, "NEUTRAL", entry_price, entry_price, "NONE",
                         notes=["No hay target de liquidez visible"])

        distance = abs(target - entry_price)
        max_dist_pips = self._max_distance * pip * 10
        if distance > max_dist_pips:
            return Route(market_map.symbol, "NEUTRAL", entry_price, entry_price, "NONE",
                         notes=[f"Target {target:.5f} demasiado lejos ({distance/pip:.0f}p)"])

        route_nodes: List[RouteNode] = []
        all_zones = []
        for tf_zones in market_map.zones.values():
            all_zones.extend(tf_zones)

        if direction == "BUY":
            relevant = sorted(
                [z for z in all_zones if z.price_level > entry_price and z.price_level < target
                 and z.direction == "BUY"],
                key=lambda z: z.price_level,
            )
        else:
            relevant = sorted(
                [z for z in all_zones if z.price_level < entry_price and z.price_level > target
                 and z.direction == "SELL"],
                key=lambda z: z.price_level, reverse=True,
            )

        for i, zone in enumerate(relevant):
            if zone.strength < 0.3:
                continue
            dist = abs(zone.price_level - entry_price)
            node = RouteNode(
                price=zone.price_level,
                node_type=zone.zone_type.value if hasattr(zone.zone_type, 'value') else str(zone.zone_type),
                timeframe=zone.timeframe,
                direction=zone.direction,
                strength=zone.strength,
                is_required=zone.strength >= 0.5,
                distance_from_entry=dist,
                notes=zone.notes,
            )
            route_nodes.append(node)

        has_sl_zone = False
        sl_zone_price = None
        if direction == "BUY":
            zones_below = [z for z in all_zones if z.price_level < entry_price
                           and z.direction == "SELL"]
            if zones_below:
                nearest_below = max(zones_below, key=lambda z: z.price_level)
                sl_dist = entry_price - nearest_below.price_level
                if sl_dist >= atr_val * 0.5:
                    has_sl_zone = True
                    sl_zone_price = nearest_below.price_level
        else:
            zones_above = [z for z in all_zones if z.price_level > entry_price
                           and z.direction == "BUY"]
            if zones_above:
                nearest_above = min(zones_above, key=lambda z: z.price_level)
                sl_dist = nearest_above.price_level - entry_price
                if sl_dist >= atr_val * 0.5:
                    has_sl_zone = True
                    sl_zone_price = nearest_above.price_level

        if not has_sl_zone:
            if direction == "BUY":
                sl_zone_price = entry_price - atr_val * 0.8
            else:
                sl_zone_price = entry_price + atr_val * 0.8
            has_sl_zone = True

        dist_to_target = abs(target - entry_price)
        min_dist = self._min_target_atr * atr_val
        if dist_to_target < min_dist:
            return Route(market_map.symbol, "NEUTRAL", entry_price, entry_price, "NONE",
                         notes=[f"Target demasiado cerca ({dist_to_target/pip:.1f}p < {min_dist/pip:.1f}p)"])

        confidence = 0.3 + len(route_nodes) * 0.05
        if route_nodes:
            confidence += sum(n.strength for n in route_nodes) * 0.1
        confidence = min(0.95, confidence)

        route = Route(
            symbol=market_map.symbol,
            direction=direction,
            entry_price=entry_price,
            target_price=target,
            target_type=target_type,
            nodes=route_nodes,
            total_distance=dist_to_target,
            confidence=confidence,
            has_valid_sl_zone=has_sl_zone,
            sl_zone_price=sl_zone_price,
        )

        target_label = f"liquidez {target:.5f}" if "LIQUIDITY" in target_type else f"FVG {target:.5f}"
        route.notes.append(f"Ruta {direction}: {entry_price:.5f} → {target_label} ({dist_to_target/pip:.1f}p)")
        if route_nodes:
            route.notes.append(f"Zonas intermedias: {len(route_nodes)}")
        if sl_zone_price is not None:
            route.notes.append(f"SL en {sl_zone_price:.5f} ({(abs(entry_price - sl_zone_price))/pip:.1f}p)")
        if confidence >= 0.7:
            route.notes.append(f"Ruta de alta confianza ({confidence:.0%})")

        return route

    def _pip_size(self, symbol: str) -> float:
        from src.utils.helpers import pip_size
        return pip_size(symbol)

    def _atr(self, df: pd.DataFrame, period: int = 14) -> float:
        from src.utils.helpers import atr
        if df is None or len(df) < period + 1:
            return 0.0
        return float(atr(df, period).iloc[-1])

"""Confluence Scorer — Proyecta targets por confluencia de zonas SMC
Toma el MarketMap (liquidez, FVGs, OBs, breakers) y asigna probabilidad
a cada nivel basado en cantidad y calidad de confluencias.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.liquidity_mapper import MarketMap, MarketZone, LiquidityLevel
from src.core.route_planner import Route, RouteNode
from src.utils.helpers import atr, pip_size

logger = logging.getLogger(__name__)


@dataclass
class TargetProjection:
    price: float
    direction: str
    confidence: float  # 0-1
    confluence_count: int
    confluences: List[str]
    timeframe: str
    distance_pips: float
    estimated_bars_m1: int  # velas M1 estimadas para llegar
    is_primary: bool = False


@dataclass
class ConfluenceResult:
    symbol: str
    current_price: float
    targets_above: List[TargetProjection]
    targets_below: List[TargetProjection]
    primary_target: Optional[TargetProjection] = None
    notes: List[str] = field(default_factory=list)

    @property
    def has_target(self) -> bool:
        return self.primary_target is not None

    @property
    def best_buy_target(self) -> Optional[TargetProjection]:
        buy = [t for t in self.targets_above if t.direction == "BUY"]
        return max(buy, key=lambda t: t.confidence) if buy else None

    @property
    def best_sell_target(self) -> Optional[TargetProjection]:
        sell = [t for t in self.targets_below if t.direction == "SELL"]
        return max(sell, key=lambda t: t.confidence) if sell else None


CONFLUENCE_WEIGHTS = {
    "liquidity_pool": 3.0,
    "order_block": 2.5,
    "breaker_block": 2.0,
    "fvg": 2.0,
    "fibonacci_618": 2.5,
    "fibonacci_50": 1.5,
    "fibonacci_382": 1.0,
    "swing_high_low": 1.5,
    "psychological": 1.0,
    "previous_high_low": 1.5,
    "eq_zone": 1.0,
    "poc": 2.0,
    "lvn_edge": 1.5,
    "interval_point": 2.0,
    "pressure_zone": 2.5,
}


class ConfluenceScorer:
    """Evalúa confluencias en niveles clave del MarketMap."""

    def __init__(self, proximity_pip_pct: float = 0.05):
        self._proximity_pct = proximity_pip_pct

    def score(self, market_map: MarketMap, route: Optional[Route] = None,
              regime: str = "NEUTRAL", atr_val: float = 0.0) -> ConfluenceResult:
        pip = pip_size(market_map.symbol)
        current_price = market_map.current_price
        targets_above: List[TargetProjection] = []
        targets_below: List[TargetProjection] = []

        # Collect candidate levels from market map
        candidates = self._collect_candidates(market_map, current_price, pip)
        logger.debug(f"[{market_map.symbol}] Confluence candidates: "
                     f"{len(candidates['above'])} above, {len(candidates['below'])} below")

        for price, level_type, tf, direction in candidates["above"]:
            conf = self._score_single(price, level_type, tf, direction,
                                      market_map, pip, current_price, atr_val)
            if conf.confidence > 0:
                targets_above.append(conf)

        for price, level_type, tf, direction in candidates["below"]:
            conf = self._score_single(price, level_type, tf, direction,
                                      market_map, pip, current_price, atr_val)
            if conf.confidence > 0:
                targets_below.append(conf)

        # Merge nearby levels (within 1 ATR)
        targets_above = self._merge_nearby(targets_above, atr_val)
        targets_below = self._merge_nearby(targets_below, atr_val)

        # Sort by confidence
        targets_above.sort(key=lambda t: t.confidence, reverse=True)
        targets_below.sort(key=lambda t: t.confidence, reverse=True)

        # Add route target if available
        if route and route.is_valid:
            for node in route.nodes:
                if node.direction == "BUY" and node.distance_from_entry > 0:
                    tp = TargetProjection(
                        price=node.price, direction="BUY",
                        confidence=min(1.0, route.confidence * 1.1),
                        confluence_count=1,
                        confluences=[f"route_{node.node_type}"],
                        timeframe=node.timeframe,
                        distance_pips=abs(node.price - current_price) / max(pip, 0.0001),
                        estimated_bars_m1=self._estimate_bars(
                            abs(node.price - current_price), atr_val),
                    )
                    targets_above.append(tp)
                elif node.direction == "SELL" and node.distance_from_entry > 0:
                    tp = TargetProjection(
                        price=node.price, direction="SELL",
                        confidence=min(1.0, route.confidence * 1.1),
                        confluence_count=1,
                        confluences=[f"route_{node.node_type}"],
                        timeframe=node.timeframe,
                        distance_pips=abs(node.price - current_price) / max(pip, 0.0001),
                        estimated_bars_m1=self._estimate_bars(
                            abs(node.price - current_price), atr_val),
                    )
                    targets_below.append(tp)

        # Deduplicate and re-sort
        targets_above = self._deduplicate(targets_above)
        targets_below = self._deduplicate(targets_below)

        # Determine primary target
        best = None
        if targets_above and targets_below:
            best = max(targets_above + targets_below, key=lambda t: t.confidence)
        elif targets_above:
            best = targets_above[0]
        elif targets_below:
            best = targets_below[0]

        notes = []
        if best:
            notes.append(f"Primary target: {best.price:.2f} ({best.direction}) "
                         f"@{best.confidence:.0%} ({best.confluence_count} confluences)")
        notes.append(f"Targets above: {len(targets_above)}, below: {len(targets_below)}")

        return ConfluenceResult(
            symbol=market_map.symbol,
            current_price=current_price,
            targets_above=targets_above[:5],
            targets_below=targets_below[:5],
            primary_target=best,
            notes=notes,
        )

    def _collect_candidates(self, mm: MarketMap, current_price: float, pip: float
                            ) -> dict:
        above, below = [], []
        proximity = max(pip * 2, current_price * self._proximity_pct)

        # Liquidity levels
        if mm.nearest_liquidity_above:
            above.append((mm.nearest_liquidity_above, "liquidity_pool", "D1", "SELL"))
        if mm.nearest_liquidity_below:
            below.append((mm.nearest_liquidity_below, "liquidity_pool", "D1", "BUY"))
        if mm.nearest_fvg_above:
            above.append((mm.nearest_fvg_above, "fvg", "M15", "SELL"))
        if mm.nearest_fvg_below:
            below.append((mm.nearest_fvg_below, "fvg", "M15", "BUY"))
        if mm.nearest_ob_above:
            above.append((mm.nearest_ob_above, "order_block", "M15", "SELL"))
        if mm.nearest_ob_below:
            below.append((mm.nearest_ob_below, "order_block", "M15", "BUY"))

        # All zones from map
        for tf_label, tf_zones in mm.zones.items():
            for z in tf_zones:
                ztype = z.zone_type.value if hasattr(z.zone_type, 'value') else str(z.zone_type)
                ztype_lower = ztype.lower()
                # Map zone types to confluence types
                if "breake" in ztype_lower or "breaker" in ztype_lower:
                    ct = "breaker_block"
                elif "order" in ztype_lower and "block" in ztype_lower:
                    ct = "order_block"
                elif "fvg" in ztype_lower:
                    ct = "fvg"
                elif "supply" in ztype_lower or "demand" in ztype_lower:
                    ct = "order_block"
                elif "pressure" in ztype_lower:
                    ct = "pressure_zone"
                elif "eq" in ztype_lower:
                    ct = "eq_zone"
                elif "interval" in ztype_lower:
                    ct = "interval_point"
                else:
                    ct = "swing_high_low"

                zprice = z.midpoint
                if zprice > current_price + proximity:
                    above.append((zprice, ct, z.timeframe, "SELL"))
                elif zprice < current_price - proximity:
                    below.append((zprice, ct, z.timeframe, "BUY"))

        return {"above": above, "below": below}

    def _score_single(self, price: float, level_type: str, timeframe: str,
                      direction: str, market_map: MarketMap, pip: float,
                      current_price: float, atr_val: float) -> TargetProjection:
        base_weight = CONFLUENCE_WEIGHTS.get(level_type, 1.0)

        # Count nearby confluences within proximity band
        proximity = max(pip * 3, atr_val * 0.1)
        nearby = 0
        confluences = [level_type]
        rel_levels = [l for l in market_map.liquidity_levels
                      if abs(l.price - price) <= proximity and l.is_active]
        nearby += len(rel_levels)
        for rl in rel_levels:
            ct = rl.level_type.lower().replace(" ", "_")
            confluences.append(ct)

        # TF weight (higher TF = stronger)
        tf_weight = {"D1": 2.0, "H4": 1.8, "H3": 1.6, "H1": 1.4,
                     "M30": 1.2, "M15": 1.0, "M5": 0.8, "M1": 0.6}
        tmult = tf_weight.get(timeframe, 1.0)

        # Distance weight (closer = more relevant)
        dist = abs(price - current_price) / max(pip, 0.0001)
        dist_weight = max(0.2, min(1.5, 1.0 - (dist / 5000.0)))

        # Touch count bonus
        touch_bonus = 0.0
        for zlist in market_map.zones.values():
            for z in zlist:
                if abs(z.midpoint - price) <= proximity:
                    touch_bonus += min(0.3, z.touch_count * 0.05)

        raw_conf = min(1.0, (base_weight * tmult * dist_weight + touch_bonus) / 5.0)

        # Boost by nearby confluence count
        if nearby > 0:
            raw_conf = min(1.0, raw_conf + nearby * 0.08)

        confluences = list(set(confluences))
        est_bars = self._estimate_bars(abs(price - current_price), atr_val)

        return TargetProjection(
            price=price, direction=direction,
            confidence=round(raw_conf, 4),
            confluence_count=len(confluences),
            confluences=confluences,
            timeframe=timeframe,
            distance_pips=dist,
            estimated_bars_m1=est_bars,
        )

    def _estimate_bars(self, distance: float, atr_val: float) -> int:
        if atr_val <= 0:
            return 999
        # At M1, price moves ~ATR per ~3-5 bars typically
        bars = int(distance / max(atr_val * 0.3, 0.0001))
        return max(1, min(999, bars))

    def _merge_nearby(self, targets: List[TargetProjection], atr_val: float
                      ) -> List[TargetProjection]:
        if not targets:
            return []
        sorted_t = sorted(targets, key=lambda t: t.price)
        merged = [sorted_t[0]]
        for t in sorted_t[1:]:
            prev = merged[-1]
            if abs(t.price - prev.price) <= max(atr_val * 0.5, 0.001):
                # Merge: keep higher confidence
                if t.confidence > prev.confidence:
                    merged[-1] = t
                else:
                    merged[-1].confluence_count += t.confluence_count
                    merged[-1].confluences.extend(t.confluences)
                    merged[-1].confidence = min(1.0, merged[-1].confidence + t.confidence * 0.2)
            else:
                merged.append(t)
        return merged

    def _deduplicate(self, targets: List[TargetProjection]) -> List[TargetProjection]:
        seen = set()
        result = []
        for t in targets:
            key = (round(t.price, 2), t.direction)
            if key not in seen:
                seen.add(key)
                result.append(t)
        return result

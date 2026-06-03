"""Zone State Tracker — Historial de toques por zona (MEJORA SCALPING)
Cada zona (OB, FVG, Breaker, PriceInteraction, PressureZone, etc.) tiene un
contador de toques. Una zona FRESH (0-1 toques) es válida con peso completo.
Una zona EXHAUSTED (4+ toques) es liquidez, no se opera en su dirección.
Las zonas rotas (breakout) se marcan para operar el retest.
"""
import logging
import time
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ZoneStatus(Enum):
    FRESH = "FRESH"
    TOUCHED_1 = "TOUCHED_1"
    TOUCHED_2 = "TOUCHED_2"
    TOUCHED_3 = "TOUCHED_3"
    EXHAUSTED = "EXHAUSTED"
    BROKEN = "BROKEN"
    BROKEN_RETEST = "BROKEN_RETEST"


class ZoneType(Enum):
    ORDER_BLOCK = "ORDER_BLOCK"
    FVG = "FVG"
    BREAKER = "BREAKER"
    LIQUIDITY_SWEEP = "LIQUIDITY_SWEEP"
    PRICE_INTERACTION = "PRICE_INTERACTION"
    PRESSURE_ZONE = "PRESSURE_ZONE"
    HARMONIC_CYCLE = "HARMONIC_CYCLE"
    BOS_ZONE = "BOS_ZONE"
    VOID_SCALP = "VOID_SCALP"


ZONE_TOUCH_PENALTY = {
    ZoneStatus.FRESH: 1.0,
    ZoneStatus.TOUCHED_1: 0.85,
    ZoneStatus.TOUCHED_2: 0.60,
    ZoneStatus.TOUCHED_3: 0.30,
    ZoneStatus.EXHAUSTED: 0.0,
    ZoneStatus.BROKEN: 0.0,
    ZoneStatus.BROKEN_RETEST: 1.3,
}

ZONE_TOUCH_TP_REDUCTION = {
    ZoneStatus.FRESH: 1.0,
    ZoneStatus.TOUCHED_1: 0.85,
    ZoneStatus.TOUCHED_2: 0.65,
    ZoneStatus.TOUCHED_3: 0.40,
    ZoneStatus.EXHAUSTED: 0.0,
    ZoneStatus.BROKEN: 0.0,
    ZoneStatus.BROKEN_RETEST: 0.90,
}


@dataclass
class ZoneRecord:
    zone_id: str
    zone_type: ZoneType
    timeframe: str
    price_level: float
    zone_high: float
    zone_low: float
    direction: str
    touch_count: int = 0
    first_touch_time: Optional[float] = None
    last_touch_time: Optional[float] = None
    is_mitigated: bool = False
    mitigation_pct: float = 0.0
    status: ZoneStatus = ZoneStatus.FRESH
    break_direction: Optional[str] = None
    break_price: Optional[float] = None
    break_time: Optional[float] = None
    created_at: float = field(default_factory=time.time)

    def update_touch(self, price: float, timestamp: float):
        if self.first_touch_time is None:
            self.first_touch_time = timestamp
        self.last_touch_time = timestamp
        self.touch_count += 1

        if self.touch_count >= 4:
            self.status = ZoneStatus.EXHAUSTED
        elif self.touch_count == 3:
            self.status = ZoneStatus.TOUCHED_3
        elif self.touch_count == 2:
            self.status = ZoneStatus.TOUCHED_2
        elif self.touch_count == 1:
            self.status = ZoneStatus.TOUCHED_1

    def mark_broken(self, direction: str, price: float, timestamp: float):
        self.status = ZoneStatus.BROKEN
        self.break_direction = direction
        self.break_price = price
        self.break_time = timestamp

    def mark_retest(self):
        if self.status == ZoneStatus.BROKEN:
            self.status = ZoneStatus.BROKEN_RETEST

    @property
    def weight_multiplier(self) -> float:
        return ZONE_TOUCH_PENALTY.get(self.status, 1.0)

    @property
    def tp_reduction(self) -> float:
        return ZONE_TOUCH_TP_REDUCTION.get(self.status, 1.0)

    @property
    def is_valid_for_entry(self) -> bool:
        return self.status in (ZoneStatus.FRESH, ZoneStatus.TOUCHED_1,
                               ZoneStatus.TOUCHED_2, ZoneStatus.BROKEN_RETEST)

    @property
    def is_exhausted(self) -> bool:
        return self.status == ZoneStatus.EXHAUSTED


def make_zone_id(zone_type: ZoneType, timeframe: str, price: float) -> str:
    raw = f"{zone_type.value}_{timeframe}_{price:.5f}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def is_price_in_zone(price: float, zone_low: float, zone_high: float,
                     touch_buffer_pct: float = 0.05) -> bool:
    zone_range = zone_high - zone_low
    buffer = zone_range * touch_buffer_pct
    return (zone_low - buffer) <= price <= (zone_high + buffer)


def is_zone_mitigated(price: float, direction: str, zone_low: float,
                      zone_high: float, mitigation_pct: float = 0.70) -> Tuple[bool, float]:
    if direction == "BUY":
        pct = min(1.0, max(0.0, (price - zone_low) / max(zone_high - zone_low, 0.0001)))
    else:
        pct = min(1.0, max(0.0, (zone_high - price) / max(zone_high - zone_low, 0.0001)))
    return pct >= mitigation_pct, pct


class ZoneStateTracker:
    """Mantiene el estado de todas las zonas detectadas por símbolo.
    Lleva conteo de toques, mitigación, y determina si una zona
    sigue siendo válida o se ha convertido en liquidez.
    """

    def __init__(self, touch_buffer_pct: float = 0.05,
                 mitigation_threshold: float = 0.70,
                 max_zone_age_hours: float = 72.0):
        self._touch_buffer_pct = touch_buffer_pct
        self._mitigation_threshold = mitigation_threshold
        self._max_zone_age = max_zone_age_hours * 3600
        self._zones: Dict[str, Dict[str, ZoneRecord]] = defaultdict(dict)

    def register_zone(self, symbol: str, zone_type: ZoneType, timeframe: str,
                      price_level: float, zone_high: float, zone_low: float,
                      direction: str) -> ZoneRecord:
        zone_id = make_zone_id(zone_type, timeframe, price_level)
        if zone_id in self._zones[symbol]:
            return self._zones[symbol][zone_id]

        record = ZoneRecord(
            zone_id=zone_id,
            zone_type=zone_type,
            timeframe=timeframe,
            price_level=price_level,
            zone_high=zone_high,
            zone_low=zone_low,
            direction=direction,
        )
        self._zones[symbol][zone_id] = record
        return record

    def get_zone(self, symbol: str, zone_type: ZoneType, timeframe: str,
                 price_level: float) -> Optional[ZoneRecord]:
        zone_id = make_zone_id(zone_type, timeframe, price_level)
        return self._zones[symbol].get(zone_id)

    def update_price(self, symbol: str, price: float, timestamp: Optional[float] = None):
        if timestamp is None:
            timestamp = time.time()
        now = timestamp
        to_delete = []
        for zone_id, zone in self._zones[symbol].items():
            if now - zone.created_at > self._max_zone_age:
                to_delete.append(zone_id)
                continue
            if zone.status in (ZoneStatus.BROKEN, ZoneStatus.EXHAUSTED):
                continue
            if is_price_in_zone(price, zone.zone_low, zone.zone_high, self._touch_buffer_pct):
                zone.update_touch(price, timestamp)
                mitigated, pct = is_zone_mitigated(
                    price, zone.direction, zone.zone_low, zone.zone_high,
                    self._mitigation_threshold
                )
                zone.mitigation_pct = pct
                if mitigated:
                    zone.is_mitigated = True
            elif zone.status == ZoneStatus.BROKEN and zone.break_direction:
                retest_dir = "BUY" if zone.break_direction == "SELL" else "SELL"
                if is_price_in_zone(price, zone.zone_low, zone.zone_high, self._touch_buffer_pct * 3):
                    zone.mark_retest()

        for zid in to_delete:
            del self._zones[symbol][zid]

    def detect_breakout(self, symbol: str, price: float, direction: str,
                        timestamp: Optional[float] = None):
        if timestamp is None:
            timestamp = time.time()
        for zone in self._zones[symbol].values():
            if zone.status in (ZoneStatus.BROKEN, ZoneStatus.EXHAUSTED):
                continue
            if direction == "BUY" and price > zone.zone_high:
                zone.mark_broken("BUY", price, timestamp)
                logger.info(f"[{symbol}] Zona {zone.zone_type.value}@{zone.price_level:.5f} "
                            f"ROTA al alza (touch #{zone.touch_count})")
            elif direction == "SELL" and price < zone.zone_low:
                zone.mark_broken("SELL", price, timestamp)
                logger.info(f"[{symbol}] Zona {zone.zone_type.value}@{zone.price_level:.5f} "
                            f"ROTA a la baja (touch #{zone.touch_count})")

    def get_valid_zones(self, symbol: str, direction: Optional[str] = None) -> List[ZoneRecord]:
        result = []
        for zone in self._zones[symbol].values():
            if not zone.is_valid_for_entry:
                continue
            if direction and zone.direction != direction:
                continue
            result.append(zone)
        return result

    def get_exhausted_zones(self, symbol: str) -> List[ZoneRecord]:
        return [z for z in self._zones[symbol].values() if z.is_exhausted]

    def get_broken_zones(self, symbol: str,
                         retest_only: bool = False) -> List[ZoneRecord]:
        if retest_only:
            return [z for z in self._zones[symbol].values()
                    if z.status == ZoneStatus.BROKEN_RETEST]
        return [z for z in self._zones[symbol].values()
                if z.status in (ZoneStatus.BROKEN, ZoneStatus.BROKEN_RETEST)]

    def get_zone_multiplier(self, symbol: str, zone_type: ZoneType,
                            timeframe: str, price_level: float) -> float:
        zone = self.get_zone(symbol, zone_type, timeframe, price_level)
        if zone is None:
            return 1.0
        return zone.weight_multiplier

    def get_zone_tp_reduction(self, symbol: str, zone_type: ZoneType,
                              timeframe: str, price_level: float) -> float:
        zone = self.get_zone(symbol, zone_type, timeframe, price_level)
        if zone is None:
            return 1.0
        if zone.status == ZoneStatus.BROKEN_RETEST:
            return 1.0
        return zone.tp_reduction

    def cleanup(self, symbol: str):
        now = time.time()
        self._zones[symbol] = {
            zid: z for zid, z in self._zones[symbol].items()
            if now - z.created_at <= self._max_zone_age
        }

    def get_stats(self, symbol: str) -> dict:
        zones = self._zones.get(symbol, {})
        return {
            "total": len(zones),
            "fresh": sum(1 for z in zones.values() if z.status == ZoneStatus.FRESH),
            "touched_1": sum(1 for z in zones.values() if z.status == ZoneStatus.TOUCHED_1),
            "touched_2": sum(1 for z in zones.values() if z.status == ZoneStatus.TOUCHED_2),
            "touched_3": sum(1 for z in zones.values() if z.status == ZoneStatus.TOUCHED_3),
            "exhausted": sum(1 for z in zones.values() if z.status == ZoneStatus.EXHAUSTED),
            "broken": sum(1 for z in zones.values() if z.status == ZoneStatus.BROKEN),
            "broken_retest": sum(1 for z in zones.values() if z.status == ZoneStatus.BROKEN_RETEST),
        }

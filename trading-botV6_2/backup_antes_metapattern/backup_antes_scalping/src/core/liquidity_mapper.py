"""Liquidity Mapper — Mapeo vertical D1→4H→1H→M15→M5→M1 (MEJORA SCALPING)
Escanea cada timeframe y construye un mapa completo de:
  - Zonas de liquidez (swing highs/lows mayores)
  - FVGs no mitigados
  - Order Blocks no mitigados
  - Zonas de interacción de precio
  - Breakers
  - Puntos de equilibrio (EQ)

El mapa se usa para determinar hacia dónde se dirige el precio
y qué zonas están en el camino.
"""
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from src.core.zone_state_tracker import (
    ZoneStateTracker, ZoneRecord, ZoneType, ZoneStatus,
    is_price_in_zone, is_zone_mitigated,
)
from src.utils.helpers import atr, find_swing_points, pip_size

logger = logging.getLogger(__name__)

TIMEFRAME_HIERARCHY = ["D1", "H4", "H3", "H1", "M30", "M15", "M5", "M1"]


@dataclass
class LiquidityLevel:
    price: float
    strength: float
    timeframe: str
    level_type: str
    touch_count: int = 0
    is_active: bool = True

    def __lt__(self, other):
        return self.price < other.price


@dataclass
class MarketZone:
    zone_type: ZoneType
    timeframe: str
    price_level: float
    zone_high: float
    zone_low: float
    direction: str
    strength: float
    touch_count: int = 0
    is_mitigated: bool = False
    mitigation_pct: float = 0.0
    zone_id: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def midpoint(self) -> float:
        return (self.zone_high + self.zone_low) / 2.0

    @property
    def height_pips(self) -> float:
        return self.zone_high - self.zone_low


@dataclass
class MarketMap:
    symbol: str
    current_price: float
    zones: Dict[str, List[MarketZone]] = field(default_factory=dict)
    liquidity_levels: List[LiquidityLevel] = field(default_factory=list)
    primary_trend: str = "NEUTRAL"
    trend_strength: float = 0.0
    nearest_liquidity_above: Optional[float] = None
    nearest_liquidity_below: Optional[float] = None
    nearest_fvg_above: Optional[float] = None
    nearest_fvg_below: Optional[float] = None
    nearest_ob_above: Optional[float] = None
    nearest_ob_below: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    @property
    def has_target_above(self) -> bool:
        return self.nearest_liquidity_above is not None

    @property
    def has_target_below(self) -> bool:
        return self.nearest_liquidity_below is not None

    @property
    def distance_to_liquidity_above_pips(self) -> Optional[float]:
        if self.nearest_liquidity_above:
            return self.nearest_liquidity_above - self.current_price
        return None

    @property
    def distance_to_liquidity_below_pips(self) -> Optional[float]:
        if self.nearest_liquidity_below:
            return self.current_price - self.nearest_liquidity_below
        return None

    @property
    def dominant_direction(self) -> str:
        if self.primary_trend in ("BULLISH", "STRONG_BULLISH"):
            return "BUY"
        elif self.primary_trend in ("BEARISH", "STRONG_BEARISH"):
            return "SELL"
        return "NEUTRAL"


class LiquidityMapper:
    """Construye el mapa vertical del mercado desde D1 hasta M1.

    Escanea cada timeframe en orden jerárquico y detecta:
      - Swing highs/lows como liquidez
      - FVGs, OBs, Breakers, zonas de interacción
      - Tendencia primaria por convergencia de timeframes
    """

    def __init__(self, lookback_candles: int = 100,
                 swing_lookback: int = 10,
                 min_zone_strength: float = 0.3,
                 zone_overlap_pips: float = 2.0):
        self._lookback = lookback_candles
        self._swing_lookback = swing_lookback
        self._min_strength = min_zone_strength
        self._overlap_pips = zone_overlap_pips
        self._pip_cache: Dict[str, float] = {}

    def build(self, symbol: str, timeframes: Dict[str, pd.DataFrame],
              zone_tracker: Optional[ZoneStateTracker] = None) -> MarketMap:
        pip = self._get_pip(symbol)
        current_price = self._get_current_price(timeframes)
        if current_price is None:
            return self._empty_map(symbol, 0.0)

        zones: Dict[str, List[MarketZone]] = defaultdict(list)
        liq_levels: List[LiquidityLevel] = []
        primary_trend = "NEUTRAL"
        trend_votes: Dict[str, int] = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0}
        tf_directions: List[str] = []

        for tf in TIMEFRAME_HIERARCHY:
            df = timeframes.get(tf)
            if df is None or len(df) < self._swing_lookback * 2:
                continue
            strength = self._tf_weight(tf)

            swing_dir = self._detect_swing_trend(df)
            tf_directions.append(swing_dir)
            trend_votes[swing_dir] += strength

            tf_liq = self._detect_liquidity_levels(df, tf, pip)
            liq_levels.extend(tf_liq)

            self._scan_fvg_zones(df, tf, pip, strength, zones, zone_tracker, symbol)
            self._scan_ob_zones(df, tf, pip, strength, zones, zone_tracker, symbol)
            self._scan_breaker_zones(df, tf, pip, strength, zones, zone_tracker, symbol)

        trend_total = sum(trend_votes.values())
        if trend_total > 0:
            bullish_pct = trend_votes["BULLISH"] / trend_total
            bearish_pct = trend_votes["BEARISH"] / trend_total
            if bullish_pct > 0.6:
                primary_trend = "STRONG_BULLISH"
            elif bullish_pct > 0.4:
                primary_trend = "BULLISH"
            elif bearish_pct > 0.6:
                primary_trend = "STRONG_BEARISH"
            elif bearish_pct > 0.4:
                primary_trend = "BEARISH"
            else:
                primary_trend = "NEUTRAL"

        liq_above = [l for l in liq_levels if l.price > current_price and l.is_active]
        liq_below = [l for l in liq_levels if l.price < current_price and l.is_active]
        liq_above.sort()
        liq_below.sort(reverse=True)

        fvg_above = self._zones_above(zones.get("FVG", []), current_price)
        fvg_below = self._zones_below(zones.get("FVG", []), current_price)
        ob_above = self._zones_above(zones.get("ORDER_BLOCK", []), current_price)
        ob_below = self._zones_below(zones.get("ORDER_BLOCK", []), current_price)

        mm = MarketMap(
            symbol=symbol,
            current_price=current_price,
            zones=dict(zones),
            liquidity_levels=liq_levels,
            primary_trend=primary_trend,
            trend_strength=trend_votes.get(primary_trend.replace("STRONG_", ""), 0) / max(trend_total, 1),
            nearest_liquidity_above=liq_above[0].price if liq_above else None,
            nearest_liquidity_below=liq_below[0].price if liq_below else None,
            nearest_fvg_above=fvg_above[0].price_level if fvg_above else None,
            nearest_fvg_below=fvg_below[0].price_level if fvg_below else None,
            nearest_ob_above=ob_above[0].price_level if ob_above else None,
            nearest_ob_below=ob_below[0].price_level if ob_below else None,
        )

        if primary_trend in ("BULLISH", "STRONG_BULLISH") and mm.nearest_liquidity_above:
            mm.notes.append(f"Tendencia {primary_trend} → liquidez en {mm.nearest_liquidity_above:.5f}")
            if mm.nearest_fvg_above:
                mm.notes.append(f"FVG intermedio en {mm.nearest_fvg_above:.5f}")
        elif primary_trend in ("BEARISH", "STRONG_BEARISH") and mm.nearest_liquidity_below:
            mm.notes.append(f"Tendencia {primary_trend} → liquidez en {mm.nearest_liquidity_below:.5f}")
            if mm.nearest_fvg_below:
                mm.notes.append(f"FVG intermedio en {mm.nearest_fvg_below:.5f}")

        return mm

    def _empty_map(self, symbol: str, price: float) -> MarketMap:
        return MarketMap(symbol=symbol, current_price=price)

    def _get_pip(self, symbol: str) -> float:
        if symbol not in self._pip_cache:
            self._pip_cache[symbol] = pip_size(symbol)
        return self._pip_cache[symbol]

    def _get_current_price(self, timeframes: Dict[str, pd.DataFrame]) -> Optional[float]:
        for tf in TIMEFRAME_HIERARCHY:
            df = timeframes.get(tf)
            if df is not None and len(df) > 0:
                return float(df["close"].iloc[-1])
        return None

    def _tf_weight(self, tf: str) -> int:
        weights = {"D1": 7, "H4": 6, "H3": 5, "H1": 4, "M30": 3, "M15": 2, "M5": 1, "M1": 1}
        return weights.get(tf, 1)

    def _detect_swing_trend(self, df: pd.DataFrame) -> str:
        highs, lows = find_swing_points(df, lookback=self._swing_lookback)
        if len(highs) < 3 or len(lows) < 3:
            return "NEUTRAL"
        hh = df["high"].iloc[highs[-1]] > df["high"].iloc[highs[-2]] > df["high"].iloc[highs[-3]]
        ll = df["low"].iloc[lows[-1]] > df["low"].iloc[lows[-2]] > df["low"].iloc[lows[-3]]
        LH = df["high"].iloc[highs[-1]] < df["high"].iloc[highs[-2]] < df["high"].iloc[highs[-3]]
        HL = df["low"].iloc[lows[-1]] < df["low"].iloc[lows[-2]] < df["low"].iloc[lows[-3]]
        if hh and ll:
            return "BULLISH"
        elif LH and HL:
            return "BEARISH"
        elif hh:
            return "BULLISH"
        elif LH:
            return "BEARISH"
        return "NEUTRAL"

    def _detect_liquidity_levels(self, df: pd.DataFrame, tf: str,
                                 pip: float) -> List[LiquidityLevel]:
        levels = []
        highs, lows = find_swing_points(df, lookback=self._swing_lookback)
        for idx in highs[-4:]:
            price = float(df["high"].iloc[idx])
            volume = float(df["volume"].iloc[idx]) if "volume" in df.columns else 0
            strength = min(1.0, volume / 1000 + 0.3) if volume > 0 else 0.5
            levels.append(LiquidityLevel(price, strength, tf, "SWING_HIGH"))
        for idx in lows[-4:]:
            price = float(df["low"].iloc[idx])
            volume = float(df["volume"].iloc[idx]) if "volume" in df.columns else 0
            strength = min(1.0, volume / 1000 + 0.3) if volume > 0 else 0.5
            levels.append(LiquidityLevel(price, strength, tf, "SWING_LOW"))
        return levels

    def _scan_fvg_zones(self, df: pd.DataFrame, tf: str, pip: float,
                        strength: float, zones: Dict[str, List[MarketZone]],
                        tracker: Optional[ZoneStateTracker], symbol: str):
        for i in range(3, len(df) - 1):
            c1, c2, c3 = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
            if c1["high"] < c2["low"]:
                zone_type = ZoneType.FVG
                direction = "BUY"
                low, high = c1["high"], c2["low"]
                pct_strength = strength * min(1.0, (high - low) / (pip * 5))
                if pct_strength < self._min_strength:
                    continue
                price_lvl = (low + high) / 2
                mz = MarketZone(
                    zone_type=zone_type, timeframe=tf,
                    price_level=price_lvl, zone_high=high, zone_low=low,
                    direction=direction, strength=pct_strength,
                    notes=[f"{tf} FVG {direction}"],
                )
                if not self._overlaps_existing(zones.get("FVG", []), mz):
                    zones["FVG"].append(mz)
                    if tracker:
                        tracker.register_zone(symbol, zone_type, tf, price_lvl, high, low, direction)
            elif c2["low"] > c3["high"]:
                zone_type = ZoneType.FVG
                direction = "SELL"
                low, high = c3["high"], c2["low"]
                pct_strength = strength * min(1.0, (high - low) / (pip * 5))
                if pct_strength < self._min_strength:
                    continue
                price_lvl = (low + high) / 2
                mz = MarketZone(
                    zone_type=zone_type, timeframe=tf,
                    price_level=price_lvl, zone_high=high, zone_low=low,
                    direction=direction, strength=pct_strength,
                    notes=[f"{tf} FVG {direction}"],
                )
                if not self._overlaps_existing(zones.get("FVG", []), mz):
                    zones["FVG"].append(mz)
                    if tracker:
                        tracker.register_zone(symbol, zone_type, tf, price_lvl, high, low, direction)

    def _scan_ob_zones(self, df: pd.DataFrame, tf: str, pip: float,
                       strength: float, zones: Dict[str, List[MarketZone]],
                       tracker: Optional[ZoneStateTracker], symbol: str):
        for i in range(2, len(df) - 1):
            prev, curr = df.iloc[i - 1], df.iloc[i]
            body = abs(curr["close"] - curr["open"])
            range_c = curr["high"] - curr["low"]
            if range_c == 0:
                continue
            body_ratio = body / range_c
            if body_ratio < 0.4:
                continue
            if curr["close"] > curr["open"]:
                zone_type = ZoneType.ORDER_BLOCK
                direction = "BUY"
                low, high = curr["open"], curr["high"]
                pct_strength = strength * body_ratio
                if pct_strength < self._min_strength:
                    continue
                price_lvl = (low + high) / 2
                mz = MarketZone(
                    zone_type=zone_type, timeframe=tf,
                    price_level=price_lvl, zone_high=high, zone_low=low,
                    direction=direction, strength=pct_strength,
                    notes=[f"{tf} OB {direction}"],
                )
                if not self._overlaps_existing(zones.get("ORDER_BLOCK", []), mz):
                    zones["ORDER_BLOCK"].append(mz)
                    if tracker:
                        tracker.register_zone(symbol, zone_type, tf, price_lvl, high, low, direction)
            else:
                zone_type = ZoneType.ORDER_BLOCK
                direction = "SELL"
                low, high = curr["low"], curr["open"]
                pct_strength = strength * body_ratio
                if pct_strength < self._min_strength:
                    continue
                price_lvl = (low + high) / 2
                mz = MarketZone(
                    zone_type=zone_type, timeframe=tf,
                    price_level=price_lvl, zone_high=high, zone_low=low,
                    direction=direction, strength=pct_strength,
                    notes=[f"{tf} OB {direction}"],
                )
                if not self._overlaps_existing(zones.get("ORDER_BLOCK", []), mz):
                    zones["ORDER_BLOCK"].append(mz)
                    if tracker:
                        tracker.register_zone(symbol, zone_type, tf, price_lvl, high, low, direction)

    def _scan_breaker_zones(self, df: pd.DataFrame, tf: str, pip: float,
                            strength: float, zones: Dict[str, List[MarketZone]],
                            tracker: Optional[ZoneStateTracker], symbol: str):
        highs, lows = find_swing_points(df, lookback=self._swing_lookback)
        if len(highs) < 4 or len(lows) < 4:
            return
        h1, h2 = df["high"].iloc[highs[-2]], df["high"].iloc[highs[-1]]
        l1, l2 = df["low"].iloc[lows[-2]], df["low"].iloc[lows[-1]]
        close = df["close"].iloc[-1]
        if h2 > h1 and close < l1:
            zones["BREAKER"].append(MarketZone(
                zone_type=ZoneType.BREAKER, timeframe=tf,
                price_level=(l1 + h2) / 2, zone_high=h2, zone_low=l1,
                direction="SELL", strength=strength * 0.8,
                notes=[f"{tf} Breaker SELL"],
            ))
            if tracker:
                tracker.register_zone(symbol, ZoneType.BREAKER, tf, (l1 + h2) / 2, h2, l1, "SELL")
        elif l2 < l1 and close > h1:
            zones["BREAKER"].append(MarketZone(
                zone_type=ZoneType.BREAKER, timeframe=tf,
                price_level=(h1 + l2) / 2, zone_high=h1, zone_low=l2,
                direction="BUY", strength=strength * 0.8,
                notes=[f"{tf} Breaker BUY"],
            ))
            if tracker:
                tracker.register_zone(symbol, ZoneType.BREAKER, tf, (h1 + l2) / 2, h1, l2, "BUY")

    def _overlaps_existing(self, existing: List[MarketZone], candidate: MarketZone) -> bool:
        for z in existing:
            overlap = min(z.zone_high, candidate.zone_high) - max(z.zone_low, candidate.zone_low)
            if overlap > 0 and z.timeframe == candidate.timeframe:
                return True
        return False

    def _zones_above(self, zones: List[MarketZone], price: float) -> List[MarketZone]:
        return sorted([z for z in zones if z.price_level > price and z.strength >= self._min_strength],
                      key=lambda z: z.price_level)

    def _zones_below(self, zones: List[MarketZone], price: float) -> List[MarketZone]:
        return sorted([z for z in zones if z.price_level < price and z.strength >= self._min_strength],
                      key=lambda z: z.price_level, reverse=True)

    def get_opposing_zones(self, market_map: MarketMap, direction: str,
                           max_distance_pips: float = 50.0) -> List[MarketZone]:
        result = []
        for tf_zones in market_map.zones.values():
            for z in tf_zones:
                if z.direction != direction:
                    continue
                dist = abs(z.price_level - market_map.current_price)
                if dist <= max_distance_pips * self._get_pip(market_map.symbol):
                    result.append(z)
        return sorted(result, key=lambda z: abs(z.price_level - market_map.current_price))

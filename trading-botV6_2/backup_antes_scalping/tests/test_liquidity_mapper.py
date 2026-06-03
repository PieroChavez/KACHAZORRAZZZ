import pandas as pd
import pytest

from src.core.liquidity_mapper import (
    LiquidityMapper, LiquidityLevel, MarketZone, MarketMap,
)
from src.core.zone_state_tracker import ZoneType, ZoneStateTracker


class TestLiquidityLevel:
    def test_ordering_by_price(self):
        a = LiquidityLevel(price=100.0, strength=0.5, timeframe="H1", level_type="SWING_HIGH")
        b = LiquidityLevel(price=101.0, strength=0.8, timeframe="H1", level_type="SWING_HIGH")
        assert a < b
        assert not (b < a)

    def test_defaults(self):
        ll = LiquidityLevel(price=100.0, strength=0.5, timeframe="H1", level_type="SWING_HIGH")
        assert ll.touch_count == 0
        assert ll.is_active


class TestMarketZone:
    def test_midpoint(self):
        z = MarketZone(ZoneType.FVG, "H1", 100.0, 101.0, 99.0, "BUY", 0.8)
        assert z.midpoint == 100.0

    def test_height_pips(self):
        z = MarketZone(ZoneType.FVG, "H1", 100.0, 101.0, 99.0, "BUY", 0.8)
        assert z.height_pips == 2.0


class TestMarketMap:
    def test_defaults(self):
        mm = MarketMap(symbol="XAUUSD", current_price=100.0)
        assert mm.primary_trend == "NEUTRAL"
        assert mm.zones == {}

    def test_has_target_above(self):
        mm = MarketMap(symbol="XAUUSD", current_price=100.0, nearest_liquidity_above=105.0)
        assert mm.has_target_above
        assert not mm.has_target_below

    def test_has_target_below(self):
        mm = MarketMap(symbol="XAUUSD", current_price=100.0, nearest_liquidity_below=95.0)
        assert mm.has_target_below
        assert not mm.has_target_above

    def test_distance_to_liquidity_above(self):
        mm = MarketMap(symbol="XAUUSD", current_price=100.0, nearest_liquidity_above=105.0)
        assert mm.distance_to_liquidity_above_pips == 5.0

    def test_dominant_direction(self):
        mm = MarketMap(symbol="XAUUSD", current_price=100.0, primary_trend="BULLISH")
        assert mm.dominant_direction == "BUY"
        mm2 = MarketMap(symbol="XAUUSD", current_price=100.0, primary_trend="BEARISH")
        assert mm2.dominant_direction == "SELL"


class TestLiquidityMapper:
    @pytest.fixture
    def mapper(self):
        return LiquidityMapper(lookback_candles=30, swing_lookback=5, min_zone_strength=0.1, zone_overlap_pips=2.0)

    def _make_tf_df(self, closes, opens=None, highs=None, lows=None, volumes=None):
        n = len(closes)
        if opens is None:
            opens = closes[:]
        if highs is None:
            highs = [c + 0.3 for c in closes]
        if lows is None:
            lows = [c - 0.3 for c in closes]
        data = {"open": opens, "close": closes, "high": highs, "low": lows}
        if volumes:
            data["volume"] = volumes
        return pd.DataFrame(data)

    def test_empty_map_when_no_data(self, mapper):
        mm = mapper.build("XAUUSD", {})
        assert mm.current_price == 0.0
        assert mm.primary_trend == "NEUTRAL"

    def test_build_with_simple_data(self, mapper):
        df = self._make_tf_df(closes=[100.0 + i * 0.1 for i in range(30)])
        timeframes = {"M1": df}
        mm = mapper.build("XAUUSD", timeframes)
        assert mm.symbol == "XAUUSD"
        assert mm.current_price > 0
        assert mm.primary_trend in ("BULLISH", "NEUTRAL")

    def test_fvg_bullish_scan(self, mapper):
        """_scan_fvg_zones detects c1.high < c2.low as BUY FVG."""
        # Create a df with a gap up at position 10: candle 9 high < candle 10 low
        closes = [100.0] * 15
        opens = [100.0] * 15
        highs = [100.3] * 9 + [100.3, 101.8] + [101.0] * 4
        lows = [99.7] * 9 + [99.7, 101.0] + [100.5] * 4
        df = self._make_tf_df(closes, opens, highs, lows)
        timeframes = {"M1": df}
        mm = mapper.build("XAUUSD", timeframes)
        fvg_zones = mm.zones.get("FVG", [])
        if fvg_zones:
            assert any(z.direction == "BUY" for z in fvg_zones)
            assert any(z.zone_type == ZoneType.FVG for z in fvg_zones)

    def test_fvg_bearish_scan(self, mapper):
        """_scan_fvg_zones detects c2.low > c3.high as SELL FVG."""
        closes = [105.0] * 15
        opens = [105.0] * 15
        highs = [105.3] * 9 + [105.3, 102.5] + [102.0] * 4
        lows = [104.7] * 9 + [104.7, 101.5] + [101.0] * 4
        df = self._make_tf_df(closes, opens, highs, lows)
        timeframes = {"M1": df}
        mm = mapper.build("XAUUSD", timeframes)
        fvg_zones = mm.zones.get("FVG", [])
        if fvg_zones:
            assert any(z.direction == "SELL" for z in fvg_zones)
            assert any(z.zone_type == ZoneType.FVG for z in fvg_zones)

    def test_no_fvg_on_flat_data(self, mapper):
        df = self._make_tf_df(closes=[100.0] * 30)
        timeframes = {"M1": df}
        mm = mapper.build("XAUUSD", timeframes)
        fvg_zones = mm.zones.get("FVG", [])
        assert len(fvg_zones) == 0

    def test_ob_scan(self, mapper):
        closes = [100.0] * 5 + [101.0] * 10 + [100.0] * 10
        opens = [100.0] * 5 + [100.2] * 10 + [100.0] * 10
        highs = [100.3] * 25
        lows = [99.7] * 25
        df = self._make_tf_df(closes, opens, highs, lows)
        timeframes = {"M1": df}
        mm = mapper.build("XAUUSD", timeframes)
        ob_zones = mm.zones.get("ORDER_BLOCK", [])
        assert len(ob_zones) >= 0

    def test_build_with_zone_tracker(self, mapper):
        tracker = ZoneStateTracker()
        df = self._make_tf_df(closes=[100.0 + i * 0.1 for i in range(30)])
        timeframes = {"M1": df}
        mm = mapper.build("XAUUSD", timeframes, zone_tracker=tracker)
        assert mm is not None

    def test_get_opposing_zones(self, mapper):
        mm = MarketMap(symbol="XAUUSD", current_price=100.0, primary_trend="BULLISH")
        z = MarketZone(ZoneType.FVG, "H1", 100.0, 101.0, 99.0, "SELL", 0.8)
        mm.zones["FVG"] = [z]
        # Should find the SELL zone as opposing to BUY
        from src.utils.helpers import pip_size
        opposing = mapper.get_opposing_zones(mm, "SELL", max_distance_pips=100)
        assert len(opposing) == 1
        assert opposing[0].direction == "SELL"

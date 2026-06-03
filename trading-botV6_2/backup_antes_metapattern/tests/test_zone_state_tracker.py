import time
import pytest

from src.core.zone_state_tracker import (
    ZoneStateTracker, ZoneRecord, ZoneStatus, ZoneType,
    is_price_in_zone, is_zone_mitigated, make_zone_id,
)


class TestZoneUtils:
    def test_is_price_in_zone(self):
        assert is_price_in_zone(100.0, 99.0, 101.0)
        assert not is_price_in_zone(102.0, 99.0, 101.0)
        assert is_price_in_zone(101.05, 99.0, 101.0)  # within buffer

    def test_is_zone_mitigated_buy(self):
        mitigated, pct = is_zone_mitigated(101.5, "BUY", 100.0, 102.0)
        # pct = (101.5-100.0) / (102.0-100.0) = 1.5/2.0 = 0.75 >= 0.70 → True
        assert mitigated
        assert pct == 0.75

    def test_is_zone_mitigated_sell(self):
        mitigated, pct = is_zone_mitigated(100.5, "SELL", 100.0, 102.0)
        # pct = (102.0-100.5) / (102.0-100.0) = 1.5/2.0 = 0.75 >= 0.70
        assert mitigated
        assert pct == 0.75

    def test_make_zone_id_deterministic(self):
        z1 = make_zone_id(ZoneType.FVG, "H1", 100.0)
        z2 = make_zone_id(ZoneType.FVG, "H1", 100.0)
        assert z1 == z2
        assert len(z1) == 12


class TestZoneRecord:
    def test_initial_state(self):
        z = ZoneRecord("id1", ZoneType.FVG, "H1", 100.0, 101.0, 99.0, "BUY")
        assert z.status == ZoneStatus.FRESH
        assert z.touch_count == 0
        assert z.weight_multiplier == 1.0
        assert z.is_valid_for_entry

    def test_update_touch_progression(self):
        z = ZoneRecord("id1", ZoneType.FVG, "H1", 100.0, 101.0, 99.0, "BUY")
        ts = time.time()
        z.update_touch(100.0, ts)
        assert z.touch_count == 1
        assert z.status == ZoneStatus.TOUCHED_1

        z.update_touch(100.5, ts + 1)
        assert z.touch_count == 2
        assert z.status == ZoneStatus.TOUCHED_2

        z.update_touch(100.3, ts + 2)
        assert z.touch_count == 3
        assert z.status == ZoneStatus.TOUCHED_3

        z.update_touch(100.1, ts + 3)
        assert z.touch_count == 4
        assert z.status == ZoneStatus.EXHAUSTED
        assert not z.is_valid_for_entry

    def test_mark_broken(self):
        z = ZoneRecord("id1", ZoneType.FVG, "H1", 100.0, 101.0, 99.0, "BUY")
        z.mark_broken("SELL", 98.0, time.time())
        assert z.status == ZoneStatus.BROKEN
        assert z.break_direction == "SELL"

    def test_mark_retest(self):
        z = ZoneRecord("id1", ZoneType.FVG, "H1", 100.0, 101.0, 99.0, "BUY")
        z.mark_broken("SELL", 98.0, time.time())
        z.mark_retest()
        assert z.status == ZoneStatus.BROKEN_RETEST


class TestZoneStateTracker:
    @pytest.fixture
    def tracker(self):
        return ZoneStateTracker(touch_buffer_pct=0.05, mitigation_threshold=0.70, max_zone_age_hours=72)

    def test_register_and_get_zone(self, tracker):
        zone = tracker.register_zone("XAUUSD", ZoneType.FVG, "H1", 100.0, 101.0, 99.0, "BUY")
        assert zone.zone_type == ZoneType.FVG
        assert tracker.get_zone("XAUUSD", ZoneType.FVG, "H1", 100.0) is not None

    def test_register_duplicate_returns_same(self, tracker):
        z1 = tracker.register_zone("XAUUSD", ZoneType.FVG, "H1", 100.0, 101.0, 99.0, "BUY")
        z2 = tracker.register_zone("XAUUSD", ZoneType.FVG, "H1", 100.0, 101.0, 99.0, "BUY")
        assert z1 is z2

    def test_update_price_touches_zone(self, tracker):
        zone = tracker.register_zone("XAUUSD", ZoneType.FVG, "H1", 100.0, 101.0, 99.0, "BUY")
        tracker.update_price("XAUUSD", 100.5)
        assert zone.touch_count >= 1

    def test_detect_breakout(self, tracker):
        zone = tracker.register_zone("XAUUSD", ZoneType.FVG, "H1", 100.0, 101.0, 99.0, "BUY")
        tracker.detect_breakout("XAUUSD", 102.0, "BUY")
        assert zone.status == ZoneStatus.BROKEN

    def test_get_broken_zones(self, tracker):
        tracker.register_zone("XAUUSD", ZoneType.FVG, "H1", 100.0, 101.0, 99.0, "BUY")
        tracker.detect_breakout("XAUUSD", 102.0, "BUY")
        broken = tracker.get_broken_zones("XAUUSD")
        assert len(broken) == 1

    def test_get_exhausted_zones(self, tracker):
        zone = tracker.register_zone("XAUUSD", ZoneType.FVG, "H1", 100.0, 101.0, 99.0, "BUY")
        ts = time.time()
        for i in range(4):
            zone.update_touch(100.5, ts + i)
        exhausted = tracker.get_exhausted_zones("XAUUSD")
        assert len(exhausted) == 1

    def test_get_stats(self, tracker):
        z = tracker.register_zone("XAUUSD", ZoneType.FVG, "H1", 100.0, 101.0, 99.0, "BUY")
        stats = tracker.get_stats("XAUUSD")
        assert stats["total"] == 1
        assert stats["fresh"] == 1
        z.update_touch(100.5, time.time())
        stats = tracker.get_stats("XAUUSD")
        assert stats["touched_1"] == 1

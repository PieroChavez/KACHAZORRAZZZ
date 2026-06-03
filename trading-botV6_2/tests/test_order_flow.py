from unittest.mock import patch, MagicMock, PropertyMock
import numpy as np
import pandas as pd
import pytest

from src.core.order_flow import (
    OrderFlowConfig,
    OrderFlowEngine,
    OrderFlowSignal,
    DOMClusterEngine,
    DOMSnapshot,
    DOMLevel,
    DOMCluster,
    DeltaMACD,
    TickRecord,
)


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture
def cfg():
    return OrderFlowConfig(
        expert_mode=True,
        dom_depth_levels=15,
        dom_cluster_pips=2.0,
        absorption_samples=3,
        iceberg_chunk_min=3,
        stop_run_lookback=20,
        tick_history_size=1000,
        delta_ma_period=14,
        imbalance_threshold=0.20,
    )


@pytest.fixture
def df_with_volume():
    """30 velas OHLC con volumen para OrderFlowEngine._analyze_ticks."""
    n = 30
    opens = [100.0 + i * 0.1 for i in range(n)]
    closes = [o + (0.2 if i % 2 == 0 else -0.1) for i, o in enumerate(opens)]
    highs = [max(o, c) + 0.15 for o, c in zip(opens, closes)]
    lows = [min(o, c) - 0.15 for o, c in zip(opens, closes)]
    volumes = [100 + i * 10 for i in range(n)]
    return pd.DataFrame({
        "open": opens, "close": closes,
        "high": highs, "low": lows, "volume": volumes,
    })


@pytest.fixture
def df_bullish_volume():
    """30 velas con fuerte delta positivo."""
    n = 30
    opens = [100.0 + i * 0.2 for i in range(n)]
    closes = [o + 0.3 for o in opens]
    highs = [c + 0.1 for c in closes]
    lows = [o - 0.1 for o in opens]
    volumes = [200 + i * 5 for i in range(n)]
    return pd.DataFrame({
        "open": opens, "close": closes,
        "high": highs, "low": lows, "volume": volumes,
    })


def make_snapshot(
    bid_prices=(1.1000, 1.0998, 1.0995, 1.0992, 1.0988),
    ask_prices=(1.1002, 1.1005, 1.1008, 1.1010, 1.1015),
    bid_volumes=(10.0, 15.0, 5.0, 8.0, 12.0),
    ask_volumes=(12.0, 8.0, 6.0, 10.0, 14.0),
    classification=None,
):
    bid_levels = [
        DOMLevel(price=p, bid_volume=v, ask_volume=0, bid_order_count=int(v), total_volume=v)
        for p, v in zip(bid_prices, bid_volumes)
    ]
    ask_levels = [
        DOMLevel(price=p, bid_volume=0, ask_volume=v, ask_order_count=int(v), total_volume=v)
        for p, v in zip(ask_prices, ask_volumes)
    ]
    bid_total = sum(bid_volumes)
    ask_total = sum(ask_volumes)
    total = bid_total + ask_total
    imb = (bid_total - ask_total) / total if total > 0 else 0.0
    spread = ask_prices[0] - bid_prices[0] if bid_prices and ask_prices else 0.0002
    if classification is None:
        if imb > 0.30:
            classification = "STRONG_BID"
        elif imb > 0.10:
            classification = "BID_HEAVY"
        elif imb < -0.30:
            classification = "STRONG_ASK"
        elif imb < -0.10:
            classification = "ASK_HEAVY"
        else:
            classification = "NEUTRAL"
    return DOMSnapshot(
        bid_levels=bid_levels, ask_levels=ask_levels,
        bid_total_volume=bid_total, ask_total_volume=ask_total,
        spread=spread, imbalance_ratio=round(imb, 4),
        imbalance_classification=classification,
        timestamp=1_000_000.0, is_valid=True,
    )


# ─────────────────────────────────────────────
# ORDER FLOW CONFIG
# ─────────────────────────────────────────────

class TestOrderFlowConfig:
    def test_defaults(self):
        c = OrderFlowConfig()
        assert c.dom_depth_levels == 15
        assert c.absorption_volume_ratio == 2.0
        assert c.imbalance_threshold == 0.20
        assert c.exhaustion_buy_threshold == 0.75
        assert c.stop_run_atr_multiplier == 1.2
        assert c.delta_macd_fast == 5
        assert c.tick_history_size == 10000


# ─────────────────────────────────────────────
# DOM CLUSTER ENGINE
# ─────────────────────────────────────────────

class TestDOMClusterEngine:
    def test_cluster_empty(self):
        snap = make_snapshot(bid_prices=(), ask_prices=())
        clusters = DOMClusterEngine.cluster(snap, 2.0, 0.0001)
        assert clusters == []

    def test_cluster_single_group(self):
        snap = make_snapshot(
            bid_prices=(1.1000, 1.0998),
            ask_prices=(1.1002, 1.1005),
        )
        clusters = DOMClusterEngine.cluster(snap, 5.0, 0.0001)
        assert len(clusters) == 1

    def test_cluster_multiple_groups(self):
        snap = make_snapshot(
            bid_prices=(1.1000, 1.0999, 1.0900),
            ask_prices=(1.1001, 1.1003, 1.0901),
        )
        clusters = DOMClusterEngine.cluster(snap, 2.0, 0.0001)
        assert len(clusters) == 2

    def test_build_cluster_properties(self):
        items = [
            (1.1000, "bid", 10.0, 3),
            (1.1001, "ask", 8.0, 2),
        ]
        snap = make_snapshot()
        cluster = DOMClusterEngine._build_cluster(items, snap)
        assert cluster.price_high >= cluster.price_low
        assert cluster.total_volume == 18.0
        assert cluster.order_count == 5
        assert isinstance(cluster.poc, float)
        assert cluster.bid_volume == 10.0
        assert cluster.ask_volume == 8.0

    def test_cluster_handles_single_level(self):
        snap = make_snapshot(
            bid_prices=(1.1000,),
            ask_prices=(1.1002,),
        )
        clusters = DOMClusterEngine.cluster(snap, 2.0, 0.0001)
        assert len(clusters) == 1


# ─────────────────────────────────────────────
# ORDER FLOW ENGINE — TICK ANALYSIS
# ─────────────────────────────────────────────

class TestOrderFlowEngineTickAnalysis:
    @pytest.fixture
    def engine(self, cfg):
        return OrderFlowEngine(config=cfg)

    def test_init_state(self, engine):
        assert engine._tick_history == {}
        assert engine._dom_cache == {}
        assert engine._cumulative_delta == {}
        assert engine._mt5_client is None

    def test_set_mt5_client(self, engine):
        client = MagicMock()
        engine.set_mt5_client(client)
        assert engine._mt5_client is client

    def test_analyze_ticks_none_df(self, engine):
        result = engine._analyze_ticks("EURUSD", None)
        assert result["delta"] == 0
        assert result["buy_pressure"] == 0.5
        assert result["sell_pressure"] == 0.5
        assert result["delta_divergence"] is False

    def test_analyze_ticks_insufficient_data(self, engine):
        df = pd.DataFrame({"open": [100.0], "close": [100.0], "high": [100.1], "low": [99.9], "volume": [100]})
        result = engine._analyze_ticks("EURUSD", df)
        assert result["delta"] == 0

    def test_analyze_ticks_bullish(self, engine, df_bullish_volume):
        result = engine._analyze_ticks("EURUSD", df_bullish_volume)
        assert result["buy_pressure"] > 0.5
        assert result["sell_pressure"] < 0.5
        assert result["delta"] > 0

    def test_analyze_ticks_bearish(self, engine):
        n = 30
        opens = [100.0 - i * 0.2 for i in range(n)]
        closes = [o - 0.3 for o in opens]
        highs = [o + 0.1 for o in opens]
        lows = [c - 0.1 for c in closes]
        volumes = [200 + i * 5 for i in range(n)]
        df = pd.DataFrame({"open": opens, "close": closes, "high": highs, "low": lows, "volume": volumes})
        result = engine._analyze_ticks("EURUSD", df)
        assert result["delta"] < 0

    def test_analyze_ticks_accumulates_delta(self, engine, df_bullish_volume):
        r1 = engine._analyze_ticks("EURUSD", df_bullish_volume)
        r2 = engine._analyze_ticks("EURUSD", df_bullish_volume)
        assert r2["cumulative_delta"] > r1["cumulative_delta"]

    def test_analyze_ticks_delta_divergence_bullish_price_falls(self, engine):
        n = 30
        opens = [100.0] * n
        closes = [99.0 + i * 0.02 for i in range(n)]
        highs = [max(o, c) + 0.1 for o, c in zip(opens, closes)]
        lows = [min(o, c) - 0.1 for o, c in zip(opens, closes)]
        volumes = [100] * n
        for i in range(10, n):
            volumes[i] = 500
        df = pd.DataFrame({"open": opens, "close": closes, "high": highs, "low": lows, "volume": volumes})
        result = engine._analyze_ticks("EURUSD", df)
        assert "delta_divergence" in result

    def test_analyze_ticks_ad_ratio(self, engine, df_bullish_volume):
        result = engine._analyze_ticks("EURUSD", df_bullish_volume)
        assert result["ad_ratio"] > 1.0

    def test_analyze_ticks_exhaustion(self, engine):
        n = 30
        opens = [100.0] * n
        closes = [100.5 + i * 0.01 for i in range(n)]
        highs = [c + 0.1 for c in closes]
        lows = [o - 0.1 for o in opens]
        volumes = [100 + i * 80 for i in range(n)]
        df = pd.DataFrame({"open": opens, "close": closes, "high": highs, "low": lows, "volume": volumes})
        result = engine._analyze_ticks("EURUSD", df)
        assert "exhaustion_active" in result


# ─────────────────────────────────────────────
# ORDER FLOW ENGINE — DOM ANALYSIS
# ─────────────────────────────────────────────

class TestOrderFlowEngineDOM:
    @pytest.fixture
    def engine(self, cfg):
        return OrderFlowEngine(config=cfg)

    def test_classify_imbalance(self, engine):
        assert engine._classify_imbalance(0.35) == "STRONG_BID"
        assert engine._classify_imbalance(0.15) == "BID_HEAVY"
        assert engine._classify_imbalance(-0.35) == "STRONG_ASK"
        assert engine._classify_imbalance(-0.15) == "ASK_HEAVY"
        assert engine._classify_imbalance(0.05) == "NEUTRAL"

    def test_analyze_dom_no_dom(self, engine):
        result = engine._analyze_dom(None, 0.0, 0.0, "EURUSD")
        assert result["imbalance_ratio"] == 0.0
        assert result["absorption_active"] is False
        assert result["iceberg_detected"] is False
        assert result["stop_run_detected"] is False

    def test_analyze_dom_invalid_snapshot(self, engine):
        snap = make_snapshot()
        snap.is_valid = False
        result = engine._analyze_dom(snap, 0.0, 0.0, "EURUSD")
        assert result["imbalance_ratio"] == 0.0

    def test_analyze_dom_not_expert(self, engine):
        engine.config.expert_mode = False
        snap = make_snapshot()
        result = engine._analyze_dom(snap, 0.0, 0.0, "EURUSD")
        assert result["imbalance_ratio"] == 0.0

    def test_analyze_dom_imbalance(self, engine):
        snap = make_snapshot(
            bid_volumes=(50.0, 30.0, 10.0, 5.0, 3.0),
            ask_volumes=(5.0, 4.0, 3.0, 2.0, 1.0),
        )
        result = engine._analyze_dom(snap, 0.0, 0.0001, "EURUSD")
        assert result["imbalance_ratio"] > 0.3

    def test_analyze_dom_absorption(self, engine):
        snap = make_snapshot(
            bid_prices=(1.1000, 1.0995, 1.0990, 1.0985, 1.0980),
            ask_prices=(1.1005, 1.1010, 1.1015, 1.1020, 1.1025),
            bid_volumes=(100.0, 80.0, 60.0, 40.0, 20.0),
            ask_volumes=(100.0, 80.0, 60.0, 40.0, 20.0),
        )
        result = engine._analyze_dom(snap, 0.0, 0.0001, "EURUSD")
        assert isinstance(result["absorption_active"], bool)

    def test_analyze_dom_stop_run_miss(self, engine):
        snap1 = make_snapshot()
        snap2 = make_snapshot()
        engine._analyze_dom(snap1, 0.002, 0.0001, "EURUSD")
        result = engine._analyze_dom(snap2, 0.002, 0.0001, "EURUSD")
        assert result["stop_run_detected"] is False

    def test_analyze_dom_iceberg(self, engine):
        snap = make_snapshot(
            bid_prices=(1.1000, 1.0995, 1.0990, 1.0985, 1.0980),
            ask_prices=(1.1005, 1.1010, 1.1015, 1.1020, 1.1025),
            bid_volumes=(10.0, 10.0, 500.0, 10.0, 10.0),
            ask_volumes=(10.0, 10.0, 10.0, 10.0, 10.0),
        )
        result = engine._analyze_dom(snap, 0.0, 0.0001, "EURUSD")
        assert isinstance(result["iceberg_detected"], bool)


# ─────────────────────────────────────────────
# ORDER FLOW ENGINE — MERGE & SIGNAL
# ─────────────────────────────────────────────

class TestOrderFlowEngineSignal:
    @pytest.fixture
    def engine(self, cfg):
        return OrderFlowEngine(config=cfg)

    def test_merge_returns_signal(self, engine, df_bullish_volume):
        tick = engine._analyze_ticks("EURUSD", df_bullish_volume)
        dom = {
            "imbalance_ratio": 0.25,
            "imbalance_classification": "BID_HEAVY",
            "absorption_active": False,
            "absorption_levels": [],
            "absorption_clusters": [],
            "iceberg_detected": False,
            "iceberg_levels": [],
            "iceberg_chunks": 0,
            "stop_run_detected": False,
            "stop_run_direction": "",
            "stop_run_level": 0.0,
        }
        signal = engine._merge("EURUSD", tick, dom, df_bullish_volume, 0.0, 0.0001)
        assert isinstance(signal, OrderFlowSignal)
        assert signal.delta != 0
        assert signal.buy_pressure > 0.5
        assert signal.ad_ratio > 0

    def test_merge_sets_delta_macd(self, engine, df_bullish_volume):
        tick = engine._analyze_ticks("EURUSD", df_bullish_volume)
        dom = {
            "imbalance_ratio": 0.0, "imbalance_classification": "NEUTRAL",
            "absorption_active": False, "absorption_levels": [],
            "absorption_clusters": [], "iceberg_detected": False,
            "iceberg_levels": [], "iceberg_chunks": 0,
            "stop_run_detected": False, "stop_run_direction": "", "stop_run_level": 0.0,
        }
        for _ in range(5):
            engine._analyze_ticks("EURUSD", df_bullish_volume)
        signal = engine._merge("EURUSD", tick, dom, df_bullish_volume, 0.0, 0.0001)
        assert isinstance(signal.delta_macd, DeltaMACD) or signal.delta_macd is None

    def test_merge_with_absorption_note(self, engine, df_bullish_volume):
        tick = engine._analyze_ticks("EURUSD", df_bullish_volume)
        dom = {
            "imbalance_ratio": 0.0, "imbalance_classification": "NEUTRAL",
            "absorption_active": True,
            "absorption_levels": [1.1005, 1.1010],
            "absorption_clusters": [MagicMock(), MagicMock()],
            "iceberg_detected": False, "iceberg_levels": [], "iceberg_chunks": 0,
            "stop_run_detected": False, "stop_run_direction": "", "stop_run_level": 0.0,
        }
        signal = engine._merge("EURUSD", tick, dom, df_bullish_volume, 0.0, 0.0001)
        assert any("Absorption" in n for n in signal.notes)

    def test_merge_with_stop_run_note(self, engine, df_bullish_volume):
        tick = engine._analyze_ticks("EURUSD", df_bullish_volume)
        dom = {
            "imbalance_ratio": 0.0, "imbalance_classification": "NEUTRAL",
            "absorption_active": False, "absorption_levels": [],
            "absorption_clusters": [], "iceberg_detected": False,
            "iceberg_levels": [], "iceberg_chunks": 0,
            "stop_run_detected": True,
            "stop_run_direction": "SELL",
            "stop_run_level": 1.0995,
        }
        signal = engine._merge("EURUSD", tick, dom, df_bullish_volume, 0.0, 0.0001)
        assert any("Stop-run" in n for n in signal.notes)

    def test_get_signal_contribution_buy(self, engine):
        signal = OrderFlowSignal(
            delta=0.25, delta_ma=0.15, delta_divergence=False,
            delta_macd=None, imbalance_ratio=0.35,
            imbalance_classification="STRONG_BID",
            absorption_active=False, absorption_levels=[], absorption_clusters=[],
            iceberg_detected=False, iceberg_levels=[], iceberg_chunks=0,
            stop_run_detected=False, stop_run_direction="", stop_run_level=0.0,
            exhaustion_active=False, exhaustion_side="",
            buy_pressure=0.65, sell_pressure=0.35,
            cumulative_delta=2.5, cumulative_delta_ma=1.0,
            ad_ratio=1.5, notes=[],
        )
        score, reason = engine.get_signal_contribution(signal, "BUY")
        assert score > 0
        assert "Delta" in reason

    def test_get_signal_contribution_sell(self, engine):
        signal = OrderFlowSignal(
            delta=-0.25, delta_ma=-0.15, delta_divergence=False,
            delta_macd=None, imbalance_ratio=-0.35,
            imbalance_classification="STRONG_ASK",
            absorption_active=False, absorption_levels=[], absorption_clusters=[],
            iceberg_detected=False, iceberg_levels=[], iceberg_chunks=0,
            stop_run_detected=False, stop_run_direction="", stop_run_level=0.0,
            exhaustion_active=False, exhaustion_side="",
            buy_pressure=0.35, sell_pressure=0.65,
            cumulative_delta=-2.5, cumulative_delta_ma=-1.0,
            ad_ratio=0.5, notes=[],
        )
        score, reason = engine.get_signal_contribution(signal, "SELL")
        assert score > 0
        assert "Delta" in reason

    def test_signal_contribution_absorption_bonus(self, engine):
        signal = OrderFlowSignal(
            delta=0.1, delta_ma=0.1, delta_divergence=False,
            delta_macd=None, imbalance_ratio=0.1,
            imbalance_classification="NEUTRAL",
            absorption_active=True, absorption_levels=[1.10], absorption_clusters=[],
            iceberg_detected=False, iceberg_levels=[], iceberg_chunks=0,
            stop_run_detected=False, stop_run_direction="", stop_run_level=0.0,
            exhaustion_active=False, exhaustion_side="",
            buy_pressure=0.5, sell_pressure=0.5,
            cumulative_delta=0.0, cumulative_delta_ma=0.0,
            ad_ratio=1.0, notes=[],
        )
        score, reason = engine.get_signal_contribution(signal, "BUY")
        assert "Absorción" in reason

    def test_signal_contribution_iceberg_bonus(self, engine):
        signal = OrderFlowSignal(
            delta=0.1, delta_ma=0.1, delta_divergence=False,
            delta_macd=None, imbalance_ratio=0.1,
            imbalance_classification="NEUTRAL",
            absorption_active=False, absorption_levels=[], absorption_clusters=[],
            iceberg_detected=True, iceberg_levels=[1.1005], iceberg_chunks=3,
            stop_run_detected=False, stop_run_direction="", stop_run_level=0.0,
            exhaustion_active=False, exhaustion_side="",
            buy_pressure=0.5, sell_pressure=0.5,
            cumulative_delta=0.0, cumulative_delta_ma=0.0,
            ad_ratio=1.0, notes=[],
        )
        score, reason = engine.get_signal_contribution(signal, "BUY")
        assert "Iceberg" in reason

    def test_get_detailed_report_structure(self, engine):
        signal = OrderFlowSignal(
            delta=0.1, delta_ma=0.08, delta_divergence=False,
            delta_macd=None, imbalance_ratio=0.0,
            imbalance_classification="NEUTRAL",
            absorption_active=False, absorption_levels=[], absorption_clusters=[],
            iceberg_detected=False, iceberg_levels=[], iceberg_chunks=0,
            stop_run_detected=False, stop_run_direction="", stop_run_level=0.0,
            exhaustion_active=True, exhaustion_side="SELL_CLIMAX",
            buy_pressure=0.8, sell_pressure=0.2,
            cumulative_delta=1.0, cumulative_delta_ma=0.5,
            ad_ratio=1.8, notes=["test note"],
        )
        report = engine.get_detailed_report(signal)
        assert report["delta"] == 0.1
        assert report["exhaustion"]["active"] is True
        assert report["exhaustion"]["side"] == "SELL_CLIMAX"
        assert report["pressure"]["buy"] == 0.8
        assert report["notes"] == ["test note"]
        assert "absorption" in report
        assert "iceberg" in report
        assert "stop_run" in report
        assert "divergence" in report
        assert "cumulative_delta" in report

    def test_delta_macd_divergence_bullish_contribution(self, engine):
        signal = OrderFlowSignal(
            delta=0.1, delta_ma=0.1, delta_divergence=False,
            delta_macd=DeltaMACD(
                macd_line=0.001, signal_line=0.0005, histogram=0.0005,
                divergence_bullish=True, divergence_bearish=False,
            ),
            imbalance_ratio=0.1, imbalance_classification="NEUTRAL",
            absorption_active=False, absorption_levels=[], absorption_clusters=[],
            iceberg_detected=False, iceberg_levels=[], iceberg_chunks=0,
            stop_run_detected=False, stop_run_direction="", stop_run_level=0.0,
            exhaustion_active=False, exhaustion_side="",
            buy_pressure=0.5, sell_pressure=0.5,
            cumulative_delta=0.0, cumulative_delta_ma=0.0,
            ad_ratio=1.0, notes=[],
        )
        score, reason = engine.get_signal_contribution(signal, "BUY")
        assert "Delta-MACD" in reason

    def test_analyze_returns_signal(self, engine, df_bullish_volume):
        with patch.object(engine, "_fetch_dom", return_value=None):
            signal = engine.analyze("EURUSD", df_bullish_volume, atr_val=0.002, pip=0.0001)
        assert isinstance(signal, OrderFlowSignal)
        assert signal.delta != 0
        assert signal.cumulative_delta != 0

    def test_analyze_with_dom(self, engine, df_bullish_volume):
        snap = make_snapshot(
            bid_volumes=(50.0, 30.0, 10.0, 5.0, 3.0),
            ask_volumes=(5.0, 4.0, 3.0, 2.0, 1.0),
        )
        with patch.object(engine, "_fetch_dom", return_value=snap):
            signal = engine.analyze("EURUSD", df_bullish_volume, atr_val=0.002, pip=0.0001)
        assert isinstance(signal, OrderFlowSignal)

    def test_analyze_non_expert_mode_skip_dom(self, cfg, df_bullish_volume):
        cfg.expert_mode = False
        engine = OrderFlowEngine(config=cfg)
        signal = engine.analyze("EURUSD", df_bullish_volume, atr_val=0.002, pip=0.0001)
        assert isinstance(signal, OrderFlowSignal)

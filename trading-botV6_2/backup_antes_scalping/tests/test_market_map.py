"""Tests for MarketMap orchestrator."""
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

from src.core.market_map import MarketMap, MarketMapResult
from src.core.liquidity_mapper import MarketMap as LiquidityMarketMap
from src.core.micro_phase import MicroPhase, PhaseResult
from src.core.breakout_retest import BreakoutRetestSignal
from src.core.route_planner import Route
from src.core.entry_confirmer import EntryConfirmation
from src.core.candle_confirmer import CandleConfirmerResult, ConfirmerStatus
from src.core.dynamic_tp import TPTier
from src.core.md_concepts import MDConcept, MDDetection
from src.core.md_asymmetry_filter import AsymmetryResult
from src.core.md_binary_risk_filter import BinaryRiskResult


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────


def _liquidity_map(direction="BUY"):
    m = MagicMock(spec=LiquidityMarketMap)
    m.zones = {}
    m.dominant_direction = direction
    return m


def _phase(allows_entry=True, confidence=0.7, phase_name="IMPULSE_STARTING"):
    p = MagicMock()
    p.allows_entry = allows_entry
    p.phase = MagicMock(value=phase_name)
    p.confidence = confidence
    p.direction = "NEUTRAL"
    return p


def _route(direction="BUY", is_valid=True):
    return Route(
        symbol="EURUSD", direction=direction, entry_price=100.0,
        target_price=101.0, target_type="LIQUIDITY",
        nodes=[MagicMock()] if is_valid else [],
        has_valid_sl_zone=is_valid,
        confidence=0.5 if is_valid else 0.0,
    )


def _md_detection(concept=MDConcept.POD, confidence=0.6):
    return MDDetection(concept=concept, direction="BUY", confidence=confidence)


def _no_md():
    m = MagicMock()
    m.found = False
    return m


def _candle_result(confirmed=False, tf="1h"):
    cr = MagicMock()
    cr.is_confirmed = confirmed
    cr.status = ConfirmerStatus.CONFIRMED if confirmed else ConfirmerStatus.NOT_READY
    return cr


def _candle_confirmation(confirmed_1h=False, confirmed_4h=False):
    return {
        "1h": _candle_result(confirmed_1h, "1h"),
        "4h": _candle_result(confirmed_4h, "4h"),
    }


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────


@pytest.fixture
def df():
    return pd.DataFrame({
        "open": [100.0] * 15,
        "close": [100.0] * 15,
        "high": [100.1] * 15,
        "low": [99.9] * 15,
    })


@pytest.fixture
def dfs(df):
    return {"M1": df, "1h": df, "4h": df}


@pytest.fixture
def detectors():
    dets = {
        "zone_tracker": MagicMock(),
        "liquidity_mapper": MagicMock(),
        "phase_detector": MagicMock(),
        "breakout_detector": MagicMock(),
        "route_planner": MagicMock(),
        "entry_confirmer": MagicMock(),
        "candle_confirmer": MagicMock(),
        "tp_manager": MagicMock(),
        "pod_detector": MagicMock(),
        "interval_detector": MagicMock(),
        "limit_price_detector": MagicMock(),
        "price_capture_detector": MagicMock(),
        "ote_detector": MagicMock(),
        "sequence_123_detector": MagicMock(),
        "three_candle_detector": MagicMock(),
        "asymmetry_filter": MagicMock(),
        "binary_risk_filter": MagicMock(),
        "concept_tracker": None,
    }
    for key in ("pod_detector", "interval_detector", "limit_price_detector",
                 "price_capture_detector", "ote_detector",
                 "sequence_123_detector", "three_candle_detector"):
        dets[key].detect.return_value = _no_md()
    return dets


@pytest.fixture
def mm(detectors):
    return MarketMap("EURUSD", **detectors)


# ─────────────────────────────────────────────
# MarketMapResult dataclass
# ─────────────────────────────────────────────


class TestMarketMapResult:
    def test_all_fields(self):
        r = MarketMapResult(
            decision="TRADE", direction="BUY", confidence=0.85,
            market_map=MagicMock(), phase=MagicMock(),
            breakout_signals=[MagicMock()],
            route=MagicMock(), entry_confirmation=MagicMock(),
            candle_confirmation={"1h": MagicMock()},
            tp_tiers=[MagicMock()], active_tp=1.1050,
            suggested_entry=1.1000, stop_loss=1.0980,
            score_bonus=15.0, final_score=85.0,
            md_detections=[MagicMock()], notes=["test"],
        )
        assert r.decision == "TRADE"
        assert r.direction == "BUY"
        assert r.confidence == 0.85
        assert r.score_bonus == 15.0
        assert r.final_score == 85.0
        assert r.active_tp == 1.1050
        assert r.suggested_entry == 1.1000
        assert r.stop_loss == 1.0980
        assert len(r.md_detections) == 1
        assert r.notes == ["test"]

    def test_defaults(self):
        r = MarketMapResult(decision="NO_TRADE", direction="NEUTRAL", confidence=0.0)
        assert r.market_map is None
        assert r.phase is None
        assert r.breakout_signals == []
        assert r.route is None
        assert r.entry_confirmation is None
        assert r.candle_confirmation == {}
        assert r.tp_tiers == []
        assert r.active_tp is None
        assert r.suggested_entry is None
        assert r.stop_loss is None
        assert r.score_bonus == 0.0
        assert r.final_score == 0.0
        assert r.md_detections == []
        assert r.notes == []


# ─────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────


class TestMarketMapInit:
    def test_default_detectors_created(self):
        mm = MarketMap("EURUSD")
        assert mm._symbol == "EURUSD"
        assert mm._zone_tracker is not None
        assert mm._liquidity_mapper is not None
        assert mm._phase_detector is not None
        assert mm._breakout is not None
        assert mm._route_planner is not None
        assert mm._entry_confirmer is not None
        assert mm._candle_confirmer is not None
        assert mm._tp_manager is not None
        assert mm._pod_detector is not None
        assert mm._interval_detector is not None
        assert mm._limit_price_detector is not None
        assert mm._price_capture_detector is not None
        assert mm._ote_detector is not None
        assert mm._sequence_123_detector is not None
        assert mm._three_candle_detector is not None
        assert mm._asymmetry_filter is not None
        assert mm._binary_risk_filter is not None
        assert mm._concept_tracker is None

    def test_custom_detectors_injected(self, detectors):
        mm = MarketMap("EURUSD", **detectors)
        assert mm._zone_tracker is detectors["zone_tracker"]
        assert mm._liquidity_mapper is detectors["liquidity_mapper"]
        assert mm._phase_detector is detectors["phase_detector"]
        assert mm._breakout is detectors["breakout_detector"]
        assert mm._route_planner is detectors["route_planner"]
        assert mm._entry_confirmer is detectors["entry_confirmer"]
        assert mm._candle_confirmer is detectors["candle_confirmer"]
        assert mm._tp_manager is detectors["tp_manager"]
        assert mm._pod_detector is detectors["pod_detector"]
        assert mm._interval_detector is detectors["interval_detector"]
        assert mm._limit_price_detector is detectors["limit_price_detector"]
        assert mm._price_capture_detector is detectors["price_capture_detector"]
        assert mm._ote_detector is detectors["ote_detector"]
        assert mm._sequence_123_detector is detectors["sequence_123_detector"]
        assert mm._three_candle_detector is detectors["three_candle_detector"]
        assert mm._asymmetry_filter is detectors["asymmetry_filter"]
        assert mm._binary_risk_filter is detectors["binary_risk_filter"]

    def test_concept_tracker_passed(self, detectors):
        tracker = MagicMock()
        detectors["concept_tracker"] = tracker
        mm = MarketMap("EURUSD", **detectors)
        assert mm._concept_tracker is tracker


# ─────────────────────────────────────────────
# evaluate() — TRADE decision
# ─────────────────────────────────────────────


class TestMarketMapEvaluate:
    def _setup_full_trade_path(self, detectors, direction="BUY"):
        detectors["liquidity_mapper"].build.return_value = _liquidity_map(direction)
        detectors["phase_detector"].detect.return_value = _phase(allows_entry=True)
        detectors["breakout_detector"].check.return_value = []
        detectors["route_planner"].plan.return_value = _route(direction, is_valid=True)
        detectors["entry_confirmer"].confirm.return_value = EntryConfirmation(
            valid=True, confidence=0.8, reason="ok",
        )
        detectors["candle_confirmer"].check.return_value = _candle_confirmation()
        detectors["tp_manager"].build_tiers.return_value = [
            TPTier(level=1, price=1.1010, label="TP1"),
        ]
        detectors["tp_manager"].get_active_tp.return_value = 1.1010
        detectors["asymmetry_filter"].evaluate.return_value = AsymmetryResult(
            passed=True, asymmetry_factor=1.2,
        )
        detectors["binary_risk_filter"].evaluate.return_value = BinaryRiskResult(
            passed=True, passed_checks=["structural_sl"],
        )
        for key in ("pod_detector", "interval_detector", "limit_price_detector",
                     "price_capture_detector", "ote_detector",
                     "sequence_123_detector", "three_candle_detector"):
            detectors[key].detect.return_value = _no_md()

    def test_trade_decision(self, mm, detectors, dfs):
        self._setup_full_trade_path(detectors)
        with patch.object(mm, "_pip_size", return_value=0.0001), \
             patch.object(mm, "_atr", return_value=0.001):
            result = mm.evaluate(dfs)
        assert result.decision == "TRADE"
        assert result.direction == "BUY"
        assert result.confidence > 0
        assert result.final_score > 0

    def test_trade_decision_sell(self, mm, detectors, dfs):
        self._setup_full_trade_path(detectors, direction="SELL")
        with patch.object(mm, "_pip_size", return_value=0.0001), \
             patch.object(mm, "_atr", return_value=0.001):
            result = mm.evaluate(dfs)
        assert result.decision == "TRADE"
        assert result.direction == "SELL"

    def test_trade_decision_includes_md_detections(self, mm, detectors, dfs):
        self._setup_full_trade_path(detectors)
        detection = _md_detection()
        detectors["pod_detector"].detect.return_value = MagicMock(
            found=True, to_detection=lambda: detection,
        )
        with patch.object(mm, "_pip_size", return_value=0.0001), \
             patch.object(mm, "_atr", return_value=0.001):
            result = mm.evaluate(dfs)
        assert result.decision == "TRADE"
        assert any(d.concept == MDConcept.POD for d in result.md_detections)

    def test_trade_decision_with_candle_confirmation(self, mm, detectors, df):
        self._setup_full_trade_path(detectors)
        h1_result = CandleConfirmerResult(
            status=ConfirmerStatus.CONFIRMED, direction="BUY", timeframe="1h",
        )
        detectors["candle_confirmer"].check.return_value = {"1h": h1_result}
        dfs = {"M1": df, "1h": df}
        with patch.object(mm, "_pip_size", return_value=0.0001), \
             patch.object(mm, "_atr", return_value=0.001):
            result = mm.evaluate(dfs)
        assert result.decision == "TRADE"
        assert result.candle_confirmation.get("1h").is_confirmed

# ─────────────────────────────────────────────
# evaluate() — NO_TRADE when phase blocks entry
# ─────────────────────────────────────────────


    def test_no_trade_phase_not_allowed(self, mm, detectors, dfs):
        detectors["liquidity_mapper"].build.return_value = _liquidity_map("BUY")
        detectors["phase_detector"].detect.return_value = _phase(
            allows_entry=False, phase_name="INDECISION",
        )
        detectors["breakout_detector"].check.return_value = []
        with patch.object(mm, "_pip_size", return_value=0.0001), \
             patch.object(mm, "_atr", return_value=0.001):
            result = mm.evaluate(dfs)
        assert result.decision == "NO_TRADE"
        assert result.confidence == 0.0
        assert any("no permite entrada" in n.lower() for n in result.notes)

# ─────────────────────────────────────────────
# evaluate() — NO_TRADE when binary risk fails
# ─────────────────────────────────────────────


    def test_no_trade_binary_risk_fails(self, mm, detectors, dfs):
        self._setup_full_trade_path(detectors)
        detectors["binary_risk_filter"].evaluate.return_value = BinaryRiskResult(
            passed=False, failed_checks=["min_rr"],
        )
        with patch.object(mm, "_pip_size", return_value=0.0001), \
             patch.object(mm, "_atr", return_value=0.001):
            result = mm.evaluate(dfs)
        assert result.decision == "NO_TRADE"
        assert result.final_score == 0.0
        assert any("BinaryRisk FAIL" in n for n in result.notes)

# ─────────────────────────────────────────────
# evaluate() — WAIT when entry confirmer fails
# ─────────────────────────────────────────────


    def test_wait_entry_confirmer_fails(self, mm, detectors, dfs):
        self._setup_full_trade_path(detectors)
        detectors["entry_confirmer"].confirm.return_value = EntryConfirmation(
            valid=False, confidence=0.0, reason="no_tick_data",
        )
        with patch.object(mm, "_pip_size", return_value=0.0001), \
             patch.object(mm, "_atr", return_value=0.001):
            result = mm.evaluate(dfs)
        assert result.decision == "WAIT"
        assert any("no confirmada" in n.lower() for n in result.notes)

# ─────────────────────────────────────────────
# evaluate() — Breakout active path (early TRADE)
# ─────────────────────────────────────────────


    def test_breakout_active_trade(self, mm, detectors, dfs):
        detectors["liquidity_mapper"].build.return_value = _liquidity_map("BUY")
        detectors["phase_detector"].detect.return_value = _phase(allows_entry=False)
        detectors["breakout_detector"].check.return_value = [
            BreakoutRetestSignal(
                active=True, direction="BUY", breakout_level=100.5,
                retest_level=100.0, zone_touches=4, confidence=0.85,
                score_bonus=30.0,
            ),
        ]
        detectors["route_planner"].plan.return_value = _route("BUY", is_valid=True)
        detectors["tp_manager"].build_tiers.return_value = [
            TPTier(level=1, price=1.1010, label="TP1"),
        ]
        detectors["tp_manager"].get_active_tp.return_value = 1.1010
        with patch.object(mm, "_pip_size", return_value=0.0001), \
             patch.object(mm, "_atr", return_value=0.001):
            result = mm.evaluate(dfs)
        assert result.decision == "TRADE"
        assert result.direction == "BUY"
        assert len(result.breakout_signals) == 1
        assert result.breakout_signals[0].active
        assert result.score_bonus > 0

    def test_breakout_inactive_falls_through(self, mm, detectors, dfs):
        self._setup_full_trade_path(detectors)
        detectors["breakout_detector"].check.return_value = [
            BreakoutRetestSignal(
                active=False, direction="BUY", breakout_level=100.5,
                retest_level=100.0, zone_touches=3, confidence=0.5,
                score_bonus=0.0,
            ),
        ]
        with patch.object(mm, "_pip_size", return_value=0.0001), \
             patch.object(mm, "_atr", return_value=0.001):
            result = mm.evaluate(dfs)
        assert result.decision == "TRADE"
        assert len(result.breakout_signals) == 1

# ─────────────────────────────────────────────
# evaluate() — Invalid route
# ─────────────────────────────────────────────


    def test_invalid_route_no_trade(self, mm, detectors, dfs):
        self._setup_full_trade_path(detectors)
        detectors["route_planner"].plan.return_value = _route("BUY", is_valid=False)
        with patch.object(mm, "_pip_size", return_value=0.0001), \
             patch.object(mm, "_atr", return_value=0.001):
            result = mm.evaluate(dfs)
        assert result.decision == "NO_TRADE"
        assert any("Ruta inv" in n for n in result.notes)

    def test_invalid_route_with_breakout_waits(self, mm, detectors, dfs):
        detectors["liquidity_mapper"].build.return_value = _liquidity_map("BUY")
        detectors["phase_detector"].detect.return_value = _phase(allows_entry=True)
        detectors["breakout_detector"].check.return_value = [
            BreakoutRetestSignal(
                active=False, direction="BUY", breakout_level=100.5,
                retest_level=100.0, zone_touches=3, confidence=0.5,
                score_bonus=0.0,
            ),
        ]
        detectors["route_planner"].plan.return_value = _route("BUY", is_valid=False)
        with patch.object(mm, "_pip_size", return_value=0.0001), \
             patch.object(mm, "_atr", return_value=0.001):
            result = mm.evaluate(dfs)
        assert result.decision == "WAIT"

# ─────────────────────────────────────────────
# evaluate() — Error isolation (detector crash)
# ─────────────────────────────────────────────


    def test_error_isolation_md_detector_crash(self, mm, detectors, dfs):
        self._setup_full_trade_path(detectors)
        detectors["pod_detector"].detect.side_effect = ValueError("POD crash")
        with patch.object(mm, "_pip_size", return_value=0.0001), \
             patch.object(mm, "_atr", return_value=0.001):
            result = mm.evaluate(dfs)
        assert result.decision == "TRADE"
        assert not any(d.concept == MDConcept.POD for d in result.md_detections)

    def test_error_isolation_breakout_crash(self, mm, detectors, dfs):
        detectors["liquidity_mapper"].build.return_value = _liquidity_map("BUY")
        detectors["phase_detector"].detect.return_value = _phase(allows_entry=True)
        detectors["breakout_detector"].check.side_effect = RuntimeError("breakout fail")
        detectors["route_planner"].plan.return_value = _route("BUY", is_valid=True)
        detectors["entry_confirmer"].confirm.return_value = EntryConfirmation(
            valid=True, confidence=0.8, reason="ok",
        )
        detectors["candle_confirmer"].check.return_value = _candle_confirmation()
        detectors["tp_manager"].build_tiers.return_value = []
        detectors["tp_manager"].get_active_tp.return_value = None
        detectors["asymmetry_filter"].evaluate.return_value = AsymmetryResult(
            passed=True, asymmetry_factor=1.2,
        )
        detectors["binary_risk_filter"].evaluate.return_value = BinaryRiskResult(
            passed=True, passed_checks=["structural_sl"],
        )
        with patch.object(mm, "_pip_size", return_value=0.0001), \
             patch.object(mm, "_atr", return_value=0.001):
            result = mm.evaluate(dfs)
        assert result.decision == "TRADE"
        assert result.breakout_signals == []

    def test_error_isolation_asymmetry_crash(self, mm, detectors, dfs):
        self._setup_full_trade_path(detectors)
        detectors["asymmetry_filter"].evaluate.side_effect = Exception("asym crash")
        with patch.object(mm, "_pip_size", return_value=0.0001), \
             patch.object(mm, "_atr", return_value=0.001):
            result = mm.evaluate(dfs)
        assert result.decision == "TRADE"
        assert any("Asymmetry FAIL" in n for n in result.notes)

    def test_error_isolation_entry_confirmer_crash(self, mm, detectors, dfs):
        self._setup_full_trade_path(detectors)
        detectors["entry_confirmer"].confirm.side_effect = Exception("entry crash")
        with patch.object(mm, "_pip_size", return_value=0.0001), \
             patch.object(mm, "_atr", return_value=0.001):
            result = mm.evaluate(dfs)
        assert result.decision == "WAIT"
        assert any("entry_confirmer_crashed" in n for n in result.notes)

# ─────────────────────────────────────────────
# _md_bonus
# ─────────────────────────────────────────────


class TestMDBonus:
    def test_without_concept_tracker(self, detectors):
        mm = MarketMap("EURUSD", **detectors)
        det = _md_detection(confidence=0.6)
        bonus = mm._md_bonus(det)
        assert bonus == 0.6 * 1.0 * 10

    def test_with_concept_tracker(self, detectors):
        tracker = MagicMock()
        tracker.get_weight.return_value = 1.5
        detectors["concept_tracker"] = tracker
        mm = MarketMap("EURUSD", **detectors)
        det = _md_detection(confidence=0.6)
        bonus = mm._md_bonus(det)
        assert bonus == 0.6 * 1.5 * 10
        tracker.get_weight.assert_called_once_with("EURUSD", MDConcept.POD)

    def test_with_concept_tracker_low_weight(self, detectors):
        tracker = MagicMock()
        tracker.get_weight.return_value = 0.3
        detectors["concept_tracker"] = tracker
        mm = MarketMap("EURUSD", **detectors)
        det = _md_detection(confidence=0.8)
        bonus = mm._md_bonus(det)
        assert bonus == 0.8 * 0.3 * 10

    def test_with_concept_tracker_intervals(self, detectors):
        tracker = MagicMock()
        tracker.get_weight.return_value = 0.0
        detectors["concept_tracker"] = tracker
        mm = MarketMap("EURUSD", **detectors)
        det = _md_detection(concept=MDConcept.INTERVAL, confidence=0.5)
        bonus = mm._md_bonus(det)
        assert bonus == 0.0

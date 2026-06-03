"""Market Map — Orchestrador de scalping completo (MEJORA SCALPING)
Coordina todos los módulos de scalping vertical:
   1. LiquidityMapper — mapea liquidez en todos los TFs
   2. ZoneStateTracker — gestiona ciclo de vida de zonas
   3. MicroPhaseDetector — detecta micro-fase actual
   4. BreakoutRetestDetector — detecta secuencias consolidación→ruptura→retest
   5. RoutePlanner — traza ruta hacia liquidez objetivo
   6. EntryConfirmer — valida timing con tick/DOM/stop-run
   7. CandleConfirmer — valida tesis en cierres HTF
   8. DynamicTPManager — expande TP con confirmación HTF
   9. PODDetector — detecta Traditional Block + POD (deuda interna)  [F1]
  10. IntervalDetector — detecta intervalos (punto interactivo)        [F1]
  11. LimitPriceDetector — detecta mechas extremas defendidas          [F2]
  12. PriceCaptureDetector — detecta micro-acumulación pre-breakout   [F2]
  13. OTEDetector — detecta Fibonacci OTE 75-79%                     [F2]
  14. Sequence123Detector — detecta conteo 1-2-3                     [F2]
  15. ThreeCandleDetector — detecta conteo de 3 velas consecutivas   [F3]
  16. AsymmetryFilter — valida asimetría matemática del setup        [F3]
  17. BinaryRiskFilter — filtro binario go/no-go pre-trade           [F3]
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from src.core.liquidity_mapper import LiquidityMapper, MarketMap as LiquidityMarketMap
from src.core.zone_state_tracker import ZoneStateTracker, ZoneStatus
from src.core.micro_phase import MicroPhaseDetector, PhaseResult, MicroPhase
from src.core.breakout_retest import BreakoutRetestDetector, BreakoutRetestSignal
from src.core.route_planner import RoutePlanner, Route
from src.core.entry_confirmer import EntryConfirmer, EntryConfirmation
from src.core.candle_confirmer import CandleConfirmer, CandleConfirmerResult, ConfirmerStatus
from src.core.dynamic_tp import DynamicTPManager, TPTier
from src.core.md_concepts import MDConcept, MDDetection
from src.core.md_pod_detector import PODDetector, PODResult
from src.core.md_interval_detector import IntervalDetector, IntervalResult
from src.core.md_limit_price_detector import LimitPriceDetector, LimitPriceResult
from src.core.md_price_capture_detector import PriceCaptureDetector, PriceCaptureResult
from src.core.md_ote_detector import OTEDetector, OTEResult
from src.core.md_sequence_123_detector import Sequence123Detector, Sequence123Result
from src.core.md_three_candle_detector import ThreeCandleDetector, ThreeCandleResult
from src.core.md_asymmetry_filter import AsymmetryFilter, AsymmetryResult
from src.core.md_binary_risk_filter import BinaryRiskFilter, BinaryRiskResult
from src.core.concept_tracker import ConceptPerformanceTracker, ConceptStats

logger = logging.getLogger(__name__)


@dataclass
class MarketMapResult:
    decision: str  # "TRADE" | "NO_TRADE" | "WAIT"
    direction: str
    confidence: float
    market_map: Optional[LiquidityMarketMap] = None
    phase: Optional[PhaseResult] = None
    breakout_signals: List[BreakoutRetestSignal] = field(default_factory=list)
    route: Optional[Route] = None
    entry_confirmation: Optional[EntryConfirmation] = None
    candle_confirmation: Dict[str, CandleConfirmerResult] = field(default_factory=dict)
    tp_tiers: List[TPTier] = field(default_factory=list)
    active_tp: Optional[float] = None
    suggested_entry: Optional[float] = None
    stop_loss: Optional[float] = None
    score_bonus: float = 0.0
    final_score: float = 0.0
    md_detections: List[MDDetection] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


class MarketMap:
    """Orquestador principal del sistema de scalping.

    Pipeline completo:
       1. Crear/actualizar mapa de liquidez (todos los TFs)
       2. Determinar micro-fase actual
       3. Gestionar estado de zonas
       4. Detectar breakout/retest
       5. Detectar conceptos MD (Phase 1 + Phase 2 + Phase 3)
       6. Planificar ruta hacia liquidez
       7. Confirmar entrada (tick/DOM/stop-run)
       8. Confirmar tesis HTF (1H/4H)
       9. Establecer TP dinámico
      10. Filtrar asimetría + riesgo binario
      11. Decisión final: TRADE / WAIT / NO_TRADE
    """

    def __init__(self, symbol: str,
                 zone_tracker: Optional[ZoneStateTracker] = None,
                 liquidity_mapper: Optional[LiquidityMapper] = None,
                 phase_detector: Optional[MicroPhaseDetector] = None,
                 breakout_detector: Optional[BreakoutRetestDetector] = None,
                 route_planner: Optional[RoutePlanner] = None,
                 entry_confirmer: Optional[EntryConfirmer] = None,
                 candle_confirmer: Optional[CandleConfirmer] = None,
                 tp_manager: Optional[DynamicTPManager] = None,
                 pod_detector: Optional[PODDetector] = None,
                 interval_detector: Optional[IntervalDetector] = None,
                 limit_price_detector: Optional[LimitPriceDetector] = None,
                 price_capture_detector: Optional[PriceCaptureDetector] = None,
                 ote_detector: Optional[OTEDetector] = None,
                 sequence_123_detector: Optional[Sequence123Detector] = None,
                 three_candle_detector: Optional[ThreeCandleDetector] = None,
                 asymmetry_filter: Optional[AsymmetryFilter] = None,
                 binary_risk_filter: Optional[BinaryRiskFilter] = None,
                 concept_tracker: Optional[ConceptPerformanceTracker] = None):
        self._symbol = symbol
        self._zone_tracker = zone_tracker or ZoneStateTracker()
        self._liquidity_mapper = liquidity_mapper or LiquidityMapper()
        self._phase_detector = phase_detector or MicroPhaseDetector()
        self._breakout = breakout_detector or BreakoutRetestDetector()
        self._route_planner = route_planner or RoutePlanner()
        self._entry_confirmer = entry_confirmer or EntryConfirmer()
        self._candle_confirmer = candle_confirmer or CandleConfirmer()
        self._tp_manager = tp_manager or DynamicTPManager()
        self._pod_detector = pod_detector or PODDetector()
        self._interval_detector = interval_detector or IntervalDetector()
        self._limit_price_detector = limit_price_detector or LimitPriceDetector()
        self._price_capture_detector = price_capture_detector or PriceCaptureDetector()
        self._ote_detector = ote_detector or OTEDetector()
        self._sequence_123_detector = sequence_123_detector or Sequence123Detector()
        self._three_candle_detector = three_candle_detector or ThreeCandleDetector()
        self._asymmetry_filter = asymmetry_filter or AsymmetryFilter()
        self._binary_risk_filter = binary_risk_filter or BinaryRiskFilter()
        self._concept_tracker = concept_tracker

    MIN_CONVICTION_BYPASS = 0.10

    def evaluate(self, dfs: Dict[str, pd.DataFrame],
                 direction: str = "NEUTRAL",
                 ticks: Optional[List[dict]] = None,
                 dom: Optional[dict] = None,
                 conviction: float = 0.0) -> MarketMapResult:
        result_notes = []

        df_m1 = dfs.get("M1")
        df_h1 = dfs.get("1h", dfs.get("H1"))
        df_h4 = dfs.get("4h", dfs.get("H4"))
        primary_df = df_m1

        if primary_df is None or primary_df.empty:
            for tf in ["M5", "M15", "M1"]:
                if tf in dfs and dfs[tf] is not None and not dfs[tf].empty:
                    primary_df = dfs[tf]
                    break

        pip = self._pip_size(self._symbol)
        atr_val = self._atr(primary_df) if primary_df is not None else 0.0

        liquidity_map = self._liquidity_mapper.build(self._symbol, dfs)
        result_notes.append(f"Mapa creado con {sum(len(z) for z in liquidity_map.zones.values())} zonas")

        dominant = liquidity_map.dominant_direction
        if direction == "NEUTRAL":
            direction = dominant
        result_notes.append(f"Dirección dominante: {dominant} → usando {direction}")

        try:
            phase = self._phase_detector.detect(primary_df) if primary_df is not None else PhaseResult(MicroPhase.INDECISION, "NEUTRAL", 0.0)
            result_notes.append(f"Fase: {phase.phase.value} (entry_allowed={phase.allows_entry})")
        except Exception:
            logger.exception("MicroPhaseDetector failed")
            phase = PhaseResult(MicroPhase.INDECISION, "NEUTRAL", 0.0)

        if liquidity_map and primary_df is not None:
            last_close = primary_df["close"].iloc[-1]
            for tf_label, tf_zones in liquidity_map.zones.items():
                for z in tf_zones:
                    self._zone_tracker.register_zone(
                        self._symbol, z.zone_type, tf_label,
                        z.zone_low, z.zone_high,
                        direction=z.direction, strength=z.strength,
                    )
            self._zone_tracker.update_price(self._symbol, last_close)

        breakout_signals = []
        if primary_df is not None:
            try:
                breakout_signals = self._breakout.check(primary_df, self._zone_tracker, self._symbol)
                for sig in breakout_signals:
                    result_notes.append(f"BreakoutRetest: {sig.direction} ({sig.notes[0] if sig.notes else ''})")
            except Exception:
                logger.exception("BreakoutRetestDetector failed")
                breakout_signals = []

        # ── MD Concept Detectors (Phase 1: POD, Interval; Phase 2: LimitPrice, PriceCapture, OTE, 1-2-3) ──
        md_detections: List[MDDetection] = []
        md_score_bonus = 0.0
        if primary_df is not None:
            try:
                pod = self._pod_detector.detect(primary_df)
                if pod.found:
                    det = pod.to_detection()
                    if det is not None:
                        md_detections.append(det)
                        md_score_bonus += self._md_bonus(det)
                        result_notes.append(f"MD {det.summary}")
            except Exception:
                logger.exception("PODDetector failed")

            try:
                interval = self._interval_detector.detect(primary_df)
                if interval.found:
                    det = interval.to_detection()
                    if det is not None:
                        md_detections.append(det)
                        md_score_bonus += self._md_bonus(det)
                        result_notes.append(f"MD {det.summary}")
            except Exception:
                logger.exception("IntervalDetector failed")

            try:
                limit_price = self._limit_price_detector.detect(primary_df)
                if limit_price.found:
                    det = limit_price.to_detection()
                    if det is not None:
                        md_detections.append(det)
                        md_score_bonus += self._md_bonus(det)
                        result_notes.append(f"MD {det.summary}")
            except Exception:
                logger.exception("LimitPriceDetector failed")

            try:
                price_capture = self._price_capture_detector.detect(primary_df)
                if price_capture.found:
                    det = price_capture.to_detection()
                    if det is not None:
                        md_detections.append(det)
                        md_score_bonus += self._md_bonus(det)
                        result_notes.append(f"MD {det.summary}")
            except Exception:
                logger.exception("PriceCaptureDetector failed")

            try:
                ote = self._ote_detector.detect(primary_df)
                if ote.found:
                    det = ote.to_detection()
                    if det is not None:
                        md_detections.append(det)
                        md_score_bonus += self._md_bonus(det)
                        result_notes.append(f"MD {det.summary}")
            except Exception:
                logger.exception("OTEDetector failed")

            try:
                seq_123 = self._sequence_123_detector.detect(primary_df)
                if seq_123.found:
                    det = seq_123.to_detection()
                    if det is not None:
                        md_detections.append(det)
                        md_score_bonus += self._md_bonus(det)
                        result_notes.append(f"MD {det.summary}")
            except Exception:
                logger.exception("Sequence123Detector failed")

            try:
                three_candle = self._three_candle_detector.detect(primary_df)
                if three_candle.found:
                    det = three_candle.to_detection()
                    if det is not None:
                        md_detections.append(det)
                        md_score_bonus += self._md_bonus(det)
                        result_notes.append(f"MD {det.summary}")
            except Exception:
                logger.exception("ThreeCandleDetector failed")

        if not phase.allows_entry and not breakout_signals:
            if conviction >= self.MIN_CONVICTION_BYPASS or md_detections:
                result_notes.append(
                    f"Scalping bypass: fase {phase.phase.value}, conv={conviction:.0%}, "
                    f"md={len(md_detections)} detecciones"
                )
            else:
                result_notes.append(f"Fase {phase.phase.value} no permite entrada y no hay breakout")
                return MarketMapResult(
                    decision="NO_TRADE", direction=direction, confidence=0.0,
                    market_map=liquidity_map, phase=phase,
                    md_detections=md_detections,
                    notes=result_notes,
                )

        if breakout_signals and breakout_signals[0].active:
            effective_direction = breakout_signals[0].direction
            effective_confidence = breakout_signals[0].confidence
            score_bonus = breakout_signals[0].score_bonus
            suggested_price = primary_df["close"].iloc[-1] if primary_df is not None else 0.0
            sl_price = suggested_price - atr_val * 0.8 if effective_direction == "BUY" else suggested_price + atr_val * 0.8
            tp_tiers = self._tp_manager.build_tiers(
                suggested_price, effective_direction, liquidity_map, primary_df,
                confirmed_1h=False, confirmed_4h=False,
            )
            active_tp = self._tp_manager.get_active_tp(tp_tiers)
            result_notes.append(f"Breakout activo: entrada {effective_direction}, bonus={score_bonus}")

            try:
                route = self._route_planner.plan(liquidity_map, primary_df, suggested_price, effective_direction)
            except Exception:
                logger.exception("RoutePlanner failed")
                route = Route(entries=[], is_valid=False, confidence=0.0, direction=effective_direction,
                              notes=["Route planner crashed"])

            total_bonus = score_bonus + md_score_bonus
            return MarketMapResult(
                decision="TRADE",
                direction=effective_direction,
                confidence=effective_confidence,
                market_map=liquidity_map,
                phase=phase,
                breakout_signals=breakout_signals,
                route=route,
                suggested_entry=suggested_price,
                stop_loss=sl_price,
                score_bonus=total_bonus,
                final_score=effective_confidence + total_bonus / 100,
                md_detections=md_detections,
                tp_tiers=tp_tiers,
                active_tp=active_tp,
                notes=result_notes,
            )

        try:
            route = self._route_planner.plan(liquidity_map, primary_df,
                                              primary_df["close"].iloc[-1] if primary_df is not None else 0.0,
                                              direction)
        except Exception:
            logger.exception("RoutePlanner failed")
            route = Route(entries=[], is_valid=False, confidence=0.0, direction=direction,
                          notes=["Route planner crashed"])
        if not route.is_valid:
            result_notes.append(f"Scalping: ruta no disponible ({route.notes[0] if route.notes else ''}), usando SL/TP por ATR")
            route = None

        try:
            entry = self._entry_confirmer.confirm(
                self._symbol, direction, phase, primary_df, ticks, dom,
            )
        except Exception:
            logger.exception("EntryConfirmer failed")
            entry = EntryConfirmation(valid=False, reason="entry_confirmer_crashed", confidence=0.0)
        if not entry.valid:
            if conviction >= self.MIN_CONVICTION_BYPASS or md_detections:
                result_notes.append(
                    f"Scalping bypass: entrada no confirmada ({entry.reason}), "
                    f"conv={conviction:.0%}, md={len(md_detections)} detecciones"
                )
            else:
                result_notes.append(f"Entrada no confirmada: {entry.reason}")
                return MarketMapResult(
                    decision="WAIT", direction=direction, confidence=entry.confidence,
                    route=route, market_map=liquidity_map, phase=phase,
                    entry_confirmation=entry, md_detections=md_detections,
                    notes=result_notes,
                )

        # Scalping: skip HTF confirmation (demasiado lento para scalping)
        confirmed_1h = False
        confirmed_4h = False
        candle_results = {}
        result_notes.append("Scalping: skip HTF confirmation")

        tp_tiers = self._tp_manager.build_tiers(
            entry.suggested_entry or (primary_df["close"].iloc[-1] if primary_df is not None else 0.0),
            direction, liquidity_map, primary_df,
            confirmed_1h=confirmed_1h, confirmed_4h=confirmed_4h,
        )
        active_tp = self._tp_manager.get_active_tp(tp_tiers)

        if route is not None and route.sl_zone_price:
            sl_price = route.sl_zone_price
        else:
            sl_price = (entry.suggested_entry or 0) - atr_val * 0.8 if direction == "BUY" else (entry.suggested_entry or 0) + atr_val * 0.8

        # ── Phase 3: Asymmetry Filter ──
        asym_entry = entry.suggested_entry or (primary_df["close"].iloc[-1] if primary_df is not None else 0.0)
        asym_sl = sl_price
        asym_tp = active_tp or asym_entry
        try:
            asymmetry = self._asymmetry_filter.evaluate(
                primary_df, direction,
                entry_price=asym_entry, stop_loss=asym_sl, target_price=asym_tp,
            ) if primary_df is not None else AsymmetryResult(passed=False, asymmetry_factor=0.0)
        except Exception:
            logger.exception("AsymmetryFilter failed")
            asymmetry = AsymmetryResult(passed=False, asymmetry_factor=0.0, notes=["AsymmetryFilter crashed"])
        if asymmetry.passed:
            md_detections.append(asymmetry.to_detection(direction))
            result_notes.append(f"F3 Asymmetry OK (factor={asymmetry.asymmetry_factor:.1f})")
        else:
            result_notes.append(f"F3 Asymmetry FAIL ({asymmetry.notes[0] if asymmetry.notes else ''})")

        # ── Phase 3: Binary Risk Filter ──
        zone_data = {"dominant_direction": dominant}
        try:
            binary = self._binary_risk_filter.evaluate(
                primary_df, direction,
                entry_price=asym_entry, stop_loss=asym_sl, target_price=asym_tp,
                zone_data=zone_data, macro_direction=dominant,
            ) if primary_df is not None else BinaryRiskResult(passed=False, failed_checks=["no_data"])
        except Exception:
            logger.exception("BinaryRiskFilter failed")
            binary = BinaryRiskResult(passed=False, failed_checks=["binary_risk_crashed"])
        if binary.passed:
            det = binary.to_detection(direction)
            if det is not None:
                md_detections.append(det)
            result_notes.append(f"F3 BinaryRisk PASS ({len(binary.passed_checks)} checks)")
        else:
            result_notes.append(f"F3 BinaryRisk ADVERTENCIA: {binary.failed_checks}")

        route_conf = route.confidence if route else 0.3
        final_confidence = phase.confidence * 0.1 + route_conf * 0.2 + entry.confidence * 0.4 + (md_score_bonus / 100) * 0.3
        if confirmed_1h:
            final_confidence += 0.1
        if confirmed_4h:
            final_confidence += 0.1
        final_confidence = min(0.95, final_confidence)
        final_score = final_confidence * 100

        tp_str = f"{active_tp:.5f}" if active_tp else "N/A"
        result_notes.append(
            f"Decisión: TRADE {direction} | confianza={final_confidence:.0%} | "
            f"TP={tp_str} | SL={sl_price:.5f}"
        )

        total_bonus = md_score_bonus
        return MarketMapResult(
            decision="TRADE",
            direction=direction,
            confidence=final_confidence,
            market_map=liquidity_map,
            phase=phase,
            breakout_signals=breakout_signals,
            route=route,
            entry_confirmation=entry,
            candle_confirmation=candle_results,
            tp_tiers=tp_tiers,
            active_tp=active_tp,
            suggested_entry=entry.suggested_entry,
            stop_loss=sl_price,
            score_bonus=total_bonus,
            final_score=final_score,
            md_detections=md_detections,
            notes=result_notes,
        )

    def _md_bonus(self, detection: MDDetection) -> float:
        """Calcula el bonus de score para una detección MD.
        Usa el peso histórico del ConceptTracker si está disponible.
        Si no hay historial suficiente, weight=1.0 (neutral).
        """
        weight = 1.0
        if self._concept_tracker is not None:
            weight = self._concept_tracker.get_weight(self._symbol, detection.concept)
        return detection.confidence * weight * 10

    def _pip_size(self, symbol: str) -> float:
        from src.utils.helpers import pip_size
        return pip_size(symbol)

    def _atr(self, df: pd.DataFrame, period: int = 14) -> float:
        from src.utils.helpers import atr
        if df is None or len(df) < period + 1:
            return 0.0
        return float(atr(df, period).iloc[-1])

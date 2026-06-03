"""MicroPredictor — Predicción combinada de dirección, target y timing
Integra OrderFlowSignal + ConfluenceResult + MarketMapResult + régimen
para producir una predicción: dirección, target, velas estimadas, confianza.
"""
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.order_flow import OrderFlowSignal
from src.core.market_map import MarketMapResult
from src.core.liquidity_mapper import MarketMap
from src.core.route_planner import Route
from src.core.regime_detector import RegimeContext
from src.predictor.confluence_scorer import (
    ConfluenceScorer, ConfluenceResult, TargetProjection,
)
from src.utils.helpers import atr, pip_size

logger = logging.getLogger(__name__)


@dataclass
class MicroPrediction:
    symbol: str
    current_price: float
    direction: str  # "BUY" | "SELL" | "NEUTRAL"
    target_price: Optional[float]
    target_type: str  # "liquidity" | "fvg" | "ob" | "breaker" | "route" | "confluence"
    confidence: float  # 0-1
    estimated_bars_m1: int
    primary_reason: str
    confluence_count: int

    # Desglose
    orderflow_direction: str
    orderflow_confidence: float
    confluence_confidence: float
    regime_bias: str
    regime_confidence: float
    market_map_bias: str

    targets_above: List[TargetProjection] = field(default_factory=list)
    targets_below: List[TargetProjection] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def is_bullish(self) -> bool:
        return self.direction == "BUY"

    @property
    def is_bearish(self) -> bool:
        return self.direction == "SELL"

    @property
    def is_neutral(self) -> bool:
        return self.direction == "NEUTRAL" or self.confidence < 0.2

    @property
    def summary(self) -> str:
        if self.is_neutral:
            return f"{self.symbol} NEUTRAL (confianza insuficiente)"
        target_str = f"{self.target_price:.2f}" if self.target_price is not None else "N/A"
        return (f"{self.symbol} {self.direction} → target {target_str} "
                f"en ~{self.estimated_bars_m1} velas M1 | "
                f"conf={self.confidence:.0%} | {self.confluence_count} confluences | "
                f"OF={self.orderflow_direction}@{self.orderflow_confidence:.0%}")


@dataclass
class PredictorConfig:
    min_confidence_to_predict: float = 0.25
    confluence_weight: float = 0.35
    orderflow_weight: float = 0.40
    regime_weight: float = 0.15
    market_map_weight: float = 0.10
    enable_ml_model: bool = True
    model_path: Optional[Path] = None


class MicroPredictor:
    """Predicción combinada: OrderFlow + confluencias + régimen + MarketMap.

    Flujo:
      1. Analiza OrderFlowSignal → dirección + confianza OF
      2. Puntúa confluencias del MarketMap → targets con confianza
      3. Incorpora bias de régimen + MarketMap
      4. Combina todo → predice dirección + target + timing
    """

    def __init__(self, config: Optional[PredictorConfig] = None):
        self._config = config or PredictorConfig()
        self._confluence_scorer = ConfluenceScorer()
        self._ml_model = None
        self._ml_features: List[dict] = []  # ring buffer for online learning
        self._max_ml_samples = 500
        self._prediction_history: List[MicroPrediction] = []
        self._max_history = 200
        self._last_trade_outcome: Optional[dict] = None

        # Load existing model if available
        if self._config.enable_ml_model and self._config.model_path:
            self._load_model()

    def predict(self, of_signal: OrderFlowSignal, market_map: MarketMap,
                market_map_result: MarketMapResult, route: Optional[Route],
                regime: RegimeContext, current_price: float, atr_val: float,
                pip: float) -> MicroPrediction:
        """Produce predicción combinando todas las señales."""
        notes = []

        # 1. OrderFlow direction & confidence
        of_dir, of_conf = self._orderflow_bias(of_signal)
        notes.append(f"OF: {of_dir} @ {of_conf:.0%}")

        # 2. Confluence scoring
        confluence_result = self._confluence_scorer.score(
            market_map, route, regime.regime.value if regime else "NEUTRAL", atr_val,
        )
        conf_dir, conf_conf, conf_target, conf_bars = self._confluence_bias(
            confluence_result, current_price)
        notes.append(f"Confluencia: {conf_dir} @ {conf_conf:.0%} → {conf_target}")

        # 3. Regime bias
        regime_dir, regime_conf = self._regime_bias(regime, current_price,
                                                     market_map, atr_val, pip)
        notes.append(f"Regimen: {regime_dir} @ {regime_conf:.0%}")

        # 4. MarketMap bias
        mm_dir, mm_conf = self._market_map_bias(market_map_result)
        notes.append(f"MarketMap: {mm_dir} @ {mm_conf:.0%}")

        # 5. Combine all signals
        direction, confidence, target, target_type, primary_reason = self._combine(
            of_dir, of_conf, conf_dir, conf_conf, conf_target,
            regime_dir, regime_conf, mm_dir, mm_conf,
            confluence_result, of_signal, current_price,
        )

        # 6. ML refinement (if model available)
        if self._ml_model is not None and self._config.enable_ml_model:
            try:
                ml_dir, ml_conf = self._ml_refine(
                    of_signal, of_dir, of_conf, conf_dir, conf_conf,
                    regime, current_price, atr_val,
                )
                notes.append(f"ML: {ml_dir} @ {ml_conf:.0%}")
                if ml_conf >= 0.5:
                    direction = ml_dir
                    confidence = confidence * 0.4 + ml_conf * 0.6
                    primary_reason = f"ml_{ml_dir}"
            except Exception as e:
                logger.debug(f"ML refinement error: {e}")

        # Estimate bars
        if target:
            est_bars = int(abs(target - current_price) / max(atr_val * 0.3, 0.0001))
            est_bars = max(1, min(999, est_bars))
        else:
            est_bars = confluence_result.primary_target.estimated_bars_m1 if (
                confluence_result and confluence_result.primary_target) else 999

        pred = MicroPrediction(
            symbol=market_map.symbol,
            current_price=current_price,
            direction=direction,
            target_price=target,
            target_type=target_type,
            confidence=min(1.0, confidence),
            estimated_bars_m1=est_bars,
            primary_reason=primary_reason,
            confluence_count=confluence_result.primary_target.confluence_count
            if (confluence_result and confluence_result.primary_target) else 0,
            orderflow_direction=of_dir,
            orderflow_confidence=of_conf,
            confluence_confidence=conf_conf,
            regime_bias=regime_dir,
            regime_confidence=regime_conf,
            market_map_bias=mm_dir,
            targets_above=confluence_result.targets_above if confluence_result else [],
            targets_below=confluence_result.targets_below if confluence_result else [],
            notes=notes,
        )

        self._prediction_history.append(pred)
        if len(self._prediction_history) > self._max_history:
            self._prediction_history.pop(0)

        return pred

    def _orderflow_bias(self, signal: OrderFlowSignal) -> Tuple[str, float]:
        """Dirección y confianza desde OrderFlow."""
        if signal is None:
            return "NEUTRAL", 0.0

        buy_pressure = getattr(signal, 'buy_pressure', 0.0) or 0.0
        sell_pressure = getattr(signal, 'sell_pressure', 0.0) or 0.0
        delta = getattr(signal, 'delta', 0.0) or 0.0

        # Exhaustion check
        exhaustion_active = getattr(signal, 'exhaustion_active', False)
        exhaustion_side = getattr(signal, 'exhaustion_side', "")

        if exhaustion_active:
            if exhaustion_side.lower() == "buy":
                return "SELL", 0.6
            elif exhaustion_side.lower() == "sell":
                return "BUY", 0.6

        # Stop-run detection
        stop_run_detected = getattr(signal, 'stop_run_detected', False)
        stop_run_dir = getattr(signal, 'stop_run_direction', "")

        if stop_run_detected:
            if stop_run_dir.lower() == "buy":
                return "SELL", 0.55
            elif stop_run_dir.lower() == "sell":
                return "BUY", 0.55

        # Absorption
        absorption_active = getattr(signal, 'absorption_active', False)
        if absorption_active:
            return f"{'BUY' if buy_pressure > sell_pressure else 'SELL'}", 0.45

        # Delta-based
        if buy_pressure > sell_pressure * 1.3:
            conf = min(0.7, buy_pressure / max(sell_pressure, 0.001) * 0.3)
            return "BUY", round(conf, 4)
        elif sell_pressure > buy_pressure * 1.3:
            conf = min(0.7, sell_pressure / max(buy_pressure, 0.001) * 0.3)
            return "SELL", round(conf, 4)

        # MACD divergence
        delta_macd = getattr(signal, 'delta_macd', None)
        if delta_macd:
            div_bullish = getattr(delta_macd, 'divergence_bullish', False)
            div_bearish = getattr(delta_macd, 'divergence_bearish', False)
            if div_bullish:
                return "BUY", 0.55
            if div_bearish:
                return "SELL", 0.55

        # Imbalance
        imb_class = getattr(signal, 'imbalance_classification', "")
        if "bullish" in imb_class.lower():
            return "BUY", 0.4
        elif "bearish" in imb_class.lower():
            return "SELL", 0.4

        return "NEUTRAL", 0.2

    def _confluence_bias(self, result: ConfluenceResult, current_price: float
                         ) -> Tuple[str, float, Optional[float], int]:
        if not result or not result.primary_target:
            return "NEUTRAL", 0.0, None, 999

        t = result.primary_target
        return t.direction, t.confidence, t.price, t.estimated_bars_m1

    def _regime_bias(self, regime: RegimeContext, current_price: float,
                     market_map: MarketMap, atr_val: float, pip: float
                     ) -> Tuple[str, float]:
        if regime is None:
            return "NEUTRAL", 0.0

        regime_name = regime.regime.value if hasattr(regime.regime, 'value') else str(regime.regime)
        direction = regime.direction if hasattr(regime, 'direction') else "NEUTRAL"
        conf = regime.confidence if hasattr(regime, 'confidence') else 0.0

        # Regime-based confidence adjustment
        if "STRONG_TREND" in regime_name:
            return direction, conf * 0.7
        elif "WEAK_TREND" in regime_name or "TRANSITION" in regime_name:
            return direction, conf * 0.4
        elif "RANGING" in regime_name:
            return "NEUTRAL", conf * 0.2
        else:
            return direction, conf * 0.3

    def _market_map_bias(self, result: MarketMapResult) -> Tuple[str, float]:
        if result is None:
            return "NEUTRAL", 0.0
        if result.phase and hasattr(result.phase, 'allows_entry') and result.phase.allows_entry:
            return result.direction, result.confidence * 0.5
        return "NEUTRAL", 0.1

    def _combine(self, of_dir: str, of_conf: float,
                 conf_dir: str, conf_conf: float, conf_target: Optional[float],
                 regime_dir: str, regime_conf: float,
                 mm_dir: str, mm_conf: float,
                 confluence: ConfluenceResult, of_signal: OrderFlowSignal,
                 current_price: float) -> Tuple[str, float, Optional[float], str, str]:
        w_of = self._config.orderflow_weight
        w_conf = self._config.confluence_weight
        w_reg = self._config.regime_weight
        w_mm = self._config.market_map_weight

        # Direction votes per system
        votes = {"BUY": 0.0, "SELL": 0.0}
        reasons = {"BUY": [], "SELL": []}

        def add_vote(direction: str, confidence: float, weight: float, source: str):
            if direction == "BUY":
                votes["BUY"] += confidence * weight
                reasons["BUY"].append(source)
            elif direction == "SELL":
                votes["SELL"] += confidence * weight
                reasons["SELL"].append(source)

        add_vote(of_dir, of_conf, w_of, "orderflow")
        add_vote(conf_dir, conf_conf, w_conf, "confluence")
        add_vote(regime_dir, regime_conf, w_reg, "regime")
        add_vote(mm_dir, mm_conf, w_mm, "marketmap")

        # Neutral baseline
        if votes["BUY"] == 0 and votes["SELL"] == 0:
            return "NEUTRAL", 0.0, None, "none", "sin_senales"

        total = votes["BUY"] + votes["SELL"]
        if total == 0:
            return "NEUTRAL", 0.0, None, "none", "sin_peso"

        buy_pct = votes["BUY"] / total
        sell_pct = votes["SELL"] / total

        # Need minimum edge
        edge = abs(buy_pct - sell_pct)
        if edge < 0.1:
            return "NEUTRAL", 0.0, None, "none", "edge_bajo"

        if buy_pct > sell_pct:
            direction = "BUY"
            confidence = buy_pct
            primary_reason = f"of+{reasons['BUY'][0]}" if reasons["BUY"] else "buy_vote"
            # Pick best target from confluence targets or route
            target, target_type = self._pick_target(confluence, "above", current_price, of_signal)
        else:
            direction = "SELL"
            confidence = sell_pct
            primary_reason = f"of+{reasons['SELL'][0]}" if reasons["SELL"] else "sell_vote"
            target, target_type = self._pick_target(confluence, "below", current_price, of_signal)

        return direction, round(confidence, 4), target, target_type, primary_reason

    def _pick_target(self, confluence: ConfluenceResult, side: str,
                     current_price: float, of_signal: OrderFlowSignal
                     ) -> Tuple[Optional[float], str]:
        if confluence is None:
            return None, "none"

        if side == "above":
            targets = confluence.targets_above
        else:
            targets = confluence.targets_below

        if targets:
            best = targets[0]
            return best.price, best.confluences[0] if best.confluences else "confluence"
        return None, "none"

    def _ml_refine(self, of_signal: OrderFlowSignal, of_dir: str, of_conf: float,
                   conf_dir: str, conf_conf: float, regime: RegimeContext,
                   current_price: float, atr_val: float) -> Tuple[str, float]:
        """Refina predicción usando modelo ML si está disponible."""
        if self._ml_model is None:
            return "NEUTRAL", 0.0

        features = self._build_ml_features(
            of_signal, of_dir, of_conf, conf_dir, conf_conf,
            regime, current_price, atr_val,
        )
        try:
            X = np.array([features]).reshape(1, -1)
            pred = self._ml_model.predict(X)[0]
            proba = self._ml_model.predict_proba(X)[0]
            if pred == 1:
                return "BUY", float(max(proba))
            else:
                return "SELL", float(max(proba))
        except Exception as e:
            logger.debug(f"ML predict error: {e}")
            return "NEUTRAL", 0.0

    def _build_ml_features(self, of_signal, of_dir, of_conf, conf_dir, conf_conf,
                           regime, current_price, atr_val) -> List[float]:
        fs = []

        # OrderFlow features (8)
        fs.append(of_conf)
        fs.append(1.0 if of_dir == "BUY" else (0.0 if of_dir == "SELL" else 0.5))
        fs.append(getattr(of_signal, 'delta', 0.0) or 0.0)
        fs.append(getattr(of_signal, 'buy_pressure', 0.0) or 0.0)
        fs.append(getattr(of_signal, 'sell_pressure', 0.0) or 0.0)
        fs.append(getattr(of_signal, 'imbalance_ratio', 0.0) or 0.0)
        fs.append(1.0 if getattr(of_signal, 'absorption_active', False) else 0.0)
        fs.append(1.0 if getattr(of_signal, 'stop_run_detected', False) else 0.0)

        # Confluence features (4)
        fs.append(conf_conf)
        fs.append(1.0 if conf_dir == "BUY" else (0.0 if conf_dir == "SELL" else 0.5))
        max_above = 0
        if hasattr(of_signal, 'targets_above') and of_signal.targets_above:
            max_above = max(t.confidence for t in of_signal.targets_above)
        fs.append(max_above)
        max_below = 0
        if hasattr(of_signal, 'targets_below') and of_signal.targets_below:
            max_below = max(t.confidence for t in of_signal.targets_below)
        fs.append(max_below)

        # Regime features (4)
        if regime:
            regime_val = 1.0 if "STRONG" in str(regime.regime.value) else (
                0.5 if "WEAK" in str(regime.regime.value) or "TRANSITION" in str(regime.regime.value) else 0.0
            )
            fs.append(regime_val)
            fs.append(regime.confidence if hasattr(regime, 'confidence') else 0.0)
            fs.append(1.0 if "BUY" in str(regime.direction).upper() else (
                0.0 if "SELL" in str(regime.direction).upper() else 0.5))
        else:
            fs.extend([0.0, 0.0, 0.5])
        fs.append(atr_val if atr_val > 0 else 0.001)

        # Market structure (2)
        fs.append(getattr(of_signal, 'ad_ratio', 0.5) if hasattr(of_signal, 'ad_ratio') else 0.5)
        fs.append(getattr(of_signal, 'cumulative_delta', 0.0) if hasattr(of_signal, 'cumulative_delta') else 0.0)

        return fs

    def record_outcome(self, prediction: MicroPrediction, actual_direction: str,
                        actual_profit: float):
        """Registra resultado de la predicción para entrenamiento futuro."""
        self._ml_features.append({
            "features": self._build_ml_features(
                None,
                prediction.orderflow_direction,
                prediction.orderflow_confidence,
                prediction.direction,
                prediction.confluence_confidence,
                None,
                prediction.current_price,
                abs(prediction.target_price - prediction.current_price) if prediction.target_price else 0,
            ),
            "outcome": 1 if actual_direction == "BUY" else 0,
            "profit": actual_profit,
            "prediction": prediction.direction,
            "timestamp": time.time(),
        })
        if len(self._ml_features) > self._max_ml_samples:
            self._ml_features.pop(0)

    def _load_model(self):
        try:
            if self._config.model_path and self._config.model_path.exists():
                with open(self._config.model_path, "rb") as f:
                    self._ml_model = pickle.load(f)
                logger.info(f"MicroPredictor model loaded: {self._config.model_path}")
        except Exception as e:
            logger.warning(f"MicroPredictor model load failed: {e}")

    def save_model(self, path: Path):
        if self._ml_model is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(self._ml_model, f)
            logger.info(f"MicroPredictor model saved: {path}")

    def get_latest(self) -> Optional[MicroPrediction]:
        if self._prediction_history:
            return self._prediction_history[-1]
        return None

    def get_history(self, n: int = 10) -> List[MicroPrediction]:
        return list(self._prediction_history[-n:]) if self._prediction_history else []

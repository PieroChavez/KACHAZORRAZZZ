"""Integration adapter - connects NN predictions into the strategy engine"""
import logging
from typing import Optional

import numpy as np

from src.neural.features import record_to_features, FEATURE_DIM
from src.neural.trainer import load_model, load_scaler

logger = logging.getLogger(__name__)


class NeuralAdvisor:
    def __init__(self):
        self._model = None
        self._scaler = None
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            self._model = load_model()
            self._scaler = load_scaler()
            self._loaded = True
            if self._model:
                logger.info("Neural advisor: model loaded")
            else:
                logger.info("Neural advisor: no trained model yet")

    @property
    def available(self) -> bool:
        self._ensure_loaded()
        return self._model is not None

    def predict_win_probability(
        self,
        score: float,
        conviction: float,
        regime: str,
        session: str,
        primary_pattern: Optional[str],
        direction: str,
        regime_confidence: float = 0.0,
    ) -> Optional[float]:
        self._ensure_loaded()
        if self._model is None or self._scaler is None:
            return None

        features = record_to_features(
            score=score,
            conviction=conviction,
            regime=regime,
            session=session,
            primary_pattern=primary_pattern,
            direction=direction,
            regime_confidence=regime_confidence,
        )

        features_norm = (features - np.array(self._scaler["mean"])) / np.array(self._scaler["std"])
        prob = self._model.predict_proba(features_norm.reshape(1, -1))
        return float(prob[0])

    def conviction_multiplier(self, win_prob: float) -> float:
        if win_prob is None:
            return 1.0
        if win_prob >= 0.7:
            return 1.25
        elif win_prob >= 0.6:
            return 1.10
        elif win_prob <= 0.3:
            return 0.70
        elif win_prob <= 0.4:
            return 0.85
        return 1.0

    def adjust_conviction(self, base_conviction: float, win_prob: Optional[float] = None) -> float:
        if win_prob is None:
            return base_conviction
        mult = self.conviction_multiplier(win_prob)
        adjusted = base_conviction * mult
        logger.debug(
            f"NN: win_prob={win_prob:.2f} mult={mult:.2f} "
            f"conviction {base_conviction:.2f} -> {adjusted:.2f}"
        )
        return min(adjusted, 1.0)


_advisor: Optional[NeuralAdvisor] = None


def get_advisor() -> NeuralAdvisor:
    global _advisor
    if _advisor is None:
        _advisor = NeuralAdvisor()
    return _advisor

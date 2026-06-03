"""ConceptTracker — Meta-learner por concepto MD
Por cada símbolo y cada MDConcept, trackea resultados históricos.
El bot descubre qué conceptos funcionan para cada par y ajusta pesos.
Si un concepto nunca funciona → weight → 0.0 (se ignora solo).
Si un concepto siempre acierta → weight → 1.5+ (se potencia).

Sin intervención humana. El bot aprende qué hipótesis son verdaderas.
"""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.core.md_concepts import MDConcept

logger = logging.getLogger(__name__)

MIN_SAMPLES_FOR_WEIGHT = 10
WEIGHT_FLOOR = 0.0
WEIGHT_CEIL = 2.0
BASELINE_EXPECTANCY = 5.0


@dataclass
class ConceptStats:
    total_signals: int = 0
    winning_signals: int = 0
    losing_signals: int = 0
    total_pnl: float = 0.0
    total_confidence: float = 0.0
    weight: float = 1.0

    @property
    def win_rate(self) -> float:
        if self.total_signals == 0:
            return 0.0
        return self.winning_signals / self.total_signals

    @property
    def avg_confidence(self) -> float:
        if self.total_signals == 0:
            return 0.0
        return self.total_confidence / self.total_signals

    @property
    def expectancy(self) -> float:
        if self.total_signals == 0:
            return 0.0
        return self.total_pnl / self.total_signals

    def record(self, won: bool, pnl: float, confidence: float):
        self.total_signals += 1
        if won:
            self.winning_signals += 1
        else:
            self.losing_signals += 1
        self.total_pnl += pnl
        self.total_confidence += confidence
        self.weight = self._recalc_weight()

    def _recalc_weight(self) -> float:
        if self.total_signals < MIN_SAMPLES_FOR_WEIGHT:
            return 1.0
        exp = self.expectancy
        if BASELINE_EXPECTANCY > 0:
            raw = exp / BASELINE_EXPECTANCY
        else:
            raw = self.win_rate / 0.5 - 1.0
        return max(WEIGHT_FLOOR, min(WEIGHT_CEIL, raw))

    def to_dict(self) -> dict:
        return {
            "total_signals": self.total_signals,
            "winning_signals": self.winning_signals,
            "losing_signals": self.losing_signals,
            "total_pnl": self.total_pnl,
            "weight": self.weight,
            "win_rate": self.win_rate,
            "expectancy": self.expectancy,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConceptStats":
        stats = cls()
        stats.total_signals = data.get("total_signals", 0)
        stats.winning_signals = data.get("winning_signals", 0)
        stats.losing_signals = data.get("losing_signals", 0)
        stats.total_pnl = data.get("total_pnl", 0.0)
        stats.total_confidence = data.get("total_confidence", 0.0)
        stats.weight = data.get("weight", 1.0)
        return stats


SymbolConceptMap = Dict[str, Dict[str, ConceptStats]]


class ConceptPerformanceTracker:
    """Trackea rendimiento de conceptos MD por símbolo.

    Por cada par (XAUUSD, EURUSD, etc.) y cada concepto (POD, INTERVAL, etc.):
      - Cuántas señales generó
      - Cuántas ganó/perdió
      - PnL total acumulado
      - Peso ajustado dinámicamente
    """

    def __init__(self, persistence_path: Optional[Path] = None):
        if persistence_path is None:
            persistence_path = Path(__file__).resolve().parent.parent.parent / "data" / "concept_tracker.json"
        self._path = persistence_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: SymbolConceptMap = {}
        self._load()

    def record_result(self, symbol: str, concept: MDConcept,
                      won: bool, pnl: float, confidence: float):
        stats = self._get_stats(symbol, concept)
        stats.record(won, pnl, confidence)
        logger.debug(
            f"[{symbol}] Concept {concept.value}: {'WIN' if won else 'LOSS'} "
            f"pnl={pnl:.1f} conf={confidence:.0%} "
            f"→ weight={stats.weight:.2f} (n={stats.total_signals})"
        )
        self._save()

    def get_weight(self, symbol: str, concept: MDConcept) -> float:
        stats = self._data.get(symbol, {}).get(concept.value)
        if stats is None:
            return 1.0
        return stats.weight

    def get_stats(self, symbol: str, concept: MDConcept) -> ConceptStats:
        return self._get_stats(symbol, concept)

    def summary(self, symbol: str) -> Dict[str, ConceptStats]:
        return self._data.get(symbol, {})

    def all_summary(self) -> SymbolConceptMap:
        return self._data

    def _get_stats(self, symbol: str, concept: MDConcept) -> ConceptStats:
        if symbol not in self._data:
            self._data[symbol] = {}
        key = concept.value
        if key not in self._data[symbol]:
            self._data[symbol][key] = ConceptStats()
        return self._data[symbol][key]

    def _load(self):
        if not self._path.exists():
            logger.info(f"ConceptTracker: nuevo archivo en {self._path}")
            return
        try:
            with open(self._path, "r") as f:
                raw: dict = json.load(f)
            for symbol, concepts in raw.items():
                if symbol not in self._data:
                    self._data[symbol] = {}
                for concept_key, stats_data in concepts.items():
                    self._data[symbol][concept_key] = ConceptStats.from_dict(stats_data)
            logger.info(f"ConceptTracker: cargados {sum(len(v) for v in self._data.values())} conceptos")
        except Exception as e:
            logger.warning(f"ConceptTracker: error cargando {self._path}: {e}")

    def _save(self):
        try:
            raw: dict = {}
            for symbol, concepts in self._data.items():
                raw[symbol] = {k: v.to_dict() for k, v in concepts.items()}
            with open(self._path, "w") as f:
                json.dump(raw, f, indent=2)
        except Exception as e:
            logger.warning(f"ConceptTracker: error guardando {self._path}: {e}")

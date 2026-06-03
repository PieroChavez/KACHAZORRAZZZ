"""MD Concepts — Conceptos de la metodología MD (Fases 1-3)
Define los patrones institucionales que el bot puede detectar y aprender a usar.
Cada detector MD es una hipótesis que el bot valida con datos reales.
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class MDConcept(Enum):
    POD = "pod"
    INTERVAL = "interval"
    LIMIT_PRICE = "limit_price"
    PRICE_CAPTURE = "price_capture"
    OTE_75_79 = "ote_75_79"
    SEQUENCE_123 = "sequence_123"
    THREE_CANDLE = "three_candle"
    ASYMMETRY = "asymmetry"
    BINARY_RISK = "binary_risk"


CONCEPT_DESCRIPTION = {
    MDConcept.POD: "Deuda interna: gap entre OB y Bloque Tradicional, entrada al 50%",
    MDConcept.INTERVAL: "Vela de cuerpo mínimo con mechas bilaterales largas + volumen",
    MDConcept.LIMIT_PRICE: "Mecha extrema defendida por cuerpos de velas siguientes",
    MDConcept.PRICE_CAPTURE: "Micro-acumulación Doji justo antes de breakout masivo",
    MDConcept.OTE_75_79: "Retroceso Fibonacci 75-79% para imbalances grandes",
    MDConcept.SEQUENCE_123: "Secuencia 1-breakout 2-inducement 3-resteo final",
    MDConcept.THREE_CANDLE: "Conteo de 3 velas consecutivas, entrada al 50% de vela 2",
    MDConcept.ASYMMETRY: "Asimetría matemática: impulso rápido vs retroceso lento + R:R",
    MDConcept.BINARY_RISK: "Filtro binario: SL estructural, R:R mínimo, volumen, confluencia",
}


@dataclass
class MDDetection:
    concept: MDConcept
    direction: str
    confidence: float
    suggested_price: Optional[float] = None
    timeframe: str = ""
    metadata: Dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.concept.value

    @property
    def summary(self) -> str:
        price = f"{self.suggested_price:.5f}" if self.suggested_price else "N/A"
        return f"{self.concept.value} {self.direction} conf={self.confidence:.0%} @{price}"

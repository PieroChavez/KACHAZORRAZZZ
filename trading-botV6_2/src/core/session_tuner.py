"""Session Auto-Tuner (MEJORA 17)
Cada sesión (Asia, London, NY, Overlap, Close) tiene su PROPIO
conjunto de parámetros: SL/TP multipliers, pattern weights, score mínimo,
número de escalas, RR mínimo, convicción mínima.
Reconoce que el mercado se comporta fundamentalmente distinto según hora.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.core.session_profiler import TradingSession, SESSION_HOURS_UTC

logger = logging.getLogger(__name__)


@dataclass
class SessionTunedParams:
    """Conjunto completo de parámetros ajustados para una sesión específica."""
    session_label: str
    min_score: float
    min_conviction: float
    scale_entries: int
    sl_multiplier: float
    tp_multiplier: float
    volume_multiplier: float
    aggressiveness: str
    pattern_weight_mult: Dict[str, float]
    preferred_patterns: List[str]
    avoided_patterns: List[str]
    rr_min: float
    notes: List[str] = field(default_factory=list)


SESSION_CONFIGS: Dict[str, dict] = {
    "ASIAN": {
        "min_score": 72.0,
        "min_conviction": 0.25,
        "scale_entries": 3,
        "sl_multiplier": 1.3,
        "tp_multiplier": 1.8,
        "volume_multiplier": 0.6,
        "aggressiveness": "conservative",
        "pattern_weight_mult": {
            "FVG": 0.7, "BREAKER": 0.5, "VOID_SCALP": 0.3, "SWEEP": 1.1,
            "OB": 1.2, "CYCLE": 0.8, "BOS_ZONE": 0.9, "WYCKOFF": 0.4,
            "PRESSURE_ZONE": 1.1, "HARMONIC_CYCLE": 1.1, "PRICE_INTERACTION": 0.8,
            "TRIPLE_CONF": 0.6, "SWEEP_CONF": 0.8, "MF_ALIGN": 1.0,
            "INTERVAL_POINT": 0.7,
        },
        "preferred": ["OB", "SWEEP", "PRESSURE_ZONE", "HARMONIC_CYCLE"],
        "avoided": ["VOID_SCALP", "BREAKER", "WYCKOFF"],
        "rr_min": 3.0,
        "notes": [
            "Asia: rango estrecho, preferir OB/SWEEP",
            "SL +30% por spreads mayores y movimiento lento",
        ],
    },
    "LONDON_OPEN": {
        "min_score": 65.0,
        "min_conviction": 0.15,
        "scale_entries": 5,
        "sl_multiplier": 0.9,
        "tp_multiplier": 2.0,
        "volume_multiplier": 1.1,
        "aggressiveness": "aggressive",
        "pattern_weight_mult": {
            "FVG": 1.3, "BREAKER": 1.2, "VOID_SCALP": 1.2, "SWEEP": 1.0,
            "OB": 0.9, "CYCLE": 0.8, "BOS_ZONE": 1.2, "WYCKOFF": 0.7,
            "PRESSURE_ZONE": 0.8, "HARMONIC_CYCLE": 0.8, "PRICE_INTERACTION": 0.7,
            "TRIPLE_CONF": 1.0, "SWEEP_CONF": 0.9, "MF_ALIGN": 1.2,
            "INTERVAL_POINT": 1.0,
        },
        "preferred": ["FVG", "BREAKER", "VOID_SCALP", "BOS_ZONE"],
        "avoided": ["WYCKOFF", "PRESSURE_ZONE"],
        "rr_min": 4.0,
        "notes": [
            "London Open: alta volatilidad, FVG/BREAKER/VOID funcionan bien",
            "SL -10% por movimiento rápido, buscar RR >= 4:1",
        ],
    },
    "LONDON_MID": {
        "min_score": 68.0,
        "min_conviction": 0.20,
        "scale_entries": 4,
        "sl_multiplier": 1.0,
        "tp_multiplier": 1.5,
        "volume_multiplier": 0.9,
        "aggressiveness": "moderate",
        "pattern_weight_mult": {
            "FVG": 0.9, "BREAKER": 0.7, "VOID_SCALP": 0.6, "SWEEP": 1.1,
            "OB": 1.2, "CYCLE": 1.1, "BOS_ZONE": 0.8, "WYCKOFF": 0.8,
            "PRESSURE_ZONE": 1.1, "HARMONIC_CYCLE": 1.2, "PRICE_INTERACTION": 1.2,
            "TRIPLE_CONF": 0.9, "SWEEP_CONF": 1.0, "MF_ALIGN": 1.1,
            "INTERVAL_POINT": 0.9,
        },
        "preferred": ["OB", "SWEEP", "CYCLE", "PRICE_INTERACTION"],
        "avoided": ["VOID_SCALP", "BREAKER"],
        "rr_min": 3.5,
        "notes": [
            "London Mid: consolidación tras apertura, OB/CYCLE/HARMONIC",
        ],
    },
    "NY_OPEN": {
        "min_score": 62.0,
        "min_conviction": 0.12,
        "scale_entries": 6,
        "sl_multiplier": 0.85,
        "tp_multiplier": 2.2,
        "volume_multiplier": 1.2,
        "aggressiveness": "aggressive",
        "pattern_weight_mult": {
            "FVG": 1.4, "BREAKER": 1.3, "VOID_SCALP": 1.3, "SWEEP": 1.0,
            "OB": 0.8, "CYCLE": 0.7, "BOS_ZONE": 1.3, "WYCKOFF": 0.6,
            "PRESSURE_ZONE": 0.7, "HARMONIC_CYCLE": 0.8, "PRICE_INTERACTION": 0.6,
            "TRIPLE_CONF": 1.1, "SWEEP_CONF": 0.8, "MF_ALIGN": 1.3,
            "INTERVAL_POINT": 1.1,
        },
        "preferred": ["FVG", "BREAKER", "BOS_ZONE", "VOID_SCALP"],
        "avoided": ["WYCKOFF"],
        "rr_min": 4.5,
        "notes": [
            "NY Open: máxima volatilidad, FVG/BREAKER/BOS_ZONE excelentes",
            "SL -15%, TP +20%, buscar grandes movimientos",
        ],
    },
    "LONDON_NY_OVERLAP": {
        "min_score": 60.0,
        "min_conviction": 0.10,
        "scale_entries": 6,
        "sl_multiplier": 0.8,
        "tp_multiplier": 2.5,
        "volume_multiplier": 1.3,
        "aggressiveness": "aggressive",
        "pattern_weight_mult": {
            "FVG": 1.5, "BREAKER": 1.4, "VOID_SCALP": 1.2, "SWEEP": 1.1,
            "OB": 0.9, "CYCLE": 1.1, "BOS_ZONE": 1.4, "WYCKOFF": 0.7,
            "PRESSURE_ZONE": 0.8, "HARMONIC_CYCLE": 0.9, "PRICE_INTERACTION": 0.8,
            "TRIPLE_CONF": 1.2, "SWEEP_CONF": 0.9, "MF_ALIGN": 1.4,
            "INTERVAL_POINT": 1.2,
        },
        "preferred": ["FVG", "BOS_ZONE", "CYCLE", "BREAKER", "SWEEP"],
        "avoided": [],
        "rr_min": 5.0,
        "notes": [
            "Overlap: mejor momento para operar, todos los patrones funcionan",
            "SL -20%, TP +50%, volumen +30%, buscar RR >= 5:1",
        ],
    },
    "NY_AFTERNOON": {
        "min_score": 72.0,
        "min_conviction": 0.25,
        "scale_entries": 3,
        "sl_multiplier": 1.2,
        "tp_multiplier": 1.5,
        "volume_multiplier": 0.7,
        "aggressiveness": "conservative",
        "pattern_weight_mult": {
            "FVG": 0.6, "BREAKER": 0.5, "VOID_SCALP": 0.4, "SWEEP": 1.1,
            "OB": 1.3, "CYCLE": 1.2, "BOS_ZONE": 0.7, "WYCKOFF": 0.5,
            "PRESSURE_ZONE": 1.2, "HARMONIC_CYCLE": 1.3, "PRICE_INTERACTION": 1.2,
            "TRIPLE_CONF": 0.6, "SWEEP_CONF": 1.1, "MF_ALIGN": 0.8,
            "INTERVAL_POINT": 1.0,
        },
        "preferred": ["OB", "SWEEP", "PRESSURE_ZONE", "HARMONIC_CYCLE"],
        "avoided": ["FVG", "VOID_SCALP"],
        "rr_min": 3.0,
        "notes": [
            "NY Afternoon: baja volatilidad, preferir OB/SWEEP/HARMONIC",
            "Volumen -30%, SL +20% por spreads crecientes",
        ],
    },
    "CLOSE": {
        "min_score": 78.0,
        "min_conviction": 0.35,
        "scale_entries": 2,
        "sl_multiplier": 1.5,
        "tp_multiplier": 1.2,
        "volume_multiplier": 0.3,
        "aggressiveness": "conservative",
        "pattern_weight_mult": {
            "FVG": 0.3, "BREAKER": 0.3, "VOID_SCALP": 0.2, "SWEEP": 0.8,
            "OB": 0.7, "CYCLE": 0.6, "BOS_ZONE": 0.4, "WYCKOFF": 0.3,
            "PRESSURE_ZONE": 0.7, "HARMONIC_CYCLE": 0.6, "PRICE_INTERACTION": 0.6,
            "TRIPLE_CONF": 0.3, "SWEEP_CONF": 0.5, "MF_ALIGN": 0.4,
            "INTERVAL_POINT": 0.5,
        },
        "preferred": [],
        "avoided": ["FVG", "VOID_SCALP", "BREAKER", "BOS_ZONE"],
        "rr_min": 2.5,
        "notes": [
            "Close: evitar operar, spreads amplios, solo señales excepcionales",
            "Volumen -70%, Score mínimo 78, convicción mínima 0.35",
        ],
    },
}


SESSION_LABEL_MAP = {
    TradingSession.ASIAN: "ASIAN",
    TradingSession.LONDON_OPEN: "LONDON_OPEN",
    TradingSession.LONDON_MID: "LONDON_MID",
    TradingSession.NY_OPEN: "NY_OPEN",
    TradingSession.LONDON_NY_OVERLAP: "LONDON_NY_OVERLAP",
    TradingSession.NY_AFTERNOON: "NY_AFTERNOON",
    TradingSession.CLOSE: "CLOSE",
}


PATTERN_WEIGHT_KEYS = [
    "FVG", "BREAKER", "VOID_SCALP", "SWEEP", "OB", "CYCLE", "BOS_ZONE",
    "WYCKOFF", "PRESSURE_ZONE", "HARMONIC_CYCLE", "PRICE_INTERACTION",
    "TRIPLE_CONF", "SWEEP_CONF", "MF_ALIGN", "INTERVAL_POINT",
]

SCOCFG_ATTR_MAP = {
    "FVG": "fvg_detected",
    "BREAKER": "breaker_retest",
    "VOID_SCALP": "void_scalp_confirmed",
    "SWEEP": "liquidity_sweep_ltf",
    "OB": "order_block_valid",
    "CYCLE": "triple_confluence",
    "BOS_ZONE": "bos_zone_retest",
    "WYCKOFF": "wyckoff_phase_c_spring",
    "PRESSURE_ZONE": "pressure_zone_bonus",
    "HARMONIC_CYCLE": "harmonic_cycle_aligned",
    "PRICE_INTERACTION": "price_interaction_bonus",
    "TRIPLE_CONF": "triple_confluence",
    "SWEEP_CONF": "ltf_sweep_confirmation",
    "MF_ALIGN": "multiframe_alignment",
    "INTERVAL_POINT": "interval_point_bonus",
}


class SessionTuner:
    """Auto-tuning de parámetros por sesión de trading.

    Cada sesión (Asia, London Open, London Mid, NY Open, Overlap,
    NY Afternoon, Close) tiene configuraciones expertas independientes
    para score mínimo, convicción, escalas, SL/TP, pesos de patrones, etc.
    """

    def __init__(self, user_overrides: Optional[Dict[str, dict]] = None):
        self._configs: Dict[str, dict] = {}
        for label, config in SESSION_CONFIGS.items():
            base = dict(config)
            base["pattern_weight_mult"] = dict(config["pattern_weight_mult"])
            base["preferred"] = list(config["preferred"])
            base["avoided"] = list(config["avoided"])
            base["notes"] = list(config["notes"])
            if user_overrides and label in user_overrides:
                override = user_overrides[label]
                for k, v in override.items():
                    if k == "pattern_weight_mult" and isinstance(v, dict):
                        base["pattern_weight_mult"].update(v)
                    elif k in ("preferred", "avoided", "notes") and isinstance(v, list):
                        base[k] = list(v)
                    else:
                        base[k] = v
            self._configs[label] = base

    def get_config(self, session: TradingSession) -> dict:
        label = SESSION_LABEL_MAP.get(session, "ASIAN")
        return self._configs.get(label, self._configs["ASIAN"])

    def tune(self, session: TradingSession,
             base_min_score: float = 65.0,
             base_scale_entries: int = 5,
             base_rr: float = 4.0,
             session_profile_vol_adj: float = 1.0,
             ) -> SessionTunedParams:
        cfg = self.get_config(session)
        label = SESSION_LABEL_MAP.get(session, "ASIAN")

        min_score = cfg.get("min_score", base_min_score)
        min_conviction = cfg.get("min_conviction", 0.15)
        scale_entries = cfg.get("scale_entries", base_scale_entries)
        sl_mult = cfg.get("sl_multiplier", 1.0)
        tp_mult = cfg.get("tp_multiplier", 1.0)
        vol_mult = cfg.get("volume_multiplier", 1.0) * session_profile_vol_adj
        aggressiveness = cfg.get("aggressiveness", "moderate")
        rr_min = cfg.get("rr_min", base_rr)
        preferred = cfg.get("preferred", [])
        avoided = cfg.get("avoided", [])
        weight_mult = cfg.get("pattern_weight_mult", {})

        notes = list(cfg.get("notes", []))
        if session_profile_vol_adj != 1.0:
            notes.append(f"SessionProfiler vol adj: ×{session_profile_vol_adj:.2f}")

        return SessionTunedParams(
            session_label=label,
            min_score=min_score,
            min_conviction=min_conviction,
            scale_entries=scale_entries,
            sl_multiplier=sl_mult,
            tp_multiplier=tp_mult,
            volume_multiplier=vol_mult,
            aggressiveness=aggressiveness,
            pattern_weight_mult=dict(weight_mult),
            preferred_patterns=list(preferred),
            avoided_patterns=list(avoided),
            rr_min=rr_min,
            notes=notes,
        )

    def get_min_score(self, session: TradingSession,
                      base_min_score: float = 65.0) -> float:
        return self.get_config(session).get("min_score", base_min_score)

    def get_scale_entries(self, session: TradingSession,
                          base_scale: int = 5) -> int:
        return self.get_config(session).get("scale_entries", base_scale)

    def get_sl_tp_multipliers(self, session: TradingSession
                              ) -> Tuple[float, float]:
        cfg = self.get_config(session)
        return cfg.get("sl_multiplier", 1.0), cfg.get("tp_multiplier", 1.0)

    def get_pattern_weight_multipliers(self, session: TradingSession
                                       ) -> Dict[str, float]:
        return dict(self.get_config(session).get("pattern_weight_mult", {}))

    @staticmethod
    def apply_weight_multipliers(base_score: float, mult: float) -> float:
        """Aplica un multiplicador a un peso base de ScoringConfig."""
        return round(base_score * mult, 1)

    def get_rr_min(self, session: TradingSession,
                   base_rr: float = 4.0) -> float:
        return self.get_config(session).get("rr_min", base_rr)

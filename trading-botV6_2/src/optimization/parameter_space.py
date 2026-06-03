"""Parameter Space Definition for Genetic Optimization (Mejora 5)
Defines search bounds, types, and constraints for all tunable parameters.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple


@dataclass
class ParamDef:
    name: str
    lo: float
    hi: float
    ptype: str  # "float", "int", "categorical"
    step: float = None
    categories: List = None
    description: str = ""
    constraint: Callable = None  # validation function


BASE_SPACE: Dict[str, ParamDef] = {
    "risk_per_trade": ParamDef("risk_per_trade", 0.003, 0.03, "float", description="Riesgo por operación"),
    "atr_multiplier_sl": ParamDef("atr_multiplier_sl", 0.8, 3.5, "float", description="Multiplicador ATR para SL"),
    "atr_multiplier_tp": ParamDef("atr_multiplier_tp", 1.2, 5.0, "float", description="Multiplicador ATR para TP"),
    "min_reward_risk_ratio": ParamDef("min_reward_risk_ratio", 1.0, 3.5, "float", description="Ratio reward/risk mínimo"),
    "min_score_to_trade": ParamDef("min_score_to_trade", 40.0, 85.0, "int", step=1, description="Score mínimo para operar"),
    "high_confidence_score": ParamDef("high_confidence_score", 70.0, 98.0, "int", step=1, description="Score de alta confianza"),
    "min_net_score": ParamDef("min_net_score", 10.0, 50.0, "int", step=1, description="Score neto mínimo"),
    "min_reversal_score": ParamDef("min_reversal_score", 55.0, 95.0, "int", step=1, description="Score para reversión"),
    "conviction_threshold": ParamDef("conviction_threshold", 0.08, 0.25, "float", description="Umbral de convicción"),
    "sl_min_pips": ParamDef("sl_min_pips", 3.0, 12.0, "int", step=1, description="SL mínimo en pips"),
    "sl_max_pips": ParamDef("sl_max_pips", 15.0, 50.0, "int", step=1, description="SL máximo en pips"),
    "cooldown_bars_m1": ParamDef("cooldown_bars_m1", 0, 15, "int", step=1, description="Velas de cooldown"),
    "max_concurrent_trades": ParamDef("max_concurrent_trades", 1, 5, "int", step=1, description="Máximas posiciones simultáneas"),
    "base_risk_pct": ParamDef("base_risk_pct", 0.01, 0.05, "float", description="Porcentaje de riesgo base"),
}

EXPERT_SPACE: Dict[str, ParamDef] = {
    **BASE_SPACE,
    "vol_base_sl_mult": ParamDef("vol_base_sl_mult", 0.8, 2.5, "float", description="Base SL mult para VolatilityScaler"),
    "vol_base_tp_mult": ParamDef("vol_base_tp_mult", 1.2, 3.5, "float", description="Base TP mult para VolatilityScaler"),
    "corr_threshold": ParamDef("corr_threshold", 0.5, 0.9, "float", description="Umbral de correlación para filtro"),
    "corr_volume_reduction": ParamDef("corr_volume_reduction", 0.1, 0.5, "float", description="Reducción de volumen por correlación"),
    "score_weight_trend": ParamDef("score_weight_trend", 0.1, 0.5, "float", description="Peso de tendencia en scoring"),
    "score_weight_pattern": ParamDef("score_weight_pattern", 0.1, 0.5, "float", description="Peso de patrones en scoring"),
    "score_weight_momentum": ParamDef("score_weight_momentum", 0.05, 0.3, "float", description="Peso de momentum en scoring"),
    "score_weight_volume": ParamDef("score_weight_volume", 0.05, 0.3, "float", description="Peso de volumen en scoring"),
}

FITNESS_WEIGHTS = {
    "sharpe": 3.0,
    "profit": 2.0,
    "win_rate": 1.5,
    "profit_factor": 2.0,
    "max_drawdown": -2.0,
    "trades": 0.5,
    "avg_risk_reward": 1.0,
}


def decode_chromosome(chrom: Dict, space: Dict) -> Dict:
    """Convert normalized [0,1] values back to parameter values"""
    decoded = {}
    for name, val in chrom.items():
        pdef = space.get(name)
        if pdef is None:
            decoded[name] = val
            continue
        raw = pdef.lo + val * (pdef.hi - pdef.lo)
        if pdef.ptype == "int":
            raw = int(round(raw))
        decoded[name] = raw
    return decoded


def encode_params(params: Dict, space: Dict) -> Dict:
    """Convert parameter values to normalized [0,1]"""
    encoded = {}
    for name, val in params.items():
        pdef = space.get(name)
        if pdef is None:
            encoded[name] = val
            continue
        norm = (val - pdef.lo) / (pdef.hi - pdef.lo) if pdef.hi > pdef.lo else 0.5
        encoded[name] = max(0.0, min(1.0, norm))
    return encoded

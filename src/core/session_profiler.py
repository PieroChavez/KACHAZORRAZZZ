"""Time-Aware Session Profiler
Analyzes market behavior by time of day and day of week.
Each trading session (Asian, London, NY, Overlap) has distinct characteristics.
The profiler adjusts strategy parameters accordingly.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum

import pandas as pd
import numpy as np

from src.utils.helpers import atr

logger = logging.getLogger(__name__)


class TradingSession(Enum):
    ASIAN = "ASIAN"
    LONDON_OPEN = "LONDON_OPEN"
    LONDON_MID = "LONDON_MID"
    NY_OPEN = "NY_OPEN"
    LONDON_NY_OVERLAP = "LONDON_NY_OVERLAP"
    NY_AFTERNOON = "NY_AFTERNOON"
    CLOSE = "CLOSE"


SESSION_HOURS_UTC = {
    TradingSession.ASIAN: (0, 7),
    TradingSession.LONDON_OPEN: (7, 9),
    TradingSession.LONDON_MID: (9, 12),
    TradingSession.NY_OPEN: (12, 14),
    TradingSession.LONDON_NY_OVERLAP: (12, 16),
    TradingSession.NY_AFTERNOON: (16, 21),
    TradingSession.CLOSE: (21, 24),
}

SESSION_WEAK_WINDOWS = {
    TradingSession.ASIAN: [(0, 3), (5, 7)],
    TradingSession.LONDON_OPEN: [(7, 8)],
    TradingSession.NY_OPEN: [(12, 13)],
    TradingSession.LONDON_NY_OVERLAP: [(14, 15)],
    TradingSession.NY_AFTERNOON: [(18, 20)],
}

SESSION_PEAK_WINDOWS = {
    TradingSession.LONDON_OPEN: [(8, 9)],
    TradingSession.NY_OPEN: [(13, 14)],
    TradingSession.LONDON_NY_OVERLAP: [(12, 14)],
}


@dataclass
class SessionProfile:
    session: TradingSession
    label: str
    volatility_pct: float
    direction_bias: Optional[str]
    preferred_patterns: List[str]
    avoided_patterns: List[str]
    sl_adjustment: float
    tp_adjustment: float
    volume_adjustment: float
    aggressiveness: str
    is_peak: bool
    is_weak: bool
    notes: List[str] = field(default_factory=list)


DEFAULT_PATTERNS_BY_SESSION = {
    TradingSession.ASIAN: {
        "preferred": ["OB", "SWEEP", "PRESSURE_ZONE", "HARMONIC_CYCLE"],
        "avoided": ["VOID_SCALP", "BREAKER", "WYCKOFF"],
    },
    TradingSession.LONDON_OPEN: {
        "preferred": ["FVG", "BREAKER", "VOID_SCALP", "BOS_ZONE"],
        "avoided": ["WYCKOFF", "PRESSURE_ZONE"],
    },
    TradingSession.LONDON_MID: {
        "preferred": ["OB", "SWEEP", "CYCLE", "PRICE_INTERACTION"],
        "avoided": ["VOID_SCALP", "BREAKER"],
    },
    TradingSession.NY_OPEN: {
        "preferred": ["FVG", "BREAKER", "BOS_ZONE", "VOID_SCALP"],
        "avoided": ["WYCKOFF", "INTERVAL_POINT"],
    },
    TradingSession.LONDON_NY_OVERLAP: {
        "preferred": ["FVG", "BOS_ZONE", "CYCLE", "BREAKER", "SWEEP"],
        "avoided": [],
    },
    TradingSession.NY_AFTERNOON: {
        "preferred": ["OB", "SWEEP", "PRESSURE_ZONE", "HARMONIC_CYCLE"],
        "avoided": ["FVG", "VOID_SCALP"],
    },
    TradingSession.CLOSE: {
        "preferred": [],
        "avoided": ["FVG", "VOID_SCALP", "BREAKER", "BOS_ZONE"],
    },
}


class SessionProfiler:
    def __init__(self, asian_volatility_reduction: float = 0.7,
                 london_peak_boost: float = 1.2,
                 ny_overlap_boost: float = 1.3,
                 close_session_penalty: float = 0.3):
        self._volatility_cache: Dict[str, Dict[int, float]] = {}
        self._session_volatility_mult = {
            TradingSession.ASIAN: asian_volatility_reduction,
            TradingSession.LONDON_OPEN: london_peak_boost,
            TradingSession.LONDON_MID: 1.0,
            TradingSession.NY_OPEN: ny_overlap_boost,
            TradingSession.LONDON_NY_OVERLAP: ny_overlap_boost * 1.1,
            TradingSession.NY_AFTERNOON: 1.0,
            TradingSession.CLOSE: close_session_penalty,
        }

    def get_session(self, dt: Optional[datetime] = None) -> TradingSession:
        if dt is None:
            dt = datetime.utcnow()
        hour = dt.hour

        if 0 <= hour < 7:
            return TradingSession.ASIAN
        if 7 <= hour < 9:
            return TradingSession.LONDON_OPEN
        if 9 <= hour < 12:
            return TradingSession.LONDON_MID
        if 12 <= hour < 16:
            if 12 <= hour < 14:
                return TradingSession.NY_OPEN
            return TradingSession.LONDON_NY_OVERLAP
        if 16 <= hour < 21:
            return TradingSession.NY_AFTERNOON
        return TradingSession.CLOSE

    def is_peak_time(self, session: TradingSession, dt: Optional[datetime] = None) -> bool:
        if dt is None:
            dt = datetime.utcnow()
        hour = dt.hour
        for start, end in SESSION_PEAK_WINDOWS.get(session, []):
            if start <= hour < end:
                return True
        return False

    def is_weak_time(self, session: TradingSession, dt: Optional[datetime] = None) -> bool:
        if dt is None:
            dt = datetime.utcnow()
        hour = dt.hour
        for start, end in SESSION_WEAK_WINDOWS.get(session, []):
            if start <= hour < end:
                return True
        return False

    def profile(self, symbol: str, ltf_df: Optional[pd.DataFrame] = None,
                 dt: Optional[datetime] = None) -> SessionProfile:
        session = self.get_session(dt)
        is_peak = self.is_peak_time(session, dt)
        is_weak = self.is_weak_time(session, dt)

        volatility_pct = self._estimate_volatility(symbol, session, ltf_df)

        preferred = DEFAULT_PATTERNS_BY_SESSION[session]["preferred"]
        avoided = DEFAULT_PATTERNS_BY_SESSION[session]["avoided"]

        direction_bias = self._get_session_bias(session)

        if is_peak:
            sl_adj = 0.9
            tp_adj = 1.2
            vol_adj = 1.2
            aggressiveness = "aggressive"
        elif is_weak:
            sl_adj = 1.2
            tp_adj = 0.8
            vol_adj = 0.5
            aggressiveness = "conservative"
        else:
            sl_adj = 1.0
            tp_adj = 1.0
            vol_adj = 1.0
            aggressiveness = "moderate"

        notes = [
            f"Sesión: {session.value}",
            f"Volatilidad relativa: {volatility_pct:.0%}",
            f"{'HORA PICO' if is_peak else 'HORA DÉBIL' if is_weak else 'Normal'}",
            f"Patrones preferidos: {', '.join(preferred) if preferred else 'ninguno'}",
            f"Patrones evitados: {', '.join(avoided) if avoided else 'ninguno'}",
        ]
        if direction_bias:
            notes.append(f"Sesgo direccional: {direction_bias}")

        return SessionProfile(
            session=session,
            label=session.value,
            volatility_pct=volatility_pct,
            direction_bias=direction_bias,
            preferred_patterns=preferred,
            avoided_patterns=avoided,
            sl_adjustment=sl_adj,
            tp_adjustment=tp_adj,
            volume_adjustment=vol_adj,
            aggressiveness=aggressiveness,
            is_peak=is_peak,
            is_weak=is_weak,
            notes=notes,
        )

    def _estimate_volatility(self, symbol: str, session: TradingSession,
                              ltf_df: Optional[pd.DataFrame]) -> float:
        session_mult = self._session_volatility_mult.get(session, 1.0)
        if ltf_df is not None and len(ltf_df) >= 14:
            atr_val = float(atr(ltf_df, 14).iloc[-1])
            avg_price = float(ltf_df["close"].mean())
            current_vol = atr_val / avg_price if avg_price > 0 else 0
            current_vol *= session_mult

            cache_key = f"{symbol}_{session.value}"
            if cache_key not in self._volatility_cache:
                self._volatility_cache[cache_key] = current_vol
            else:
                cached = self._volatility_cache[cache_key]
                alpha = 0.3
                self._volatility_cache[cache_key] = cached * (1 - alpha) + current_vol * alpha

            return self._volatility_cache[cache_key]

        session_vol_map = {
            TradingSession.ASIAN: 0.003 * session_mult,
            TradingSession.LONDON_OPEN: 0.008 * session_mult,
            TradingSession.LONDON_MID: 0.005 * session_mult,
            TradingSession.NY_OPEN: 0.010 * session_mult,
            TradingSession.LONDON_NY_OVERLAP: 0.012 * session_mult,
            TradingSession.NY_AFTERNOON: 0.006 * session_mult,
            TradingSession.CLOSE: 0.002 * session_mult,
        }
        return session_vol_map.get(session, 0.005 * session_mult)

    def _get_session_bias(self, session: TradingSession) -> Optional[str]:
        bias_map = {
            TradingSession.ASIAN: None,
            TradingSession.LONDON_OPEN: None,
            TradingSession.LONDON_MID: None,
            TradingSession.NY_OPEN: None,
            TradingSession.LONDON_NY_OVERLAP: None,
            TradingSession.NY_AFTERNOON: None,
            TradingSession.CLOSE: None,
        }
        return bias_map.get(session)

    def adjust_decision(self, session_profile: SessionProfile,
                         conviction: float) -> Tuple[float, float, float, str]:
        session_mult = self._session_volatility_mult.get(session_profile.session, 1.0)
        vol_adj = session_profile.volume_adjustment * session_mult
        sl_adj = session_profile.sl_adjustment
        tp_adj = session_profile.tp_adjustment

        if session_profile.is_peak:
            conviction_adj = min(1.0, conviction * 1.2)
        elif session_profile.is_weak:
            conviction_adj = conviction * 0.8
        else:
            conviction_adj = conviction

        return vol_adj, sl_adj, tp_adj, session_profile.aggressiveness

    def get_preferred_patterns(self, dt: Optional[datetime] = None) -> List[str]:
        session = self.get_session(dt)
        return DEFAULT_PATTERNS_BY_SESSION[session]["preferred"]

    def get_avoided_patterns(self, dt: Optional[datetime] = None) -> List[str]:
        session = self.get_session(dt)
        return DEFAULT_PATTERNS_BY_SESSION[session]["avoided"]

"""Feature engineering for neural network training and inference"""
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REGIMES = [
    "STRONG_TREND_BULLISH", "STRONG_TREND_BEARISH", "RANGING",
    "HIGH_VOLATILITY", "LOW_VOLATILITY", "TRANSITION",
]

SESSIONS = [
    "ASIAN", "LONDON_OPEN", "LONDON_MID", "NY_OPEN",
    "LONDON_NY_OVERLAP", "NY_AFTERNOON", "CLOSE",
]

PATTERN_GROUPS = [
    "FVG", "OB", "BREAKER", "SWEEP", "WYCKOFF",
    "SEQUENCE", "BOS_ZONE", "ESTABLISHMENT", "OTHER", "NONE",
]

DIRECTIONS = ["BUY", "SELL"]

FEATURE_NAMES = [
    "score_norm", "conviction", "regime_confidence",
] + [f"dir_{d}" for d in DIRECTIONS] \
  + [f"regime_{r}" for r in REGIMES] \
  + [f"session_{s}" for s in SESSIONS] \
  + [f"pattern_{g}" for g in PATTERN_GROUPS]


def _pattern_to_group(pattern_name: Optional[str]) -> str:
    if not pattern_name:
        return "NONE"
    p = pattern_name.upper()
    if "FVG" in p or "VOID_SCALP" in p:
        return "FVG"
    if "OB_" in p:
        return "OB"
    if "BREAKER" in p:
        return "BREAKER"
    if "SWEEP" in p or "CYCLE" in p:
        return "SWEEP"
    if "SPRING" in p or "UTAD" in p or "SOS" in p or "SOW" in p:
        return "WYCKOFF"
    if "SEQUENCE" in p:
        return "SEQUENCE"
    if "BOS_ZONE" in p:
        return "BOS_ZONE"
    if "ESTABLISHMENT" in p:
        return "ESTABLISHMENT"
    return "OTHER"


def _one_hot(value: str, categories: List[str]) -> List[float]:
    return [1.0 if value == c else 0.0 for c in categories]


def record_to_features(
    score: float,
    conviction: float,
    regime: str,
    session: str,
    primary_pattern: Optional[str],
    direction: str,
    regime_confidence: float = 0.0,
) -> np.ndarray:
    feats = [
        score / 100.0,
        conviction,
        regime_confidence,
    ]
    feats += _one_hot(direction, DIRECTIONS)
    feats += _one_hot(regime, REGIMES)
    feats += _one_hot(session, SESSIONS)
    feats += _one_hot(_pattern_to_group(primary_pattern), PATTERN_GROUPS)
    return np.array(feats, dtype=np.float32)


def load_trade_records(db_path: Path, min_trades: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(
        "SELECT score, conviction, regime, session, primary_pattern, "
        "direction, regime_confidence, profit FROM trade_records "
        "WHERE profit IS NOT NULL AND profit != 0",
        conn,
    )
    conn.close()

    if len(df) < min_trades:
        raise ValueError(f"Not enough trades ({len(df)} < {min_trades})")

    features = []
    targets = []
    for _, row in df.iterrows():
        try:
            f = record_to_features(
                score=row["score"],
                conviction=row["conviction"],
                regime=row["regime"],
                session=row["session"],
                primary_pattern=row["primary_pattern"],
                direction=row["direction"],
                regime_confidence=row["regime_confidence"] or 0.0,
            )
            features.append(f)
            targets.append(1.0 if row["profit"] > 0 else 0.0)
        except Exception:
            continue

    X = np.array(features, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X = (X - mean) / std

    return X, y, mean, std


FEATURE_DIM = len(FEATURE_NAMES)

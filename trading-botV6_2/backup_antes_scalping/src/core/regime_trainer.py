"""ML-based Market Regime Trainer
Trains a Random Forest classifier to detect market regimes
using multi-feature technical analysis.
"""
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.helpers import atr, find_swing_points

logger = logging.getLogger(__name__)

REGIME_LABELS = {
    0: "STRONG_TREND_BULLISH",
    1: "STRONG_TREND_BEARISH",
    2: "RANGING",
    3: "HIGH_VOLATILITY",
    4: "LOW_VOLATILITY",
    5: "TRANSITION",
}
LABEL_TO_REGIME = {v: k for k, v in REGIME_LABELS.items()}

REGIME_PATTERN_MULTIPLIERS = {
    "FVG": {0: 1.8, 1: 1.8, 2: 0.8, 3: 1.3, 4: 0.7, 5: 0.9},
    "OB": {0: 1.2, 1: 1.2, 2: 1.5, 3: 1.1, 4: 1.3, 5: 1.4},
    "BREAKER": {0: 1.3, 1: 1.3, 2: 0.8, 3: 0.6, 4: 1.1, 5: 1.5},
    "SWEEP": {0: 1.5, 1: 1.5, 2: 1.8, 3: 1.2, 4: 1.4, 5: 1.6},
    "WYCKOFF": {0: 0.6, 1: 0.6, 2: 1.8, 3: 0.5, 4: 1.6, 5: 1.5},
    "VOID_SCALP": {0: 1.6, 1: 1.6, 2: 0.7, 3: 1.5, 4: 0.6, 5: 0.8},
    "BOS_ZONE": {0: 1.7, 1: 1.7, 2: 0.8, 3: 1.4, 4: 0.8, 5: 0.7},
    "CYCLE": {0: 1.4, 1: 1.4, 2: 0.9, 3: 1.2, 4: 0.9, 5: 1.3},
    "SEQUENCE": {0: 1.3, 1: 1.3, 2: 0.8, 3: 1.1, 4: 0.6, 5: 0.8},
    "INTERVAL_POINT": {0: 1.1, 1: 1.1, 2: 1.6, 3: 0.8, 4: 1.5, 5: 1.4},
    "PRICE_INTERACTION": {0: 1.2, 1: 1.2, 2: 1.4, 3: 0.9, 4: 1.3, 5: 1.3},
    "HARMONIC_CYCLE": {0: 1.5, 1: 1.5, 2: 0.9, 3: 1.2, 4: 0.8, 5: 0.9},
    "PRESSURE_ZONE": {0: 1.3, 1: 1.3, 2: 1.7, 3: 0.7, 4: 1.5, 5: 1.6},
    "TRB": {0: 0.5, 1: 0.5, 2: 1.9, 3: 0.4, 4: 1.7, 5: 1.2},
}


def compute_ema(arr: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1)
    result = np.zeros_like(arr)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = arr[i] * alpha + result[i - 1] * (1 - alpha)
    return result


def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    diff = np.diff(close)
    gains = np.where(diff > 0, diff, 0)
    losses = np.where(diff < 0, -diff, 0)
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period
    rs = avg_gain / np.where(avg_loss > 0, avg_loss, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_adx(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ),
    )
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    tr_ema = compute_ema(tr, period)
    plus_ema = compute_ema(plus_dm, period)
    minus_ema = compute_ema(minus_dm, period)
    plus_di = 100 * plus_ema / np.where(tr_ema > 0, tr_ema, 1)
    minus_di = 100 * minus_ema / np.where(tr_ema > 0, tr_ema, 1)
    dx = 100 * np.abs(plus_di - minus_di) / np.where(plus_di + minus_di > 0, plus_di + minus_di, 1)
    adx_vals = np.full_like(close, np.nan)
    if len(dx) >= period:
        adx_vals[period:] = np.concatenate([
            [np.mean(dx[:period])],
            [np.mean(dx[i - period + 1:i + 1]) for i in range(period, len(dx))],
        ])
    return adx_vals


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values
    n = len(df)

    features = pd.DataFrame(index=df.index)
    features["time"] = pd.to_datetime(df["time"])

    ema8 = compute_ema(close, 8)
    ema21 = compute_ema(close, 21)
    ema50 = compute_ema(close, 50)
    ema200 = compute_ema(close, 200) if n >= 200 else np.full(n, np.nan)

    features["ema8_21_ratio"] = ema8 / np.where(ema21 > 0, ema21, 1e-10)
    features["ema21_50_ratio"] = ema21 / np.where(ema50 > 0, ema50, 1e-10)
    if n >= 200:
        features["ema50_200_ratio"] = ema50 / np.where(ema200 > 0, ema200, 1e-10)
    else:
        features["ema50_200_ratio"] = np.nan

    features["price_to_ema8_pct"] = (close - ema8) / np.where(ema8 > 0, ema8, 1e-10)
    features["price_to_ema21_pct"] = (close - ema21) / np.where(ema21 > 0, ema21, 1e-10)

    atr_vals = atr(df, 14).values if len(df) >= 15 else np.full(n, np.nan)
    atr_sma = pd.Series(atr_vals).rolling(50, min_periods=10).mean().values
    features["atr_ratio"] = atr_vals / np.where(atr_sma > 0, atr_sma, 1e-10)

    atr_price = atr_vals / np.where(close > 0, close, 1e-10)
    features["atr_price_pct"] = atr_price

    adx_vals = compute_adx(df, 14)
    features["adx"] = adx_vals

    plus_di_vals = np.full(n, np.nan)
    minus_di_vals = np.full(n, np.nan)
    high_np, low_np, close_np = high, low, close
    tr_v = np.maximum(
        high_np[1:] - low_np[1:],
        np.maximum(np.abs(high_np[1:] - close_np[:-1]), np.abs(low_np[1:] - close_np[:-1])),
    )
    up = high_np[1:] - high_np[:-1]
    dn = low_np[:-1] - low_np[1:]
    pdm = np.where((up > dn) & (up > 0), up, 0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0)
    tr_e = compute_ema(tr_v, 14)
    pd_e = compute_ema(pdm, 14)
    md_e = compute_ema(mdm, 14)
    pdi = 100 * pd_e / np.where(tr_e > 0, tr_e, 1)
    mdi = 100 * md_e / np.where(tr_e > 0, tr_e, 1)
    features["plus_di"] = np.concatenate([[np.nan], pdi])
    features["minus_di"] = np.concatenate([[np.nan], mdi])
    di_sum = (features["plus_di"] + features["minus_di"]).values
    di_diff = (features["plus_di"] - features["minus_di"]).values
    features["di_cross"] = di_diff / np.where(di_sum > 0, di_sum, 1e-10)

    rsi_vals = compute_rsi(close, 14)
    features["rsi_14"] = rsi_vals / 100.0

    volume_sma = pd.Series(volume).rolling(20, min_periods=5).mean().values
    features["volume_ratio"] = volume / np.where(volume_sma > 0, volume_sma, 1e-10)

    sma50 = pd.Series(close).rolling(50, min_periods=10).mean().values
    sma200 = pd.Series(close).rolling(200, min_periods=40).mean().values if n >= 200 else np.full(n, np.nan)
    features["close_sma50_ratio"] = close / np.where(sma50 > 0, sma50, 1e-10)
    features["close_sma200_ratio"] = close / np.where(sma200 > 0, sma200, 1e-10)

    rolling_max = pd.Series(high).rolling(20, min_periods=5).max().values
    rolling_min = pd.Series(low).rolling(20, min_periods=5).min().values
    range_val = rolling_max - rolling_min
    features["price_range_position"] = (close - rolling_min) / np.where(range_val > 0, range_val, 1e-10)

    candle_range = high - low
    features["candle_range_atr_ratio"] = candle_range / np.where(atr_vals > 0, atr_vals, 1e-10)

    for period in [1, 3, 5, 10]:
        shifted = pd.Series(close).shift(period)
        features[f"price_change_{period}"] = (close - shifted.values) / np.where(shifted.values > 0, shifted.values, 1e-10)

    if len(df) >= 20:
        atr_shifted = pd.Series(atr_vals).shift(10)
        features["atr_change_10"] = atr_vals / np.where(atr_shifted.values > 0, atr_shifted.values, 1e-10)
    else:
        features["atr_change_10"] = np.nan

    try:
        timestamps = pd.to_datetime(df["time"])
        hour = timestamps.dt.hour.values
        features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    except (KeyError, AttributeError):
        features["hour_sin"] = 0
        features["hour_cos"] = 0

    try:
        highs_idx, lows_idx = find_swing_points(df, lookback=5)
        swing_high_count = np.zeros(n)
        swing_low_count = np.zeros(n)
        for idx in highs_idx:
            swing_high_count[idx] = 1
        for idx in lows_idx:
            swing_low_count[idx] = 1
        features["swing_high_count_10"] = pd.Series(swing_high_count).rolling(10, min_periods=3).sum().values
        features["swing_low_count_10"] = pd.Series(swing_low_count).rolling(10, min_periods=3).sum().values
    except Exception:
        features["swing_high_count_10"] = 0
        features["swing_low_count_10"] = 0

    feature_cols = [
        "ema8_21_ratio", "ema21_50_ratio", "ema50_200_ratio",
        "price_to_ema8_pct", "price_to_ema21_pct",
        "atr_ratio", "atr_price_pct", "adx",
        "plus_di", "minus_di", "di_cross",
        "rsi_14", "volume_ratio",
        "close_sma50_ratio", "close_sma200_ratio",
        "price_range_position", "candle_range_atr_ratio",
        "price_change_1", "price_change_3", "price_change_5", "price_change_10",
        "atr_change_10",
        "hour_sin", "hour_cos",
        "swing_high_count_10", "swing_low_count_10",
    ]

    return features[feature_cols]


def generate_labels(df: pd.DataFrame) -> np.ndarray:
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)

    labels = np.full(n, 5, dtype=int)

    atr_vals = atr(df, 14).values
    atr_sma = pd.Series(atr_vals).rolling(50, min_periods=10).mean().values
    atr_ratio = atr_vals / np.where(atr_sma > 0, atr_sma, 1e-10)

    adx_vals = compute_adx(df, 14)

    ema8 = compute_ema(close, 8)
    ema21 = compute_ema(close, 21)
    ema_slope = np.full(n, np.nan)
    if n >= 5:
        ema_slope[:5] = 0
        ema_slope[5:] = (ema8[5:] - ema8[:-5]) / np.where(ema8[:-5] > 0, ema8[:-5], 1e-10)

    trend_strength = np.full(n, 0.0)
    ema_dist = np.abs(ema8 - ema21) / np.where(close > 0, close, 1e-10)
    if n >= 50:
        rolling_vol = pd.Series(high - low).rolling(20, min_periods=5).std().values
        norm_vol = rolling_vol / np.where(rolling_vol.mean() > 0, rolling_vol.mean(), 1e-10)
    else:
        norm_vol = np.ones(n)

    for i in range(50, n):
        if np.isnan(atr_ratio[i]) or np.isnan(adx_vals[i]) or np.isnan(ema_slope[i]):
            labels[i] = 5
            continue

        is_compressed = atr_ratio[i] < 0.7
        is_expanding = atr_ratio[i] > 1.5

        if is_expanding and (np.isnan(adx_vals[i]) or adx_vals[i] < 25):
            labels[i] = 3
            continue

        strength = min(1.0, abs(ema_slope[i]) * 500 + ema_dist[i] * 200 + 0.0)
        if not np.isnan(adx_vals[i]) and adx_vals[i] >= 30 and strength >= 0.6:
            if ema_slope[i] > 0:
                labels[i] = 0
            else:
                labels[i] = 1
            continue

        if is_compressed:
            labels[i] = 4
            continue

        if not np.isnan(adx_vals[i]) and 20 <= adx_vals[i] < 30 and 0.7 <= atr_ratio[i] <= 1.3:
            labels[i] = 2
            continue

        labels[i] = 5

    return labels


class RegimeTrainer:
    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or Path(__file__).resolve().parent.parent.parent / "data" / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._model: Optional[RandomForestClassifier] = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_names: List[str] = []
        self._train_accuracy: float = 0.0
        self._test_accuracy: float = 0.0
        self._feature_importance: Dict[str, float] = {}

    def collect_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        features = compute_features(df)
        labels = generate_labels(df)
        valid = ~features.isna().any(axis=1).values & ~np.isnan(labels)
        features = features[valid]
        labels = labels[valid]
        return features, labels

    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        features_df, labels = self.collect_training_data(df)
        if len(features_df) < 1000:
            logger.warning(
                f"Solo {len(features_df)} muestras válidas de {len(df)} totales. "
                f"Se necesitan al menos 1000 para un modelo confiable."
            )
            return {"samples": len(features_df), "status": "insufficient_data"}

        self._feature_names = list(features_df.columns)
        X = features_df.values
        y = labels

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y,
        )

        self._scaler = StandardScaler()
        X_train_scaled = self._scaler.fit_transform(X_train)
        X_test_scaled = self._scaler.transform(X_test)

        self._model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        self._model.fit(X_train_scaled, y_train)

        train_acc = self._model.score(X_train_scaled, y_train)
        test_acc = self._model.score(X_test_scaled, y_test)
        self._train_accuracy = train_acc
        self._test_accuracy = test_acc

        importances = self._model.feature_importances_
        self._feature_importance = {
            name: float(imp)
            for name, imp in sorted(
                zip(self._feature_names, importances),
                key=lambda x: x[1], reverse=True,
            )
        }

        class_counts = pd.Series(y_train).value_counts().sort_index()
        label_dist = {}
        for label_id, name in REGIME_LABELS.items():
            label_dist[name] = int(class_counts.get(label_id, 0))

        result = {
            "samples": len(features_df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "train_accuracy": round(train_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "overfit_gap": round(train_acc - test_acc, 4),
            "label_distribution": label_dist,
            "feature_importance": self._feature_importance,
            "status": "success",
        }

        logger.info(f"RF Regime Model trained: {result['train_samples']} train / {result['test_samples']} test "
                    f"→ train_acc={train_acc:.1%} test_acc={test_acc:.1%}")
        logger.info(f"Top 5 features: {list(self._feature_importance.keys())[:5]}")
        logger.info(f"Label distribution: {label_dist}")

        return result

    def save_model(self, symbol: str) -> Path:
        if self._model is None or self._scaler is None:
            raise ValueError("No trained model to save")
        path = self.model_dir / f"regime_rf_{symbol}.pkl"
        data = {
            "model": self._model,
            "scaler": self._scaler,
            "feature_names": self._feature_names,
            "train_accuracy": self._train_accuracy,
            "test_accuracy": self._test_accuracy,
            "feature_importance": self._feature_importance,
            "trained_at": datetime.now().isoformat(),
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Model saved: {path}")
        return path

    @staticmethod
    def load_model(path: Path) -> Optional[Dict]:
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            logger.warning(f"Failed to load model {path}: {e}")
            return None

    @staticmethod
    def ml_predict(
        features_df: pd.DataFrame,
        model_data: Dict,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        model: RandomForestClassifier = model_data["model"]
        scaler: StandardScaler = model_data["scaler"]
        feature_names: List[str] = model_data["feature_names"]

        missing = [c for c in feature_names if c not in features_df.columns]
        if missing:
            for c in missing:
                features_df[c] = 0.0

        X = features_df[feature_names].values
        X_scaled = scaler.transform(X)

        nan_rows = np.isnan(X_scaled).any(axis=1)
        if nan_rows.any():
            X_scaled[nan_rows] = 0.0

        preds = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)
        confidences = np.max(probs, axis=1)

        return preds, confidences, probs

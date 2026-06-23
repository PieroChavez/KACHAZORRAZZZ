"""Training pipeline - loads trade history, trains NN, saves model"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.neural.model import NeuralNetwork
from src.neural.features import load_trade_records, FEATURE_DIM

logger = logging.getLogger(__name__)

BASE_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"


def _model_path(symbol: str) -> Path:
    model_dir = BASE_MODEL_DIR / symbol
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / "trade_predictor.npz"


def train(
    symbol: str = "",
    db_path: Optional[Path] = None,
    hidden_layers: list = None,
    epochs: int = 500,
    lr: float = 0.01,
    val_split: float = 0.2,
    force: bool = False,
) -> NeuralNetwork:
    if db_path is None:
        db_path = Path(__file__).resolve().parent.parent.parent / "data" / "db" / symbol / "meta_learning.db"

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    min_trades = 1 if force else 10
    X, y, mean, std = load_trade_records(db_path, min_trades=min_trades)
    logger.info(f"Loaded {len(X)} trade records, {int(y.sum())} wins / {int((1 - y).sum())} losses")

    if hidden_layers is None:
        n_features = X.shape[1]
        hidden_layers = [max(16, n_features * 2), max(8, n_features)]

    layers = [X.shape[1]] + hidden_layers + [1]
    logger.info(f"Network architecture: {layers}")

    nn = NeuralNetwork(layers, learning_rate=lr)

    if len(X) >= 3:
        split = int(len(X) * (1 - val_split))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        val_data = (X_val, y_val)
    else:
        X_train, y_train = X, y
        val_data = None

    nn.train(
        X_train, y_train,
        epochs=epochs,
        batch_size=min(32, len(X_train)),
        validation_data=val_data,
        patience=15,
        verbose=True,
    )

    train_acc = nn.accuracy(y_train, nn.forward(X_train))
    logger.info(f"Train accuracy: {train_acc:.2%}")
    if val_data:
        val_acc = nn.accuracy(y_val, nn.forward(X_val))
        logger.info(f"Validation accuracy: {val_acc:.2%}")

    model_path = _model_path(symbol)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    nn.save(model_path)

    with np.load(model_path, allow_pickle=True) as data:
        arrays = dict(data)
    arrays["scaler_mean"] = np.array(mean)
    arrays["scaler_std"] = np.array(std)
    np.savez_compressed(model_path, **arrays)

    logger.info(f"Model + scaler saved to {model_path}")

    return nn


def load_model(symbol: str = "") -> Optional[NeuralNetwork]:
    model_path = _model_path(symbol)
    if not model_path.exists():
        logger.warning(f"No trained model found at {model_path}")
        return None
    return NeuralNetwork.load(model_path)


def load_scaler(symbol: str = "") -> Optional[dict]:
    model_path = _model_path(symbol)
    if not model_path.exists():
        return None
    with np.load(model_path, allow_pickle=True) as data:
        if "scaler_mean" not in data:
            return None
        scaler = {
            "mean": data["scaler_mean"].tolist(),
            "std": data["scaler_std"].tolist(),
        }
    return scaler


def get_prediction(features: np.ndarray, symbol: str = "") -> Optional[float]:
    nn = load_model(symbol)
    scaler = load_scaler(symbol)
    if nn is None or scaler is None:
        return None

    features_norm = (features - np.array(scaler["mean"])) / np.array(scaler["std"])
    prob = nn.predict_proba(features_norm.reshape(1, -1))
    return float(prob[0])

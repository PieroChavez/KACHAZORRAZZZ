"""Neural Network Model - Feedforward MLP with backpropagation (NumPy only)"""
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class NeuralNetwork:
    def __init__(self, layers: List[int], learning_rate: float = 0.01):
        self.layers = layers
        self.lr = learning_rate
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        for i in range(len(layers) - 1):
            limit = np.sqrt(6.0 / (layers[i] + layers[i + 1]))
            w = np.random.uniform(-limit, limit, (layers[i], layers[i + 1]))
            b = np.zeros((1, layers[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

        self._activations: List[np.ndarray] = []
        self._z_values: List[np.ndarray] = []

    # --- Activations ---

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def _sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        return x * (1.0 - x)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def _relu_derivative(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    # --- Forward ---

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._activations = [X]
        self._z_values = []

        a = X
        for i in range(len(self.weights)):
            z = a @ self.weights[i] + self.biases[i]
            self._z_values.append(z)
            if i < len(self.weights) - 1:
                a = self._relu(z)
            else:
                a = self._sigmoid(z)
            self._activations.append(a)
        return a

    # --- Backward ---

    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray):
        m = X.shape[0]
        y = y.reshape(-1, 1)

        delta = output - y
        deltas = [delta]

        for i in range(len(self.weights) - 1, 0, -1):
            delta = (deltas[0] @ self.weights[i].T) * self._relu_derivative(self._activations[i])
            deltas.insert(0, delta)

        for i in range(len(self.weights)):
            dW = self._activations[i].T @ deltas[i] / m
            db = np.sum(deltas[i], axis=0, keepdims=True) / m
            self.weights[i] -= self.lr * dW
            self.biases[i] -= self.lr * db

    # --- Training ---

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 1000, batch_size: int = 32,
              validation_data: Optional[tuple] = None,
              patience: int = 20, verbose: bool = True):
        m = X.shape[0]
        best_loss = float("inf")
        best_weights = None
        best_biases = None
        stale = 0

        for epoch in range(1, epochs + 1):
            idx = np.random.permutation(m)
            X_s, y_s = X[idx], y[idx]

            for start in range(0, m, batch_size):
                end = min(start + batch_size, m)
                out = self.forward(X_s[start:end])
                self.backward(X_s[start:end], y_s[start:end], out)

            if epoch % 25 == 0:
                train_loss = self.binary_cross_entropy(y, self.forward(X))
                train_acc = self.accuracy(y, self.forward(X))

                if validation_data:
                    X_val, y_val = validation_data
                    val_loss = self.binary_cross_entropy(y_val, self.forward(X_val))
                    val_acc = self.accuracy(y_val, self.forward(X_val))
                    current_loss = val_loss

                    if verbose:
                        logger.info(
                            f"Epoch {epoch:4d}/{epochs}  "
                            f"train_loss={train_loss:.4f}  train_acc={train_acc:.2%}  "
                            f"val_loss={val_loss:.4f}  val_acc={val_acc:.2%}"
                        )
                else:
                    current_loss = train_loss
                    if verbose:
                        logger.info(
                            f"Epoch {epoch:4d}/{epochs}  "
                            f"loss={train_loss:.4f}  acc={train_acc:.2%}"
                        )

                if current_loss < best_loss:
                    best_loss = current_loss
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

        if best_weights is not None:
            self.weights = best_weights
            self.biases = best_biases

    # --- Predict ---

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X).ravel()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    # --- Loss & Metrics ---

    def binary_cross_entropy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))

    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_pred.ravel() >= 0.5).astype(int) == y_true.ravel()))

    # --- Persistence ---

    def save(self, path: Path):
        weights_arrays = {f"weight_{i}": w for i, w in enumerate(self.weights)}
        biases_arrays = {f"bias_{i}": b for i, b in enumerate(self.biases)}
        metadata = np.array([len(self.layers), *self.layers, self.lr])
        np.savez_compressed(
            path,
            metadata=metadata,
            **weights_arrays,
            **biases_arrays,
        )
        logger.info(f"Model saved to {path} (npz)")

    @classmethod
    def load(cls, path: Path) -> "NeuralNetwork":
        data = np.load(path, allow_pickle=True)
        meta = data["metadata"]
        layers = [int(x) for x in meta[1:1 + int(meta[0])]]
        lr = float(meta[-1])
        nn = cls(layers, lr)
        nn.weights = [data[f"weight_{i}"] for i in range(len(layers) - 1)]
        nn.biases = [data[f"bias_{i}"] for i in range(len(layers) - 1)]
        data.close()
        logger.info(f"Model loaded from {path} (npz)")
        return nn

#!/usr/bin/env python
"""Train the neural network from historical trade data.

Usage:
    python scripts/train_neural.py                    # train with defaults
    python scripts/train_neural.py --epochs 1000 --lr 0.005
    python scripts/train_neural.py --layers 32 16 8   # custom hidden layers
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.neural.trainer import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Train trade prediction neural network")
    parser.add_argument("--db", type=str, default=None, help="Path to meta_learning.db")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Hidden layer sizes (e.g. --layers 32 16)")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else None

    nn = train(
        db_path=db_path,
        hidden_layers=args.layers,
        epochs=args.epochs,
        lr=args.lr,
        val_split=args.val_split,
    )

    print(f"\nModel saved to models/trade_predictor.json")
    print(f"Layers: {nn.layers}")
    print(f"Learning rate: {nn.lr}")


if __name__ == "__main__":
    main()

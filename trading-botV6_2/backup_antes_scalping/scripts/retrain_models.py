"""Retrain ML regime models and hot-reload into running bot"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from loguru import logger

from src.core.regime_trainer import RegimeTrainer
from src.adapters.mt5_client import MT5Client
from src.adapters.mt5_api import LiveMT5API

CONFIG_SYMBOLS = ["XAUUSD..", "XAGUSD..", "BTCUSD..", "USTEC_x100"]
MODEL_DIR = Path(__file__).resolve().parent.parent / "data" / "models"
TIMEFRAME_M5 = 5
NUM_CANDLES = 5000


def resolve(mt5: MT5Client, cfg_sym: str) -> str:
    base = cfg_sym.rstrip("m.").split("_")[0]
    resolved = mt5.resolve_symbol(base)
    if not resolved:
        raise RuntimeError(f"Cannot resolve {cfg_sym} (base={base})")
    if resolved != cfg_sym:
        logger.info(f"  {cfg_sym} -> {resolved}")
    return resolved


def fetch_data(mt5: MT5Client, resolved_sym: str, cfg_sym: str) -> pd.DataFrame:
    for attempt in range(3):
        raw = mt5.get_candles(resolved_sym, timeframe=TIMEFRAME_M5, count=NUM_CANDLES)
        if raw and len(raw) >= 2000:
            records = []
            for c in raw:
                records.append({
                    "time": c.timestamp,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                })
            df = pd.DataFrame(records)
            df["time"] = pd.to_datetime(df["time"])
            logger.info(f"[{cfg_sym}] Loaded {len(df)} M5 candles")
            return df
        logger.warning(f"[{cfg_sym}] Attempt {attempt+1}: got {len(raw) if raw else 0}/{NUM_CANDLES}")
        time.sleep(2)
    raise RuntimeError(f"Could not fetch enough data for {cfg_sym}")


def main():
    mt5 = MT5Client(api=LiveMT5API())
    if not mt5.connect():
        logger.error("Cannot connect to MT5")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer = RegimeTrainer(MODEL_DIR)

    resolved_map = {}
    for cfg_sym in CONFIG_SYMBOLS:
        try:
            resolved_map[cfg_sym] = resolve(mt5, cfg_sym)
        except RuntimeError as e:
            logger.error(e)
            return

    succeeded = []
    for cfg_sym, resolved_sym in resolved_map.items():
        try:
            logger.info(f"[{cfg_sym}] Fetching data via {resolved_sym}...")
            df = fetch_data(mt5, resolved_sym, cfg_sym)
            logger.info(f"[{cfg_sym}] Training model...")
            result = trainer.train(df)
            if result.get("status") != "success":
                logger.warning(f"[{cfg_sym}] Training {result.get('status')}: {result.get('samples', 0)} samples")
                continue
            acc = result.get("test_accuracy", 0)
            logger.info(f"[{cfg_sym}] Test acc={acc:.1%}, samples={result.get('samples', 0)}")
            trainer.save_model(resolved_sym)
            succeeded.append(cfg_sym)
        except Exception as e:
            logger.error(f"[{cfg_sym}] Failed: {e}")

    logger.info(f"=== Retrained {len(succeeded)}/{len(CONFIG_SYMBOLS)} models: {succeeded} ===")
    logger.info("Models saved to disk.")
    logger.info("The running bot will auto-detect the new models within a few seconds (hot-reload).")


if __name__ == "__main__":
    main()

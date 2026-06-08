"""SMC Trading Bot — Main Entry Point
Bot de trading para MT5 usando SMC fractal cascade
con detector de velocidad/acumulación de mercado.
"""
import json
import signal
import sys
import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict
from loguru import logger

_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))
del _proj_root

from src.adapters.mt5_client import MT5Client
from src.core.multi_timeframe import MultiTimeframeFetcher
from src.core.market_velocity import MarketVelocityDetector, VelocityResult
from src.utils.state_persistence import StatePersistence
from src.learning.meta_learner import MetaLearner
from src.scheduler.timeframe_scheduler import TimeframeScheduler
from src.strategies.fractal_cascade import FractalCascadeStrategy
import pandas as pd


ACCUMULATION_SKIP_SCORE = 0.80


def setup_logging():
    logger.remove()
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.add(log_dir / "trading_bot_{time}.log", rotation="00:00", retention="7 days",
              level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    logger.add(sys.stderr, level="INFO")


def load_config(config_dir: Path) -> dict:
    with open(config_dir / "broker.json") as f:
        broker = json.load(f)
    with open(config_dir / "strategy.json") as f:
        strategy = json.load(f)
    with open(config_dir / "risk.json") as f:
        risk = json.load(f)
    return {"broker": broker, "strategy": strategy, "risk": risk}


def candles_to_dataframe(candles: list) -> pd.DataFrame:
    records = []
    for c in candles:
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
    return df


class TradingBot:
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.config = load_config(config_dir)
        setup_logging()

        import os
        broker_config = self.config["broker"]["mt5"]
        self.mt5 = MT5Client(
            login=os.environ.get("MT5_LOGIN") or broker_config.get("login"),
            password=os.environ.get("MT5_PASSWORD") or broker_config.get("password"),
            server=os.environ.get("MT5_SERVER") or broker_config.get("server"),
            path=os.environ.get("MT5_PATH") or broker_config.get("path"),
        )

        self.fetcher = MultiTimeframeFetcher(self.mt5)
        str_cfg = self.config["strategy"]
        self.active_symbols = str_cfg.get("active_symbols", ["XAUUSDm"])

        self.symbols = {}
        for sym in self.active_symbols:
            sym_cfg = str_cfg.get("symbols", {}).get(sym, {})
            engine = FractalCascadeStrategy(sym, self.mt5, self.fetcher)
            self.symbols[sym] = {
                "engine": engine,
                "last_trade_time": None,
            }

        self.mt5.connect()

        self.scheduler = TimeframeScheduler(self.fetcher, self.active_symbols[0])

        self.state_persistence: Dict[str, StatePersistence] = {}
        self.meta_learner: Dict[str, MetaLearner] = {}
        for sym in self.active_symbols:
            self.state_persistence[sym] = StatePersistence(sym)
            self.meta_learner[sym] = MetaLearner(sym)

        self.velocity_detector = MarketVelocityDetector()

        self.running = False
        self.start_time = 0.0
        self._last_meta_analysis: Dict[str, float] = {}
        self._meta_analysis_interval = 14400  # cada 4 horas

    async def _initialize_state(self):
        for sym in self.active_symbols:
            await self.state_persistence[sym].initialize()

    async def _save_state_periodic(self):
        for sym in self.active_symbols:
            await self.state_persistence[sym].save_daily_state(
                daily_loss=0.0,
                trades_count=0,
            )

    def start(self, max_duration: int = 0):
        logger.info("=" * 50)
        logger.info("SMC Fractal Cascade Bot starting...")
        logger.info(f"Symbols: {', '.join(self.active_symbols)}")
        if max_duration:
            logger.info(f"Max duration: {max_duration // 60} min")
        logger.info("=" * 50)

        if not self.mt5.connect():
            logger.error("Failed to connect to MT5. Exiting.")
            return

        for sym in self.active_symbols:
            self.fetcher.init_historical(sym, count=5000)

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.running = True
        self.start_time = time.time()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._initialize_state())

        self._auto_train_model()

        self.scheduler.add_callback(self._on_new_candle)
        self.scheduler.start()
        self._evaluate()

        logger.info("Bot running. Press Ctrl+C to stop.")

        while self.running:
            self.loop.run_until_complete(asyncio.sleep(
                self.config["strategy"].get("loop_sleep_seconds", 5)
            ))
            self.loop.run_until_complete(self._save_state_periodic())

            self._manage_positions()

            for sym in self.active_symbols:
                last_time = self._last_meta_analysis.get(sym, 0)
                if time.time() - last_time > self._meta_analysis_interval:
                    self._last_meta_analysis[sym] = time.time()
                    try:
                        meta_result = self.meta_learner[sym].analyze_performance()
                        if meta_result.get("analyzed"):
                            logger.info(
                                f"[{sym}] Meta-Learning: {len(meta_result.get('adjustments', []))} ajustes"
                            )
                            for adj in meta_result["adjustments"]:
                                logger.info(f"  Ajuste: {adj}")
                    except Exception as e:
                        logger.warning(f"[{sym}] Meta-Learning analysis error: {e}")

            if max_duration and (time.time() - self.start_time) >= max_duration:
                logger.info(f"Max duration ({max_duration // 60} min) reached, stopping.")
                break

    def stop(self):
        logger.info("Stopping bot...")
        self.running = False
        self.scheduler.stop()
        if hasattr(self, 'loop') and self.loop.is_running():
            self.loop.run_until_complete(self._save_state_periodic())
            for sym in self.active_symbols:
                self.loop.run_until_complete(self.state_persistence[sym].close())
        self.mt5.disconnect()
        logger.info("Bot stopped.")

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}")
        self.stop()
        sys.exit(0)

    def _manage_positions(self):
        for sym, sym_data in self.symbols.items():
            try:
                sym_data["engine"].manage_orders()
            except Exception as e:
                logger.warning(f"[{sym}] Error gestionando trailing: {e}")

    def _on_new_candle(self, timeframe: str, candle_time: datetime):
        logger.info(f"New {timeframe} candle at {candle_time}")
        if timeframe == "5min":
            self._evaluate()

    def _auto_train_model(self):
        try:
            from src.neural.trainer import train
            import sqlite3
            for sym in self.active_symbols:
                db_path = Path(__file__).parent.parent / "data" / "db" / sym / "meta_learning.db"
                if not db_path.exists():
                    logger.info(f"[{sym}] No meta_learning.db found, skipping auto-train")
                    continue
                conn = sqlite3.connect(str(db_path))
                count = conn.execute(
                    "SELECT COUNT(*) FROM trade_records WHERE profit IS NOT NULL AND profit != 0"
                ).fetchone()[0]
                conn.close()
                if count < 3:
                    logger.info(f"[{sym}] Only {count} trades with profit, need at least 3 for training")
                    continue
                logger.info(f"[{sym}] Auto-training neural model with {count} trades...")
                train(symbol=sym, db_path=db_path, epochs=200, lr=0.005, force=True)
                logger.info(f"[{sym}] Auto-training complete")
        except Exception as e:
            logger.warning(f"Auto-train skipped: {e}")

    def _evaluate(self):
        logger.info("=" * 40)
        logger.info(f"Fractal Cascade Evaluation at {datetime.now()}")
        logger.info("=" * 40)

        for sym, sym_data in self.symbols.items():
            try:
                timeframes = self.fetcher.get_dataframes(sym, count=300)
                if len(timeframes) < 3:
                    continue

                ltf_df = None
                for tf in ["5min"]:
                    df = timeframes.get(tf)
                    if df is not None and len(df) >= 50:
                        ltf_df = df
                        break
                velocity: Optional[VelocityResult] = None
                if ltf_df is not None:
                    velocity = self.velocity_detector.detect(ltf_df)
                    sym_data["velocity"] = velocity
                    if velocity.regime == "ACCUMULATION":
                        logger.info(
                            f"[{sym}] Mercado en ACUMULACIÓN "
                            f"(score={velocity.accumulation_score:.0%}, "
                            f"ATR ratio={velocity.atr_ratio:.2f})"
                        )

                    if velocity.regime == "EXPANSION":
                        logger.info(
                            f"[{sym}] Mercado en EXPANSIÓN "
                            f"(momentum={velocity.momentum:+.2f}%, "
                            f"ATR ratio={velocity.atr_ratio:.2f})"
                        )

                now = datetime.now(timezone.utc).replace(tzinfo=None)
                skip_entries = (velocity is not None
                               and velocity.regime == "ACCUMULATION"
                               and velocity.accumulation_score >= ACCUMULATION_SKIP_SCORE)
                sym_data["engine"].evaluate(timeframes, now, skip_entries=skip_entries)
                status = sym_data["engine"].get_status()
                logger.info(
                    f"[{sym}] Cazador: {status.get('active_fractals', 0)} fractales, "
                    f"{status.get('alerts', 0)} alertas, "
                    f"{status.get('active_packs', 0)} packs activos"
                )

            except Exception as e:
                logger.exception(f"[{sym}] Error en evaluación: {e}")


def main():
    project_root = Path(__file__).parent.parent
    from dotenv import load_dotenv
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    bot = TradingBot(project_root / "config")
    duration = 0
    for i, arg in enumerate(sys.argv):
        if arg == "--duration" and i + 1 < len(sys.argv):
            duration = int(sys.argv[i + 1])
    bot.start(max_duration=duration)


if __name__ == "__main__":
    main()

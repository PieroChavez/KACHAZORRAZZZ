"""Timeframe Scheduler
Manages evaluation scheduling based on candle timeframes
"""
import time
from datetime import datetime, timedelta
from typing import Callable, Optional
from threading import Thread, Event
from loguru import logger

from ..core.multi_timeframe import MultiTimeframeFetcher


class TimeframeScheduler:
    """Scheduler that triggers evaluation when new candles form"""

    def __init__(self, mt5_fetcher: MultiTimeframeFetcher, symbol: str = "XAUUSDm"):
        self.fetcher = mt5_fetcher
        self.symbol = symbol
        self.running = False
        self._thread: Optional[Thread] = None
        self._stop_event = Event()
        self._callbacks: list[Callable] = []

        self._intervals = {
            "1min": 60,
            "3min": 3 * 60,
            "5min": 5 * 60,
            "1H": 3600,
            "2H": 7200,
            "3H": 10800,
            "4H": 14400,
        }

    def add_callback(self, callback: Callable):
        """Add a callback function to be called on new candle"""
        self._callbacks.append(callback)

    def start(self):
        """Start the scheduler in a background thread"""
        if self.running:
            logger.warning("Scheduler already running")
            return

        self.running = True
        self._stop_event.clear()
        self._thread = Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Scheduler started")

    def stop(self):
        """Stop the scheduler"""
        if not self.running:
            return

        self.running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Scheduler stopped")

    def _run_loop(self):
        """Main scheduler loop"""
        last_candle_times = {
            "1min": None,
            "3min": None,
            "5min": None,
            "1H": None,
            "2H": None,
            "3H": None,
            "4H": None,
        }

        while not self._stop_event.is_set():
            try:
                # Fetch all timeframes
                data = self.fetcher.fetch_all(self.symbol, count=5)

                # Check each timeframe for new candles
                for tf_name, tf_data in data.items():
                    if not tf_data.candles:
                        continue

                    current_time = tf_data.candles[0].timestamp.to_pydatetime()
                    last_time = last_candle_times.get(tf_name)

                    # If we haven't seen this timeframe before or candle is new
                    if last_time is None or current_time > last_time:
                        if last_time is not None:
                            # New candle formed - trigger callbacks
                            logger.info(f"New {tf_name} candle detected")
                            for callback in self._callbacks:
                                try:
                                    callback(tf_name, current_time)
                                except Exception as e:
                                    logger.error(f"Callback error: {e}")

                        last_candle_times[tf_name] = current_time

                # Sleep for a bit before next check
                self._stop_event.wait(timeout=5)

            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(10)

    def wait_for_next_candle(self, timeframe: str) -> datetime:
        """Block until next candle forms for given timeframe

        Args:
            timeframe: "M15", "H1", or "H4"

        Returns:
            Timestamp of the new candle
        """
        interval = self._intervals.get(timeframe, 60)
        start_time = datetime.now()

        while True:
            elapsed = (datetime.now() - start_time).total_seconds()
            remaining = interval - (elapsed % interval)

            if remaining < 1:  # Less than 1 second remaining
                # Wait for candle to actually close
                time.sleep(remaining + 1)
                return datetime.now()

            time.sleep(min(remaining, 5))  # Check more frequently


def get_next_candle_time(timeframe: str, from_time: Optional[datetime] = None) -> datetime:
    """Calculate when the next candle will close

    Args:
        timeframe: "M15", "H1", or "H4"
        from_time: Reference time (default: now)

    Returns:
        Datetime when current candle will close
    """
    if from_time is None:
        from_time = datetime.now()

    intervals = {
        "M5": 5 * 60,
        "M15": 15 * 60,
        "M30": 30 * 60,
    }

    interval = intervals.get(timeframe, 60)
    elapsed = (from_time.timestamp()) % interval
    remaining = interval - elapsed

    return from_time + timedelta(seconds=remaining)

"""Multi-timeframe data fetcher
Fetches and synchronizes data across all timeframes for full SMC cascade analysis
Order: 4H -> 3H -> 1H -> 30min -> 15min -> 10min -> 5min -> 3min -> 1min
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from loguru import logger

from .candle_closure_ratings import CandleData, CandleRating, rate_candle


TIMEFRAME_ORDER = [
    "4H", "3H", "2H", "1H",
    "30min", "15min", "10min", "5min", "3min", "1min",
]

TIMEFRAME_CODES = {
    "4H": 16388,
    "3H": 16387,
    "2H": 16386,
    "1H": 16385,
    "30min": 30,
    "15min": 15,
    "10min": 10,
    "5min": 5,
    "3min": 3,
    "1min": 1,
}

TIMEFRAME_GROUPS = {
    "HTF": ["4H", "3H", "2H", "1H"],
    "MID": ["30min", "15min", "10min"],
    "LTF": ["5min", "3min", "1min"],
}


@dataclass
class TimeframeData:
    timeframe: str
    candles: list[CandleData] = field(default_factory=list)
    ratings: list[CandleRating] = field(default_factory=list)
    last_update: Optional[datetime] = None


class MultiTimeframeFetcher:
    def __init__(self, mt5_client):
        self.mt5 = mt5_client
        self._data: Dict[str, TimeframeData] = {}
        for tf in TIMEFRAME_ORDER:
            self._data[tf] = TimeframeData(timeframe=tf)

    def fetch_all(self, symbol: str, count: int = 200) -> Dict[str, TimeframeData]:
        for tf_name in TIMEFRAME_ORDER:
            tf_code = TIMEFRAME_CODES.get(tf_name)
            if tf_code is None:
                continue
            try:
                candles = self.mt5.get_candles(symbol, timeframe=tf_code, count=count)
                if candles:
                    ratings = [rate_candle(c) for c in candles]
                    self._data[tf_name] = TimeframeData(
                        timeframe=tf_name,
                        candles=candles,
                        ratings=ratings,
                        last_update=datetime.now()
                    )
            except Exception as e:
                logger.warning(f"Failed to fetch {tf_name} for {symbol}: {e}")
        return self._data

    def get_dataframes(self, symbol: str, count: int = 200) -> Dict[str, pd.DataFrame]:
        data = self.fetch_all(symbol, count)
        result = {}
        for tf_name, tf_data in data.items():
            if not tf_data.candles:
                continue
            records = []
            for c in tf_data.candles:
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
            result[tf_name] = df
        return result

    def get_latest_rating(self, timeframe: str) -> Optional[CandleRating]:
        data = self._data.get(timeframe)
        if data and data.ratings:
            return data.ratings[0]
        return None

    def get_latest_candle(self, timeframe: str) -> Optional[CandleData]:
        data = self._data.get(timeframe)
        if data and data.candles:
            return data.candles[0]
        return None

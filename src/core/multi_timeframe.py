"""Multi-timeframe data fetcher
Fetches and synchronizes data across all timeframes for full SMC cascade analysis
Order: 4H -> 2H -> 30min -> 15min -> 5min
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from loguru import logger

from src.scoring.candle_closure_ratings import CandleData, CandleRating, rate_candle


TIMEFRAME_ORDER = [
    "3H", "1H",
    "30min", "3min",
]

TIMEFRAME_CODES = {
    "3H": 16387,
    "1H": 16385,
    "30min": 30,
    "3min": 3,
}

TIMEFRAME_GROUPS = {
    "HTF": ["3H", "1H"],
    "MID": ["30min"],
    "LTF": ["3min"],
}

HISTORICAL_COUNT = 5000
FRESH_COUNT = 100


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
        self._historical: Dict[str, Dict[str, pd.DataFrame]] = {}

    def init_historical(self, symbol: str,
                         count: int = HISTORICAL_COUNT) -> Dict[str, pd.DataFrame]:
        logger.info(f"[{symbol}] Inicializando histórico ({count} velas por TF)...")
        result = {}
        for tf_name in TIMEFRAME_ORDER:
            tf_code = TIMEFRAME_CODES.get(tf_name)
            if tf_code is None:
                continue
            try:
                candles = self.mt5.get_candles(symbol, timeframe=tf_code, count=count)
                if candles:
                    df = self._candles_to_df(candles)
                    result[tf_name] = df
                    logger.info(f"  {tf_name}: {len(df)} velas ({df['time'].iloc[0].strftime('%Y-%m-%d')} → {df['time'].iloc[-1].strftime('%Y-%m-%d %H:%M')})")
                else:
                    logger.warning(f"  {tf_name}: sin datos")
            except Exception as e:
                logger.warning(f"  {tf_name}: error → {e}")
        self._historical[symbol] = result
        logger.info(f"[{symbol}] Histórico cargado: {len(result)} timeframes")
        return result

    def get_historical(self, symbol: str) -> Optional[Dict[str, pd.DataFrame]]:
        return self._historical.get(symbol)

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

    def get_dataframes(self, symbol: str, count: int = FRESH_COUNT) -> Dict[str, pd.DataFrame]:
        fresh = self.fetch_all(symbol, count)
        result = {}
        for tf_name in TIMEFRAME_ORDER:
            tf_data = fresh.get(tf_name)
            if not tf_data or not tf_data.candles:
                continue
            fresh_df = self._candles_to_df(tf_data.candles)
            hist = self._historical.get(symbol, {}).get(tf_name)
            if hist is not None and len(hist) > 0:
                last_hist_time = hist["time"].iloc[-1]
                new_candles = fresh_df[fresh_df["time"] > last_hist_time]
                if len(new_candles) > 0:
                    merged = pd.concat([hist, new_candles], ignore_index=True)
                    merged = merged.tail(HISTORICAL_COUNT).reset_index(drop=True)
                    self._historical[symbol][tf_name] = merged
                    result[tf_name] = merged
                else:
                    result[tf_name] = hist
            else:
                result[tf_name] = fresh_df
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

    def _candles_to_df(self, candles: list[CandleData]) -> pd.DataFrame:
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

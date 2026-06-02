"""OANDA Client - Data Feed Reference
Provides supplementary data from OANDA for reference
"""
import requests
from datetime import datetime
from typing import Optional
from dataclasses import dataclass
import pandas as pd
from loguru import logger

from ..scoring.candle_closure_ratings import CandleData


@dataclass
class OANDAConfig:
    """OANDA API configuration"""
    api_key: str
    account_id: str
    environment: str = "practice"  # "practice" or "live"


class OANDAClient:
    """Client for OANDA API - reference data feed"""

    BASE_URLS = {
        "practice": "https://api-fxpractice.oanda.com",
        "live": "https://api-fxtrade.oanda.com"
    }

    def __init__(self, config: OANDAConfig):
        self.config = config
        self.base_url = self.BASE_URLS.get(config.environment, self.BASE_URLS["practice"])
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        })

    def get_candles(self, instrument: str, timeframe: str = "H1",
                    count: int = 100) -> list[CandleData]:
        """Get candles from OANDA

        Args:
            instrument: OANDA instrument code (e.g., "XAU_USD")
            timeframe: OANDA granularity (e.g., "H1", "H4", "M15")
            count: Number of candles

        Returns:
            List of CandleData objects (newest first)
        """
        # Note: OANDA uses XAU_USD format, not XAUUSD
        endpoint = f"{self.base_url}/v3/instruments/{instrument}/candles"

        params = {
            "granularity": timeframe,
            "count": count
        }

        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            candles = []

            for c in data.get("candles", []):
                candle = CandleData(
                    timestamp=pd.Timestamp(c["time"]),
                    open=float(c["mid"]["o"]),
                    high=float(c["mid"]["h"]),
                    low=float(c["mid"]["l"]),
                    close=float(c["mid"]["c"]),
                    volume=float(c["volume"])
                )
                candles.append(candle)

            return candles

        except requests.RequestException as e:
            logger.error(f"OANDA API error: {e}")
            return []

    def get_account_summary(self) -> Optional[dict]:
        """Get account summary from OANDA"""
        endpoint = f"{self.base_url}/v3/accounts/{self.config.account_id}/summary"

        try:
            response = self.session.get(endpoint, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"OANDA account error: {e}")
            return None

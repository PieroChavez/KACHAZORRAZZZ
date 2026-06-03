"""News event calendar for filtering high-impact news during trading.
Loads events from config/news_events.json.
"""
from datetime import datetime, timedelta
from typing import List, Optional
import json
from pathlib import Path


class NewsEvent:
    def __init__(self, dt: datetime, title: str, impact: str, currency: str):
        self.dt = dt
        self.title = title
        self.impact = impact
        self.currency = currency


CURRENCY_FOR_SYMBOL = {
    "XAUUSD": "USD", "XAGUSD": "USD", "GOLD": "USD", "SILVER": "USD",
    "NAS100": "USD", "US100": "USD", "NDX": "USD",
    "US30": "USD", "DJI30": "USD",
    "SPX500": "USD", "SP500": "USD",
    "EURUSD": "EUR", "GBPUSD": "GBP", "USDJPY": "JPY",
    "USDCAD": "CAD", "AUDUSD": "AUD", "NZDUSD": "NZD",
}


class NewsCalendar:
    def __init__(self, config_dir: Optional[Path] = None, buffer_minutes: int = 30):
        self.events: List[NewsEvent] = []
        self.buffer_minutes = buffer_minutes
        if config_dir:
            self._load_events(config_dir)

    def _load_events(self, config_dir: Path):
        news_file = config_dir / "news_events.json"
        if news_file.exists():
            with open(news_file) as f:
                data = json.load(f)
            for item in data:
                time_str = item.get("time")
                if time_str is None:
                    continue
                self.events.append(NewsEvent(
                    dt=datetime.fromisoformat(time_str),
                    title=item.get("title", ""),
                    impact=item.get("impact", "HIGH"),
                    currency=item.get("currency", "USD"),
                ))

    def is_high_impact_active(self, current_dt: datetime, symbol: str) -> bool:
        sym_currency = self._symbol_currency(symbol)
        if not sym_currency:
            return False
        for event in self.events:
            if event.impact.upper() != "HIGH":
                continue
            if event.currency != sym_currency:
                continue
            time_diff = abs((current_dt - event.dt).total_seconds())
            if time_diff <= self.buffer_minutes * 60:
                return True
        return False

    def get_upcoming(self, symbol: str, max_count: int = 5) -> List[NewsEvent]:
        sym_currency = self._symbol_currency(symbol)
        now = datetime.now()
        matching = [
            e for e in self.events
            if e.currency == sym_currency and e.impact.upper() == "HIGH" and e.dt > now
        ]
        matching.sort(key=lambda e: e.dt)
        return matching[:max_count]

    def log_upcoming(self, symbol: str):
        upcoming = self.get_upcoming(symbol)
        if upcoming:
            import logging
            logger = logging.getLogger(__name__)
            for ev in upcoming:
                logger.info(f"  News: {ev.title} @ {ev.dt.isoformat()} [{ev.impact}]")

    @staticmethod
    def _symbol_currency(symbol: str) -> Optional[str]:
        s = symbol.upper()
        for key, curr in CURRENCY_FOR_SYMBOL.items():
            if key in s:
                return curr
        return None

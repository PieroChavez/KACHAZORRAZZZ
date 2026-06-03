"""MarketHoursController — Auto-regula evaluación de símbolos según disponibilidad de mercado.

Detecta apertura/cierre mediante freshness de candles 1m.
No requiere configuración de horarios, timezones, ni feriados.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Optional, Set

from loguru import logger


class AssetClass(str, Enum):
    CRYPTO = "crypto"
    FOREX = "forex"
    INDEX = "index"
    COMMODITY = "commodity"
    UNKNOWN = "unknown"


class MarketState(str, Enum):
    BOOTSTRAP = "BOOTSTRAP"
    OPEN = "OPEN"
    BLACKOUT_PRE = "BLACKOUT_PRE"
    CLOSED = "CLOSED"
    COOLDOWN_POST = "COOLDOWN_POST"


_CRYPTO_RE = re.compile(
    r'^(BTC|ETH|XRP|LTC|ADA|DOT|SOL|BNB|LINK|MATIC|AVAX|DASH|EOS|'
    r'TRX|XLM|ATOM|VET|FIL|AAVE|UNI|SUSHI|ALGO|NEAR|FTT|XMR|ZEC|DOGE|SHIB)',
    re.IGNORECASE,
)
_COMMODITY_RE = re.compile(r'^(XAU|XAG|XPT|XPD|XCU|UKOIL|USOIL)', re.IGNORECASE)
_INDEX_RE = re.compile(
    r'^(USTEC|US30|US100|DAX|NK225|SPX|SP500|CAC|FTSE|IBEX|VIX)', re.IGNORECASE
)
_FOREX_RE = re.compile(r'^[A-Z]{6}(\.|\b|m|$)')


def classify_symbol(symbol: str) -> AssetClass:
    if _CRYPTO_RE.match(symbol):
        return AssetClass.CRYPTO
    if _COMMODITY_RE.match(symbol):
        return AssetClass.COMMODITY
    if _INDEX_RE.match(symbol):
        return AssetClass.INDEX
    if _FOREX_RE.match(symbol):
        return AssetClass.FOREX
    return AssetClass.UNKNOWN


class SymbolHoursState:
    __slots__ = (
        'asset_class', 'state', 'last_candle_time', 'last_update',
        'cooldown_until', 'blackout_start', 'bootstrap_until',
        'intervals_min',
    )

    def __init__(self, asset_class: AssetClass):
        self.asset_class = asset_class
        self.state = MarketState.BOOTSTRAP
        self.last_candle_time: Optional[datetime] = None
        self.last_update: Optional[datetime] = None
        self.cooldown_until: Optional[datetime] = None
        self.blackout_start: Optional[datetime] = None
        self.bootstrap_until: Optional[datetime] = None
        self.intervals_min: list = []

    def to_dict(self) -> dict:
        def _ts(dt):
            return dt.timestamp() if dt else None
        return {
            'asset_class': self.asset_class.value,
            'state': self.state.value,
            'last_candle_time': _ts(self.last_candle_time),
            'last_update': _ts(self.last_update),
            'cooldown_until': _ts(self.cooldown_until),
            'blackout_start': _ts(self.blackout_start),
            'bootstrap_until': _ts(self.bootstrap_until),
        }

    @classmethod
    def from_dict(cls, data: dict) -> SymbolHoursState:
        obj = cls(AssetClass(data.get('asset_class', 'unknown')))
        obj.state = MarketState(data.get('state', 'BOOTSTRAP'))

        def _dt(ts):
            return datetime.fromtimestamp(ts) if ts else None

        obj.last_candle_time = _dt(data.get('last_candle_time'))
        obj.last_update = _dt(data.get('last_update'))
        obj.cooldown_until = _dt(data.get('cooldown_until'))
        obj.blackout_start = _dt(data.get('blackout_start'))
        obj.bootstrap_until = _dt(data.get('bootstrap_until'))
        return obj


class MarketHoursController:
    STALE_CANDLE_MINUTES: float = 25.0
    CLOSED_CANDLE_MINUTES: float = 35.0
    COOLDOWN_MINUTES: float = 10.0
    BOOTSTRAP_MINUTES: float = 60.0
    FRESH_CANDLE_MAX_AGE: float = 3.0

    SPREAD_ANOMALY_RATIO: float = 5.0
    SPREAD_BASELINE_SAMPLES: int = 20
    VOLUME_DROP_RATIO: float = 0.10
    VOLUME_BASELINE_SAMPLES: int = 20

    def __init__(self):
        self._states: Dict[str, SymbolHoursState] = {}
        self._start_time: Optional[datetime] = None
        self._spread_baselines: Dict[str, list] = {}
        self._volume_baselines: Dict[str, list] = {}
        self._last_persisted: Dict[str, dict] = {}
        self._changed: bool = False

    def classify_all(self, symbols: Set[str]):
        self._start_time = datetime.now()
        for sym in sorted(symbols):
            asset_class = classify_symbol(sym)
            state = SymbolHoursState(asset_class)
            state.bootstrap_until = self._start_time + timedelta(
                minutes=self.BOOTSTRAP_MINUTES
            )
            self._states[sym] = state
            logger.info(f"[MarketHours] {sym} → {asset_class.value}")

    def get_state(self, symbol: str) -> MarketState:
        state = self._states.get(symbol)
        return state.state if state else MarketState.OPEN

    def get_asset_class(self, symbol: str) -> AssetClass:
        state = self._states.get(symbol)
        return state.asset_class if state else AssetClass.UNKNOWN

    def on_new_candle(self, symbol: str, timeframe: str, candle_time: datetime):
        if timeframe != "1min":
            return
        state = self._states.get(symbol)
        if state is None:
            return

        if candle_time.tzinfo is not None:
            candle_time = candle_time.replace(tzinfo=None)

        now = datetime.now()
        prev = state.last_candle_time

        if prev is not None:
            interval = (candle_time - prev).total_seconds() / 60.0
            state.intervals_min.append(interval)
            if len(state.intervals_min) > 5:
                state.intervals_min.pop(0)

        state.last_candle_time = candle_time
        state.last_update = now

    def update_states(self, now: Optional[datetime] = None):
        if now is None:
            now = datetime.now()
        for sym, state in list(self._states.items()):
            try:
                self._update_state(sym, state, now)
            except Exception as e:
                logger.warning(f"[MarketHours] Error updating {sym}: {e}")

    def _update_state(self, sym: str, state: SymbolHoursState, now: datetime):
        if state.asset_class == AssetClass.CRYPTO:
            state.state = MarketState.OPEN
            return

        if state.last_candle_time is None:
            return

        if state.bootstrap_until and now < state.bootstrap_until:
            state.state = MarketState.BOOTSTRAP
            return

        age_min = (now - state.last_candle_time).total_seconds() / 60.0
        current_state = state.state

        if age_min < self.FRESH_CANDLE_MAX_AGE:
            if current_state in (MarketState.CLOSED, MarketState.BLACKOUT_PRE):
                state.state = MarketState.COOLDOWN_POST
                state.cooldown_until = now + timedelta(minutes=self.COOLDOWN_MINUTES)
                logger.info(
                    f"[MarketHours] {sym}: reabrió → COOLDOWN_POST "
                    f"({self.COOLDOWN_MINUTES}min)"
                )
            elif current_state == MarketState.COOLDOWN_POST:
                if state.cooldown_until and now >= state.cooldown_until:
                    state.state = MarketState.OPEN
                    logger.info(
                        f"[MarketHours] {sym}: COOLDOWN_POST expirado → OPEN"
                    )
            elif current_state == MarketState.BOOTSTRAP:
                state.state = MarketState.OPEN

        elif age_min >= self.CLOSED_CANDLE_MINUTES:
            if current_state != MarketState.CLOSED:
                state.state = MarketState.CLOSED
                logger.info(
                    f"[MarketHours] {sym}: {age_min:.0f}min sin candle → CLOSED"
                )

        elif age_min >= self.STALE_CANDLE_MINUTES:
            if current_state not in (
                MarketState.CLOSED,
                MarketState.BLACKOUT_PRE,
                MarketState.COOLDOWN_POST,
            ):
                state.state = MarketState.BLACKOUT_PRE
                state.blackout_start = now
                logger.info(
                    f"[MarketHours] {sym}: {age_min:.0f}min sin candle → "
                    f"BLACKOUT_PRE"
                )

        if current_state == MarketState.COOLDOWN_POST:
            if state.cooldown_until and now >= state.cooldown_until:
                state.state = MarketState.OPEN
                logger.info(
                    f"[MarketHours] {sym}: COOLDOWN_POST expirado → OPEN"
                )

    def should_evaluate(self, symbol: str) -> bool:
        state = self._states.get(symbol)
        if state is None:
            return True
        if state.asset_class == AssetClass.CRYPTO:
            return True
        if state.asset_class in (AssetClass.COMMODITY, AssetClass.INDEX, AssetClass.FOREX):
            if datetime.now().weekday() >= 5:
                return False
        return state.state in (
            MarketState.OPEN,
            MarketState.BOOTSTRAP,
            MarketState.BLACKOUT_PRE,
            MarketState.COOLDOWN_POST,
        )

    def can_open_new_position(self, symbol: str) -> bool:
        state = self._states.get(symbol)
        if state is None:
            return True
        if state.asset_class == AssetClass.CRYPTO:
            return True
        return state.state in (MarketState.OPEN, MarketState.BOOTSTRAP)

    def is_traditional_market_open(self) -> bool:
        for sym, state in self._states.items():
            if state.asset_class != AssetClass.CRYPTO and state.state in (
                MarketState.OPEN,
                MarketState.COOLDOWN_POST,
            ):
                return True
        return False

    def on_spread_data(self, symbol: str, current_spread: float):
        """Feed spread tick data for anomaly detection (secondary close signal).
        Si el spread se dispara > 5x el promedio, fuerza BLACKOUT_PRE.
        """
        state = self._states.get(symbol)
        if state is None or state.asset_class == AssetClass.CRYPTO:
            return

        baseline = self._spread_baselines.setdefault(symbol, [])
        baseline.append(current_spread)
        if len(baseline) > self.SPREAD_BASELINE_SAMPLES * 2:
            baseline[:self.SPREAD_BASELINE_SAMPLES] = []

        if len(baseline) >= self.SPREAD_BASELINE_SAMPLES:
            avg_spread = sum(baseline[-self.SPREAD_BASELINE_SAMPLES:]) / self.SPREAD_BASELINE_SAMPLES
            if avg_spread > 0 and current_spread / avg_spread > self.SPREAD_ANOMALY_RATIO:
                if state.state == MarketState.OPEN:
                    state.state = MarketState.BLACKOUT_PRE
                    state.blackout_start = datetime.now()
                    self._changed = True
                    logger.info(
                        f"[MarketHours] {symbol}: spread spike "
                        f"{current_spread/avg_spread:.1f}x → BLACKOUT_PRE"
                    )

    def on_volume_data(self, symbol: str, tick_volume: float):
        """Feed tick volume for zombie market detection (tertiary close signal).
        Si el volumen del último candle 1m es < 10% del promedio, fuerza BLACKOUT_PRE.
        """
        state = self._states.get(symbol)
        if state is None or state.asset_class == AssetClass.CRYPTO:
            return

        baseline = self._volume_baselines.setdefault(symbol, [])
        baseline.append(tick_volume)
        if len(baseline) > self.VOLUME_BASELINE_SAMPLES * 2:
            baseline[:self.VOLUME_BASELINE_SAMPLES] = []

        if len(baseline) >= self.VOLUME_BASELINE_SAMPLES:
            avg_vol = sum(baseline[-self.VOLUME_BASELINE_SAMPLES:]) / self.VOLUME_BASELINE_SAMPLES
            if avg_vol > 0 and tick_volume / avg_vol < self.VOLUME_DROP_RATIO:
                if state.state == MarketState.OPEN:
                    state.state = MarketState.BLACKOUT_PRE
                    state.blackout_start = datetime.now()
                    self._changed = True
                    logger.info(
                        f"[MarketHours] {symbol}: volume plunge "
                        f"{tick_volume/avg_vol:.1%} of avg → BLACKOUT_PRE"
                    )

    def get_all_states(self) -> Dict[str, dict]:
        return {sym: state.to_dict() for sym, state in self._states.items()}

    def get_changed_states(self) -> Dict[str, dict]:
        """Return states that changed since last call, and update baseline."""
        current = self.get_all_states()
        changed = {}
        for sym, data in current.items():
            prev = self._last_persisted.get(sym)
            if prev is None or prev.get('state') != data.get('state'):
                changed[sym] = data
        self._last_persisted = current
        self._changed = bool(changed)
        return changed

    def mark_persisted(self):
        """Sync the persisted baseline so get_changed_states returns empty next time."""
        self._last_persisted = self.get_all_states()
        self._changed = False

    def restore_states(self, states_data: Dict[str, dict]):
        for sym, data in states_data.items():
            if sym in self._states:
                restored = SymbolHoursState.from_dict(data)
                restored.intervals_min = self._states[sym].intervals_min
                self._states[sym] = restored
        self.mark_persisted()

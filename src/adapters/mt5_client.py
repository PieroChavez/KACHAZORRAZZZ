"""MT5 Client Adapter
Connects to MetaTrader 5 terminal for data and execution
"""
import time
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
from loguru import logger

from ..scoring.candle_closure_ratings import CandleData


# Symbol alias map: canonical name -> list of known broker variants to try
SYMBOL_ALIASES = {
    "USDOLLAR": ["USDOLLAR", "USDollar", "USDX", "DXY", "DX"],
    "US100": ["US100", "NAS100", "USTEC", "USTEC_x100"],
    "US30": ["US30", "DJI30"],
    "SPX500": ["SPX500", "US500"],
}


class MT5Client:
    """MetaTrader 5 client for data retrieval and order execution"""

    def __init__(self, login: Optional[int] = None, password: Optional[str] = None,
                 server: Optional[str] = None, path: Optional[str] = None):
        self.login = int(login) if login is not None else None
        self.password = password
        self.server = server
        self.path = path
        self.connected = False
        self._symbol_cache: Dict[str, str] = {}

    def _find_mt5_path(self) -> Optional[str]:
        return self.path

    def resolve_symbol(self, symbol: str) -> str:
        """Try to resolve a canonical symbol name to the broker's actual symbol.
        Caches successful resolution to avoid repeated broker calls.

        Args:
            symbol: The canonical symbol name (e.g. 'USDOLLAR')

        Returns:
            The broker's actual symbol name, or the original if no alias found
        """
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]

        if not self.ensure_connected():
            return symbol

        if mt5.symbol_info(symbol) is not None:
            self._symbol_cache[symbol] = symbol
            return symbol

        aliases = SYMBOL_ALIASES.get(symbol, [])
        for alias in aliases:
            if mt5.symbol_info(alias) is not None:
                self._symbol_cache[symbol] = alias
                logger.info(f"Symbol '{symbol}' resolved to '{alias}' on this broker")
                return alias

        logger.warning(f"Could not resolve symbol '{symbol}' - tried aliases: {aliases}")
        return symbol

    def connect(self) -> bool:
        mt5.shutdown()
        time.sleep(1)

        path = self._find_mt5_path()
        ok = mt5.initialize(path=path) if path else mt5.initialize()
        if not ok:
            err = mt5.last_error()
            logger.warning(f"MT5 initialize failed: {err}, retrying without path...")
            mt5.shutdown()
            time.sleep(1)
            ok = mt5.initialize()
            if not ok:
                err = mt5.last_error()
                logger.error(f"MT5 initialize failed (2nd attempt): {err}")
                logger.error("Asegúrate de: 1) MT5 esté instalado, 2) Abrirlo manualmente una vez para aceptar licencia")
                return False

        if self.login and self.password and self.server:
            account = mt5.account_info()
            if account is not None and account.login == self.login:
                logger.info(f"Already logged in as {account.login}@{account.server}")
                self.connected = True
                return True
            logger.info(f"Logging in as {self.login}@{self.server}...")
            if not mt5.login(self.login, password=self.password, server=self.server):
                err = mt5.last_error()
                logger.error(f"MT5 login failed (login={self.login}, server={self.server}): {err}")
                mt5.shutdown()
                return False

        self.connected = True
        logger.info(f"MT5 connected: login={self.login}, server={self.server}")
        return True

    def ensure_connected(self) -> bool:
        """Reconnect if MT5 terminal disconnected"""
        if self.connected:
            try:
                info = mt5.terminal_info()
                if info is not None and info.connected:
                    return True
            except Exception:
                pass
        logger.warning("MT5 disconnected, reconnecting...")
        mt5.shutdown()
        time.sleep(1)
        return self.connect()

    def disconnect(self):
        """Shutdown MT5 connection"""
        mt5.shutdown()
        self.connected = False
        logger.info("MT5 disconnected")

    def is_connected(self) -> bool:
        """Check if MT5 is connected and terminal is online"""
        if not self.connected:
            return False
        info = mt5.terminal_info()
        return info is not None and info.connected

    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """Get symbol info"""
        self.ensure_connected()
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        return {
            "symbol": info.name,
            "bid": info.bid,
            "ask": info.ask,
            "last": info.last,
            "volume": info.volume,
            "digits": info.digits,
            "point": info.point,
            "spread": info.spread,
            "trade_tick_size": info.trade_tick_size,
            "volume_min": info.volume_min,
            "volume_step": info.volume_step,
        }

    def _ensure_symbol(self, symbol: str) -> bool:
        resolved = self.resolve_symbol(symbol)
        for attempt in range(3):
            if mt5.symbol_select(resolved, True):
                return True
            if attempt == 0:
                logger.warning(f"Could not select {resolved} (from '{symbol}'), retrying...")
            time.sleep(2)
        logger.error(f"Could not select {resolved} (from '{symbol}') after 3 attempts")
        return False

    def get_candles(self, symbol: str, timeframe: int = 60,
                    count: int = 100) -> list[CandleData]:
        self.ensure_connected()
        resolved = self.resolve_symbol(symbol)
        if not self._ensure_symbol(symbol):
            return []

        rates = mt5.copy_rates_from_pos(resolved, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            return []

        candles = []
        for rate in rates:
            candle = CandleData(
                timestamp=pd.Timestamp.fromtimestamp(rate['time'], tz='UTC'),
                open=float(rate['open']),
                high=float(rate['high']),
                low=float(rate['low']),
                close=float(rate['close']),
                volume=float(rate['tick_volume'])
            )
            candles.append(candle)

        return candles

    def get_latest_candle_time(self, symbol: str, timeframe: int) -> Optional[datetime]:
        """Get the timestamp of the most recent candle"""
        candles = self.get_candles(symbol, timeframe, count=1)
        if candles:
            return candles[0].timestamp.to_pydatetime()
        return None

    def get_account_info(self) -> Optional[dict]:
        """Get trading account information"""
        if not self.connected:
            return None

        info = mt5.account_info()
        if info is None:
            return None

        return {
            "login": info.login,
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "leverage": info.leverage,
            "profit": info.profit,
        }

    def get_positions(self, symbol: Optional[str] = None) -> list[dict]:
        """Get open positions"""
        positions = mt5.positions_get()
        if positions is None:
            return []

        result = []
        for pos in positions:
            if symbol is None or pos.symbol == symbol:
                result.append({
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "type": "buy" if pos.type == mt5.POSITION_TYPE_BUY else "sell",
                    "volume": pos.volume,
                    "price_open": pos.price_open,
                    "price_current": pos.price_current,
                    "profit": pos.profit,
                    "sl": pos.sl,
                    "tp": pos.tp,
                    "magic": pos.magic,
                    "time": pd.Timestamp.fromtimestamp(pos.time, tz='UTC'),
                })

        return result

    def has_open_position(self, symbol: str) -> bool:
        """Check if symbol has an open position"""
        positions = self.get_positions(symbol)
        return len(positions) > 0

    def get_pending_orders(self, symbol: Optional[str] = None) -> list[dict]:
        """Get pending orders (limit/stop)"""
        orders = mt5.orders_get()
        if orders is None:
            return []
        result = []
        for o in orders:
            if symbol is None or o.symbol == symbol:
                result.append({
                    "ticket": o.ticket,
                    "symbol": o.symbol,
                    "type": o.type,
                    "volume": o.volume_initial,
                    "price": o.price_open,
                    "sl": o.sl,
                    "tp": o.tp,
                    "time": pd.Timestamp.fromtimestamp(o.time_setup, tz='UTC'),
                    "comment": o.comment,
                })
        return result

    def modify_position(self, ticket: int, new_sl: float, new_tp: float = 0.0) -> bool:
        """Modify position SL and TP

        Args:
            ticket: Position ticket number
            new_sl: New stop loss price
            new_tp: New take profit price (0 to leave unchanged)

        Returns:
            True if modification successful
        """
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": new_sl,
            "tp": new_tp,
            "comment": "Trailing/BE"
        }

        result = mt5.order_send(request)

        if result is None:
            logger.error(f"Position {ticket} modify failed: {mt5.last_error()}")
            return False

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Position {ticket} modify failed: {result.comment}")
            return False

        logger.info(f"Position {ticket} modified: SL={new_sl}, TP={new_tp}")
        return True

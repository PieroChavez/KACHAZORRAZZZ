from abc import ABC, abstractmethod
from typing import Any, Optional


class MT5API(ABC):
    """Abstract interface to the MetaTrader5 module.

    Allows dependency injection so MT5Client can be tested with a fake.
    """

    COPY_TICKS_ALL: int = 2
    POSITION_TYPE_BUY: int = 0
    TRADE_ACTION_SLTP: int = 5
    TRADE_RETCODE_DONE: int = 10009

    @abstractmethod
    def initialize(self, path: Optional[str] = None) -> bool:
        ...

    @abstractmethod
    def shutdown(self) -> None:
        ...

    @abstractmethod
    def login(self, login: int, password: str, server: str) -> bool:
        ...

    @abstractmethod
    def account_info(self) -> Any:
        """Returns an object with .login, .server, .balance, .equity, etc."""

    @abstractmethod
    def terminal_info(self) -> Any:
        """Returns an object with .connected"""

    @abstractmethod
    def last_error(self) -> str:
        ...

    @abstractmethod
    def symbol_info(self, symbol: str) -> Any:
        """Returns an object with .name, .bid, .ask, ... or None"""

    @abstractmethod
    def symbol_select(self, symbol: str, enable: bool) -> bool:
        ...

    @abstractmethod
    def market_book_add(self, symbol: str) -> bool:
        ...

    @abstractmethod
    def market_book_release(self, symbol: str) -> None:
        ...

    @abstractmethod
    def market_book_get(self, symbol: str) -> Any:
        """Returns list of book entries with .price, .volume, .volume_dbl, .type"""

    @abstractmethod
    def copy_ticks_from(self, symbol: str, from_time: Any, count: int,
                        flags: int) -> Any:
        """Returns array of ticks with .time, .bid, .ask, .last, .volume, .flags"""

    @abstractmethod
    def copy_rates_from_pos(self, symbol: str, timeframe: int,
                            start: int, count: int) -> Any:
        """Returns array of rates with .time, .open, .high, .low, .close, .tick_volume"""

    @abstractmethod
    def positions_get(self) -> Any:
        """Returns list of positions with .ticket, .symbol, .type, .volume, ..."""

    @abstractmethod
    def orders_get(self) -> Any:
        """Returns list of orders with .ticket, .symbol, .type, .volume_initial, ..."""

    @abstractmethod
    def order_send(self, request: dict) -> Any:
        """Returns result with .retcode, .comment"""


class LiveMT5API(MT5API):
    """Delegates directly to the MetaTrader5 module."""

    def __init__(self):
        import MetaTrader5 as _mt5
        self._mt5 = _mt5
        self.COPY_TICKS_ALL = _mt5.COPY_TICKS_ALL
        self.POSITION_TYPE_BUY = _mt5.POSITION_TYPE_BUY
        self.TRADE_ACTION_SLTP = _mt5.TRADE_ACTION_SLTP
        self.TRADE_RETCODE_DONE = _mt5.TRADE_RETCODE_DONE

    def initialize(self, path=None):
        if path:
            return self._mt5.initialize(path=path)
        return self._mt5.initialize()

    def shutdown(self):
        self._mt5.shutdown()

    def login(self, login, password, server):
        return self._mt5.login(login, password=password, server=server)

    def account_info(self):
        return self._mt5.account_info()

    def terminal_info(self):
        return self._mt5.terminal_info()

    def last_error(self):
        return self._mt5.last_error()

    def symbol_info(self, symbol):
        return self._mt5.symbol_info(symbol)

    def symbol_select(self, symbol, enable):
        return self._mt5.symbol_select(symbol, enable)

    def market_book_add(self, symbol):
        return self._mt5.market_book_add(symbol)

    def market_book_release(self, symbol):
        self._mt5.market_book_release(symbol)

    def market_book_get(self, symbol):
        return self._mt5.market_book_get(symbol)

    def copy_ticks_from(self, symbol, from_time, count, flags):
        return self._mt5.copy_ticks_from(symbol, from_time, count, flags)

    def copy_rates_from_pos(self, symbol, timeframe, start, count):
        return self._mt5.copy_rates_from_pos(symbol, timeframe, start, count)

    def positions_get(self):
        return self._mt5.positions_get()

    def orders_get(self):
        return self._mt5.orders_get()

    def order_send(self, request):
        return self._mt5.order_send(request)

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Optional

import pandas as pd
import pytest

from src.adapters.mt5_api import MT5API
from src.adapters.mt5_client import MT5Client


class FakeMT5API(MT5API):

    COPY_TICKS_ALL = 2
    POSITION_TYPE_BUY = 0
    TRADE_ACTION_SLTP = 5
    TRADE_RETCODE_DONE = 10009

    def __init__(self):
        self.reset()

    def reset(self):
        self.calls = []
        self._initialize = True
        self._logged_in = False
        self._terminal_connected = True
        self._account = SimpleNamespace(
            login=123456, server="ICMarkets-Demo",
            balance=10000.0, equity=10000.0,
            margin=0.0, margin_free=10000.0,
            leverage=500, profit=0.0,
        )
        self._symbols = {}
        self._positions = []
        self._orders = []
        self._order_send_result = SimpleNamespace(retcode=10009, comment="Done")
        self._last_error_str = "No error"
        self._rates = []

    def initialize(self, path=None):
        self.calls.append(("initialize", path))
        return self._initialize

    def shutdown(self):
        self.calls.append(("shutdown",))

    def login(self, login, password, server):
        self.calls.append(("login", login, server))
        self._logged_in = True
        return True

    def account_info(self):
        self.calls.append(("account_info",))
        return self._account

    def terminal_info(self):
        self.calls.append(("terminal_info",))
        return SimpleNamespace(connected=self._terminal_connected)

    def last_error(self):
        return self._last_error_str

    def symbol_info(self, symbol):
        self.calls.append(("symbol_info", symbol))
        return self._symbols.get(symbol)

    def symbol_select(self, symbol, enable):
        self.calls.append(("symbol_select", symbol, enable))
        return True

    def market_book_add(self, symbol):
        self.calls.append(("market_book_add", symbol))
        return True

    def market_book_release(self, symbol):
        self.calls.append(("market_book_release", symbol))

    def market_book_get(self, symbol):
        self.calls.append(("market_book_get", symbol))
        return []

    def copy_ticks_from(self, symbol, from_time, count, flags):
        self.calls.append(("copy_ticks_from", symbol, count, flags))
        return []

    def copy_rates_from_pos(self, symbol, timeframe, start, count):
        self.calls.append(("copy_rates_from_pos", symbol, timeframe, start, count))
        return self._rates

    def positions_get(self):
        self.calls.append(("positions_get",))
        return self._positions

    def orders_get(self):
        self.calls.append(("orders_get",))
        return self._orders

    def order_send(self, request):
        self.calls.append(("order_send", request))
        return self._order_send_result


# ── helpers ──

def _rate(time: int, open=1.0, high=1.0, low=1.0, close=1.0, tick_volume=0):
    return SimpleNamespace(time=time, open=open, high=high,
                           low=low, close=close, tick_volume=tick_volume)


def _pos(ticket=1, symbol="XAUUSD", type_=0, volume=0.1,
         price_open=1900.0, price_current=1900.0,
         profit=0.0, sl=0.0, tp=0.0, time=1000):
    return SimpleNamespace(ticket=ticket, symbol=symbol, type=type_,
                           volume=volume, price_open=price_open,
                           price_current=price_current, profit=profit,
                           sl=sl, tp=tp, time=time)


def _order(ticket=1, symbol="XAUUSD", type_=0, volume_initial=0.1,
           price_open=1900.0, sl=0.0, tp=0.0, time_setup=1000,
           comment="test"):
    return SimpleNamespace(ticket=ticket, symbol=symbol, type=type_,
                           volume_initial=volume_initial,
                           price_open=price_open, sl=sl, tp=tp,
                           time_setup=time_setup, comment=comment)


# ── Fixtures ──

@pytest.fixture
def fake_api():
    return FakeMT5API()


@pytest.fixture
def client(fake_api):
    return MT5Client(api=fake_api)


# ── Tests ──

class TestInit:
    def test_default_api_is_live(self):
        c = MT5Client()
        from src.adapters.mt5_api import LiveMT5API
        assert isinstance(c.api, LiveMT5API)

    def test_stores_credentials(self):
        c = MT5Client(login=1, password="p", server="s", path="/mt5")
        assert c.login == 1
        assert c.password == "p"
        assert c.server == "s"
        assert c.path == "/mt5"

    def test_connected_false_by_default(self, client):
        assert not client.connected


class TestConnect:
    def test_connect_calls_initialize(self, client, fake_api):
        assert client.connect()
        assert any(c[0] == "initialize" for c in fake_api.calls)

    def test_connect_sets_connected(self, client, fake_api):
        client.connect()
        assert client.connected

    def test_connect_with_login(self, fake_api):
        c = MT5Client(login=1, password="p", server="s", api=fake_api)
        assert c.connect()
        assert any(c[0] == "login" for c in fake_api.calls)

    def test_connect_failure(self, client, fake_api):
        fake_api._initialize = False
        assert not client.connect()
        assert not client.connected

    def test_connect_already_logged_in(self, fake_api):
        c = MT5Client(login=123456, password="p", server="ICMarkets-Demo",
                       api=fake_api)
        assert c.connect()
        # Should NOT call login because already on same account/server
        assert not any(c[0] == "login" for c in fake_api.calls)

    def test_connect_switches_account(self, fake_api):
        c = MT5Client(login=999999, password="p", server="Other-Server",
                       api=fake_api)
        assert c.connect()
        assert any(c[0] == "login" for c in fake_api.calls)


class TestConnectionManagement:
    def test_ensure_connected_already_ok(self, client, fake_api):
        client.connected = True
        assert client.ensure_connected()
        calls = [c[0] for c in fake_api.calls]
        assert "terminal_info" in calls

    def test_ensure_connected_reconnects(self, client, fake_api):
        client.connected = True
        fake_api._terminal_connected = False
        assert client.ensure_connected()
        calls = [c[0] for c in fake_api.calls]
        assert "shutdown" in calls
        assert "initialize" in calls

    def test_disconnect(self, client, fake_api):
        client.connected = True
        client.disconnect()
        assert not client.connected
        assert any(c[0] == "shutdown" for c in fake_api.calls)

    def test_is_connected_true(self, client, fake_api):
        client.connected = True
        assert client.is_connected()

    def test_is_connected_false_not_connected(self, client):
        assert not client.is_connected()

    def test_is_connected_terminal_down(self, client, fake_api):
        client.connected = True
        fake_api._terminal_connected = False
        assert not client.is_connected()


class TestSymbolResolution:
    def test_resolve_symbol_found(self, client, fake_api):
        fake_api._symbols["XAUUSD"] = SimpleNamespace(name="XAUUSD")
        assert client.resolve_symbol("XAUUSD") == "XAUUSD"

    def test_resolve_symbol_not_found(self, client, fake_api):
        assert client.resolve_symbol("UNKNOWN") is None

    def test_resolve_symbol_tries_variants(self, client, fake_api):
        fake_api._symbols["XAUUSDm"] = SimpleNamespace(name="XAUUSDm")
        assert client.resolve_symbol("XAUUSD") == "XAUUSDm"


class TestSymbolInfo:
    def test_get_symbol_info(self, client, fake_api):
        fake_api._symbols["XAUUSD"] = SimpleNamespace(
            name="XAUUSD", bid=1900.0, ask=1900.5, last=1900.2,
            volume=100, digits=2, point=0.01, spread=5,
        )
        info = client.get_symbol_info("XAUUSD")
        assert info is not None
        assert info["symbol"] == "XAUUSD"
        assert info["bid"] == 1900.0
        assert info["ask"] == 1900.5

    def test_get_symbol_info_not_found(self, client):
        assert client.get_symbol_info("NONEXIST") is None


class TestDOM:
    def test_subscribe_dom(self, client, fake_api):
        assert client.subscribe_dom("XAUUSD")
        assert any(c[0] == "market_book_add" for c in fake_api.calls)

    def test_unsubscribe_dom(self, client, fake_api):
        client.unsubscribe_dom("XAUUSD")
        assert any(c[0] == "market_book_release" for c in fake_api.calls)

    def test_get_dom_snapshot_no_books(self, client, fake_api):
        snap = client.get_dom_snapshot("XAUUSD")
        assert snap is None

    def test_get_dom_sequential_empty(self, client, fake_api):
        snaps = client.get_dom_sequential("XAUUSD", samples=2, interval=0.01)
        assert snaps == []


class TestTicks:
    def test_get_real_ticks_empty(self, client, fake_api):
        ticks = client.get_real_ticks("XAUUSD")
        assert ticks == []

    def test_get_real_ticks_since_empty(self, client, fake_api):
        ticks = client.get_real_ticks_since("XAUUSD", datetime.now(timezone.utc))
        assert ticks == []

    def test_get_tick_delta_default(self, client, fake_api):
        delta = client.get_tick_delta("XAUUSD")
        assert delta["buy_volume"] == 0
        assert delta["sell_volume"] == 0
        assert delta["delta"] == 0


class TestCandles:
    def test_get_candles_empty(self, client, fake_api):
        candles = client.get_candles("XAUUSD", count=10)
        assert candles == []

    def test_get_candles_returns_candle_data(self, client, fake_api):
        fake_api._rates = [
            _rate(1000, open=1.0, high=1.1, low=0.9, close=1.05, tick_volume=100),
        ]
        candles = client.get_candles("XAUUSD", count=5)
        assert len(candles) == 1
        c = candles[0]
        assert c.open == 1.0
        assert c.high == 1.1
        assert c.low == 0.9
        assert c.close == 1.05
        assert c.volume == 100

    def test_get_latest_candle_time(self, client, fake_api):
        fake_api._rates = [
            _rate(1000, open=1.0, high=1.1, low=0.9, close=1.05, tick_volume=100),
        ]
        t = client.get_latest_candle_time("XAUUSD", 60)
        assert t is not None
        assert isinstance(t, datetime)

    def test_get_latest_candle_time_empty(self, client, fake_api):
        assert client.get_latest_candle_time("XAUUSD", 60) is None


class TestAccount:
    def test_get_account_info_not_connected(self, client, fake_api):
        assert client.get_account_info() is None

    def test_get_account_info(self, client, fake_api):
        client.connected = True
        info = client.get_account_info()
        assert info is not None
        assert info["login"] == 123456
        assert info["balance"] == 10000.0

    def test_get_account_info_none_from_api(self, client, fake_api):
        client.connected = True
        fake_api._account = None
        assert client.get_account_info() is None


class TestPositions:
    def test_get_positions_empty(self, client, fake_api):
        assert client.get_positions() == []

    def test_get_positions_buy(self, client, fake_api):
        fake_api._positions = [
            _pos(ticket=1, symbol="XAUUSD", type_=0, volume=0.1,
                 price_open=1900.0),
        ]
        positions = client.get_positions()
        assert len(positions) == 1
        assert positions[0]["type"] == "buy"

    def test_get_positions_sell(self, client, fake_api):
        fake_api._positions = [
            _pos(ticket=1, symbol="XAUUSD", type_=1, volume=0.1),
        ]
        positions = client.get_positions()
        assert positions[0]["type"] == "sell"

    def test_get_positions_filtered(self, client, fake_api):
        fake_api._positions = [
            _pos(ticket=1, symbol="XAUUSD"),
            _pos(ticket=2, symbol="EURUSD"),
        ]
        assert len(client.get_positions("XAUUSD")) == 1
        assert len(client.get_positions("NONEXIST")) == 0

    def test_has_open_position_true(self, client, fake_api):
        fake_api._positions = [_pos(ticket=1, symbol="XAUUSD")]
        assert client.has_open_position("XAUUSD")

    def test_has_open_position_false(self, client, fake_api):
        assert not client.has_open_position("XAUUSD")


class TestOrders:
    def test_get_pending_orders_empty(self, client, fake_api):
        assert client.get_pending_orders() == []

    def test_get_pending_orders(self, client, fake_api):
        fake_api._orders = [
            _order(ticket=10, symbol="XAUUSD", type_=2,
                   price_open=1850.0),
        ]
        orders = client.get_pending_orders()
        assert len(orders) == 1
        assert orders[0]["ticket"] == 10

    def test_get_pending_orders_filtered(self, client, fake_api):
        fake_api._orders = [
            _order(ticket=10, symbol="XAUUSD"),
            _order(ticket=11, symbol="EURUSD"),
        ]
        assert len(client.get_pending_orders("XAUUSD")) == 1
        assert len(client.get_pending_orders()) == 2


class TestModifyPosition:
    def test_modify_position_success(self, client, fake_api):
        assert client.modify_position(1, new_sl=1890.0, new_tp=1920.0)

    def test_modify_position_sends_request(self, client, fake_api):
        client.modify_position(1, new_sl=1890.0, new_tp=1920.0)
        req = [c[1] for c in fake_api.calls if c[0] == "order_send"][0]
        assert req["action"] == fake_api.TRADE_ACTION_SLTP
        assert req["position"] == 1
        assert req["sl"] == 1890.0
        assert req["tp"] == 1920.0

    def test_modify_position_retcode_fail(self, client, fake_api):
        fake_api._order_send_result = SimpleNamespace(retcode=10010, comment="Rejected")
        assert not client.modify_position(1, new_sl=1890.0)

    def test_modify_position_none_result(self, client, fake_api):
        fake_api._order_send_result = None
        assert not client.modify_position(1, new_sl=1890.0)

    def test_modify_position_default_tp(self, client, fake_api):
        client.modify_position(1, new_sl=1890.0)
        req = [c[1] for c in fake_api.calls if c[0] == "order_send"][0]
        assert req["tp"] == 0.0  # default leaves TP unchanged

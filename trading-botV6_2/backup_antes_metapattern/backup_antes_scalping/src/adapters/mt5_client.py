"""MT5 Client Adapter
Connects to MetaTrader 5 terminal for data and execution
"""
import time
import pandas as pd
from datetime import datetime
from typing import Optional
from loguru import logger

from ..core.candle_closure_ratings import CandleData
from .mt5_api import MT5API, LiveMT5API


class MT5Client:
    """MetaTrader 5 client for data retrieval and order execution"""

    def __init__(self, login: Optional[int] = None, password: Optional[str] = None,
                 server: Optional[str] = None, path: Optional[str] = None,
                 api: Optional[MT5API] = None):
        self.login = int(login) if login is not None else None
        self.password = password
        self.server = server
        self.path = path
        self.api = api or LiveMT5API()
        self.connected = False

    def connect(self) -> bool:
        self.api.shutdown()
        time.sleep(1)
        if self.path:
            if not self.api.initialize(path=self.path):
                logger.error(f"MT5 initialization failed with path {self.path}")
                return False
        elif not self.api.initialize():
            logger.error("MT5 initialization failed")
            return False

        if self.login and self.password and self.server:
            account = self.api.account_info()
            if account is not None and account.login == self.login and account.server == self.server:
                logger.info(f"Already connected to {account.server} (login={account.login})")
                self.connected = True
                return True
            if account is not None:
                logger.info(f"Switching account from {account.login}@{account.server} to {self.login}@{self.server}")
            if not self.api.login(self.login, password=self.password, server=self.server):
                logger.error(f"MT5 login failed: {self.api.last_error()}")
                return False

        self.connected = True
        logger.info("MT5 connected successfully")
        return True

    def ensure_connected(self) -> bool:
        """Reconnect if MT5 terminal disconnected"""
        if self.connected:
            try:
                info = self.api.terminal_info()
                if info is not None and info.connected:
                    return True
            except Exception:
                pass
        logger.warning("MT5 disconnected, reconnecting...")
        self.api.shutdown()
        time.sleep(1)
        return self.connect()

    def disconnect(self):
        """Shutdown MT5 connection"""
        self.api.shutdown()
        self.connected = False
        logger.info("MT5 disconnected")

    def is_connected(self) -> bool:
        """Check if MT5 is connected and terminal is online"""
        if not self.connected:
            return False
        info = self.api.terminal_info()
        return info is not None and info.connected

    SYMBOL_VARIANTS = {
        "XAUUSD": ["XAUUSD..", "XAUUSDm", "XAUUSD", "XAUUSD.cash", "XAUUSDr", "GOLD"],
        "XAGUSD": ["XAGUSD..", "XAGUSDm", "XAGUSD", "XAGUSD.cash", "XAGUSDr", "SILVER"],
    }

    def resolve_symbol(self, base: str) -> Optional[str]:
        """Try multiple symbol variants and return the first one that exists on the broker."""
        self.ensure_connected()
        candidates = self.SYMBOL_VARIANTS.get(base.upper(), [base])
        for sym in candidates:
            info = self.api.symbol_info(sym)
            if info is not None:
                self.api.symbol_select(sym, True)
                return info.name
        logger.warning(f"No variant found for {base} (tried: {candidates})")
        return None

    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """Get symbol info"""
        self.ensure_connected()
        info = self.api.symbol_info(symbol)
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
            "stoplevel": info.stoplevel,
            "trade_stops_level": info.trade_stops_level,
        }

    # ── DOM / Market Book (Modo Experto) ──

    def subscribe_dom(self, symbol: str) -> bool:
        """Subscribe to DOM updates for a symbol"""
        self.ensure_connected()
        try:
            return self.api.market_book_add(symbol)
        except Exception as e:
            logger.debug(f"market_book_add({symbol}) failed: {e}")
            return False

    def unsubscribe_dom(self, symbol: str):
        """Unsubscribe from DOM updates"""
        try:
            self.api.market_book_release(symbol)
        except Exception as e:
            logger.debug(f"market_book_release({symbol}) failed: {e}")

    def get_dom_snapshot(self, symbol: str, max_levels: int = 20) -> Optional[dict]:
        """Get full DOM snapshot via MT5 market book.
        
        Returns dict with:
          - bid_levels: [{price, volume, volume_dbl, orders}]
          - ask_levels: [{price, volume, volume_dbl, orders}]
          - bid_total, ask_total, spread, imbalance
          - timestamp
        """
        self.ensure_connected()
        try:
            self.subscribe_dom(symbol)
            time.sleep(0.05)
            books = self.api.market_book_get(symbol)
            if books is None or len(books) < 2:
                self.unsubscribe_dom(symbol)
                return None

            bid_levels = {}
            ask_levels = {}
            for b in books:
                price = round(b.price, 5)
                vol = float(getattr(b, 'volume_dbl', 0) or b.volume)
                orders = int(getattr(b, 'volume', 0))
                t = b.type
                if t in (2, 4):
                    if price not in bid_levels:
                        bid_levels[price] = {"price": price, "volume": 0.0, "volume_dbl": 0.0, "orders": 0}
                    entry = bid_levels[price]
                    entry["volume_dbl"] += vol
                    entry["volume"] += float(vol)
                    entry["orders"] += max(1, orders)
                elif t in (1, 3):
                    if price not in ask_levels:
                        ask_levels[price] = {"price": price, "volume": 0.0, "volume_dbl": 0.0, "orders": 0}
                    entry = ask_levels[price]
                    entry["volume_dbl"] += vol
                    entry["volume"] += float(vol)
                    entry["orders"] += max(1, orders)

            bid_sorted = sorted(bid_levels.values(), key=lambda x: x["price"], reverse=True)[:max_levels]
            ask_sorted = sorted(ask_levels.values(), key=lambda x: x["price"])[:max_levels]

            bid_total = sum(e["volume_dbl"] for e in bid_sorted)
            ask_total = sum(e["volume_dbl"] for e in ask_sorted)
            total = bid_total + ask_total
            imbalance = (bid_total - ask_total) / total if total > 0 else 0.0
            spread = 0.0
            if bid_sorted and ask_sorted:
                spread = ask_sorted[0]["price"] - bid_sorted[0]["price"]

            return {
                "bid_levels": bid_sorted,
                "ask_levels": ask_sorted,
                "bid_total": bid_total,
                "ask_total": ask_total,
                "spread": spread,
                "imbalance": round(imbalance, 4),
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.debug(f"DOM snapshot failed for {symbol}: {e}")
            return None
        finally:
            self.unsubscribe_dom(symbol)

    def get_dom_sequential(self, symbol: str, samples: int = 3, interval: float = 0.1) -> list:
        """Take multiple DOM snapshots sequentially for absorption/iceberg analysis"""
        snapshots = []
        for i in range(samples):
            snap = self.get_dom_snapshot(symbol)
            if snap and snap["bid_levels"] and snap["ask_levels"]:
                snapshots.append(snap)
                if i < samples - 1:
                    time.sleep(interval)
        return snapshots

    # ── Real Tick Data (Modo Experto) ──

    def get_real_ticks(self, symbol: str, count: int = 1000) -> list[dict]:
        """Fetch real tick data from MT5.
        Returns list of {time, bid, ask, last, volume, flags}
        """
        self.ensure_connected()
        try:
            ticks = self.api.copy_ticks_from(symbol, pd.Timestamp.now(), count, self.api.COPY_TICKS_ALL)
            if ticks is None or len(ticks) == 0:
                ticks = self.api.copy_ticks_from(symbol, pd.Timestamp.now() - pd.Timedelta(hours=1), count, self.api.COPY_TICKS_ALL)
        except Exception:
            try:
                ticks = self.api.copy_ticks_from(symbol, pd.Timestamp.now() - pd.Timedelta(hours=1), count, self.api.COPY_TICKS_ALL)
            except Exception:
                return []

        if ticks is None or len(ticks) == 0:
            return []
        result = []
        for t in ticks:
            result.append({
                "time": pd.Timestamp.fromtimestamp(t['time'], tz="UTC"),
                "bid": float(t['bid']),
                "ask": float(t['ask']),
                "last": float(t['last']),
                "volume": float(t['volume']),
                "flags": t['flags'],
            })
        return result

    def get_real_ticks_since(self, symbol: str, from_time: datetime, count: int = 10000) -> list[dict]:
        """Fetch ticks from a specific datetime"""
        self.ensure_connected()
        try:
            ticks = self.api.copy_ticks_from(symbol, from_time, count, self.api.COPY_TICKS_ALL)
        except Exception:
            return []
        if ticks is None:
            return []
        result = []
        for t in ticks:
            result.append({
                "time": pd.Timestamp.fromtimestamp(t['time'], tz="UTC"),
                "bid": float(t['bid']),
                "ask": float(t['ask']),
                "last": float(t['last']),
                "volume": float(t['volume']),
                "flags": t['flags'],
            })
        return result

    def get_tick_delta(self, symbol: str, count: int = 1000) -> dict:
        """Calculate buy/sell delta from real ticks.
        Returns {buy_volume, sell_volume, delta, total, buy_pct, sell_pct}
        """
        ticks = self.get_real_ticks(symbol, count)
        if not ticks:
            return {"buy_volume": 0, "sell_volume": 0, "delta": 0, "total": 0, "buy_pct": 0.5, "sell_pct": 0.5}

        buy_vol = 0.0
        sell_vol = 0.0
        for t in ticks:
            vol = t["volume"]
            if t["last"] >= t["ask"]:
                buy_vol += vol
            elif t["last"] <= t["bid"]:
                sell_vol += vol
            else:
                mid = (t["bid"] + t["ask"]) / 2
                if t["last"] > mid:
                    buy_vol += vol
                elif t["last"] < mid:
                    sell_vol += vol
                else:
                    buy_vol += vol * 0.5
                    sell_vol += vol * 0.5

        total = buy_vol + sell_vol
        delta = (buy_vol - sell_vol) / total if total > 0 else 0
        return {
            "buy_volume": round(buy_vol, 2),
            "sell_volume": round(sell_vol, 2),
            "delta": round(delta, 4),
            "total": round(total, 2),
            "buy_pct": round(buy_vol / total, 4) if total > 0 else 0.5,
            "sell_pct": round(sell_vol / total, 4) if total > 0 else 0.5,
        }

    def _ensure_symbol(self, symbol: str) -> bool:
        for attempt in range(3):
            if self.api.symbol_select(symbol, True):
                return True
            if attempt == 0:
                logger.warning(f"Could not select {symbol}, retrying...")
            time.sleep(2)
        logger.error(f"Could not select {symbol} after 3 attempts")
        return False

    def get_candles(self, symbol: str, timeframe: int = 60,
                    count: int = 100, timeout: float = 30) -> list[CandleData]:
        self.ensure_connected()
        if not self._ensure_symbol(symbol):
            return []

        from concurrent.futures import ThreadPoolExecutor, TimeoutError as _TO
        with ThreadPoolExecutor(max_workers=1) as _ex:
            _f = _ex.submit(self.api.copy_rates_from_pos, symbol, timeframe, 0, count)
            try:
                rates = _f.result(timeout=timeout)
            except _TO:
                logger.warning(f"Timeout ({timeout}s) get_candles {symbol} tf={timeframe}")
                return []
            except Exception:
                return []

        if rates is None or len(rates) == 0:
            return []

        def _get(rate, name: str, idx: int):
            try:
                return rate[name]
            except (TypeError, ValueError, IndexError):
                return rate[idx]

        candles = []
        for rate in rates:
            candle = CandleData(
                timestamp=pd.Timestamp.fromtimestamp(_get(rate, 'time', 0), tz='UTC'),
                open=float(_get(rate, 'open', 1)),
                high=float(_get(rate, 'high', 2)),
                low=float(_get(rate, 'low', 3)),
                close=float(_get(rate, 'close', 4)),
                volume=float(_get(rate, 'tick_volume', 5))
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

        info = self.api.account_info()
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
        positions = self.api.positions_get()
        if positions is None:
            return []

        result = []
        for pos in positions:
            if symbol is None or pos.symbol == symbol:
                result.append({
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "type": "buy" if pos.type == self.api.POSITION_TYPE_BUY else "sell",
                    "volume": pos.volume,
                    "price_open": pos.price_open,
                    "price_current": pos.price_current,
                    "profit": pos.profit,
                    "sl": pos.sl,
                    "tp": pos.tp,
                    "time": pd.Timestamp.fromtimestamp(pos.time, tz='UTC'),
                })

        return result

    def has_open_position(self, symbol: str) -> bool:
        """Check if symbol has an open position"""
        positions = self.get_positions(symbol)
        return len(positions) > 0

    def get_pending_orders(self, symbol: Optional[str] = None) -> list[dict]:
        """Get pending orders (limit/stop)"""
        orders = self.api.orders_get()
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
            "action": self.api.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": new_sl,
            "tp": new_tp,
            "comment": "Trailing/BE"
        }

        result = self.api.order_send(request)

        if result is None:
            logger.error(f"Position {ticket} modify failed: {self.api.last_error()}")
            return False

        if result.retcode != self.api.TRADE_RETCODE_DONE:
            logger.error(f"Position {ticket} modify failed: {result.comment}")
            return False

        logger.info(f"Position {ticket} modified: SL={new_sl}, TP={new_tp}")
        return True

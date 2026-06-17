"""Dual Order Pack Manager — 2 simultaneous positions per fractal entry
Manages entry, SL, TP1/TP2, breakeven +10, and dynamic trailing for runner.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict
from pathlib import Path
import sqlite3
import MetaTrader5 as mt5
from loguru import logger

from ..adapters.mt5_client import MT5Client
from ..utils.helpers import pip_size, atr


MAGIC = 20260521


@dataclass
class SubOrder:
    id: int = 0
    pack_id: int = 0
    position_number: int = 0
    ticket: int = 0
    direction: str = ""
    symbol: str = ""
    volume: float = 0.0
    entry_price: float = 0.0
    sl_initial: float = 0.0
    tp_target: float = 0.0
    sl_current: float = 0.0
    status: str = "active"
    profit: float = 0.0
    closed_at: Optional[datetime] = None


@dataclass
class OrderPack:
    id: int = 0
    fractal_id: int = 0
    symbol: str = ""
    direction: str = ""
    entry_price: float = 0.0
    sl_initial: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    volume_total: float = 0.0
    volume_per: float = 0.0
    status: str = "active"
    breakeven_activated: bool = False
    trailing_activated: bool = False
    created_at: Optional[datetime] = None
    source_timeframe: str = ""


class OrderPackManager:
    def __init__(self, mt5_client: MT5Client, symbol: str, copy_enabled: bool = True):
        self.mt5_client = mt5_client
        self.symbol = symbol
        self.pip = pip_size(symbol)
        self._copy_enabled = copy_enabled
        self._executor = None
        db_path = Path(__file__).parent.parent.parent / "data" / "db" / symbol / "order_packs.db"
        db_path.parent.mkdir(exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._init_db()
        self._packs: Dict[int, OrderPack] = {}
        self._subs: Dict[int, List[SubOrder]] = {}
        self._peak_prices: Dict[int, float] = {}
        self._load_active()
        self._init_peaks()

    def _init_db(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS order_packs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fractal_id INTEGER, symbol TEXT, direction TEXT,
                entry_price REAL, sl_initial REAL,
                tp1 REAL, tp2 REAL,
                volume_total REAL, volume_per REAL,
                status TEXT DEFAULT 'active',
                breakeven_activated INTEGER DEFAULT 0,
                trailing_activated INTEGER DEFAULT 0,
                source_timeframe TEXT DEFAULT '',
                created_at TEXT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS sub_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pack_id INTEGER, position_number INTEGER,
                ticket INTEGER, direction TEXT, symbol TEXT,
                volume REAL, entry_price REAL,
                sl_initial REAL, tp_target REAL, sl_current REAL,
                status TEXT DEFAULT 'active',
                profit REAL DEFAULT 0,
                closed_at TEXT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS copy_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                symbol TEXT,
                direction TEXT,
                volume REAL,
                price REAL,
                sl REAL,
                tp REAL,
                pack_id INTEGER,
                sub_number INTEGER DEFAULT 0,
                executed INTEGER DEFAULT 0,
                real_ticket INTEGER,
                error TEXT,
                created_at TEXT,
                executed_at TEXT
            )
        """)
        self._conn.commit()

    def _load_active(self):
        rows = self._conn.execute(
            "SELECT * FROM order_packs WHERE symbol=? AND status='active'",
            (self.symbol,)
        ).fetchall()
        for r in rows:
            tp1_val = r[6] if len(r) > 6 else 0
            tp2_val = r[7] if len(r) > 7 else 0
            vol_total = r[8] if len(r) > 8 else 0
            vol_per = r[9] if len(r) > 9 else 0
            p = OrderPack(id=r[0], fractal_id=r[1], symbol=r[2], direction=r[3],
                          entry_price=r[4], sl_initial=r[5], tp1=tp1_val, tp2=tp2_val,
                          volume_total=vol_total, volume_per=vol_per,
                          status=r[10],
                          breakeven_activated=bool(r[11]) if len(r) > 11 else False,
                          trailing_activated=bool(r[12]) if len(r) > 12 else False,
                          source_timeframe=r[13] if len(r) > 13 else "",
                           created_at=self._parse_datetime(r[14]) if len(r) > 14 else None)
            self._packs[p.id] = p
            sub_rows = self._conn.execute(
                "SELECT * FROM sub_orders WHERE pack_id=?", (p.id,)
            ).fetchall()
            subs = []
            for s in sub_rows:
                sub = SubOrder(id=s[0], pack_id=s[1], position_number=s[2],
                               ticket=s[3], direction=s[4], symbol=s[5],
                               volume=s[6], entry_price=s[7],
                               sl_initial=s[8], tp_target=s[9], sl_current=s[10],
                               status=s[11], profit=s[12],
                               closed_at=self._parse_datetime(s[13]))
                subs.append(sub)
            self._subs[p.id] = subs

    def _init_peaks(self):
        for subs in self._subs.values():
            for sub in subs:
                if sub.status == "active" and sub.ticket != 0:
                    pos = self._get_position(sub.ticket)
                    if pos:
                        self._peak_prices[sub.ticket] = pos["price_current"]

    def _write_signal(self, action: str, pack_id: int, sub_number: int = 0,
                       symbol: str = "", direction: str = "",
                       volume: float = 0.0, price: float = 0.0,
                       sl: float = 0.0, tp: float = 0.0):
        if not self._copy_enabled:
            return
        now = datetime.utcnow().isoformat()
        self._conn.execute("""
            INSERT INTO copy_signals
                (action, symbol, direction, volume, price, sl, tp,
                 pack_id, sub_number, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (action, symbol or self.symbol, direction, volume, price,
              sl, tp, pack_id, sub_number, now))
        self._conn.commit()

    def _place_limit_order(self, direction: str, volume: float, entry_price: float,
                           sl: float, tp: float, comment: str) -> Optional[int]:
        _ = tp  # TP se asigna cuando el LIMIT se ejecuta
        direction = direction.upper()
        order_type = mt5.ORDER_TYPE_BUY_LIMIT if direction == "BUY" else mt5.ORDER_TYPE_SELL_LIMIT
        info = self.mt5_client.get_symbol_info(self.symbol)
        if not info:
            logger.error(f"[{self.symbol}] No symbol info")
            return None
        tick_size = info.get("trade_tick_size", info.get("point", 0.0001))
        if tick_size <= 0:
            tick_size = info.get("point", 0.0001)

        price_adj = round(entry_price / tick_size) * tick_size
        sl_adj = round(sl / tick_size) * tick_size

        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": self.symbol,
            "volume": volume,
            "type": order_type,
            "price": price_adj,
            "sl": sl_adj,
            "tp": 0.0,
            "deviation": 10,
            "magic": MAGIC,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"[{self.symbol}] LIMIT {direction} {volume}@{price_adj} SL={sl_adj} → ticket={result.order}")
            return result.order
        logger.error(f"[{self.symbol}] LIMIT {direction} failed: {result.retcode if result else 'no result'} {result.comment if result else ''}")
        return None

    @staticmethod
    def _parse_datetime(val):
        if val is None:
            return None
        if isinstance(val, datetime):
            return val
        if isinstance(val, str):
            try:
                return datetime.fromisoformat(val)
            except ValueError:
                return None
        return None

    def place_pack(self, fractal_id: int, direction: str, entry_price: float,
                   sl_price: float, timeframe: str,
                   volume_total: float = 0.03) -> Optional[OrderPack]:
        for p in self._packs.values():
            if p.fractal_id == fractal_id and p.status == "active":
                logger.warning(f"[{self.symbol}] Pack ya activo para fractal #{fractal_id}, saltando")
                return None
        direction = direction.upper()
        is_buy = direction == "BUY"
        risk_dist = abs(entry_price - sl_price)
        if risk_dist == 0:
            logger.error(f"[{self.symbol}] Risk distance zero, cannot place pack")
            return None

        tp1_price = entry_price + risk_dist if is_buy else entry_price - risk_dist
        tp2_price = entry_price + 2 * risk_dist if is_buy else entry_price - 2 * risk_dist
        vol_per = round(volume_total / 2, 2)

        now = datetime.utcnow()
        cur = self._conn.execute("""
            INSERT INTO order_packs (fractal_id, symbol, direction, entry_price,
                sl_initial, tp1, tp2, volume_total, volume_per, status,
                source_timeframe, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (fractal_id, self.symbol, direction, entry_price, sl_price,
              tp1_price, tp2_price, volume_total, vol_per,
              "active", timeframe, now.isoformat()))
        self._conn.commit()
        pack_id = cur.lastrowid

        pack = OrderPack(id=pack_id, fractal_id=fractal_id, symbol=self.symbol,
                         direction=direction, entry_price=entry_price,
                         sl_initial=sl_price, tp1=tp1_price, tp2=tp2_price,
                         volume_total=volume_total,
                         volume_per=vol_per, status="active", created_at=now,
                         source_timeframe=timeframe)
        self._packs[pack_id] = pack

        tickets = []
        tp_targets = [tp1_price, tp2_price]
        for i in range(2):
            ticket = self._place_limit_order(
                direction, vol_per, entry_price, sl_price, tp_targets[i],
                f"F{fractal_id}P{i+1}"
            )
            tickets.append(ticket or 0)

        # Write signals BEFORE sub_orders commit so CopyTrader gets them
        for i in range(2):
            self._write_signal("PLACE_LIMIT", pack_id, i + 1,
                               direction=direction, volume=vol_per,
                               price=entry_price, sl=sl_price, tp=tp_targets[i])

        subs = []
        for i in range(2):
            sub = SubOrder(pack_id=pack_id, position_number=i + 1,
                          ticket=tickets[i], direction=direction,
                          symbol=self.symbol, volume=vol_per,
                          entry_price=entry_price, sl_initial=sl_price,
                          tp_target=tp_targets[i], sl_current=sl_price)
            self._conn.execute("""
                INSERT INTO sub_orders (pack_id, position_number, ticket,
                    direction, symbol, volume, entry_price, sl_initial,
                    tp_target, sl_current)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (pack_id, i + 1, tickets[i], direction, self.symbol, vol_per,
                  entry_price, sl_price, tp_targets[i], sl_price))
            subs.append(sub)

        # Retry commit up to 3 times to handle concurrent access
        for attempt in range(3):
            try:
                self._conn.commit()
                break
            except Exception as e:
                if attempt < 2:
                    import time
                    time.sleep(0.2 * (attempt + 1))
                else:
                    logger.error(f"[{self.symbol}] Commit falló tras 3 intentos: {e}")
                    # Cancel MT5 orders to avoid orphans
                    for t in tickets:
                        if t:
                            mt5.order_delete(t)
                    return None

        self._subs[pack_id] = subs

        logger.info(f"[{self.symbol}] Pack #{pack_id} {direction} @ {entry_price:.2f} "
                     f"SL={sl_price:.2f} | 1:1={tp1_price:.2f} 1:2={tp2_price:.2f}")

        return pack

    def _is_pending(self, ticket: int) -> bool:
        if ticket == 0:
            return False
        orders = mt5.orders_get(ticket=ticket)
        return orders is not None and len(orders) > 0

    def _build_active_assigned(self) -> set:
        """Return set of position tickets already assigned to any active pack."""
        assigned = set()
        for pid, pk in self._packs.items():
            if pk.status != "active":
                continue
            for s in self._subs.get(pid, []):
                if s.ticket:
                    pos = mt5.positions_get(ticket=s.ticket)
                    if pos and len(pos) > 0:
                        assigned.add(s.ticket)
        return assigned

    def _sync_pending_fills(self):
        """Detect LIMIT orders that have filled — update sub.ticket, sin TP."""
        all_assigned = self._build_active_assigned()
        for pack_id, pack in list(self._packs.items()):
            if pack.status != "active":
                continue
            subs = self._subs.get(pack_id, [])
            for sub in subs:
                if sub.status != "active" or sub.ticket == 0:
                    continue
                if self._is_pending(sub.ticket):
                    continue
                if sub.ticket in all_assigned:
                    continue
                positions = mt5.positions_get(symbol=self.symbol)
                if not positions:
                    continue
                is_buy = pack.direction == "BUY"
                mt5_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
                for p in positions:
                    if p.magic != MAGIC or p.type != mt5_type:
                        continue
                    if p.ticket in all_assigned:
                        continue
                    if abs(p.volume - sub.volume) > 0.001:
                        continue
                    sub.ticket = p.ticket
                    sub.tp_target = 0.0
                    all_assigned.add(p.ticket)
                    self._peak_prices[p.ticket] = p.price_current
                    self._conn.execute(
                        "UPDATE sub_orders SET ticket=? WHERE pack_id=? AND position_number=?",
                        (p.ticket, pack_id, sub.position_number)
                    )
                    self._conn.commit()
                    logger.info(f"[{self.symbol}] Pack #{pack.id} P{sub.position_number} filled → "
                               f"pos ticket={p.ticket}, sin TP")
                    break

    def manage_all(self, current_time: datetime, df_5m):
        atr_val = atr(df_5m, 14).iloc[-1] if df_5m is not None and len(df_5m) > 14 else 0
        self._sync_pending_fills()
        for pack_id, pack in list(self._packs.items()):
            if pack.status != "active":
                continue
            subs = self._subs.get(pack_id, [])
            if not subs:
                continue
            self._check_sl_hit(pack, subs)
            self._check_breakeven(pack, subs)
            if pack.breakeven_activated:
                self._check_trailing(pack, subs, df_5m, atr_val)

    def _get_position(self, ticket: int) -> Optional[dict]:
        if ticket == 0:
            return None
        positions = mt5.positions_get(ticket=ticket)
        if positions is not None and len(positions) > 0:
            p = positions[0]
            return {
                "ticket": p.ticket,
                "symbol": p.symbol,
                "type": p.type,
                "volume": p.volume,
                "price_open": p.price_open,
                "sl": p.sl,
                "tp": p.tp,
                "price_current": p.price_current,
                "profit": p.profit,
            }
        return None

    def _check_sl_hit(self, pack: OrderPack, subs: List[SubOrder]):
        for sub in subs:
            if sub.status != "active" or sub.ticket == 0:
                continue
            pos = self._get_position(sub.ticket)
            if pos is None:
                continue
            current_price = pos["price_current"]
            is_buy = pack.direction == "BUY"
            sl_hit = (is_buy and current_price <= sub.sl_current) or \
                     (not is_buy and current_price >= sub.sl_current)
            if sl_hit:
                sub.status = "closed_by_sl"
                sub.profit = pos.get("profit", 0)
                sub.closed_at = datetime.utcnow()
                self._update_sub_status(sub)
                self._peak_prices.pop(sub.ticket, None)

        all_closed = all(s.status != "active" for s in subs)
        if all_closed:
            pack.status = "closed"
            self._update_pack_status(pack)
            logger.info(f"[{self.symbol}] Pack #{pack.id} fully closed")

    def _check_breakeven(self, pack: OrderPack, subs: List[SubOrder]):
        if pack.breakeven_activated:
            return
        for sub in subs:
            if sub.status != "active" or sub.ticket == 0:
                continue
            pos = self._get_position(sub.ticket)
            if pos is None:
                continue
            current_price = pos["price_current"]
            is_buy = pack.direction == "BUY"
            prev_peak = self._peak_prices.get(sub.ticket, current_price)
            peak_price = max(current_price, prev_peak) if is_buy else min(current_price, prev_peak)
            self._peak_prices[sub.ticket] = peak_price
            be_price = pack.entry_price + (400 * self.pip) if is_buy else pack.entry_price - (400 * self.pip)
            reached = (is_buy and peak_price >= be_price) or \
                      (not is_buy and peak_price <= be_price)
            if reached:
                self._apply_breakeven(pack, subs)
                break

    def _apply_breakeven(self, pack: OrderPack, subs: List[SubOrder]):
        is_buy = pack.direction == "BUY"
        distance = 300 * self.pip

        current_price = None
        for sub in subs:
            if sub.status == "active" and sub.ticket != 0:
                pos = self._get_position(sub.ticket)
                if pos:
                    current_price = pos["price_current"]
                    break
        if current_price is None:
            return

        new_sl = current_price - distance if is_buy else current_price + distance

        for sub in subs:
            if sub.status != "active" or sub.ticket == 0:
                continue
            if sub.ticket:
                self.mt5_client.modify_position(sub.ticket, new_sl, 0.0)
                sub.sl_current = new_sl
                self._update_sub_sl(sub)

        pack.breakeven_activated = True
        self._conn.execute(
            "UPDATE order_packs SET breakeven_activated=1 WHERE id=?", (pack.id,)
        )
        self._conn.commit()
        logger.info(f"[{self.symbol}] Pack #{pack.id} breakeven activado → SL={new_sl:.2f} (trailing {distance:.2f} pts detrás)")

        for sub in subs:
            if sub.status == "active" and sub.ticket != 0:
                self._write_signal("MODIFY_SLTP", pack.id, sub.position_number,
                                   sl=new_sl)

    def _check_trailing(self, pack: OrderPack, subs: List[SubOrder], df_5m, atr_val):
        _ = atr_val, df_5m
        is_buy = pack.direction == "BUY"
        distance = 300 * self.pip

        for sub in subs:
            if sub.status != "active" or sub.ticket == 0:
                continue
            pos = self._get_position(sub.ticket)
            if pos is None:
                continue

            current_price = pos["price_current"]
            new_sl = current_price - distance if is_buy else current_price + distance

            better_sl = (new_sl > sub.sl_current) if is_buy else (new_sl < sub.sl_current)
            if better_sl:
                self.mt5_client.modify_position(sub.ticket, new_sl, 0.0)
                sub.sl_current = new_sl
                self._update_sub_sl(sub)
                pack.trailing_activated = True
                self._write_signal("MODIFY_SLTP", pack.id, sub.position_number,
                                   sl=new_sl)

        if pack.trailing_activated:
            self._conn.execute(
                "UPDATE order_packs SET trailing_activated=1 WHERE id=?", (pack.id,)
            )
            self._conn.commit()

    def _update_sub_status(self, sub: SubOrder):
        self._conn.execute("""
            UPDATE sub_orders SET status=?, profit=?, closed_at=?
            WHERE id=?
        """, (sub.status, sub.profit,
              sub.closed_at.isoformat() if sub.closed_at else None, sub.id))
        self._conn.commit()

    def _update_sub_sl(self, sub: SubOrder):
        self._conn.execute(
            "UPDATE sub_orders SET sl_current=?, tp_target=? WHERE id=?",
            (sub.sl_current, sub.tp_target, sub.id)
        )
        self._conn.commit()

    def _update_pack_status(self, pack: OrderPack):
        self._conn.execute(
            "UPDATE order_packs SET status=? WHERE id=?", (pack.status, pack.id)
        )
        self._conn.commit()

    def get_active_packs(self) -> List[OrderPack]:
        return [p for p in self._packs.values() if p.status == "active"]

    def get_pack_by_id(self, pack_id: int) -> Optional[OrderPack]:
        return self._packs.get(pack_id)

    def get_pack_total_profit(self, pack_id: int) -> float:
        subs = self._subs.get(pack_id, [])
        return sum(s.profit for s in subs if s.status != "active")

    def get_all_pack_ids(self) -> List[int]:
        return list(self._packs.keys())

    def get_all_subs(self, pack_id: int) -> List[SubOrder]:
        return self._subs.get(pack_id, [])

    def cancel_pack(self, pack_id: int):
        """Cancela todas las órdenes pendientes de un pack y lo marca como cerrado."""
        pack = self._packs.get(pack_id)
        if not pack:
            return
        subs = self._subs.get(pack_id, [])
        for sub in subs:
            self._write_signal("CANCEL", pack_id, sub.position_number)
            if sub.status == "active" and sub.ticket != 0:
                result = mt5.order_send({
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order": sub.ticket,
                })
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"[{self.symbol}] Cancelada orden {sub.ticket} del pack #{pack_id}")
                sub.status = "cancelled"
                self._update_sub_status(sub)
        pack.status = "closed"
        self._update_pack_status(pack)

    def get_pack_summary(self) -> List[dict]:
        result = []
        for p in self.get_active_packs():
            subs = self._subs.get(p.id, [])
            result.append({
                "pack_id": p.id,
                "fractal_id": p.fractal_id,
                "direction": p.direction,
                "entry": p.entry_price,
                "sl": p.sl_initial,
                "tp1": p.tp1, "tp2": p.tp2,
                "be": p.breakeven_activated,
                "trailing": p.trailing_activated,
                "source_tf": p.source_timeframe,
                "subs": [{"n": s.position_number, "ticket": s.ticket,
                          "status": s.status, "sl": s.sl_current} for s in subs]
            })
        return result

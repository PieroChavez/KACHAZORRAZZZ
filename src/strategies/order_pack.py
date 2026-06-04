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
    def __init__(self, mt5_client: MT5Client, symbol: str):
        self.mt5_client = mt5_client
        self.symbol = symbol
        self.pip = pip_size(symbol)
        self._executor = None
        db_path = Path(__file__).parent.parent.parent / "data" / "order_packs.db"
        db_path.parent.mkdir(exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._init_db()
        self._packs: Dict[int, OrderPack] = {}
        self._subs: Dict[int, List[SubOrder]] = {}
        self._load_active()

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
        self._conn.commit()

    def _load_active(self):
        rows = self._conn.execute(
            "SELECT * FROM order_packs WHERE symbol=? AND status='active'",
            (self.symbol,)
        ).fetchall()
        for r in rows:
            tp1_val = r[6] if len(r) > 6 else 0
            tp2_val = r[7] if len(r) > 7 else 0
            # r[8] is tp3 (added later, shifts indices)
            vol_total = r[9] if len(r) > 9 else r[8]
            vol_per = r[10] if len(r) > 10 else r[9]
            p = OrderPack(id=r[0], fractal_id=r[1], symbol=r[2], direction=r[3],
                          entry_price=r[4], sl_initial=r[5], tp1=tp1_val, tp2=tp2_val,
                          volume_total=vol_total, volume_per=vol_per,
                          status=r[11] if len(r) > 11 else r[10],
                          breakeven_activated=bool(r[12] if len(r) > 12 else r[11]),
                          trailing_activated=bool(r[13] if len(r) > 13 else r[12]),
                          source_timeframe=r[14] if len(r) > 14 else r[13],
                          created_at=datetime.fromisoformat(r[15] if len(r) > 15 else r[14]) if (r[15] if len(r) > 15 else r[14]) else None)
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
                               closed_at=datetime.fromisoformat(s[13]) if s[13] else None)
                subs.append(sub)
            self._subs[p.id] = subs

    def _place_limit_order(self, direction: str, volume: float, entry_price: float,
                           sl: float, tp: float, comment: str) -> Optional[int]:
        _ = tp  # TP se asigna cuando el LIMIT se ejecuta
        direction = direction.upper()
        order_type = mt5.ORDER_TYPE_BUY_LIMIT if direction == "BUY" else mt5.ORDER_TYPE_SELL_LIMIT
        info = self.mt5_client.get_symbol_info(self.symbol)
        if not info:
            logger.error(f"[{self.symbol}] No symbol info")
            return None
        digits = info["digits"]

        price_adj = round(entry_price, digits)
        sl_adj = round(sl, digits)

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

    def place_pack(self, fractal_id: int, direction: str, entry_price: float,
                   sl_price: float, timeframe: str,
                   volume_total: float = 0.03) -> Optional[OrderPack]:
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
        self._conn.commit()
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
        """Detect LIMIT orders that have filled — update sub.ticket y asigna TP."""
        all_assigned = self._build_active_assigned()
        for pack_id, pack in list(self._packs.items()):
            if pack.status != "active":
                continue
            subs = self._subs.get(pack_id, [])
            # Tickets ya asignados a posiciones de este pack
            pack_assigned = set()
            for s in subs:
                if s.ticket:
                    pos = mt5.positions_get(ticket=s.ticket)
                    if pos and len(pos) > 0:
                        pack_assigned.add(s.ticket)
            for sub in subs:
                if sub.status != "active" or sub.ticket == 0:
                    continue
                if self._is_pending(sub.ticket):
                    continue  # aun no se ejecuta
                if sub.ticket in all_assigned:
                    continue  # ya asignado a otro pack activo
                # La orden ya no esta pendiente — buscar la posicion
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
                    # Coincidencia encontrada
                    sub.ticket = p.ticket
                    entries = {1: pack.tp1, 2: pack.tp2}
                    tp_price = entries.get(sub.position_number, pack.tp2)
                    self.mt5_client.modify_position(p.ticket, sub.sl_current, tp_price)
                    sub.tp_target = tp_price
                    all_assigned.add(p.ticket)
                    self._conn.execute(
                        "UPDATE sub_orders SET ticket=?, tp_target=? WHERE pack_id=? AND position_number=?",
                        (p.ticket, tp_price, pack_id, sub.position_number)
                    )
                    self._conn.commit()
                    logger.info(f"[{self.symbol}] Pack #{pack.id} P{sub.position_number} filled → "
                               f"pos ticket={p.ticket}, TP={tp_price:.2f}")
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
            self._check_tp_hit(pack, subs)
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

    def _check_tp_hit(self, pack: OrderPack, subs: List[SubOrder]):
        for sub in subs:
            if sub.status != "active" or sub.ticket == 0:
                continue
            pos = self._get_position(sub.ticket)
            if pos is None:
                if self._is_pending(sub.ticket):
                    continue  # LIMIT aun pendiente
                sub.status = "closed_manual"
                self._update_sub_status(sub)
                continue
            current_price = pos["price_current"]
            is_buy = pack.direction == "BUY"
            tp_hit = (is_buy and current_price >= sub.tp_target) or \
                     (not is_buy and current_price <= sub.tp_target)
            if tp_hit:
                sub.status = "closed_by_tp"
                sub.profit = pos.get("profit", 0)
                sub.closed_at = datetime.utcnow()
                self._update_sub_status(sub)
                logger.info(f"[{self.symbol}] Pack #{pack.id} P{sub.position_number} TP hit")

                if sub.position_number == 1:
                    pack.breakeven_activated = True
                    self._apply_breakeven(pack, subs)

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
            be_price = pack.entry_price + (10 * self.pip) if is_buy else pack.entry_price - (10 * self.pip)
            reached = (is_buy and current_price >= be_price) or \
                      (not is_buy and current_price <= be_price)
            if reached:
                self._apply_breakeven(pack, subs)
                break

    def _apply_breakeven(self, pack: OrderPack, subs: List[SubOrder]):
        is_buy = pack.direction == "BUY"
        be_sl = pack.entry_price + (10 * self.pip) if is_buy else pack.entry_price - (10 * self.pip)
        tp1_risk = abs(pack.tp1 - pack.entry_price)
        assurance_sl = pack.entry_price + tp1_risk if is_buy else pack.entry_price - tp1_risk

        for sub in subs:
            if sub.status != "active" or sub.ticket == 0:
                continue
            # Posición 1 va a BE+10; runner (pos 2) va a assurance (1:1)
            if sub.position_number == 1:
                new_sl = be_sl
            else:
                new_sl = assurance_sl if pack.breakeven_activated else be_sl
            if sub.ticket:
                self.mt5_client.modify_position(sub.ticket, new_sl, 0.0)
                sub.sl_current = new_sl
                self._update_sub_sl(sub)

        pack.breakeven_activated = True
        self._conn.execute(
            "UPDATE order_packs SET breakeven_activated=1 WHERE id=?", (pack.id,)
        )
        self._conn.commit()
        logger.info(f"[{self.symbol}] Pack #{pack.id} BE+10 → SL={be_sl:.2f} (runner asegurado en {assurance_sl:.2f})")

    def _check_trailing(self, pack: OrderPack, subs: List[SubOrder], df_5m, atr_val):
        _ = atr_val, df_5m
        is_buy = pack.direction == "BUY"
        entry = pack.entry_price
        pip = self.pip

        # --- Averiguar cuántas posiciones activas quedan ---
        active = [s for s in subs if s.status == "active" and s.ticket != 0 and self._get_position(s.ticket)]
        if not active:
            return

        # Precio actual del mercado (del primer activo)
        first_pos = self._get_position(active[0].ticket)
        if first_pos is None:
            return
        price = first_pos["price_current"]

        # Determinar si TP1 ya fue alcanzado (posición 1 ya no está activa)
        tp1_hit = not any(s.position_number == 1 and s.status == "active" and self._get_position(s.ticket)
                          for s in subs)
        # Determinar si solo queda el runner
        only_runner = len(active) == 1 and active[0].position_number == 2

        for sub in active:
            raw_profit = (price - entry) if is_buy else (entry - price)

            # --- Runner dinámico: SL persigue el precio sin esperar ---
            if sub.position_number == 2 and (tp1_hit or only_runner):
                # Runner: lockea 50% del profit actual, mínimo 10 pips
                lock_ratio = 0.50
                locked = max(raw_profit * lock_ratio, 10 * pip)
                new_sl = entry + locked if is_buy else entry - locked
                better_sl = (new_sl > sub.sl_current) if is_buy else (new_sl < sub.sl_current)
                if better_sl and sub.ticket:
                    self.mt5_client.modify_position(sub.ticket, new_sl, 0.0)
                    sub.sl_current = new_sl
                    self._update_sub_sl(sub)
                    pack.trailing_activated = True

            # --- Posición 1 (scalp): trailing suave solo si está activa ---
            elif sub.position_number == 1:
                lock_ratio = 0.20
                locked = max(raw_profit * lock_ratio, 10 * pip)
                new_sl = entry + locked if is_buy else entry - locked
                better_sl = (new_sl > sub.sl_current) if is_buy else (new_sl < sub.sl_current)
                if better_sl and sub.ticket:
                    self.mt5_client.modify_position(sub.ticket, new_sl, 0.0)
                    sub.sl_current = new_sl
                    self._update_sub_sl(sub)
                    pack.trailing_activated = True

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

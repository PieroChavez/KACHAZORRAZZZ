"""CopyTrader — Replica señales del bot DEMO a una cuenta REAL
Ejecutar como proceso separado: python scripts/copy_trader.py
Lee de la tabla copy_signals en order_packs.db y ejecuta en la cuenta REAL.
"""
import os
import sys
import time
import sqlite3
from datetime import datetime
from pathlib import Path

_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

from dotenv import load_dotenv
env_file = _proj_root / ".env"
if env_file.exists():
    load_dotenv(env_file)

import MetaTrader5 as mt5
from loguru import logger

SYMBOL = "XAUUSDm"
SYMBOL_MAP = {"XAUUSDm": "XAUUSDc"}
POLL_INTERVAL = 2
MAGIC = 20260521


def setup_logging():
    logger.remove()
    log_dir = _proj_root / "logs"
    log_dir.mkdir(exist_ok=True)
    logger.add(log_dir / "copy_trader_{time}.log", rotation="00:00", retention="7 days",
               level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    logger.add(sys.stderr, level="INFO")


def connect_real() -> bool:
    login = os.environ.get("REAL_MT5_LOGIN")
    password = os.environ.get("REAL_MT5_PASSWORD")
    server = os.environ.get("REAL_MT5_SERVER")
    path = os.environ.get("REAL_MT5_PATH") or os.environ.get("MT5_PATH")
    if not all([login, password, server]):
        logger.error("Faltan credenciales REAL en .env (REAL_MT5_LOGIN, REAL_MT5_PASSWORD, REAL_MT5_SERVER)")
        return False
    mt5.shutdown()
    time.sleep(1)
    ok = mt5.initialize(path=path) if path else mt5.initialize()
    if not ok:
        logger.error(f"MT5 initialize failed: {mt5.last_error()}")
        return False
    account = mt5.account_info()
    if account is not None and account.login == int(login):
        logger.info(f"Ya autenticado como {account.login}@{account.server}")
        return True
    if not mt5.login(int(login), password=password, server=server):
        logger.error(f"Login REAL falló: {mt5.last_error()}")
        mt5.shutdown()
        return False
    logger.info(f"Conectado a REAL: {login}@{server}")
    return True


def real_symbol(demo_symbol: str) -> str:
    return SYMBOL_MAP.get(demo_symbol, demo_symbol)


def get_db() -> sqlite3.Connection:
    db_path = _proj_root / "data" / "db" / SYMBOL / "order_packs.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


def get_pending_signals(conn: sqlite3.Connection) -> list:
    return conn.execute(
        "SELECT * FROM copy_signals WHERE executed=0 ORDER BY id"
    ).fetchall()


def get_real_ticket(conn: sqlite3.Connection, pack_id: int, sub_number: int) -> int:
    row = conn.execute(
        "SELECT real_ticket FROM copy_signals "
        "WHERE pack_id=? AND sub_number=? AND action='PLACE_LIMIT' AND executed=1 "
        "AND real_ticket IS NOT NULL AND real_ticket > 0 "
        "ORDER BY id DESC LIMIT 1",
        (pack_id, sub_number)
    ).fetchone()
    return row["real_ticket"] if row else 0


def _resolve_position_ticket(pack_id: int, sub_number: int, fallback_ticket: int) -> int:
    """Encuentra el ticket activo (posición o pendiente) para un pack/sub.
    En MT5, cuando un LIMIT se ejecuta, el ticket de la posición es diferente
    al ticket de la orden pendiente. Buscamos por comentario y magic.
    """
    comment = f"COPY_F{pack_id}P{sub_number}"
    sym = real_symbol(SYMBOL)

    # Buscar en posiciones abiertas
    positions = mt5.positions_get(group=f"*{sym}*")
    if positions:
        for p in positions:
            if p.magic == MAGIC and comment in (p.comment or ""):
                return p.ticket

    # Buscar en órdenes pendientes
    orders = mt5.orders_get(symbol=sym)
    if orders:
        for o in orders:
            if o.magic == MAGIC and comment in (o.comment or ""):
                return o.ticket

    return fallback_ticket


def execute_place_limit(signal: dict, conn: sqlite3.Connection) -> bool:
    demo_sym = signal["symbol"]
    symbol = real_symbol(demo_sym)
    direction = signal["direction"]
    volume = signal["volume"]
    price = signal["price"]
    sl = signal["sl"]
    pack_id = signal["pack_id"]
    sub_number = signal["sub_number"]

    order_type = mt5.ORDER_TYPE_BUY_LIMIT if direction.upper() == "BUY" else mt5.ORDER_TYPE_SELL_LIMIT
    info = mt5.symbol_info(symbol)
    if info is None:
        logger.error(f"[SIG#{signal['id']}] No symbol info for {symbol}")
        return False
    mt5.symbol_select(symbol, True)
    tick_size = info.trade_tick_size or info.point or 0.0001
    price_adj = round(price / tick_size) * tick_size
    sl_adj = round(sl / tick_size) * tick_size

    request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price_adj,
        "sl": sl_adj,
        "tp": 0.0,
        "deviation": 10,
        "magic": MAGIC,
        "comment": f"COPY_F{pack_id}P{sub_number}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"[SIG#{signal['id']}] PLACE_LIMIT {direction} {volume}@{price_adj} REAL ticket={result.order}")
        conn.execute(
            "UPDATE copy_signals SET executed=1, real_ticket=?, executed_at=? WHERE id=?",
            (result.order, datetime.utcnow().isoformat(), signal["id"])
        )
        conn.commit()
        return True
    err = result.comment if result else str(mt5.last_error())
    logger.error(f"[SIG#{signal['id']}] PLACE_LIMIT falló: {err}")
    conn.execute(
        "UPDATE copy_signals SET executed=-1, error=?, executed_at=? WHERE id=?",
        (err, datetime.utcnow().isoformat(), signal["id"])
    )
    conn.commit()
    return False


def execute_modify_sltp(signal: dict, conn: sqlite3.Connection) -> bool:
    pack_id = signal["pack_id"]
    sub_number = signal["sub_number"]
    new_sl = signal["sl"]

    fallback = get_real_ticket(conn, pack_id, sub_number)
    real_ticket = _resolve_position_ticket(pack_id, sub_number, fallback)

    if not real_ticket:
        logger.warning(f"[SIG#{signal['id']}] No REAL ticket para pack={pack_id} sub={sub_number}")
        conn.execute(
            "UPDATE copy_signals SET executed=-1, error='No REAL ticket mapping', executed_at=? WHERE id=?",
            (datetime.utcnow().isoformat(), signal["id"])
        )
        conn.commit()
        return False

    result = mt5.order_send({
        "action": mt5.TRADE_ACTION_SLTP,
        "position": real_ticket,
        "sl": new_sl,
        "tp": 0.0,
    })
    if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"[SIG#{signal['id']}] MODIFY_SLTP ticket={real_ticket} SL={new_sl}")
        conn.execute(
            "UPDATE copy_signals SET executed=1, real_ticket=?, executed_at=? WHERE id=?",
            (real_ticket, datetime.utcnow().isoformat(), signal["id"])
        )
        conn.commit()
        return True

    err = result.comment if result else str(mt5.last_error())
    logger.warning(f"[SIG#{signal['id']}] MODIFY_SLTP ticket={real_ticket} falló: {err}")
    conn.execute(
        "UPDATE copy_signals SET executed=-1, error=?, executed_at=? WHERE id=?",
        (err, datetime.utcnow().isoformat(), signal["id"])
    )
    conn.commit()
    return False


def execute_cancel(signal: dict, conn: sqlite3.Connection) -> bool:
    pack_id = signal["pack_id"]
    sub_number = signal["sub_number"]

    fallback = get_real_ticket(conn, pack_id, sub_number)
    real_ticket = _resolve_position_ticket(pack_id, sub_number, fallback)

    if not real_ticket:
        conn.execute(
            "UPDATE copy_signals SET executed=-1, error='No REAL ticket mapping', executed_at=? WHERE id=?",
            (datetime.utcnow().isoformat(), signal["id"])
        )
        conn.commit()
        return False

    result = mt5.order_send({
        "action": mt5.TRADE_ACTION_REMOVE,
        "order": real_ticket,
    })
    if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"[SIG#{signal['id']}] CANCEL ticket={real_ticket}")
        conn.execute(
            "UPDATE copy_signals SET executed=1, real_ticket=?, executed_at=? WHERE id=?",
            (real_ticket, datetime.utcnow().isoformat(), signal["id"])
        )
        conn.commit()
        return True

    err = result.comment if result else str(mt5.last_error())
    logger.warning(f"[SIG#{signal['id']}] CANCEL ticket={real_ticket} falló: {err}")
    conn.execute(
        "UPDATE copy_signals SET executed=-1, error=?, executed_at=? WHERE id=?",
        (err, datetime.utcnow().isoformat(), signal["id"])
    )
    conn.commit()
    return False


def log_account_status():
    info = mt5.account_info()
    if info:
        logger.info(
            f"📊 REAL: balance={info.balance:.2f} equity={info.equity:.2f} "
            f"profit={info.profit:+.2f} margin={info.margin:.2f}"
        )


def main():
    setup_logging()
    logger.info("=" * 50)
    logger.info("CopyTrader iniciando...")
    logger.info("=" * 50)

    if not connect_real():
        logger.error("No se pudo conectar a la cuenta REAL. Saliendo.")
        return

    log_account_status()
    last_log = time.time()

    while True:
        try:
            conn = get_db()
            signals = get_pending_signals(conn)
            for sig in signals:
                s = dict(sig)
                action = s["action"]
                if action == "PLACE_LIMIT":
                    execute_place_limit(s, conn)
                elif action == "MODIFY_SLTP":
                    execute_modify_sltp(s, conn)
                elif action == "CANCEL":
                    execute_cancel(s, conn)
                else:
                    logger.warning(f"[SIG#{s['id']}] Acción desconocida: {action}")
                    conn.execute(
                        "UPDATE copy_signals SET executed=-1, error='Unknown action', executed_at=? WHERE id=?",
                        (datetime.utcnow().isoformat(), s["id"])
                    )
                    conn.commit()
            conn.close()

            if time.time() - last_log > 300:
                last_log = time.time()
                log_account_status()

            time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            logger.info("CopyTrader detenido por el usuario.")
            break
        except Exception as e:
            logger.exception(f"Error en loop: {e}")
            time.sleep(5)

    mt5.shutdown()


if __name__ == "__main__":
    main()

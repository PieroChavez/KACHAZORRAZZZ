"""Clean up all positions, pending orders, and state to start fresh"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import MetaTrader5 as mt5
from loguru import logger
from dotenv import load_dotenv

env_file = Path(__file__).resolve().parent.parent / ".env"
if env_file.exists():
    load_dotenv(env_file)

# --- MT5 connection ---
if not mt5.initialize():
    logger.error("MT5 initialize failed")
    sys.exit(1)

logger.info(f"MT5 version: {mt5.version()}")

# --- Close all positions ---
positions = mt5.positions_get()
if positions:
    logger.info(f"Closing {len(positions)} open positions...")
    for pos in positions:
        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": order_type,
            "position": pos.ticket,
            "price": pos.price_current if order_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(pos.symbol).ask,
            "deviation": 20,
            "magic": 0,
            "comment": "Cleanup",
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"  Closed {pos.ticket} {pos.symbol} {pos.type} {pos.volume}")
        else:
            err = mt5.last_error() if result is None else result.comment
            logger.warning(f"  Failed to close {pos.ticket}: {err}")
else:
    logger.info("No open positions")

# --- Cancel all pending orders ---
orders = mt5.orders_get()
if orders:
    logger.info(f"Cancelling {len(orders)} pending orders...")
    for o in orders:
        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": o.ticket,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"  Cancelled {o.ticket} {o.symbol}")
        else:
            err = mt5.last_error() if result is None else result.comment
            logger.warning(f"  Failed to cancel {o.ticket}: {err}")
else:
    logger.info("No pending orders")

mt5.shutdown()

# --- Clear state database ---
state_db = Path(__file__).resolve().parent.parent / "data" / "trading_state.db"
if state_db.exists():
    state_db.unlink()
    logger.info(f"Deleted {state_db}")

# --- Clear trade records from meta_learning.db ---
meta_db = Path(__file__).resolve().parent.parent / "data" / "meta_learning.db"
if meta_db.exists():
    meta_db.unlink()
    logger.info(f"Deleted {meta_db}")

logger.info("Reset complete — bot is clean and ready to restart")

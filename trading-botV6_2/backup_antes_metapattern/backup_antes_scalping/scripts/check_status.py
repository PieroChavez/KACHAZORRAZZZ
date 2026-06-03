import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import MetaTrader5 as mt5
from datetime import datetime

from src.core.market_hours import classify_symbol

mt5.initialize()
account = mt5.account_info()
if account:
    print(f"Account: {account.login} Balance: ${account.balance:.2f} Equity: ${account.equity:.2f}")

symbols = ["XAUUSDm", "XAGUSDm"]

print(f"\n{'Symbol':<12} {'Class':<12} {'State':<16} {'Age':<10} {'Positions':<10}")
print("-" * 60)

for sym in symbols:
    asset_class = classify_symbol(sym).value

    state = "UNKNOWN"
    last_candle = None

    # Try to read from persisted state
    try:
        import sqlite3
        db_path = Path(__file__).resolve().parent.parent / "data" / "trading_state.db"
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            cursor = conn.execute(
                "SELECT state, last_candle_time FROM market_hours_state WHERE symbol = ?", (sym,)
            )
            row = cursor.fetchone()
            if row:
                state = row[0]
                if row[1]:
                    last_candle = datetime.fromtimestamp(row[1])
            conn.close()
    except Exception:
        pass

    age_str = ""
    if last_candle:
        age_min = (datetime.now() - last_candle).total_seconds() / 60.0
        age_str = f"{age_min:.0f}min"
    else:
        age_str = "N/A"

    positions = mt5.positions_get(symbol=sym)
    pos_count = len(positions) if positions else 0

    print(f"{sym:<12} {asset_class:<12} {state:<16} {age_str:<10} {pos_count:<10}")
    if positions:
        for p in positions:
            is_long = p.type == 0
            direction = "BUY" if is_long else "SELL"
            print(f"  #{p.ticket} {direction} {p.volume:.2f}L @ {p.price_open:.5f} "
                  f"Curr={p.price_current:.5f} SL={p.sl:.5f} TP={p.tp:.5f} P/L=${p.profit:.2f}")

    orders = mt5.orders_get(symbol=sym)
    if orders:
        for o in orders:
            direction = "BUY" if o.type in (0,2) else "SELL"
            print(f"  ORDER #{o.ticket} {direction} {o.volume:.2f}L @ {o.price:.5f}")

# Check if bot process is running
import psutil
bot_running = any("python" in p.name() and "src.bot" in " ".join(p.cmdline()) for p in psutil.process_iter() if p.cmdline())
print(f"\nBot process running: {bot_running}")
mt5.shutdown()

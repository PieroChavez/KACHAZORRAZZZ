import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import MetaTrader5 as mt5
from datetime import datetime

mt5.initialize()
account = mt5.account_info()
if account:
    print(f"Account: {account.login} Balance: ${account.balance:.2f} Equity: ${account.equity:.2f}")

for sym in ["XAUUSDm", "XAGUSDm"]:
    positions = mt5.positions_get(symbol=sym)
    if positions:
        for p in positions:
            is_long = p.type == 0
            direction = "BUY" if is_long else "SELL"
            dist_sl = abs(p.price_open - p.sl) if p.sl else 0
            dist_tp = abs(p.price_open - p.tp) if p.tp else 0
            rr = dist_tp/dist_sl if dist_sl>0 else 0
            print(f"POS {sym} #{p.ticket} {direction} {p.volume:.2f}L @ {p.price_open:.5f} "
                  f"Curr={p.price_current:.5f} SL={p.sl:.5f}({dist_sl:.5f}) TP={p.tp:.5f}({dist_tp:.5f}) "
                  f"RR={rr:.2f} P/L=${p.profit:.2f}")
    else:
        print(f"{sym}: No positions")

    orders = mt5.orders_get(symbol=sym)
    if orders:
        for o in orders:
            direction = "BUY" if o.type in (0,2) else "SELL"
            print(f"ORDER {sym} #{o.ticket} {direction} {o.volume:.2f}L @ {o.price:.5f} SL={o.sl:.5f} TP={o.tp:.5f}")
    else:
        print(f"{sym}: No pending orders")

# Check if bot process is running
import psutil
bot_running = any("python" in p.name() and "src.main" in " ".join(p.cmdline()) for p in psutil.process_iter() if p.cmdline())
print(f"\nBot process running: {bot_running}")
mt5.shutdown()

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import MetaTrader5 as mt5
mt5.initialize()

positions = mt5.positions_get(symbol="XAUUSDc")
if positions:
    for p in positions:
        is_long = p.type == 0
        if is_long:
            price_profit = p.price_current - p.price_open
        else:
            price_profit = p.price_open - p.price_current
        tp_dist = abs(p.tp - p.price_open) if p.tp else 0
        pct_of_tp = (price_profit / tp_dist * 100) if tp_dist > 0 else 0
        pip_dist = abs(p.sl - p.price_open)
        be_active = pip_dist < 0.1
        print(f"XAUUSDc #{p.ticket} SELL")
        print(f"  Entry={p.price_open:.5f} Current={p.price_current:.5f}")
        print(f"  Profit_pts={price_profit:.5f} pips={price_profit*100:.1f}")
        print(f"  TP distance={tp_dist:.5f}")
        print(f"  30% of TP = {tp_dist*0.3:.5f}")
        print(f"  Reached 30%? {'YES' if price_profit >= tp_dist*0.3 else 'NO'} ({pct_of_tp:.0f}% of TP)")
        print(f"  SL={p.sl:.5f} (diff from entry={pip_dist:.5f})")
        print(f"  BE active? {'YES' if be_active else 'NO'}")
else:
    print("No XAUUSDc positions")

mt5.shutdown()

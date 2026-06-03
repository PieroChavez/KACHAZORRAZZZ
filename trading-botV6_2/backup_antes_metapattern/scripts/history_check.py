import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import MetaTrader5 as mt5
from datetime import datetime

mt5.initialize()

# Get ALL bot opening deals (magic match)
deals = mt5.history_deals_get(0, datetime.now())
bot_position_ids = set()
for d in deals:
    comment = getattr(d, 'comment', '') or ''
    if d.magic == 20260520 and 'smc' in comment and d.entry == 0:
        bot_position_ids.add(d.position_id)

print(f"Bot position IDs found: {len(bot_position_ids)}")

# For each position_id, find ALL deals (any magic) to get closing P&L
closed = []
for pid in bot_position_ids:
    entries = []
    exits = []
    for d in deals:
        if d.position_id == pid:
            if d.entry == 0:
                entries.append(d)
            elif d.entry in (1, 2):
                exits.append(d)
    if exits:
        total_pnl = sum(d.profit for d in exits)
        first_entry = entries[0] if entries else None
        if first_entry:
            direction = "BUY" if first_entry.type == 0 else "SELL"
            closed.append((pid, first_entry.symbol, direction, first_entry.volume, first_entry.price, total_pnl, first_entry.time))

closed.sort(key=lambda x: x[6], reverse=True)

print(f"Closed positions (with P&L): {len(closed)}")
print()

if closed:
    winners = [c for c in closed if c[5] > 0]
    losers = [c for c in closed if c[5] < 0]
    be = [c for c in closed if c[5] == 0]
    total = sum(c[5] for c in closed)
    print(f"G: {len(winners)} | P: {len(losers)} | BE: {len(be)} | Net: ${total:.2f}")
    print()
    print("--- ULTIMOS 30 TRADES CERRADOS ---")
    for pid, sym, dir_, vol, entry, pnl, ts in closed[:30]:
        tag = "GANADORA" if pnl > 0 else ("PERDEDORA" if pnl < 0 else "BE")
        dt = datetime.fromtimestamp(ts).strftime("%m/%d %H:%M")
        print(f"  {dt} #{pid} {sym} {dir_} {vol:.2f}L @ {entry:.5f} {tag} ${pnl:.2f}")

mt5.shutdown()

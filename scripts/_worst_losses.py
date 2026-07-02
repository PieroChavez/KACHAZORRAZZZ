"""Find which timeframe has the worst losses"""
import sqlite3
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "data" / "db" / "XAUUSDc"

# Fractal learner - losses per TF
fl = BASE / "fractal_learner.db"
conn = sqlite3.connect(str(fl))
conn.row_factory = sqlite3.Row

print("=== Fractal Learner - Perdidas por TF ===")
all_losses = []
for tf in ['5min','15min','30min','2H','4H']:
    rows = conn.execute("""
        SELECT timeframe, direction, profit, opened_at, session
        FROM fractal_trades
        WHERE timeframe = ? AND profit IS NOT NULL AND profit < 0
        ORDER BY profit ASC
    """, (tf,)).fetchall()
    if rows:
        total = sum(r['profit'] for r in rows)
        cnt = len(rows)
        worst = rows[0]
        print(f"{tf:>6}: {cnt:>2} perdidas | total=${total:>+7.2f} | peor=${worst['profit']:>+7.2f} {worst['direction']} {str(worst['opened_at'])[:16]}")
        all_losses.extend(rows)
    else:
        print(f"{tf:>6}: 0 perdidas registradas")

# Overall worst
if all_losses:
    worst = min(all_losses, key=lambda r: r['profit'])
    print(f"\nPeor trade de todo: {worst['timeframe']} {worst['direction']} ${worst['profit']:.2f} el {str(worst['opened_at'])[:16]}")
conn.close()

# Order packs - sub_orders losses
print("\n=== Order Packs (sub_orders) - Perdidas por TF ===")
op = BASE / "order_packs.db"
conn2 = sqlite3.connect(str(op))
conn2.row_factory = sqlite3.Row

total_losses = []
for tf in ['5min','15min','30min','2H','4H']:
    rows = conn2.execute("""
        SELECT p.source_timeframe as tf, s.direction, s.profit, s.closed_at, s.volume
        FROM sub_orders s
        JOIN order_packs p ON s.pack_id = p.id
        WHERE p.source_timeframe = ? AND s.profit IS NOT NULL AND s.profit < 0
        ORDER BY s.profit ASC
    """, (tf,)).fetchall()
    if rows:
        total = sum(r['profit'] for r in rows)
        cnt = len(rows)
        worst = rows[0]
        print(f"{tf:>6}: {cnt:>4} sub_orders | total=${total:>+8.2f} | peor=${worst['profit']:>+7.2f} {worst['direction']} {str(worst['closed_at'])[:16]}")
        total_losses.extend(rows)

# Summary at pack level (aggregate sub_orders by pack)
print("\n=== Perdidas por PACK (agregando sub_orders) ===")
pack_losses = conn2.execute("""
    SELECT p.source_timeframe as tf, p.id, SUM(s.profit) as pack_profit
    FROM order_packs p
    JOIN sub_orders s ON s.pack_id = p.id
    WHERE s.profit IS NOT NULL AND s.profit < 0
    GROUP BY p.id
    ORDER BY pack_profit ASC
""").fetchall()

by_tf = {}
for r in pack_losses:
    tf = r['tf']
    pnl = r['pack_profit']
    if tf not in by_tf:
        by_tf[tf] = {'count': 0, 'total': 0.0, 'worst': 0.0}
    by_tf[tf]['count'] += 1
    by_tf[tf]['total'] += pnl
    if pnl < by_tf[tf]['worst']:
        by_tf[tf]['worst'] = pnl

for tf in sorted(by_tf.keys()):
    s = by_tf[tf]
    print(f"{tf:>6}: {s['count']:>4} packs perdidos | total=${s['total']:>+8.2f} | peor pack=${s['worst']:>+7.2f}")

conn2.close()

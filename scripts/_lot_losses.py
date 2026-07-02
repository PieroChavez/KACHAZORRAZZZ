"""Show lot sizes of losing trades"""
import sqlite3
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "data" / "db" / "XAUUSDc"

# Fractal learner
fl = BASE / "fractal_learner.db"
conn = sqlite3.connect(str(fl))
conn.row_factory = sqlite3.Row

print("=== Fractal Learner - Perdidas con lotajes ===")
rows = conn.execute("""
    SELECT timeframe, direction, profit, volume, opened_at
    FROM fractal_trades
    WHERE profit IS NOT NULL AND profit < 0
    ORDER BY profit ASC
""").fetchall()
for r in rows:
    print(f"  {r['timeframe']:>6} {r['direction']:>7} profit={r['profit']:>+7.2f} lot={r['volume']:>6} {str(r['opened_at'])[:16]}")
conn.close()

# Order packs - sub_orders with losses + lot
op = BASE / "order_packs.db"
conn2 = sqlite3.connect(str(op))
conn2.row_factory = sqlite3.Row

print("\n=== Order Packs (sub_orders) - Perdidas con lotajes ===")
rows = conn2.execute("""
    SELECT p.source_timeframe as tf, s.direction, s.profit, s.volume, s.closed_at
    FROM sub_orders s
    JOIN order_packs p ON s.pack_id = p.id
    WHERE s.profit IS NOT NULL AND s.profit < 0
    ORDER BY s.profit ASC
""").fetchall()
for r in rows:
    print(f"  {r['tf']:>6} {r['direction']:>7} profit={r['profit']:>+7.2f} lot={r['volume']:>6} {str(r['closed_at'])[:16]}")

print("\n=== Perdidas a nivel PACK (agregado) ===")
rows = conn2.execute("""
    SELECT p.source_timeframe as tf, p.id, SUM(s.profit) as pack_profit, SUM(s.volume) as total_vol
    FROM order_packs p
    JOIN sub_orders s ON s.pack_id = p.id
    WHERE s.profit IS NOT NULL AND s.profit < 0
    GROUP BY p.id
    ORDER BY pack_profit ASC
""").fetchall()
for r in rows:
    print(f"  pack#{r['id']} {r['tf']:>6} profit={r['pack_profit']:>+7.2f} lot={r['total_vol']:.2f}")
conn2.close()

# Also check wins for comparison
print("\n=== Fractal Learner - GANANCIAS con lotajes ===")
conn3 = sqlite3.connect(str(fl))
conn3.row_factory = sqlite3.Row
rows = conn3.execute("""
    SELECT timeframe, direction, profit, volume, opened_at
    FROM fractal_trades
    WHERE profit IS NOT NULL AND profit > 0
    ORDER BY profit DESC
""").fetchall()
for r in rows:
    print(f"  {r['timeframe']:>6} {r['direction']:>7} profit={r['profit']:>+7.2f} lot={r['volume']:>6} {str(r['opened_at'])[:16]}")
conn3.close()

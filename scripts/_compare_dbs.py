"""Comparar trades por sesion entre fractal_learner y meta_learning"""
import sqlite3
from pathlib import Path

base = Path(__file__).resolve().parent.parent / "data" / "db" / "XAUUSDc"

# fractal_learner.db
fl = base / "fractal_learner.db"
conn1 = sqlite3.connect(str(fl))
rows1 = conn1.execute("""
    SELECT session, COUNT(*),
           SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END),
           SUM(CASE WHEN outcome='loss' THEN 1 ELSE 0 END),
           ROUND(SUM(profit),2)
    FROM fractal_trades
    WHERE outcome IS NOT NULL AND outcome != 'open' AND session IS NOT NULL AND session != ''
    GROUP BY session
    ORDER BY SUM(profit)
""").fetchall()
print("=== fractal_learner.db por sesion ===")
print(f"{'Session':<20} {'Trades':>7} {'W':>4} {'L':>4} {'Profit':>10}")
print("-" * 50)
for r in rows1:
    print(f"{r[0]:<20} {r[1]:>7} {r[2]:>4} {r[3]:>4} {r[4]:>+10.2f}")
conn1.close()

# meta_learning.db
ml = base / "meta_learning.db"
conn2 = sqlite3.connect(str(ml))
rows2 = conn2.execute("""
    SELECT session, COUNT(*),
           SUM(CASE WHEN profit>0 THEN 1 ELSE 0 END),
           SUM(CASE WHEN profit<=0 THEN 1 ELSE 0 END),
           ROUND(SUM(profit),2)
    FROM trade_records
    WHERE profit IS NOT NULL AND profit != 0 AND session IS NOT NULL AND session != ''
    GROUP BY session
    ORDER BY SUM(profit)
""").fetchall()
print("\n=== meta_learning.db por sesion ===")
print(f"{'Session':<20} {'Trades':>7} {'W':>4} {'L':>4} {'Profit':>10}")
print("-" * 50)
for r in rows2:
    print(f"{r[0]:<20} {r[1]:>7} {r[2]:>4} {r[3]:>4} {r[4]:>+10.2f}")
conn2.close()

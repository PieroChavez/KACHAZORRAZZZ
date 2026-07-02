"""Resumen ultimas 8h"""
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

db = Path(__file__).resolve().parent.parent / "data" / "db" / "XAUUSDc" / "fractal_learner.db"
conn = sqlite3.connect(str(db))

since = (datetime.utcnow() - timedelta(hours=8)).isoformat()

# por TF
rows = conn.execute("""
    SELECT timeframe, COUNT(*),
           SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END),
           SUM(CASE WHEN outcome='loss' THEN 1 ELSE 0 END),
           ROUND(SUM(profit),2)
    FROM fractal_trades
    WHERE outcome IN ('win','loss') AND closed_at >= ?
    GROUP BY timeframe
    ORDER BY COUNT(*) DESC
""", (since,)).fetchall()

print("=== ULTIMAS 8 HORAS - Por TF ===")
print(f"{'TF':>8} {'Trades':>7} {'W':>4} {'L':>4} {'Profit':>10}")
print("-" * 40)
total_t, total_w, total_l, total_p = 0,0,0,0
for r in rows:
    print(f"{r[0]:>8} {r[1]:>7} {r[2]:>4} {r[3]:>4} {r[4]:>+10.2f}")
    total_t += r[1]; total_w += r[2]; total_l += r[3]; total_p += r[4]

# por sesion
rows2 = conn.execute("""
    SELECT session, COUNT(*),
           SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END),
           SUM(CASE WHEN outcome='loss' THEN 1 ELSE 0 END),
           ROUND(SUM(profit),2)
    FROM fractal_trades
    WHERE outcome IN ('win','loss') AND closed_at >= ?
    GROUP BY session
    ORDER BY COUNT(*) DESC
""", (since,)).fetchall()

print(f"\n=== ULTIMAS 8 HORAS - Por Sesion ===")
print(f"{'Session':<20} {'Trades':>7} {'W':>4} {'L':>4} {'Profit':>10}")
print("-" * 50)
for r in rows2:
    print(f"{r[0]:<20} {r[1]:>7} {r[2]:>4} {r[3]:>4} {r[4]:>+10.2f}")

print(f"\nTOTAL: {total_t} trades | {total_w}W / {total_l}L | P&L=${total_p:+.2f}")
conn.close()

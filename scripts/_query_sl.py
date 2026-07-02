"""Consulta trades que llegaron a SL vs TP por temporalidad"""
import sqlite3
from pathlib import Path

db = Path(__file__).resolve().parent.parent / "data" / "db" / "XAUUSDc" / "fractal_learner.db"
conn = sqlite3.connect(str(db))
c = conn.cursor()

print("=== TRADES QUE LLEGARON A SL (loss) ===")
print(f"{'TF':>8} {'Trades':>7} {'Vol':>7} {'Avg Loss':>10} {'Total Loss':>12}")
print("-" * 50)
rows = c.execute("""
    SELECT timeframe, COUNT(*) as total, SUM(volume) as vol,
           ROUND(AVG(profit),2) as avg_loss, ROUND(SUM(profit),2) as total_loss
    FROM fractal_trades
    WHERE outcome = 'loss'
    GROUP BY timeframe
    ORDER BY total DESC
""").fetchall()
for r in rows:
    print(f"{r[0]:>8} {r[1]:>7} {r[2]:>7.2f} {r[3]:>+10.2f} {r[4]:>+12.2f}")

print()
print("=== TRADES QUE LLEGARON A TP (win) ===")
print(f"{'TF':>8} {'Trades':>7} {'Vol':>7} {'Avg Win':>10} {'Total Win':>12}")
print("-" * 50)
rows2 = c.execute("""
    SELECT timeframe, COUNT(*) as total, SUM(volume) as vol,
           ROUND(AVG(profit),2) as avg_win, ROUND(SUM(profit),2) as total_win
    FROM fractal_trades
    WHERE outcome = 'win'
    GROUP BY timeframe
    ORDER BY total DESC
""").fetchall()
for r in rows2:
    print(f"{r[0]:>8} {r[1]:>7} {r[2]:>7.2f} {r[3]:>+10.2f} {r[4]:>+12.2f}")

print()
total = c.execute("SELECT COUNT(*), ROUND(SUM(profit),2) FROM fractal_trades WHERE outcome IN ('win','loss')").fetchone()
print(f"TOTAL: {total[0]} trades cerrados, P&L=${total[1]:.2f}")
conn.close()

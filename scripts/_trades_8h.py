"""Trades de las ultimas 8 horas"""
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

db = Path(__file__).resolve().parent.parent / "data" / "db" / "XAUUSDc" / "fractal_learner.db"
conn = sqlite3.connect(str(db))
conn.row_factory = sqlite3.Row

since = (datetime.utcnow() - timedelta(hours=8)).isoformat()

rows = conn.execute("""
    SELECT timeframe, direction, outcome, profit, volume, session,
           entry_price, exit_price, closed_at
    FROM fractal_trades
    WHERE outcome IN ('win','loss') AND closed_at >= ?
    ORDER BY closed_at DESC
""", (since,)).fetchall()

total_profit = 0
wins = 0
losses = 0

print(f"{'TF':>8} {'Dir':>5} {'Res':>6} {'Profit':>10} {'Vol':>6} {'Sesion':<16} {'Entry':>9} {'Exit':>9} {'Hora':>16}")
print("-" * 90)
for r in rows:
    total_profit += r["profit"]
    if r["outcome"] == "win":
        wins += 1
    else:
        losses += 1
    closed = str(r["closed_at"])[11:19] if r["closed_at"] else ""
    print(f"{r['timeframe']:>8} {r['direction']:>5} {r['outcome']:>6} {r['profit']:>+10.2f} {r['volume']:>6.2f} {r['session']:<16} {r['entry_price']:>9.2f} {r['exit_price']:>9.2f} {closed:>16}")

print(f"\nTotal: {len(rows)} trades | {wins}W / {losses}L | P&L=${total_profit:+.2f}")
conn.close()

"""Trades de London y NY de hoy"""
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

db = Path(__file__).resolve().parent.parent / "data" / "db" / "XAUUSDc" / "meta_learning.db"
conn = sqlite3.connect(str(db))
conn.row_factory = sqlite3.Row

since = (datetime.utcnow() - timedelta(days=1)).isoformat()

rows = conn.execute("""
    SELECT session, direction, entry_price, exit_price, profit, volume,
           duration_minutes, timestamp
    FROM trade_records
    WHERE profit IS NOT NULL AND profit != 0
      AND (session = 'LONDON_OPEN' OR session = 'NY_AFTERNOON')
    ORDER BY timestamp DESC
""").fetchall()

print(f"{'Session':<16} {'Dir':>5} {'Entry':>9} {'Exit':>9} {'Profit':>9} {'Vol':>5} {'Dur(min)':>9} {'Time':>16}")
print("-" * 85)
for r in rows:
    ts = str(r["timestamp"])[11:19] if r["timestamp"] else ""
    print(f"{r['session']:<16} {r['direction']:>5} {r['entry_price']:>9.2f} {r['exit_price']:>9.2f} {r['profit']:>+9.2f} {r['volume']:>5.2f} {r['duration_minutes']:>9} {ts:>16}")

tot = conn.execute("""
    SELECT session, COUNT(*), ROUND(SUM(profit),2)
    FROM trade_records
    WHERE profit IS NOT NULL AND profit != 0
      AND session IN ('LONDON_OPEN', 'NY_AFTERNOON')
      AND timestamp >= ?
    GROUP BY session
""", (since,)).fetchall()
print()
for t in tot:
    print(f"  {t[0]}: {t[1]} trades, P&L=${t[2]:.2f}")

conn.close()

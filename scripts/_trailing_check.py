"""Check trailing stats in DB"""
import sqlite3
from pathlib import Path

db = Path("P:/PROYECTOS/OneDrive/Escritorio/trading-botV3/data/db/XAUUSDc/order_packs.db")
conn = sqlite3.connect(str(db))
conn.row_factory = sqlite3.Row

total = conn.execute("SELECT COUNT(*) FROM order_packs").fetchone()[0]
with_trailing = conn.execute("SELECT COUNT(*) FROM order_packs WHERE trailing_activated=1").fetchone()[0]
with_be = conn.execute("SELECT COUNT(*) FROM order_packs WHERE breakeven_activated=1").fetchone()[0]
active = conn.execute("SELECT COUNT(*) FROM order_packs WHERE status='active'").fetchone()[0]

print(f"Total packs:  {total}")
print(f"Con trailing: {with_trailing}")
print(f"Con BE:       {with_be}")
print(f"Activos:      {active}")

rows = conn.execute("""
    SELECT source_timeframe, COUNT(*) as total,
           SUM(CASE WHEN trailing_activated=1 THEN 1 ELSE 0 END) as trailed,
           SUM(CASE WHEN breakeven_activated=1 THEN 1 ELSE 0 END) as be
    FROM order_packs
    GROUP BY source_timeframe
    ORDER BY total DESC
""").fetchall()
print()
for r in rows:
    print(f"{r['source_timeframe']:>6}: total={r['total']:>4} trailed={r['trailed']:>4} be={r['be']:>4}")

# Count trailing activations in logs
import subprocess, os
log_dir = Path("P:/PROYECTOS/OneDrive/Escritorio/trading-botV3/logs")
print("\n--- Trailing activity in logs ---")
for log in sorted(log_dir.glob("trading_bot_*.log"))[-3:]:
    try:
        with open(log) as f:
            content = f.read()
            trail_count = content.count("TRAIL")
            be_count = content.count("BREAKEVEN")
            scan_count = content.count("TrailingGuard")
            print(f"{log.name[:19]}: scans={scan_count:>4} TRAIL={trail_count:>4} BREAKEVEN={be_count:>4}")
    except:
        pass

conn.close()

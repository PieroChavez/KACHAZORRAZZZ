import sqlite3
from pathlib import Path

db = Path(__file__).parent / "data" / "db" / "XAUEURm"

# trading_state
conn = sqlite3.connect(str(db / "trading_state.db"))
cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tables:", cursor.fetchall())
cursor = conn.execute("SELECT * FROM daily_state ORDER BY date DESC LIMIT 5")
print("\nDaily state:")
for r in cursor.fetchall():
    print(r)
conn.close()

# meta_learning
conn = sqlite3.connect(str(db / "meta_learning.db"))
cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("\nMeta tables:", cursor.fetchall())
cursor = conn.execute("SELECT * FROM trade_records ORDER BY id DESC LIMIT 10")
print("\nLast trades:")
for r in cursor.fetchall():
    print(r)
conn.close()

"""Set lot_override to 0.02"""
import sqlite3
from pathlib import Path

BASE = Path("P:/PROYECTOS/OneDrive/Escritorio/trading-botV3/data/db")

for sym in ["XAUUSDc", "XAUUSDm", "XAUEURm"]:
    db = BASE / sym / "fractal_state.db"
    if not db.exists():
        print(f"{sym}: no db found")
        continue
    conn = sqlite3.connect(str(db), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?,?)",
                 ("lot_override", "0.02"))
    conn.commit()
    val = conn.execute("SELECT value FROM config WHERE key='lot_override'").fetchone()
    print(f"{sym}: lot_override = {val[0]}")
    conn.close()

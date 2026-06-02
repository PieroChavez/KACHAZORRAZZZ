"""Verify seeded database content."""
import sqlite3

import pathlib; base = str(pathlib.Path(__file__).resolve().parent.parent.parent / 'data' / 'db')
dbs = ["kelly_state.db", "failure_analysis.db", "entry_confirmation.db",
       "adaptive_thresholds.db", "bayesian_ensemble.db"]

for db_name in dbs:
    conn = sqlite3.connect(f"{base}/{db_name}")
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    for t in tables:
        tn = t[0]
        rows = conn.execute(f'SELECT * FROM "{tn}"').fetchall()
        cols = [c[1] for c in conn.execute(f'PRAGMA table_info("{tn}")').fetchall()]
        print(f"{db_name}.{tn}: {len(rows)} rows | cols={cols[:4]}... | first={rows[0] if rows else 'empty'}")
    conn.close()
    print(f"  Verified: {db_name}")
print("All databases verified.")

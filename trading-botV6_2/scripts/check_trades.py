import sqlite3, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from datetime import datetime, timezone

db_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
print("=" * 70)
print("ANALISIS DE OPERACIONES - BASE DE DATOS")
print("=" * 70)

for f in sorted(os.listdir(db_dir)):
    if f.endswith('.db'):
        fp = os.path.join(db_dir, f)
        size = os.path.getsize(fp)
        try:
            conn = sqlite3.connect(fp)
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            print(f'\n--- {f} ({size/1024:.0f} KB) ---')
            for t in tables:
                tn = t[0]
                cnt = conn.execute(f'SELECT COUNT(*) FROM [{tn}]').fetchone()[0]
                cols = [d[1] for d in conn.execute(f'PRAGMA table_info([{tn}])').fetchall()]
                print(f'  [{tn}] {cnt} rows | cols: {cols}')
                if cnt > 0 and 'result' in cols:
                    wins = conn.execute(f"SELECT COUNT(*) FROM [{tn}] WHERE result='WIN'").fetchone()[0]
                    losses = conn.execute(f"SELECT COUNT(*) FROM [{tn}] WHERE result='LOSS'").fetchone()[0]
                    pnl = conn.execute(f"SELECT COALESCE(SUM(pnl),0) FROM [{tn}]").fetchone()[0]
                    print(f'    WIN={wins} LOSS={losses} PnL={pnl:.2f}')
                    sample = conn.execute(f'SELECT * FROM [{tn}] ORDER BY rowid DESC LIMIT 5').fetchall()
                    for r in sample:
                        print(f'    -> {r}')
            conn.close()
        except Exception as e:
            print(f'  Error: {e}')

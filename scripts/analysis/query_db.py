import sqlite3, sys
sys.path.insert(0, '.')
import pathlib
dbs = ['trading_state.db', 'meta_learning.db', 'bayesian_ensemble.db', 'kelly_state.db', 'failure_analysis.db', 'entry_confirmation.db', 'adaptive_thresholds.db']
base = str(pathlib.Path(__file__).resolve().parent.parent.parent / 'data' / 'db' / '')
for db_name in dbs:
    print(f'\n{"="*60}')
    print(f'DATABASE: {db_name}')
    print(f'{"="*60}')
    try:
        conn = sqlite3.connect(base + db_name)
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        for t in tables:
            tn = t[0]
            print(f'\n  --- TABLE: {tn} ---')
            rows = conn.execute(f'SELECT * FROM "{tn}"').fetchall()
            cols = [c[1] for c in conn.execute(f'PRAGMA table_info("{tn}")').fetchall()]
            print(f'  Columns: {cols}')
            print(f'  Rows: {len(rows)}')
            for r in rows[:50]:
                print(f'    {r}')
        conn.close()
    except Exception as e:
        print(f'  Error: {e}')

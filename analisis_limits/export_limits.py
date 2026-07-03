import sqlite3, csv, os
from collections import defaultdict

symbols = ['DEMO/XAUEURm', 'XAUEURm', 'XAUUSDc', 'XAUUSDm']
base = 'data/db'
out_dir = 'analisis_limits'
os.makedirs(out_dir, exist_ok=True)

for sym in symbols:
    db_path = os.path.join(base, sym, 'order_packs.db')
    if not os.path.exists(db_path):
        continue
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute('''
        SELECT id, fractal_id, symbol, direction, entry_price, sl_initial,
               volume_total, volume_per, status, source_timeframe, created_at
        FROM order_packs ORDER BY created_at DESC
    ''').fetchall()

    safe_sym = sym.replace('/', '_')
    out = os.path.join(out_dir, f'limits_{safe_sym}.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'fractal_id', 'symbol', 'direction', 'entry_price',
                    'sl_initial', 'volume_total', 'volume_per', 'status',
                    'source_timeframe', 'created_at'])
        for r in rows:
            w.writerow([r['id'], r['fractal_id'], r['symbol'], r['direction'],
                       r['entry_price'], r['sl_initial'], r['volume_total'],
                       r['volume_per'], r['status'], r['source_timeframe'],
                       r['created_at']])

    conn.close()
    print(f'{sym}: {len(rows)} limits exportados a {out}')

print()
print('=== RESUMEN POR TEMPORALIDAD ===')
for sym in symbols:
    db_path = os.path.join(base, sym, 'order_packs.db')
    if not os.path.exists(db_path):
        continue
    conn = sqlite3.connect(db_path)
    cur = conn.execute('''
        SELECT source_timeframe, status, COUNT(*) as cnt
        FROM order_packs
        GROUP BY source_timeframe, status
        ORDER BY source_timeframe, status
    ''')
    data = cur.fetchall()
    print(f'\n--- {sym} ---')
    tf_totals = defaultdict(lambda: {'active': 0, 'closed': 0})
    for tf, st, cnt in data:
        tf_totals[tf][st] = cnt
    for tf in sorted(tf_totals.keys()):
        d = tf_totals[tf]
        total = d['active'] + d['closed']
        print(f'  {tf:8s} -> activos: {d["active"]:4d}  cerrados: {d["closed"]:4d}  total: {total:4d}')
    conn.close()

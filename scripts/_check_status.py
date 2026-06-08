import sqlite3, os

base = 'P:/PROYECTOS/OneDrive/Escritorio/trading-botV3/data/db'
for sym in sorted(os.listdir(base)):
    db = os.path.join(base, sym, 'order_packs.db')
    if not os.path.exists(db):
        continue
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    
    print(f'===== {sym} =====')
    
    cur.execute("SELECT COUNT(*) FROM order_packs WHERE status='active'")
    active_packs = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM order_packs WHERE breakeven_activated=1 AND status='active'")
    be_active = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM order_packs WHERE trailing_activated=1 AND status='active'")
    trail_active = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM order_packs WHERE status='closed'")
    closed_packs = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM sub_orders WHERE status='active' AND ticket!=0")
    subs_with = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM sub_orders WHERE status='active' AND ticket=0")
    subs_without = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM sub_orders WHERE status!='active'")
    closed_subs = cur.fetchone()[0]
    
    print(f'  Packs activos: {active_packs}')
    print(f'  Packs cerrados: {closed_packs}')
    print(f'  Breakeven activado: {be_active}')
    print(f'  Trailing activado: {trail_active}')
    print(f'  Sub-orders con ticket: {subs_with}')
    print(f'  Sub-orders sin ticket (pending): {subs_without}')
    print(f'  Sub-orders cerradas: {closed_subs}')
    
    cur.execute("SELECT id, direction, entry_price, breakeven_activated, trailing_activated FROM order_packs WHERE (breakeven_activated=1 OR trailing_activated=1) AND status='active'")
    be_rows = cur.fetchall()
    for r in be_rows:
        print(f'  >> Pack #{r[0]} {r[1]} entry={r[2]:.2f} BE={r[3]!=0} Trail={r[4]!=0}')
    
    if not be_rows:
        print(f'  >> Ningun pack con BE o Trailing activado')
    
    conn.close()
    print()

"""Query order and trade stats per timeframe"""
import sqlite3
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent / "data" / "db"
SYMBOLS = ["XAUUSDc"]

def query_order_packs(sym: str):
    db = BASE / sym / "order_packs.db"
    if not db.exists():
        print(f"  {db} not found")
        return
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    # Total limits by TF (unique sub_orders)
    total_rows = conn.execute("""
        SELECT p.source_timeframe as tf, COUNT(*) as total
        FROM sub_orders s
        JOIN order_packs p ON s.pack_id = p.id
        GROUP BY p.source_timeframe
        ORDER BY total DESC
    """).fetchall()

    print(f"\n=== XAUUSDc - ORDER PACKS ===")
    print(f"\nTotal limits colocados por TF:")
    for r in total_rows:
        print(f"  {str(r['tf']):>10}: {r['total']}")

    # Profit at PACK level (aggregate sub_orders per pack)
    rows = conn.execute("""
        SELECT
            p.source_timeframe as tf,
            p.id as pack_id,
            p.direction,
            p.status,
            SUM(s.profit) as pack_profit
        FROM order_packs p
        LEFT JOIN sub_orders s ON s.pack_id = p.id
        WHERE s.profit IS NOT NULL
        GROUP BY p.id
    """).fetchall()

    stats = defaultdict(lambda: {"count": 0, "total_profit": 0.0, "wins": 0, "losses": 0})

    for r in rows:
        tf = r["tf"] if r["tf"] else "unknown"
        p = r["pack_profit"] or 0.0
        stats[tf]["count"] += 1
        stats[tf]["total_profit"] += p
        if p > 0:
            stats[tf]["wins"] += 1
        else:
            stats[tf]["losses"] += 1

    print(f"\nProfit por TF (por PACK agregado):")
    best_tf = None
    best_profit = float("-inf")
    for tf, s in sorted(stats.items()):
        avg = s["total_profit"] / s["count"] if s["count"] else 0
        wr = s["wins"] / s["count"] * 100 if s["count"] else 0
        print(f"  {str(tf):>10}: {s['count']:>3} packs | profit={s['total_profit']:>+8.2f} | avg={avg:>+7.2f} | wr={wr:>5.1f}%")
        if s["total_profit"] > best_profit:
            best_profit = s["total_profit"]
            best_tf = tf

    if best_tf:
        print(f"\n  >>> Mejor ganancia total: {best_tf} (${best_profit:.2f})")

    conn.close()

def query_meta_learning(sym: str):
    db = BASE / sym / "meta_learning.db"
    if not db.exists():
        return
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    table_name = "trade_records"
    try:
        cols = [c[1] for c in conn.execute(f"PRAGMA table_info({table_name})").fetchall()]
    except:
        print("  (no trade_records table)")
        conn.close()
        return

    tf_col = "timeframe" if "timeframe" in cols else "source_timeframe" if "source_timeframe" in cols else "'?'"
    profit_col = "profit" if "profit" in cols else "pnl" if "pnl" in cols else "result" if "result" in cols else "0"

    rows = conn.execute(f"""
        SELECT {tf_col} as tf, COUNT(*) as cnt,
               SUM(CASE WHEN {profit_col} > 0 THEN 1 ELSE 0 END) as wins,
               SUM({profit_col}) as total_profit,
               AVG({profit_col}) as avg_profit
        FROM trade_records
        WHERE {profit_col} IS NOT NULL
        GROUP BY {tf_col}
        ORDER BY cnt DESC
    """).fetchall()

    if rows:
        print(f"\n=== XAUUSDc - META-LEARNING ===")
        best_tf = None
        best_profit = float("-inf")
        for r in rows:
            tf = r["tf"]
            wr = r["wins"] / r["cnt"] * 100 if r["cnt"] else 0
            print(f"  {str(tf):>10}: {r['cnt']:>3} trades | profit={r['total_profit']:>+8.2f} | avg={r['avg_profit']:>+7.2f} | wr={wr:>5.1f}%")
            if r["total_profit"] > best_profit:
                best_profit = r["total_profit"]
                best_tf = tf
        if best_tf:
            print(f"  >>> Mejor ganancia: {best_tf} (${best_profit:.2f})")

    conn.close()

def query_fractal_learner(sym: str):
    db = BASE / sym / "fractal_learner.db"
    if not db.exists():
        return
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    cols = [c[1] for c in conn.execute("PRAGMA table_info(fractal_trades)").fetchall()]

    tf_col = "timeframe" if "timeframe" in cols else "source_timeframe"
    profit_col = "profit" if "profit" in cols else "pnl"

    rows = conn.execute(f"""
        SELECT {tf_col} as tf, COUNT(*) as cnt,
               SUM(CASE WHEN {profit_col} > 0 THEN 1 ELSE 0 END) as wins,
               SUM({profit_col}) as total_profit,
               AVG({profit_col}) as avg_profit
        FROM fractal_trades
        WHERE {profit_col} IS NOT NULL
        GROUP BY {tf_col}
        ORDER BY cnt DESC
    """).fetchall()

    if rows:
        print(f"\n=== XAUUSDc - FRACTAL LEARNER ===")
        best_tf = None
        best_profit = float("-inf")
        for r in rows:
            tf = r["tf"]
            wr = r["wins"] / r["cnt"] * 100 if r["cnt"] else 0
            print(f"  {str(tf):>10}: {r['cnt']:>3} trades | profit={r['total_profit']:>+8.2f} | avg={r['avg_profit']:>+7.2f} | wr={wr:>5.1f}%")
            if r["total_profit"] > best_profit:
                best_profit = r["total_profit"]
                best_tf = tf
        if best_tf:
            print(f"  >>> Mejor ganancia: {best_tf} (${best_profit:.2f})")

    conn.close()

for sym in SYMBOLS:
    query_order_packs(sym)
    query_fractal_learner(sym)
    query_meta_learning(sym)

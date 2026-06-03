import sqlite3, os

db_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

conn = sqlite3.connect(os.path.join(db_dir, 'meta_learning.db'))

# Pattern performance
print("PATTERN PERFORMANCE")
rows = conn.execute("SELECT * FROM pattern_performance ORDER BY total_occurrences DESC").fetchall()
cols = [d[1] for d in conn.execute("PRAGMA table_info(pattern_performance)").fetchall()]
for r in rows:
    d = dict(zip(cols, r))
    wr = d['wins']/(d['wins']+d['losses'])*100 if (d['wins']+d['losses']) > 0 else 0
    print(f"  {d['pattern_type']:25s} | regime={d['regime']:25s} | {d['total_occurrences']:3d}x | "
          f"WIN={d['wins']:2d} LOSS={d['losses']:2d} WR={wr:5.1f}% | PnL=${d['total_profit']:>7.2f}")

# Feature stats
print("\n\nFEATURE STATS")
conn2 = sqlite3.connect(os.path.join(db_dir, 'bayesian_weights.db'))
rows2 = conn2.execute("SELECT * FROM feature_stats ORDER BY occurrences DESC").fetchall()
cols2 = [d[1] for d in conn2.execute("PRAGMA table_info(feature_stats)").fetchall()]
for r in rows2:
    d = dict(zip(cols2, r))
    wr = d['positive_outcomes']/(d['positive_outcomes']+d['negative_outcomes'])*100 if (d['positive_outcomes']+d['negative_outcomes']) > 0 else 0
    print(f"  {d['name']:30s} | {d['occurrences']:3d}x | pos={d['positive_outcomes']:2d} neg={d['negative_outcomes']:2d} "
          f"WR={wr:5.1f}% | weight={d['total_weight']:.2f}")

conn.close()
conn2.close()

# Also check outcome_log for recent trades
print("\n\nRECENT OUTCOME LOG")
conn3 = sqlite3.connect(os.path.join(db_dir, 'bayesian_weights.db'))
rows3 = conn3.execute("SELECT * FROM outcome_log ORDER BY id DESC LIMIT 10").fetchall()
cols3 = [d[1] for d in conn3.execute("PRAGMA table_info(outcome_log)").fetchall()]
for r in rows3:
    d = dict(zip(cols3, r))
    print(f"  #{d['id']} | {d['symbol']} {d['direction']} | profit={d['profit']} | feature={d['feature_key']} | contrib={d['contribution']}")
conn3.close()

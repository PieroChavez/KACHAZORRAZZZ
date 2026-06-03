import sqlite3, os, sys

db_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

# Meta learning - trade records
conn = sqlite3.connect(os.path.join(db_dir, 'meta_learning.db'))
print("=" * 70)
print("TRADE RECORDS")
print("=" * 70)
rows = conn.execute("SELECT * FROM trade_records ORDER BY id DESC LIMIT 20").fetchall()
cols = [d[1] for d in conn.execute("PRAGMA table_info(trade_records)").fetchall()]
for r in rows:
    d = dict(zip(cols, r))
    print(f"  #{d['id']} | {d['symbol']} {d['direction']} | score={d['score']} conv={d['conviction']} | "
          f"profit={d['profit']} | regime={d['regime']} session={d['session']} | "
          f"exit={d['exit_reason']} | pattern={d['primary_pattern']}")

# Summary
wins = conn.execute("SELECT COUNT(*) FROM trade_records WHERE profit > 0").fetchone()[0]
losses = conn.execute("SELECT COUNT(*) FROM trade_records WHERE profit <= 0").fetchone()[0]
total = conn.execute("SELECT COUNT(*) FROM trade_records").fetchone()[0]
avg_win = conn.execute("SELECT COALESCE(AVG(profit),0) FROM trade_records WHERE profit > 0").fetchone()[0]
avg_loss = conn.execute("SELECT COALESCE(AVG(profit),0) FROM trade_records WHERE profit <= 0").fetchone()[0]
total_pnl = conn.execute("SELECT COALESCE(SUM(profit),0) FROM trade_records").fetchone()[0]
print(f"\nRESUMEN: {total} trades | WIN={wins} ({wins/total*100:.0f}%) LOSS={losses} ({losses/total*100:.0f}%)")
print(f"  Avg WIN=${avg_win:.2f} | Avg LOSS=${avg_loss:.2f} | Total PnL=${total_pnl:.2f}")

# Exit reasons
print(f"\nEXIT REASONS:")
exit_reasons = conn.execute("SELECT exit_reason, COUNT(*), AVG(profit) FROM trade_records GROUP BY exit_reason").fetchall()
for er in exit_reasons:
    print(f"  {er[0]}: {er[1]} trades, avg profit=${er[2]:.2f}")

# SKIPPED SIGNALS
print(f"\n" + "=" * 70)
print("SKIPPED SIGNALS (ultimos 10)")
print("=" * 70)
skip_rows = conn.execute("SELECT * FROM skipped_signals ORDER BY id DESC LIMIT 10").fetchall()
skip_cols = [d[1] for d in conn.execute("PRAGMA table_info(skipped_signals)").fetchall()]
for r in skip_rows:
    d = dict(zip(skip_cols, r))
    print(f"  #{d['id']} | {d['symbol']} {d['direction']} | score={d['score']} conv={d['conviction']} | "
          f"regime={d['regime']} session={d['session']} | reason={d['reason']}")

conn.close()

# Failure analysis
conn2 = sqlite3.connect(os.path.join(db_dir, 'failure_analysis.db'))
print(f"\n" + "=" * 70)
print("FAILURE ANALYSIS")
print("=" * 70)
rows2 = conn2.execute("SELECT * FROM trade_analysis ORDER BY id DESC LIMIT 10").fetchall()
cols2 = [d[1] for d in conn2.execute("PRAGMA table_info(trade_analysis)").fetchall()]
for r in rows2:
    d = dict(zip(cols2, r))
    print(f"  #{d['id']} | {d['symbol']} {d['direction']} | profit={d['profit']} | score={d['score']} conv={d['conviction']} | "
          f"exit={d['exit_reason']} | fail_cat={d['failure_category']} | mfe={d['mfe_pct']} mae={d['mae_pct']} | "
          f"sl={d['sl_dist']} tp={d['tp_dist']} session={d['session']}")

conn2.close()

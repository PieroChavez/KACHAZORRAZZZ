"""Detailed trade report per timeframe with running balance"""
import sqlite3
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parent.parent / "data" / "db" / "XAUUSDc"

# Check trading_state for initial balance
ts_db = BASE / "trading_state.db"
if ts_db.exists():
    conn = sqlite3.connect(str(ts_db))
    try:
        rows = conn.execute("SELECT * FROM daily_state ORDER BY date ASC").fetchall()
        if rows:
            cols = [c[1] for c in conn.execute("PRAGMA table_info(daily_state)").fetchall()]
            print("=== TRADING STATE (daily) ===")
            for r in rows:
                print(dict(zip(cols, r)))
        else:
            print("trading_state.db: empty")
    except Exception as e:
        print(f"trading_state error: {e}")
    conn.close()

# Fractal Learner - detailed trades
fl_db = BASE / "fractal_learner.db"
if fl_db.exists():
    conn = sqlite3.connect(str(fl_db))
    conn.row_factory = sqlite3.Row
    cols = [c[1] for c in conn.execute("PRAGMA table_info(fractal_trades)").fetchall()]
    print(f"\n=== FRACTAL LEARNER columns: {cols} ===")

    # Get all trades ordered by opened_at
    rows = conn.execute("""
        SELECT * FROM fractal_trades
        WHERE profit IS NOT NULL
        ORDER BY opened_at ASC
    """).fetchall()

    if rows:
        print(f"\nTotal trades: {len(rows)}")
        balance = 0.0
        # Find initial balance if available
        # Check if there's a balance column
        if "balance" in cols:
            balance = rows[0]["balance"] if rows[0]["balance"] else 0.0
            print(f"Initial balance from DB: {balance:.2f}")
        else:
            print("Initial balance: unknown (starting from 0 for relative calc)")

        print(f"\n{'#':>4} {'TF':>6} {'Dir':>6} {'Profit':>8} {'Entry':>18} {'Session':>12}")
        print("-" * 60)
        for i, r in enumerate(rows, 1):
            d = dict(zip(cols, r))
            tf = d.get("timeframe", "?")
            direction = d.get("direction", "?")
            profit = d.get("profit", 0) or 0
            opened = str(d.get("opened_at", ""))[:16]
            session = str(d.get("session", ""))[:10]
            print(f"{i:>4} {tf:>6} {direction:>6} {profit:>+8.2f} {opened:>18} {session:>12}")

        # Summary per timeframe
        conn2 = sqlite3.connect(str(fl_db))
        summary = conn2.execute("""
            SELECT timeframe,
                   COUNT(*) as cnt,
                   SUM(profit) as total_pnl,
                   AVG(profit) as avg_pnl,
                   SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN profit < 0 THEN 1 ELSE 0 END) as losses
            FROM fractal_trades
            WHERE profit IS NOT NULL
            GROUP BY timeframe
            ORDER BY total_pnl DESC
        """).fetchall()

        print(f"\n=== SUMMARY PER TF (Fractal Learner) ===")
        print(f"{'TF':>6} {'Trades':>7} {'Profit':>10} {'Avg':>8} {'Wins':>5} {'Losses':>5} {'WR':>6}")
        print("-" * 50)
        total_trades = 0
        total_pnl = 0.0
        for r in summary:
            tf = r[0]
            cnt = r[1]
            pnl = r[2]
            avg = r[3]
            wins = r[4]
            losses = r[5]
            wr = wins / cnt * 100 if cnt else 0
            print(f"{tf:>6} {cnt:>7} {pnl:>+10.2f} {avg:>+8.2f} {wins:>5} {losses:>5} {wr:>5.1f}%")
            total_trades += cnt
            total_pnl += pnl
        print("-" * 50)
        print(f"{'TOTAL':>6} {total_trades:>7} {total_pnl:>+10.2f}")

        # Best and worst trades
        best = conn2.execute("""
            SELECT timeframe, direction, profit, opened_at
            FROM fractal_trades
            WHERE profit IS NOT NULL
            ORDER BY profit DESC LIMIT 1
        """).fetchone()
        worst = conn2.execute("""
            SELECT timeframe, direction, profit, opened_at
            FROM fractal_trades
            WHERE profit IS NOT NULL
            ORDER BY profit ASC LIMIT 1
        """).fetchone()

        if best:
            print(f"\nBest trade: {best[0]} {best[1]} +${best[2]:.2f} on {str(best[3])[:16]}")
        if worst:
            print(f"Worst trade: {worst[0]} {worst[1]} ${worst[2]:.2f} on {str(worst[3])[:16]}")
    conn.close()

# Check account info from logs
import re
log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
log_files = sorted(log_dir.glob("trading_bot_*.log"))
if log_files:
    latest = log_files[-1]
    print(f"\n=== Latest log: {latest.name} ===")
    with open(latest) as f:
        lines = f.readlines()
        # Find balance lines
        for line in lines:
            if "balance=" in line and "equity=" in line:
                print(line.strip())
                break

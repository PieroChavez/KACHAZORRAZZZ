"""Ránking de temporalidades por rendimiento"""
import sqlite3
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "data" / "db"

def rank_timeframes(sym: str):
    db = BASE / sym / "fractal_learner.db"
    if not db.exists():
        print(f"  {db} no encontrado")
        return

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    # Ranking por TF
    rows = conn.execute("""
        SELECT
            timeframe,
            COUNT(*) as trades,
            SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
            ROUND(AVG(profit), 2) as avg_profit,
            ROUND(SUM(profit), 2) as total_profit,
            ROUND(SUM(volume), 2) as total_volume,
            ROUND(SUM(CASE WHEN outcome = 'win' THEN profit ELSE 0 END), 2) as gross_wins,
            ROUND(SUM(CASE WHEN outcome = 'loss' THEN profit ELSE 0 END), 2) as gross_losses
        FROM fractal_trades
        WHERE outcome IS NOT NULL AND outcome != 'open'
        GROUP BY timeframe
        ORDER BY total_profit DESC
    """).fetchall()

    if not rows:
        print(f"  {sym}: sin trades cerrados")
        conn.close()
        return

    print(f"\n{'='*60}")
    print(f"  RANKING POR TEMPORALIDAD — {sym}")
    print(f"{'='*60}")
    print(f"{'#':>3} {'TF':>8} {'Trades':>7} {'W':>4} {'L':>4} {'WR':>6} "
          f"{'Profit':>10} {'Avg':>8} {'Vol':>6}")
    print(f"{'-'*60}")

    for i, r in enumerate(rows, 1):
        tf = r["timeframe"]
        trades = r["trades"]
        wins = r["wins"]
        losses = r["losses"]
        wr = wins / trades * 100 if trades > 0 else 0
        profit = r["total_profit"]
        avg = r["avg_profit"]
        vol = r["total_volume"]
        print(f"{i:>3} {tf:>8} {trades:>7} {wins:>4} {losses:>4} "
              f"{wr:>5.1f}% {profit:>+10.2f} {avg:>+8.2f} {vol:>6.2f}")

    # Best and worst
    best = max(rows, key=lambda r: r["total_profit"])
    worst = min(rows, key=lambda r: r["total_profit"])
    print(f"\n  {'>>':>3} MEJOR:  {best['timeframe']} → ${best['total_profit']:.2f} ({best['trades']} trades)")
    print(f"  {'>>':>3} PEOR:   {worst['timeframe']} → ${worst['total_profit']:.2f} ({worst['trades']} trades)")

    # Últimos 20 trades
    print(f"\n{'='*60}")
    print(f"  ÚLTIMOS 20 TRADES")
    print(f"{'='*60}")
    print(f"{'TF':>8} {'Dir':>6} {'Profit':>10} {'Vol':>6} {'Salida':>16}")
    print(f"{'-'*50}")

    recent = conn.execute("""
        SELECT timeframe, direction, profit, volume, outcome, closed_at
        FROM fractal_trades
        WHERE outcome IS NOT NULL AND outcome != 'open'
        ORDER BY closed_at DESC LIMIT 20
    """).fetchall()

    for r in recent:
        closed = str(r["closed_at"])[11:19] if r["closed_at"] else ""
        print(f"{r['timeframe']:>8} {r['direction']:>6} {r['profit']:>+10.2f} "
              f"{r['volume']:>6.2f} {closed:>16}")

    conn.close()

if __name__ == "__main__":
    import sys
    symbols = sys.argv[1:] if len(sys.argv) > 1 else ["XAUUSDc"]
    for sym in symbols:
        rank_timeframes(sym)

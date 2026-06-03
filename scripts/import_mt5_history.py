"""Import MT5 trade history into FractalLearner DB
Groups positions opened within the same minute as a 'pack'.

Usage:
    python scripts/import_mt5_history.py [--symbol XAUEURm] [--magic 20260521] [--days 30]
"""
import sys, os, json, argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict

_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

import MetaTrader5 as mt5
from loguru import logger

from src.core.session_profiler import SessionProfiler, TradingSession
from src.strategies.fractal_learner import FractalLearner

SESSION_LABELS = {
    TradingSession.ASIAN: "ASIAN",
    TradingSession.LONDON_OPEN: "LONDON_OPEN",
    TradingSession.LONDON_MID: "LONDON_MID",
    TradingSession.NY_OPEN: "NY_OPEN",
    TradingSession.LONDON_NY_OVERLAP: "LONDON_NY_OVERLAP",
    TradingSession.NY_AFTERNOON: "NY_AFTERNOON",
    TradingSession.CLOSE: "CLOSE",
}


def get_session_for_dt(dt: datetime) -> str:
    profiler = SessionProfiler()
    sess = profiler.get_session(dt.replace(tzinfo=None))
    return SESSION_LABELS.get(sess, sess.value)


def now_utc():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def main():
    parser = argparse.ArgumentParser(description="Import MT5 history to FractalLearner")
    parser.add_argument("--symbol", default="XAUEURm")
    parser.add_argument("--magic", type=int, default=20260521)
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    with open(_proj_root / "config" / "broker.json") as f:
        broker = json.load(f)["mt5"]
    login = os.environ.get("MT5_LOGIN") or broker.get("login")
    password = os.environ.get("MT5_PASSWORD") or broker.get("password")
    server = os.environ.get("MT5_SERVER") or broker.get("server")

    if not mt5.initialize(login=int(login), password=password, server=server):
        print(f"Error MT5: {mt5.last_error()}")
        return

    since = now_utc() - timedelta(days=args.days)
    deals = mt5.history_deals_get(since, now_utc())
    if not deals:
        print("No deals found.")
        mt5.shutdown()
        return

    # Filter by symbol + magic + closed (profit != 0)
    target = [d for d in deals
              if d.symbol == args.symbol
              and d.magic == args.magic
              and d.profit != 0]
    print(f"Deals encontrados: {len(target)} (filtrados de {len(deals)})")

    if not target:
        mt5.shutdown()
        return

    # Group into packs: positions opened within same 60s window
    pos_groups = defaultdict(list)
    for d in target:
        pos_groups[d.position_id].append(d)

    # Collect opening times: look at the order open time via order history
    orders = mt5.history_orders_get(since, now_utc())
    order_time = {}
    if orders:
        for o in orders:
            if o.symbol == args.symbol and o.magic == args.magic:
                order_time[o.ticket] = o.time_setup

    # Group positions that were opened within 60s of each other
    pos_list = []
    for pid, deals_list in pos_groups.items():
        total_profit = sum(d.profit for d in deals_list)
        total_volume = sum(d.volume for d in deals_list)
        first_deal = deals_list[0]
        # Try to find order open time
        open_time = order_time.get(first_deal.order)
        if open_time is None:
            open_time = first_deal.time  # fallback to deal time
        open_dt = datetime.fromtimestamp(open_time, tz=timezone.utc).replace(tzinfo=None)
        direction = "bullish" if first_deal.type == 0 else "bearish"
        outcome = "win" if total_profit > 0 else "loss"
        session = get_session_for_dt(open_dt)
        pos_list.append({
            "position_id": pid,
            "open_time": open_dt,
            "direction": direction,
            "volume": total_volume,
            "profit": total_profit,
            "outcome": outcome,
            "session": session,
            "entry_price": first_deal.price,
            "deals": len(deals_list),
        })

    # Sort by open time
    pos_list.sort(key=lambda x: x["open_time"])

    # Now import into FractalLearner
    learner = FractalLearner(args.symbol)
    imported = 0
    for p in pos_list:
        pack_id = p["position_id"]  # use position_id as pack_id
        # Check if already imported
        existing = learner._conn.execute(
            "SELECT id FROM fractal_trades WHERE pack_id=? AND symbol=?",
            (pack_id, args.symbol)
        ).fetchone()
        if existing:
            continue
        learner.record_entry(
            pack_id=pack_id,
            fractal_id=0,  # unknown from MT5
            timeframe="imported",  # marked as imported
            direction=p["direction"],
            is_subfractal=False,
            entry_price=p["entry_price"],
            sl_price=0.0,
            tp_price=0.0,
            volume=round(p["volume"], 2),
            range_size=0.0,
            fib_level=0.0,
            session=p["session"],
        )
        learner.record_exit(pack_id, p["outcome"], 0.0, round(p["profit"], 2))
        imported += 1

    print(f"Importados {imported} trades nuevos en fractal_learner.db")
    print(f"Total ahora: {learner.get_summary()['closed_trades']} trades cerrados")

    # Run analysis
    result = learner.analyze(force=True)
    if result.get("analyzed"):
        print(f"Analisis ML ejecutado: {len(result.get('adjustments', []))} ajustes")
        for adj in result.get("adjustments", []):
            print(f"  {adj}")
    else:
        print(f"Analisis no ejecutado: {result.get('reason', '?')}")

    print(f"\nResumen:")
    summary = learner.get_summary()
    print(f"  Trades: {summary['closed_trades']} | Win rate: {summary['win_rate']:.1%} | Profit: ${summary['total_profit']:+.2f}")
    if summary['volume_multipliers']:
        print(f"  Multiplicadores: {json.dumps(summary['volume_multipliers'])}")

    learner.close()
    mt5.shutdown()


if __name__ == "__main__":
    main()

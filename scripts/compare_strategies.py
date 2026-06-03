"""Compare strategy performance report
Fetches MT5 trade history for today, groups by magic number,
and shows which strategy had the best profit.

Usage:
    python scripts/compare_strategies.py              # today only
    python scripts/compare_strategies.py --days 7     # last 7 days
    python scripts/compare_strategies.py --html        # HTML report
"""
import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta, date

_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

import MetaTrader5 as mt5
from loguru import logger


MAGIC_MAP = {
    20260520: ("SMC Classic", "XAUUSDm — SMC scoring + patterns"),
    20260521: ("Fractal Cascade", "XAUEURm — CHoCH/BOS + 0.72 fib"),
    20260522: ("Fibonacci Ruiz", "XAUEURm — swing retracement 0.618/0.75 + EMA50"),
}

STRATEGY_NAMES = {
    20260520: "SMC",
    20260521: "Fractal",
    20260522: "Fibonacci",
}


def load_credentials():
    config_dir = _proj_root / "config"
    with open(config_dir / "broker.json") as f:
        broker = json.load(f)["mt5"]
    login = os.environ.get("MT5_LOGIN") or broker.get("login")
    password = os.environ.get("MT5_PASSWORD") or broker.get("password")
    server = os.environ.get("MT5_SERVER") or broker.get("server")
    return login, password, server


def fetch_trades(since_date: datetime) -> list:
    """Get all closed deal history from MT5 grouped by magic."""
    history = mt5.history_deals_get(since_date, datetime.now())
    if history is None or len(history) == 0:
        return []

    trades = []
    for deal in history:
        if deal.deal_type != mt5.DEAL_TYPE_BUY and deal.deal_type != mt5.DEAL_TYPE_SELL:
            continue
        if deal.entry != mt5.DEAL_ENTRY_OUT:
            continue
        trades.append({
            "ticket": deal.ticket,
            "symbol": deal.symbol,
            "magic": deal.magic,
            "type": "BUY" if deal.type == mt5.DEAL_TYPE_BUY else "SELL",
            "volume": deal.volume,
            "price": deal.price,
            "profit": deal.profit,
            "commission": deal.commission,
            "swap": deal.swap,
            "time": datetime.fromtimestamp(deal.time),
        })

    return trades


def report_text(trades: list, days: int):
    """Plain text report."""
    today = date.today()

    strategy_data = {}
    for t in trades:
        magic = t["magic"]
        if magic not in strategy_data:
            strategy_data[magic] = {"trades": [], "total_profit": 0.0,
                                     "wins": 0, "losses": 0, "volumes": 0}
        sd = strategy_data[magic]
        sd["trades"].append(t)
        sd["total_profit"] += t["profit"]
        sd["volumes"] += t["volume"]
        if t["profit"] > 0:
            sd["wins"] += 1
        elif t["profit"] < 0:
            sd["losses"] += 1

    print()
    print("=" * 72)
    print(f"  📊 REPORTE COMPARATIVO DE ESTRATEGIAS")
    print(f"  Período: últimos {days} día(s) — {today}")
    print("=" * 72)

    if not strategy_data:
        print("\n  ⚠️  No se encontraron trades cerrados en el período.")
        print("=" * 72)
        return

    # Header
    print(f"\n  {'Estrategia':<18} {'Símbolo':<12} {'Trades':>7} {'Ganadas':>8} "
          f"{'Perdidas':>9} {'Win%':>7} {'Profit':>10} {'Vol':>7}")
    print(f"  {'─'*18} {'─'*12} {'─'*7} {'─'*8} {'─'*9} {'─'*7} {'─'*10} {'─'*7}")

    sorted_magics = sorted(strategy_data.keys(),
                           key=lambda m: strategy_data[m]["total_profit"],
                           reverse=True)

    best_magic = sorted_magics[0]
    best_data = strategy_data[best_magic]

    for magic in sorted_magics:
        sd = strategy_data[magic]
        name, desc = MAGIC_MAP.get(magic, (f"Desconocido ({magic})", ""))
        total = len(sd["trades"])
        win_rate = (sd["wins"] / total * 100) if total > 0 else 0
        sym = sd["trades"][0]["symbol"] if sd["trades"] else "—"
        profit_str = f"{sd['total_profit']:+.2f}"
        if profit_str.startswith("+"):
            profit_str = " " + profit_str

        print(f"  {name:<18} {sym:<12} {total:>7} {sd['wins']:>8} "
              f"{sd['losses']:>9} {win_rate:>6.1f}% {profit_str:>10} {sd['volumes']:>7.2f}")

    print(f"  {'─'*72}")

    winner_name, winner_desc = MAGIC_MAP.get(best_magic,
                                             (f"Estrategia {best_magic}", ""))
    print(f"\n  🏆  Estrategia con MAYOR PROFIT: {winner_name}")
    print(f"     {winner_desc}")
    print(f"     Profit total: ${best_data['total_profit']:+.2f}")
    print(f"     Win rate: {best_data['wins']/max(len(best_data['trades']),1)*100:.1f}%")
    print()

    # Per-day breakdown
    if any(len(sd["trades"]) >= 3 for sd in strategy_data.values()):
        print(f"  {'─'*72}")
        print(f"  Desglose diario:")
        print(f"  {'─'*72}")
        daily: dict = {}
        for magic, sd in strategy_data.items():
            name = MAGIC_MAP.get(magic, (f"Magic {magic}", ""))[0]
            for t in sd["trades"]:
                d = t["time"].date()
                daily.setdefault(d, {}).setdefault(name, []).append(t)

        for d in sorted(daily.keys()):
            day_data = daily[d]
            parts = []
            for sname, sts in sorted(day_data.items(),
                                     key=lambda x: sum(t["profit"] for t in x[1]),
                                     reverse=True):
                p = sum(t["profit"] for t in sts)
                parts.append(f"{sname}: ${p:+.2f}")
            print(f"  {d}:  {'  |  '.join(parts)}")

    print("=" * 72)


def report_html(trades: list, days: int):
    """Generate HTML report file."""
    strategy_data = {}
    for t in trades:
        magic = t["magic"]
        if magic not in strategy_data:
            strategy_data[magic] = {"trades": [], "total_profit": 0.0,
                                     "wins": 0, "losses": 0}
        sd = strategy_data[magic]
        sd["trades"].append(t)
        sd["total_profit"] += t["profit"]
        if t["profit"] > 0:
            sd["wins"] += 1
        elif t["profit"] < 0:
            sd["losses"] += 1

    sorted_magics = sorted(strategy_data.keys(),
                           key=lambda m: strategy_data[m]["total_profit"],
                           reverse=True)

    rows = ""
    for magic in sorted_magics:
        sd = strategy_data[magic]
        name = STRATEGY_NAMES.get(magic, f"???({magic})")
        sym = sd["trades"][0]["symbol"] if sd["trades"] else "—"
        total = len(sd["trades"])
        wr = sd["wins"] / total * 100 if total > 0 else 0
        color = "#22c55e" if sd["total_profit"] >= 0 else "#ef4444"
        rows += f"""
        <tr>
            <td><strong>{name}</strong></td>
            <td>{sym}</td>
            <td>{total}</td>
            <td>{sd['wins']}</td>
            <td>{sd['losses']}</td>
            <td>{wr:.1f}%</td>
            <td style="color:{color};font-weight:bold">{sd['total_profit']:+.2f}</td>
        </tr>"""

    winner = STRATEGY_NAMES.get(sorted_magics[0], "—") if sorted_magics else "—"

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Comparativa de Estrategias — {date.today()}</title>
<style>
    body {{ font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; padding: 40px; }}
    h1 {{ color: #f59e0b; text-align: center; font-size: 2em; }}
    h2 {{ color: #94a3b8; text-align: center; }}
    table {{ margin: 30px auto; border-collapse: collapse; width: 90%; max-width: 1000px; }}
    th {{ background: #1e293b; color: #f1f5f9; padding: 12px; text-align: left; font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.1em; }}
    td {{ padding: 10px 12px; border-bottom: 1px solid #1e293b; }}
    tr:hover {{ background: #1e293b; }}
    .winner {{ text-align: center; font-size: 1.5em; padding: 20px; background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 12px; margin: 30px auto; width: fit-content; }}
    .winner span {{ color: #f59e0b; }}
    .subtle {{ color: #64748b; text-align: center; }}
</style>
</head>
<body>
    <h1>📊 Comparativa de Estrategias</h1>
    <h2>Últimos {days} día(s) — {date.today()}</h2>
    <table>
        <thead>
            <tr><th>Estrategia</th><th>Símbolo</th><th>Trades</th><th>Ganadas</th><th>Perdidas</th><th>Win%</th><th>Profit</th></tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    <div class="winner">
        🏆 Estrategia con mayor profit: <span>{winner}</span>
    </div>
    <p class="subtle">Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</body>
</html>"""

    report_dir = _proj_root / "data" / "reports"
    report_dir.mkdir(exist_ok=True)
    path = report_dir / f"comparativa_{date.today().isoformat()}.html"
    path.write_text(html, encoding="utf-8")
    print(f"\n📄 Reporte HTML guardado: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Comparativa de estrategias")
    parser.add_argument("--days", type=int, default=1, help="Días hacia atrás (default: 1 = hoy)")
    parser.add_argument("--html", action="store_true", help="Generar reporte HTML")
    args = parser.parse_args()

    login, password, server = load_credentials()
    if not mt5.initialize(login=int(login), password=password, server=server):
        print(f"Error conectando a MT5: {mt5.last_error()}")
        return

    print(f"Conectado a MT5 — {server}")
    info = mt5.terminal_info()
    print(f"  Balance: ${info.balance:.2f}  |  Equity: ${info.equity:.2f}")
    print()

    since = datetime.now() - timedelta(days=args.days)
    trades = fetch_trades(since)

    if args.html:
        report_html(trades, args.days)
    report_text(trades, args.days)

    mt5.shutdown()


if __name__ == "__main__":
    main()

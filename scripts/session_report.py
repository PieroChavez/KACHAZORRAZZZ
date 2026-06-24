"""Session Performance Report — market movement + trade performance per session
Integrated into the bot (runs every 4h) and usable as standalone script.

Usage:
    python scripts/session_report.py [--html] [--days 1]
"""
import sys
import os
import json
import argparse
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict

_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

from loguru import logger
from src.core.session_profiler import SessionProfiler, TradingSession

SESSION_LABELS = {
    TradingSession.ASIAN: "Asia",
    TradingSession.LONDON_OPEN: "London Open",
    TradingSession.LONDON_MID: "London Mid",
    TradingSession.NY_OPEN: "NY Open",
    TradingSession.LONDON_NY_OVERLAP: "LondonxNY",
    TradingSession.NY_AFTERNOON: "NY Afternoon",
    TradingSession.CLOSE: "Close",
}

SESSION_ORDER = [
    TradingSession.ASIAN,
    TradingSession.LONDON_OPEN,
    TradingSession.LONDON_MID,
    TradingSession.NY_OPEN,
    TradingSession.LONDON_NY_OVERLAP,
    TradingSession.NY_AFTERNOON,
    TradingSession.CLOSE,
]


def now_utc():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def get_session_for_dt(dt: datetime) -> TradingSession:
    return SessionProfiler().get_session(dt.replace(tzinfo=None))


def fetch_market_movement(symbol: str, since: datetime, mt5_client=None) -> dict:
    if mt5_client is None:
        import MetaTrader5 as mt5
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, since, datetime.now())
    else:
        rates = mt5_client.copy_rates_range(symbol, mt5.TIMEFRAME_H1, since, datetime.now())
    if rates is None or len(rates) == 0:
        return {}
    import pandas as pd
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)

    sessions = {}
    for idx, row in df.iterrows():
        sess = get_session_for_dt(idx.to_pydatetime())
        label = SESSION_LABELS.get(sess, sess.value)
        if label not in sessions:
            sessions[label] = {"open": float(row["open"]), "high": float(row["high"]),
                               "low": float(row["low"]), "close": float(row["close"]),
                               "count": 1}
        else:
            s = sessions[label]
            s["high"] = max(s["high"], float(row["high"]))
            s["low"] = min(s["low"], float(row["low"]))
            s["close"] = float(row["close"])
            s["count"] += 1

    for label, s in sessions.items():
        s["range"] = round(s["high"] - s["low"], 2)
        s["change"] = round(s["close"] - s["open"], 2)
        s["direction"] = "ALCISTA" if s["change"] > 0 else "BAJISTA" if s["change"] < 0 else "PLANO"
    return sessions


def load_trade_stats(symbol: str, db_path: Path) -> list:
    """Read trade performance per session from meta_learning.db"""
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("""
            SELECT session,
                   COUNT(*) as total,
                   SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN profit <= 0 THEN 1 ELSE 0 END) as losses,
                   SUM(profit) as total_profit,
                   AVG(volume) as avg_volume,
                   AVG(duration_minutes) as avg_duration
            FROM trade_records
            WHERE profit IS NOT NULL AND profit != 0 AND symbol=?
            GROUP BY session
            ORDER BY total_profit DESC
        """, (symbol,)).fetchall()
        return [
            {
                "session": r[0] or "SIN_SESION",
                "total": r[1], "wins": r[2] or 0, "losses": r[3] or 0,
                "total_profit": r[4] or 0.0, "avg_volume": r[5] or 0.0,
                "avg_duration": r[6] or 0.0,
                "win_rate": (r[2] or 0) / r[1] * 100 if r[1] > 0 else 0.0,
            }
            for r in rows
        ]
    finally:
        conn.close()


def load_global_stats(symbol: str, db_path: Path) -> dict:
    if not db_path.exists():
        return {"total_trades": 0, "closed_trades": 0, "win_rate": 0.0, "total_profit": 0.0}
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("""
            SELECT COUNT(*),
                   SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END),
                   SUM(profit)
            FROM trade_records
            WHERE profit IS NOT NULL AND profit != 0 AND symbol=?
        """, (symbol,)).fetchone()
        total, wins, profit = row
        wins = wins or 0
        profit = profit or 0.0
        return {
            "total_trades": total,
            "closed_trades": total,
            "win_rate": wins / total if total > 0 else 0.0,
            "total_profit": profit,
        }
    finally:
        conn.close()


def generate_report_data(symbol: str, days: int = 1, mt5_client=None) -> dict:
    since = now_utc() - timedelta(days=days)
    db_path = _proj_root / "data" / "db" / symbol / "meta_learning.db"

    market = {}
    try:
        market = fetch_market_movement(symbol, since, mt5_client)
    except Exception as e:
        logger.warning(f"No se pudo obtener datos de mercado: {e}")

    trades = load_trade_stats(symbol, db_path)
    global_stats = load_global_stats(symbol, db_path)

    return {
        "symbol": symbol,
        "period_days": days,
        "period_start": since.isoformat(),
        "period_end": now_utc().isoformat(),
        "market": market,
        "trades_by_session": trades,
        "global_stats": global_stats,
    }


def report_text(symbol: str, days: int = 1, mt5_client=None):
    data = generate_report_data(symbol, days, mt5_client)

    print(f"\n{'='*70}")
    print(f"  INFORME DE SESION - {data['symbol']}")
    print(f"  Periodo: ultimos {data['period_days']} dia(s) - {data['period_start'][:10]} -> {data['period_end'][:16]} UTC")
    print(f"{'='*70}")

    market = data["market"]
    if market:
        print(f"\n  -- MOVIMIENTO DEL MERCADO --")
        print(f"  {'Session':<18} {'Apertura':<12} {'Maximo':<12} {'Minimo':<12} {'Cierre':<12} {'Rango':<10} {'Dir':<10}")
        print(f"  {'-'*76}")
        for sess in SESSION_ORDER:
            label = SESSION_LABELS.get(sess, sess.value)
            s = market.get(label)
            if s:
                print(f"  {label:<18} {s['open']:<12.2f} {s['high']:<12.2f} {s['low']:<12.2f} {s['close']:<12.2f} {s['range']:<10.2f} {s['direction']:<10}")
    else:
        print("\n  (Sin datos de mercado)")

    trades = data["trades_by_session"]
    print(f"\n  -- RENDIMIENTO POR SESION --")
    if trades:
        print(f"  {'Session':<18} {'Trades':>7} {'Ganadas':>8} {'Perdidas':>9} {'Win%':>7} {'Profit':>10} {'Vol':>7}")
        print(f"  {'-'*68}")
        for t in trades:
            wr = t["win_rate"]
            print(f"  {t['session']:<18} {t['total']:>7} {t['wins']:>8} {t['losses']:>9} {wr:>6.1f}% {t['total_profit']:>+10.2f} {t['avg_volume']:>7.2f}")
    else:
        print("  No hay trades cerrados en este periodo.")

    gs = data["global_stats"]
    print(f"\n  -- TOTALES --")
    print(f"  Trades: {gs['total_trades']} | Win Rate: {gs['win_rate']:.1%} | Profit: ${gs['total_profit']:+.2f}")

    if trades:
        best = max(trades, key=lambda t: t["win_rate"])
        worst = min(trades, key=lambda t: t["win_rate"])
        print(f"\n  -- MEJOR SESION --")
        print(f"  {best['session']}: {best['win_rate']:.1f}% WR, ${best['total_profit']:+.2f} profit ({best['total']} trades)")
        print(f"\n  -- PEOR SESION --")
        print(f"  {worst['session']}: {worst['win_rate']:.1f}% WR, ${worst['total_profit']:+.2f} profit ({worst['total']} trades)")

    print(f"\n{'='*70}\n")


def report_html(symbol: str, days: int = 1, mt5_client=None) -> Path:
    data = generate_report_data(symbol, days, mt5_client)

    mkt_rows = ""
    for sess in SESSION_ORDER:
        label = SESSION_LABELS.get(sess, sess.value)
        s = data["market"].get(label) if data["market"] else None
        if s:
            cls = "green" if s["direction"] == "ALCISTA" else ("red" if s["direction"] == "BAJISTA" else "")
            mkt_rows += f"<tr><td>{label}</td><td class='{cls}'>{s['open']:.2f}</td><td>{s['high']:.2f}</td><td>{s['low']:.2f}</td><td class='{cls}'>{s['close']:.2f}</td><td>{s['range']:.2f}</td></tr>"

    trade_rows = ""
    for t in data["trades_by_session"]:
        color = "green" if t["total_profit"] >= 0 else "red"
        trade_rows += f"<tr><td>{t['session']}</td><td>{t['total']}</td><td>{t['wins']}</td><td>{t['losses']}</td><td>{t['win_rate']:.1f}%</td><td style='color:{color}'>{t['total_profit']:+.2f}</td><td>{t['avg_volume']:.2f}</td></tr>"

    if data["trades_by_session"]:
        best = max(data["trades_by_session"], key=lambda t: t["win_rate"])
        worst = min(data["trades_by_session"], key=lambda t: t["win_rate"])
        best_sec = f"<p><strong>Mejor:</strong> {best['session']} ({best['win_rate']:.1f}% WR, ${best['total_profit']:+.2f})</p>"
        worst_sec = f"<p><strong>Peor:</strong> {worst['session']} ({worst['win_rate']:.1f}% WR, ${worst['total_profit']:+.2f})</p>"
    else:
        best_sec = worst_sec = ""

    gs = data["global_stats"]
    profit_color = "green" if gs["total_profit"] >= 0 else "red"

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Informe de Sesion - {symbol}</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Segoe UI',Arial,sans-serif; background:#0d1117; color:#c9d1d9; padding:2rem; }}
  h1 {{ color:#f59e0b; }}
  h2 {{ color:#58a6ff; margin-top:2rem; }}
  .meta {{ color:#8b949e; font-size:.9rem; margin-bottom:1.5rem; }}
  table {{ width:100%; border-collapse:collapse; margin-top:.5rem; }}
  th, td {{ padding:.5rem .75rem; text-align:left; border-bottom:1px solid #21262d; font-size:.9rem; }}
  th {{ background:#161b22; color:#8b949e; text-transform:uppercase; letter-spacing:.05em; }}
  .green {{ color:#3fb950 !important; }}
  .red {{ color:#f85149 !important; }}
  .stats {{ display:flex; gap:1rem; margin:1rem 0; flex-wrap:wrap; }}
  .stat {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:1rem 1.5rem; }}
  .stat .num {{ font-size:1.5rem; font-weight:700; color:#58a6ff; }}
  .stat .label {{ font-size:.8rem; color:#8b949e; }}
  .rec {{ margin-top:1.5rem; background:#161b22; border:1px solid #30363d; border-radius:8px; padding:1rem; }}
  .rec h3 {{ color:#f59e0b; margin-bottom:.5rem; }}
</style>
</head>
<body>
<h1>Informe de Sesion - {symbol}</h1>
<div class='meta'>Periodo: ultimos {data['period_days']} dia(s) - {data['period_start'][:10]} -> {data['period_end'][:16]} UTC</div>

<div class='stats'>
  <div class='stat'><div class='num'>{gs['closed_trades']}</div><div class='label'>Trades</div></div>
  <div class='stat'><div class='num'>{gs['win_rate']:.0%}</div><div class='label'>Win Rate</div></div>
  <div class='stat'><div class='num' style='color:{profit_color}'>{gs['total_profit']:+.2f}</div><div class='label'>Profit</div></div>
</div>

<h2>Movimiento del Mercado</h2>
<table><thead><tr><th>Session</th><th>Apertura</th><th>Maximo</th><th>Minimo</th><th>Cierre</th><th>Rango</th></tr></thead><tbody>
{mkt_rows if mkt_rows else '<tr><td colspan="6">Sin datos</td></tr>'}
</tbody></table>

<h2>Rendimiento por Sesion</h2>
<table><thead><tr><th>Session</th><th>Trades</th><th>Ganadas</th><th>Perdidas</th><th>Win%</th><th>Profit</th><th>Vol Prom</th></tr></thead><tbody>
{trade_rows if trade_rows else '<tr><td colspan="7">Sin trades cerrados</td></tr>'}
</tbody></table>

<div class='rec'>
<h3>Evaluacion por Sesion</h3>
{best_sec}{worst_sec}
<p>Proximo reporte: en 4h.</p>
</div>
</body></html>"""

    report_dir = _proj_root / "data" / "reports"
    report_dir.mkdir(exist_ok=True)
    path = report_dir / f"session_report_{symbol}_{now_utc().strftime('%Y%m%d_%H%M')}.html"
    path.write_text(html, encoding="utf-8")
    logger.info(f"Reporte de sesion generado: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Session Performance Report")
    parser.add_argument("--symbol", default="XAUUSDc", help="Simbolo")
    parser.add_argument("--days", type=int, default=1, help="Dias a analizar")
    parser.add_argument("--html", action="store_true", help="Generar HTML")
    args = parser.parse_args()

    report_text(args.symbol, args.days)
    if args.html:
        report_html(args.symbol, args.days)


if __name__ == "__main__":
    main()

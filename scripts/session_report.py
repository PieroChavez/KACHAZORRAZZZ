"""Session Performance Report
Shows how the market moved per session (Asia/London/NY) and how the
strategy performed in each. Includes ML recommendations for next sessions.

Usage:
    python scripts/session_report.py [--html] [--days 1]
"""
import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

import MetaTrader5 as mt5
import pandas as pd
from loguru import logger

from src.core.session_profiler import SessionProfiler, TradingSession
from src.strategies.fractal_learner import FractalLearner
from src.strategies.fractal_db import FractalDB

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


def load_credentials():
    config_dir = _proj_root / "config"
    with open(config_dir / "broker.json") as f:
        broker = json.load(f)["mt5"]
    login = os.environ.get("MT5_LOGIN") or broker.get("login")
    password = os.environ.get("MT5_PASSWORD") or broker.get("password")
    server = os.environ.get("MT5_SERVER") or broker.get("server")
    return login, password, server


def get_session_for_dt(dt: datetime) -> TradingSession:
    profiler = SessionProfiler()
    return profiler.get_session(dt.replace(tzinfo=None))


def fetch_market_movement(symbol: str, since: datetime) -> dict:
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, since, datetime.now())
    if rates is None or len(rates) == 0:
        return {}
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)

    sessions = {}
    for idx, row in df.iterrows():
        sess = get_session_for_dt(idx)
        label = SESSION_LABELS.get(sess, sess.value)
        if label not in sessions:
            sessions[label] = {"open": row["open"], "high": row["high"],
                               "low": row["low"], "close": row["close"],
                               "count": 1, "direction": ""}
        else:
            s = sessions[label]
            s["high"] = max(s["high"], row["high"])
            s["low"] = min(s["low"], row["low"])
            s["close"] = row["close"]
            s["count"] += 1

    for label, s in sessions.items():
        s["range"] = round(s["high"] - s["low"], 2)
        s["change"] = round(s["close"] - s["open"], 2)
        s["direction"] = "ALCISTA" if s["change"] > 0 else "BAJISTA" if s["change"] < 0 else "PLANO"
    return sessions


def now_utc():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def report_text(symbol: str, days: int):
    since = now_utc() - timedelta(days=days)
    learner = FractalLearner(symbol)

    print(f"\n{'='*70}")
    print(f"  INFORME DE SESION - {symbol}")
    print(f"  Periodo: ultimos {days} dia(s) - {since.strftime('%Y-%m-%d')} -> {now_utc().strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"{'='*70}")

    market = fetch_market_movement(symbol, since)
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
        print("\n  No hay datos de mercado disponibles.")

    print(f"\n  -- RENDIMIENTO POR SESION --")
    rows = learner._conn.execute("""
        SELECT session,
               COUNT(*) as total,
               SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN outcome='loss' THEN 1 ELSE 0 END) as losses,
               SUM(profit) as total_profit,
               SUM(volume) as total_volume,
               AVG(duration_hours) as avg_dur
        FROM fractal_trades
        WHERE outcome != 'open' AND symbol=?
        GROUP BY session
        ORDER BY total_profit DESC
    """, (symbol,)).fetchall()

    if rows:
        print(f"  {'Session':<18} {'Trades':>7} {'Ganadas':>8} {'Perdidas':>9} {'Win%':>7} {'Profit':>10} {'Vol':>7} {'Dur(h)':>8}")
        print(f"  {'-'*76}")
        for r in rows:
            session, total, wins, losses, total_profit, total_volume, avg_dur = r
            wins = wins or 0
            losses = losses or 0
            total_profit = total_profit or 0.0
            total_volume = total_volume or 0.0
            win_rate = wins / total * 100 if total > 0 else 0.0
            label = session if session else "SIN_SESION"
            print(f"  {label:<18} {total:>7} {wins:>8} {losses:>9} {win_rate:>6.1f}% {total_profit:>+10.2f} {total_volume:>7.2f} {avg_dur:>7.1f}h")
    else:
        print("  No hay trades cerrados en este periodo.")

    print(f"\n  -- ESTADO DEL APRENDIZAJE --")
    summary = learner.get_summary()
    print(f"  Trades registrados:  {summary['total_trades']} ({summary['closed_trades']} cerrados)")
    print(f"  Win rate global:     {summary['win_rate']:.1%}")
    print(f"  Profit total:        ${summary['total_profit']:+.2f}")
    if summary['volume_multipliers']:
        print(f"  Multiplicadores activos:")
        for k, v in sorted(summary['volume_multipliers'].items()):
            print(f"    {k}: {v}x")

    print(f"\n  -- RECOMENDACIONES PARA PROXIMAS SESIONES --")
    hourly_adjustments = {
        TradingSession.ASIAN: ("bajo", "Reducir volumen (x0.7). Esperar setups de alta probabilidad solamente."),
        TradingSession.LONDON_OPEN: ("medio-alto", "Volumen normal (x1.0). Activar alertas de proximidad."),
        TradingSession.LONDON_MID: ("medio", "Reducir ligeramente (x0.85). Buscar continuaciones."),
        TradingSession.NY_OPEN: ("alto", "Aumentar volumen (x1.2). Alta volatilidad al inicio."),
        TradingSession.LONDON_NY_OVERLAP: ("muy alto", "Maximo volumen (x1.3). Mayor oportunidad de movimientos fuertes."),
        TradingSession.NY_AFTERNOON: ("medio-bajo", "Volumen normal-bajo (x0.9). Reduccion de liquidez."),
        TradingSession.CLOSE: ("muy bajo", "Volumen minimo (x0.4). Evitar nuevas entradas."),
    }
    now = now_utc()
    current_session = get_session_for_dt(now)
    for sess in SESSION_ORDER:
        label = SESSION_LABELS.get(sess, sess.value)
        level, advice = hourly_adjustments[sess]
        now_marker = " << ACTUAL" if sess == current_session else ""
        print(f"  {label:<18} volatilidad {level:<12} - {advice}{now_marker}")

    print(f"\n{'='*70}")
    print(f"  Proximo analisis ML: cada 4h el sistema ajusta multiplicadores automaticamente")
    print(f"{'='*70}\n")

    learner.close()


def report_html(symbol: str, days: int):
    since = now_utc() - timedelta(days=days)
    learner = FractalLearner(symbol)
    summary = learner.get_summary()

    market = fetch_market_movement(symbol, since)

    rows = learner._conn.execute("""
        SELECT session,
               COUNT(*) as total,
               SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN outcome='loss' THEN 1 ELSE 0 END) as losses,
               SUM(profit) as total_profit,
               SUM(volume) as total_volume,
               AVG(duration_hours) as avg_dur
        FROM fractal_trades
        WHERE outcome != 'open' AND symbol=?
        GROUP BY session
        ORDER BY total_profit DESC
    """, (symbol,)).fetchall()

    mkt_rows = ""
    for sess in SESSION_ORDER:
        label = SESSION_LABELS.get(sess, sess.value)
        s = market.get(label) if market else None
        if s:
            cls = "green" if s["direction"] == "ALCISTA" else ("red" if s["direction"] == "BAJISTA" else "")
            mkt_rows += f"<tr><td>{label}</td><td class='{cls}'>{s['open']:.2f}</td><td>{s['high']:.2f}</td><td>{s['low']:.2f}</td><td class='{cls}'>{s['close']:.2f}</td><td>{s['range']:.2f}</td></tr>"

    trade_rows = ""
    for r in rows:
        session, total, wins, losses, tp, tv, dur = r
        wins = wins or 0
        losses = losses or 0
        tp = tp or 0.0
        tv = tv or 0.0
        wr = wins / total * 100 if total > 0 else 0.0
        color = "green" if tp >= 0 else "red"
        trade_rows += f"<tr><td>{session or '&mdash;'}</td><td>{total}</td><td>{wins}</td><td>{losses}</td><td>{wr:.1f}%</td><td style='color:{color}'>{tp:+.2f}</td><td>{tv:.2f}</td></tr>"

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
  .rec p {{ color:#c9d1d9; }}
</style>
</head>
<body>
<h1>Informe de Sesion - {symbol}</h1>
<div class='meta'>Periodo: ultimos {days} dia(s) - {since.strftime('%Y-%m-%d')} -> {now_utc().strftime('%Y-%m-%d %H:%M')} UTC</div>

<div class='stats'>
  <div class='stat'><div class='num'>{summary['closed_trades']}</div><div class='label'>Trades</div></div>
  <div class='stat'><div class='num'>{summary['win_rate']:.0%}</div><div class='label'>Win Rate</div></div>
  <div class='stat'><div class='num' style='color:{"#3fb950" if summary["total_profit"]>=0 else "#f85149"}'>{summary['total_profit']:+.2f}</div><div class='label'>Profit</div></div>
</div>

<h2>Movimiento del Mercado</h2>
<table><thead><tr><th>Session</th><th>Apertura</th><th>Maximo</th><th>Minimo</th><th>Cierre</th><th>Rango</th></tr></thead><tbody>
{mkt_rows if mkt_rows else '<tr><td colspan="6">Sin datos</td></tr>'}
</tbody></table>

<h2>Rendimiento por Sesion</h2>
<table><thead><tr><th>Session</th><th>Trades</th><th>Ganadas</th><th>Perdidas</th><th>Win%</th><th>Profit</th><th>Vol</th></tr></thead><tbody>
{trade_rows if trade_rows else '<tr><td colspan="7">Sin trades cerrados</td></tr>'}
</tbody></table>

<div class='rec'>
<h3>Recomendaciones del Aprendiz</h3>
<p>Multiplicadores de volumen activos: {json.dumps(summary['volume_multipliers'])}</p>
<p>Proximo analisis automatico: cada 4h.</p>
</div>
</body></html>"""

    report_dir = _proj_root / "data" / "reports"
    report_dir.mkdir(exist_ok=True)
    path = report_dir / f"session_report_{symbol}_{now_utc().strftime('%Y%m%d_%H%M')}.html"
    path.write_text(html, encoding="utf-8")
    print(f"\nReporte HTML: file:///{path.as_posix()}")
    learner.close()
    return path


def main():
    parser = argparse.ArgumentParser(description="Session Performance Report")
    parser.add_argument("--symbol", default="XAUEURm", help="Simbolo")
    parser.add_argument("--days", type=int, default=1, help="Dias a analizar")
    parser.add_argument("--html", action="store_true", help="Generar HTML")
    args = parser.parse_args()

    login, password, server = load_credentials()
    if not mt5.initialize(login=int(login), password=password, server=server):
        print(f"Error conectando a MT5: {mt5.last_error()}")
        return

    acc = mt5.account_info()
    if acc:
        print(f"Conectado a MT5 - {server} | Balance: ${acc.balance:.2f}")
    else:
        print(f"Conectado a MT5 - {server}")

    report_text(args.symbol, args.days)
    if args.html:
        report_html(args.symbol, args.days)

    mt5.shutdown()


if __name__ == "__main__":
    main()

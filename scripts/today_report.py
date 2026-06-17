"""Reporte detallado de operaciones del día de hoy"""
import sqlite3
from datetime import datetime, date, timedelta
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
SYMBOL = "XAUUSDm"
TODAY = date.today()
TODAY_ISO = TODAY.isoformat()
YESTERDAY_ISO = (TODAY - timedelta(days=1)).isoformat()

DB_DIR = ROOT / "data" / "db" / SYMBOL


def fmt_ts(ts_str):
    if not ts_str:
        return "---"
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt.strftime("%H:%M:%S")
    except:
        return ts_str[:19]


def fmt_price(v):
    if v is None:
        return "---"
    return f"{v:.2f}"


def fmt_profit(v):
    if v is None:
        return "---"
    return f"{v:+.2f}"


def direction_arrow(dir_):
    if dir_ and dir_.upper() == "BUY":
        return "🟢 BUY"
    if dir_ and dir_.upper() == "SELL":
        return "🔴 SELL"
    if dir_ == "bullish":
        return "🟢 BULL"
    if dir_ == "bearish":
        return "🔴 BEAR"
    return dir_ or "---"


def load_order_packs():
    """Carga packs de order_packs.db con sub_orders"""
    db = DB_DIR / "order_packs.db"
    if not db.exists():
        print(f"⚠ No se encuentra {db}")
        return []
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT op.*, so.id as sub_id, so.position_number,
               so.ticket, so.status as sub_status,
               so.entry_price as sub_entry, so.sl_current, so.profit, so.closed_at
        FROM order_packs op
        LEFT JOIN sub_orders so ON so.pack_id = op.id
        WHERE date(op.created_at) = ? OR date(op.created_at) = ?
        ORDER BY op.id, so.position_number
    """, (TODAY_ISO, YESTERDAY_ISO)).fetchall()
    conn.close()

    packs = defaultdict(lambda: {"subs": [], "fractal": None, "learner": None})
    for r in rows:
        pid = r["id"]
        if pid not in packs:
            packs[pid] = {
                "id": pid,
                "fractal_id": r["fractal_id"],
                "direction": r["direction"],
                "entry_price": r["entry_price"],
                "sl_initial": r["sl_initial"],
                "tp1": r["tp1"],
                "tp2": r["tp2"],
                "tp3": r["tp3"] if "tp3" in r.keys() else None,
                "volume_total": r["volume_total"],
                "status": r["status"],
                "breakeven": bool(r["breakeven_activated"]),
                "trailing": bool(r["trailing_activated"]),
                "source_tf": r["source_timeframe"],
                "created_at": r["created_at"],
                "subs": [],
            }
        if r["sub_id"]:
            packs[pid]["subs"].append({
                "sub_id": r["sub_id"],
                "n": r["position_number"],
                "ticket": r["ticket"],
                "status": r["sub_status"],
                "entry": r["sub_entry"],
                "sl": r["sl_current"],
                "profit": r["profit"],
                "closed_at": r["closed_at"],
            })
    return list(packs.values())


def load_fractals(pack_ids):
    """Carga detalles de fractal_state.db para cada fractal_id"""
    if not pack_ids:
        return {}
    db = DB_DIR / "fractal_state.db"
    if not db.exists():
        return {}
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" for _ in pack_ids)
    rows = conn.execute(f"""
        SELECT * FROM fractals WHERE id IN ({placeholders})
    """, pack_ids).fetchall()
    conn.close()
    return {r["id"]: dict(r) for r in rows}


def load_learner_trades():
    """Carga trades de fractal_learner.db para los packs de hoy"""
    db = DB_DIR / "fractal_learner.db"
    if not db.exists():
        return {}
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT * FROM fractal_trades
        WHERE date(opened_at) = ? OR date(closed_at) = ?
        ORDER BY opened_at
    """, (TODAY_ISO, TODAY_ISO)).fetchall()
    conn.close()
    return {r["pack_id"]: dict(r) for r in rows}


def load_active_fractals():
    """Carga fractales activos NO hit de fractal_state.db"""
    db = DB_DIR / "fractal_state.db"
    if not db.exists():
        return []
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT * FROM fractals
        WHERE active=1 AND hit_entry=0
        ORDER BY created_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def describe_fractal(f):
    if not f:
        return "---"
    arrow = direction_arrow(f["direction"])
    sub = "SUB" if f["is_subfractal"] else "MACRO"
    return f"{arrow} {f['timeframe']} [{sub}] L1={fmt_price(f['level1'])} L0={fmt_price(f['level0'])} entry={fmt_price(f['fib_072'])}"


def main():
    print("=" * 72)
    print(f"📊 REPORTE DE OPERACIONES - {TODAY}")
    print(f"   Símbolo: {SYMBOL}")
    print("=" * 72)

    # ── 1. Packs de hoy ────────────────────────────────────────────────
    packs = load_order_packs()
    pack_ids = [p["fractal_id"] for p in packs if p["fractal_id"]]
    fractals = load_fractals(pack_ids)
    learner_map = load_learner_trades()

    print(f"\n📦 PACKS DEL DÍA: {len(packs)}\n")

    if not packs:
        print("   (ningún pack creado hoy)")
    else:
        for p in packs:
            f = fractals.get(p["fractal_id"])
            lr = learner_map.get(p["id"])
            arrow = direction_arrow(p["direction"])
            status_icon = {
                "active": "🟡",
                "closed": "✅",
                "cancelled": "❌",
                "cancelled_restart": "💀",
            }.get(p["status"], "⚪")

            print(f"  {status_icon} Pack #{p['id']} {arrow} | TF: {p['source_tf']} | "
                  f"Entry: {fmt_price(p['entry_price'])} | SL: {fmt_price(p['sl_initial'])}")
            print(f"     TP1: {fmt_price(p['tp1'])} | TP2: {fmt_price(p['tp2'])} | "
                  f"Vol: {p['volume_total']} | Status: {p['status'].upper()}")
            print(f"     BE: {'✅' if p['breakeven'] else '⬜'} | "
                  f"Trailing: {'✅' if p['trailing'] else '⬜'} | "
                  f"Creado: {fmt_ts(p['created_at'])}")

            # Motivo (fractal)
            if f:
                print(f"     📐 Motivo: {describe_fractal(f)}")
                if f.get("note"):
                    print(f"        Nota: {f['note']}")
                if lr:
                    outcome = lr.get("outcome", "open")
                    profit = lr.get("profit", 0)
                    dur = lr.get("duration_hours", 0)
                    if outcome != "open":
                        print(f"     💰 Resultado: {'🏆 GANANCIA' if profit > 0 else '💀 PÉRDIDA'} "
                              f"{fmt_profit(profit)} | Duración: {dur:.1f}h | {outcome.upper()}")
            else:
                print(f"     📐 Fractal #{p['fractal_id']} (no encontrado en DB)")

            # Sub-órdenes
            for s in p["subs"]:
                sub_icon = {"active": "🟡", "filled": "🟢", "closed": "✅",
                            "closed_by_sl": "💀", "cancelled": "❌"}.get(s["status"], "⚪")
                print(f"     {sub_icon}  P{s['n']} | Ticket: {s['ticket']} | "
                      f"SL: {fmt_price(s['sl'])} | "
                      f"Profit: {fmt_profit(s['profit'])} | "
                      f"Status: {s['status']}")
            print()

    # ── 2. Fractales activos esperando entrada ─────────────────────────
    active_f = load_active_fractals()
    macro = [f for f in active_f if not f["is_subfractal"]]
    sub = [f for f in active_f if f["is_subfractal"]]

    print(f"⏳ FRACTALES ESPERANDO ENTRY: {len(active_f)} "
          f"(Macro: {len(macro)}, Sub: {len(sub)})\n")

    if not active_f:
        print("   (ninguno)")
    else:
        for f in active_f:
            tag = "SUB" if f["is_subfractal"] else "M"
            arrow = direction_arrow(f["direction"])
            print(f"  · #{f['id']} {arrow} {f['timeframe']} [{tag}] "
                  f"L1={fmt_price(f['level1'])} → Entry={fmt_price(f['fib_072'])} "
                  f"(rango={fmt_price(f['level0'] - f['level1'])})")
            if f.get("note"):
                print(f"    {f['note']}")
        print()

    # ── 3. Fractales hit hoy (entry alcanzado) ─────────────────────────
    db_fs = DB_DIR / "fractal_state.db"
    if db_fs.exists():
        conn = sqlite3.connect(str(db_fs))
        conn.row_factory = sqlite3.Row
        hit_today = conn.execute("""
            SELECT * FROM fractals
            WHERE hit_entry=1 AND date(created_at) = ?
            ORDER BY created_at DESC
        """, (TODAY_ISO,)).fetchall()
        conn.close()
        if hit_today:
            print(f"🎯 FRACTALES CON ENTRY ALCANZADO HOY: {len(hit_today)}\n")
            for row in hit_today:
                r = dict(row)
                tag = "SUB" if r["is_subfractal"] else "M"
                arrow = direction_arrow(r["direction"])
                print(f"  · #{r['id']} {arrow} {r['timeframe']} [{tag}] "
                      f"Entry={fmt_price(r['fib_072'])} "
                      f"→ Ejecutado en {fmt_price(r['entry_price'])}")
                if r.get("note"):
                    print(f"    {r['note']}")
            print()

    # ── 4. Resumen MT5 (si es posible conectar) ────────────────────────
    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            positions = mt5.positions_get(symbol=SYMBOL)
            if positions:
                print(f"📈 POSICIONES ABIERTAS EN MT5 ({SYMBOL}): {len(positions)}\n")
                for pos in positions:
                    arrow = "🟢 BUY" if pos.type == 0 else "🔴 SELL"
                    print(f"  {arrow} Vol: {pos.volume} | "
                          f"Open: {fmt_price(pos.price_open)} | "
                          f"SL: {fmt_price(pos.sl)} | TP: {fmt_price(pos.tp)} | "
                          f"Profit: {fmt_profit(pos.profit)}")
                    if pos.comment:
                        print(f"     Comment: {pos.comment}")
                    print(f"     Ticket: {pos.ticket} | Magic: {pos.magic}")
                print()
            mt5.shutdown()
        else:
            print("⚠ No se pudo conectar a MT5\n")
    except ImportError:
        print("⚠ MetaTrader5 no disponible\n")
    except Exception as e:
        print(f"⚠ Error conectando MT5: {e}\n")

    print("=" * 72)
    print(f"Fin del reporte - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)


if __name__ == "__main__":
    main()

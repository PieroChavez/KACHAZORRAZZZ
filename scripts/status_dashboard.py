"""Fractal Cascade Strategy — Status Dashboard
Shows active fractals, pending entries, and open order packs.
Outputs both a terminal report and a standalone HTML file.

Usage:
    python scripts/status_dashboard.py [--html]
"""
import sys
import json
import os
from pathlib import Path
from datetime import datetime

_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

from src.strategies.fractal_db import FractalDB
from src.strategies.order_pack import OrderPackManager
from src.core.multi_timeframe import TIMEFRAME_ORDER


def report_terminal(symbol: str = "XAUEURm"):
    db = FractalDB(symbol)
    fractals = db.get_active_fractals()
    not_hit = [f for f in fractals if not f.hit_entry]
    hit = [f for f in fractals if f.hit_entry]

    sub = [f for f in fractals if f.is_subfractal]
    macro = [f for f in fractals if not f.is_subfractal]

    print(f"\n{'='*65}")
    print(f"  CAZADOR FRACTAL — {symbol}")
    print(f"{'='*65}")
    print(f"  Fractales activos:  {len(fractals)}  (macro={len(macro)}, 5M sub={len(sub)})")
    print(f"  Esperando entrada:  {len(not_hit)}")
    print(f"  Entrados (hit):     {len(hit)}")
    print(f"{'='*65}")

    if not_hit:
        print(f"\n  ── ESPERANDO ENTRADA (0.72) ──")
        print(f"  {'ID':<4} {'TF':<8} {'Dir':<8} {'L1 (SL)':<12} {'L0':<12} {'0.72':<12}  {'Tipo':<6}")
        print(f"  {'─'*62}")
        for f in sorted(not_hit, key=lambda x: x.timeframe, reverse=True):
            tip = "SUB" if f.is_subfractal else "MACRO"
            print(f"  {f.id:<4} {f.timeframe:<8} {f.direction:<8} {f.level1:<12.2f} {f.level0:<12.2f} {f.fib_072:<12.2f}  {tip:<6}")

    if hit:
        print(f"\n  ── POSICIONES ACTIVAS ──")
        print(f"  {'ID':<4} {'TF':<8} {'Dir':<8} {'Entry':<12} {'SL':<12} {'Tipo':<6} {'Note'}")
        print(f"  {'─'*62}")
        for f in sorted(hit, key=lambda x: x.created_at, reverse=True):
            tip = "SUB" if f.is_subfractal else "MACRO"
            print(f"  {f.id:<4} {f.timeframe:<8} {f.direction:<8} {f.entry_price:<12.2f} {f.sl_price:<12.2f} {tip:<6} {f.note[:24]}")


def html_dashboard(symbol: str = "XAUEURm"):
    db = FractalDB(symbol)
    fractals = db.get_active_fractals()
    not_hit = [f for f in fractals if not f.hit_entry]
    hit = [f for f in fractals if f.hit_entry]
    macro = [f for f in fractals if not f.is_subfractal]
    sub = [f for f in fractals if f.is_subfractal]

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Cazador Fractal — {symbol}</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Segoe UI',Arial,sans-serif; background:#0d1117; color:#c9d1d9; padding:2rem; }}
  h1 {{ color:#58a6ff; margin-bottom:.5rem; }}
  .stats {{ display:flex; gap:1rem; margin:1rem 0 2rem; flex-wrap:wrap; }}
  .stat {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:1rem 1.5rem; min-width:130px; }}
  .stat .num {{ font-size:2rem; font-weight:700; color:#58a6ff; }}
  .stat .label {{ font-size:.85rem; color:#8b949e; }}
  .stat.sub .num {{ color:#f87171; }}
  .stat.alert .num {{ color:#d29922; }}
  table {{ width:100%; border-collapse:collapse; margin-top:1rem; }}
  th, td {{ padding:.5rem .75rem; text-align:left; border-bottom:1px solid #21262d; font-size:.9rem; }}
  th {{ background:#161b22; color:#8b949e; font-weight:600; text-transform:uppercase; letter-spacing:.05em; }}
  tr:hover td {{ background:#1c2333; }}
  .bullish {{ color:#3fb950; }}
  .bearish {{ color:#f85149; }}
  .status-hit {{ color:#3fb950; }}
  .status-wait {{ color:#d29922; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:12px; font-size:.75rem; }}
  .badge-4H {{ background:#1f3a5f; color:#58a6ff; }}
  .badge-2H {{ background:#1f3a5f; color:#58a6ff; }}
  .badge-30min {{ background:#27272a; color:#a1a1aa; }}
  .badge-15min {{ background:#27272a; color:#a1a1aa; }}
  .badge-5min {{ background:#3b1a1a; color:#f87171; }}
</style>
</head>
<body>
<h1>🧬 Cazador Fractal — {symbol}</h1>
<div class="stats">
  <div class="stat"><div class="num">{len(fractals)}</div><div class="label">Fractales</div></div>
  <div class="stat"><div class="num">{len(macro)}</div><div class="label">Macro</div></div>
  <div class="stat sub"><div class="num">{len(sub)}</div><div class="label">Sub 5M</div></div>
  <div class="stat alert"><div class="num">{len(not_hit)}</div><div class="label">Esperando 0.72</div></div>
  <div class="stat"><div class="num">{len(hit)}</div><div class="label">Entrados</div></div>
</div>
"""

    if not_hit:
        html += """<h2 style="color:#d29922">⏳ Esperando Entrada (0.72)</h2><table>
<tr><th>ID</th><th>TF</th><th>Tipo</th><th>Dir</th><th>L1 (SL)</th><th>L0</th><th>0.72</th><th>Rango</th></tr>"""
        for f in sorted(not_hit, key=lambda x: x.timeframe, reverse=True):
            rng = abs(f.level0 - f.level1)
            cls = "bullish" if f.direction == "bullish" else "bearish"
            tip = "SUB" if f.is_subfractal else "MACRO"
            html += f"<tr><td>{f.id}</td><td><span class='badge badge-{f.timeframe}'>{f.timeframe}</span></td><td>{tip}</td><td class='{cls}'>{f.direction.upper()}</td><td>{f.level1:.2f}</td><td>{f.level0:.2f}</td><td class='status-wait'>{f.fib_072:.2f}</td><td>{rng:.2f}</td></tr>"
        html += "</table>"

    if hit:
        html += """<h2 style="color:#3fb950">✅ Posiciones Activas</h2><table>
<tr><th>ID</th><th>TF</th><th>Tipo</th><th>Dir</th><th>Entry</th><th>SL</th><th>Note</th></tr>"""
        for f in sorted(hit, key=lambda x: x.created_at, reverse=True):
            cls = "bullish" if f.direction == "bullish" else "bearish"
            tip = "SUB" if f.is_subfractal else "MACRO"
            html += f"<tr><td>{f.id}</td><td><span class='badge badge-{f.timeframe}'>{f.timeframe}</span></td><td>{tip}</td><td class='{cls}'>{f.direction.upper()}</td><td>{f.entry_price:.2f}</td><td>{f.sl_price:.2f}</td><td>{f.note}</td></tr>"
        html += "</table>"

    if not not_hit and not hit:
        html += "<p style='color:#8b949e; margin-top:2rem;'>No hay fractales activos.</p>"

    html += f"""<p style="margin-top:2rem;font-size:.8rem;color:#484f58;">
      Reporte generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
    </p></body></html>"""

    out_path = _proj_root / "data" / f"dashboard_{symbol}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"  Dashboard HTML: file:///{out_path.as_posix()}")
    return out_path


if __name__ == "__main__":
    use_html = "--html" in sys.argv
    symbol = "XAUEURm"
    for a in sys.argv[1:]:
        if not a.startswith("--"):
            symbol = a

    report_terminal(symbol)
    if use_html:
        path = html_dashboard(symbol)
        print(f"\nAbrir en navegador: file:///{path.as_posix()}")

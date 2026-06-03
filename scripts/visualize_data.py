"""Visual inspection of downloaded historical data with fractal Fibonacci markers
Usage:
    python scripts/visualize_data.py [symbol] [--tf 4H 2H 5min]
"""
import sys
import os
import json
from pathlib import Path
from datetime import datetime, date

_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
import numpy as np

from src.adapters.mt5_client import MT5Client
from src.core.multi_timeframe import (
    MultiTimeframeFetcher, TIMEFRAME_ORDER, TIMEFRAME_GROUPS
)
from src.strategies.fractal_db import FractalDB, Fractal
from src.utils.helpers import pip_size, find_swing_points


FIB_LEVEL = 0.72
HISTORICAL_COUNT = 5000


def load_credentials():
    config_dir = _proj_root / "config"
    with open(config_dir / "broker.json") as f:
        broker = json.load(f)["mt5"]
    login = os.environ.get("MT5_LOGIN") or broker.get("login")
    password = os.environ.get("MT5_PASSWORD") or broker.get("password")
    server = os.environ.get("MT5_SERVER") or broker.get("server")
    return login, password, server


def detect_fractals_for_tf(tf: str, df: pd.DataFrame, db: FractalDB):
    """Detect bullish and bearish fractals matching the strategy logic"""
    if df is None or len(df) < 30:
        return
    highs, lows = find_swing_points(df, lookback=3)
    if len(highs) < 3 or len(lows) < 3:
        return

    # Bullish: low (level1) → bos to high (level0), entry at 0.72 retrace down
    for j in range(1, len(highs)):
        curr_high_idx = highs[j]
        for k in range(len(lows)):
            low_idx = lows[k]
            if low_idx > curr_high_idx:
                low_price = df["low"].iloc[low_idx]
                if k > 0 and low_price > df["low"].iloc[lows[k - 1]]:
                    level1 = df["low"].iloc[low_idx]
                    level0 = df["high"].iloc[curr_high_idx]
                    if level0 > level1:
                        fib_range = level0 - level1
                        fib_072 = level1 + FIB_LEVEL * fib_range
                        note = f"BULL {tf} L1={level1:.2f} L0={level0:.2f}"
                        f = Fractal(symbol=db.symbol, timeframe=tf, direction="bullish",
                                    level0=level0, level1=level1, fib_072=fib_072,
                                    swing_high=level0, swing_low=level1,
                                    bos_index=curr_high_idx, note=note)
                        db.add_fractal(f)
                break

    # Bearish: high (level1) → bos to low (level0), entry at 0.72 retrace up
    for j in range(1, len(lows)):
        curr_low_idx = lows[j]
        for k in range(len(highs)):
            high_idx = highs[k]
            if high_idx > curr_low_idx:
                high_price = df["high"].iloc[high_idx]
                if k > 0 and high_price < df["high"].iloc[highs[k - 1]]:
                    level1 = df["high"].iloc[high_idx]
                    level0 = df["low"].iloc[curr_low_idx]
                    if level1 > level0:
                        fib_range = level1 - level0
                        fib_072 = level0 + FIB_LEVEL * fib_range
                        note = f"BEAR {tf} L1={level1:.2f} L0={level0:.2f}"
                        f = Fractal(symbol=db.symbol, timeframe=tf, direction="bearish",
                                    level0=level0, level1=level1, fib_072=fib_072,
                                    swing_high=level1, swing_low=level0,
                                    bos_index=curr_low_idx, note=note)
                        db.add_fractal(f)
                break


def print_summary(symbol: str, data: dict):
    print(f"\n{'='*60}")
    print(f"  {symbol} — Resumen de datos históricos ({HISTORICAL_COUNT} velas por TF)")
    print(f"{'='*60}")
    print(f"  {'TF':<8} {'Velas':>7} {'Desde':<20} {'Hasta':<20} {'Rango'}")
    print(f"  {'-'*60}")
    for tf in TIMEFRAME_ORDER:
        df = data.get(tf)
        if df is None or len(df) == 0:
            print(f"  {tf:<8} {'—':>7}")
            continue
        n = len(df)
        start = df["time"].iloc[0].strftime("%Y-%m-%d %H:%M")
        end = df["time"].iloc[-1].strftime("%Y-%m-%d %H:%M")
        days = (df["time"].iloc[-1] - df["time"].iloc[0]).days
        print(f"  {tf:<8} {n:>7}  {start:<20} {end:<20}  {days} días")
    print(f"{'='*60}\n")


def plot_timeframe(symbol: str, tf_name: str, df: pd.DataFrame,
                   fractals: list, save_dir: Path):
    if df is None or len(df) < 5:
        print(f"  Datos insuficientes para graficar {tf_name}")
        return

    plot_df = df.set_index("time")
    plot_df.index = pd.to_datetime(plot_df.index)

    tf_fractals = [f for f in fractals if f.timeframe == tf_name and f.active]

    title = f"{symbol} — {tf_name} ({len(df)} velas, {len(tf_fractals)} fractales)"
    save_path = save_dir / f"{symbol}_{tf_name}.png"

    style = mpf.make_mpf_style(base_mpf_style="charles",
        rc={"figure.facecolor": "#0d1117", "axes.facecolor": "#0d1117",
            "axes.edgecolor": "#30363d", "axes.labelcolor": "#c9d1d9",
            "text.color": "#c9d1d9", "grid.color": "#21262d",
            "grid.alpha": 0.3, "xtick.color": "#8b949e",
            "ytick.color": "#8b949e"})

    extra_plots = []
    volumes = mpf.make_addplot(plot_df["volume"], panel=1, color="#30363d", alpha=0.5, width=0.8)
    extra_plots.append(volumes)

    colors = {"bullish": "#3fb950", "bearish": "#f85149"}
    for i, f in enumerate(tf_fractals):
        if f.fib_072 == 0:
            continue
        fib_line = pd.Series(f.fib_072, index=plot_df.index)
        color = colors.get(f.direction, "#58a6ff")
        extra_plots.append(
            mpf.make_addplot(fib_line, panel=0, color=color,
                             linestyle="--", alpha=0.7, width=1.0)
        )

    fig, axes = mpf.plot(
        plot_df, type="candle", style=style, title=title,
        volume=False, addplot=extra_plots,
        figsize=(16, 10), returnfig=True,
        savefig=str(save_path), tight_layout=True,
    )

    ax_main = axes[0]
    colors_patch = {"bullish": "#3fb950", "bearish": "#f85149"}
    for f in tf_fractals:
        if f.fib_072 == 0:
            continue
        color = colors_patch.get(f.direction, "#58a6ff")
        label = f"{'🟢' if f.direction == 'bullish' else '🔴'} {f.direction.upper()} 0.72={f.fib_072:.2f}"
        ax_main.axhline(y=f.fib_072, color=color, linestyle="--", alpha=0.6, linewidth=1)
        ax_main.axhline(y=f.level1, color="#f0883e", linestyle=":", alpha=0.4, linewidth=0.7)
        ax_main.axhline(y=f.level0, color="#f0883e", linestyle=":", alpha=0.4, linewidth=0.7)
        ax_main.annotate(label, xy=(plot_df.index[-1], f.fib_072),
                        xytext=(10, 0), textcoords="offset points",
                        fontsize=7, color=color, alpha=0.8,
                        va="center", ha="left")

    ax_main.set_ylabel("Precio", color="#8b949e")
    ax_main.legend_.remove() if ax_main.legend_ else None

    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gráfico guardado: {save_path}")


def main():
    args = sys.argv[1:]
    symbols = []
    target_tfs = None

    i = 0
    while i < len(args):
        if args[i] == "--tf":
            i += 1
            target_tfs = []
            while i < len(args) and not args[i].startswith("--"):
                target_tfs.append(args[i])
                i += 1
        else:
            symbols.append(args[i])
            i += 1

    if not symbols:
        symbols = ["XAUUSDm", "XAUEURm"]

    login, password, server = load_credentials()
    if not all([login, password, server]):
        print("Error: Credenciales MT5 no encontradas")
        sys.exit(1)

    print("Conectando a MT5...")
    mt5 = MT5Client(login=login, password=password, server=server)
    if not mt5.connect():
        print("Error: No se pudo conectar a MT5")
        sys.exit(1)

    fetcher = MultiTimeframeFetcher(mt5)

    save_dir = _proj_root / "data" / "charts"
    save_dir.mkdir(exist_ok=True)

    for sym in symbols:
        print(f"\nDescargando histórico para {sym} ({HISTORICAL_COUNT} velas)...")
        data = fetcher.init_historical(sym, count=HISTORICAL_COUNT)

        if not data:
            print(f"  Sin datos para {sym}")
            continue

        print_summary(sym, data)

        db = FractalDB(sym)

        tfs = target_tfs if target_tfs else TIMEFRAME_ORDER
        for tf in tfs:
            if tf not in data:
                print(f"  TF {tf} no disponible para {sym}")
                continue
            detect_fractals_for_tf(tf, data[tf], db)

        fractals = db.get_active_fractals()
        print(f"  Fractales detectados: {len(fractals)}")
        for f in fractals:
            cls = "🟢" if f.direction == "bullish" else "🔴"
            print(f"    {cls} #{f.id} {f.timeframe} {f.direction.upper():7s} "
                   f"L1={f.level1:.2f} L0={f.level0:.2f} 0.72={f.fib_072:.2f}")

        for tf in tfs:
            if tf not in data:
                continue
            plot_timeframe(sym, tf, data[tf], fractals, save_dir)

    mt5.disconnect()
    print(f"\nGráficos guardados en: {save_dir}")
    print("Listo.")


if __name__ == "__main__":
    main()

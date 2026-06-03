"""Integrated Backtester for SMC Trading Bot
Fetches M15/M1 data from MT5, runs strategy, simulates trades with slippage/spread,
reports comprehensive performance metrics.
"""
import json
import sys
import random
import math
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import MetaTrader5 as mt5

sys.path.insert(0, str(Path(__file__).parent.parent))
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")

from src.core.strategy_engine import StrategyEngine, ScoringConfig
from src.utils.helpers import pip_size
from src.bot import SymbolProfile, StrategyParams

SYMBOL = "XAUUSDm"
HTF_TF = mt5.TIMEFRAME_M15
LTF_TF = mt5.TIMEFRAME_M1
HTF_BARS = 300
LTF_BARS = 2000
MIN_SCORE = 65.0
SLIPPAGE_PIPS = 0.5
SPREAD_PIPS = 0.25

def to_df(rates):
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df

def simulate_trade_candle(entry, sl, tp, direction, candles_df, slippage_pips, pip):
    for i in range(len(candles_df)):
        c = candles_df.iloc[i]
        sl_with_slippage = sl
        tp_with_slippage = tp
        if direction == "BUY":
            if c["low"] <= sl_with_slippage:
                fill_price = min(c["high"], sl_with_slippage + slippage_pips * pip)
                return "SL", fill_price, i
            if c["high"] >= tp_with_slippage:
                fill_price = max(c["low"], tp_with_slippage - slippage_pips * pip)
                return "TP", fill_price, i
        else:
            if c["high"] >= sl_with_slippage:
                fill_price = max(c["low"], sl_with_slippage - slippage_pips * pip)
                return "SL", fill_price, i
            if c["low"] <= tp_with_slippage:
                fill_price = min(c["high"], tp_with_slippage + slippage_pips * pip)
                return "TP", fill_price, i
    return "OPEN", candles_df.iloc[-1]["close"], len(candles_df)

def dict_to_profile(d):
    return SymbolProfile(
        symbol=d["symbol"],
        htf_context=d.get("htf_context", "M15"),
        htf_secondary=d.get("htf_secondary", "M5"),
        ltf_trigger=d.get("ltf_trigger", "M1"),
        ltf_refine=d.get("ltf_refine", "M1"),
        min_retracement=d.get("min_retracement", 0.50),
        fvg_entry_level=d.get("fvg_entry_level", 0.50),
        sl_buffer_pips=d.get("sl_buffer_pips", 3.0),
        sl_fixed_pips=d.get("sl_fixed_pips", 2.0),
        min_rr_ratio=d.get("min_rr_ratio", 2.0),
        tp_fixed_pips=d.get("tp_fixed_pips", 0),
        risk_per_trade_pct=d.get("risk_per_trade_pct", 1.0),
        max_concurrent_trades=d.get("max_concurrent_trades", 1),
        max_daily_loss_pct=d.get("max_daily_loss_pct", 4.0),
        max_daily_trades=d.get("max_daily_trades", 15),
        allowed_sessions=d.get("allowed_sessions", [(7, 11), (12, 16)]),
        sl_min_pips=d.get("sl_min_pips", 5.0),
        cooldown_bars_m1=d.get("cooldown_bars_m1", 5),
    )

def dict_to_params(d):
    return StrategyParams(
        consolidation_max_atr_ratio=d.get("consolidation_max_atr_ratio", 0.5),
        consolidation_min_bars=d.get("consolidation_min_bars", 8),
        expansion_tick_acceleration=d.get("expansion_tick_acceleration", 1.5),
        gap_detection_pips=d.get("gap_detection_pips", 5.0),
        macro_event_filter=d.get("macro_event_filter", True),
        min_retracement_level=d.get("min_retracement_level", 0.50),
        breakout_validation=d.get("breakout_validation", "BodyClose"),
        fvg_min_size_atr_ratio=d.get("fvg_min_size_atr_ratio", 0.2),
        fvg_entry_level=d.get("fvg_entry_level", 0.50),
        void_min_size_pips=d.get("void_min_size_pips", 3.0),
        pivot_length=d.get("pivot_length", 8),
        wyckoff_min_phase_bars=d.get("wyckoff_min_phase_bars", 10),
        sequence_length=d.get("sequence_length", 3),
        news_filter_active=d.get("news_filter_active", True),
        news_buffer_minutes=d.get("news_buffer_minutes", 30),
    )

def calculate_metrics(trades, pip, initial_balance=10000.0):
    """Calculate comprehensive performance metrics"""
    if not trades:
        return {}
    total = len(trades)
    wins = [t for t in trades if t["result"] == "TP"]
    losses = [t for t in trades if t["result"] == "SL"]
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / total * 100 if total else 0
    gross_profit = sum(t["pips"] for t in wins) * pip * 10
    gross_loss = sum(abs(t["pips"]) for t in losses) * pip * 10
    net_pips = sum(t["pips"] for t in trades)
    net_profit = gross_profit - gross_loss
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    avg_win = (sum(t["pips"] for t in wins) / win_count) if win_count else 0
    avg_loss = (sum(abs(t["pips"]) for t in losses) / loss_count) if loss_count else 0
    expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)
    equity = [initial_balance]
    peak = initial_balance
    max_dd = 0.0
    for t in trades:
        pnl = t["pips"] * pip * 10
        equity.append(equity[-1] + pnl)
        if equity[-1] > peak:
            peak = equity[-1]
        dd = (peak - equity[-1]) / peak * 100
        if dd > max_dd:
            max_dd = dd
    equity_series = pd.Series(equity)
    returns = equity_series.pct_change().dropna()
    sharpe = 0.0
    if len(returns) > 1 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * math.sqrt(252)
    avg_rr = sum(t["rr"] for t in trades) / total
    return {
        "total_signals": total,
        "wins": win_count,
        "losses": loss_count,
        "win_rate_pct": round(win_rate, 1),
        "net_pips": round(net_pips, 1),
        "net_profit": round(net_profit, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "avg_win_pips": round(avg_win, 1),
        "avg_loss_pips": round(avg_loss, 1),
        "expectancy_pips": round(expectancy, 2),
        "avg_rr": round(avg_rr, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 3),
        "final_balance": round(equity[-1], 2),
    }

def run_backtest():
    if not mt5.initialize():
        print("MT5 init failed")
        return
    random.seed(42)
    np.random.seed(42)
    config_path = Path(__file__).parent.parent / "config" / "strategy.json"
    with open(config_path) as f:
        config = json.load(f)
    sym_cfg = config["symbols"][SYMBOL]
    params_cfg = config.get("params", {})
    scfg = ScoringConfig(**config["scoring"])
    profile = dict_to_profile(sym_cfg)
    params = dict_to_params(params_cfg)
    strategy = StrategyEngine(profile=profile, params=params, weights=scfg, min_score=MIN_SCORE)
    pip = pip_size(SYMBOL)
    slippage = SLIPPAGE_PIPS
    r_htf = mt5.copy_rates_from_pos(SYMBOL, HTF_TF, 0, HTF_BARS)
    r_ltf = mt5.copy_rates_from_pos(SYMBOL, LTF_TF, 0, LTF_BARS)
    if r_htf is None or r_ltf is None:
        print("Data fetch failed"); mt5.shutdown(); return
    htf_df = to_df(r_htf)
    ltf_df = to_df(r_ltf)
    logger.info(f"HTF={len(htf_df)} LTF={len(ltf_df)} | {htf_df.iloc[0]['time']} {htf_df.iloc[-1]['time']}")
    spread = SPREAD_PIPS
    sl_min = round(spread * 2 + 0.02, 2)
    trades = []
    sim_start = max(30, int(len(htf_df) * 0.2))
    step = 3
    for i in range(sim_start, len(htf_df), step):
        hw = htf_df.iloc[:i+1].copy()
        t_cutoff = hw.iloc[-1]["time"]
        lw = ltf_df[ltf_df["time"] <= t_cutoff].copy()
        if len(lw) < 30:
            continue
        sig = strategy.evaluate(hw, lw, t_cutoff, news_active=False)
        if sig.direction not in ("BUY", "SELL"):
            continue
        entry_raw = sig.entry_price
        sl_raw = sig.stop_loss
        tp_raw = sig.take_profit
        if sig.direction == "BUY":
            entry_price = entry_raw + spread * 0.5
            sl = sl_raw
            tp = tp_raw
        else:
            entry_price = entry_raw - spread * 0.5
            sl = sl_raw
            tp = tp_raw
        risk = abs(entry_price - sl)
        if risk < sl_min:
            sl = entry_price - sl_min if sig.direction == "BUY" else entry_price + sl_min
            risk = sl_min
        reward = abs(tp - entry_price)
        rr = reward / risk if risk > 0 else 0
        later = ltf_df[ltf_df["time"] > t_cutoff].copy().reset_index(drop=True)
        if len(later) < 2:
            continue
        slippage_random = slippage * (0.5 + random.random())
        res, price, bars = simulate_trade_candle(entry_price, sl, tp, sig.direction, later, slippage_random, pip)
        pips = abs(price - entry_price) / pip
        if res == "SL":
            pips = -pips
        trades.append({
            "time": t_cutoff, "dir": sig.direction, "score": sig.score,
            "entry": entry_price, "sl": sl, "tp": tp, "rr": rr,
            "result": res, "pips": pips, "bars_to_exit": bars,
            "pat": sig.primary_pattern.type.name if sig.primary_pattern else "none",
        })
    metrics = calculate_metrics(trades, pip)
    total = metrics["total_signals"]
    wr = metrics["win_rate_pct"]
    print(f"\n{'='*80}")
    print(f"  BACKTEST: {SYMBOL} M15-M1 | {total} senales (score>={MIN_SCORE})")
    print(f"{'='*80}")
    print(f"  Spread: {SPREAD_PIPS}p | Slippage: {SLIPPAGE_PIPS}p | SL min: {sl_min}p")
    print(f"  Data range: {htf_df.iloc[0]['time']} -> {htf_df.iloc[-1]['time']} ({len(htf_df)} HTF bars)")
    print()
    print(f"  {'Metric':30s} {'Value':>10s}")
    print(f"  {'-'*30} {'-'*10}")
    print(f"  {'Total Signals':30s} {metrics['total_signals']:>10d}")
    print(f"  {'Win/Loss':30s} {metrics['wins']}W / {metrics['losses']}L")
    print(f"  {'Win Rate':30s} {metrics['win_rate_pct']:>9.1f}%")
    print(f"  {'Net Pips':30s} {metrics['net_pips']:>+10.1f}")
    print(f"  {'Gross Profit / Loss':30s} ${metrics['gross_profit']:>+.0f} / ${metrics['gross_loss']:>+.0f}")
    print(f"  {'Net Profit':30s} ${metrics['net_profit']:>+10.2f}")
    print(f"  {'Profit Factor':30s} {metrics['profit_factor']:>10.2f}")
    print(f"  {'Avg Win / Avg Loss (pips)':30s} {metrics['avg_win_pips']:>+10.1f} / {metrics['avg_loss_pips']:>+10.1f}")
    print(f"  {'Expectancy (pips/trade)':30s} {metrics['expectancy_pips']:>+10.2f}")
    print(f"  {'Avg RR Ratio':30s} {metrics['avg_rr']:>10.2f}")
    print(f"  {'Max Drawdown':30s} {metrics['max_drawdown_pct']:>9.2f}%")
    print(f"  {'Sharpe Ratio (annual)':30s} {metrics['sharpe_ratio']:>10.3f}")
    print(f"  {'Final Balance ($10k start)':30s} ${metrics['final_balance']:>+10.2f}")
    print()
    pats = {}
    for t in trades:
        p = t["pat"]
        if p not in pats:
            pats[p] = {"n": 0, "w": 0, "rr": [], "pips": []}
        pats[p]["n"] += 1
        if t["result"] == "TP":
            pats[p]["w"] += 1
        pats[p]["rr"].append(t["rr"])
        pats[p]["pips"].append(t["pips"])
    print(f"  {'Pattern':28s} {'#':4s} {'W%':6s} {'AvgRR':6s} {'NetPips':8s}")
    print(f"  {'-'*28} {'-'*4} {'-'*6} {'-'*6} {'-'*8}")
    for p, d in sorted(pats.items(), key=lambda x: -x[1]["n"])[:10]:
        pw = d["w"] / d["n"] * 100
        pr = sum(d["rr"]) / len(d["rr"])
        pp = sum(d["pips"])
        print(f"  {p:28s} {d['n']:4d} {pw:5.1f}% {pr:5.2f} {pp:+7.1f}")
    if trades:
        print(f"\n  Ultimas {min(5, len(trades))} operaciones:")
        for t in trades[-5:]:
            print(f"    {str(t['time'])[:16]} {t['dir']:4s} sc={t['score']:.0f} "
                  f"rr={t['rr']:.2f} bars={t['bars_to_exit']:2d} -> {t['result']} ({t['pips']:+.1f}p)")
    csv_path = Path(__file__).parent.parent / "data" / f"backtest_{SYMBOL}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    pd.DataFrame(trades).to_csv(csv_path, index=False)
    print(f"\n  CSV export: {csv_path}")
    mt5.shutdown()
    return trades

if __name__ == "__main__":
    if len(sys.argv) > 1:
        SYMBOL = sys.argv[1]
    run_backtest()

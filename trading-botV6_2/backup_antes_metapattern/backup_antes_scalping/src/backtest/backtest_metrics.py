"""Backtest Metrics Calculator
Computes Sharpe, Sortino, drawdown, profit factor, win rate,
and per-segment breakdowns (by pattern, regime, session).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .simulated_executor import SimulatedTrade


@dataclass
class BacktestMetrics:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_risk_reward: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    total_return_pct: float = 0.0
    avg_bars_held: float = 0.0
    expectancy: float = 0.0
    kelly_fraction: float = 0.0
    by_pattern: Dict[str, dict] = field(default_factory=dict)
    by_regime: Dict[str, dict] = field(default_factory=dict)
    by_session: Dict[str, dict] = field(default_factory=dict)
    by_exit_reason: Dict[str, dict] = field(default_factory=dict)
    monthly_return_pct: float = 0.0
    trades_per_day: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    recovery_factor: float = 0.0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


def compute_metrics(
    trades: List[SimulatedTrade],
    equity_curve: List[float],
    initial_balance: float,
    total_bars: int,
) -> BacktestMetrics:
    m = BacktestMetrics()
    if not trades:
        return m

    m.total_trades = len(trades)
    wins = [t for t in trades if t.profit > 0]
    losses = [t for t in trades if t.profit <= 0]
    m.winning_trades = len(wins)
    m.losing_trades = len(losses)
    m.win_rate = m.winning_trades / m.total_trades if m.total_trades > 0 else 0.0

    m.gross_profit = sum(t.profit for t in wins)
    m.gross_loss = abs(sum(t.profit for t in losses))
    m.net_profit = m.gross_profit - m.gross_loss
    m.profit_factor = m.gross_profit / m.gross_loss if m.gross_loss > 0 else 0.0

    m.avg_win = m.gross_profit / m.winning_trades if m.winning_trades > 0 else 0.0
    m.avg_loss = m.gross_loss / m.losing_trades if m.losing_trades > 0 else 0.0
    m.avg_bars_held = np.mean([t.bars_held for t in trades]) if trades else 0.0

    rrs = []
    for t in trades:
        entry, sl = t.entry_price, t.stop_loss
        direction = 1.0 if t.direction == "BUY" else -1.0
        r = abs(t.take_profit - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0
        rrs.append(r)
    m.avg_risk_reward = float(np.mean(rrs)) if rrs else 0.0

    final_balance = equity_curve[-1] if equity_curve else initial_balance
    m.total_return_pct = ((final_balance - initial_balance) / initial_balance) * 100.0

    # Drawdown
    peak = equity_curve[0] if equity_curve else initial_balance
    max_dd = 0.0
    max_dd_pct = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = peak - val
        dd_pct = dd / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
    m.max_drawdown = max_dd
    m.max_drawdown_pct = max_dd_pct

    # Sharpe / Sortino
    if len(equity_curve) > 1:
        returns = np.diff(equity_curve) / equity_curve[:-1]
        if len(returns) > 1 and np.std(returns) > 0:
            m.sharpe_ratio = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
            downside = returns[returns < 0]
            if len(downside) > 0 and np.std(downside) > 0:
                m.sortino_ratio = float(np.mean(returns) / np.std(downside) * np.sqrt(252))

    # Expectancy
    m.expectancy = (m.win_rate * m.avg_win) - ((1 - m.win_rate) * m.avg_loss)

    # Kelly
    if m.avg_loss > 0:
        b = m.avg_win / m.avg_loss if m.avg_win > 0 else 0.0
        p = m.win_rate
        q = 1 - p
        kelly = (p * b - q) / b if b > 0 else 0.0
        m.kelly_fraction = max(0.0, kelly)

    # Recovery factor
    if m.max_drawdown > 0:
        m.recovery_factor = m.net_profit / m.max_drawdown

    # Consecutive wins/losses
    max_cw = 0
    max_cl = 0
    cw = 0
    cl = 0
    for t in trades:
        if t.profit > 0:
            cw += 1
            cl = 0
            max_cw = max(max_cw, cw)
        else:
            cl += 1
            cw = 0
            max_cl = max(max_cl, cl)
    m.consecutive_wins = max_cw
    m.consecutive_losses = max_cl

    # Per-pattern breakdown
    by_pattern: Dict[str, dict] = {}
    for t in trades:
        key = t.pattern_type or "UNKNOWN"
        if key not in by_pattern:
            by_pattern[key] = {"trades": 0, "wins": 0, "losses": 0, "profit": 0.0}
        by_pattern[key]["trades"] += 1
        by_pattern[key]["profit"] += t.profit
        if t.profit > 0:
            by_pattern[key]["wins"] += 1
        else:
            by_pattern[key]["losses"] += 1
    for k, v in by_pattern.items():
        v["win_rate"] = v["wins"] / v["trades"] if v["trades"] > 0 else 0.0
    m.by_pattern = by_pattern

    # Per-regime
    by_regime: Dict[str, dict] = {}
    for t in trades:
        key = t.regime or "UNKNOWN"
        if key not in by_regime:
            by_regime[key] = {"trades": 0, "wins": 0, "losses": 0, "profit": 0.0}
        by_regime[key]["trades"] += 1
        by_regime[key]["profit"] += t.profit
        if t.profit > 0:
            by_regime[key]["wins"] += 1
        else:
            by_regime[key]["losses"] += 1
    for k, v in by_regime.items():
        v["win_rate"] = v["wins"] / v["trades"] if v["trades"] > 0 else 0.0
    m.by_regime = by_regime

    # Per-session
    by_session: Dict[str, dict] = {}
    for t in trades:
        key = t.session or "UNKNOWN"
        if key not in by_session:
            by_session[key] = {"trades": 0, "wins": 0, "losses": 0, "profit": 0.0}
        by_session[key]["trades"] += 1
        by_session[key]["profit"] += t.profit
        if t.profit > 0:
            by_session[key]["wins"] += 1
        else:
            by_session[key]["losses"] += 1
    for k, v in by_session.items():
        v["win_rate"] = v["wins"] / v["trades"] if v["trades"] > 0 else 0.0
    m.by_session = by_session

    # Per exit reason
    by_exit: Dict[str, dict] = {}
    for t in trades:
        key = t.exit_reason or "UNKNOWN"
        if key not in by_exit:
            by_exit[key] = {"trades": 0, "wins": 0, "losses": 0, "profit": 0.0}
        by_exit[key]["trades"] += 1
        by_exit[key]["profit"] += t.profit
        if t.profit > 0:
            by_exit[key]["wins"] += 1
        else:
            by_exit[key]["losses"] += 1
    for k, v in by_exit.items():
        v["win_rate"] = v["wins"] / v["trades"] if v["trades"] > 0 else 0.0
    m.by_exit_reason = by_exit

    return m


def metrics_to_fitness(m: BacktestMetrics, weights: Optional[Dict[str, float]] = None) -> float:
    if weights is None:
        weights = {
            "sharpe": 3.0, "profit_factor": 2.0, "win_rate": 1.5,
            "total_return_pct": 2.0, "max_drawdown_pct": -2.0,
            "trades": 0.5, "avg_risk_reward": 1.0,
        }
    score = 0.0
    score += weights.get("sharpe", 0) * min(m.sharpe_ratio, 5.0)
    score += weights.get("profit_factor", 0) * min(m.profit_factor, 10.0)
    score += weights.get("win_rate", 0) * m.win_rate * 100.0
    score += weights.get("total_return_pct", 0) * min(m.total_return_pct / 10.0, 5.0)
    score += weights.get("max_drawdown_pct", 0) * min(m.max_drawdown_pct * 100.0, 50.0)
    score += weights.get("trades", 0) * min(m.total_trades / 20.0, 5.0)
    score += weights.get("avg_risk_reward", 0) * min(m.avg_risk_reward, 5.0)
    return score

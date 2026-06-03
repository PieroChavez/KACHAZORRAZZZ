"""SMC Backtesting Module (Mejora 9)
Replicates the full _evaluate_symbol() pipeline offline using:
- Real SMC components (RegimeDetector, PatternDetector, StrategyEngine, etc.)
- SimulatedExecutor for order tracking
- BacktestMetrics with Sharpe, drawdown, win rate, profit factor
- Per-pattern / per-regime / per-session breakdowns
"""

from .backtest_engine import BacktestEngine, BacktestResult
from .simulated_executor import SimulatedExecutor, SimulatedTrade
from .backtest_metrics import BacktestMetrics, compute_metrics, metrics_to_fitness

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "SimulatedExecutor",
    "SimulatedTrade",
    "BacktestMetrics",
    "compute_metrics",
    "metrics_to_fitness",
]

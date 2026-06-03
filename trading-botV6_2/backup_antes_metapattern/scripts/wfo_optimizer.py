"""Walk-Forward Optimization Engine: optimiza thresholds del bot usando
datos históricos de trades registrados por FailureAnalyzer y MetaLearner.
NO modifica el runtime del bot. Solo produce recomendaciones."""
import sqlite3
import json
import itertools
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class TradeSample:
    profit: float
    score: float
    conviction: float
    regime: str
    direction: str
    exit_reason: str
    failure_category: str
    primary_pattern: Optional[str]
    timestamp: float


@dataclass
class ParamSet:
    min_reversal_strong: float = 80.0
    min_reversal_loss: float = 50.0
    be_threshold: float = 0.20
    trail_stage1_mult: float = 3.0
    trail_stage2_mult: float = 2.5
    trail_stage3_mult: float = 2.0
    trail_stage4_mult: float = 1.5


@dataclass
class WFOResult:
    params: ParamSet
    in_sample_pf: float
    out_sample_pf: float
    in_sample_winrate: float
    out_sample_winrate: float
    in_sample_trades: int
    out_sample_trades: int
    in_sample_avg_profit: float
    out_sample_avg_profit: float


class WFOptimizer:
    """Optimiza thresholds mediante Walk-Forward Analysis.
    Grid search sobre parámetros → valida en datos fuera de muestra."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self):
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row

    def load_trades(self, min_samples: int = 20) -> List[TradeSample]:
        if self._conn is None:
            return []
        cur = self._conn.execute("""
            SELECT profit, score, conviction, regime, direction, exit_reason,
                   failure_category, primary_pattern, timestamp
            FROM trade_analysis ORDER BY timestamp ASC
        """)
        trades = []
        for r in cur.fetchall():
            trades.append(TradeSample(
                profit=r["profit"], score=r["score"],
                conviction=r["conviction"], regime=r["regime"],
                direction=r["direction"], exit_reason=r["exit_reason"],
                failure_category=r["failure_category"],
                primary_pattern=r["primary_pattern"],
                timestamp=r["timestamp"],
            ))
        return trades

    def evaluate_params(self, params: ParamSet, trades: List[TradeSample]) -> dict:
        """Simula cuántas pérdidas se habrían evitado con estos thresholds."""
        reversals_attempted = 0
        reversals_won = 0
        reversals_lost = 0
        total_profit = 0.0

        for t in trades:
            if t.exit_reason != "reversal":
                continue
            reversals_attempted += 1
            rev_threshold = params.min_reversal_strong if t.profit >= 0 else params.min_reversal_loss
            if t.score >= rev_threshold:
                total_profit += t.profit
                if t.profit >= 0:
                    reversals_won += 1
                else:
                    reversals_lost += 1

        pf = (reversals_won / max(reversals_lost, 1)) * (abs(total_profit) / max(abs(total_profit), 1))
        winrate = reversals_won / max(reversals_attempted, 1)
        avg_profit = total_profit / max(reversals_attempted, 1)

        return {
            "profit_factor": pf if pf < 100 else 99.9,
            "winrate": winrate,
            "total_trades": reversals_attempted,
            "avg_profit": avg_profit,
            "total_profit": total_profit,
        }

    def walk_forward(self, trades: List[TradeSample],
                     train_ratio: float = 0.7) -> List[WFOResult]:
        if len(trades) < 30:
            logger.warning(f"Only {len(trades)} trades, need >= 30 for WFO")
            return []

        split = int(len(trades) * train_ratio)
        train = trades[:split]
        test = trades[split:]

        param_grid = self._build_grid()
        results = []

        for p in param_grid:
            train_metrics = self.evaluate_params(p, train)
            test_metrics = self.evaluate_params(p, test)
            results.append(WFOResult(
                params=p,
                in_sample_pf=train_metrics["profit_factor"],
                out_sample_pf=test_metrics["profit_factor"],
                in_sample_winrate=train_metrics["winrate"],
                out_sample_winrate=test_metrics["winrate"],
                in_sample_trades=train_metrics["total_trades"],
                out_sample_trades=test_metrics["total_trades"],
                in_sample_avg_profit=train_metrics["avg_profit"],
                out_sample_avg_profit=test_metrics["avg_profit"],
            ))

        return results

    def _build_grid(self) -> List[ParamSet]:
        grids = {
            "min_reversal_strong": [70, 75, 80, 85, 90],
            "min_reversal_loss": [40, 45, 50, 55, 60],
            "be_threshold": [0.16, 0.18, 0.20, 0.22, 0.25],
        }
        params_list = []
        for rev_s, rev_l, be in itertools.product(
            grids["min_reversal_strong"], grids["min_reversal_loss"], grids["be_threshold"]
        ):
            params_list.append(ParamSet(
                min_reversal_strong=rev_s, min_reversal_loss=rev_l, be_threshold=be,
            ))
        return params_list

    def rank_results(self, results: List[WFOResult], top_n: int = 5) -> List[WFOResult]:
        scored = []
        for r in results:
            if r.out_sample_trades < 3:
                continue
            score = r.out_sample_pf * (1 + r.out_sample_winrate) * max(0, r.out_sample_avg_profit + 1)
            if r.in_sample_pf < 1.0 or r.out_sample_pf < 0.5:
                score *= 0.5
            scored.append((score, r))

        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored[:top_n]]

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None


def main():
    project_root = Path(__file__).parent.parent
    db_path = project_root / "data" / "failure_analysis.db"
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}. Run the bot first to collect trades.")
        return

    optimizer = WFOptimizer(db_path)
    optimizer.connect()
    trades = optimizer.load_trades()
    logger.info(f"Loaded {len(trades)} trades from {db_path}")

    if len(trades) < 30:
        logger.warning(f"Only {len(trades)} trades. Need >= 30 for meaningful WFO. "
                       f"Collect more trades by running the bot.")
        optimizer.close()
        return

    results = optimizer.walk_forward(trades, train_ratio=0.7)
    logger.info(f"Evaluated {len(results)} parameter combinations")

    top = optimizer.rank_results(results, top_n=5)
    if not top:
        logger.info("No valid results found")
        optimizer.close()
        return

    print("\n" + "=" * 70)
    print("WALK-FORWARD OPTIMIZATION RESULTS (Top 5)")
    print("=" * 70)
    print(f"{'Rank':<6} {'Rev Strong':<12} {'Rev Loss':<10} {'BE%':<8} "
          f"{'IS PF':<8} {'OOS PF':<8} {'IS WR%':<8} {'OOS WR%':<8} {'OOS Trades':<10}")
    print("-" * 70)
    for i, r in enumerate(top, 1):
        print(f"{i:<6} {r.params.min_reversal_strong:<12.0f} {r.params.min_reversal_loss:<10.0f} "
              f"{r.params.be_threshold*100:<8.0f} {r.in_sample_pf:<8.2f} {r.out_sample_pf:<8.2f} "
              f"{r.in_sample_winrate*100:<8.0f} {r.out_sample_winrate*100:<8.0f} {r.out_sample_trades:<10}")
    print("-" * 70)

    best = top[0]
    print(f"\nRecommended thresholds:")
    print(f"  min_reversal_score (strong): {best.params.min_reversal_strong:.0f}")
    print(f"  min_reversal_score (loss):   {best.params.min_reversal_loss:.0f}")
    print(f"  BE activation threshold:      {best.params.be_threshold*100:.0f}%")
    print(f"\nOut-of-sample performance:")
    print(f"  Profit Factor: {best.out_sample_pf:.2f}")
    print(f"  Win Rate:      {best.out_sample_winrate*100:.1f}%")
    print(f"  # Trades:      {best.out_sample_trades}")
    print(f"  Avg Profit:    ${best.out_sample_avg_profit:.2f}")

    params_path = project_root / "data" / "wfo_recommended_params.json"
    with open(params_path, "w") as f:
        json.dump({
            "min_reversal_strong": best.params.min_reversal_strong,
            "min_reversal_loss": best.params.min_reversal_loss,
            "be_threshold": best.params.be_threshold,
            "in_sample_pf": best.in_sample_pf,
            "out_sample_pf": best.out_sample_pf,
            "oos_trades": best.out_sample_trades,
            "timestamp": time.time(),
        }, f, indent=2)
    logger.info(f"Recommended params saved to {params_path}")

    optimizer.close()


if __name__ == "__main__":
    main()

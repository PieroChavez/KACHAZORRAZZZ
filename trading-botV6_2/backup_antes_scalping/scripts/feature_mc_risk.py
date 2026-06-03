"""Feature Importance + Monte Carlo Risk Simulation.
Lee historical trades desde failure_analysis.db, analiza qué features
correlacionan con resultados, y simula riesgo mediante bootstrap.
Sin modificar runtime del bot."""
import sqlite3
import json
import time
import random
import math
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from collections import defaultdict
from dataclasses import dataclass
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


class FeatureAnalyzer:
    """Correlaciona features con outcomes de trades."""

    def __init__(self, trades: List[TradeSample]):
        self.trades = trades

    @staticmethod
    def from_db(db_path: Path):
        if not db_path.exists():
            logger.error(f"DB not found: {db_path}")
            return FeatureAnalyzer([])
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        trades = []
        try:
            cur = conn.execute("SELECT profit, score, conviction, regime, direction, "
                               "exit_reason, failure_category, primary_pattern "
                               "FROM trade_analysis")
            for r in cur.fetchall():
                trades.append(TradeSample(
                    profit=r["profit"], score=r["score"],
                    conviction=r["conviction"], regime=r["regime"],
                    direction=r["direction"], exit_reason=r["exit_reason"],
                    failure_category=r["failure_category"],
                    primary_pattern=r["primary_pattern"],
                ))
        except sqlite3.OperationalError as e:
            logger.error(f"DB error: {e}")
        finally:
            conn.close()
        return FeatureAnalyzer(trades)

    def pattern_importance(self) -> List[dict]:
        """Winrate y profit por patrón. Ranking de importancia."""
        grouped = defaultdict(list)
        for t in self.trades:
            grouped[t.primary_pattern or "unknown"].append(t)
        rows = []
        for pattern, grp in grouped.items():
            wins = sum(1 for t in grp if t.profit > 0)
            pnl = sum(t.profit for t in grp)
            rows.append({
                "feature": "pattern",
                "value": pattern,
                "trades": len(grp),
                "winrate": round(wins / len(grp), 3),
                "pnl": round(pnl, 2),
                "impact": round(pnl / max(len(grp), 1), 2),
            })
        rows.sort(key=lambda x: -abs(x["impact"]))
        return rows

    def regime_importance(self) -> List[dict]:
        grouped = defaultdict(list)
        for t in self.trades:
            grouped[t.regime or "unknown"].append(t)
        rows = []
        for regime, grp in grouped.items():
            wins = sum(1 for t in grp if t.profit > 0)
            pnl = sum(t.profit for t in grp)
            rows.append({
                "feature": "regime",
                "value": regime,
                "trades": len(grp),
                "winrate": round(wins / len(grp), 3),
                "pnl": round(pnl, 2),
                "impact": round(pnl / max(len(grp), 1), 2),
            })
        rows.sort(key=lambda x: -abs(x["impact"]))
        return rows

    def direction_importance(self) -> List[dict]:
        grouped = defaultdict(list)
        for t in self.trades:
            grouped[t.direction or "unknown"].append(t)
        rows = []
        for direction, grp in grouped.items():
            wins = sum(1 for t in grp if t.profit > 0)
            pnl = sum(t.profit for t in grp)
            rows.append({
                "feature": "direction",
                "value": direction,
                "trades": len(grp),
                "winrate": round(wins / len(grp), 3),
                "pnl": round(pnl, 2),
                "impact": round(pnl / max(len(grp), 1), 2),
            })
        rows.sort(key=lambda x: -abs(x["impact"]))
        return rows

    def exit_reason_importance(self) -> List[dict]:
        """Analiza rentabilidad por motivo de salida."""
        grouped = defaultdict(list)
        for t in self.trades:
            grouped[t.exit_reason or "unknown"].append(t)
        rows = []
        for reason, grp in grouped.items():
            wins = sum(1 for t in grp if t.profit > 0)
            pnl = sum(t.profit for t in grp)
            rows.append({
                "feature": "exit_reason",
                "value": reason,
                "trades": len(grp),
                "winrate": round(wins / len(grp), 3),
                "pnl": round(pnl, 2),
                "impact": round(pnl / max(len(grp), 1), 2),
            })
        rows.sort(key=lambda x: -abs(x["impact"]))
        return rows

    def best_worst_patterns(self, top_n: int = 5) -> Tuple[List[dict], List[dict]]:
        """Top N mejores y peores patrones por winrate (min 3 trades)."""
        rows = self.pattern_importance()
        qualified = [r for r in rows if r["trades"] >= 3]
        qualified.sort(key=lambda x: -x["winrate"])
        best = qualified[:top_n]
        qualified.sort(key=lambda x: x["winrate"])
        worst = qualified[:top_n]
        return best, worst

    def export_all(self) -> dict:
        return {
            "timestamp": time.time(),
            "total_trades": len(self.trades),
            "pattern_importance": self.pattern_importance(),
            "regime_importance": self.regime_importance(),
            "direction_importance": self.direction_importance(),
            "exit_reason_importance": self.exit_reason_importance(),
        }


class MonteCarloSim:
    """Simula N caminos aleatorios a partir de trades históricos.
    Reporta VaR, CVaR, max drawdown, probabilidad de ruina."""

    def __init__(self, trades: List[TradeSample], initial_capital: float = 10000.0):
        self.profits = [t.profit for t in trades] if trades else [0.0]
        self.wins = [p for p in self.profits if p > 0]
        self.losses = [p for p in self.profits if p <= 0]
        self.n = len(trades)
        self.capital = initial_capital

    @staticmethod
    def from_db(db_path: Path, capital: float = 10000.0):
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        profits = []
        try:
            cur = conn.execute("SELECT profit FROM trade_analysis")
            for r in cur.fetchall():
                profits.append(r["profit"])
        except sqlite3.OperationalError:
            pass
        finally:
            conn.close()
        trades = [TradeSample(profit=p, score=0, conviction=0, regime="", direction="",
                              exit_reason="", failure_category="", primary_pattern=None)
                  for p in profits]
        return MonteCarloSim(trades, capital)

    def run(self, n_simulations: int = 10000,
            trades_per_sim: Optional[int] = None) -> dict:
        if not self.profits:
            return {"error": "no profit data"}
        if trades_per_sim is None:
            trades_per_sim = len(self.profits)

        final_equities = []
        peak_equities = []
        max_dds = []
        ruin_count = 0

        for _ in range(n_simulations):
            equity = self.capital
            peak = self.capital
            max_dd = 0.0
            for _ in range(trades_per_sim):
                p = random.choice(self.profits)
                equity += p
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
                if equity <= 0:
                    ruin_count += 1
                    break
            final_equities.append(equity)
            peak_equities.append(peak)
            max_dds.append(max_dd)

        final_equities.sort()
        max_dds.sort()

        var_95 = final_equities[int(len(final_equities) * 0.05)]
        var_99 = final_equities[int(len(final_equities) * 0.01)]
        cvar_95 = sum(final_equities[:int(len(final_equities) * 0.05)]) / max(int(len(final_equities) * 0.05), 1)
        median_final = final_equities[len(final_equities) // 2]
        mean_final = sum(final_equities) / len(final_equities)
        dd_95 = max_dds[int(len(max_dds) * 0.95)]
        dd_99 = max_dds[int(len(max_dds) * 0.99)]
        expected_dd = sum(max_dds) / len(max_dds)

        return {
            "n_simulations": n_simulations,
            "trades_per_sim": trades_per_sim,
            "initial_capital": self.capital,
            "historical_trades": self.n,
            "historical_winrate": len(self.wins) / max(self.n, 1),
            "historical_avg_profit": round(sum(self.profits) / max(self.n, 1), 2),
            "var_95": round(var_95, 2),
            "var_99": round(var_99, 2),
            "cvar_95": round(cvar_95, 2),
            "median_final_equity": round(median_final, 2),
            "mean_final_equity": round(mean_final, 2),
            "expected_max_dd_pct": round(expected_dd * 100, 2),
            "dd_95_pct": round(dd_95 * 100, 2),
            "dd_99_pct": round(dd_99 * 100, 2),
            "ruin_probability": round(ruin_count / n_simulations, 4),
        }


def main():
    project_root = Path(__file__).parent.parent
    db_path = project_root / "data" / "failure_analysis.db"

    # --- Feature Importance ---
    fa = FeatureAnalyzer.from_db(db_path)
    report = fa.export_all()

    out_path = project_root / "data" / "feature_importance.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Feature importance saved to {out_path}")

    if report["total_trades"] > 0:
        print(f"\n{'='*60}")
        print(f"FEATURE IMPORTANCE ({report['total_trades']} trades)")
        print(f"{'='*60}")

        for feature_type in ["pattern_importance", "regime_importance",
                             "direction_importance", "exit_reason_importance"]:
            rows = report.get(feature_type, [])
            if not rows:
                continue
            print(f"\n--- By {feature_type.replace('_importance', '')} ---")
            print(f"{'Value':<25} {'Trades':<8} {'WR%':<8} {'PnL':<12} {'Impact':<10}")
            print("-" * 63)
            for r in rows[:8]:
                val = r["value"][:22] + ".." if len(r["value"]) > 24 else r["value"]
                print(f"{val:<25} {r['trades']:<8} {r['winrate']*100:<8.1f} "
                      f"${r['pnl']:<10.2f} ${r['impact']:<8.2f}")

        best, worst = fa.best_worst_patterns()
        print(f"\n--- BEST 5 PATTERNS (by winrate, >= 3 trades) ---")
        for b in best:
            print(f"  {b['value']:<25} WR: {b['winrate']*100:.1f}%  {b['trades']} trades  PnL: ${b['pnl']:.2f}")
        print(f"\n--- WORST 5 PATTERNS (by winrate, >= 3 trades) ---")
        for w in worst:
            print(f"  {w['value']:<25} WR: {w['winrate']*100:.1f}%  {w['trades']} trades  PnL: ${w['pnl']:.2f}")

    # --- Monte Carlo Risk ---
    mc = MonteCarloSim.from_db(db_path, capital=10000.0)
    risk = mc.run(n_simulations=10000)

    risk_path = project_root / "data" / "monte_carlo_risk.json"
    with open(risk_path, "w") as f:
        json.dump(risk, f, indent=2)
    logger.info(f"Monte Carlo risk saved to {risk_path}")

    if "error" not in risk:
        print(f"\n{'='*60}")
        print(f"MONTE CARLO RISK (10,000 simulations)")
        print(f"{'='*60}")
        print(f"Historical trades:  {risk['historical_trades']}")
        print(f"Historical WR:      {risk['historical_winrate']*100:.1f}%")
        print(f"Avg trade profit:   ${risk['historical_avg_profit']:.2f}")
        print(f"Initial capital:    ${risk['initial_capital']:,.2f}")
        print(f"\n=== Value at Risk ===")
        print(f"VaR 95%:            ${risk['var_95']:,.2f}")
        print(f"VaR 99%:            ${risk['var_99']:,.2f}")
        print(f"CVaR 95%:           ${risk['cvar_95']:,.2f}")
        print(f"\n=== Equity Projection ===")
        print(f"Median final:       ${risk['median_final_equity']:,.2f}")
        print(f"Mean final:         ${risk['mean_final_equity']:,.2f}")
        print(f"\n=== Drawdown ===")
        print(f"Expected max DD:    {risk['expected_max_dd_pct']:.2f}%")
        print(f"DD 95% (worst 5%):  {risk['dd_95_pct']:.2f}%")
        print(f"DD 99%:             {risk['dd_99_pct']:.2f}%")
        print(f"\nRuin probability:   {risk['ruin_probability']*100:.2f}%")
        print(f"\nFull reports: {out_path}, {risk_path}")
    else:
        logger.warning("Monte Carlo: no trade data available")


if __name__ == "__main__":
    main()

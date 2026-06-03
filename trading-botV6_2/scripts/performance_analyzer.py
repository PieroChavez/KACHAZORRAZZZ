"""Trade Performance Analyzer: lee datos históricos de failure_analysis.db
y meta_learner.db, produce reportes de performance por feature (patrón,
régimen, sesión, score bin). No modifica el runtime del bot."""
import sqlite3
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict
from loguru import logger


@dataclass
class TradeRecord:
    profit: float
    score: float
    conviction: float
    regime: str
    direction: str
    exit_reason: str
    failure_category: str
    primary_pattern: Optional[str]
    session: Optional[str]


@dataclass
class SignalRecord:
    score: float
    conviction: float
    regime: str
    direction: str
    session: str
    taken: int
    skipped_reason: Optional[str]


class PerformanceAnalyzer:
    """Carga trades desde ambas DBs y produce reportes agregados."""

    def __init__(self, failures_path: Path, meta_path: Optional[Path] = None):
        self.failures_path = failures_path
        self.meta_path = meta_path
        self.trades: List[TradeRecord] = []
        self.signals: List[SignalRecord] = []

    def load_all(self):
        self._load_failure_trades()
        self._load_meta_signals()
        logger.info(f"Loaded {len(self.trades)} trades, {len(self.signals)} signals")

    def _load_failure_trades(self):
        if not self.failures_path.exists():
            logger.warning(f"DB not found: {self.failures_path}")
            return
        conn = sqlite3.connect(str(self.failures_path))
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute("SELECT profit, score, conviction, regime, direction, "
                               "exit_reason, failure_category, primary_pattern "
                               "FROM trade_analysis")
            for r in cur.fetchall():
                self.trades.append(TradeRecord(
                    profit=r["profit"], score=r["score"],
                    conviction=r["conviction"], regime=r["regime"],
                    direction=r["direction"], exit_reason=r["exit_reason"],
                    failure_category=r["failure_category"],
                    primary_pattern=r["primary_pattern"],
                    session=None,
                ))
        except sqlite3.OperationalError as e:
            logger.error(f"Error reading {self.failures_path}: {e}")
        finally:
            conn.close()

    def _load_meta_signals(self):
        if not self.meta_path or not self.meta_path.exists():
            logger.info(f"No meta learner DB at {self.meta_path}, skipping signals")
            return
        conn = sqlite3.connect(str(self.meta_path))
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute("SELECT score, conviction, regime, direction, "
                               "session, reason FROM skipped_signals")
            for r in cur.fetchall():
                self.signals.append(SignalRecord(
                    score=r["score"], conviction=r["conviction"],
                    regime=r["regime"], direction=r["direction"],
                    session=r["session"], taken=0,
                    skipped_reason=r["reason"],
                ))
        except sqlite3.OperationalError as e:
            logger.error(f"Error reading {self.meta_path}: {e}")
        finally:
            conn.close()

    def report_trade_performance(self) -> dict:
        """Agrega métricas globales de trades."""
        t = self.trades
        if not t:
            return {"error": "no trades"}
        wins = [x for x in t if x.profit > 0]
        losses = [x for x in t if x.profit <= 0]
        total_pnl = sum(x.profit for x in t)
        return {
            "total_trades": len(t),
            "wins": len(wins),
            "losses": len(losses),
            "winrate": len(wins) / max(len(t), 1),
            "total_pnl": round(total_pnl, 2),
            "avg_win": round(sum(x.profit for x in wins) / max(len(wins), 1), 2),
            "avg_loss": round(sum(x.profit for x in losses) / max(len(losses), 1), 2),
            "profit_factor": round(
                abs(sum(x.profit for x in wins)) / max(abs(sum(x.profit for x in losses)), 1), 2
            ),
            "max_drawdown_pct": round(
                self._compute_max_drawdown(t) * 100, 2
            ),
        }

    def report_by_pattern(self) -> List[dict]:
        """Winrate y profit por patrón primario."""
        grouped = defaultdict(list)
        for t in self.trades:
            key = t.primary_pattern or "unknown"
            grouped[key].append(t)
        rows = []
        for pattern, trades in sorted(grouped.items(), key=lambda x: -len(x[1])):
            wins = sum(1 for t in trades if t.profit > 0)
            pnl = sum(t.profit for t in trades)
            rows.append({
                "pattern": pattern,
                "trades": len(trades),
                "winrate": round(wins / len(trades), 3),
                "pnl": round(pnl, 2),
                "avg_profit": round(pnl / len(trades), 2),
            })
        return rows

    def report_by_regime(self) -> List[dict]:
        """Winrate y profit por régimen."""
        grouped = defaultdict(list)
        for t in self.trades:
            key = t.regime or "unknown"
            grouped[key].append(t)
        rows = []
        for regime, trades in sorted(grouped.items(), key=lambda x: -len(x[1])):
            wins = sum(1 for t in trades if t.profit > 0)
            pnl = sum(t.profit for t in trades)
            rows.append({
                "regime": regime,
                "trades": len(trades),
                "winrate": round(wins / len(trades), 3),
                "pnl": round(pnl, 2),
                "avg_profit": round(pnl / len(trades), 2),
            })
        return rows

    def report_by_score_bin(self, bin_size: int = 5) -> List[dict]:
        """Winrate por bin de score."""
        grouped = defaultdict(list)
        for t in self.trades:
            key = int(t.score // bin_size * bin_size)
            grouped[key].append(t)
        rows = []
        for score_bin in sorted(grouped):
            trades = grouped[score_bin]
            wins = sum(1 for t in trades if t.profit > 0)
            pnl = sum(t.profit for t in trades)
            rows.append({
                "score_bin": f"{score_bin}-{score_bin + bin_size}",
                "trades": len(trades),
                "winrate": round(wins / len(trades), 3),
                "pnl": round(pnl, 2),
            })
        return rows

    def report_signal_quality(self) -> dict:
        """Calidad de señales vs trades tomados."""
        s = self.signals
        if not s:
            return {"error": "no signals"}
        taken = [x for x in s if x.taken]
        skipped = [x for x in s if not x.taken]
        return {
            "total_signals": len(s),
            "taken": len(taken),
            "skipped": len(skipped),
            "taken_pct": round(len(taken) / max(len(s), 1) * 100, 1),
            "top_skip_reasons": self._top_skip_reasons(skipped),
        }

    def _top_skip_reasons(self, skipped: List[SignalRecord], top_n: int = 5) -> List[dict]:
        reasons = defaultdict(int)
        for s in skipped:
            reasons[s.skipped_reason or "unknown"] += 1
        sorted_r = sorted(reasons.items(), key=lambda x: -x[1])
        return [{"reason": r, "count": c} for r, c in sorted_r[:top_n]]

    def _compute_max_drawdown(self, trades: List[TradeRecord]) -> float:
        equity = 0.0
        peak = 0.0
        mdd = 0.0
        for t in trades:
            equity += t.profit
            if equity > peak:
                peak = equity
            dd = (peak - equity) / max(peak, 1)
            if dd > mdd:
                mdd = dd
        return mdd

    def export_json(self, path: Path) -> dict:
        """Exporta todos los reportes a JSON."""
        report = {
            "timestamp": time.time(),
            "global": self.report_trade_performance(),
            "by_pattern": self.report_by_pattern(),
            "by_regime": self.report_by_regime(),
            "by_score_bin": self.report_by_score_bin(),
            "signal_quality": self.report_signal_quality(),
        }
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Performance report saved to {path}")
        return report


def main():
    project_root = Path(__file__).parent.parent
    failures_db = project_root / "data" / "failure_analysis.db"
    meta_db = project_root / "data" / "meta_learner.db"

    analyzer = PerformanceAnalyzer(failures_db, meta_db)
    analyzer.load_all()

    out_path = project_root / "data" / "performance_report.json"
    report = analyzer.export_json(out_path)

    g = report.get("global", {})
    if "error" not in g:
        print(f"\n{'='*60}")
        print(f"PERFORMANCE REPORT")
        print(f"{'='*60}")
        print(f"Trades: {g['total_trades']}  Wins: {g['wins']}  Losses: {g['losses']}")
        print(f"Winrate: {g['winrate']*100:.1f}%  Profit Factor: {g['profit_factor']}")
        print(f"Total PnL: ${g['total_pnl']}  Avg Win: ${g['avg_win']}  Avg Loss: ${g['avg_loss']}")
        print(f"Max DD: {g['max_drawdown_pct']}%")
        print()

        patterns = report.get("by_pattern", [])
        if patterns:
            print(f"{'Pattern':<20} {'Trades':<8} {'WR%':<8} {'PnL':<12} {'Avg':<10}")
            print("-" * 58)
            for p in patterns[:10]:
                print(f"{p['pattern']:<20} {p['trades']:<8} {p['winrate']*100:<8.1f} "
                      f"${p['pnl']:<10.2f} ${p['avg_profit']:<8.2f}")
            print()

        sig = report.get("signal_quality", {})
        if "error" not in sig:
            print(f"Signals: {sig['total_signals']}  Taken: {sig['taken']} ({sig['taken_pct']}%)")
            if sig.get("top_skip_reasons"):
                print("Top skip reasons:")
                for r in sig["top_skip_reasons"][:3]:
                    print(f"  - {r['reason']}: {r['count']}")

    print(f"\nFull report: {out_path}")


if __name__ == "__main__":
    main()

"""Portfolio Risk Manager Multisímbolo — MEJORA 10 (Modo Experto)
Gestiona exposición consolidada, matriz de correlación, asignación de capital
entre los N mejores setups, y límite de drawdown a nivel portfolio.

Evita sobreexposición cuando múltiples símbolos correlacionados dan señal
simultánea.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

logger = logging.getLogger(__name__)


@dataclass
class PortfolioExposure:
    total_long_notional: float = 0.0
    total_short_notional: float = 0.0
    net_notional: float = 0.0
    gross_notional: float = 0.0
    long_count: int = 0
    short_count: int = 0
    total_positions: int = 0
    directional_ratio: float = 0.0
    margin_used: float = 0.0
    margin_available: float = 0.0


@dataclass
class RankedSignal:
    symbol: str
    direction: str
    score: float
    conviction: float
    regime: str
    pattern_type: str
    risk_pct: float
    rank: int = 0
    allocated: bool = False


@dataclass
class AllocationResult:
    selected: List[RankedSignal]
    skipped: List[RankedSignal]
    total_risk_budget: float
    used_risk_budget: float
    notes: List[str] = field(default_factory=list)


class PortfolioRiskManager:
    """Portfolio-level risk management: consolidated exposure, correlation matrix,
    capital allocation, and portfolio drawdown limit.
    """

    def __init__(
        self,
        correlation_engine=None,
        max_portfolio_risk_pct: float = 0.06,
        max_correlation_for_full_size: float = 0.7,
        max_net_directional_exposure_pct: float = 0.10,
        max_gross_exposure_pct: float = 0.20,
        portfolio_drawdown_limit_pct: float = 0.15,
        max_concurrent_setups: int = 3,
        balance: float = 10_000.0,
    ):
        self.correlation_engine = correlation_engine
        self.max_portfolio_risk_pct = max_portfolio_risk_pct
        self.max_correlation_for_full_size = max_correlation_for_full_size
        self.max_net_directional_exposure_pct = max_net_directional_exposure_pct
        self.max_gross_exposure_pct = max_gross_exposure_pct
        self.portfolio_drawdown_limit_pct = portfolio_drawdown_limit_pct
        self.max_concurrent_setups = max_concurrent_setups
        self.balance = balance
        self.peak_balance = balance

        self._open_positions: Dict[str, dict] = {}
        self._pending_signals: List[RankedSignal] = []
        self._daily_net_pnl: float = 0.0
        self._consecutive_portfolio_losses: int = 0
        self._allocation_history: List[Dict] = []
        self._correlation_matrix_cache: Optional[Dict] = None
        self._last_cache_update: float = 0.0

    def update_balance(self, new_balance: float):
        """Track peak balance for portfolio drawdown computation."""
        self.balance = new_balance
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance

    def update_positions(self, positions: Dict[str, dict]):
        """Update open positions dict from MT5/executor.

        Args:
            positions: dict mapping symbol -> {direction, volume, entry, sl, tp, profit}
        """
        self._open_positions = dict(positions)
        self._correlation_matrix_cache = None

    def compute_exposure(self) -> PortfolioExposure:
        """Compute consolidated portfolio exposure."""
        exp = PortfolioExposure()
        for sym, pos in self._open_positions.items():
            volume = pos.get("volume", 0)
            entry = pos.get("entry", 0)
            notional = volume * entry * self._get_contract_size(sym)
            direction = pos.get("direction", "BUY")
            if direction.upper() == "BUY":
                exp.total_long_notional += notional
                exp.long_count += 1
            else:
                exp.total_short_notional += notional
                exp.short_count += 1

        exp.gross_notional = exp.total_long_notional + exp.total_short_notional
        exp.net_notional = exp.total_long_notional - exp.total_short_notional
        exp.total_positions = exp.long_count + exp.short_count

        divisor = max(exp.total_long_notional, exp.total_short_notional, 1.0)
        exp.directional_ratio = (
            (exp.total_long_notional - exp.total_short_notional) / divisor
        )
        return exp

    def get_portfolio_drawdown_pct(self) -> float:
        """Portfolio-level drawdown as fraction of peak balance."""
        if self.peak_balance <= 0:
            return 0.0
        return (self.peak_balance - self.balance) / self.peak_balance

    def check_portfolio_drawdown(self) -> Tuple[bool, str]:
        """Check if portfolio drawdown exceeds the limit.

        Returns:
            (can_trade, reason)
        """
        dd = self.get_portfolio_drawdown_pct()
        if dd >= self.portfolio_drawdown_limit_pct:
            return (
                False,
                f"Portfolio drawdown {dd:.1%} >= limit {self.portfolio_drawdown_limit_pct:.1%}",
            )
        return True, ""

    def get_correlation_penalty(
        self, symbol: str, direction: str, all_symbols: List[str]
    ) -> Tuple[float, str]:
        """Compute volume penalty based on correlation with existing positions.

        Returns:
            (multiplier, reason)
        """
        if not self._open_positions or not self.correlation_engine:
            return 1.0, "no_other_positions"

        net_side = 1.0 if direction.upper() == "BUY" else -1.0
        weighted_sum = 0.0
        total_volume = 0.0
        notes = []

        for other_sym, pos in self._open_positions.items():
            if other_sym == symbol:
                continue
            other_dir = 1.0 if pos.get("direction", "BUY").upper() == "BUY" else -1.0
            other_vol = pos.get("volume", 0)
            if other_vol <= 0:
                continue

            try:
                corr_data = self.correlation_engine.correlation(
                    symbol, other_sym, lookback="medium"
                )
                corr = corr_data.get("correlation", 0.0)
            except Exception:
                corr = 0.0

            alignment = net_side * other_dir
            if alignment > 0 and corr > self.max_correlation_for_full_size:
                penalty = max(0.2, 1.0 - abs(corr))
                weighted_sum += other_vol * penalty
                notes.append(
                    f"{other_sym} corr={corr:.2f} aligned → ×{penalty:.2f}"
                )
            elif alignment < 0 and abs(corr) > self.max_correlation_for_full_size:
                penalty = max(0.3, 1.0 - abs(corr) * 0.5)
                weighted_sum += other_vol * penalty
                notes.append(
                    f"{other_sym} corr={corr:.2f} opposite → ×{penalty:.2f}"
                )
            else:
                weighted_sum += other_vol
            total_volume += other_vol

        if total_volume <= 0:
            return 1.0, "no_volume"

        avg_penalty = weighted_sum / total_volume
        avg_penalty = max(0.2, min(1.0, avg_penalty))
        return avg_penalty, "; ".join(notes) if notes else "no_corr_penalty"

    def get_net_directional_penalty(
        self, symbol: str, direction: str
    ) -> Tuple[float, str]:
        """Penalize if adding this trade would exceed net directional exposure limit."""
        exp = self.compute_exposure()
        side = 1.0 if direction.upper() == "BUY" else -1.0

        current_net = exp.net_notional
        hypothetical_notional = self._estimate_trade_notional(symbol)
        new_net = current_net + side * hypothetical_notional

        max_net = self.balance * self.max_net_directional_exposure_pct
        max_gross = self.balance * self.max_gross_exposure_pct

        notes = []
        penalty = 1.0

        if abs(new_net) > max_net and abs(current_net) > 0:
            overage = abs(new_net) - max_net
            penalty = max(0.2, 1.0 - overage / max_net)
            notes.append(
                f"net directional {new_net:.0f} > {max_net:.0f} → ×{penalty:.2f}"
            )

        gross = exp.gross_notional + hypothetical_notional
        if gross > max_gross:
            gross_penalty = max(0.3, 1.0 - (gross - max_gross) / max_gross)
            penalty = min(penalty, gross_penalty)
            notes.append(
                f"gross exposure {gross:.0f} > {max_gross:.0f} → ×{gross_penalty:.2f}"
            )

        return penalty, "; ".join(notes) if notes else "within_limits"

    def rank_and_allocate(
        self,
        signals: List[RankedSignal],
        max_setups: Optional[int] = None,
        risk_budget_pct: Optional[float] = None,
    ) -> AllocationResult:
        """Rank all candidate signals and allocate risk budget to the best K.

        Args:
            signals: list of candidate signals from all symbols
            max_setups: max concurrent setups (default self.max_concurrent_setups)
            risk_budget_pct: total portfolio risk budget (default self.max_portfolio_risk_pct)

        Returns:
            AllocationResult with selected and skipped signals
        """
        max_setups = max_setups or self.max_concurrent_setups
        risk_budget_pct = risk_budget_pct or self.max_portfolio_risk_pct
        total_budget = self.balance * risk_budget_pct

        ranked = sorted(signals, key=lambda s: s.score * s.conviction, reverse=True)
        for i, sig in enumerate(ranked):
            sig.rank = i + 1

        selected: List[RankedSignal] = []
        skipped: List[RankedSignal] = []
        used_budget = 0.0
        notes = []

        if self._open_positions:
            already_open = len(self._open_positions)
            available = max(0, max_setups - already_open)
        else:
            available = max_setups

        if available <= 0 and signals:
            skipped.extend(ranked)
            notes.append(
                f"max concurrent ({max_setups}) already in positions"
            )
            return AllocationResult(
                selected=[], skipped=skipped,
                total_risk_budget=total_budget, used_risk_budget=0.0,
                notes=notes,
            )

        selected_count = 0
        for sig in ranked:
            if selected_count >= available:
                skipped.append(sig)
                continue
            sig_budget = total_budget * (sig.risk_pct / risk_budget_pct)
            if used_budget + sig_budget > total_budget * 1.1:
                skipped.append(sig)
                continue
            sig.allocated = True
            selected.append(sig)
            used_budget += sig_budget
            selected_count += 1

        notes.append(
            f"selected {len(selected)}/{len(ranked)} setups, "
            f"budget used {used_budget:.2f}/{total_budget:.2f}"
        )
        return AllocationResult(
            selected=selected, skipped=skipped,
            total_risk_budget=total_budget,
            used_risk_budget=used_budget,
            notes=notes,
        )

    def pre_check(
        self,
        symbol: str,
        direction: str,
        score: float,
        conviction: float,
        all_symbols: List[str],
    ) -> Tuple[bool, str, float]:
        """Pre-flight check before evaluating a symbol for entry.

        Returns:
            (can_trade: bool, reason: str, volume_multiplier: float)
        """
        can_dd, dd_reason = self.check_portfolio_drawdown()
        if not can_dd:
            return False, dd_reason, 0.0

        corr_mult, corr_reason = self.get_correlation_penalty(
            symbol, direction, all_symbols
        )
        dir_mult, dir_reason = self.get_net_directional_penalty(symbol, direction)

        combined = corr_mult * dir_mult
        reasons = []
        if corr_mult < 1.0:
            reasons.append(corr_reason)
        if dir_mult < 1.0:
            reasons.append(dir_reason)

        return True, "; ".join(reasons) if reasons else "ok", combined

    def record_trade_result(self, symbol: str, profit: float):
        """Record a closed trade for portfolio-level tracking."""
        self._daily_net_pnl += profit
        if profit < 0:
            self._consecutive_portfolio_losses += 1
        else:
            self._consecutive_portfolio_losses = 0
        self.update_balance(self.balance + profit)

    def get_consolidated_risk_metrics(self) -> Dict:
        """Return a snapshot of portfolio-level risk metrics for logging/reporting."""
        dd = self.get_portfolio_drawdown_pct()
        exp = self.compute_exposure()
        return {
            "balance": round(self.balance, 2),
            "peak_balance": round(self.peak_balance, 2),
            "drawdown_pct": round(dd * 100, 2),
            "exposure_long": round(exp.total_long_notional, 2),
            "exposure_short": round(exp.total_short_notional, 2),
            "exposure_net": round(exp.net_notional, 2),
            "exposure_gross": round(exp.gross_notional, 2),
            "positions": exp.total_positions,
            "directional_ratio": round(exp.directional_ratio, 4),
            "daily_pnl": round(self._daily_net_pnl, 2),
            "max_portfolio_risk_pct": self.max_portfolio_risk_pct,
            "portfolio_drawdown_limit_pct": self.portfolio_drawdown_limit_pct,
            "max_concurrent_setups": self.max_concurrent_setups,
        }

    def reset_daily(self):
        """Reset daily counters (call at start of each trading day)."""
        self._daily_net_pnl = 0.0
        self._consecutive_portfolio_losses = 0

    def _get_contract_size(self, symbol: str) -> float:
        s = symbol.upper()
        if "XAU" in s or s == "GOLD":
            return 100.0
        if "XAG" in s or s == "SILVER":
            return 5000.0
        if s in ("NAS100", "US100", "NDX", "DJI30", "US30", "SPX500"):
            return 1.0
        return 100000.0

    def _estimate_trade_notional(self, symbol: str) -> float:
        lots = 0.01
        return lots * self._get_contract_size(symbol) * self._estimate_entry(symbol)

    @staticmethod
    def _estimate_entry(symbol: str) -> float:
        if "XAU" in symbol or symbol == "GOLD":
            return 2000.0
        if "XAG" in symbol or symbol == "SILVER":
            return 25.0
        return 1.0

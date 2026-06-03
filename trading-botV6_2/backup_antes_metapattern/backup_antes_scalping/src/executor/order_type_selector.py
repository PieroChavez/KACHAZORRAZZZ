from typing import Optional
from loguru import logger

from .order_types import OrderType, OrderTypeDecision


class OrderTypeSelector:
    """Selects optimal order type based on market context, signal, and opportunity.

    Decision matrix:
        Entry vs Market     | Trend strength | Spread   | Recommendation
        --------------------|----------------|----------|---------------
        Entry below mkt BUY | any            | any      | LIMIT (retrace)
        Entry above mkt BUY | strong         | tight    | STOP (breakout)
        Entry above mkt BUY | weak/range     | any      | LIMIT (wait)
        Entry at mkt BUY    | strong         | tight    | MARKET
        Entry at mkt BUY    | any            | wide     | LIMIT (avoid slippage)
    """

    def __init__(self, params):
        self.params = params

    def select(
        self,
        direction: str,
        signal_entry: float,
        current_ask: float,
        current_bid: float,
        atr_val: float,
        spread_pips: float,
        score: float,
        conviction: float,
        is_gap_pattern: bool,
        regime_trend_alignment: str = "NEUTRAL",
        adx: float = 0.0,
        min_stop_distance_atr: float = 0.3,
    ) -> OrderTypeDecision:
        entry = signal_entry
        is_buy = direction.upper() == "BUY"
        current = current_ask if is_buy else current_bid
        spread_ratio = spread_pips / max(atr_val, 0.0001)

        is_strong_trend = adx >= 25 and regime_trend_alignment in ("BULLISH_ALIGNED", "BEARISH_ALIGNED", "STRONG")
        is_high_conviction = conviction >= 0.50 and score >= 60
        is_tight_spread = spread_pips <= 3.0 or spread_ratio <= 0.15
        is_wide_spread = spread_pips >= 8.0 or spread_ratio >= 0.4

        if is_buy:
            above_market = entry >= current_ask
            near_market = abs(entry - current_ask) <= atr_val * 0.3
            far_below = entry <= current_bid - atr_val * 1.5
        else:
            above_market = entry <= current_bid
            near_market = abs(entry - current_bid) <= atr_val * 0.3
            far_above = entry >= current_ask + atr_val * 1.5

        def _stop_or_degrade(entry_val: float, reason: str, base_ca: float) -> OrderTypeDecision:
            stop_dist = abs(entry_val - current)
            if stop_dist >= atr_val * min_stop_distance_atr:
                return OrderTypeDecision(
                    OrderType.STOP, entry_val, reason,
                    confidence_adjustment=base_ca,
                )
            if is_high_conviction and is_strong_trend and is_tight_spread:
                return OrderTypeDecision(
                    OrderType.MARKET, current,
                    f"STOP too close ({stop_dist/atr_val:.2f} ATR), → MARKET @ {current:.5f}",
                    confidence_adjustment=base_ca * 0.95,
                )
            limit_price = current_ask if is_buy else current_bid
            return OrderTypeDecision(
                OrderType.LIMIT, limit_price,
                f"STOP too close ({stop_dist/atr_val:.2f} ATR), → LIMIT @ {limit_price:.5f}",
                confidence_adjustment=base_ca * 0.9,
            )

        if is_gap_pattern and near_market:
            return OrderTypeDecision(
                OrderType.MARKET, entry,
                "Gap pattern at market price → market entry",
                confidence_adjustment=0.9,
            )

        if is_high_conviction and is_strong_trend and is_tight_spread and near_market:
            return OrderTypeDecision(
                OrderType.MARKET, current,
                f"Strong trend + high conviction + tight spread → market entry @ {current:.5f}",
                confidence_adjustment=1.0,
            )

        if is_buy:
            if above_market:
                if is_strong_trend and is_tight_spread:
                    return _stop_or_degrade(
                        entry,
                        f"BUY entry {entry:.5f} above ask {current_ask:.5f} in strong trend → BUY STOP",
                        1.1,
                    )
                if is_high_conviction and not is_wide_spread:
                    return _stop_or_degrade(
                        entry,
                        f"BUY entry above ask with good conviction → BUY STOP",
                        1.0,
                    )
                return OrderTypeDecision(
                    OrderType.LIMIT, current_ask,
                    f"BUY entry {entry:.5f} above ask in weak/range → wait as LIMIT at ask",
                    confidence_adjustment=0.85,
                )
            else:
                if far_below and is_high_conviction:
                    narrow = atr_val * 0.3
                    improved = entry + narrow
                    return OrderTypeDecision(
                        OrderType.LIMIT, improved,
                        f"BUY entry {entry:.5f} far below bid → tighter LIMIT @ {improved:.5f}",
                        confidence_adjustment=1.05,
                    )
                return OrderTypeDecision(
                    OrderType.LIMIT, entry,
                    f"BUY entry {entry:.5f} below ask → BUY LIMIT",
                    confidence_adjustment=1.0,
                )
        else:
            if above_market:
                if is_strong_trend and is_tight_spread:
                    return _stop_or_degrade(
                        entry,
                        f"SELL entry {entry:.5f} below bid {current_bid:.5f} in strong trend → SELL STOP",
                        1.1,
                    )
                if is_high_conviction and not is_wide_spread:
                    return _stop_or_degrade(
                        entry,
                        f"SELL entry below bid with good conviction → SELL STOP",
                        1.0,
                    )
                return OrderTypeDecision(
                    OrderType.LIMIT, current_bid,
                    f"SELL entry {entry:.5f} below bid in weak/range → wait as LIMIT at bid",
                    confidence_adjustment=0.85,
                )
            else:
                if far_above and is_high_conviction:
                    narrow = atr_val * 0.3
                    improved = entry - narrow
                    return OrderTypeDecision(
                        OrderType.LIMIT, improved,
                        f"SELL entry {entry:.5f} far above ask → tighter LIMIT @ {improved:.5f}",
                        confidence_adjustment=1.05,
                    )
                return OrderTypeDecision(
                    OrderType.LIMIT, entry,
                    f"SELL entry {entry:.5f} above bid → SELL LIMIT",
                    confidence_adjustment=1.0,
                )

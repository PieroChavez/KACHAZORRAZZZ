"""Order Executor via MT5
Handles order placement, modification, and closing
"""
import time
import MetaTrader5 as mt5
from datetime import datetime
from typing import Optional
from dataclasses import dataclass
from loguru import logger

from ..adapters.mt5_client import MT5Client


@dataclass
class OrderResult:
    """Result of an order operation"""
    success: bool
    order_ticket: Optional[int]
    message: str
    error_code: Optional[int] = None


class OrderExecutor:
    """Executes orders via MT5"""

    def __init__(self, mt5_client: MT5Client):
        self.mt5 = mt5_client

    def place_pending_limit(self, signal, volume: float, limit_price: float) -> OrderResult:
        """Place a pending LIMIT order at the specified price

        Args:
            signal: TradingSignal object
            volume: Lot size
            limit_price: Price at which to place the limit order

        Returns:
            OrderResult with the pending order ticket
        """
        if signal.direction in ("HOLD", "NEUTRAL", "neutral"):
            return OrderResult(success=False, order_ticket=None, message="Neutral signal")

        order_type = (mt5.ORDER_TYPE_BUY_LIMIT if signal.direction.upper() == "BUY"
                      else mt5.ORDER_TYPE_SELL_LIMIT)

        symbol_info = self.mt5.get_symbol_info(signal.symbol)
        if not symbol_info:
            return OrderResult(success=False, order_ticket=None,
                               message=f"No symbol info for {signal.symbol}")

        sl_dist = abs(signal.stop_loss - signal.entry_price) if signal.stop_loss else 0
        tp_dist = abs(signal.take_profit - signal.entry_price) if signal.take_profit else 0
        is_long = signal.direction.upper() == "BUY"
        adjusted_sl = round(limit_price - sl_dist, symbol_info["digits"]) if is_long else round(limit_price + sl_dist, symbol_info["digits"])
        adjusted_tp = round(limit_price + tp_dist, symbol_info["digits"]) if is_long else round(limit_price - tp_dist, symbol_info["digits"])

        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": signal.symbol,
            "volume": volume,
            "type": order_type,
            "price": limit_price,
            "sl": adjusted_sl,
            "tp": adjusted_tp,
            "deviation": 10,
            "magic": 20260520,
            "comment": f"limit_{signal.direction.upper()}_{signal.score:.0f}",
            "type_time": mt5.ORDER_TIME_GTC,
        }

        PENDING_FILLING = [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK]
        for fill in PENDING_FILLING:
            req = {**request, "type_filling": fill}
            result = mt5.order_send(req)
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Pending LIMIT {signal.direction} {volume} lots {signal.symbol} @ {limit_price}")
                return OrderResult(success=True, order_ticket=result.order,
                                   message=f"Pending order: {result.order}")
            if result is not None:
                logger.debug(f"Pending fill mode {fill}: {result.retcode} {result.comment}")

        req = {k: v for k, v in request.items() if k != "type_filling"}
        result = mt5.order_send(req)
        if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Pending LIMIT {signal.direction} {volume} lots {signal.symbol} @ {limit_price}")
            return OrderResult(success=True, order_ticket=result.order,
                               message=f"Pending order: {result.order}")

        return OrderResult(success=False, order_ticket=None,
                           message=f"Rejected by broker: {result.comment if result else 'unknown'}",
                           error_code=result.retcode if result else None)

    def place_pending_entry(self, signal, volume: float, entry_price: float,
                            custom_tp_distance: float = None) -> OrderResult:
        """Place a pending LIMIT or STOP order at the specified entry price.
        Chooses LIMIT if price would need to retrace, STOP if price would need to break out.

        Args:
            signal: TradingSignal object
            volume: Lot size
            entry_price: Price at which to place the pending order
            custom_tp_distance: If set, override TP distance (pips) for this order
                               (used for scale entries with different TP ratios)

        Returns:
            OrderResult with the pending order ticket
        """
        if signal.direction in ("HOLD", "NEUTRAL", "neutral"):
            return OrderResult(success=False, order_ticket=None, message="Neutral signal")

        symbol_info = self.mt5.get_symbol_info(signal.symbol)
        if not symbol_info:
            return OrderResult(success=False, order_ticket=None,
                               message=f"No symbol info for {signal.symbol}")

        current_ask = symbol_info["ask"]
        current_bid = symbol_info["bid"]
        is_buy = signal.direction.upper() == "BUY"

        if is_buy:
            if entry_price >= current_ask:
                return OrderResult(success=False, order_ticket=None,
                                   message=f"BUY {entry_price} >= ask {current_ask}, would need STOP, skipping")
            order_type = mt5.ORDER_TYPE_BUY_LIMIT
        else:
            if entry_price <= current_bid:
                return OrderResult(success=False, order_ticket=None,
                                   message=f"SELL {entry_price} <= bid {current_bid}, would need STOP, skipping")
            order_type = mt5.ORDER_TYPE_SELL_LIMIT

        sl_dist = abs(signal.stop_loss - signal.entry_price) if signal.stop_loss else 0
        tp_dist = custom_tp_distance if custom_tp_distance is not None else (abs(signal.take_profit - signal.entry_price) if signal.take_profit else 0)
        if is_buy:
            adjusted_sl = round(entry_price - sl_dist, symbol_info["digits"])
            adjusted_tp = round(entry_price + tp_dist, symbol_info["digits"])
        else:
            adjusted_sl = round(entry_price + sl_dist, symbol_info["digits"])
            adjusted_tp = round(entry_price - tp_dist, symbol_info["digits"])

        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": signal.symbol,
            "volume": volume,
            "type": order_type,
            "price": entry_price,
            "sl": adjusted_sl,
            "tp": adjusted_tp,
            "deviation": 10,
            "magic": 20260520,
            "comment": f"pend_{signal.direction.upper()}_{signal.score:.0f}",
            "type_time": mt5.ORDER_TIME_GTC,
        }

        for fill in [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, None]:
            req = {**request}
            if fill is not None:
                req["type_filling"] = fill
            result = mt5.order_send(req)
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                label = "LIMIT" if order_type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT) else "STOP"
                logger.info(f"Pending {label} {signal.direction} {volume} lots {signal.symbol} @ {entry_price}")
                return OrderResult(success=True, order_ticket=result.order,
                                   message=f"Pending order: {result.order}")
            if result is not None:
                logger.debug(f"Pending fill mode {fill}: {result.retcode} {result.comment}")

        return OrderResult(success=False, order_ticket=None,
                           message=f"Pending order rejected by broker",
                           error_code=result.retcode if result else None)

    def cancel_pending_order(self, ticket: int) -> OrderResult:
        """Cancel a pending order by ticket number

        Args:
            ticket: Pending order ticket number

        Returns:
            OrderResult with status
        """
        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": ticket,
        }
        result = mt5.order_send(request)
        if result is None:
            err = mt5.last_error()
            logger.error(f"Cancel pending {ticket} failed: {err}")
            return OrderResult(success=False, order_ticket=ticket,
                               message=f"Cancel failed: {err}")
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Pending order {ticket} cancelled")
            return OrderResult(success=True, order_ticket=ticket,
                               message="Pending order cancelled")
        logger.warning(f"Cancel pending {ticket}: {result.retcode} {result.comment}")
        return OrderResult(success=False, order_ticket=ticket,
                           message=f"Cancel failed: {result.comment}", error_code=result.retcode)

    def execute_signal(self, signal, volume: float) -> OrderResult:
        if signal.direction in ("HOLD", "NEUTRAL", "neutral"):
            return OrderResult(
                success=False, order_ticket=None,
                message="Neutral signal - no execution"
            )

        order_type = (
            mt5.ORDER_TYPE_BUY if signal.direction.upper() == "BUY"
            else mt5.ORDER_TYPE_SELL
        )

        max_retries = 3
        for attempt in range(max_retries):
            symbol_info = self.mt5.get_symbol_info(signal.symbol)
            if not symbol_info:
                if attempt == max_retries - 1:
                    return OrderResult(
                        success=False, order_ticket=None,
                        message=f"Could not get symbol info for {signal.symbol}"
                    )
                continue

            price = symbol_info["ask"] if order_type == mt5.ORDER_TYPE_BUY else symbol_info["bid"]
            score = getattr(signal, 'score', getattr(signal, 'confidence', 0.5))

            sl_dist = abs(signal.stop_loss - signal.entry_price) if signal.stop_loss else 0
            tp_dist = abs(signal.take_profit - signal.entry_price) if signal.take_profit else 0
            if signal.direction.upper() == "BUY":
                adjusted_sl = round(price - sl_dist, symbol_info["digits"])
                adjusted_tp = round(price + tp_dist, symbol_info["digits"])
            else:
                adjusted_sl = round(price + sl_dist, symbol_info["digits"])
                adjusted_tp = round(price - tp_dist, symbol_info["digits"])

            price_diff = price - signal.entry_price
            logger.debug(f"Order request: side={order_type}, entry={signal.entry_price}, market={price}, "
                         f"diff={price_diff:.4f}, raw_sl={signal.stop_loss}, raw_tp={signal.take_profit}, "
                         f"adj_sl={adjusted_sl}, adj_tp={adjusted_tp}")

            FILLING_MODES = [
                mt5.ORDER_FILLING_IOC,
                mt5.ORDER_FILLING_FOK,
                None,
            ]

            def _order_req(sl_val=None, tp_val=None, filling=None):
                req = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": signal.symbol,
                    "volume": volume,
                    "type": order_type,
                    "price": price,
                    "deviation": 10,
                    "magic": 20260520,
                    "comment": f"smc_{signal.direction.upper()}_{score:.1f}",
                }
                if filling is not None:
                    req["type_filling"] = filling
                if sl_val is not None:
                    req["sl"] = sl_val
                if tp_val is not None:
                    req["tp"] = tp_val
                return req

            def _send(req):
                res = mt5.order_send(req)
                if res is None:
                    return None, mt5.last_error()
                return res, None

            placed = None
            for fill_mode in FILLING_MODES:
                result, err = _send(_order_req(adjusted_sl, adjusted_tp, filling=fill_mode))
                if result is None:
                    continue
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"Order executed ({fill_mode}): {signal.direction} {volume} lots {signal.symbol} @ {price}")
                    return OrderResult(
                        success=True, order_ticket=result.order,
                        message=f"Order filled: {result.order}"
                    )
                if result.retcode == 10030:
                    continue
                logger.debug(f"Fill mode {fill_mode} failed: {result.retcode} {result.comment}")

            for fill_mode in FILLING_MODES:
                result_bare, err_bare = _send(_order_req(filling=fill_mode))
                if result_bare is None:
                    continue
                if result_bare.retcode == mt5.TRADE_RETCODE_DONE:
                    placed = result_bare
                    break
                if result_bare.retcode == 10030:
                    continue
                logger.debug(f"Bare order mode {fill_mode}: {result_bare.retcode} {result_bare.comment}")

            if placed is None:
                if attempt < max_retries - 1:
                    logger.info(f"Retrying {symbol_info['symbol']} order (attempt {attempt + 2}/{max_retries})")
                    continue
                return OrderResult(
                    success=False, order_ticket=None,
                    message="All fill modes rejected by broker"
                )

            logger.info(f"Bare order executed: {signal.direction} {volume} lots {signal.symbol} @ {price}")

            sl_dist = abs(signal.entry_price - signal.stop_loss)
            tp_dist = abs(signal.take_profit - signal.entry_price)
            pos_price = getattr(placed, 'price', None) or getattr(placed, 'bid', None) or price
            if signal.direction.upper() == "BUY":
                sl_fill = round(pos_price - sl_dist, symbol_info["digits"])
                tp_fill = round(pos_price + tp_dist, symbol_info["digits"])
            else:
                sl_fill = round(pos_price + sl_dist, symbol_info["digits"])
                tp_fill = round(pos_price - tp_dist, symbol_info["digits"])

            for mod_attempt in range(3):
                mod_result = self.modify_position(placed.order, sl_fill, tp_fill)
                if mod_result.success:
                    logger.info(f"SL/TP set after fill: price={pos_price}, sl={sl_fill}, tp={tp_fill}")
                    break
                logger.warning(f"Modify attempt {mod_attempt+1} failed: {mod_result.message} (sl={sl_fill}, tp={tp_fill})")
                if mod_attempt < 2:
                    time.sleep(0.3)
            else:
                logger.warning(f"Could not set SL/TP after 3 attempts, position {placed.order} is unprotected")

            return OrderResult(
                success=True, order_ticket=placed.order,
                message=f"Order filled (bare): {placed.order}"
            )

    def close_position(self, ticket: int, volume: Optional[float] = None) -> OrderResult:
        """Close an open position

        Args:
            ticket: Position ticket number
            volume: Volume to close (None = close all)

        Returns:
            OrderResult with status
        """
        positions = self.mt5.get_positions()
        position = next((p for p in positions if p["ticket"] == ticket), None)

        if not position:
            return OrderResult(
                success=False,
                order_ticket=ticket,
                message=f"Position {ticket} not found"
            )

        order_type = (
            mt5.ORDER_TYPE_SELL if position["type"] == "buy"
            else mt5.ORDER_TYPE_BUY
        )

        symbol_info = self.mt5.get_symbol_info(position["symbol"])
        if not symbol_info:
            return OrderResult(
                success=False,
                order_ticket=ticket,
                message="Could not get symbol info"
            )

        price = symbol_info["bid"] if order_type == mt5.ORDER_TYPE_BUY else symbol_info["ask"]

        volume_to_close = volume if volume else position["volume"]

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position["symbol"],
            "volume": volume_to_close,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 10,
            "magic": 20230505,
            "comment": "IFVG_close",
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        result = mt5.order_send(request)

        if result is None:
            return OrderResult(
                success=False,
                order_ticket=ticket,
                message=f"Close failed: {mt5.last_error()}",
            )

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Position {ticket} closed successfully")
            return OrderResult(
                success=True,
                order_ticket=ticket,
                message="Position closed"
            )

        return OrderResult(
            success=False,
            order_ticket=ticket,
            message=f"Close failed: {result.comment}",
            error_code=result.retcode
        )

    def modify_position(self, ticket: int, new_sl: Optional[float] = None,
                        new_tp: Optional[float] = None) -> OrderResult:
        """Modify SL/TP of an open position

        Args:
            ticket: Position ticket number
            new_sl: New stop loss price
            new_tp: New take profit price

        Returns:
            OrderResult with status
        """
        positions = self.mt5.get_positions()
        position = next((p for p in positions if p["ticket"] == ticket), None)

        if not position:
            return OrderResult(
                success=False,
                order_ticket=ticket,
                message=f"Position {ticket} not found"
            )

        sl = new_sl if new_sl is not None else position["sl"]
        tp = new_tp if new_tp is not None else position["tp"]

        request = {
            "action": 6,
            "position": ticket,
            "symbol": position["symbol"],
            "sl": sl,
            "tp": tp,
            "magic": 20230505,
        }

        result = mt5.order_send(request)

        if result is None:
            return OrderResult(
                success=False,
                order_ticket=ticket,
                message=f"Modify failed: {mt5.last_error()}",
            )

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Position {ticket} modified: SL={sl}, TP={tp}")
            return OrderResult(
                success=True,
                order_ticket=ticket,
                message="Position modified"
            )

        return OrderResult(
            success=False,
            order_ticket=ticket,
            message=f"Modify failed: {result.comment}",
            error_code=result.retcode
        )

import pytest
from datetime import datetime
from src.backtest.simulated_executor import SimulatedExecutor, SimulatedTrade


class TestSimulatedExecutor:
    def test_init_sets_balance_equity_empty_positions(self):
        ex = SimulatedExecutor(initial_balance=10_000.0, spread_pips=0.5)
        assert ex.balance == 10_000.0
        assert ex.equity == 10_000.0
        assert ex.positions == {}
        assert ex.closed_trades == []
        assert ex.open_positions == []
        assert ex.spread_pips == 0.5
        assert ex.equity_curve == [10_000.0]

    def test_open_position_buy_creates_correct_trade(self):
        ex = SimulatedExecutor()
        tid = ex.open_position("XAUUSD", "BUY", 100.0, 0.01, 99.0, 105.0)
        trade = ex.positions[tid]
        assert trade.direction == "BUY"
        assert trade.symbol == "XAUUSD"
        assert trade.volume == 0.01
        assert trade.stop_loss == 99.0
        assert trade.take_profit == 105.0

    def test_open_position_sell_creates_correct_trade(self):
        ex = SimulatedExecutor()
        tid = ex.open_position("XAGUSD", "SELL", 30.0, 0.1, 31.0, 28.0)
        trade = ex.positions[tid]
        assert trade.direction == "SELL"
        assert trade.symbol == "XAGUSD"
        assert trade.volume == 0.1
        assert trade.stop_loss == 31.0
        assert trade.take_profit == 28.0

    def test_open_position_adjusts_entry_price_for_spread(self):
        ex = SimulatedExecutor(spread_pips=0.5)
        tid = ex.open_position("XAUUSD", "BUY", 100.0, 0.01, 99.0, 105.0)
        trade = ex.positions[tid]
        expected = 100.0 + 0.5 * 0.0001
        assert trade.entry_price == pytest.approx(expected)

    def test_open_position_sell_adjusts_entry_down_for_spread(self):
        ex = SimulatedExecutor(spread_pips=0.5)
        tid = ex.open_position("XAUUSD", "SELL", 100.0, 0.01, 101.0, 98.0)
        trade = ex.positions[tid]
        expected = 100.0 - 0.5 * 0.0001
        assert trade.entry_price == pytest.approx(expected)

    def test_open_position_returns_incremental_ticket_ids(self):
        ex = SimulatedExecutor()
        tid1 = ex.open_position("XAUUSD", "BUY", 100.0, 0.01, 99.0, 105.0)
        tid2 = ex.open_position("XAGUSD", "SELL", 30.0, 0.1, 31.0, 28.0)
        tid3 = ex.open_position("NAS100", "BUY", 15000.0, 0.1, 14900.0, 15200.0)
        assert tid1 == 1
        assert tid2 == 2
        assert tid3 == 3

    def test_update_positions_hits_stop_loss_for_buy(self):
        ex = SimulatedExecutor(spread_pips=0.0)
        tid = ex.open_position("XAUUSD", "BUY", 100.0, 0.01, 99.0, 105.0)
        ex.update_positions(high=101.0, low=98.0, close=100.0, timestamp=datetime.now())
        assert tid not in ex.positions
        assert len(ex.closed_trades) == 1
        trade = ex.closed_trades[0]
        assert trade.exit_reason == "sl"
        assert trade.exit_price == 99.0

    def test_update_positions_hits_take_profit_for_buy(self):
        ex = SimulatedExecutor(spread_pips=0.0)
        tid = ex.open_position("XAUUSD", "BUY", 100.0, 0.01, 99.0, 105.0)
        ex.update_positions(high=106.0, low=101.0, close=103.0, timestamp=datetime.now())
        assert tid not in ex.positions
        assert len(ex.closed_trades) == 1
        trade = ex.closed_trades[0]
        assert trade.exit_reason == "tp"
        assert trade.exit_price == 105.0

    def test_update_positions_hits_stop_loss_for_sell(self):
        ex = SimulatedExecutor(spread_pips=0.0)
        tid = ex.open_position("XAUUSD", "SELL", 100.0, 0.01, 101.0, 95.0)
        ex.update_positions(high=102.0, low=99.0, close=100.5, timestamp=datetime.now())
        assert tid not in ex.positions
        assert len(ex.closed_trades) == 1
        trade = ex.closed_trades[0]
        assert trade.exit_reason == "sl"
        assert trade.exit_price == 101.0

    def test_update_positions_hits_take_profit_for_sell(self):
        ex = SimulatedExecutor(spread_pips=0.0)
        tid = ex.open_position("XAUUSD", "SELL", 100.0, 0.01, 101.0, 95.0)
        ex.update_positions(high=100.5, low=94.0, close=98.0, timestamp=datetime.now())
        assert tid not in ex.positions
        assert len(ex.closed_trades) == 1
        trade = ex.closed_trades[0]
        assert trade.exit_reason == "tp"
        assert trade.exit_price == 95.0

    def test_close_position_closes_specific_trade(self):
        ex = SimulatedExecutor(spread_pips=0.0)
        tid = ex.open_position("XAUUSD", "BUY", 100.0, 0.01, 99.0, 105.0)
        ex.close_position(tid, reason="signal", timestamp=datetime.now())
        assert tid not in ex.positions
        assert len(ex.closed_trades) == 1
        assert ex.closed_trades[0].exit_reason == "signal"

    def test_close_all_closes_all_positions(self):
        ex = SimulatedExecutor(spread_pips=0.0)
        ex.open_position("XAUUSD", "BUY", 100.0, 0.01, 99.0, 105.0)
        ex.open_position("XAGUSD", "SELL", 30.0, 0.1, 31.0, 28.0)
        ex.close_all(reason="end", timestamp=datetime.now())
        assert ex.positions == {}
        assert len(ex.closed_trades) == 2

    def test_closed_trades_returns_list_of_closed_trades(self):
        ex = SimulatedExecutor(spread_pips=0.0)
        assert ex.closed_trades == []
        tid = ex.open_position("XAUUSD", "BUY", 100.0, 0.01, 99.0, 105.0)
        ex.close_position(tid, timestamp=datetime.now())
        assert len(ex.closed_trades) == 1
        assert isinstance(ex.closed_trades[0], SimulatedTrade)

    def test_calc_risk_calculates_correct_amount(self):
        ex = SimulatedExecutor(spread_pips=0.0)
        trade = SimulatedTrade(
            symbol="XAUUSD", direction="BUY", entry_time=datetime.now(),
            entry_price=100.0, volume=0.01, stop_loss=99.0, take_profit=105.0,
        )
        risk = ex._calc_risk(trade)
        sl_distance = 100.0 - 99.0
        expected = sl_distance / 0.0001 * 10.0 * 0.01
        assert risk == pytest.approx(expected)

    def test_calc_risk_zero_when_sl_beyond_entry(self):
        ex = SimulatedExecutor(spread_pips=0.0)
        trade = SimulatedTrade(
            symbol="XAUUSD", direction="BUY", entry_time=datetime.now(),
            entry_price=100.0, volume=0.01, stop_loss=101.0, take_profit=105.0,
        )
        risk = ex._calc_risk(trade)
        assert risk == 0.0

    def test_apply_trailing_moves_sl_for_buy_on_new_high(self):
        ex = SimulatedExecutor()
        trade = SimulatedTrade(
            symbol="XAUUSD", direction="BUY", entry_time=datetime.now(),
            entry_price=100.0, volume=0.01, stop_loss=98.0, take_profit=110.0,
        )
        ex._apply_trailing(trade, high=105.0, low=100.0, close=103.0,
                           config={"trail_mult": 2.0, "atr": 2.0})
        assert trade.stop_loss == pytest.approx(101.0)

    def test_apply_trailing_does_not_move_sl_when_candidate_below(self):
        ex = SimulatedExecutor()
        trade = SimulatedTrade(
            symbol="XAUUSD", direction="BUY", entry_time=datetime.now(),
            entry_price=100.0, volume=0.01, stop_loss=99.5, take_profit=110.0,
        )
        ex._apply_trailing(trade, high=101.0, low=100.0, close=100.5,
                           config={"trail_mult": 2.0, "atr": 2.0})
        assert trade.stop_loss == 99.5

    def test_apply_trailing_moves_sl_for_sell_on_new_low(self):
        ex = SimulatedExecutor()
        trade = SimulatedTrade(
            symbol="XAUUSD", direction="SELL", entry_time=datetime.now(),
            entry_price=100.0, volume=0.01, stop_loss=105.0, take_profit=90.0,
        )
        ex._apply_trailing(trade, high=100.0, low=95.0, close=97.0,
                           config={"trail_mult": 2.0, "atr": 2.0})
        assert trade.stop_loss == pytest.approx(99.0)

    def test_apply_trailing_no_op_when_atr_zero(self):
        ex = SimulatedExecutor()
        trade = SimulatedTrade(
            symbol="XAUUSD", direction="BUY", entry_time=datetime.now(),
            entry_price=100.0, volume=0.01, stop_loss=98.0, take_profit=110.0,
        )
        ex._apply_trailing(trade, high=105.0, low=100.0, close=103.0,
                           config={"trail_mult": 2.0, "atr": 0.0})
        assert trade.stop_loss == 98.0

    def test_pip_value_returns_correct_for_various_symbols(self):
        ex = SimulatedExecutor()
        assert ex._pip_value("XAUUSD") == 10.0
        assert ex._pip_value("GOLD") == 10.0
        assert ex._pip_value("XAGUSD") == 50.0
        assert ex._pip_value("SILVER") == 50.0
        assert ex._pip_value("NAS100") == 1.0
        assert ex._pip_value("US100") == 1.0
        assert ex._pip_value("NDX") == 1.0
        assert ex._pip_value("DJI30") == 1.0
        assert ex._pip_value("US30") == 1.0
        assert ex._pip_value("SPX500") == 1.0
        assert ex._pip_value("EURUSD") == 10.0
        assert ex._pip_value("BTCUSD") == 10.0

    def test_can_open_respects_max_positions(self):
        ex = SimulatedExecutor(spread_pips=0.0)
        assert ex.can_open(1)
        ex.open_position("XAUUSD", "BUY", 100.0, 0.01, 99.0, 105.0)
        assert not ex.can_open(1)
        assert ex.can_open(2)

    def test_get_drawdown_pct_calculates_correctly(self):
        ex = SimulatedExecutor(initial_balance=10_000.0)
        ex.equity_curve = [10_000.0, 11_000.0, 9_000.0]
        ex.equity = 9_000.0
        expected = (11_000.0 - 9_000.0) / 11_000.0
        assert ex.get_peak_equity() == 11_000.0
        assert ex.get_drawdown_pct() == pytest.approx(expected)

    def test_get_peak_equity_returns_max(self):
        ex = SimulatedExecutor(initial_balance=10_000.0)
        ex.equity_curve = [10_000.0, 12_000.0, 11_000.0, 15_000.0, 13_000.0]
        ex.equity = 13_000.0
        assert ex.get_peak_equity() == 15_000.0

    def test_get_peak_equity_falls_back_to_equity_when_empty(self):
        ex = SimulatedExecutor(initial_balance=10_000.0)
        ex.equity_curve = []
        ex.equity = 5_000.0
        assert ex.get_peak_equity() == 5_000.0

    def test_multiple_trades_accumulate_balance_correctly(self):
        ex = SimulatedExecutor(initial_balance=10_000.0, spread_pips=0.0)
        tid1 = ex.open_position("XAUUSD", "BUY", 100.0, 0.01, 99.0, 105.0)
        ex.update_positions(high=106.0, low=101.0, close=103.0, timestamp=datetime.now())
        balance_after_first = ex.balance
        tid2 = ex.open_position("XAUUSD", "BUY", 100.0, 0.01, 99.0, 105.0)
        ex.update_positions(high=106.0, low=101.0, close=103.0, timestamp=datetime.now())
        assert ex.balance > balance_after_first

    def test_update_positions_sets_exit_price_to_close_when_no_sl_tp(self):
        ex = SimulatedExecutor(spread_pips=0.0)
        tid = ex.open_position("XAUUSD", "BUY", 100.0, 0.01, 90.0, 110.0)
        ex.update_positions(high=102.0, low=99.0, close=101.5, timestamp=datetime.now())
        assert tid in ex.positions
        assert ex.positions[tid].exit_price == 101.5

    def test_open_position_deducts_risk_from_balance(self):
        ex = SimulatedExecutor(initial_balance=10_000.0, spread_pips=0.0)
        ex.open_position("XAUUSD", "BUY", 100.0, 0.01, 99.0, 105.0)
        expected_risk = 1.0 / 0.0001 * 10.0 * 0.01
        assert ex.balance == pytest.approx(10_000.0 - expected_risk)

    def test_close_position_updates_balance_with_profit(self):
        ex = SimulatedExecutor(initial_balance=10_000.0, spread_pips=0.0)
        tid = ex.open_position("XAUUSD", "BUY", 100.0, 0.01, 99.0, 105.0)
        ex.update_positions(high=103.0, low=101.0, close=102.0, timestamp=datetime.now())
        balance_after_open = ex.balance
        ex.close_position(tid, timestamp=datetime.now())
        assert ex.balance > balance_after_open

    def test_bars_held_increments_on_update(self):
        ex = SimulatedExecutor(spread_pips=0.0)
        tid = ex.open_position("XAUUSD", "BUY", 100.0, 0.01, 90.0, 110.0)
        ex.update_positions(high=101.0, low=99.0, close=100.0, timestamp=datetime.now())
        ex.update_positions(high=102.0, low=100.0, close=101.0, timestamp=datetime.now())
        assert ex.positions[tid].bars_held == 2

    def test_update_positions_closes_sl_and_tp_in_same_bar_buy(self):
        ex = SimulatedExecutor(spread_pips=0.0)
        tid = ex.open_position("XAUUSD", "BUY", 100.0, 0.01, 99.0, 105.0)
        ex.update_positions(high=106.0, low=98.0, close=102.0, timestamp=datetime.now())
        assert tid not in ex.positions
        assert len(ex.closed_trades) == 1

    def test_update_positions_closes_sl_and_tp_in_same_bar_sell(self):
        ex = SimulatedExecutor(spread_pips=0.0)
        tid = ex.open_position("XAUUSD", "SELL", 100.0, 0.01, 101.0, 95.0)
        ex.update_positions(high=102.0, low=94.0, close=98.0, timestamp=datetime.now())
        assert tid not in ex.positions
        assert len(ex.closed_trades) == 1

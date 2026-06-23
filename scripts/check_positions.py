import MetaTrader5 as mt5
mt5.initialize()
mt5.login(196559696, '12P@ola12', 'Exness-MT5Real11')
positions = mt5.positions_get(symbol='XAUUSDc')
if positions:
    for p in positions:
        direction = "BUY" if p.type == 0 else "SELL"
        print(f"Ticket: {p.ticket}, {direction}, Entry: {p.price_open}, Current: {p.price_current:.2f}, SL: {p.sl}, TP: {p.tp}, Profit: ${p.profit:.2f}")
else:
    print("No open positions for XAUUSDc")
mt5.shutdown()

import MetaTrader5 as mt5
mt5.initialize()
positions = mt5.positions_get()
print(f"Posiciones abiertas: {len(positions) if positions else 0}")
total_pnl = 0.0
for p in (positions or []):
    d = "BUY" if p.type == 0 else "SELL"
    print(f"  Ticket:{p.ticket} {p.symbol} {d} Vol:{p.volume} Open:{p.price_open:.5f} Cur:{p.price_current:.5f} SL:{p.sl} TP:{p.tp} P/L:${p.profit:.2f}")
    total_pnl += p.profit
orders = mt5.orders_get()
print(f"\nOrdenes pendientes: {len(orders) if orders else 0}")
for o in (orders or []):
    print(f"  Ticket:{o.ticket} {o.symbol} Type:{o.type} Price:{o.price_open:.5f}")
print(f"\nP/L Total: ${total_pnl:.2f}")
mt5.shutdown()

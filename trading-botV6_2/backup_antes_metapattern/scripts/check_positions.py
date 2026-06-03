import MetaTrader5 as mt5
mt5.initialize()
for p in (mt5.positions_get() or []):
    print(f'pos: ticket={p.ticket} sym={p.symbol} type={"SELL" if p.type==1 else "BUY"} vol={p.volume} entry={p.price_open} sl={p.sl} tp={p.tp} profit={p.profit:.2f}')
if not mt5.positions_get():
    print("No positions")
mt5.shutdown()

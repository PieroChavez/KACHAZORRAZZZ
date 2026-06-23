import MetaTrader5 as mt5
import time

mt5.initialize()
tick = mt5.symbol_info_tick('XAUUSDc')
print(f'Bid: {tick.bid}, Ask: {tick.ask}, Spread: {tick.ask-tick.bid:.2f}')
si = mt5.symbol_info('XAUUSDc')
print(f'Digits: {si.digits}, Point: {si.point}, StopsLevel: {si.trade_stops_level}, FreezeLevel: {si.trade_freeze_level}')

ps = mt5.positions_get(symbol='XAUUSDc') or []
for p in ps:
    ptype = 'SELL' if p.type == 1 else 'BUY'
    print(f'Position: type={ptype} entry={p.price_open} sl={p.sl} tp={p.tp} cur={p.price_current}')
    
    if p.type == 1:
        sl_val = round(p.price_open + 0.20, 2)
        tp_val = round(p.price_open - 0.52, 2)
        print(f'  SL={sl_val} > bid={tick.bid}? {sl_val > tick.bid}')
        print(f'  TP={tp_val} < bid={tick.bid}? {tp_val < tick.bid}')
        
        req = {
            'action': 6,
            'position': p.ticket,
            'symbol': 'XAUUSDc',
            'sl': sl_val,
            'tp': tp_val
        }
        r = mt5.order_send(req)
        print(f'  retcode={r.retcode} comment={r.comment}')
    else:
        sl_val = round(p.price_open - 0.20, 2)
        tp_val = round(p.price_open + 0.52, 2)
        print(f'  SL={sl_val} < ask={tick.ask}? {sl_val < tick.ask}')
        print(f'  TP={tp_val} > ask={tick.ask}? {tp_val > tick.ask}')
        
        req = {
            'action': 6,
            'position': p.ticket,
            'symbol': 'XAUUSDc',
            'sl': sl_val,
            'tp': tp_val
        }
        r = mt5.order_send(req)
        print(f'  retcode={r.retcode} comment={r.comment}')

if not ps:
    print('No positions to modify')
mt5.shutdown()

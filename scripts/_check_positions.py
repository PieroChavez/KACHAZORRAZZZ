import MetaTrader5 as mt5
import os
from dotenv import load_dotenv

load_dotenv("P:/PROYECTOS/OneDrive/Escritorio/trading-botV3/.env")

login = int(os.getenv("MT5_LOGIN", 0))
password = os.getenv("MT5_PASSWORD", "")
server = os.getenv("MT5_SERVER", "")

if not mt5.initialize():
    print("Error: No se pudo iniciar MT5")
    mt5.shutdown()
    exit()

print(f"MT5 version: {mt5.version()}")

if not mt5.login(login, password, server):
    print(f"Error login: {mt5.last_error()}")
    mt5.shutdown()
    exit()

print(f"Login OK: {login}")

# Account info
acc = mt5.account_info()
if acc:
    print(f"Cuenta: {acc.login} | Balance: ${acc.balance:.2f} | Equity: ${acc.equity:.2f} | Profit: ${acc.profit:.2f}")
    print(f"MarginFree: ${acc.margin_free:.2f}")

# Positions
positions = mt5.positions_get()
if positions:
    print(f"\nPosiciones abiertas: {len(positions)}")
    for p in positions:
        sl_str = f'{p.sl:.2f}' if p.sl else 'N/A'
        tp_str = f'{p.tp:.2f}' if p.tp else 'N/A'
        print(f"  #{p.ticket} {p.symbol} {'BUY' if p.type==0 else 'SELL'} vol={p.volume:.2f} "
              f"open={p.price_open:.2f} current={p.price_current:.2f} "
              f"SL={sl_str} TP={tp_str} "
              f"profit=${p.profit:.2f} swap=${p.swap:.2f}")
else:
    print("\nNo hay posiciones abiertas")

# Orders
orders = mt5.orders_get()
if orders:
    print(f"\nOrdenes pendientes: {len(orders)}")
    for o in orders:
        dir_map = {0:'BUY',1:'SELL',2:'BUY_LIMIT',3:'SELL_LIMIT',4:'BUY_STOP',5:'SELL_STOP'}
        vol = getattr(o, 'volume_initial', getattr(o, 'volume', 0))
        price = getattr(o, 'price_open', getattr(o, 'price', 0))
        sl = o.sl if hasattr(o, 'sl') and o.sl else 0
        tp = o.tp if hasattr(o, 'tp') and o.tp else 0
        print(f"  #{o.ticket} {o.symbol} {dir_map.get(o.type, f'TYPE_{o.type}')} "
              f"vol={vol:.2f} price={price:.2f} SL={sl:.2f} TP={tp:.2f}")
else:
    print("\nNo hay ordenes pendientes")

mt5.shutdown()

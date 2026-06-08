import sys, os
sys.path.insert(0, 'P:/PROYECTOS/OneDrive/Escritorio/trading-botV3')
from dotenv import load_dotenv
load_dotenv('P:/PROYECTOS/OneDrive/Escritorio/trading-botV3/.env')

import MetaTrader5 as mt5
mt5.initialize()
login = int(os.getenv('MT5_LOGIN', 0))
password = os.getenv('MT5_PASSWORD', '')
server = os.getenv('MT5_SERVER', '')
mt5.login(login, password, server)

from src.adapters.mt5_client import MT5Client
client = MT5Client()
from src.strategies.order_pack import OrderPackManager

mgr = OrderPackManager(client, 'XAUEURm')

print("BEFORE manage_all:")
for pid, pack in mgr._packs.items():
    subs = mgr._subs.get(pid, [])
    active_subs = [s for s in subs if s.status == 'active']
    if active_subs:
        print(f'  Pack #{pid} BE={pack.breakeven_activated} Trail={pack.trailing_activated} status={pack.status}')
        for s in active_subs:
            pos = mgr._get_position(s.ticket)
            if pos:
                print(f'    Sub P{s.position_number} ticket={s.ticket} sl_db={s.sl_current:.2f} sl_mt5={pos["sl"]} price={pos["price_current"]:.2f}')
            else:
                print(f'    Sub P{s.position_number} ticket={s.ticket} **NO POSITION**')

from datetime import datetime
mgr.manage_all(datetime.utcnow(), None)

print("\nAFTER manage_all:")
for pid, pack in mgr._packs.items():
    if pack.breakeven_activated or pack.trailing_activated:
        print(f'  Pack #{pid} BE={pack.breakeven_activated} Trail={pack.trailing_activated}')
    subs = mgr._subs.get(pid, [])
    for s in subs:
        if s.status == 'active':
            pos = mgr._get_position(s.ticket)
            if pos:
                print(f'    Sub P{s.position_number} sl_db={s.sl_current:.2f} sl_mt5={pos["sl"]}')

mt5.shutdown()

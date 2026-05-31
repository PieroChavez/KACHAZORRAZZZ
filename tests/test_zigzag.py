from data.market_data import MarketData
from strategies.zigzag import detect_zigzag

import MetaTrader5 as mt5

from core.config import (
    MT5_LOGIN,
    MT5_PASSWORD,
    MT5_SERVER,
    MT5_PATH
)

# ==========================
# CONNECT MT5
# ==========================

if MT5_PATH:
    mt5.initialize(path=MT5_PATH)
else:
    mt5.initialize()

authorized = mt5.login(
    login=MT5_LOGIN,
    password=MT5_PASSWORD,
    server=MT5_SERVER
)

if not authorized:

    print(mt5.last_error())

    quit()

# ==========================
# DATA
# ==========================

market = MarketData(
    symbol="XAUUSDm"
)

df = market.get_historical_data(
    timeframe="M15",
    bars=1000
)

print(df.tail())

# ==========================
# ZIGZAG
# ==========================

pivots = detect_zigzag(
    df,
    deviation=0.5,
    depth=10
)

print()

print("TOTAL PIVOTS:")
print(len(pivots))

print()

for p in pivots[-10:]:

    print(p)

mt5.shutdown()
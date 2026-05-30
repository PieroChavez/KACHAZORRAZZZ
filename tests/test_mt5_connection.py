import MetaTrader5 as mt5

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))



from core.config import (
    MT5_LOGIN,
    MT5_PASSWORD,
    MT5_SERVER,
    MT5_PATH,
    DEFAULT_SYMBOL
)


print("=" * 50)
print("PRUEBA DE CONEXIÓN MT5")
print("=" * 50)

# Inicializar terminal
if MT5_PATH:

    connected = mt5.initialize(path=MT5_PATH)

else:

    connected = mt5.initialize()

if not connected:

    print("❌ Error al inicializar MT5")
    print("Error:", mt5.last_error())
    quit()

print("✅ Terminal MT5 iniciada")

# Login
authorized = mt5.login(
    login=MT5_LOGIN,
    password=MT5_PASSWORD,
    server=MT5_SERVER
)

if not authorized:

    print("❌ Error de login")
    print("Error:", mt5.last_error())

    mt5.shutdown()
    quit()

print("✅ Login correcto")

# Información de cuenta
account = mt5.account_info()

if account:

    print("\nDATOS DE LA CUENTA")
    print("-" * 30)
    print("Login:", account.login)
    print("Servidor:", account.server)
    print("Balance:", account.balance)
    print("Equity:", account.equity)
    print("Moneda:", account.currency)

else:

    print("❌ No se pudo obtener account_info()")

# Verificar símbolo
print("\nVERIFICANDO SÍMBOLO")
print("-" * 30)

symbol = mt5.symbol_info(DEFAULT_SYMBOL)

if symbol:

    print("✅ Símbolo encontrado:", DEFAULT_SYMBOL)
    print("Bid:", symbol.bid)
    print("Ask:", symbol.ask)
    print("Spread:", symbol.spread)

else:

    print(f"❌ El símbolo {DEFAULT_SYMBOL} no existe")

# Obtener último tick
tick = mt5.symbol_info_tick(DEFAULT_SYMBOL)

if tick:

    print("\nÚLTIMO TICK")
    print("-" * 30)
    print("Bid:", tick.bid)
    print("Ask:", tick.ask)

# Cerrar conexión
mt5.shutdown()

print("\n🔌 Conexión cerrada correctamente")
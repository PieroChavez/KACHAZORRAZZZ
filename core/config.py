"""
Configuración y variables de entorno
"""
import os
from dotenv import load_dotenv

load_dotenv()

MT5_LOGIN = int(os.getenv("MT5_LOGIN", 0))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")
MT5_PATH = os.getenv("MT5_PATH", "")  # ruta al terminal64.exe si es necesario

DEFAULT_SYMBOL = os.getenv("DEFAULT_SYMBOL", "XAUUSDm")  # Exness usa XAUUSDm, otros brokers pueden usar XAUUSD
DEFAULT_TIMEFRAME = os.getenv("DEFAULT_TIMEFRAME", "H1")
DEFAULT_BARS = int(os.getenv("DEFAULT_BARS", 500))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = "logs"



# execution_config.py o en core/config.py
SCALED_ORDERS = {
    "enabled": True,
    "num_orders": 3,
    "total_lot": 0.03,
    "rr_ratios": [1, 2, 3],
    "breakeven_buffer": 0.3,  # Para XAUUSD (ajustar según símbolo)
    "manage_on_tp1": True
}
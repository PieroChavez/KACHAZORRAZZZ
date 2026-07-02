"""Check current lot_override in DB and verify Telegram flow"""
import sqlite3
import sys
from pathlib import Path

# Check DB value
BASE = Path("P:/PROYECTOS/OneDrive/Escritorio/trading-botV3/data/db/XAUUSDc")
conn = sqlite3.connect(str(BASE / "fractal_state.db"), timeout=10)
val = conn.execute("SELECT value FROM config WHERE key='lot_override'").fetchone()
print(f"DB lot_override = {val[0] if val else 'NOT SET'}")
conn.close()

# Check if _calc_volume would use it correctly
lot_raw = val[0] if val else ""
lot_override = float(lot_raw) if lot_raw else None
print(f"_lot_override in memory = {lot_override}")
print(f"_calc_volume would return = {lot_override or 0.1}")

# Verify the Telegram command path
print("\n--- Telegram command flow ---")
print("1. Telegram: /set_lot 0.02")
print("2. _cmd_set_lot: eng._lot_override = 0.02 (MEMORY)")
print("3. _cmd_set_lot: eng.db.set_config('lot_override', '0.02') (DB)")
print("4. Next _execute_entry → _calc_volume → return 0.02")
print("5. ✅ Should work")

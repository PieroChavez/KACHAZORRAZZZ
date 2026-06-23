import sqlite3
from pathlib import Path
from datetime import datetime, timezone

db_path = Path(__file__).parent.parent / "data" / "trading_state.db"
logs_dir = Path(__file__).parent.parent / "logs"

print("=" * 60)
print("REPORTE DE OPERACIONES - ÚLTIMAS 4 HORAS")
print("=" * 60)

now = datetime.now(timezone.utc)
cutoff = now.timestamp() - 4 * 3600
cutoff_dt = datetime.fromtimestamp(cutoff, tz=timezone.utc)
print(f"Periodo: {cutoff_dt.strftime('%H:%M UTC')} -> {now.strftime('%H:%M UTC')}")
print()

# 1. Leer la DB de estado
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [t[0] for t in cursor.fetchall()]

for table in tables:
    cursor.execute(f'SELECT * FROM "{table}"')
    rows = cursor.fetchall()
    col_names = [d[0] for d in cursor.description]
    print(f"\n[DB] {table}:")
    for r in rows:
        print(f"  {dict(zip(col_names, r))}")

conn.close()

# 2. Buscar logs de las últimas 4h
print("\n" + "=" * 60)
print("LOGS DE LAS ÚLTIMAS 4 HORAS")
print("=" * 60)

trades = []

for log_file in sorted(logs_dir.glob("trading_bot_*.log")):
    # Parse timestamp from filename: trading_bot_2026-05-25_02-33-02_253963.log
    parts = log_file.stem.split("_")
    if len(parts) >= 4:
        date_part = parts[2]
        time_part = parts[3]
        try:
            log_dt = datetime.strptime(f"{date_part}_{time_part}", "%Y-%m-%d_%H-%M-%S")
            log_dt = log_dt.replace(tzinfo=timezone.utc)
            if log_dt.timestamp() < cutoff:
                continue
        except:
            continue
    
    with open(log_file) as f:
        for line in f:
            line = line.strip()
            # Orden colocada
            if "| INFO" in line and "Pending LIMIT" in line:
                trades.append(("PENDIENTE", line))
            # Orden ejecutada/llena
            if "| INFO" in line and "Order executed" in line:
                trades.append(("EJECUTADA", line))
            if "| INFO" in line and "Bare order executed" in line:
                trades.append(("EJECUTADA", line))
            # Posición cerrada
            if "| INFO" in line and "closed successfully" in line:
                trades.append(("CERRADA", line))
            # Scale-out
            if "| INFO" in line and "Scale-out" in line:
                trades.append(("SCALE-OUT", line))
            # Señal generada
            if "| INFO" in line and "Signal:" in line and "Score:" in line:
                trades.append(("SEÑAL", line))
            # Order cancelada (no por POI)
            if "| INFO" in line and "Pending order" in line and "cancelled" in line:
                trades.append(("CANCELADA", line))
            # POI invalidated
            if "POI" in line and "invalidated" in line:
                trades.append(("INVALIDADA", line))

# Procesar señales por activo
signals = {"XAUUSDc": {"SELL": 0, "BUY": 0}, "XAGUSDm": {"SELL": 0, "BUY": 0}}
orders_placed = {"XAUUSDc": 0, "XAGUSDm": 0}
orders_cancelled = {"XAUUSDc": 0, "XAGUSDm": 0}

for ttype, line in trades:
    if ttype == "SEÑAL":
        for sym in ["XAUUSDc", "XAGUSDm"]:
            if f"[{sym}]" in line:
                if "BUY" in line:
                    signals[sym]["BUY"] += 1
                elif "SELL" in line:
                    signals[sym]["SELL"] += 1
    elif ttype == "PENDIENTE":
        for sym in ["XAUUSDc", "XAGUSDm"]:
            if sym in line:
                orders_placed[sym] += 1
    elif ttype == "CANCELADA":
        for sym in ["XAUUSDc", "XAGUSDm"]:
            if sym in line:
                orders_cancelled[sym] += 1

for sym in ["XAUUSDc", "XAGUSDm"]:
    print(f"\n--- {sym} ---")
    print(f"  Señales generadas: {signals[sym]['SELL']} SELL / {signals[sym]['BUY']} BUY")
    print(f"  Órdenes LIMIT colocadas: {orders_placed[sym]}")
    print(f"  Órdenes canceladas: {orders_cancelled[sym]}")
    net = orders_placed[sym] - orders_cancelled[sym]
    print(f"  Órdenes activas/netas: {net}")

# Posiciones actuales en MT5
print("\n" + "=" * 60)
print("POSICIONES ABIERTAS AHORA")
print("=" * 60)
import MetaTrader5 as mt5
if mt5.initialize():
    positions = mt5.positions_get()
    if positions:
        for p in positions:
            print(f"  Ticket: {p.ticket} | {p.symbol} | {'BUY' if p.type==0 else 'SELL'} | Vol: {p.volume} | "
                  f"Open: {p.price_open:.5f} | Current: {p.price_current:.5f} | "
                  f"SL: {p.sl} | TP: {p.tp} | Profit: ${p.profit:.2f}")
        total_pnl = sum(p.profit for p in positions)
        print(f"  P/L Total: ${total_pnl:.2f}")
    else:
        print("  No hay posiciones abiertas")
    mt5.shutdown()
else:
    print("  No se pudo conectar a MT5")

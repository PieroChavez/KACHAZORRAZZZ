import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta
from collections import defaultdict

mt5.initialize()
now = datetime.now(timezone.utc)
start = now - timedelta(days=7)
deals = mt5.history_deals_get(int(start.timestamp()), int(now.timestamp()))

if not deals:
    print("No deals")
    mt5.shutdown()
    exit()

# By symbol + direction + date
summary = defaultdict(lambda: {"count": 0, "profit": 0.0, "buy": 0, "sell": 0, "buy_pnl": 0.0, "sell_pnl": 0.0})
by_day = defaultdict(lambda: defaultdict(float))

for d in deals:
    if not d.symbol or d.profit == 0:
        continue
    sym = d.symbol
    day = datetime.fromtimestamp(d.time).strftime("%m-%d %a")
    summary[sym]["count"] += 1
    summary[sym]["profit"] += d.profit
    by_day[day][sym] += d.profit

print("=" * 75)
print("P&L POR ACTIVO Y DIA")
print("=" * 75)
for day in sorted(by_day):
    line = f"  {day}:"
    for sym in sorted(by_day[day]):
        p = by_day[day][sym]
        line += f"  {sym}=${p:+.2f}"
    print(line)

print(f"\n{'='*75}")
print("RESUMEN TOTAL")
print("="*75)
total = 0
for sym, data in sorted(summary.items()):
    total += data["profit"]
    print(f"  {sym}: {data['count']} trades, ${data['profit']:+.2f}")
print(f"  {'='*50}")
print(f"  TOTAL: ${total:+.2f}")

# Show batch fills (same price, same time ≈ batch trigger)
print(f"\n{'='*75}")
print("ÓRDENES SIMULTÁNEAS (BATCHES) - XAUUSDc")
print("="*75)
xau = [d for d in deals if d.symbol == "XAUUSDc" and d.profit != 0]
from collections import Counter
time_groups = Counter()
for d in xau:
    key = datetime.fromtimestamp(d.time).strftime("%H:%M:%S")
    time_groups[key] += 1

for t, count in sorted(time_groups.items(), key=lambda x: -x[1])[:15]:
    print(f"  {t}: {count} órdenes ejecutadas simultáneamente")

# Total commission + swap
print(f"\n{'='*75}")
print("COSTOS TOTALES")
print("="*75)
commission = sum(d.commission for d in deals if d.symbol)
swap = sum(d.swap for d in deals if d.symbol)
print(f"  Commission: ${commission:.2f}")
print(f"  Swap: ${swap:.2f}")
print(f"  Total costs: ${commission + swap:.2f}")

mt5.shutdown()

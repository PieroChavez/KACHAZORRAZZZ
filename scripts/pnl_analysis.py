import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta
from collections import defaultdict

mt5.initialize()
now = datetime.now(timezone.utc)
start = now - timedelta(days=7)
start_ts = int(start.timestamp())
end_ts = int(now.timestamp())

deals = mt5.history_deals_get(start_ts, end_ts)
if not deals:
    print("No deals found")
    mt5.shutdown()
    exit()

# Group by symbol
by_symbol = defaultdict(lambda: {"count": 0, "profit": 0.0, "buy_vol": 0.0, "sell_vol": 0.0, "deals": []})
for d in deals:
    if not d.symbol:
        continue
    sym = d.symbol
    by_symbol[sym]["count"] += 1
    by_symbol[sym]["profit"] += d.profit
    vol = d.volume if d.profit != 0 else 0
    by_symbol[sym]["deals"].append(d)

print("=" * 70)
print("RESUMEN POR ACTIVO (7 días)")
print("=" * 70)
total_pnl = 0
for sym, data in sorted(by_symbol.items()):
    pnl = data["profit"]
    total_pnl += pnl
    print(f"\n{sym}:")
    print(f"  Deals: {data['count']}")
    print(f"  P&L: ${pnl:.2f}")

print(f"\n{'=' * 70}")
print(f"P&L TOTAL: ${total_pnl:.2f}")
print(f"{'=' * 70}")

# Deep dive on XAUUSDc
xau = by_symbol.get("XAUUSDc")
if xau:
    print(f"\n{'=' * 70}")
    print("ANÁLISIS PROFUNDO XAUUSDc")
    print("=" * 70)
    
    # Separate buy/sell
    buys = [d for d in xau["deals"] if d.profit != 0 and d.type == 0]
    sells = [d for d in xau["deals"] if d.profit != 0 and d.type == 1]
    
    print(f"  BUY trades: {len(buys)}")
    print(f"  SELL trades: {len(sells)}")
    
    # Group by price level
    by_price = defaultdict(lambda: {"count": 0, "profit": 0.0, "volume": 0.0})
    for d in xau["deals"]:
        if d.profit != 0:
            key = round(d.price, 0)
            by_price[key]["count"] += 1
            by_price[key]["profit"] += d.profit
            by_price[key]["volume"] += d.volume
    
    print(f"\n  Desglose por nivel de precio:")
    for price, data in sorted(by_price.items()):
        print(f"    ~{price:.0f}: {data['count']} deals, {data['volume']:.2f} vol, ${data['profit']:.2f}")
    
    # Show worst trades
    losing = sorted([d for d in xau["deals"] if d.profit < 0], key=lambda d: d.profit)
    print(f"\n  Peores trades ({len(losing)} perdedores):")
    for d in losing[:5]:
        dt = datetime.fromtimestamp(d.time)
        print(f"    ${d.profit:.2f} | Vol:{d.volume} | @ {d.price:.2f} | {dt}")

# Show all non-zero profit deals grouped by symbol and date
print(f"\n{'=' * 70}")
print("OPERACIONES CON P&L NO CERO POR DÍA")
print("=" * 70)
by_date = defaultdict(lambda: defaultdict(float))
for sym, data in by_symbol.items():
    for d in data["deals"]:
        if d.profit != 0:
            day = datetime.fromtimestamp(d.time).strftime("%m-%d")
            by_date[day][sym] += d.profit

for day in sorted(by_date):
    row = f"  {day}:"
    for sym in sorted(by_date[day]):
        row += f"  {sym}=${by_date[day][sym]:.2f}"
    print(row)

mt5.shutdown()

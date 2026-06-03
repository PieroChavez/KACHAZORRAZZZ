import MetaTrader5 as mt5
from datetime import datetime, timezone

mt5.initialize()
info = mt5.account_info()
print(f"Balance: ${info.balance}")
print(f"Server: {info.server}")
print(f"Login: {info.login}")

# Get today's history
now = datetime.now(timezone.utc)
from datetime import timedelta
start = now - timedelta(days=7)
start_ts = int(start.timestamp())
end_ts = int(now.timestamp())

deals = mt5.history_deals_get(start_ts, end_ts)
if deals:
    print(f"\nTotal deals (last 7 days): {len(deals)}")
    for d in deals[-20:]:
        print(f"  {d.symbol} {d.volume} @ {d.price} Profit: ${d.profit:.2f} Time: {datetime.fromtimestamp(d.time)}")
else:
    print(f"\nNo deals found in last 7 days")

orders = mt5.history_orders_get(start_ts, end_ts)
if orders:
    print(f"\nTotal orders (last 7 days): {len(orders)}")
    for o in orders[-10:]:
        print(f"  Ticket:{o.ticket} {o.symbol} Type:{o.type} {o.price_open} {datetime.fromtimestamp(o.time_setup)} -> {datetime.fromtimestamp(o.time_done)}")
else:
    print(f"\nNo orders found in last 7 days")

mt5.shutdown()

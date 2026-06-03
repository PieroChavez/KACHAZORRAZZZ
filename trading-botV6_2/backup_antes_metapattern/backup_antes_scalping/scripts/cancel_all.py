import MetaTrader5 as mt5
mt5.initialize()
orders = mt5.orders_get()
if orders:
    print(f"Cancelando {len(orders)} ordenes pendientes...")
    for o in orders:
        req = {"action": mt5.TRADE_ACTION_REMOVE, "order": o.ticket}
        result = mt5.order_send(req)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"  OK Ticket:{o.ticket} {o.symbol}")
        else:
            err = result.comment if result else mt5.last_error()
            print(f"  FAIL Ticket:{o.ticket}: {err}")
    print("Completado")
else:
    print("No hay ordenes pendientes")
mt5.shutdown()

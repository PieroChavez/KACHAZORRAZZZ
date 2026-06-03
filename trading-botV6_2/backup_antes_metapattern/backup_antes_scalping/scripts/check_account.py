import MetaTrader5 as mt5
if mt5.initialize():
    info = mt5.account_info()
    if info:
        print(f"Login: {info.login}")
        print(f"Server: {info.server}")
        print(f"Balance: ${info.balance:.2f}")
        print(f"Equity: ${info.equity:.2f}")
        print(f"Margin: ${info.margin:.2f}")
        print(f"Free Margin: ${info.margin_free:.2f}")
        print(f"Profit: ${info.profit:.2f}")
        term = mt5.terminal_info()
        if term:
            print(f"Terminal: {term.path}")
            print(f"Connected: {term.connected}")
    else:
        print("No account info - not connected?")
        print(f"Last error: {mt5.last_error()}")
    mt5.shutdown()
else:
    print(f"MT5 initialize failed: {mt5.last_error()}")

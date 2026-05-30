import streamlit as st
from streamlit_autorefresh import st_autorefresh

import MetaTrader5 as mt5
import plotly.graph_objects as go

from core.config import (
    MT5_LOGIN,
    MT5_PASSWORD,
    MT5_SERVER,
    MT5_PATH,
    DEFAULT_SYMBOL
)

from data.market_data import MarketDataManager


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="BOT2 Dashboard",
    layout="wide"
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Configuración")

symbol = st.sidebar.selectbox(
    "Símbolo",
    [
        "XAUUSD",
        "EURUSD",
        "BTCUSD"
    ],
    index=0
)

timeframe = st.sidebar.selectbox(
    "Timeframe",
    [
        "M15",
        "H1",
        "H4",
        "D1"
    ],
    index=1
)

bars = st.sidebar.slider(
    "Velas",
    min_value=100,
    max_value=5000,
    value=500,
    step=100
)

refresh_seconds = st.sidebar.slider(
    "Refresh (segundos)",
    min_value=5,
    max_value=60,
    value=10
)

# =========================================================
# AUTO REFRESH
# =========================================================

st_autorefresh(
    interval=refresh_seconds * 1000,
    key="dashboard_refresh"
)

# =========================================================
# TITLE
# =========================================================

st.title("📈 BOT2 Trading Dashboard")

# =========================================================
# MT5 CONNECTION
# =========================================================

if MT5_PATH:

    initialized = mt5.initialize(
        path=MT5_PATH
    )

else:

    initialized = mt5.initialize()

if not initialized:

    st.error(
        f"Error inicializando MT5: {mt5.last_error()}"
    )

    st.stop()

authorized = mt5.login(
    login=MT5_LOGIN,
    password=MT5_PASSWORD,
    server=MT5_SERVER
)

if not authorized:

    st.error(
        f"Error login: {mt5.last_error()}"
    )

    mt5.shutdown()

    st.stop()

# =========================================================
# ACCOUNT INFO
# =========================================================

account = mt5.account_info()

if account is None:

    st.error(
        "No se pudo obtener información de cuenta"
    )

    mt5.shutdown()

    st.stop()

st.subheader("💰 Cuenta")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Balance",
        f"${account.balance:.2f}"
    )

with col2:
    st.metric(
        "Equity",
        f"${account.equity:.2f}"
    )

with col3:
    st.metric(
        "Profit",
        f"${account.profit:.2f}"
    )

with col4:
    st.metric(
        "Margin",
        f"${account.margin:.2f}"
    )

# =========================================================
# MARKET INFO
# =========================================================

st.subheader("📊 Mercado")

market = MarketDataManager(symbol)

real_symbol = market.verify_symbol()

if real_symbol is None:

    st.error(
        f"No se encontró símbolo válido para {symbol}"
    )

    mt5.shutdown()

    st.stop()

tick = mt5.symbol_info_tick(real_symbol)

if tick:

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Bid",
            f"{tick.bid:.3f}"
        )

    with c2:
        st.metric(
            "Ask",
            f"{tick.ask:.3f}"
        )

    with c3:
        st.metric(
            "Spread",
            f"{tick.ask - tick.bid:.3f}"
        )

# =========================================================
# DOWNLOAD DATA
# =========================================================

df = market.get_historical_data(
    timeframe=timeframe,
    bars=bars
)

if df is None or df.empty:

    st.error(
        "No se pudieron descargar datos"
    )

    mt5.shutdown()

    st.stop()

# =========================================================
# FEATURES
# =========================================================

df["ema20"] = (
    df["close"]
    .rolling(20)
    .mean()
)

df["ema50"] = (
    df["close"]
    .rolling(50)
    .mean()
)

# =========================================================
# CHART
# =========================================================

st.subheader(
    f"📈 {real_symbol} - {timeframe}"
)

fig = go.Figure()

fig.add_trace(

    go.Candlestick(

        x=df["time"],

        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],

        name="Precio"
    )
)

fig.add_trace(

    go.Scatter(

        x=df["time"],
        y=df["ema20"],

        name="EMA20"
    )
)

fig.add_trace(

    go.Scatter(

        x=df["time"],
        y=df["ema50"],

        name="EMA50"
    )
)

fig.update_layout(

    height=700,

    xaxis_rangeslider_visible=False,

    legend=dict(
        orientation="h"
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================================
# LAST CANDLE
# =========================================================

st.subheader("🕯 Última vela")

last = df.iloc[-1]

st.json(
    {
        "time": str(last["time"]),
        "open": float(last["open"]),
        "high": float(last["high"]),
        "low": float(last["low"]),
        "close": float(last["close"]),
        "volume": float(last["volume"])
    }
)

# =========================================================
# LAST ROWS
# =========================================================

st.subheader("📄 Últimas 10 velas")

st.dataframe(
    df.tail(10),
    use_container_width=True
)

# =========================================================
# CLOSE MT5
# =========================================================

mt5.shutdown()
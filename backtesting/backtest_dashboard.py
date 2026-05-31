# app.py

import streamlit as st
import MetaTrader5 as mt5
import pandas as pd
import plotly.graph_objects as go

# =====================================
# CONFIG
# =====================================

SYMBOL = "XAUUSDm"
TIMEFRAME = mt5.TIMEFRAME_H1
BARS = 1000

# =====================================
# MT5 CONNECT
# =====================================

def connect_mt5():

    if not mt5.initialize():

        st.error(
            f"Error MT5: {mt5.last_error()}"
        )

        return False

    return True

# =====================================
# LOAD DATA
# =====================================

def load_data():

    rates = mt5.copy_rates_from_pos(
        SYMBOL,
        TIMEFRAME,
        0,
        BARS
    )

    if rates is None:
        st.error("No se pudieron obtener datos desde MT5")
        return pd.DataFrame()

    df = pd.DataFrame(rates)

    if df.empty:
        st.error("No hay datos disponibles")
        return df

    df["time"] = pd.to_datetime(
        df["time"],
        unit="s"
    )

    return df

# =====================================
# SMA
# =====================================

def add_indicators(df):

    df["sma20"] = (
        df["close"]
        .rolling(window=20)
        .mean()
    )

    return df

# =====================================
# SIGNALS
# =====================================

def generate_signals(df):

    df["buy"] = False
    df["sell"] = False

    for i in range(20, len(df)):

        prev_close = df.iloc[i - 1]["close"]
        close = df.iloc[i]["close"]

        prev_sma = df.iloc[i - 1]["sma20"]
        sma = df.iloc[i]["sma20"]

        if pd.isna(prev_sma) or pd.isna(sma):
            continue

        # Cruce alcista
        if prev_close < prev_sma and close > sma:

            df.loc[df.index[i], "buy"] = True

        # Cruce bajista
        elif prev_close > prev_sma and close < sma:

            df.loc[df.index[i], "sell"] = True

    return df

# =====================================
# CHART
# =====================================

def draw_chart(df):

    fig = go.Figure()

    # Velas
    fig.add_trace(
        go.Candlestick(
            x=df["time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=SYMBOL
        )
    )

    # SMA
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["sma20"],
            mode="lines",
            name="SMA20"
        )
    )

    # BUY
    buys = df[df["buy"]]

    fig.add_trace(
        go.Scatter(
            x=buys["time"],
            y=buys["low"],
            mode="markers",
            name="BUY",
            marker=dict(
                size=12,
                color="green",
                symbol="triangle-up"
            )
        )
    )

    # SELL
    sells = df[df["sell"]]

    fig.add_trace(
        go.Scatter(
            x=sells["time"],
            y=sells["high"],
            mode="markers",
            name="SELL",
            marker=dict(
                size=12,
                color="red",
                symbol="triangle-down"
            )
        )
    )

    fig.update_layout(
        title=f"{SYMBOL} - Trading Dashboard",
        height=800,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================
# MAIN
# =====================================

st.set_page_config(
    page_title="XAUUSDm Dashboard",
    layout="wide"
)

st.title("XAUUSDm Trading Dashboard")

if connect_mt5():

    try:

        df = load_data()

        if not df.empty:

            df = add_indicators(df)

            df = generate_signals(df)

            draw_chart(df)

            st.subheader("Últimas 20 velas")

            st.dataframe(
                df.tail(20),
                use_container_width=True
            )

    finally:

        mt5.shutdown()
        
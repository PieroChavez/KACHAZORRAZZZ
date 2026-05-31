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
from strategies.zigzag import detect_zigzag
from strategies.order_block import detect_order_blocks  # ← NUEVO: Import del detector de OBs


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

show_order_blocks = st.sidebar.checkbox(
    "Mostrar Order Blocks",
    value=True
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
# HELPER: DETECTAR BOS DESDE ZIGZAG PIVOTS
# =========================================================

def detect_bos_from_zigzag(df, pivots, lookback=5):
    """
    Detecta Break of Structure (BOS) basándose en pivotes ZigZag.
    
    - BULLISH_BOS: Cuando un HIGH rompe el HIGH anterior significativo
    - BEARISH_BOS: Cuando un LOW rompe el LOW anterior significativo
    """
    bos_list = []
    
    if len(pivots) < 3:
        return bos_list
    
    # Recorrer pivotes buscando rupturas de estructura
    for i in range(2, len(pivots)):
        current = pivots[i]
        prev_significant = None
        
        # Buscar el pivote significativo anterior del tipo opuesto
        for j in range(i - 1, -1, -1):
            if pivots[j]["type"] != current["type"]:
                prev_significant = pivots[j]
                break
        
        if prev_significant is None:
            continue
        
        # Índice de la vela actual en el DataFrame
        try:
            current_idx = df[df["time"] == current["time"]].index[0]
        except IndexError:
            continue
        
        # 🔹 BULLISH_BOS: HIGH actual > HIGH anterior significativo
        if current["type"] == "HIGH":
            if float(current["price"]) > float(prev_significant["price"]):
                bos_list.append({
                    "type": "BULLISH_BOS",
                    "index": int(current_idx),
                    "price": float(current["price"]),
                    "time": current["time"],
                    "broken_level": float(prev_significant["price"])
                })
        
        # 🔹 BEARISH_BOS: LOW actual < LOW anterior significativo  
        elif current["type"] == "LOW":
            if float(current["price"]) < float(prev_significant["price"]):
                bos_list.append({
                    "type": "BEARISH_BOS",
                    "index": int(current_idx),
                    "price": float(current["price"]),
                    "time": current["time"],
                    "broken_level": float(prev_significant["price"])
                })
    
    return bos_list


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
# ZIGZAG
# =========================================================

pivots = detect_zigzag(
    df,
    deviation=0.5,
    depth=10
)

# =========================================================
# BREAK OF STRUCTURE (BOS)
# =========================================================

bos_list = detect_bos_from_zigzag(df, pivots)

# =========================================================
# ORDER BLOCKS
# =========================================================

order_blocks = []

if show_order_blocks and len(bos_list) > 0:
    
    order_blocks = detect_order_blocks(df, bos_list)
    
    # 🔹 Actualizar mitigación: si el precio actual tocó la zona del OB
    if len(order_blocks) > 0 and len(df) > 0:
        current_price = df["close"].iloc[-1]
        for ob in order_blocks:
            if ob["low"] <= current_price <= ob["high"]:
                ob["mitigated"] = True

# =========================================================
# CHART
# =========================================================

st.subheader(
    f"📈 {real_symbol} - {timeframe}"
)

fig = go.Figure()

# =========================================================
# CANDLES
# =========================================================

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

# =========================================================
# EMA20
# =========================================================

fig.add_trace(

    go.Scatter(

        x=df["time"],
        y=df["ema20"],

        mode="lines",

        name="EMA20",
        line=dict(width=1, color="orange")
    )
)

# =========================================================
# EMA50
# =========================================================

fig.add_trace(

    go.Scatter(

        x=df["time"],
        y=df["ema50"],

        mode="lines",

        name="EMA50",
        line=dict(width=1, color="blue")
    )
)

# =========================================================
# ZIGZAG
# =========================================================

if len(pivots) > 0:

    pivot_times = [
        p["time"]
        for p in pivots
    ]

    pivot_prices = [
        float(p["price"])
        for p in pivots
    ]

    fig.add_trace(

        go.Scatter(

            x=pivot_times,

            y=pivot_prices,

            mode="lines+markers",

            name="ZigZag",
            line=dict(width=1, color="purple"),
            marker=dict(size=6)
        )
    )

    for pivot in pivots:

        label = (
            "HH"
            if pivot["type"] == "HIGH"
            else "LL"
        )

        fig.add_annotation(

            x=pivot["time"],

            y=float(
                pivot["price"]
            ),

            text=label,

            showarrow=True,

            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="purple" if pivot["type"] == "HIGH" else "brown",
            bgcolor="white",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=9)
        )

# =========================================================
# ORDER BLOCKS - DIBUJAR ZONAS
# =========================================================

if show_order_blocks and len(order_blocks) > 0:
    
    for ob in order_blocks:
        
        # Colores según tipo y estado de mitigación
        if ob["type"] == "BULLISH_OB":
            color = "rgba(0, 200, 83, 0.15)" if not ob["mitigated"] else "rgba(0, 200, 83, 0.05)"
            border_color = "rgba(0, 200, 83, 0.8)"
            label = "🟢 Bullish OB"
        else:
            color = "rgba(255, 82, 82, 0.15)" if not ob["mitigated"] else "rgba(255, 82, 82, 0.05)"
            border_color = "rgba(255, 82, 82, 0.8)"
            label = "🔴 Bearish OB"
        
        # Añadir rectángulo para la zona del Order Block
        fig.add_vrect(
            x0=ob["time"],
            x1=ob["time"],  # Se extiende hacia la derecha visualmente
            y0=ob["low"],
            y1=ob["high"],
            fillcolor=color,
            line=dict(width=1, color=border_color, dash="dot"),
            annotation_text=label if not ob["mitigated"] else f"{label} ✓",
            annotation_position="top right",
            annotation_font_size=9,
            annotation_font_color=border_color,
            layer="below"
        )
        
        # Añadir línea horizontal de referencia en el centro del OB
        mid_price = (ob["high"] + ob["low"]) / 2
        fig.add_hrect(
            y0=ob["low"],
            y1=ob["high"],
            fillcolor=color,
            line_width=0,
            opacity=0,
            layer="below"
        )

# =========================================================
# LAYOUT DEL GRÁFICO
# =========================================================

fig.update_layout(

    height=850,

    xaxis_rangeslider_visible=False,

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    
    hovermode="x unified",
    
    xaxis_title="Tiempo",
    yaxis_title="Precio",
    
    margin=dict(t=60, b=20, l=20, r=20)
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================================
# PANEL INFERIOR: INFO ADICIONAL
# =========================================================

col_chart_info1, col_chart_info2 = st.columns(2)

# 🔹 Order Blocks Detectados
with col_chart_info1:
    st.subheader("🧱 Order Blocks Detectados")
    
    if show_order_blocks:
        st.write(f"Total OBs: **{len(order_blocks)}**")
        
        if len(order_blocks) > 0:
            # Preparar datos para tabla
            ob_table = []
            for ob in order_blocks[-10:]:  # Últimos 10
                ob_table.append({
                    "Tipo": "🟢 Bullish" if ob["type"] == "BULLISH_OB" else "🔴 Bearish",
                    "Tiempo": str(ob["time"]),
                    "High": f"{ob['high']:.3f}",
                    "Low": f"{ob['low']:.3f}",
                    "Mitigado": "✅ Sí" if ob["mitigated"] else "❌ No"
                })
            
            st.dataframe(
                ob_table,
                use_container_width=True,
                hide_index=True
            )
    else:
        st.info("Activa 'Mostrar Order Blocks' en el sidebar para ver las zonas.")

# 🔹 BOS Detectados
with col_chart_info2:
    st.subheader("⚡ Break of Structure (BOS)")
    st.write(f"Total BOS: **{len(bos_list)}**")
    
    if len(bos_list) > 0:
        bos_table = []
        for bos in bos_list[-10:]:
            bos_table.append({
                "Tipo": "📈 Bullish" if bos["type"] == "BULLISH_BOS" else "📉 Bearish",
                "Tiempo": str(bos["time"]),
                "Precio Ruptura": f"{bos['price']:.3f}",
                "Nivel Roto": f"{bos['broken_level']:.3f}"
            })
        
        st.dataframe(
            bos_table,
            use_container_width=True,
            hide_index=True
        )

# =========================================================
# ZIGZAG INFO
# =========================================================

st.subheader("🔺 ZigZag Pivots")

st.write(
    f"Total pivots detectados: {len(pivots)}"
)

if len(pivots) > 0:

    st.dataframe(
        pivots[-20:],
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
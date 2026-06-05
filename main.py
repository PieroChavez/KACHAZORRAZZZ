import streamlit as st
from streamlit_autorefresh import st_autorefresh

import MetaTrader5 as mt5
import plotly.graph_objects as go
import pandas as pd

from core.config import (
    MT5_LOGIN,
    MT5_PASSWORD,
    MT5_SERVER,
    MT5_PATH,
)

from data.market_data import MarketDataManager
from strategies.kachazorraz import SmartMoneyConcepts


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="BOT2 SMC Dashboard",
    layout="wide"
)

# =========================================================
# SIDEBAR - CONFIG GENERAL
# =========================================================

st.sidebar.title("Configuración")

symbol = "XAUEURm"

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["M15", "H1", "H4", "D1"],
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
    max_value=120,
    value=10
)

st.sidebar.markdown("---")
st.sidebar.markdown("### SMC Parameters")

swing_length = st.sidebar.slider(
    "Swing Length",
    min_value=10,
    max_value=100,
    value=50,
    step=5
)

show_swings = st.sidebar.checkbox("Swings (HH/HL/LH/LL)", value=True)
show_bos = st.sidebar.checkbox("Break of Structure (BOS)", value=True)
show_choch = st.sidebar.checkbox("Change of Character (CHoCH)", value=True)
show_eq = st.sidebar.checkbox("Equal Highs/Lows", value=True)
show_fvg = st.sidebar.checkbox("Fair Value Gaps (FVG)", value=True)
show_order_blocks = st.sidebar.checkbox("Order Blocks", value=True)
show_pd_zones = st.sidebar.checkbox("Premium/Discount Zones", value=True)
show_fibonacci = st.sidebar.checkbox("Fibonacci Automático", value=True)
show_signals = st.sidebar.checkbox("Señales BUY/SELL", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Ejecución de Órdenes")

auto_trade = st.sidebar.checkbox("Trading automático", value=False)
lot_size = st.sidebar.number_input("Lotes", min_value=0.01, max_value=10.0, value=0.01, step=0.01)

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

st.title("BOT2 SMC Trading Dashboard")


# =========================================================
# MT5 CONNECTION
# =========================================================

# =========================================================
# MT5 CONNECTION (con reconexión automática)
# =========================================================

@st.cache_resource
def get_mt5():
    if MT5_PATH:
        ok = mt5.initialize(path=MT5_PATH)
    else:
        ok = mt5.initialize()
    if not ok:
        return None
    auth = mt5.login(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    if not auth:
        mt5.shutdown()
        return None
    return mt5

mt5_instance = get_mt5()
if mt5_instance is None:
    st.error(f"No se pudo conectar a MT5. Verifica que el terminal esté abierto y las credenciales en .env")
    st.stop()

# =========================================================
# ACCOUNT INFO
# =========================================================

account = mt5.account_info()
if account is None:
    st.error("No se pudo obtener información de cuenta")
    st.stop()

st.subheader("Cuenta")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Balance", f"${account.balance:.2f}")

with col2:
    st.metric("Equity", f"${account.equity:.2f}")

with col3:
    st.metric("Profit", f"${account.profit:.2f}")

with col4:
    st.metric("Margin", f"${account.margin:.2f}")

# =========================================================
# MARKET INFO
# =========================================================

st.subheader("Mercado")

market = MarketDataManager(symbol)

real_symbol = market.verify_symbol()

if real_symbol is None:
    st.error(f"No se encontró símbolo válido para {symbol}")
    st.stop()

tick = mt5.symbol_info_tick(real_symbol)

if tick:
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Bid", f"{tick.bid:.3f}")

    with c2:
        st.metric("Ask", f"{tick.ask:.3f}")

    with c3:
        st.metric("Spread", f"{tick.ask - tick.bid:.3f}")

# =========================================================
# SMC ANALYSIS (con try/except para no romper el ciclo)
# =========================================================

try:
    df = market.get_historical_data(
        timeframe=timeframe,
        bars=bars
    )

    if df is None or df.empty:
        st.warning("Esperando datos del mercado...")
        st.stop()

    smc = SmartMoneyConcepts(
        swing_length=swing_length,
        symbol=real_symbol,
        equal_threshold_atr=0.1,
        fvg_show_strength=True,
        require_fvg_for_signal=False,
        require_bos_for_signal=True,
        sl_pips=20.0,
        rr_ratio=2.0,
    )

    smc_results = smc.analyze(df)

    swings = smc_results["swings"]
    bos_list = smc_results["bos_list"]
    choch_list = smc_results["choch_list"]
    eq_list = smc_results["equal_highs_lows"]
    fvg_list = smc_results["fair_value_gaps"]
    obs = smc_results["order_blocks"]
    pd_zone = smc_results["premium_discount"]
    fib_levels = smc_results["fibonacci"]
    signals = smc_results["signals"]
    trend = smc_results["trend"]
    stats = smc_results["stats"]

    # =========================================================
    # ORDER EXECUTION
    # =========================================================

    def place_mt5_order(signal, symbol_name):
        if signal["action"] == "BUY_LIMIT":
            action = mt5.ORDER_TYPE_BUY_LIMIT
            price = signal.get("limit_price", signal["entry_price"])
        elif signal["action"] == "SELL_LIMIT":
            action = mt5.ORDER_TYPE_SELL_LIMIT
            price = signal.get("limit_price", signal["entry_price"])
        else:
            return None

        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol_name,
            "volume": lot_size,
            "type": action,
            "price": price,
            "sl": signal["stop_loss"],
            "tp": signal["take_profit"],
            "deviation": 10,
            "magic": 123456,
            "comment": f"BOT2_{signal['action']}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        return result

    if auto_trade and len(signals) > 0:
        for sig in signals:
            order_result = place_mt5_order(sig, real_symbol)
            if order_result and order_result.retcode == mt5.TRADE_RETCODE_DONE:
                st.sidebar.success(f"Orden {sig['action']} ejecutada @ {sig['entry_price']:.3f}")
            elif order_result:
                st.sidebar.warning(f"Error orden {sig['action']}: {order_result.comment} (código {order_result.retcode})")

    # =========================================================
    # CHART
    # =========================================================

    st.subheader(f"{real_symbol} - {timeframe}")

    fig = go.Figure()

    # =========================================================
    # PREMIUM / DISCOUNT ZONES
    # =========================================================

    if show_pd_zones and pd_zone is not None:
        fig.add_hrect(
            y0=pd_zone.premium_bottom,
            y1=pd_zone.premium_top,
            fillcolor="rgba(255, 82, 82, 0.08)",
            line_width=0,
            layer="below",
            name="Premium"
        )
        fig.add_hrect(
            y0=pd_zone.equilibrium_bottom,
            y1=pd_zone.equilibrium_top,
            fillcolor="rgba(128, 128, 128, 0.08)",
            line_width=0,
            layer="below",
            name="Equilibrium"
        )
        fig.add_hrect(
            y0=pd_zone.discount_bottom,
            y1=pd_zone.discount_top,
            fillcolor="rgba(0, 200, 83, 0.08)",
            line_width=0,
            layer="below",
            name="Discount"
        )
        fig.add_hline(
            y=pd_zone.premium_bottom,
            line=dict(color="red", width=1, dash="dash"),
            layer="below"
        )
        fig.add_hline(
            y=pd_zone.equilibrium_bottom,
            line=dict(color="gray", width=1, dash="dash"),
            layer="below"
        )

    # =========================================================
    # CANDLES
    # =========================================================

    fig.add_trace(
        go.Candlestick(
            x=df["time"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="Precio"
        )
    )

    # =========================================================
    # FAIR VALUE GAPS (FVG)
    # =========================================================

    if show_fvg and len(fvg_list) > 0:
        for fvg in fvg_list:
            if fvg.mitigated and fvg.fvg_type == "FVG":
                continue
            if fvg.bias == "BULLISH":
                color = "rgba(0, 200, 83, 0.20)" if not fvg.mitigated else "rgba(0, 200, 83, 0.05)"
                border = "green"
                label = "FVG" if fvg.fvg_type == "FVG" else "iFVG"
            else:
                color = "rgba(255, 82, 82, 0.20)" if not fvg.mitigated else "rgba(255, 82, 82, 0.05)"
                border = "red"
                label = "FVG" if fvg.fvg_type == "FVG" else "iFVG"
            box = fvg.box
            right_time = box.get("right_time", fvg.time) if isinstance(box, dict) else fvg.time
            fig.add_vrect(
                x0=fvg.time, x1=right_time,
                y0=fvg.bottom, y1=fvg.top,
                fillcolor=color, line=dict(width=1, color=border, dash="dot"),
                annotation_text=f"{label} {fvg.strength:.0f}%" if fvg.strength > 0 else label,
                annotation_position="top left", annotation_font_size=8, layer="below"
            )

    # =========================================================
    # ORDER BLOCKS
    # =========================================================

    if show_order_blocks and len(obs) > 0:
        for ob in obs:
            if ob.ob_type == "BULLISH_OB":
                color = "rgba(0, 200, 83, 0.15)" if not ob.mitigated else "rgba(0, 200, 83, 0.05)"
                border_color = "rgba(0, 200, 83, 0.8)"
                label = "Bullish OB"
            else:
                color = "rgba(255, 82, 82, 0.15)" if not ob.mitigated else "rgba(255, 82, 82, 0.05)"
                border_color = "rgba(255, 82, 82, 0.8)"
                label = "Bearish OB"
            fig.add_vrect(
                x0=ob.time, x1=ob.time,
                y0=ob.low, y1=ob.high,
                fillcolor=color, line=dict(width=1, color=border_color, dash="dot"),
                annotation_text=label if not ob.mitigated else f"{label} ",
                annotation_position="top right", annotation_font_size=9,
                annotation_font_color=border_color, layer="below"
            )

    # =========================================================
    # SWINGS (HH, HL, LH, LL)
    # =========================================================

    if show_swings and len(swings) > 0:
        swing_times = [s.time for s in swings]
        swing_prices = [s.price for s in swings]
        fig.add_trace(
            go.Scatter(
                x=swing_times, y=swing_prices, mode="markers", name="Swings",
                marker=dict(
                    size=8,
                    color=["green" if s.swing_type == "HIGH" else "red" for s in swings],
                    symbol=["triangle-down" if s.swing_type == "HIGH" else "triangle-up" for s in swings],
                    line=dict(width=1, color="black")
                ),
                hovertemplate="<b>%{text}</b><br>Precio: %{y:.3f}<br>",
                text=[f"{s.label}" for s in swings]
            )
        )
        for s in swings:
            if s.label in ("HH", "HL", "LH", "LL"):
                fig.add_annotation(
                    x=s.time, y=s.price, text=s.label,
                    showarrow=True, arrowhead=1, arrowsize=1, arrowwidth=1,
                    arrowcolor="green" if s.swing_type == "HIGH" else "red",
                    bgcolor="white", bordercolor="green" if s.swing_type == "HIGH" else "red",
                    borderwidth=1, font=dict(size=9, color="green" if s.swing_type == "HIGH" else "red")
                )

    # =========================================================
    # BOS (Break of Structure)
    # =========================================================

    if show_bos and len(bos_list) > 0:
        for bos in bos_list:
            color = "green" if bos["type"] == "BULLISH_BOS" else "red"
            fig.add_hline(
                y=bos["broken_level"], line=dict(color=color, width=1, dash="dot"),
                opacity=0.5, layer="below"
            )
            bos_time = bos.get("time")
            if bos_time is not None:
                fig.add_annotation(
                    x=bos_time, y=bos["price"], text="BOS",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor=color,
                    bgcolor="white", bordercolor=color, borderwidth=1, font=dict(size=9, color=color)
                )

    # =========================================================
    # CHoCH (Change of Character)
    # =========================================================

    if show_choch and len(choch_list) > 0:
        for ch in choch_list:
            color = "blue" if ch["type"] == "BULLISH_CHOCH" else "purple"
            ch_time = ch.get("time")
            if ch_time is not None:
                fig.add_annotation(
                    x=ch_time, y=ch["price"], text="CHoCH",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor=color,
                    bgcolor="white", bordercolor=color, borderwidth=1, font=dict(size=10, color=color)
                )

    # =========================================================
    # EQUAL HIGHS / LOWS
    # =========================================================

    if show_eq and len(eq_list) > 0:
        for eq in eq_list:
            eq_time = eq.get("time")
            if eq_time is not None:
                fig.add_annotation(
                    x=eq_time, y=eq["price"], text=eq["type"],
                    showarrow=True, arrowhead=1, arrowsize=1, arrowwidth=1, arrowcolor="orange",
                    bgcolor="white", bordercolor="orange", borderwidth=1, font=dict(size=9, color="orange")
                )

    # =========================================================
    # FIBONACCI
    # =========================================================

    if show_fibonacci and len(fib_levels) > 0:
        max_time = df["time"].iloc[-1]
        fib_colors = {
            0.0: "gray", 0.22: "red", 0.382: "orange", 0.50: "yellow",
            0.72: "green", 0.786: "green", 1.0: "gray", 1.618: "blue",
        }
        for fib in fib_levels:
            color = fib_colors.get(fib["level"], "gray")
            fig.add_hline(
                y=fib["price"], line=dict(color=color, width=1, dash="dash"),
                layer="below", opacity=0.5
            )
            fig.add_annotation(
                x=max_time, y=fib["price"], text=f"{fib['label']}",
                showarrow=False, yanchor="middle", xanchor="left",
                font=dict(size=9, color=color), bgcolor="rgba(255,255,255,0.7)"
            )

    # =========================================================
    # SEÑALES BUY / SELL
    # =========================================================

    if show_signals and len(signals) > 0:
        for sig in signals:
            sig_time = sig.get("time")
            if sig_time is None:
                continue
            if sig["action"] == "BUY_LIMIT":
                color = "green"
                symbol_marker = "triangle-up"
                text = f"BUY LIMIT ({sig['confidence']:.0f}%)"
            else:
                color = "red"
                symbol_marker = "triangle-down"
                text = f"SELL LIMIT ({sig['confidence']:.0f}%)"
            fig.add_trace(
                go.Scatter(
                    x=[sig_time], y=[sig["entry_price"]],
                    mode="markers+text", name=text,
                    marker=dict(size=15, color=color, symbol=symbol_marker, line=dict(width=1, color="black")),
                    text=[text], textposition="top center", textfont=dict(size=10, color=color),
                    hovertemplate=(
                        f"<b>{sig['action']}</b><br>"
                        f"Entry: {sig['entry_price']:.3f}<br>"
                        f"SL: {sig['stop_loss']:.3f}<br>"
                        f"TP: {sig['take_profit']:.3f}<br>"
                        f"Confianza: {sig['confidence']:.0f}%<br>"
                        f"Razón: {sig['reason']}<br>"
                    )
                )
            )

    # =========================================================
    # LAYOUT
    # =========================================================

    fig.update_layout(
        height=850, xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified", xaxis_title="Tiempo", yaxis_title="Precio",
        margin=dict(t=60, b=20, l=20, r=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================================================
    # PANEL INFERIOR
    # =========================================================

    col_info1, col_info2, col_info3 = st.columns(3)

    with col_info1:
        st.subheader("Smart Money Concepts")
        st.write(f"Tendencia: **{trend['label']}**")
        st.write(f"Swings: **{stats['total_pivots']}**")
        st.write(f"BOS: **{stats['total_bos']}**")
        st.write(f"CHoCH: **{stats['total_choch']}**")
        st.write(f"FVG: **{stats['total_fvg']}**")
        st.write(f"Order Blocks: **{stats['total_obs']}**")
        st.write(f"EQH/EQL: **{stats['total_eq']}**")
        st.write(f"Señales: **{stats['total_signals']}**")

    with col_info2:
        st.subheader("Señales de Trading")
        if len(signals) > 0:
            signal_data = []
            for sig in signals:
                signal_data.append({
                    "Acción": "BUY LIMIT" if sig["action"] == "BUY_LIMIT" else "SELL LIMIT",
                    "Precio": f"{sig['entry_price']:.3f}",
                    "SL": f"{sig['stop_loss']:.3f}",
                    "TP": f"{sig['take_profit']:.3f}",
                    "Confianza": f"{sig['confidence']:.0f}%",
                    "Razón": sig["reason"]
                })
            st.dataframe(signal_data, use_container_width=True, hide_index=True)
        else:
            st.info("No hay señales activas. Espera a que se cumplan las condiciones SMC.")

    with col_info3:
        st.subheader("Order Blocks Detectados")
        if show_order_blocks and len(obs) > 0:
            ob_data = []
            for ob in obs[-10:]:
                ob_data.append({
                    "Tipo": "Bullish" if ob.ob_type == "BULLISH_OB" else "Bearish",
                    "Tiempo": str(ob.time), "High": f"{ob.high:.3f}",
                    "Low": f"{ob.low:.3f}", "Mitigado": "Sí" if ob.mitigated else "No"
                })
            st.dataframe(ob_data, use_container_width=True, hide_index=True)
        else:
            st.info("Activa Order Blocks en el sidebar.")

    # =========================================================
    # SWINGS TABLE
    # =========================================================

    st.subheader("Swings (HH/HL/LH/LL)")
    if len(swings) > 0:
        swing_data = []
        for s in swings[-30:]:
            swing_data.append({
                "Tipo": s.swing_type, "Label": s.label, "Precio": f"{s.price:.3f}",
                "Fuerza": "Strong" if s.is_strong else "Weak" if s.is_weak else "Normal",
                "Tiempo": str(s.time)
            })
        st.dataframe(swing_data, use_container_width=True, hide_index=True)
    else:
        st.info("No se detectaron swings.")

    # =========================================================
    # FVG TABLE
    # =========================================================

    st.subheader("Fair Value Gaps (FVG)")
    if show_fvg and len(fvg_list) > 0:
        fvg_data = []
        for f in fvg_list[-20:]:
            fvg_data.append({
                "Tipo": f.fvg_type, "Bias": f.bias, "Top": f"{f.top:.3f}",
                "Bottom": f"{f.bottom:.3f}", "Fuerza": f"{f.strength:.1f}%",
                "Mitigado": "Sí" if f.mitigated else "No", "Tiempo": str(f.time)
            })
        st.dataframe(fvg_data, use_container_width=True, hide_index=True)
    else:
        st.info("Activa FVG en el sidebar.")

    # =========================================================
    # FIBONACCI TABLE
    # =========================================================

    if show_fibonacci and len(fib_levels) > 0:
        st.subheader("Fibonacci Automático")
        fib_data = []
        for f in fib_levels:
            fib_data.append({"Nivel": f["label"], "Precio": f"{f['price']:.3f}"})
        st.dataframe(fib_data, use_container_width=True, hide_index=True)

    # =========================================================
    # BOS TABLE
    # =========================================================

    st.subheader("Break of Structure (BOS)")
    if len(bos_list) > 0:
        bos_data = []
        for b in bos_list[-15:]:
            bos_data.append({
                "Tipo": "Bullish" if b["type"] == "BULLISH_BOS" else "Bearish",
                "Precio": f"{b['price']:.3f}", "Nivel Roto": f"{b['broken_level']:.3f}",
                "Tiempo": str(b["time"])
            })
        st.dataframe(bos_data, use_container_width=True, hide_index=True)
    else:
        st.info("No se detectaron BOS.")

    # =========================================================
    # LAST CANDLE
    # =========================================================

    st.subheader("Última vela")
    last = df.iloc[-1]
    st.json({
        "time": str(last["time"]), "open": float(last["open"]),
        "high": float(last["high"]), "low": float(last["low"]),
        "close": float(last["close"]), "volume": float(last["volume"])
    })

    # =========================================================
    # LAST ROWS
    # =========================================================

    st.subheader("Últimas 10 velas")
    st.dataframe(df.tail(10), use_container_width=True)

except Exception as e:
    st.warning(f"Error en análisis SMC: {e}")

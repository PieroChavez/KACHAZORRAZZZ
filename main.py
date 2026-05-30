"""
Trading Bot Profesional
Multi-TimeFrame + Limit Orders a 30 pips + 3 TPs escalados + Breakeven + 80s loop
"""

import threading
import time
from datetime import datetime, timedelta

import MetaTrader5 as mt5
from core.logger import logger

from data.connector import MT5Connector
from data.market_data import MarketData
from execution.orders import place_multiple_orders, place_limit_order
from models.predict import TradingModelPredictor
from strategy.entries import detect_limit_entry
from training.dataset import get_signal_performance, log_mt5_orders, log_trade

from core.config import (
    DEFAULT_SYMBOL,
    DEFAULT_BARS
)

# =========================================================
# CONFIGURACIÓN DE ESTRATEGIA
# =========================================================

MIN_HISTORY_TRADES = 6
MIN_WINRATE = 52.0
PULLBACK_PIPS = 30          # ✅ Distancia para orden LIMIT (30 pips)
PIP_VALUE = 0.10            # Valor de 1 pip para XAUUSD (ajustar según símbolo)
RISK_PIPS = 40              # Stop Loss en pips desde la entrada LIMIT
RR_RATIOS = [1, 2, 3]       # Take Profit escalados: 1:1, 1:2, 1:3
LOT_PER_ORDER = 0.01        # Lotaje por cada una de las 3 órdenes
BREAKEVEN_BUFFER = 0.3      # Buffer para evitar slippage al mover a BE

# =========================================================
# GLOBAL STOP EVENT
# =========================================================

stop_event = threading.Event()


# =========================================================
# STOP CONSOLE
# =========================================================

def _console_stop_listener():
    logger.info("Escribe 'STOP' y presiona ENTER para detener.")
    while not stop_event.is_set():
        try:
            command = input().strip().lower()
            if command == "stop":
                logger.warning("Deteniendo bot...")
                stop_event.set()
        except Exception:
            break


# =========================================================
# MULTI TIMEFRAME STRATEGY
# =========================================================

def generate_signal(df_h1, df_m15, df_m2):
    """Estrategia simple multi-timeframe"""
    if len(df_h1) < 2 or len(df_m15) < 2 or len(df_m2) < 2:
        logger.warning("No hay suficientes velas.")
        return None

    h1_last = df_h1.iloc[-1]
    h1_bullish = h1_last["close"] > h1_last["open"]
    h1_bearish = h1_last["close"] < h1_last["open"]

    m15_last = df_m15.iloc[-1]
    m15_bullish = m15_last["close"] > m15_last["open"]
    m15_bearish = m15_last["close"] < m15_last["open"]

    m2_last = df_m2.iloc[-1]
    m2_bullish = m2_last["close"] > m2_last["open"]
    m2_bearish = m2_last["close"] < m2_last["open"]

    if h1_bullish and m15_bullish and m2_bullish:
        return "BUY"
    if h1_bearish and m15_bearish and m2_bearish:
        return "SELL"
    return None


def _has_higher_high_low(df):
    if len(df) < 3:
        return False
    return df["high"].iloc[-1] > df["high"].iloc[-2] and df["low"].iloc[-1] > df["low"].iloc[-2]


def _has_lower_low_high(df):
    if len(df) < 3:
        return False
    return df["low"].iloc[-1] < df["low"].iloc[-2] and df["high"].iloc[-1] < df["high"].iloc[-2]


def _is_pullback_valid(signal, df_m15, df_m2):
    if signal not in ("BUY", "SELL"):
        return False
    if len(df_m15) < 3 or len(df_m2) < 4:
        return False

    if signal == "BUY":
        if not _has_higher_high_low(df_m15):
            return False
        return (
            df_m2["close"].iloc[-1] < df_m2["close"].iloc[-2] and
            df_m2["low"].iloc[-1] > df_m2["low"].iloc[-2] and
            df_m2["low"].iloc[-1] > df_m2["low"].iloc[-3]
        )
    return (
        df_m2["close"].iloc[-1] > df_m2["close"].iloc[-2] and
        df_m2["high"].iloc[-1] < df_m2["high"].iloc[-2] and
        df_m2["high"].iloc[-1] < df_m2["high"].iloc[-3] and
        _has_lower_low_high(df_m15)
    )


def _is_signal_allowed(signal):
    perf = get_signal_performance(signal, min_trades=MIN_HISTORY_TRADES)
    if perf["total"] == 0:
        logger.warning(f"No hay historial de trades para {signal}. Se permitirá con precaución.")
        return True
    if perf["enough_data"] and perf["winrate"] < MIN_WINRATE:
        logger.warning(f"Señal {signal} tiene winrate bajo ({perf['winrate']}%, {perf['total']} trades). Bloqueando.")
        return False
    logger.success(f"Señal {signal} viable: winrate={perf['winrate']}% sobre {perf['total']} trades.")
    return True


# =========================================================
# FUNCIONES AUXILIARES: ENTRADA LIMIT + ÓRDENES ESCALADAS + BREAKEVEN
# =========================================================

def calculate_limit_entry(current_price: float, signal: str, pips: int = PULLBACK_PIPS, pip_value: float = PIP_VALUE) -> float:
    """
    Calcula precio de entrada LIMIT a X pips del precio actual para capturar retroceso.
    
    BUY  → Limit por DEBAJO del precio (esperar bajada)
    SELL → Limit por ARRIBA del precio (esperar subida)
    """
    offset = pips * pip_value
    if signal == "BUY":
        return round(current_price - offset, 2)
    else:
        return round(current_price + offset, 2)


def calculate_tp_scaled(entry: float, sl: float, signal: str, ratio: float) -> float:
    """Calcula TP con ratio Riesgo:Beneficio"""
    risk = abs(entry - sl)
    if signal == "BUY":
        return round(entry + (risk * ratio), 2)
    else:
        return round(entry - (risk * ratio), 2)


def move_sl_to_breakeven(ticket: int, entry: float, signal: str, buffer: float = BREAKEVEN_BUFFER) -> bool:
    """Mueve SL a Breakeven + buffer para una orden específica"""
    try:
        new_sl = entry + buffer if signal == "BUY" else entry - buffer
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": DEFAULT_SYMBOL,
            "position": ticket,
            "sl": round(new_sl, 2),
            "tp": 0  # Mantener TP original
        }
        result = mt5.OrderSend(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.success(f"🛡️ Breakeven aplicado en ticket {ticket} @ {new_sl}")
            return True
        else:
            logger.warning(f"⚠️ No se pudo aplicar BE: {result.comment}")
    except Exception as e:
        logger.error(f"❌ Error en move_sl_to_breakeven(): {e}")
    return False


def check_and_apply_breakeven(orders_list: list, signal: str):
    """Verifica si TP1 se cerró y aplica BE a las órdenes restantes"""
    try:
        for order in orders_list:
            if order.get("rr_ratio") == 1:  # Es la orden TP1
                pending = mt5.OrdersGet(ticket=order["ticket"])
                position = mt5.PositionsGet(ticket=order["ticket"])
                
                if pending is None and position is None:
                    # ✅ TP1 ya se cerró → aplicar BE a las otras
                    logger.info(f"🎯 TP1 cerrado en ticket {order['ticket']}. Aplicando BE a órdenes restantes...")
                    for other in orders_list:
                        if other["rr_ratio"] > 1:
                            pos = mt5.PositionsGet(ticket=other["ticket"])
                            if pos:
                                move_sl_to_breakeven(
                                    other["ticket"],
                                    order["entry"],
                                    signal,
                                    buffer=BREAKEVEN_BUFFER
                                )
                    return True
    except Exception as e:
        logger.error(f"❌ Error en check_and_apply_breakeven(): {e}")
    return False


# =========================================================
# MAIN
# =========================================================

def main():
    logger.info("🚀 Trading Bot: Limit 30 pips + 3 TPs escalados + BE + 80s loop")

    # =====================================================
    # CONNECT MT5
    # =====================================================
    connector = MT5Connector()
    if not connector.connect():
        logger.error("❌ No se pudo conectar a MT5")
        return

    # =====================================================
    # LOAD MODEL (opcional - fallback a técnica si falla)
    # =====================================================
    predictor = TradingModelPredictor()
    try:
        predictor.load_model()
    except Exception as e:
        logger.warning(f"⚠️ Modelo no cargado ({e}). Usando estrategia técnica.")
        predictor.loaded = False

    # =====================================================
    # STOP THREAD
    # =====================================================
    stop_thread = threading.Thread(target=_console_stop_listener, daemon=True)
    stop_thread.start()

    # =====================================================
    # MARKET DATA
    # =====================================================
    market_data = MarketData(DEFAULT_SYMBOL)
    timeframes = ["H1", "M15", "M2"]

    # =====================================================
    # LOOP PRINCIPAL
    # =====================================================
    try:
        while not stop_event.is_set():
            try:
                logger.info("📥 Descargando datos...")

                datasets = market_data.get_multi_timeframes(timeframes=timeframes, bars=DEFAULT_BARS)
                df_h1 = datasets.get("H1")
                df_m15 = datasets.get("M15")
                df_m2 = datasets.get("M2")

                if df_h1 is None or df_m15 is None or df_m2 is None:
                    logger.error("❌ Faltan temporalidades.")
                    time.sleep(10)
                    continue

                market_data.save_dataset(df_h1, "H1")
                market_data.save_dataset(df_m15, "M15")
                market_data.save_dataset(df_m2, "M2")

                # Sincronizar órdenes cerradas
                try:
                    closed_orders = connector.get_closed_orders(
                        start=datetime.now() - timedelta(days=1),
                        end=datetime.now(),
                        symbol=DEFAULT_SYMBOL
                    )
                    added = log_mt5_orders(closed_orders)
                    if added > 0:
                        logger.success(f"✅ {added} operaciones MT5 añadidas al dataset.")
                except Exception as e:
                    logger.exception(f"❌ Error sincronizando operaciones MT5: {e}")

                # Mostrar precios
                current_price = df_m2.iloc[-1]["close"]
                logger.info(f"📊 Precio actual {DEFAULT_SYMBOL}: {current_price}")
                logger.info(f"📊 H1 CLOSE: {df_h1.iloc[-1]['close']}")
                logger.info(f"📊 M15 CLOSE: {df_m15.iloc[-1]['close']}")

                # Generar señal
                model_prediction = None
                if predictor.loaded:
                    model_prediction = predictor.predict(df_m15)

                if model_prediction and model_prediction["signal"] != "HOLD":
                    signal = model_prediction["signal"]
                    logger.info(f"🤖 SEÑAL (MODELO): {signal} | confianza={model_prediction['confidence']:.2f}")
                else:
                    if model_prediction:
                        logger.info("🤖 Modelo: HOLD. Usando estrategia técnica.")
                    signal = generate_signal(df_h1, df_m15, df_m2)
                    logger.info(f"📈 SEÑAL (ESTRATEGIA): {signal}")

                # Validar pullback técnico (opcional, se puede desactivar si usas solo limit entry)
                if signal is not None:
                    if not _is_pullback_valid(signal, df_m15, df_m2):
                        logger.info(f"⚠️ Señal {signal} sin retroceso técnico, pero usando entrada LIMIT a {PULLBACK_PIPS} pips.")
                    else:
                        logger.success(f"✅ Retroceso válido para {signal}.")

                # =========================================
                # EXECUTE ORDER - LIMIT ENTRY A 30 PIPS + 3 TPs ESCALADOS
                # =========================================

                if signal is not None and _is_signal_allowed(signal):
                    logger.success(f"🎯 Señal válida: {signal}. Calculando entrada LIMIT a {PULLBACK_PIPS} pips...")

                    # 🔹 Calcular entrada LIMIT para capturar retroceso
                    limit_entry = calculate_limit_entry(current_price, signal)
                    sl_price = calculate_limit_entry(limit_entry, signal, pips=RISK_PIPS)  # SL a 40 pips de la entrada LIMIT
                    
                    logger.info(f"📍 Precio actual: {current_price}")
                    logger.info(f"📍 Entrada LIMIT ({signal}): {limit_entry} ({PULLBACK_PIPS} pips)")
                    logger.info(f"📍 Stop Loss: {sl_price} ({RISK_PIPS} pips de riesgo)")

                    # 📦 Colocar 3 órdenes con mismo entry/SL pero TPs escalonados
                    orders_placed = []
                    
                    for i, rr in enumerate(RR_RATIOS):
                        tp_price = calculate_tp_scaled(limit_entry, sl_price, signal, rr)
                        
                        order = place_limit_order(
                            symbol=DEFAULT_SYMBOL,
                            order_type=signal,
                            lot=LOT_PER_ORDER,
                            price=limit_entry,
                            sl=sl_price,
                            tp=tp_price,
                            comment=f"bot_limit_{signal}_RR{rr}"
                        )
                        
                        if order:
                            order["rr_ratio"] = rr
                            order["entry"] = limit_entry
                            orders_placed.append(order)
                            logger.info(f"✅ Orden #{i+1} (RR 1:{rr}) | Entry: {limit_entry} | TP: {tp_price} | SL: {sl_price}")
                    
                    # 📝 Guardar en dataset
                    if orders_placed:
                        try:
                            for ord_info in orders_placed:
                                log_trade({
                                    "symbol": DEFAULT_SYMBOL,
                                    "timeframe": "M15",
                                    "direction": signal,
                                    "setup_type": "limit_pullback_3tp",
                                    "entry_reason": f"limit_{PULLBACK_PIPS}pips_pullback",
                                    "has_fvg": False,
                                    "current_price": current_price,
                                    "entry_price": ord_info["price"],
                                    "sl": ord_info["sl"],
                                    "tp": ord_info["tp"],
                                    "rr_ratio": ord_info["rr_ratio"],
                                    "ticket": ord_info["ticket"],
                                    "result": "PENDING_LIMIT",
                                    "notes": f"Limit entry a {PULLBACK_PIPS} pips, TP escalado {ord_info['rr_ratio']}x"
                                })
                            logger.success("✅ Órdenes LIMIT guardadas en dataset")
                        except Exception as e:
                            logger.error(f"❌ Error log_trade: {e}")
                    
                    # 🔄 Gestionar Breakeven (si TP1 ya se cerró)
                    if orders_placed:
                        check_and_apply_breakeven(orders_placed, signal)

                else:
                    logger.info("⏳ No hay señal válida en este ciclo.")

                # =========================================
                # WAIT - 80 SEGUNDOS
                # =========================================
                wait_seconds = 80
                logger.info(f"⏱️ Esperando {wait_seconds}s (próxima revisión: {datetime.now().strftime('%H:%M:%S')})...")
                for _ in range(wait_seconds):
                    if stop_event.is_set():
                        break
                    time.sleep(1)

            except Exception as e:
                logger.exception(f"❌ Error en ciclo principal: {e}")
                time.sleep(5)

    # =====================================================
    # FINALLY: LIMPIEZA
    # =====================================================
    finally:
        logger.warning("🛑 Cerrando conexión...")
        connector.disconnect()
        logger.success("✅ Bot finalizado correctamente")


# =========================================================
# START
# =========================================================

if __name__ == "__main__":
    main()
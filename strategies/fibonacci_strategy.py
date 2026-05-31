"""
fibonacci_strategy.py

Estrategia basada en Fibonacci (Alex Ruiz) para bot de trading.
Funciona con múltiples temporalidades: Diario, 4H, 5min.

Requisitos: pandas, numpy, loguru, MetaTrader5
"""

import pandas as pd
import numpy as np
from loguru import logger
from typing import Optional, Tuple
import MetaTrader5 as mt5


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
def find_swing_points(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Detecta pivotes de swing (máximos/mínimos) en un DataFrame.
    Marca con 1 = swing high, -1 = swing low.
    """
    high_swing = (
        (df['high'] > df['high'].rolling(window=2*lookback+1, center=True).max().shift(-lookback)) &
        (df['high'] == df['high'].rolling(window=lookback, center=True).max())
    )
    low_swing = (
        (df['low'] < df['low'].rolling(window=2*lookback+1, center=True).min().shift(-lookback)) &
        (df['low'] == df['low'].rolling(window=lookback, center=True).min())
    )
    df = df.copy()
    df['swing'] = 0
    df.loc[high_swing, 'swing'] = 1
    df.loc[low_swing, 'swing'] = -1
    return df

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# ============================================================
# CLASE PRINCIPAL DE LA ESTRATEGIA
# ============================================================
class FibonacciStrategy:
    """
    Estrategia basada en Fibonacci para bot de trading multi-timeframe.
    """

    def __init__(
        self,
        daily_df: pd.DataFrame,
        h4_df: pd.DataFrame,
        m5_df: pd.DataFrame,
        ema_period: int = 50,
        fib_level_entry: float = 0.720,
        fib_level_stop: float = 0.75,
        fib_level_tp: float = 1.618,
        support_tolerance: float = 0.01,
        fib_tolerance: float = 0.005,
        lookback_swing: int = 20,
        min_tests: int = 2,
    ):
        self.daily = daily_df.copy()
        self.h4 = h4_df.copy()
        self.m5 = m5_df.copy()
        self.ema_period = ema_period
        self.fib_entry = fib_level_entry
        self.fib_stop = fib_level_stop
        self.fib_tp = fib_level_tp
        self.support_tol = support_tolerance
        self.fib_tol = fib_tolerance
        self.lookback = lookback_swing
        self.min_tests = min_tests

        # Estado interno
        self.current_support: Optional[float] = None
        self.impulse_start_idx: Optional[int] = None
        self.impulse_end_idx: Optional[int] = None
        self.entry_ready: bool = False
        self.last_signal_bar: Optional[int] = None

        # Precalcular indicadores
        self._prepare_data()

    def _prepare_data(self):
        """Calcula indicadores y swings en todos los timeframes."""
        self.daily['ema50'] = ema(self.daily['close'], self.ema_period)
        self.daily = find_swing_points(self.daily, self.lookback)
        self.h4['ema50'] = ema(self.h4['close'], self.ema_period)
        self.m5['ema50'] = ema(self.m5['close'], self.ema_period)

    def _get_recent_swing_low(self, df: pd.DataFrame) -> Optional[float]:
        """Retorna el último swing low significativo."""
        swings = df[df['swing'] == -1]
        if swings.empty:
            return None
        last_swing = swings.iloc[-1]
        level = last_swing['low']
        tests = df[(df['low'] <= level * (1 + self.support_tol)) & 
                   (df['low'] >= level * (1 - self.support_tol))].shape[0]
        if tests >= self.min_tests:
            return level
        return None

    def _is_bullish_reversal_daily(self, support_level: float) -> bool:
        """Verifica patrón de giro alcista en diario."""
        last = self.daily.iloc[-1]
        near_support = abs(last['low'] - support_level) / support_level < self.support_tol
        return near_support and (last['close'] > last['open'])

    def _ema50_breakout_4h(self) -> bool:
        """Detecta cruce alcista de EMA50 en 4H."""
        if len(self.h4) < 3:
            return False
        last = self.h4.iloc[-1]
        prev = self.h4.iloc[-2]
        return last['close'] > last['ema50'] and prev['close'] < prev['ema50']

    def _find_impulse_wave(self) -> Tuple[Optional[int], Optional[int], float, float]:
        """Identifica el impulso alcista en 4H."""
        df = self.h4.copy()
        df['swing_low'] = (df['low'] == df['low'].rolling(window=10, center=True).min())
        lows = df[df['swing_low']]
        if lows.empty:
            return None, None, None, None

        last_low = lows.iloc[-1]
        start_idx = last_low.name
        start_price = last_low['low']

        segment = df.loc[start_idx:]
        highs_above_ema = segment[(segment['close'] > segment['ema50'])]
        if highs_above_ema.empty:
            return None, None, None, None
        end_idx = highs_above_ema['high'].idxmax()
        end_price = df.loc[end_idx, 'high']
        return start_idx, end_idx, start_price, end_price

    def _is_at_fib_entry(self, start_price: float, end_price: float) -> bool:
        """Verifica confluencia Fibonacci + EMA en 4H."""
        current = self.h4.iloc[-1]
        retracement = end_price - (end_price - start_price) * self.fib_entry
        diff_pct = abs(current['low'] - retracement) / retracement
        near_fib = diff_pct < self.fib_tol
        near_ema = abs(current['close'] - current['ema50']) / current['ema50'] < self.fib_tol
        return near_fib and near_ema

    def _ema50_breakout_5min(self) -> bool:
        """Detecta cruce alcista de EMA50 en 5min."""
        if len(self.m5) < 2:
            return False
        last = self.m5.iloc[-1]
        prev = self.m5.iloc[-2]
        return last['close'] > last['ema50'] and prev['close'] <= prev['ema50']

    def _calculate_levels(self, start_price: float, end_price: float, pullback_low: float):
        """Calcula SL y TP base (fib_tp para TP3)."""
        diff = end_price - start_price
        tp = end_price + diff * (self.fib_tp - 1)
        sl_fib = end_price - diff * self.fib_stop
        sl = min(start_price, pullback_low, sl_fib)
        sl *= 0.998  # buffer de seguridad
        return sl, tp

    def check_signal(self) -> Optional[dict]:
        """Evalúa todos los pasos y retorna señal BUY si se cumplen condiciones."""
        try:
            # PASO 1: Soporte diario
            logger.info("Paso 1: Buscando soporte diario")
            support = self._get_recent_swing_low(self.daily)
            if support is None:
                logger.warning("No se encontró soporte válido")
                return None
            logger.success(f"Soporte encontrado: {support:.2f}")

            # PASO 2: Giro alcista diario
            logger.info("Paso 2: Validando giro alcista diario")
            if not self._is_bullish_reversal_daily(support):
                logger.warning("No existe giro alcista diario")
                return None
            logger.success("Giro alcista diario OK")

            # PASO 3: Ruptura EMA50 H4
            logger.info("Paso 3: Ruptura EMA50 H4")
            if not self._ema50_breakout_4h():
                logger.warning("No existe ruptura EMA50 H4")
                return None
            logger.success("Break EMA50 H4 OK")

            # PASO 4: Impulso H4
            logger.info("Paso 4: Buscando impulso H4")
            start_idx, end_idx, start_price, end_price = self._find_impulse_wave()
            if start_price is None:
                logger.warning("No se encontró impulso válido")
                return None
            logger.success(f"Impulso encontrado {start_price:.2f} -> {end_price:.2f}")

            # PASO 4B: Confluencia Fibonacci
            logger.info("Paso 4B: Validando Fibonacci")
            if not self._is_at_fib_entry(start_price, end_price):
                logger.warning("Precio fuera de zona Fibonacci")
                return None
            logger.success("Confluencia Fibonacci OK")

            # PASO 5: Break EMA50 M5
            logger.info("Paso 5: Break EMA50 M5")
            if not self._ema50_breakout_5min():
                logger.warning("No existe ruptura EMA50 M5")
                return None
            logger.success("Break EMA50 M5 OK")

            # Evitar duplicados
            current_m5_time = self.m5.iloc[-1]["time"]
            if self.last_signal_bar == current_m5_time:
                logger.warning("Señal duplicada ignorada")
                return None

            # Calcular niveles
            pullback_low = self.h4["low"].iloc[-1]
            sl, fib_tp = self._calculate_levels(start_price, end_price, pullback_low)
            entry_price = self.m5.iloc[-1]["close"]
            risk = entry_price - sl

            tp1 = entry_price + risk
            tp2 = entry_price + (risk * 2)
            tp3 = fib_tp

            self.last_signal_bar = current_m5_time

            logger.success(
                f"BUY detectado | "
                f"Entry={entry_price:.2f} "
                f"SL={sl:.2f} "
                f"TP1={tp1:.2f} "
                f"TP2={tp2:.2f} "
                f"TP3={tp3:.2f}"
            )

            return {
                "action": "BUY",
                "entry_price": entry_price,
                "stop_loss": sl,
                "tp1": tp1,
                "tp2": tp2,
                "tp3": tp3,
                "time": current_m5_time,
                "reason": "Fibonacci + EMA Confluence"
            }

        except Exception as e:
            logger.exception(f"Error en check_signal(): {e}")
            return None
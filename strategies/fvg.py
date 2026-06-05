"""
fvg.py

Fair Value Gap & Inverted FVG Detector

Basado en la lógica de Smart Money Concepts [LuxAlgo]
Detecta FVGs e iFVGs en una serie de velas.
"""

import pandas as pd
import numpy as np
from loguru import logger


class FVGDetector:
    """
    Detector de Fair Value Gaps (FVG) e Inverted Fair Value Gaps (iFVG).

    Parámetros
    ----------
    atr_length : int, default 200
        Período para calcular el ATR (Average True Range) y determinar la fuerza del FVG.
    extend_bars : int, default 1
        Número de barras a extender las cajas del FVG hacia la derecha (para dibujo).
    min_gap_percent : float, default 0.0
        Porcentaje mínimo (como decimal, ej. 0.1 para 0.1%) del precio para filtrar gaps pequeños.
        Si es 0.0, no se aplica filtro.
    show_strength : bool, default True
        Calcular la fuerza del FVG como porcentaje del ATR.
    auto_threshold : bool, default False
        Si es True, intenta calcular un umbral dinámico (simplificado). 
        Si es False, usa min_gap_percent.
    """

    def __init__(
        self,
        atr_length: int = 200,
        extend_bars: int = 1,
        min_gap_percent: float = 0.0,
        show_strength: bool = True,
        auto_threshold: bool = False
    ):
        self.atr_length = atr_length
        self.extend_bars = extend_bars
        self.min_gap_percent = min_gap_percent
        self.show_strength = show_strength
        self.auto_threshold = auto_threshold

        logger.info("FVGDetector iniciado")

    # =====================================================
    # DETECTAR FVGs E iFVGs
    # =====================================================
    def detect(self, df: pd.DataFrame) -> list:
        """
        Aplica la detección de FVGs e iFVGs sobre el DataFrame.

        Parámetros
        ----------
        df : pd.DataFrame
            Debe contener las columnas: 'time' (timestamp Unix), 'open', 'high', 'low', 'close'.

        Retorna
        -------
        list[dict]
            Lista de FVGs e iFVGs detectados con las siguientes claves:
                - type: 'FVG' o 'iFVG'
                - bias: 'BULLISH' o 'BEARISH'
                - time: int (timestamp de la barra actual donde se detectó)
                - index: int (índice de la barra actual en el DataFrame)
                - top: float (precio superior de la zona)
                - bottom: float (precio inferior de la zona)
                - strength: float (0-100) si show_strength=True, sino 0
                - mitigated: bool (si ya fue neutralizado por el precio)
                - box: dict con coordenadas para dibujo (left_time, right_time, top, bottom)
        """
        fvgs = []
        try:
            required = ['time', 'open', 'high', 'low', 'close']
            for col in required:
                if col not in df.columns:
                    raise ValueError(f"Falta la columna '{col}' en el DataFrame")

            df = df.reset_index(drop=True)
            n = len(df)

            atr = self._calculate_atr(df, self.atr_length)
            active_fvgs = []

            for i in range(2, n):
                last2_high = df['high'].iloc[i - 2]
                last2_low = df['low'].iloc[i - 2]
                last_close = df['close'].iloc[i - 1]
                last_open = df['open'].iloc[i - 1]
                last_time = df['time'].iloc[i - 1]
                current_high = df['high'].iloc[i]
                current_low = df['low'].iloc[i]
                current_time = df['time'].iloc[i]

                bar_delta_pct = abs(last_close - last_open) / last_open if last_open != 0 else 0

                if self.auto_threshold:
                    threshold = 0.2 * atr[i] / ((last_open + last_close) / 2) if last_open != 0 else 0
                else:
                    threshold = self.min_gap_percent / 100.0

                bullish_fvg = (current_low > last2_high) and (last_close > last2_high) and (bar_delta_pct > threshold)
                bearish_fvg = (current_high < last2_low) and (last_close < last2_low) and (bar_delta_pct > threshold)

                new_ifvgs = []
                for fvg in active_fvgs:
                    if fvg['type'] == 'iFVG':
                        continue

                    if fvg['bias'] == 'BEARISH' and current_low <= fvg['top'] and current_high >= fvg['bottom'] and df['close'].iloc[i] > fvg['bottom']:
                        gap_size = fvg['top'] - fvg['bottom']
                        strength = self._strength(gap_size, atr[i]) if self.show_strength else 0.0
                        ifvg = {
                            'type': 'iFVG',
                            'bias': 'BULLISH',
                            'time': current_time,
                            'index': i,
                            'top': fvg['bottom'],
                            'bottom': fvg['top'],
                            'strength': strength,
                            'mitigated': False,
                            'box': self._make_box(last_time, current_time, fvg['bottom'], fvg['top'], self.extend_bars)
                        }
                        new_ifvgs.append(ifvg)
                        fvg['mitigated'] = True
                        active_fvgs.remove(fvg)
                        break

                    elif fvg['bias'] == 'BULLISH' and current_high >= fvg['bottom'] and current_low <= fvg['top'] and df['close'].iloc[i] < fvg['top']:
                        gap_size = fvg['top'] - fvg['bottom']
                        strength = self._strength(gap_size, atr[i]) if self.show_strength else 0.0
                        ifvg = {
                            'type': 'iFVG',
                            'bias': 'BEARISH',
                            'time': current_time,
                            'index': i,
                            'top': fvg['top'],
                            'bottom': fvg['bottom'],
                            'strength': strength,
                            'mitigated': False,
                            'box': self._make_box(last_time, current_time, fvg['top'], fvg['bottom'], self.extend_bars)
                        }
                        new_ifvgs.append(ifvg)
                        fvg['mitigated'] = True
                        active_fvgs.remove(fvg)
                        break

                fvgs = new_ifvgs + fvgs

                if bullish_fvg:
                    gap_size = current_low - last2_high
                    strength = self._strength(gap_size, atr[i]) if self.show_strength else 0.0
                    fvg_new = {
                        'type': 'FVG',
                        'bias': 'BULLISH',
                        'time': current_time,
                        'index': i,
                        'top': current_low,
                        'bottom': last2_high,
                        'strength': strength,
                        'mitigated': False,
                        'box': self._make_box(last_time, current_time, current_low, last2_high, self.extend_bars)
                    }
                    fvgs.insert(0, fvg_new)
                    active_fvgs.append(fvg_new)

                if bearish_fvg:
                    gap_size = last2_low - current_high
                    strength = self._strength(gap_size, atr[i]) if self.show_strength else 0.0
                    fvg_new = {
                        'type': 'FVG',
                        'bias': 'BEARISH',
                        'time': current_time,
                        'index': i,
                        'top': current_high,
                        'bottom': last2_low,
                        'strength': strength,
                        'mitigated': False,
                        'box': self._make_box(last_time, current_time, current_high, last2_low, self.extend_bars)
                    }
                    fvgs.insert(0, fvg_new)
                    active_fvgs.append(fvg_new)

                for fvg in active_fvgs:
                    if fvg['mitigated']:
                        continue
                    if fvg['bias'] == 'BULLISH' and current_low <= fvg['bottom']:
                        fvg['mitigated'] = True
                    elif fvg['bias'] == 'BEARISH' and current_high >= fvg['top']:
                        fvg['mitigated'] = True

                active_fvgs = [f for f in active_fvgs if not f['mitigated']]

            logger.success(f"FVGs detectados: {len(fvgs)}")
            return fvgs

        except Exception as e:
            logger.exception(f"Error detectando FVGs: {e}")
            return fvgs

    def _calculate_atr(self, df: pd.DataFrame, length: int) -> list:
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        tr = np.zeros(len(df))
        for i in range(1, len(df)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )
        atr = np.full(len(df), np.nan)
        if len(df) > length:
            atr[length:] = pd.Series(tr).rolling(window=length).mean().values[length:]
        first_valid = length if length < len(df) else 0
        for i in range(first_valid, len(df)):
            if i >= first_valid:
                atr[i] = atr[i] if not np.isnan(atr[i]) else atr[i-1]
        return atr

    def _strength(self, gap_size: float, atr_val: float) -> float:
        if atr_val and atr_val > 0:
            return min((gap_size / atr_val) * 100, 100.0)
        return 0.0

    def _make_box(self, left_time, right_time, top, bottom, extend=1):
        bar_duration = right_time - left_time
        return {
            'left_time': left_time,
            'right_time': right_time + extend * bar_duration,
            'top': top,
            'bottom': bottom
        }


def detect_fvg(df: pd.DataFrame, **kwargs) -> list:
    """
    Función de conveniencia para detectar FVGs e iFVGs.
    """
    detector = FVGDetector(**kwargs)
    return detector.detect(df)

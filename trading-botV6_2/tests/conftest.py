import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def random_seed():
    return 42


def _make_ohlc(open_prices: list[float], closes: list[float],
               highs: list[float], lows: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "open": open_prices,
        "close": closes,
        "high": highs,
        "low": lows,
    })


@pytest.fixture
def df_flat() -> pd.DataFrame:
    """15 velas planas sin tendencia — ideal para probar INDECISION."""
    n = 15
    base = 100.0
    return _make_ohlc(
        open_prices=[base] * n,
        closes=[base] * n,
        highs=[base + 0.1] * n,
        lows=[base - 0.1] * n,
    )


@pytest.fixture
def df_bullish_trend() -> pd.DataFrame:
    """15 velas en tendencia alcista constante."""
    opens = [100.0 + i * 0.5 for i in range(15)]
    closes = [100.5 + i * 0.5 for i in range(15)]
    highs = [c + 0.3 for c in closes]
    lows = [o - 0.3 for o in opens]
    return _make_ohlc(opens, closes, highs, lows)


@pytest.fixture
def df_bearish_trend() -> pd.DataFrame:
    """15 velas en tendencia bajista constante."""
    opens = [100.0 - i * 0.5 for i in range(15)]
    closes = [99.5 - i * 0.5 for i in range(15)]
    highs = [o + 0.3 for o in opens]
    lows = [c - 0.3 for c in closes]
    return _make_ohlc(opens, closes, highs, lows)


@pytest.fixture
def df_bullish_fvg() -> pd.DataFrame:
    """15 velas donde las últimas 4 crean un FVG alcista.
    c1[high] < c2[low]  (gap UP entre c1 y c2)
    c1 = iloc[-4], c2 = iloc[-3]
    """
    opens = [100.0] * 11 + [100.0, 101.0, 101.5, 101.5]
    closes = [100.0] * 11 + [100.0, 101.0, 101.5, 101.8]
    highs = [100.3] * 11 + [100.3, 101.5, 102.0, 102.2]
    lows = [99.7] * 11 + [99.7, 101.2, 101.0, 101.0]
    return _make_ohlc(opens, closes, highs, lows)


@pytest.fixture
def df_bearish_fvg() -> pd.DataFrame:
    """15 velas donde las últimas 4 crean un FVG bajista.
    c2[low] > c3[high]  (gap DOWN entre c2 y c3)
    """
    opens = [105.0] * 11 + [105.0, 105.0, 102.0, 102.0]
    closes = [105.0] * 11 + [105.0, 105.0, 102.0, 101.7]
    highs = [105.3] * 11 + [105.3, 105.3, 102.5, 102.2]
    lows = [104.7] * 11 + [104.7, 104.7, 101.5, 101.3]
    return _make_ohlc(opens, closes, highs, lows)


@pytest.fixture
def df_sweep_high() -> pd.DataFrame:
    """16 velas: sweep de máximo con vela bajista fuerte (body/rng=0.67)."""
    opens = [100.0] * 11 + [
        100.0, 100.0, 100.0,
        103.0,  # sweep: high=104, low=101.5, range=2.5, close=101.0, body=2.0 → 0.8 > 0.6
        102.0,
        101.0,
    ]
    closes = [100.0] * 11 + [
        100.0, 100.0, 100.0,
        101.0,  # bearish (close < open), body=2.0
        101.8,
        101.0,
    ]
    highs = [100.3] * 11 + [
        100.3, 100.3, 100.3,
        104.0,  # > max(prev 3 highs = 100.3)
        102.5,
        101.5,
    ]
    lows = [99.7] * 11 + [
        99.7, 99.7, 99.7,
        101.5,  # body/rng = 2.0/2.5 = 0.8 > 0.6 ✓
        101.5,
        100.5,
    ]
    return _make_ohlc(opens, closes, highs, lows)


@pytest.fixture
def df_sweep_low() -> pd.DataFrame:
    """16 velas: sweep de mínimo con vela alcista fuerte (body/rng=0.83)."""
    opens = [100.0] * 11 + [
        100.0, 100.0, 100.0,
        96.0,  # sweep: low=96 < min(prev), close=98.5 > open, body=2.5, rng=3.0, ratio=0.83
        99.0,
        100.0,
    ]
    closes = [100.0] * 11 + [
        100.0, 100.0, 100.0,
        98.5,  # bullish (close > open), body=2.5
        99.5,
        100.5,
    ]
    highs = [100.3] * 11 + [
        100.3, 100.3, 100.3,
        99.0,  # range=99.0-96.0=3.0, body/rng=2.5/3.0=0.833 > 0.6 ✓
        100.0,
        101.0,
    ]
    lows = [99.7] * 11 + [
        99.7, 99.7, 99.7,
        96.0,  # < min(prev 3 lows = 99.7)
        98.5,
        99.0,
    ]
    return _make_ohlc(opens, closes, highs, lows)


@pytest.fixture
def df_asymmetric_buy() -> pd.DataFrame:
    """20 velas: 5 impulse alcistas + 5 retrace bajistas + 2 final alcistas.
    Para AsymmetryFilter con direction=BUY:
      - retrace: últimas velas bajistas (close < open)
      - impulse: velas antes del retrace que son alcistas (close > open)
    """
    opens = [100.0] * 8 + [
        101.0, 101.5, 102.0, 102.5, 103.0,
        103.5, 103.2, 102.8, 102.5, 102.0,
        102.2, 102.5,
    ]
    closes = [100.0] * 8 + [
        101.5, 102.0, 102.5, 103.0, 103.5,
        103.0, 102.8, 102.5, 102.0, 101.8,
        102.5, 102.8,
    ]
    highs = [100.5] * 8 + [c + 0.3 for c in closes[8:]]
    lows = [99.5] * 8 + [o - 0.3 for o in opens[8:]]
    return _make_ohlc(opens, closes, highs, lows)


@pytest.fixture
def df_asymmetric_sell() -> pd.DataFrame:
    """20 velas: 5 impulse bajistas + 5 retrace alcistas + 2 final bajistas."""
    opens = [100.0] * 8 + [
        103.0, 102.5, 102.0, 101.5, 101.0,
        100.5, 100.8, 101.2, 101.5, 102.0,
        101.8, 101.5,
    ]
    closes = [100.0] * 8 + [
        102.5, 102.0, 101.5, 101.0, 100.5,
        101.0, 101.2, 101.5, 102.0, 102.2,
        101.5, 101.2,
    ]
    highs = [100.5] * 8 + [o + 0.3 for o in opens[8:]]
    lows = [99.5] * 8 + [c - 0.3 for c in closes[8:]]
    return _make_ohlc(opens, closes, highs, lows)


@pytest.fixture
def df_stop_run_bullish() -> pd.DataFrame:
    """12 velas: stop run alcista.
    mid = 12-5-2 = 5 → high[5:10], low[5:10] son región estable
    Index 10: low rompe before_min (99.7)
    Index 11: close > before_min → reversión
    """
    opens = [100.0] * 10 + [99.5, 100.0]
    closes = [100.0] * 10 + [99.0, 100.5]
    highs = [100.3] * 10 + [100.0, 101.0]
    lows = [99.7] * 10 + [98.0, 99.5]
    return _make_ohlc(opens, closes, highs, lows)


@pytest.fixture
def df_stop_run_bearish() -> pd.DataFrame:
    """12 velas: stop run bajista.
    mid = 12-5-2 = 5 → high[5:10], low[5:10] son región estable
    Index 10: high rompe before_max (100.3)
    Index 11: close < before_max → reversión
    """
    opens = [100.0] * 10 + [101.5, 100.0]
    closes = [100.0] * 10 + [102.0, 99.5]
    highs = [100.3] * 10 + [103.0, 100.5]
    lows = [99.7] * 10 + [101.0, 99.0]
    return _make_ohlc(opens, closes, highs, lows)


@pytest.fixture
def df_retrace_confirmed() -> pd.DataFrame:
    """8 velas: 3 impulse down + 2 retrace up confirmado.
    trend DOWN: close[-5]=100 > close[-4]=99 > close[-3]=98
    retrace UP: nc1=close[-2]=98.5 > c3=98 and nc2=close[-1]=99 > nc1=98.5 and > c3=98
    """
    opens = [100.0] * 3 + [100.5, 99.5, 98.5, 98.5, 99.0]
    closes = [100.0] * 3 + [100.0, 99.0, 98.0, 98.5, 99.0]
    highs = [100.3] * 3 + [101.0, 99.5, 98.5, 99.0, 99.5]
    lows = [99.7] * 3 + [99.5, 98.5, 97.5, 98.0, 98.5]
    return _make_ohlc(opens, closes, highs, lows)


@pytest.fixture
def df_candle_confirm_buy() -> pd.DataFrame:
    """10 velas para CandleConfirmer: última vela alcista con cuerpo grande."""
    opens = [100.0] * 9 + [100.0]
    closes = [100.0] * 9 + [102.0]
    highs = [100.3] * 9 + [102.3]
    lows = [99.7] * 9 + [99.7]
    return _make_ohlc(opens, closes, highs, lows)


@pytest.fixture
def df_candle_confirm_sell() -> pd.DataFrame:
    """10 velas: última vela bajista con cuerpo grande."""
    opens = [100.0] * 9 + [102.0]
    closes = [100.0] * 9 + [100.0]
    highs = [100.3] * 9 + [102.3]
    lows = [99.7] * 9 + [99.7]
    return _make_ohlc(opens, closes, highs, lows)

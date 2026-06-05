"""
kachazorraz.py

Réplica fiel del indicador Smart Money Concepts [LuxAlgo] + Auto Fibonacci.
Traducido de Pine Script v6 a Python para MT5.

Conceptos:
  1. Swing Fractals con sistema de piernas (legs)
  2. Break of Structure (BOS)
  3. Change of Character (CHoCH)
  4. Equal Highs / Equal Lows (EQH/EQL)
  5. Fair Value Gaps (FVG) + Inverted FVG
  6. Strong / Weak classification
  7. Premium / Discount Zones
  8. Auto Fibonacci (22% compra, 72% venta)
  9. Order Blocks con filtro de volatilidad
"""

import pandas as pd
import numpy as np
from loguru import logger
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ============================================================
# CONSTANTES
# ============================================================
BULLISH_LEG = 1
BEARISH_LEG = 0

BULLISH = 1
BEARISH = -1

HISTORICAL = "Historical"
PRESENT = "Present"


# ============================================================
# DATACLASSES
# ============================================================

@dataclass
class Pivot:
    current_level: float = 0.0
    last_level: float = 0.0
    crossed: bool = False
    bar_time: any = None
    bar_index: int = 0

@dataclass
class Trend:
    bias: int = 0

@dataclass
class TrailingExtremes:
    top: float = 0.0
    bottom: float = 1e9
    bar_time: any = None
    bar_index: int = 0
    last_top_time: any = None
    last_bottom_time: any = None

@dataclass
class SwingPoint:
    index: int
    time: any
    price: float
    swing_type: str
    label: str = ""
    is_strong: bool = False
    is_weak: bool = False

@dataclass
class BOS:
    index: int
    time: any
    price: float
    bos_type: str
    broken_level: float

@dataclass
class CHoCH:
    index: int
    time: any
    price: float
    choc_type: str
    previous_trend: str

@dataclass
class EqualHL:
    index: int
    time: any
    price: float
    eql_type: str
    count: int

@dataclass
class FVG:
    index: int
    time: any
    fvg_type: str
    bias: str
    top: float
    bottom: float
    strength: float
    mitigated: bool
    box: Dict

@dataclass
class OrderBlock:
    bar_high: float
    bar_low: float
    bar_time: any
    bias: int
    index: int

@dataclass
class PremiumDiscountZone:
    swing_high: float
    swing_low: float
    premium_top: float
    premium_bottom: float
    equilibrium_top: float
    equilibrium_bottom: float
    discount_top: float
    discount_bottom: float

@dataclass
class FibonacciLevel:
    level: float
    price: float
    label: str

@dataclass
class TradeSignal:
    index: int
    time: any
    action: str          # BUY_LIMIT or SELL_LIMIT
    entry_price: float    # precio del limit (22% para buy, 72% para sell)
    stop_loss: float
    take_profit: float
    reason: str
    confidence: float
    limit_price: float = 0.0
    factors: List[str] = field(default_factory=list)


# ============================================================
# CLASE PRINCIPAL
# ============================================================

class SmartMoneyConcepts:
    """
    Smart Money Concepts [LuxAlgo] + Auto Fibonacci.
    Réplica fiel de la lógica Pine Script v6.
    """

    def __init__(
        self,
        # Swings
        swing_length: int = 50,
        equal_length: int = 3,
        equal_threshold: float = 0.1,
        # FVG
        fvg_auto_threshold: bool = True,
        fvg_extend_bars: int = 1,
        show_fvg_strength: bool = True,
        show_ifvg: bool = True,
        # Fibonacci
        show_fib: bool = True,
        fib_extend: bool = True,
        # Señales
        require_fvg_for_signal: bool = True,
        require_bos_for_signal: bool = True,
        # SL/TP
        symbol: str = "XAUUSD",
        sl_pips: float = 20.0,
        rr_ratio: float = 2.0,
    ):
        self.swing_length = swing_length
        self.equal_length = equal_length
        self.equal_threshold = equal_threshold
        self.fvg_auto_threshold = fvg_auto_threshold
        self.fvg_extend_bars = fvg_extend_bars
        self.show_fvg_strength = show_fvg_strength
        self.show_ifvg = show_ifvg
        self.show_fib = show_fib
        self.fib_extend = fib_extend
        self.require_fvg_for_signal = require_fvg_for_signal
        self.require_bos_for_signal = require_bos_for_signal
        self.symbol = symbol
        self.sl_pips = sl_pips
        self.rr_ratio = rr_ratio

        # State
        self.swing_high = Pivot()
        self.swing_low = Pivot()
        self.swing_trend = Trend()
        self.trailing = TrailingExtremes()
        self._leg_state = 0
        self._leg_state_prev = 0

        self.parsed_highs: List[float] = []
        self.parsed_lows: List[float] = []
        self.highs: List[float] = []
        self.lows: List[float] = []
        self.times: List[any] = []

        self.swing_order_blocks: List[OrderBlock] = []
        self.fair_value_gaps: List[FVG] = []
        self.fvg_active: List[FVG] = []

        logger.info(f"kachazorraz iniciado | SwingLength={swing_length} | Symbol={symbol} | SL={sl_pips}pips")

    @property
    def pip_value(self) -> float:
        if self.symbol in ("XAUUSD", "GOLD", "XAUUSDm", "XAUEUR", "XAUEURm"):
            return 0.1
        elif self.symbol in ("BTCUSD", "BTCUSDm"):
            return 1.0
        elif self.symbol in ("ETHUSD", "ETHUSDm"):
            return 0.1
        elif self.symbol in ("EURUSD", "GBPUSD", "AUDUSD", "USDCAD", "NZDUSD", "EURUSDm"):
            return 0.0001
        elif self.symbol in ("USDJPY", "USDJPYm"):
            return 0.01
        return 0.0001

    @property
    def pip_distance(self) -> float:
        return self.pip_value * self.sl_pips

    # ============================================================
    # 1. LEG / SWING DETECTION (como Pine Script)
    # ============================================================

    def _leg(self, df: pd.DataFrame, i: int, size: int) -> int:
        """
        Determina el leg actual, réplica del Pine Script:
          leg(int size) =>
            var leg = 0
            newLegHigh = high[size] > ta.highest(high, size)   // high[i-size] > max(high[i-size+1..i])
            newLegLow  = low[size]  < ta.lowest(low, size)     // low[i-size]  < min(low[i-size+1..i])
        """
        if i < size:
            return self._leg_state

        high_at_pivot = float(df["high"].iloc[i - size])
        low_at_pivot = float(df["low"].iloc[i - size])

        segment_high = float(df["high"].iloc[i - size + 1:i + 1].max()) if size > 1 else float(df["high"].iloc[i])
        segment_low = float(df["low"].iloc[i - size + 1:i + 1].min()) if size > 1 else float(df["low"].iloc[i])

        new_leg_high = high_at_pivot > segment_high
        new_leg_low = low_at_pivot < segment_low

        if new_leg_high:
            self._leg_state = BEARISH_LEG
        elif new_leg_low:
            self._leg_state = BULLISH_LEG

        return self._leg_state

    def _start_of_new_leg(self, leg_prev: int, leg_curr: int) -> bool:
        return leg_curr != leg_prev

    def _start_of_bearish_leg(self, leg_prev: int, leg_curr: int) -> bool:
        return leg_curr == BEARISH_LEG and leg_prev != BEARISH_LEG

    def _start_of_bullish_leg(self, leg_prev: int, leg_curr: int) -> bool:
        return leg_curr == BULLISH_LEG and leg_prev != BULLISH_LEG

    # ============================================================
    # 2. SWING STRUCTURE DETECTION
    # ============================================================

    def _get_current_structure(self, df: pd.DataFrame, i: int, size: int):
        """
        Detecta pivotes swing (réplica de getCurrentStructure en Pine).
        Pine: currentLeg = leg(size); newPivot = ta.change(leg) != 0;
              pivotLow = ta.change(leg) == 1; pivotHigh = ta.change(leg) == -1
        """
        if i < size:
            return

        self._leg_state_prev = self._leg_state
        current_leg = self._leg(df, i, size)
        leg_changed = current_leg != self._leg_state_prev

        if not leg_changed:
            return

        # pivotLow: leg changed FROM BEARISH_LEG (0) TO BULLISH_LEG (1)
        pivot_low = current_leg == BULLISH_LEG and self._leg_state_prev == BEARISH_LEG
        # pivotHigh: leg changed FROM BULLISH_LEG (1) TO BEARISH_LEG (0)
        pivot_high = current_leg == BEARISH_LEG and self._leg_state_prev == BULLISH_LEG

        if pivot_low:
            idx = i - size
            if idx < 0 or idx >= len(df):
                return
            low_val = float(df["low"].iloc[idx])
            self.swing_low.last_level = self.swing_low.current_level
            self.swing_low.current_level = low_val
            self.swing_low.crossed = False
            self.swing_low.bar_time = df["time"].iloc[idx]
            self.swing_low.bar_index = idx

            self.trailing.bottom = low_val
            self.trailing.bar_time = self.swing_low.bar_time
            self.trailing.bar_index = idx
            self.trailing.last_bottom_time = self.swing_low.bar_time

        elif pivot_high:
            idx = i - size
            if idx < 0 or idx >= len(df):
                return
            high_val = float(df["high"].iloc[idx])
            self.swing_high.last_level = self.swing_high.current_level
            self.swing_high.current_level = high_val
            self.swing_high.crossed = False
            self.swing_high.bar_time = df["time"].iloc[idx]
            self.swing_high.bar_index = idx

            self.trailing.top = high_val
            self.trailing.bar_time = self.swing_high.bar_time
            self.trailing.bar_index = idx
            self.trailing.last_top_time = self.swing_high.bar_time

    # ============================================================
    # 3. DISPLAY STRUCTURE (BOS / CHoCH)
    # ============================================================

    def _display_structure(
        self,
        df: pd.DataFrame,
        i: int,
        bos_list: List[Dict],
        choch_list: List[Dict],
        swings: List[SwingPoint],
        is_internal: bool = False
    ):
        """Detecta BOS y CHoCH como el Pine Script."""
        if i < 1:
            return

        close = float(df["close"].iloc[i])
        time_i = df["time"].iloc[i]

        # --- BULLISH: crossover(close, swingHigh.currentLevel) ---
        h_pivot = self.swing_high
        if (
            h_pivot.current_level > 0
            and close > h_pivot.current_level
            and not h_pivot.crossed
        ):
            is_choch = self.swing_trend.bias == BEARISH
            h_pivot.crossed = True
            self.swing_trend.bias = BULLISH

            tag = "CHoCH" if is_choch else "BOS"

            bos_list.append({
                "index": i,
                "time": time_i,
                "price": close,
                "type": "BULLISH_BOS",
                "broken_level": float(h_pivot.current_level),
            })

            if is_choch:
                choch_list.append({
                    "index": i,
                    "time": time_i,
                    "price": close,
                    "type": "BULLISH_CHOCH",
                    "previous_trend": "BEARISH",
                })

            # Store order block
            self._store_order_block(self.swing_high, BULLISH, df, i)

            if not is_internal:
                swings.append(SwingPoint(
                    index=i, time=time_i, price=close,
                    swing_type="HIGH", label=tag
                ))

        # --- BEARISH: crossunder(close, swingLow.currentLevel) ---
        l_pivot = self.swing_low
        if (
            l_pivot.current_level > 0
            and close < l_pivot.current_level
            and not l_pivot.crossed
        ):
            is_choch = self.swing_trend.bias == BULLISH
            l_pivot.crossed = True
            self.swing_trend.bias = BEARISH

            tag = "CHoCH" if is_choch else "BOS"

            bos_list.append({
                "index": i,
                "time": time_i,
                "price": close,
                "type": "BEARISH_BOS",
                "broken_level": float(l_pivot.current_level),
            })

            if is_choch:
                choch_list.append({
                    "index": i,
                    "time": time_i,
                    "price": close,
                    "type": "BEARISH_CHOCH",
                    "previous_trend": "BULLISH",
                })

            self._store_order_block(self.swing_low, BEARISH, df, i)

            if not is_internal:
                swings.append(SwingPoint(
                    index=i, time=time_i, price=close,
                    swing_type="LOW", label=tag
                ))

    # ============================================================
    # 4. ORDER BLOCKS (con filtro de volatilidad)
    # ============================================================

    def _store_order_block(self, pivot: Pivot, bias: int, df: pd.DataFrame, current_idx: int):
        """Almacena un order block como en Pine Script."""
        if pivot.bar_index >= current_idx:
            return

        segment = df.iloc[pivot.bar_index:current_idx + 1]

        if bias == BEARISH:
            # Buscar el máximo de parsedHighs entre pivot y barra actual
            idx_offset = segment["high"].idxmax() - pivot.bar_index if len(segment) > 0 else 0
            parsed_idx = pivot.bar_index + idx_offset
        else:
            idx_offset = segment["low"].idxmin() - pivot.bar_index if len(segment) > 0 else 0
            parsed_idx = pivot.bar_index + idx_offset

        if parsed_idx >= len(df) or parsed_idx < 0:
            return

        ob = OrderBlock(
            bar_high=float(df.iloc[parsed_idx]["high"]),
            bar_low=float(df.iloc[parsed_idx]["low"]),
            bar_time=df.iloc[parsed_idx]["time"],
            bias=bias,
            index=parsed_idx,
        )

        self.swing_order_blocks.insert(0, ob)
        if len(self.swing_order_blocks) > 100:
            self.swing_order_blocks.pop()

    def _delete_order_blocks(self, df: pd.DataFrame, current_idx: int):
        """Elimina order blocks mitigados."""
        if current_idx < 0:
            return
        high = float(df.iloc[current_idx]["high"])
        low = float(df.iloc[current_idx]["low"])
        close = float(df.iloc[current_idx]["close"])

        to_remove = []
        for ob in self.swing_order_blocks:
            if ob.bias == BEARISH and close > ob.bar_high:
                to_remove.append(ob)
            elif ob.bias == BULLISH and close < ob.bar_low:
                to_remove.append(ob)

        for ob in to_remove:
            if ob in self.swing_order_blocks:
                self.swing_order_blocks.remove(ob)

    def get_order_blocks(self, limit: int = 5) -> List[Dict]:
        """Retorna order blocks para visualización."""
        result = []
        for ob in self.swing_order_blocks[:limit]:
            result.append({
                "type": "BULLISH_OB" if ob.bias == BULLISH else "BEARISH_OB",
                "time": ob.bar_time,
                "high": ob.bar_high,
                "low": ob.bar_low,
                "index": ob.index,
                "mitigated": False,
                "bias": ob.bias,
                "bar_time": ob.bar_time,
                "bar_high": ob.bar_high,
                "bar_low": ob.bar_low,
            })
        return result

    # ============================================================
    # 5. EQUAL HIGHS / LOWS
    # ============================================================

    def _detect_equal_highs_lows(
        self,
        df: pd.DataFrame,
        i: int,
        size: int,
        atr: np.ndarray
    ) -> Optional[Dict]:
        """Detecta EQH/EQL en el pivote actual (como Pine)."""
        if i < size * 2:
            return None

        pivot_low = self._get_pivot_low(df, i, size)
        pivot_high = self._get_pivot_high(df, i, size)

        atr_val = atr[i] if i < len(atr) else 1.0

        if pivot_low is not None:
            p = pivot_low
            threshold = self.equal_threshold * atr_val
            # Buscar otro low cercano
            for j in range(i - size * 2, i):
                if j < 0:
                    continue
                if abs(df.iloc[j]["low"] - p["price"]) < threshold:
                    return {
                        "index": i,
                        "time": df.iloc[i]["time"],
                        "price": p["price"],
                        "type": "EQL",
                        "count": 2,
                    }

        if pivot_high is not None:
            p = pivot_high
            threshold = self.equal_threshold * atr_val
            for j in range(i - size * 2, i):
                if j < 0:
                    continue
                if abs(df.iloc[j]["high"] - p["price"]) < threshold:
                    return {
                        "index": i,
                        "time": df.iloc[i]["time"],
                        "price": p["price"],
                        "type": "EQH",
                        "count": 2,
                    }

        return None

    def _get_pivot_low(self, df: pd.DataFrame, i: int, size: int) -> Optional[Dict]:
        if i < size:
            return None
        left = df["low"].iloc[i - size:i].min()
        right = df["low"].iloc[i + 1:i + 1 + size].min() if i + 1 + size <= len(df) else None
        if right is None:
            return None
        if df.iloc[i]["low"] < left and df.iloc[i]["low"] < right:
            return {"index": i, "time": df.iloc[i]["time"], "price": float(df.iloc[i]["low"])}
        return None

    def _get_pivot_high(self, df: pd.DataFrame, i: int, size: int) -> Optional[Dict]:
        if i < size:
            return None
        left = df["high"].iloc[i - size:i].max()
        right = df["high"].iloc[i + 1:i + 1 + size].max() if i + 1 + size <= len(df) else None
        if right is None:
            return None
        if df.iloc[i]["high"] > left and df.iloc[i]["high"] > right:
            return {"index": i, "time": df.iloc[i]["time"], "price": float(df.iloc[i]["high"])}
        return None

    # ============================================================
    # 6. TRAILING EXTREMES
    # ============================================================

    def _update_trailing(self, df: pd.DataFrame, i: int):
        """Actualiza trailing extremes (como en Pine)."""
        if i < 0:
            return
        high = float(df.iloc[i]["high"])
        low = float(df.iloc[i]["low"])
        time_i = df.iloc[i]["time"]

        if high > self.trailing.top:
            self.trailing.top = high
            self.trailing.last_top_time = time_i
        if low < self.trailing.bottom:
            self.trailing.bottom = low
            self.trailing.last_bottom_time = time_i

        self.trailing.bar_index = i
        self.trailing.bar_time = time_i

    # ============================================================
    # 7. FVG (Fair Value Gaps) + iFVG
    # ============================================================

    def _calculate_atr(self, df: pd.DataFrame, length: int) -> np.ndarray:
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        close = df["close"].values.astype(float)
        tr = np.zeros(len(df))
        for i in range(1, len(df)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        atr = np.full(len(df), np.nan)
        if len(df) > length:
            atr[length:] = pd.Series(tr).rolling(window=length).mean().values[length:]
        atr = np.nan_to_num(atr, nan=0.0)
        return atr

    def _fvg_strength(self, gap_size: float, atr_val: float) -> float:
        if atr_val and atr_val > 0:
            return min((gap_size / atr_val) * 100, 100.0)
        return 0.0

    def _detect_fvg(self, df: pd.DataFrame) -> List[FVG]:
        """Detecta FVG e iFVG exactamente como el Pine Script."""
        fvg_list = []
        active_fvgs: List[FVG] = []
        atr = self._calculate_atr(df, 200)

        for i in range(2, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i - 1]
            prev2 = df.iloc[i - 2]

            last_close = prev["close"]
            last_open = prev["open"]
            bar_delta_pct = (last_close - last_open) / (last_open * 100) if last_open != 0 else 0

            # Threshold automático
            threshold = 0.0
            if self.fvg_auto_threshold:
                cumulative = 0.0
                count = 0
                for j in range(1, min(i, 200)):
                    if j < len(df):
                        co = df.iloc[j]["close"]
                        op = df.iloc[j]["open"]
                        if op != 0:
                            cumulative += abs((co - op) / (op * 100))
                            count += 1
                threshold = (cumulative / count) * 2 if count > 0 else 0

            bullish_fvg = (
                curr["low"] > prev2["high"]
                and last_close > prev2["high"]
                and bar_delta_pct > threshold
            )
            bearish_fvg = (
                curr["high"] < prev2["low"]
                and last_close < prev2["low"]
                and -bar_delta_pct > threshold
            )

            # iFVG: verificar FVGs activos
            new_ifvgs = []
            for fvg in active_fvgs:
                if fvg.fvg_type == "iFVG":
                    continue

                if fvg.bias == "BEARISH":
                    if (
                        curr["low"] <= fvg.top
                        and curr["high"] >= fvg.bottom
                        and prev["close"] > fvg.bottom
                    ):
                        gap_size = fvg.top - fvg.bottom
                        strength = self._fvg_strength(gap_size, atr[i]) if atr[i] > 0 else 0.0
                        ifvg = FVG(
                            index=i, time=curr["time"],
                            fvg_type="iFVG", bias="BULLISH",
                            top=fvg.bottom, bottom=fvg.top,
                            strength=strength, mitigated=False,
                            box={"left_time": prev["time"], "right_time": curr["time"],
                                 "top": fvg.bottom, "bottom": fvg.top}
                        )
                        new_ifvgs.append(ifvg)
                        fvg.mitigated = True

                elif fvg.bias == "BULLISH":
                    if (
                        curr["high"] >= fvg.bottom
                        and curr["low"] <= fvg.top
                        and prev["close"] < fvg.top
                    ):
                        gap_size = fvg.top - fvg.bottom
                        strength = self._fvg_strength(gap_size, atr[i]) if atr[i] > 0 else 0.0
                        ifvg = FVG(
                            index=i, time=curr["time"],
                            fvg_type="iFVG", bias="BEARISH",
                            top=fvg.top, bottom=fvg.bottom,
                            strength=strength, mitigated=False,
                            box={"left_time": prev["time"], "right_time": curr["time"],
                                 "top": fvg.top, "bottom": fvg.bottom}
                        )
                        new_ifvgs.append(ifvg)
                        fvg.mitigated = True

            fvg_list.extend(new_ifvgs)
            active_fvgs = [f for f in active_fvgs if not f.mitigated]

            # Nuevos FVG
            if bullish_fvg:
                gap_size = curr["low"] - prev2["high"]
                strength = self._fvg_strength(gap_size, atr[i]) if atr[i] > 0 else 0.0
                fvg = FVG(
                    index=i, time=curr["time"],
                    fvg_type="FVG", bias="BULLISH",
                    top=float(curr["low"]), bottom=float(prev2["high"]),
                    strength=strength, mitigated=False,
                    box={"left_time": prev["time"], "right_time": curr["time"],
                         "top": float(curr["low"]), "bottom": float(prev2["high"])}
                )
                fvg_list.append(fvg)
                active_fvgs.append(fvg)

            if bearish_fvg:
                gap_size = prev2["low"] - curr["high"]
                strength = self._fvg_strength(gap_size, atr[i]) if atr[i] > 0 else 0.0
                fvg = FVG(
                    index=i, time=curr["time"],
                    fvg_type="FVG", bias="BEARISH",
                    top=float(curr["high"]), bottom=float(prev2["low"]),
                    strength=strength, mitigated=False,
                    box={"left_time": prev["time"], "right_time": curr["time"],
                         "top": float(curr["high"]), "bottom": float(prev2["low"])}
                )
                fvg_list.append(fvg)
                active_fvgs.append(fvg)

            # Mitigación de FVGs
            for fvg in active_fvgs:
                if fvg.mitigated:
                    continue
                if fvg.bias == "BULLISH" and curr["low"] <= fvg.bottom:
                    fvg.mitigated = True
                elif fvg.bias == "BEARISH" and curr["high"] >= fvg.top:
                    fvg.mitigated = True

            active_fvgs = [f for f in active_fvgs if not f.mitigated]

        return fvg_list

    # ============================================================
    # 8. STRONG / WEAK (según trend)
    # ============================================================

    def _classify_strong_weak(self) -> Tuple[str, str]:
        """
        Clasifica el último extremo como Strong/Weak según el trend.
        - Trend BULLISH → Strong Low, Weak High
        - Trend BEARISH → Strong High, Weak Low
        """
        if self.swing_trend.bias == BULLISH:
            return "Weak High", "Strong Low"
        elif self.swing_trend.bias == BEARISH:
            return "Strong High", "Weak Low"
        return "Neutral", "Neutral"

    # ============================================================
    # 9. FIBONACCI AUTOMÁTICO (con etiquetas LuxAlgo)
    # ============================================================

    def _draw_fibonacci(self) -> List[Dict]:
        """Calcula niveles Fibonacci como en Pine Script modificado."""
        if self.trailing.top <= 0 or self.trailing.bottom >= 1e8:
            return []
        if self.trailing.top <= self.trailing.bottom:
            return []

        is_bullish = self.swing_trend.bias == BULLISH
        src_high = float(self.trailing.top)
        src_low = float(self.trailing.bottom)
        src_high_time = self.trailing.last_top_time
        src_low_time = self.trailing.last_bottom_time

        start_time = src_low_time if is_bullish else src_high_time
        end_time = src_high_time if is_bullish else src_low_time

        fib_levels = [0.0, 0.220, 0.382, 0.5, 0.720, 0.786, 1.0, 1.618]
        fib_texts = [
            "0%",
            "22.0%  [COMPRA]",
            "38.2%  [Entrada]",
            "50%",
            "72.0%  [VENTA - Golden]",
            "78.6%",
            "100%",
            "161.8%",
        ]
        fib_show = [True, True, True, True, True, True, True, False]

        result = []
        for j, (level, text, show) in enumerate(zip(fib_levels, fib_texts, fib_show)):
            if not show:
                continue

            if is_bullish:
                price = src_high - level * (src_high - src_low)
            else:
                price = src_low + level * (src_high - src_low)

            label = f"{text}  {price:.2f}"
            result.append({
                "level": level,
                "price": price,
                "label": label,
                "is_buy_zone": level == 0.220,
                "is_sell_zone": level == 0.720,
            })

        return result

    # ============================================================
    # 10. PREMIUM / DISCOUNT ZONES (basado en trailing)
    # ============================================================

    def _compute_premium_discount(self) -> Optional[PremiumDiscountZone]:
        """Calcula zonas Premium/Discount como en Pine (basado en trailing)."""
        if self.trailing.top <= 0 or self.trailing.bottom >= 1e8:
            return None

        top, bot = self.trailing.top, self.trailing.bottom
        if top <= bot:
            return None

        diff = top - bot
        return PremiumDiscountZone(
            swing_high=top,
            swing_low=bot,
            premium_top=top,
            premium_bottom=top - diff * 0.382,
            equilibrium_top=top - diff * 0.382,
            equilibrium_bottom=top - diff * 0.618,
            discount_top=top - diff * 0.618,
            discount_bottom=bot,
        )

    # ============================================================
    # 11. SEÑALES DE TRADING
    # ============================================================

    def generate_signals(
        self,
        df: pd.DataFrame,
        bos_list: List[Dict],
        choch_list: List[Dict],
        fvg_list: List[FVG],
        fib_levels: List[Dict],
        pd_zone: Optional[PremiumDiscountZone],
    ) -> List[TradeSignal]:
        """Genera señales BUY/SELL basadas en SMC."""
        signals = []
        last_idx = len(df) - 1
        if last_idx < 3:
            return signals

        current_price = float(df.iloc[last_idx]["close"])
        current_time = df.iloc[last_idx]["time"]
        current_open = float(df.iloc[last_idx]["open"])

        is_bullish_candle = current_price > current_open
        is_bearish_candle = current_price < current_open

        # Verificar BOS recientes (últimas 20 velas)
        recent_bos_bullish = any(
            b["index"] >= last_idx - 20 and b["type"] == "BULLISH_BOS"
            for b in bos_list
        )
        recent_bos_bearish = any(
            b["index"] >= last_idx - 20 and b["type"] == "BEARISH_BOS"
            for b in bos_list
        )

        # Verificar FVG recientes no mitigados
        recent_bullish_fvg = any(
            f.bias == "BULLISH" and not f.mitigated and f.index >= last_idx - 50
            for f in fvg_list
        )
        recent_bearish_fvg = any(
            f.bias == "BEARISH" and not f.mitigated and f.index >= last_idx - 50
            for f in fvg_list
        )

        # Niveles Fibonacci
        fib_72 = next((f for f in fib_levels if abs(f["level"] - 0.720) < 0.001), None)
        fib_22 = next((f for f in fib_levels if abs(f["level"] - 0.220) < 0.001), None)

        # Zonas
        in_discount = False
        in_premium = False
        if pd_zone:
            in_discount = pd_zone.discount_top >= current_price >= pd_zone.discount_bottom
            in_premium = pd_zone.premium_top >= current_price >= pd_zone.premium_bottom

        # Señal BUY LIMIT en 22% Fibonacci
        factors_buy = []
        buy_score = 0
        if not self.require_bos_for_signal or recent_bos_bullish:
            buy_score += 1
            factors_buy.append("BOS alcista")
        if not self.require_fvg_for_signal or recent_bullish_fvg:
            buy_score += 1
            factors_buy.append("Bullish FVG")
        if in_discount:
            buy_score += 1
            factors_buy.append("Discount Zone")
        if is_bullish_candle:
            buy_score += 1
            factors_buy.append("Vela alcista confirmación")

        can_buy_limit = fib_22 is not None and current_price > fib_22["price"] * 1.001
        if buy_score >= 3 and can_buy_limit:
            limit_price = fib_22["price"]
            sl = limit_price - self.pip_distance
            tp = limit_price + self.pip_distance * self.rr_ratio
            signals.append(TradeSignal(
                index=last_idx, time=current_time,
                action="BUY_LIMIT", entry_price=limit_price,
                limit_price=limit_price,
                stop_loss=sl, take_profit=tp,
                reason=" + ".join(factors_buy),
                confidence=buy_score * 20,
                factors=factors_buy,
            ))

        # Señal SELL LIMIT en 72% Fibonacci
        factors_sell = []
        sell_score = 0
        if not self.require_bos_for_signal or recent_bos_bearish:
            sell_score += 1
            factors_sell.append("BOS bajista")
        if not self.require_fvg_for_signal or recent_bearish_fvg:
            sell_score += 1
            factors_sell.append("Bearish FVG")
        if in_premium:
            sell_score += 1
            factors_sell.append("Premium Zone")
        if is_bearish_candle:
            sell_score += 1
            factors_sell.append("Vela bajista confirmación")

        can_sell_limit = fib_72 is not None and current_price * 1.001 < fib_72["price"]
        if sell_score >= 3 and can_sell_limit:
            limit_price = fib_72["price"]
            sl = limit_price + self.pip_distance
            tp = limit_price - self.pip_distance * self.rr_ratio
            signals.append(TradeSignal(
                index=last_idx, time=current_time,
                action="SELL_LIMIT", entry_price=limit_price,
                limit_price=limit_price,
                stop_loss=sl, take_profit=tp,
                reason=" + ".join(factors_sell),
                confidence=sell_score * 20,
                factors=factors_sell,
            ))

        return signals

    # ============================================================
    # 12. MÉTODO PRINCIPAL analyze()
    # ============================================================

    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Ejecuta el análisis completo SMC LuxAlgo sobre el DataFrame.
        Retorna diccionario con todos los resultados.
        """
        results = {
            "swings": [],
            "bos_list": [],
            "choch_list": [],
            "equal_highs_lows": [],
            "fair_value_gaps": [],
            "order_blocks": [],
            "premium_discount": None,
            "fibonacci": [],
            "signals": [],
            "trend": {},
            "stats": {},
        }

        if df is None or len(df) < 100:
            logger.warning("Datos insuficientes")
            return results

        # Resetear estado
        self.swing_high = Pivot()
        self.swing_low = Pivot()
        self.swing_trend = Trend()
        self.trailing = TrailingExtremes()
        self.swing_order_blocks = []
        self.fair_value_gaps = []
        self.fvg_active = []

        bos_list: List[Dict] = []
        choch_list: List[Dict] = []
        swing_points: List[SwingPoint] = []
        eq_list: List[Dict] = []

        atr = self._calculate_atr(df, 14)

        # --- BUCLE PRINCIPAL (barra por barra como en Pine) ---
        for i in range(self.swing_length, len(df)):
            # 1. Swing detection
            self._get_current_structure(df, i, self.swing_length)

            # 2. BOS / CHoCH
            self._display_structure(df, i, bos_list, choch_list, swing_points)

            # 3. Order blocks mitigation
            self._delete_order_blocks(df, i)

            # 4. Trailing extremes
            self._update_trailing(df, i)

            # 5. EQH/EQL
            eq = self._detect_equal_highs_lows(df, i, self.equal_length, atr)
            if eq:
                eq_list.append(eq)

        # 6. FVG (fuera del bucle para eficiencia)
        fvg_list = self._detect_fvg(df)

        # 7. Fibonacci
        fib_levels = self._draw_fibonacci()

        # 8. Premium/Discount
        pd_zone = self._compute_premium_discount()

        # 9. Strong/Weak
        strong_text, weak_text = self._classify_strong_weak()

        # 10. Señales
        signals = self.generate_signals(
            df, bos_list, choch_list, fvg_list, fib_levels, pd_zone
        )

        # 11. Swing points con clasificación HH/HL/LH/LL
        classified_swings = self._classify_swing_labels(swing_points)

        # 12. Trends
        trend_label = "BULLISH" if self.swing_trend.bias == BULLISH else "BEARISH" if self.swing_trend.bias == BEARISH else "NEUTRAL"

        # --- RESULTADOS ---
        results["swings"] = classified_swings
        results["bos_list"] = bos_list
        results["choch_list"] = choch_list
        results["equal_highs_lows"] = eq_list
        results["fair_value_gaps"] = fvg_list
        results["order_blocks"] = self.get_order_blocks(limit=10)
        results["premium_discount"] = pd_zone
        results["fibonacci"] = fib_levels
        results["signals"] = [{
            "index": s.index,
            "time": s.time,
            "action": s.action,
            "entry_price": s.entry_price,
            "stop_loss": s.stop_loss,
            "take_profit": s.take_profit,
            "reason": s.reason,
            "confidence": s.confidence,
            "factors": s.factors,
        } for s in signals]
        results["trend"] = {
            "swing_bias": self.swing_trend.bias,
            "label": trend_label,
            "strong_label": strong_text,
            "weak_label": weak_text,
        }
        results["trailing"] = {
            "top": self.trailing.top,
            "bottom": self.trailing.bottom,
        }
        results["stats"] = {
            "total_pivots": len(classified_swings),
            "total_bos": len(bos_list),
            "total_choch": len(choch_list),
            "total_fvg": len(fvg_list),
            "total_obs": len(self.swing_order_blocks),
            "total_eq": len(eq_list),
            "total_signals": len(signals),
            "trend": trend_label,
        }

        logger.success(f"Análisis kachazorraz completo | {results['stats']}")
        return results

    def _classify_swing_labels(self, swing_points: List[SwingPoint]) -> List[SwingPoint]:
        """Clasifica swings como HH/HL/LH/LL."""
        if len(swing_points) < 2:
            for s in swing_points:
                s.label = "N/A"
            return swing_points

        last_high = None
        last_low = None

        for s in swing_points:
            if s.swing_type == "HIGH":
                if last_high is None:
                    s.label = "HIGH"
                elif s.price > last_high:
                    s.label = "HH"
                else:
                    s.label = "LH"
                last_high = s.price
            else:
                if last_low is None:
                    s.label = "LOW"
                elif s.price > last_low:
                    s.label = "HL"
                else:
                    s.label = "LL"
                last_low = s.price

        return swing_points

    def get_current_trend(self) -> str:
        """Retorna tendencia actual."""
        return "BULLISH" if self.swing_trend.bias == BULLISH else "BEARISH" if self.swing_trend.bias == BEARISH else "NEUTRAL"


# ============================================================
# FUNCIÓN DE ACCESO RÁPIDO
# ============================================================

def detect_smc(df: pd.DataFrame, **kwargs) -> Dict:
    """Función de conveniencia para ejecutar análisis SMC completo."""
    detector = SmartMoneyConcepts(**kwargs)
    return detector.analyze(df)

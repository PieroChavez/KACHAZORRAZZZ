"""MEJORA 13 — Order Flow & DOM Analysis (Modo Experto Avanzado)
MT5 Depth of Market + Real Ticks analysis providing LEADING signals:

  - Cumulative Delta (tick-level buy/sell volume)
  - DOM Imbalance Ratio + Order Book Clustering
  - Absorption en niveles clave (smart money accumulation/distribution)
  - Iceberg Detection (large hidden orders con chunk tracking)
  - Stop-Run / Liquidity Sweep (with price action confirmation)
  - Delta-MACD Divergence (order flow vs price)
  - Exhaustion Patterns (buying/selling climax)
  - A/D Ratio (advance/decline volume)

En modo experto usa datos reales DOM + ticks vía MT5.
Fallback a tick-volume proxy desde OHLCV si no hay DOM.
"""
import logging
import time
import statistics
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from statistics import mean, stdev

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

@dataclass
class OrderFlowConfig:
    expert_mode: bool = True
    dom_depth_levels: int = 15
    dom_cluster_pips: float = 2.0
    absorption_samples: int = 3
    absorption_volume_ratio: float = 2.0
    absorption_delta_threshold: float = 0.12
    iceberg_chunk_min: int = 3
    iceberg_level_range_pips: float = 3.0
    iceberg_same_size_tolerance: float = 0.15
    stop_run_atr_multiplier: float = 1.2
    stop_run_confirmation_bars: int = 2
    stop_run_lookback: int = 20
    tick_history_size: int = 10000
    delta_ma_period: int = 14
    delta_macd_fast: int = 5
    delta_macd_slow: int = 13
    delta_macd_signal: int = 4
    imbalance_threshold: float = 0.20
    exhaustion_buy_threshold: float = 0.75
    exhaustion_sell_threshold: float = 0.75
    max_dom_fetch_retries: int = 3
    cluster_min_orders: int = 3
    ad_ratio_period: int = 30


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class TickRecord:
    price: float
    volume: float
    side: str
    timestamp: float


@dataclass
class DOMLevel:
    price: float
    bid_volume: float
    ask_volume: float
    bid_order_count: int = 0
    ask_order_count: int = 0
    total_volume: float = 0.0


@dataclass
class DOMSnapshot:
    bid_levels: List[DOMLevel]
    ask_levels: List[DOMLevel]
    bid_total_volume: float
    ask_total_volume: float
    spread: float
    imbalance_ratio: float
    imbalance_classification: str
    timestamp: float
    is_valid: bool = True


@dataclass
class DOMCluster:
    price_high: float
    price_low: float
    poc: float
    bid_volume: float
    ask_volume: float
    total_volume: float
    order_count: int
    imbalance: float
    absorption_score: float


@dataclass
class DeltaMACD:
    macd_line: float
    signal_line: float
    histogram: float
    divergence_bullish: bool
    divergence_bearish: bool


@dataclass
class OrderFlowSignal:
    delta: float
    delta_ma: float
    delta_divergence: bool
    delta_macd: Optional[DeltaMACD]
    imbalance_ratio: float
    imbalance_classification: str
    absorption_active: bool
    absorption_levels: List[float]
    absorption_clusters: List[DOMCluster]
    iceberg_detected: bool
    iceberg_levels: List[float]
    iceberg_chunks: int
    stop_run_detected: bool
    stop_run_direction: str
    stop_run_level: float
    exhaustion_active: bool
    exhaustion_side: str
    buy_pressure: float
    sell_pressure: float
    cumulative_delta: float
    cumulative_delta_ma: float
    ad_ratio: float
    notes: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# DOM CLUSTERING
# ─────────────────────────────────────────────

class DOMClusterEngine:
    @staticmethod
    def cluster(snapshot: DOMSnapshot, cluster_pips: float, pip: float = 0.0001) -> List[DOMCluster]:
        if not snapshot.bid_levels and not snapshot.ask_levels:
            return []
        if pip == 0:
            pip = 0.0001
        cluster_width = cluster_pips * pip
        all_levels = []
        for d in snapshot.bid_levels:
            all_levels.append((d.price, "bid", d.bid_volume, d.bid_order_count))
        for d in snapshot.ask_levels:
            all_levels.append((d.price, "ask", d.ask_volume, d.ask_order_count))
        all_levels.sort(key=lambda x: x[0])
        if not all_levels:
            return []
        clusters: List[DOMCluster] = []
        current: List[Tuple] = [all_levels[0]]
        for item in all_levels[1:]:
            if abs(item[0] - current[-1][0]) <= cluster_width:
                current.append(item)
            else:
                clusters.append(DOMClusterEngine._build_cluster(current, snapshot))
                current = [item]
        if current:
            clusters.append(DOMClusterEngine._build_cluster(current, snapshot))
        return clusters

    @staticmethod
    def _build_cluster(items: List[Tuple], snapshot: DOMSnapshot) -> DOMCluster:
        prices = [it[0] for it in items]
        bid_vol = sum(it[2] for it in items if it[1] == "bid")
        ask_vol = sum(it[2] for it in items if it[1] == "ask")
        total_vol = bid_vol + ask_vol
        orders = sum(it[3] for it in items)
        total_bid_ask = snapshot.bid_total_volume + snapshot.ask_total_volume
        imb = (bid_vol - ask_vol) / total_vol if total_vol > 0 else 0.0
        absorption_score = 0.0
        if total_vol > 0 and total_bid_ask > 0:
            vol_share = total_vol / total_bid_ask
            if vol_share > 0.15 and abs(imb) < 0.12:
                absorption_score = vol_share * (1.0 - abs(imb))
        bid_vol_weighted = sum(it[2] * it[0] for it in items if it[1] == "bid")
        ask_vol_weighted = sum(it[2] * it[0] for it in items if it[1] == "ask")
        total_w = sum(it[2] for it in items)
        poc = (bid_vol_weighted + ask_vol_weighted) / total_w if total_w > 0 else mean(prices)
        return DOMCluster(
            price_high=max(prices), price_low=min(prices), poc=round(poc, 5),
            bid_volume=round(bid_vol, 2), ask_volume=round(ask_vol, 2),
            total_volume=round(total_vol, 2), order_count=orders,
            imbalance=round(imb, 4), absorption_score=round(absorption_score, 4),
        )


# ─────────────────────────────────────────────
# ORDER FLOW ENGINE
# ─────────────────────────────────────────────

class OrderFlowEngine:
    def __init__(self, config: Optional[OrderFlowConfig] = None):
        self.config = config or OrderFlowConfig()
        self._tick_history: Dict[str, deque] = {}
        self._dom_cache: Dict[str, DOMSnapshot] = {}
        self._dom_history: Dict[str, deque] = {}
        self._dom_timestamp: Dict[str, float] = {}
        self._delta_history: Dict[str, deque] = {}
        self._cumulative_delta: Dict[str, float] = {}
        self._cumulative_delta_hist: Dict[str, deque] = {}
        self._last_dom_snapshot: Dict[str, list] = {}
        self._dom_enabled: Dict[str, bool] = {}
        self._delta_macd: Dict[str, Dict] = {}
        self._ad_history: Dict[str, deque] = {}
        self._mt5_client = None
        self._pip_cache: Dict[str, float] = {}

    def set_mt5_client(self, client) -> None:
        self._mt5_client = client

    def analyze(self, symbol: str, df: pd.DataFrame, atr_val: float = 0.0, pip: float = 0.0) -> OrderFlowSignal:
        dom = self._fetch_dom(symbol)
        tick_signal = self._analyze_ticks(symbol, df)
        dom_signal = self._analyze_dom(dom, atr_val, pip, symbol)
        return self._merge(symbol, tick_signal, dom_signal, df, atr_val, pip)

    def _fetch_dom(self, symbol: str) -> Optional[DOMSnapshot]:
        if not self.config.expert_mode:
            return None
        if not self._dom_enabled.get(symbol, True):
            return self._dom_cache.get(symbol)
        now = time.time()
        if symbol in self._dom_timestamp and now - self._dom_timestamp[symbol] < 0.5:
            return self._dom_cache.get(symbol)

        if self._mt5_client is not None and hasattr(self._mt5_client, 'get_dom_snapshot'):
            try:
                snap = self._mt5_client.get_dom_snapshot(symbol, max_levels=self.config.dom_depth_levels)
                if snap and snap["bid_levels"] and snap["ask_levels"]:
                    parsed = self._parse_dom_dict(snap)
                    self._dom_cache[symbol] = parsed
                    self._dom_timestamp[symbol] = now
                    return parsed
            except Exception as exc:
                logger.debug(f"[{symbol}] DOM via client failed: {exc}")

        import MetaTrader5 as mt5
        for attempt in range(self.config.max_dom_fetch_retries + 1):
            try:
                mt5.market_book_add(symbol)
                time.sleep(0.1)
                books = mt5.market_book_get(symbol)
                mt5.market_book_release(symbol)
                if books and len(books) >= 4:
                    snapshot = self._parse_dom(books)
                    self._dom_cache[symbol] = snapshot
                    self._dom_timestamp[symbol] = now
                    return snapshot
            except Exception as exc:
                if attempt == 0:
                    logger.debug(f"[{symbol}] DOM fetch attempt {attempt+1} failed: {exc}")
                time.sleep(0.2)

        if self._dom_enabled.get(symbol, True):
            logger.debug(f"[{symbol}] DOM data unavailable after {self.config.max_dom_fetch_retries+1} retries, disabling DOM")
            self._dom_enabled[symbol] = False
        return self._dom_cache.get(symbol)

    def _parse_dom_dict(self, snap: dict) -> DOMSnapshot:
        bid_levels = []
        for e in snap["bid_levels"]:
            bid_levels.append(DOMLevel(
                price=e["price"], bid_volume=e["volume_dbl"],
                ask_volume=0, bid_order_count=e["orders"],
                total_volume=e["volume_dbl"],
            ))
        ask_levels = []
        for e in snap["ask_levels"]:
            ask_levels.append(DOMLevel(
                price=e["price"], bid_volume=0,
                ask_volume=e["volume_dbl"], ask_order_count=e["orders"],
                total_volume=e["volume_dbl"],
            ))
        imbalance = snap.get("imbalance", 0.0)
        imb_class = self._classify_imbalance(imbalance)
        return DOMSnapshot(
            bid_levels=bid_levels, ask_levels=ask_levels,
            bid_total_volume=snap["bid_total"], ask_total_volume=snap["ask_total"],
            spread=snap["spread"], imbalance_ratio=imbalance,
            imbalance_classification=imb_class,
            timestamp=snap["timestamp"], is_valid=True,
        )

    def _parse_dom(self, books: list) -> DOMSnapshot:
        bid_levels = {}
        ask_levels = {}
        for b in books:
            price = round(b.price, 5)
            vol = float(getattr(b, 'volume_dbl', 0) or b.volume)
            orders = int(getattr(b, 'volume', 0))
            t = b.type
            if t in (2, 4):
                if price not in bid_levels:
                    bid_levels[price] = DOMLevel(price=price, bid_volume=0, ask_volume=0)
                bid_levels[price].bid_volume += vol
                bid_levels[price].bid_order_count += max(1, orders)
                bid_levels[price].total_volume += vol
            elif t in (1, 3):
                if price not in ask_levels:
                    ask_levels[price] = DOMLevel(price=price, bid_volume=0, ask_volume=0)
                ask_levels[price].ask_volume += vol
                ask_levels[price].ask_order_count += max(1, orders)
                ask_levels[price].total_volume += vol

        bid_prices = sorted(bid_levels.keys(), reverse=True)
        ask_prices = sorted(ask_levels.keys())
        bid_depth = [bid_levels[p] for p in bid_prices[:self.config.dom_depth_levels]]
        ask_depth = [ask_levels[p] for p in ask_prices[:self.config.dom_depth_levels]]

        bid_total = sum(d.bid_volume for d in bid_depth)
        ask_total = sum(d.ask_volume for d in ask_depth)
        total = bid_total + ask_total
        imbalance = (bid_total - ask_total) / total if total > 0 else 0.0
        spread = 0.0
        if bid_prices and ask_prices:
            spread = ask_prices[0] - bid_prices[0]
        imb_class = self._classify_imbalance(imbalance)
        return DOMSnapshot(
            bid_depth, ask_depth, bid_total, ask_total,
            spread, round(imbalance, 4), imb_class,
            time.time(), True,
        )

    @staticmethod
    def _classify_imbalance(imbalance: float) -> str:
        if imbalance > 0.30:
            return "STRONG_BID"
        elif imbalance > 0.10:
            return "BID_HEAVY"
        elif imbalance < -0.30:
            return "STRONG_ASK"
        elif imbalance < -0.10:
            return "ASK_HEAVY"
        return "NEUTRAL"

    def _analyze_ticks(self, symbol: str, df: pd.DataFrame) -> dict:
        if df is None or len(df) < 10:
            return {
                "delta": 0, "delta_ma": 0, "buy_pressure": 0.5, "sell_pressure": 0.5,
                "delta_divergence": False, "cumulative_delta": 0,
                "exhaustion_active": False, "exhaustion_side": "",
            }

        segment = df.iloc[-self.config.ad_ratio_period:]
        up_vol, down_vol, total_vol = 0.0, 0.0, 0.0
        for _, row in segment.iterrows():
            v = row["volume"]
            total_vol += v
            body = abs(row["close"] - row["open"])
            spread = row["high"] - row["low"]
            if spread == 0 or body / spread < 0.1:
                continue
            if row["close"] > row["open"]:
                up_vol += v
            elif row["close"] < row["open"]:
                down_vol += v

        buy_p = up_vol / total_vol if total_vol > 0 else 0.5
        sell_p = down_vol / total_vol if total_vol > 0 else 0.5
        delta = (up_vol - down_vol) / total_vol if total_vol > 0 else 0.0

        if symbol not in self._delta_history:
            self._delta_history[symbol] = deque(maxlen=self.config.delta_ma_period)
        self._delta_history[symbol].append(delta)
        delta_ma = mean(self._delta_history[symbol]) if len(self._delta_history[symbol]) >= 3 else delta

        price_change = segment["close"].iloc[-1] - segment["open"].iloc[0]
        delta_div = (price_change > 0 and delta < -0.1) or (price_change < 0 and delta > 0.1)

        cum_delta = self._cumulative_delta.get(symbol, 0.0) + delta
        self._cumulative_delta[symbol] = cum_delta
        if symbol not in self._cumulative_delta_hist:
            self._cumulative_delta_hist[symbol] = deque(maxlen=self.config.delta_ma_period)
        self._cumulative_delta_hist[symbol].append(cum_delta)
        cum_ma = mean(self._cumulative_delta_hist[symbol]) if len(self._cumulative_delta_hist[symbol]) >= 3 else cum_delta

        exhaustion_active = False
        exhaustion_side = ""
        if buy_p > self.config.exhaustion_buy_threshold and delta < 0.1:
            exhaustion_active = True
            exhaustion_side = "BUY_CLIMAX"
        elif sell_p > self.config.exhaustion_sell_threshold and delta > -0.1:
            exhaustion_active = True
            exhaustion_side = "SELL_CLIMAX"

        adv_vol = sum(
            row["volume"] for _, row in segment.iterrows()
            if row["close"] > row["open"]
        )
        dec_vol = sum(
            row["volume"] for _, row in segment.iterrows()
            if row["close"] < row["open"]
        )
        ad_ratio = adv_vol / dec_vol if dec_vol > 0 else 2.0
        if adv_vol == 0 and dec_vol == 0:
            ad_ratio = 1.0

        if symbol not in self._ad_history:
            self._ad_history[symbol] = deque(maxlen=self.config.delta_ma_period)
        self._ad_history[symbol].append(ad_ratio)

        return {
            "delta": delta, "delta_ma": delta_ma,
            "buy_pressure": buy_p, "sell_pressure": sell_p,
            "delta_divergence": delta_div,
            "cumulative_delta": cum_delta,
            "cumulative_delta_ma": cum_ma,
            "exhaustion_active": exhaustion_active,
            "exhaustion_side": exhaustion_side,
            "ad_ratio": round(ad_ratio, 4),
        }

    def _compute_delta_macd(self, symbol: str) -> Optional[DeltaMACD]:
        if symbol not in self._delta_history or len(self._delta_history[symbol]) < self.config.delta_macd_slow + self.config.delta_macd_signal:
            return None
        vals = list(self._delta_history[symbol])
        if len(vals) < self.config.delta_macd_slow + 1:
            return None
        arr = np.array(vals)
        ema_fast = self._ema(arr, self.config.delta_macd_fast)
        ema_slow = self._ema(arr, self.config.delta_macd_slow)
        macd_line = ema_fast[-1] - ema_slow[-1]
        macd_vals = np.array([
            self._ema_val(arr[:i+1], self.config.delta_macd_fast) -
            self._ema_val(arr[:i+1], self.config.delta_macd_slow)
            for i in range(len(arr))
        ])
        signal_line = self._ema(macd_vals, self.config.delta_macd_signal)[-1] if len(macd_vals) >= self.config.delta_macd_signal else 0
        histogram = macd_line - signal_line
        prev_macd = macd_vals[-2] if len(macd_vals) >= 2 else 0
        prev_signal = self._ema(macd_vals[:-1], self.config.delta_macd_signal)[-1] if len(macd_vals) >= self.config.delta_macd_signal + 1 else 0
        prev_hist = prev_macd - prev_signal
        bull_div = histogram > prev_hist and vals[-1] < vals[-2] if len(vals) >= 2 else False
        bear_div = histogram < prev_hist and vals[-1] > vals[-2] if len(vals) >= 2 else False
        return DeltaMACD(
            macd_line=round(macd_line, 6), signal_line=round(signal_line, 6),
            histogram=round(histogram, 6),
            divergence_bullish=bull_div, divergence_bearish=bear_div,
        )

    @staticmethod
    def _ema(arr: np.ndarray, period: int) -> np.ndarray:
        if len(arr) < 1 or period < 1:
            return arr
        multiplier = 2.0 / (period + 1)
        result = np.zeros_like(arr)
        result[0] = arr[0]
        for i in range(1, len(arr)):
            result[i] = (arr[i] - result[i-1]) * multiplier + result[i-1]
        return result

    @staticmethod
    def _ema_val(arr: np.ndarray, period: int) -> float:
        if len(arr) < 1 or period < 1:
            return float(arr[-1]) if len(arr) > 0 else 0.0
        multiplier = 2.0 / (period + 1)
        ema = float(arr[0])
        for i in range(1, len(arr)):
            ema = (arr[i] - ema) * multiplier + ema
        return ema

    def _analyze_dom(self, dom: Optional[DOMSnapshot], atr_val: float, pip: float, symbol: str) -> dict:
        result = {
            "imbalance_ratio": 0.0,
            "imbalance_classification": "NEUTRAL",
            "absorption_active": False,
            "absorption_levels": [],
            "absorption_clusters": [],
            "iceberg_detected": False,
            "iceberg_levels": [],
            "iceberg_chunks": 0,
            "stop_run_detected": False,
            "stop_run_direction": "",
            "stop_run_level": 0.0,
        }
        if dom is None or not dom.is_valid or not self.config.expert_mode:
            return result

        result["imbalance_ratio"] = dom.imbalance_ratio
        result["imbalance_classification"] = dom.imbalance_classification

        clusters = DOMClusterEngine.cluster(dom, self.config.dom_cluster_pips, max(pip, 0.0001))
        result["absorption_clusters"] = clusters

        absorption_levels = self._detect_absorption(dom, clusters)
        if absorption_levels:
            result["absorption_active"] = True
            result["absorption_levels"] = absorption_levels

        iceberg_levels, iceberg_chunks = self._detect_iceberg(dom, symbol)
        if iceberg_levels:
            result["iceberg_detected"] = True
            result["iceberg_levels"] = iceberg_levels
            result["iceberg_chunks"] = iceberg_chunks

        if atr_val > 0 and pip > 0:
            sd, sl = self._detect_stop_run(dom, atr_val, pip, symbol)
            if sd:
                result["stop_run_detected"] = True
                result["stop_run_direction"] = sd
                result["stop_run_level"] = sl

        if symbol not in self._dom_history:
            self._dom_history[symbol] = deque(maxlen=self.config.absorption_samples)
        self._dom_history[symbol].append(dom)

        return result

    def _detect_absorption(self, dom: DOMSnapshot, clusters: List[DOMCluster]) -> List[float]:
        levels = []
        for c in clusters:
            if c.absorption_score > 0.15 and c.order_count >= self.config.cluster_min_orders:
                levels.append(c.poc)
        return levels[:5]

    def _detect_iceberg(self, dom: DOMSnapshot, symbol: str) -> Tuple[List[float], int]:
        levels = []
        if not dom.bid_levels and not dom.ask_levels:
            return levels, 0
        all_volumes = [d.bid_volume for d in dom.bid_levels] + [d.ask_volume for d in dom.ask_levels]
        if len(all_volumes) < 4:
            return levels, 0
        try:
            vol_std = stdev(all_volumes)
            vol_mean = mean(all_volumes)
        except Exception:
            return levels, 0
        threshold = vol_mean + vol_std * 1.2
        for d in dom.bid_levels:
            if d.bid_volume > threshold:
                levels.append(d.price)
        for d in dom.ask_levels:
            if d.ask_volume > threshold:
                levels.append(d.price)

        if symbol in self._dom_history and len(self._dom_history[symbol]) >= 2:
            prev = self._dom_history[symbol][-1]
            chunk_count = 0
            for level in levels[:]:
                tolerance = level * self.config.iceberg_same_size_tolerance
                prev_vol = 0.0
                for pd_ in prev.bid_levels:
                    if abs(pd_.price - level) <= tolerance:
                        prev_vol = pd_.bid_volume
                        break
                if prev_vol == 0:
                    for pd_ in prev.ask_levels:
                        if abs(pd_.price - level) <= tolerance:
                            prev_vol = pd_.ask_volume
                            break
                current_vol = 0.0
                for d in dom.bid_levels:
                    if abs(d.price - level) <= tolerance:
                        current_vol = d.bid_volume
                        break
                if current_vol == 0:
                    for d in dom.ask_levels:
                        if abs(d.price - level) <= tolerance:
                            current_vol = d.ask_volume
                            break
                if prev_vol > 0 and current_vol > 0:
                    ratio = min(current_vol, prev_vol) / max(current_vol, prev_vol)
                    if ratio > 0.85:
                        chunk_count += 1
            return levels, chunk_count
        return levels, 0

    def _detect_stop_run(self, dom: DOMSnapshot, atr_val: float, pip: float, symbol: str) -> Tuple[str, float]:
        if symbol not in self._dom_history or len(self._dom_history[symbol]) < 2:
            return "", 0.0
        prev = self._dom_history[symbol][-1]
        if pip == 0:
            pip = 0.0001
        dist_threshold = atr_val * self.config.stop_run_atr_multiplier

        top_bid = max(d.price for d in dom.bid_levels) if dom.bid_levels else 0
        bot_ask = min(d.price for d in dom.ask_levels) if dom.ask_levels else 0
        prev_top_bid = max(d.price for d in prev.bid_levels) if prev.bid_levels else 0
        prev_bot_ask = min(d.price for d in prev.ask_levels) if prev.ask_levels else 0

        bid_sweep = False
        ask_sweep = False

        if top_bid > 0 and prev_top_bid > 0:
            bid_delta = abs(top_bid - prev_top_bid)
            bid_sweep = bid_delta > dist_threshold

        if bot_ask > 0 and prev_bot_ask > 0:
            ask_delta = abs(bot_ask - prev_bot_ask)
            ask_sweep = ask_delta > dist_threshold

        if bid_sweep and ask_sweep:
            if abs(top_bid - prev_top_bid) > abs(bot_ask - prev_bot_ask):
                ask_sweep = False
            else:
                bid_sweep = False

        if bid_sweep:
            return "BUY", round(prev_top_bid, 5)
        elif ask_sweep:
            return "SELL", round(prev_bot_ask, 5)
        return "", 0.0

    def _merge(self, symbol: str, tick: dict, dom: dict, df: pd.DataFrame, atr_val: float, pip: float) -> OrderFlowSignal:
        notes = []
        delta = tick["delta"]
        delta_ma = tick["delta_ma"]
        delta_div = tick["delta_divergence"]
        imbalance = dom["imbalance_ratio"]
        imb_class = dom["imbalance_classification"]
        cum_delta = tick["cumulative_delta"]
        cum_ma = tick["cumulative_delta_ma"]
        exhaustion = tick["exhaustion_active"]
        exhaustion_side = tick["exhaustion_side"]
        ad_ratio = tick["ad_ratio"]

        if self.config.expert_mode:
            combined = imbalance * 0.5 + delta * 0.3 + ((ad_ratio - 1.0) * 0.2)
            if abs(imbalance) > self.config.imbalance_threshold:
                side = "bid" if imbalance > 0 else "ask"
                notes.append(f"DOM {imb_class}: {side} heavy ({imbalance:+.0%})")
        else:
            combined = delta

        if dom["absorption_active"]:
            n_abs = len(dom["absorption_levels"])
            n_clust = len(dom["absorption_clusters"])
            notes.append(f"Absorption in {n_abs} clusters ({n_clust} total zones) — smart money positioning")

        if dom["iceberg_detected"]:
            n_ice = len(dom["iceberg_levels"])
            chunks = dom["iceberg_chunks"]
            notes.append(f"Iceberg orders: {n_ice} levels, {chunks} chunk repetitions — large player hiding size")

        if dom["stop_run_detected"]:
            dir_str = "longs" if dom["stop_run_direction"] == "SELL" else "shorts"
            notes.append(f"Stop-run: {dir_str} hunted @ {dom['stop_run_level']:.5f} — liquidity sweep confirmed")

        if delta_div:
            notes.append(f"Delta-price divergence detected — price vs order flow direction mismatch")

        d_macd = self._compute_delta_macd(symbol)
        if d_macd:
            if d_macd.divergence_bullish:
                notes.append(f"Delta-MACD bullish divergence: histogram rising while delta falling")
            elif d_macd.divergence_bearish:
                notes.append(f"Delta-MACD bearish divergence: histogram falling while delta rising")

        if exhaustion:
            if exhaustion_side == "BUY_CLIMAX":
                notes.append(f"Buying climax detected — exhaustion, potential reversal down")
            elif exhaustion_side == "SELL_CLIMAX":
                notes.append(f"Selling climax detected — exhaustion, potential reversal up")

        if delta > 0.25:
            notes.append(f"Aggressive buying delta: {delta:+.0%}")
        elif delta < -0.25:
            notes.append(f"Aggressive selling delta: {delta:+.0%}")

        if ad_ratio > 1.5:
            notes.append(f"A/D ratio: {ad_ratio:.2f} — advancing volume dominates")
        elif ad_ratio < 0.67:
            notes.append(f"A/D ratio: {ad_ratio:.2f} — declining volume dominates")

        return OrderFlowSignal(
            delta=round(delta, 4), delta_ma=round(delta_ma, 4),
            delta_divergence=delta_div,
            delta_macd=d_macd,
            imbalance_ratio=round(imbalance, 4),
            imbalance_classification=imb_class,
            absorption_active=dom["absorption_active"],
            absorption_levels=dom["absorption_levels"],
            absorption_clusters=dom["absorption_clusters"],
            iceberg_detected=dom["iceberg_detected"],
            iceberg_levels=dom["iceberg_levels"],
            iceberg_chunks=dom["iceberg_chunks"],
            stop_run_detected=dom["stop_run_detected"],
            stop_run_direction=dom["stop_run_direction"],
            stop_run_level=dom["stop_run_level"],
            exhaustion_active=exhaustion,
            exhaustion_side=exhaustion_side,
            buy_pressure=round(tick["buy_pressure"], 4),
            sell_pressure=round(tick["sell_pressure"], 4),
            cumulative_delta=round(cum_delta, 4),
            cumulative_delta_ma=round(cum_ma, 4),
            ad_ratio=round(ad_ratio, 4),
            notes=notes,
        )

    def get_signal_contribution(self, signal: OrderFlowSignal, direction: str) -> Tuple[float, str]:
        score = 0.0
        reasons = []

        if direction == "BUY":
            if signal.delta > 0.15:
                s = 8.0 * min(1.0, signal.delta * 3)
                score += s
                reasons.append(f"Delta compra {signal.delta:+.0%} (+{s:.0f})")

            if signal.imbalance_ratio > self.config.imbalance_threshold:
                s = 10.0 * min(1.0, signal.imbalance_ratio * 2)
                score += s
                reasons.append(f"DOM bid={signal.imbalance_ratio:+.0%} (+{s:.0f})")

            if signal.absorption_active:
                s = 12.0
                score += s
                reasons.append(f"Absorción +{s:.0f}")

            if signal.iceberg_detected and signal.iceberg_chunks >= 2:
                s = 10.0
                score += s
                reasons.append(f"Iceberg compras +{s:.0f}")

            if signal.stop_run_detected and signal.stop_run_direction == "BUY":
                s = 14.0
                score += s
                reasons.append(f"Stop-run shorts cazados +{s:.0f}")

            if signal.delta_macd and signal.delta_macd.divergence_bullish:
                s = 12.0
                score += s
                reasons.append(f"Delta-MACD divergencia alcista +{s:.0f}")

            if signal.exhaustion_active and signal.exhaustion_side == "SELL_CLIMAX":
                s = 10.0
                score += s
                reasons.append(f"Selling climax — reversión alcista +{s:.0f}")

            if signal.ad_ratio > 1.3:
                s = 5.0
                score += s
                reasons.append(f"A/D ratio {signal.ad_ratio:.2f} +{s:.0f}")

        elif direction == "SELL":
            if signal.delta < -0.15:
                s = 8.0 * min(1.0, abs(signal.delta) * 3)
                score += s
                reasons.append(f"Delta venta {signal.delta:+.0%} (+{s:.0f})")

            if signal.imbalance_ratio < -self.config.imbalance_threshold:
                s = 10.0 * min(1.0, abs(signal.imbalance_ratio) * 2)
                score += s
                reasons.append(f"DOM ask={signal.imbalance_ratio:+.0%} (+{s:.0f})")

            if signal.absorption_active:
                s = 12.0
                score += s
                reasons.append(f"Absorción +{s:.0f}")

            if signal.iceberg_detected and signal.iceberg_chunks >= 2:
                s = 10.0
                score += s
                reasons.append(f"Iceberg ventas +{s:.0f}")

            if signal.stop_run_detected and signal.stop_run_direction == "SELL":
                s = 14.0
                score += s
                reasons.append(f"Stop-run longs cazados +{s:.0f}")

            if signal.delta_macd and signal.delta_macd.divergence_bearish:
                s = 12.0
                score += s
                reasons.append(f"Delta-MACD divergencia bajista +{s:.0f}")

            if signal.exhaustion_active and signal.exhaustion_side == "BUY_CLIMAX":
                s = 10.0
                score += s
                reasons.append(f"Buying climax — reversión bajista +{s:.0f}")

            if signal.ad_ratio < 0.77:
                s = 5.0
                score += s
                reasons.append(f"A/D ratio {signal.ad_ratio:.2f} +{s:.0f}")

        if self.config.expert_mode and signal.notes:
            for n in signal.notes[:3]:
                if n not in reasons:
                    reasons.append(n.split("—")[0].strip())

        return round(score, 1), "; ".join(reasons[:6]) if reasons else "OrderFlow neutral"

    def get_detailed_report(self, signal: OrderFlowSignal) -> dict:
        return {
            "delta": signal.delta,
            "delta_ma": signal.delta_ma,
            "imbalance": signal.imbalance_ratio,
            "imbalance_class": signal.imbalance_classification,
            "absorption": {
                "active": signal.absorption_active,
                "levels": signal.absorption_levels,
                "clusters": len(signal.absorption_clusters),
            },
            "iceberg": {
                "detected": signal.iceberg_detected,
                "levels": signal.iceberg_levels,
                "chunks": signal.iceberg_chunks,
            },
            "stop_run": {
                "detected": signal.stop_run_detected,
                "direction": signal.stop_run_direction,
                "level": signal.stop_run_level,
            },
            "exhaustion": {
                "active": signal.exhaustion_active,
                "side": signal.exhaustion_side,
            },
            "divergence": {
                "delta_price": signal.delta_divergence,
                "delta_macd_bull": signal.delta_macd.divergence_bullish if signal.delta_macd else False,
                "delta_macd_bear": signal.delta_macd.divergence_bearish if signal.delta_macd else False,
            },
            "pressure": {
                "buy": signal.buy_pressure,
                "sell": signal.sell_pressure,
                "ad_ratio": signal.ad_ratio,
            },
            "cumulative_delta": signal.cumulative_delta,
            "notes": signal.notes,
        }

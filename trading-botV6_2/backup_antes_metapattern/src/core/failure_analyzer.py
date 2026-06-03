"""Failure Analyzer — Detección y Desactivación Automática de Patrones Fallidos (Mejora 6, Modo Experto)
Analiza resultados de trades, detecta patrones que fallan consistentemente
y los desactiva temporalmente sin intervención manual.
"""
import logging
import math
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "min_samples": 5,
    "max_loss_rate": 0.70,
    "max_consecutive_losses": 5,
    "cooldown_minutes": 120,
    "confidence_level": 0.95,
    "volume_reduction_steps": [0.5, 0.25, 0.0],
    "decay_half_life_hours": 4,
    "re_enable_threshold": 0.50,
    "re_enable_min_samples": 3,
}

EXPERT_CONFIG = {
    "min_samples": 4,
    "max_loss_rate": 0.65,
    "max_consecutive_losses": 4,
    "cooldown_minutes": 90,
    "confidence_level": 0.90,
    "volume_reduction_steps": [0.6, 0.3, 0.0],
    "decay_half_life_hours": 3,
    "re_enable_threshold": 0.55,
    "re_enable_min_samples": 3,
    "wilson_score": True,
    "similar_patterns": True,
    "auto_adjust_thresholds": True,
    "per_symbol_mode": True,
    "per_regime_mode": True,
    "per_session_mode": True,
    "pattern_similarity_threshold": 0.7,
}


@dataclass
class TradePostmortem:
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    profit: float
    score: float
    conviction: float
    regime: str
    session: str
    primary_pattern: Optional[str]
    patterns_found: List[str]
    exit_reason: str
    timestamp: float
    duration_minutes: float


class PatternStats:
    def __init__(self):
        self.wins: List[float] = []
        self.losses: List[float] = []
        self.consecutive_losses = 0
        self.total_losses = 0
        self.total_wins = 0
        self.disabled_until: float = 0
        self.volume_multiplier = 1.0
        self.last_result_time: float = 0
        self._timestamps: List[float] = []

    @property
    def total(self) -> int:
        return self.total_wins + self.total_losses

    @property
    def loss_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.total_losses / self.total

    @property
    def win_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.total_wins / self.total

    @property
    def avg_pnl(self) -> float:
        all_pnl = self.wins + self.losses
        if not all_pnl:
            return 0.0
        return sum(all_pnl) / len(all_pnl)

    def is_disabled(self) -> bool:
        return time.time() < self.disabled_until

    def record(self, result: str, pnl_pct: float, timestamp: float = None):
        ts = timestamp or time.time()
        self._timestamps.append(ts)
        if result == "win":
            self.wins.append(pnl_pct)
            self.total_wins += 1
            self.consecutive_losses = 0
        else:
            self.losses.append(pnl_pct)
            self.total_losses += 1
            self.consecutive_losses += 1
        self.last_result_time = ts
        self._trim()

    def _trim(self, max_samples: int = 100):
        if len(self._timestamps) > max_samples:
            self.wins = self.wins[-max_samples:]
            self.losses = self.losses[-max_samples:]
            self._timestamps = self._timestamps[-max_samples:]

    def weighted_win_rate(self, half_life_hours: float = 4) -> float:
        if self.total == 0:
            return 0.0
        now = time.time()
        half_life = half_life_hours * 3600
        weights = [math.exp(-(now - ts) / half_life) for ts in self._timestamps]
        if not weights:
            return self.win_rate
        total_w = sum(weights)
        if total_w == 0:
            return self.win_rate
        weighted_wins = sum(w for w, r in zip(weights, self._get_results()) if r == "win")
        return weighted_wins / total_w

    def _get_results(self) -> List[str]:
        results = ["win"] * len(self.wins) + ["loss"] * len(self.losses)
        zipped = sorted(zip(self._timestamps, results), key=lambda x: x[0])
        return [r for _, r in zipped]

    def wilson_lower_bound(self, confidence: float = 0.95) -> float:
        if self.total == 0:
            return 0.0
        z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
        p = self.win_rate
        n = self.total
        denominator = 1 + z * z / n
        center = p + z * z / (2 * n)
        margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return max(0, (center - margin) / denominator)

    def should_disable(self, config: Dict) -> Tuple[bool, str]:
        if self.is_disabled():
            return False, "already_disabled"
        if self.total < config["min_samples"]:
            return False, "insufficient_samples"
        if self.consecutive_losses >= config["max_consecutive_losses"]:
            return True, f"{self.consecutive_losses} pérdidas consecutivas"
        if config.get("wilson_score"):
            lb = self.wilson_lower_bound(config["confidence_level"])
            if lb < 1 - config["max_loss_rate"] and self.total >= config["min_samples"]:
                return True, f"Wilson CI < {1-config['max_loss_rate']:.0%}"
        if self.loss_rate >= config["max_loss_rate"]:
            return True, f"loss_rate={self.loss_rate:.0%} >= {config['max_loss_rate']:.0%}"
        return False, ""

    def disable(self, cooldown_minutes: int):
        self.disabled_until = time.time() + cooldown_minutes * 60
        self.consecutive_losses = 0

    def should_re_enable(self, config: Dict) -> bool:
        if not self.is_disabled():
            return False
        if self.total < config["re_enable_min_samples"]:
            return False
        return self.win_rate >= config["re_enable_threshold"]

    def reduce_volume(self) -> float:
        steps = [1.0, 0.6, 0.3, 0.0]
        idx = min(len(steps) - 1, self.consecutive_losses // 2)
        self.volume_multiplier = steps[idx]
        return self.volume_multiplier


class FailureAnalyzer:
    def __init__(self, db_or_config=None, expert_mode: bool = True):
        if isinstance(db_or_config, (str, Path)):
            self.db_path = Path(db_or_config)
            self.cfg = {**EXPERT_CONFIG}
        elif isinstance(db_or_config, dict):
            self.db_path = None
            self.cfg = {**(EXPERT_CONFIG if expert_mode else DEFAULT_CONFIG), **db_or_config}
        else:
            self.db_path = None
            self.cfg = {**(EXPERT_CONFIG if expert_mode else DEFAULT_CONFIG)}
        self.expert_mode = expert_mode
        self._conn: Optional[sqlite3.Connection] = None
        self._global_stats: Dict[str, PatternStats] = defaultdict(PatternStats)
        self._symbol_stats: Dict[str, Dict[str, PatternStats]] = defaultdict(lambda: defaultdict(PatternStats))
        self._regime_stats: Dict[str, Dict[str, PatternStats]] = defaultdict(lambda: defaultdict(PatternStats))
        self._session_stats: Dict[str, Dict[str, PatternStats]] = defaultdict(lambda: defaultdict(PatternStats))
        self._trade_history: List[TradePostmortem] = []
        self._disabled_log: List[Dict] = []
        self._re_enabled_log: List[Dict] = []
        self._similar_patterns_cache: Dict[str, List[str]] = {}
        self._thompson_sampler = None

    def set_thompson_sampler(self, sampler) -> None:
        """Link ThompsonSampler from AdaptiveCooldownEngine for probabilistic re-enablement."""
        self._thompson_sampler = sampler
        logger.info("[FAILURE] Thompson Sampler vinculado para re-activación probabilística")

    def initialize(self):
        if self.db_path:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS failure_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT, direction TEXT, pattern TEXT,
                    profit REAL, score REAL, conviction REAL,
                    regime TEXT, session TEXT, exit_reason TEXT,
                    timestamp REAL, duration_minutes REAL
                )
            """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS failure_disabled (
                    pattern TEXT, scope TEXT, reason TEXT,
                    disabled_at REAL, cooldown_minutes INTEGER
                )
            """)
            self._conn.commit()
            self._load_from_db()
        logger.info(f"[FAILURE] FailureAnalyzer initialized (expert_mode={self.expert_mode})")

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def _load_from_db(self):
        if not self._conn:
            return
        try:
            cursor = self._conn.execute(
                "SELECT symbol, direction, pattern, profit, score, conviction, "
                "regime, session, exit_reason, timestamp, duration_minutes "
                "FROM failure_trades ORDER BY id DESC LIMIT 500"
            )
            for row in cursor.fetchall():
                symbol, direction, pattern, profit, score, conviction, regime, session, exit_reason, ts, duration = row
                result = "win" if profit >= 0 else "loss"
                pnl_pct = profit
                if pattern:
                    self._global_stats[pattern].record(result, pnl_pct, ts)
                    self._symbol_stats[symbol][pattern].record(result, pnl_pct, ts)
                    if regime:
                        self._regime_stats[regime][pattern].record(result, pnl_pct, ts)
                    if session:
                        self._session_stats[session][pattern].record(result, pnl_pct, ts)
            logger.info(f"[FAILURE] Cargados {cursor.rowcount} trades desde DB")
        except Exception as e:
            logger.warning(f"[FAILURE] Error cargando DB: {e}")

    def record_trade(self, pm: TradePostmortem):
        pattern = pm.primary_pattern or "UNKNOWN"
        result = "win" if pm.profit >= 0 else "loss"
        pnl_pct = pm.profit

        self._trade_history.append(pm)
        if len(self._trade_history) > 1000:
            self._trade_history = self._trade_history[-1000:]

        self._global_stats[pattern].record(result, pnl_pct, pm.timestamp)
        self._symbol_stats[pm.symbol][pattern].record(result, pnl_pct, pm.timestamp)
        if pm.regime:
            self._regime_stats[pm.regime][pattern].record(result, pnl_pct, pm.timestamp)
        if pm.session:
            self._session_stats[pm.session][pattern].record(result, pnl_pct, pm.timestamp)

        self._check_auto_disable(pm.symbol, pattern, pm.regime, pm.session)
        self._check_auto_re_enable(pm.symbol, pattern, pm.regime, pm.session)

        self._save_to_db(pm, pattern, result)

    def _save_to_db(self, pm: TradePostmortem, pattern: str, result: str):
        if not self._conn:
            return
        try:
            self._conn.execute(
                "INSERT INTO failure_trades "
                "(symbol, direction, pattern, profit, score, conviction, "
                "regime, session, exit_reason, timestamp, duration_minutes) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (pm.symbol, pm.direction, pattern, pm.profit, pm.score,
                 pm.conviction, pm.regime, pm.session, pm.exit_reason,
                 pm.timestamp, pm.duration_minutes),
            )
            self._conn.commit()
        except Exception as e:
            logger.warning(f"[FAILURE] Error guardando en DB: {e}")

    def _check_auto_disable(self, symbol: str, pattern: str,
                            regime: str, session: str):
        stats_list = [(self._global_stats[pattern], "global")]
        if self.expert_mode:
            stats_list.append((self._symbol_stats[symbol][pattern], f"symbol={symbol}"))
            if regime:
                stats_list.append((self._regime_stats[regime][pattern], f"regime={regime}"))
            if session:
                stats_list.append((self._session_stats[session][pattern], f"session={session}"))

        for stats, scope in stats_list:
            should, reason = stats.should_disable(self.cfg)
            if should:
                cooldown = self.cfg["cooldown_minutes"]
                if self._thompson_sampler and stats.consecutive_losses >= 3:
                    cooldown = min(cooldown * (1.0 + stats.consecutive_losses * 0.5), 1440.0)
                    self._thompson_sampler.disable(pattern, cooldown)
                stats.disable(cooldown)
                entry = {
                    "pattern": pattern, "scope": scope,
                    "reason": reason, "time": time.time(),
                    "cooldown": cooldown,
                    "stats": {
                        "total": stats.total, "wins": stats.total_wins,
                        "losses": stats.total_losses,
                        "loss_rate": stats.loss_rate,
                        "consecutive_losses": stats.consecutive_losses,
                    },
                }
                self._disabled_log.append(entry)
                logger.warning(
                    f"[FAILURE] Patrón '{pattern}' DESACTIVADO ({scope}): "
                    f"{reason} ({stats.total_wins}W/{stats.total_losses}L "
                    f"rate={stats.loss_rate:.0%}, cooldown={cooldown:.0f}min)"
                )
                self._save_disabled_to_db(pattern, scope, reason, cooldown)
                if self.expert_mode and len(self._disabled_log) > 200:
                    self._disabled_log = self._disabled_log[-200:]

    def _save_disabled_to_db(self, pattern: str, scope: str,
                             reason: str, cooldown: int):
        if not self._conn:
            return
        try:
            self._conn.execute(
                "INSERT INTO failure_disabled "
                "(pattern, scope, reason, disabled_at, cooldown_minutes) "
                "VALUES (?,?,?,?,?)",
                (pattern, scope, reason, time.time(), cooldown),
            )
            self._conn.commit()
        except Exception as e:
            logger.warning(f"[FAILURE] Error guardando disable en DB: {e}")

    def _check_auto_re_enable(self, symbol: str, pattern: str,
                              regime: str, session: str):
        stats_list = [(self._global_stats[pattern], "global")]
        if self.expert_mode:
            stats_list.append((self._symbol_stats[symbol][pattern], f"symbol={symbol}"))

        for stats, scope in stats_list:
            standard_re_enable = stats.should_re_enable(self.cfg)
            thompson_re_enable = False
            thompson_p = 0.0
            if self._thompson_sampler and stats.total >= self.cfg.get("re_enable_min_samples", 3):
                thompson_re_enable, thompson_p = self._thompson_sampler.should_re_enable(pattern)
            should_re_enable = standard_re_enable or thompson_re_enable
            if should_re_enable:
                stats.disabled_until = 0
                stats.volume_multiplier = 1.0
                entry = {
                    "pattern": pattern, "scope": scope,
                    "time": time.time(),
                    "win_rate": stats.win_rate,
                    "thompson_p": thompson_p,
                }
                self._re_enabled_log.append(entry)
                reason = f"Thompson p={thompson_p:.0%}" if thompson_re_enable else f"WR={stats.win_rate:.0%}"
                logger.info(
                    f"[FAILURE] Patrón '{pattern}' RE-ACTIVADO ({scope}): {reason}"
                )

    def is_blocked(self, symbol: str, pattern: str,
                   regime: str = "", session: str = "") -> Tuple[bool, str]:
        if self._global_stats[pattern].is_disabled():
            return True, f"patrón '{pattern}' desactivado globalmente"
        if self.expert_mode:
            if self._symbol_stats[symbol][pattern].is_disabled():
                return True, f"patrón '{pattern}' desactivado para {symbol}"
            if regime and self._regime_stats[regime][pattern].is_disabled():
                return True, f"patrón '{pattern}' desactivado en régimen {regime}"
            if session and self._session_stats[session][pattern].is_disabled():
                return True, f"patrón '{pattern}' desactivado en sesión {session}"
        return False, ""

    def get_volume_multiplier(self, symbol: str, pattern: str) -> float:
        mult = self._global_stats[pattern].volume_multiplier
        if self.expert_mode:
            sym_mult = self._symbol_stats[symbol][pattern].volume_multiplier
            mult = min(mult, sym_mult)
        return mult

    def get_pattern_stats(self, symbol: str = "", pattern: str = "") -> Dict:
        if symbol and pattern:
            s = self._symbol_stats[symbol].get(pattern)
            if s is None:
                return {}
            return self._stats_to_dict(s)
        if pattern:
            s = self._global_stats.get(pattern)
            if s is None:
                return {}
            return self._stats_to_dict(s)
        return {
            pat: self._stats_to_dict(st)
            for pat, st in sorted(
                self._global_stats.items(),
                key=lambda x: x[1].total,
                reverse=True,
            )[:20]
        }

    def _stats_to_dict(self, s: PatternStats) -> Dict:
        return {
            "total": s.total, "wins": s.total_wins, "losses": s.total_losses,
            "win_rate": s.win_rate, "loss_rate": s.loss_rate,
            "consecutive_losses": s.consecutive_losses,
            "avg_pnl": s.avg_pnl, "disabled": s.is_disabled(),
            "volume_mult": s.volume_multiplier,
            "weighted_wr": s.weighted_win_rate(self.cfg["decay_half_life_hours"]),
        }

    def get_disabled_patterns(self) -> List[Dict]:
        now = time.time()
        active = []
        for entry in self._disabled_log:
            remaining = entry["time"] + entry["cooldown"] * 60 - now
            if remaining > 0:
                active.append({**entry, "remaining_min": round(remaining / 60, 1)})
        return active

    def get_failing_patterns(self, min_samples: int = 3) -> List[Dict]:
        failing = []
        for pattern, stats in self._global_stats.items():
            if stats.total >= min_samples and stats.loss_rate > self.cfg["max_loss_rate"]:
                failing.append({
                    "pattern": pattern,
                    "win_rate": stats.win_rate,
                    "loss_rate": stats.loss_rate,
                    "total": stats.total,
                    "avg_pnl": stats.avg_pnl,
                })
        return sorted(failing, key=lambda x: x["loss_rate"], reverse=True)[:20]

    def get_summary(self) -> Dict:
        disabled = self.get_disabled_patterns()
        failing = self.get_failing_patterns()
        return {
            "total_trades": len(self._trade_history),
            "tracked_patterns": len(self._global_stats),
            "disabled_patterns": len(disabled),
            "failing_patterns": len(failing),
            "re_enabled_count": len(self._re_enabled_log),
            "disabled_list": disabled,
            "failing_list": failing,
        }

    def get_trade_history(self, last_n: int = 20) -> List[Dict]:
        return [
            {
                "symbol": t.symbol, "pattern": t.primary_pattern,
                "result": "win" if t.profit >= 0 else "loss",
                "pnl_pct": t.profit,
                "regime": t.regime, "session": t.session,
                "direction": t.direction,
                "time": datetime.fromtimestamp(t.timestamp, tz=timezone.utc).isoformat(),
            }
            for t in self._trade_history[-last_n:]
        ]

    def get_similar_patterns(self, pattern: str) -> List[str]:
        if not self.expert_mode:
            return []
        cached = self._similar_patterns_cache.get(pattern)
        if cached is not None:
            return cached
        similar = []
        p_trades = [t for t in self._trade_history if t.primary_pattern == pattern]
        if len(p_trades) < 3:
            self._similar_patterns_cache[pattern] = []
            return []
        p_sessions = set(t.session for t in p_trades if t.session)
        p_regimes = set(t.regime for t in p_trades if t.regime)
        for other in self._global_stats:
            if other == pattern:
                continue
            o_trades = [t for t in self._trade_history if t.primary_pattern == other]
            if len(o_trades) < 3:
                continue
            o_sessions = set(t.session for t in o_trades if t.session)
            o_regimes = set(t.regime for t in o_trades if t.regime)
            session_overlap = len(p_sessions & o_sessions) / max(len(p_sessions | o_sessions), 1)
            regime_overlap = len(p_regimes & o_regimes) / max(len(p_regimes | o_regimes), 1)
            similarity = (session_overlap + regime_overlap) / 2
            if similarity >= self.cfg.get("pattern_similarity_threshold", 0.7):
                similar.append((other, similarity))
        similar.sort(key=lambda x: -x[1])
        result = [s[0] for s in similar[:5]]
        self._similar_patterns_cache[pattern] = result
        return result

    def reset_pattern(self, pattern: str = None):
        if pattern:
            self._global_stats.pop(pattern, None)
            for sym_stats in self._symbol_stats.values():
                sym_stats.pop(pattern, None)
            for reg_stats in self._regime_stats.values():
                reg_stats.pop(pattern, None)
            for ses_stats in self._session_stats.values():
                ses_stats.pop(pattern, None)
            logger.info(f"[FAILURE] Estadísticas reseteadas para patrón '{pattern}'")
        else:
            self._global_stats.clear()
            self._symbol_stats.clear()
            self._regime_stats.clear()
            self._session_stats.clear()
            self._disabled_log.clear()
            self._re_enabled_log.clear()
            logger.info("[FAILURE] Todas las estadísticas reseteadas")

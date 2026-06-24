"""Operation Registry — registra todas las operaciones del bot por sesión
with SQLite persistence for post-hoc analysis and reporting.
"""
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict
from loguru import logger


# ── Operation Type Constants ──────────────────────────────────────────
SESSION_START       = "SESSION_START"
SESSION_END         = "SESSION_END"
MT5_CONNECT         = "MT5_CONNECT"
MT5_DISCONNECT      = "MT5_DISCONNECT"
MT5_RECONNECT       = "MT5_RECONNECT"
MT5_ERROR           = "MT5_ERROR"
ACCOUNT_STATUS      = "ACCOUNT_STATUS"
NEW_CANDLE          = "NEW_CANDLE"
EVALUATION          = "EVALUATION"
CLEANUP             = "CLEANUP"
META_ANALYSIS       = "META_ANALYSIS"
NEURAL_TRAIN        = "NEURAL_TRAIN"
FRACTAL_SCAN        = "FRACTAL_SCAN"
SUBFRACTAL_SCAN     = "SUBFRACTAL_SCAN"
FRACTAL_ACTIVATED   = "FRACTAL_ACTIVATED"
FRACTAL_CLEANSED    = "FRACTAL_CLEANSED"
ENTRY_CHECK         = "ENTRY_CHECK"
SIGNAL_GENERATED    = "SIGNAL_GENERATED"
ORDER_PLACED        = "ORDER_PLACED"
ORDER_FILLED        = "ORDER_FILLED"
ORDER_CANCELLED     = "ORDER_CANCELLED"
ORDER_CLOSED        = "ORDER_CLOSED"
SL_HIT              = "SL_HIT"
BREAKEVEN           = "BREAKEVEN"
TRAILING            = "TRAILING"
SL_MODIFIED         = "SL_MODIFIED"
PACK_CLOSED         = "PACK_CLOSED"
PACK_OUTCOME        = "PACK_OUTCOME"
ERROR               = "ERROR"
WARNING             = "WARNING"


@dataclass
class OperationRecord:
    id: int = 0
    session_id: str = ""
    timestamp: str = ""
    operation_type: str = ""
    symbol: str = ""
    direction: str = ""
    volume: float = 0.0
    price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    ticket: int = 0
    pack_id: int = 0
    fractal_id: int = 0
    status: str = ""
    details: str = ""
    message: str = ""
    duration_ms: int = 0


class OperationRegistry:
    """Central registry for all trading bot operations per session.
    
    Usage:
        registry = OperationRegistry()
        registry.register(ORDER_PLACED, symbol="XAUUSDc", direction="BUY",
                          price=1920.5, volume=0.03, ticket=12345,
                          status="SUCCESS", message="LIMIT placed")
        registry.get_session_summary()
        registry.update_session_end(balance=10000, equity=10050, pnl=50)
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.session_id = str(uuid.uuid4())
        self.session_start = datetime.now(timezone.utc)
        self._records: List[OperationRecord] = []
        self._op_counter = 0

        if db_path is None:
            db_path = Path(__file__).resolve().parent.parent.parent / "data" / "operation_log.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()

        self.register(
            SESSION_START, status="SUCCESS",
            message=f"Session {self.session_id[:8]} started",
            details={"session_start": self.session_start.isoformat()}
        )

    def _init_db(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                operation_type TEXT NOT NULL,
                symbol TEXT DEFAULT '',
                direction TEXT DEFAULT '',
                volume REAL DEFAULT 0,
                price REAL DEFAULT 0,
                sl REAL DEFAULT 0,
                tp REAL DEFAULT 0,
                ticket INTEGER DEFAULT 0,
                pack_id INTEGER DEFAULT 0,
                fractal_id INTEGER DEFAULT 0,
                status TEXT DEFAULT '',
                details TEXT DEFAULT '',
                message TEXT DEFAULT '',
                duration_ms INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                start_time TEXT NOT NULL,
                end_time TEXT,
                symbol TEXT DEFAULT '',
                total_operations INTEGER DEFAULT 0,
                orders_placed INTEGER DEFAULT 0,
                orders_filled INTEGER DEFAULT 0,
                orders_closed INTEGER DEFAULT 0,
                sl_hit INTEGER DEFAULT 0,
                breakeven_count INTEGER DEFAULT 0,
                errors_count INTEGER DEFAULT 0,
                evaluations_count INTEGER DEFAULT 0,
                final_balance REAL DEFAULT 0,
                final_equity REAL DEFAULT 0,
                pnl REAL DEFAULT 0,
                status TEXT DEFAULT 'running'
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ops_session ON operations(session_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ops_type ON operations(operation_type)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ops_time ON operations(timestamp)
        """)
        conn.commit()
        conn.close()

    def register(self, operation_type: str, symbol: str = "",
                 direction: str = "", volume: float = 0.0,
                 price: float = 0.0, sl: float = 0.0, tp: float = 0.0,
                 ticket: int = 0, pack_id: int = 0, fractal_id: int = 0,
                 status: str = "INFO", details: Optional[dict] = None,
                 message: str = "", duration_ms: int = 0) -> int:
        """Register an operation and persist to SQLite."""
        self._op_counter += 1

        rec = OperationRecord(
            id=self._op_counter,
            session_id=self.session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            operation_type=operation_type,
            symbol=symbol,
            direction=direction,
            volume=volume,
            price=price,
            sl=sl,
            tp=tp,
            ticket=ticket,
            pack_id=pack_id,
            fractal_id=fractal_id,
            status=status,
            details=json.dumps(details or {}),
            message=message,
            duration_ms=duration_ms,
        )

        self._records.append(rec)
        self._persist(rec)
        return rec.id

    def _persist(self, rec: OperationRecord):
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                INSERT INTO operations
                    (session_id, timestamp, operation_type, symbol,
                     direction, volume, price, sl, tp,
                     ticket, pack_id, fractal_id, status,
                     details, message, duration_ms)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                rec.session_id, rec.timestamp, rec.operation_type,
                rec.symbol, rec.direction, rec.volume, rec.price,
                rec.sl, rec.tp, rec.ticket, rec.pack_id, rec.fractal_id,
                rec.status, rec.details, rec.message, rec.duration_ms,
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"[Registry] Failed to persist: {e}")
        finally:
            conn.close()

    def update_session_end(self, balance: float = 0.0, equity: float = 0.0,
                           pnl: float = 0.0):
        """Mark current session as completed with final stats."""
        orders  = self.count_by_type(ORDER_PLACED)
        fills   = self.count_by_type(ORDER_FILLED)
        closed  = self.count_by_type(ORDER_CLOSED)
        sls     = self.count_by_type(SL_HIT)
        bes     = self.count_by_type(BREAKEVEN)
        errs    = self.count_by_type(ERROR)
        evals   = self.count_by_type(EVALUATION)

        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                INSERT OR REPLACE INTO sessions
                    (session_id, start_time, end_time, symbol,
                     total_operations, orders_placed, orders_filled,
                     orders_closed, sl_hit, breakeven_count,
                     errors_count, evaluations_count,
                     final_balance, final_equity, pnl, status)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                self.session_id,
                self.session_start.isoformat(),
                datetime.now(timezone.utc).isoformat(),
                "",
                self._op_counter, orders, fills,
                closed, sls, bes,
                errs, evals,
                balance, equity, pnl, "completed",
            ))
            conn.commit()
        finally:
            conn.close()

    def count_by_type(self, op_type: str) -> int:
        return sum(1 for r in self._records if r.operation_type == op_type)

    def get_session_summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat(),
            "total_operations": self._op_counter,
            "orders_placed":     self.count_by_type(ORDER_PLACED),
            "orders_filled":     self.count_by_type(ORDER_FILLED),
            "orders_closed":     self.count_by_type(ORDER_CLOSED),
            "sl_hit":            self.count_by_type(SL_HIT),
            "breakeven":         self.count_by_type(BREAKEVEN),
            "trailing_updates":  self.count_by_type(TRAILING),
            "errors":            self.count_by_type(ERROR),
            "evaluations":       self.count_by_type(EVALUATION),
            "fractals_scanned":  self.count_by_type(FRACTAL_SCAN),
            "entries_checked":   self.count_by_type(ENTRY_CHECK),
        }

    def close(self):
        self.register("REGISTRY_CLOSE", status="INFO",
                       message="Operation registry closed")

    # ── Static query helpers ─────────────────────────────────────

    @staticmethod
    def query_by_session(db_path: str, session_id: str) -> List[dict]:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM operations WHERE session_id=? ORDER BY id",
            (session_id,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    @staticmethod
    def query_by_type(db_path: str, op_type: str, limit: int = 100) -> List[dict]:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM operations WHERE operation_type=? ORDER BY timestamp DESC LIMIT ?",
            (op_type, limit)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    @staticmethod
    def query_recent(db_path: str, limit: int = 50) -> List[dict]:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM operations ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    @staticmethod
    def query_by_date(db_path: str, date_str: str) -> List[dict]:
        """Query operations for a specific date (YYYY-MM-DD)."""
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM operations WHERE timestamp LIKE ? ORDER BY id",
            (f"{date_str}%",)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    @staticmethod
    def list_sessions(db_path: str) -> List[dict]:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY start_time DESC"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    @staticmethod
    def get_session(db_path: str, session_id: str) -> Optional[dict]:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None

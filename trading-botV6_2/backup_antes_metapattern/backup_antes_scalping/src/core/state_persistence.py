"""State Persistence with SQLite
Handles daily state and trade history for crash recovery
"""
import aiosqlite
import json
import asyncio
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List
from loguru import logger


class StatePersistence:
    """Async SQLite persistence for trading bot state"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self):
        """Initialize database and create tables with multi-path fallback"""
        from pathlib import Path as _Path

        def _ensure_writable_dir(path: _Path) -> bool:
            try:
                path.mkdir(parents=True, exist_ok=True)
                _probe = path / ".write_test"
                _probe.write_text("ok")
                _probe.unlink()
                return True
            except (PermissionError, OSError, IOError) as _e:
                logger.warning(f"No se puede escribir en {path}: {_e}")
                return False

        _candidates = [
            self.db_path.parent,
            _Path("data"),
            _Path.home() / ".trading_bot",
            _Path.home() / "AppData" / "Local" / "Temp" / "trading_bot_state",
        ]
        _chosen = None
        for _cand in _candidates:
            if _ensure_writable_dir(_cand):
                _chosen = _cand
                break

        if _chosen is None:
            import tempfile as _tf
            _chosen = _Path(_tf.mkdtemp(prefix="trading_bot_state_"))
            logger.warning(f"Fallback extremo (datos VOLÁTILES) en: {_chosen}")

        self.db_path = _chosen / self.db_path.name
        logger.info(f"Base de datos persistencia: {self.db_path.resolve()}")

        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS daily_state (
                date TEXT PRIMARY KEY,
                daily_loss REAL DEFAULT 0,
                trades_count INTEGER DEFAULT 0,
                last_save TEXT,
                state_json TEXT
            )
        """)

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS trade_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket INTEGER,
                symbol TEXT,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                volume REAL,
                profit REAL,
                sl_original REAL,
                tp_original REAL,
                open_time TEXT,
                close_time TEXT,
                be_activated INTEGER DEFAULT 0,
                trailing_activated INTEGER DEFAULT 0,
                trailing_distance REAL DEFAULT 0
            )
        """)

        await self._db.execute("""
            DELETE FROM trade_history WHERE rowid NOT IN (
                SELECT MAX(rowid) FROM trade_history GROUP BY ticket
            )
        """)

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS market_hours_state (
                symbol TEXT PRIMARY KEY,
                asset_class TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'UNKNOWN',
                last_candle_time REAL,
                cooldown_end REAL,
                blackout_start REAL,
                updated_at REAL NOT NULL
            )
        """)

        await self._db.commit()
        logger.info(f"State persistence initialized at {self.db_path}")

    async def close(self):
        """Close database connection"""
        if self._db:
            await self._db.close()
            self._db = None

    async def save_daily_state(self, daily_loss: float, trades_count: int,
                                extra_state: Optional[dict] = None):
        """Save daily trading state"""
        today = date.today().isoformat()
        now = datetime.now().isoformat()

        await self._db.execute("""
            INSERT INTO daily_state (date, daily_loss, trades_count, last_save, state_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                daily_loss = excluded.daily_loss,
                trades_count = excluded.trades_count,
                last_save = excluded.last_save,
                state_json = excluded.state_json
        """, (today, daily_loss, trades_count, now,
              json.dumps(extra_state) if extra_state else None))

        await self._db.commit()
        logger.debug(f"Daily state saved: date={today}, loss={daily_loss}, trades={trades_count}")

    async def load_daily_state(self) -> dict:
        """Load today's trading state"""
        today = date.today().isoformat()

        async with self._db.execute(
            "SELECT * FROM daily_state WHERE date = ?", (today,)
        ) as cursor:
            row = await cursor.fetchone()

        if row:
            state = {
                "daily_loss": row["daily_loss"],
                "trades_count": row["trades_count"],
                "state_json": json.loads(row["state_json"]) if row["state_json"] else {}
            }
            logger.info(f"Loaded daily state: loss={state['daily_loss']}, trades={state['trades_count']}")
            return state

        return {"daily_loss": 0.0, "trades_count": 0, "state_json": {}}

    async def save_trade(self, trade: dict):
        """Save completed or open trade to history"""
        await self._db.execute("""
            INSERT INTO trade_history (
                ticket, symbol, direction, entry_price, exit_price,
                volume, profit, sl_original, tp_original,
                open_time, close_time, be_activated, trailing_activated, trailing_distance
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.get("ticket"),
            trade.get("symbol"),
            trade.get("direction"),
            trade.get("entry_price"),
            trade.get("exit_price", 0),
            trade.get("volume"),
            trade.get("profit", 0),
            trade.get("sl_original"),
            trade.get("tp_original"),
            trade.get("open_time"),
            trade.get("close_time"),
            1 if trade.get("be_activated") else 0,
            1 if trade.get("trailing_activated") else 0,
            trade.get("trailing_distance", 0)
        ))

        await self._db.commit()
        logger.debug(f"Trade saved: ticket={trade.get('ticket')}, profit={trade.get('profit')}")

    async def save_market_hours(self, states: dict):
        """Persist market hours state for all symbols"""
        now = datetime.now().timestamp()
        for sym, data in states.items():
            await self._db.execute("""
                INSERT INTO market_hours_state (symbol, asset_class, state, last_candle_time, cooldown_end, blackout_start, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    state = excluded.state,
                    last_candle_time = excluded.last_candle_time,
                    cooldown_end = excluded.cooldown_end,
                    blackout_start = excluded.blackout_start,
                    updated_at = excluded.updated_at
            """, (
                sym,
                data.get('asset_class', 'unknown'),
                data.get('state', 'UNKNOWN'),
                data.get('last_candle_time'),
                data.get('cooldown_until'),
                data.get('blackout_start'),
                now,
            ))
        await self._db.commit()
        logger.debug(f"Market hours state saved for {len(states)} symbols")

    async def load_market_hours(self) -> dict:
        """Load persisted market hours state"""
        async with self._db.execute(
            "SELECT * FROM market_hours_state"
        ) as cursor:
            rows = await cursor.fetchall()
        result = {}
        for row in rows:
            result[row["symbol"]] = {
                "asset_class": row["asset_class"],
                "state": row["state"],
                "last_candle_time": row["last_candle_time"],
                "cooldown_until": row["cooldown_end"],
                "blackout_start": row["blackout_start"],
            }
        return result

    async def load_open_positions(self) -> List[dict]:
        """Load open positions (trades without close_time)"""
        async with self._db.execute(
            "SELECT * FROM trade_history WHERE close_time IS NULL OR close_time = ''"
        ) as cursor:
            rows = await cursor.fetchall()

        seen = set()
        positions = []
        for row in rows:
            ticket = row["ticket"]
            if ticket in seen:
                continue
            seen.add(ticket)
            positions.append({
                "ticket": ticket,
                "symbol": row["symbol"],
                "direction": row["direction"],
                "entry_price": row["entry_price"],
                "volume": row["volume"],
                "sl_original": row["sl_original"],
                "tp_original": row["tp_original"],
                "open_time": row["open_time"],
                "be_activated": bool(row["be_activated"]),
                "trailing_activated": bool(row["trailing_activated"]),
                "trailing_distance": row["trailing_distance"]
            })

        return positions
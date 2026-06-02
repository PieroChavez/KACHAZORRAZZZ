"""Seed V6_2_3 learning databases with initial data for immediate learning."""
import sqlite3
from pathlib import Path

base = Path(__file__).resolve().parent.parent.parent / 'data' / 'db'

def seed_kelly():
    conn = sqlite3.connect(str(base / "kelly_state.db"))
    conn.execute("""CREATE TABLE IF NOT EXISTS kelly_state (
        symbol TEXT, direction TEXT, regime TEXT,
        wins INTEGER DEFAULT 0, losses INTEGER DEFAULT 0,
        total_trades INTEGER DEFAULT 0, kelly_fraction REAL DEFAULT 0.0,
        PRIMARY KEY (symbol, direction, regime)
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS trade_outcomes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT, direction TEXT, regime TEXT,
        won INTEGER, profit REAL, timestamp TEXT
    )""")
    seeds = [
        ("XAUUSDm", "BUY", "STRONG_TREND_BULLISH", 8, 2, 10, 0.60),
        ("XAUUSDm", "SELL", "STRONG_TREND_BEARISH", 7, 3, 10, 0.55),
        ("XAUUSDm", "BUY", "RANGING", 4, 3, 7, 0.35),
        ("XAUUSDm", "SELL", "RANGING", 5, 2, 7, 0.40),
        ("XAUUSDm", "BUY", "HIGH_VOLATILITY", 3, 4, 7, 0.25),
        ("XAUUSDm", "SELL", "HIGH_VOLATILITY", 4, 3, 7, 0.30),
        ("XAUUSDm", "BUY", "TRANSITION", 5, 5, 10, 0.30),
        ("XAUUSDm", "SELL", "TRANSITION", 6, 4, 10, 0.35),
    ]
    for row in seeds:
        conn.execute("INSERT OR REPLACE INTO kelly_state (symbol, direction, regime, wins, losses, total_trades, kelly_fraction) VALUES (?,?,?,?,?,?,?)", row)
    conn.commit()
    conn.close()
    print(f"  kelly_state.db: {len(seeds)} seeds inserted")

def seed_failure():
    conn = sqlite3.connect(str(base / "failure_analysis.db"))
    conn.execute("""CREATE TABLE IF NOT EXISTS pattern_stats (
        pattern_type TEXT, symbol TEXT,
        wins INTEGER DEFAULT 0, losses INTEGER DEFAULT 0,
        total INTEGER DEFAULT 0, win_rate REAL DEFAULT 0.0,
        wilson_lower REAL DEFAULT 0.0, is_disabled INTEGER DEFAULT 0,
        consecutive_losses INTEGER DEFAULT 0,
        PRIMARY KEY (pattern_type, symbol)
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS disabled_patterns (
        pattern_type TEXT, symbol TEXT, disabled_at TEXT, reason TEXT
    )""")
    seeds = [
        ("FVG_BULLISH", "XAUUSDm", 6, 2, 8, 0.75, 0.65, 0, 0),
        ("FVG_BEARISH", "XAUUSDm", 5, 3, 8, 0.63, 0.52, 0, 0),
        ("ORDER_BLOCK_BULLISH", "XAUUSDm", 4, 1, 5, 0.80, 0.68, 0, 0),
        ("ORDER_BLOCK_BEARISH", "XAUUSDm", 3, 2, 5, 0.60, 0.47, 0, 0),
        ("LIQUIDITY_SWEEP_BULLISH", "XAUUSDm", 4, 2, 6, 0.67, 0.54, 0, 0),
        ("LIQUIDITY_SWEEP_BEARISH", "XAUUSDm", 5, 1, 6, 0.83, 0.72, 0, 0),
        ("INTERVAL_POINT_BULLISH", "XAUUSDm", 3, 3, 6, 0.50, 0.38, 0, 0),
        ("INTERVAL_POINT_BEARISH", "XAUUSDm", 4, 2, 6, 0.67, 0.54, 0, 0),
        ("WYCKOFF_SPRING_BULLISH", "XAUUSDm", 3, 1, 4, 0.75, 0.60, 0, 0),
        ("WYCKOFF_UTAD_BEARISH", "XAUUSDm", 2, 2, 4, 0.50, 0.35, 0, 0),
    ]
    for row in seeds:
        conn.execute("INSERT OR REPLACE INTO pattern_stats (pattern_type, symbol, wins, losses, total, win_rate, wilson_lower, is_disabled, consecutive_losses) VALUES (?,?,?,?,?,?,?,?,?)", row)
    conn.commit()
    conn.close()
    print(f"  failure_analysis.db: {len(seeds)} seeds inserted")

def seed_entry_confirmation():
    conn = sqlite3.connect(str(base / "entry_confirmation.db"))
    conn.execute("""CREATE TABLE IF NOT EXISTS regime_stats (
        regime TEXT, symbol TEXT,
        immediate_entries INTEGER DEFAULT 0, immediate_wins INTEGER DEFAULT 0,
        deferred_entries INTEGER DEFAULT 0, deferred_wins INTEGER DEFAULT 0,
        optimal_defer_bars INTEGER DEFAULT 0,
        PRIMARY KEY (regime, symbol)
    )""")
    seeds = [
        ("STRONG_TREND_BULLISH", "XAUUSDm", 10, 7, 2, 1, 1),
        ("STRONG_TREND_BEARISH", "XAUUSDm", 8, 6, 2, 1, 1),
        ("RANGING", "XAUUSDm", 10, 5, 4, 2, 3),
        ("HIGH_VOLATILITY", "XAUUSDm", 6, 2, 4, 1, 5),
        ("TRANSITION", "XAUUSDm", 12, 5, 5, 2, 3),
    ]
    for row in seeds:
        conn.execute("INSERT OR REPLACE INTO regime_stats (regime, symbol, immediate_entries, immediate_wins, deferred_entries, deferred_wins, optimal_defer_bars) VALUES (?,?,?,?,?,?,?)", row)
    conn.commit()
    conn.close()
    print(f"  entry_confirmation.db: {len(seeds)} seeds inserted")

def seed_adaptive_thresholds():
    conn = sqlite3.connect(str(base / "adaptive_thresholds.db"))
    conn.execute("""CREATE TABLE IF NOT EXISTS cooldown_state (
        symbol TEXT PRIMARY KEY,
        stage TEXT DEFAULT 'NORMAL',
        consecutive_losses INTEGER DEFAULT 0,
        cooldown_minutes INTEGER DEFAULT 0,
        last_trade_time REAL DEFAULT 0,
        thompson_alpha REAL DEFAULT 1.0,
        thompson_beta REAL DEFAULT 1.0,
        total_trades INTEGER DEFAULT 0,
        wins INTEGER DEFAULT 0
    )""")
    seeds = [
        ("XAUUSDm", "NORMAL", 0, 0, 0, 3.0, 2.0, 5, 3),
        ("XAGUSDm", "NORMAL", 0, 0, 0, 2.0, 2.0, 4, 2),
    ]
    for row in seeds:
        conn.execute("INSERT OR REPLACE INTO cooldown_state (symbol, stage, consecutive_losses, cooldown_minutes, last_trade_time, thompson_alpha, thompson_beta, total_trades, wins) VALUES (?,?,?,?,?,?,?,?,?)", row)
    conn.commit()
    conn.close()
    print(f"  adaptive_thresholds.db: {len(seeds)} seeds inserted")

def seed_bayesian_ensemble():
    conn = sqlite3.connect(str(base / "bayesian_ensemble.db"))
    conn.execute("""CREATE TABLE IF NOT EXISTS expert_state (
        name TEXT PRIMARY KEY,
        alpha REAL DEFAULT 1.0, beta REAL DEFAULT 1.0,
        predictions INTEGER DEFAULT 0, correct INTEGER DEFAULT 0,
        weight REAL DEFAULT 1.0
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS ensemble_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT, direction TEXT, conviction REAL,
        uncertainty REAL, actual_outcome INTEGER, timestamp TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS sltp_learning (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT, direction TEXT,
        adverse_excursion REAL, favorable_excursion REAL,
        atr_at_entry REAL, timestamp TEXT
    )""")
    experts = [
        ("MarketMap", 4.0, 2.0, 6, 4, 0.85),
        ("OrderFlow", 3.0, 3.0, 6, 3, 0.70),
        ("MicroPredictor", 5.0, 1.0, 6, 5, 0.92),
        ("RegimeAlignment", 3.5, 2.5, 6, 3, 0.65),
    ]
    for row in experts:
        conn.execute("INSERT OR REPLACE INTO expert_state (name, alpha, beta, predictions, correct, weight) VALUES (?,?,?,?,?,?)", row)
    conn.commit()
    conn.close()
    print(f"  bayesian_ensemble.db: {len(experts)} expert seeds inserted")

print("Seeding V6_2_3 learning databases...")
seed_kelly()
seed_failure()
seed_entry_confirmation()
seed_adaptive_thresholds()
seed_bayesian_ensemble()
print("Done.")

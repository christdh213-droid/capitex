"""Database module for SQLite storage of wallets, holdings, trades, and snapshots.

Provides robust context-managed SQLite access, Write-Ahead Logging (WAL) concurrency,
schema migration/initialization, wallet seeding, and transactional helper functions.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, Generator, List, Optional

import config

logger = logging.getLogger(__name__)


@contextmanager
def get_connection(db_path: Optional[Path] = None) -> Generator[sqlite3.Connection, None, None]:
    """Provide a transactional database connection context.

    Uses WAL mode and 5000ms busy timeout to support concurrent reads/writes
    between FastAPI and Streamlit without database locks.
    """
    path = db_path or config.LEGACY_DB_PATH
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        # Enable Write-Ahead Logging and busy timeout for high-concurrency access
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        yield conn
        conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.error(f"Database error during transaction: {exc}")
        raise
    finally:
        conn.close()


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize database tables, apply column migrations, and seed initial wallets.

    Safe to run repeatedly on every rerun or API startup without data loss.
    """
    with get_connection(db_path) as conn:
        conn.execute('BEGIN IMMEDIATE')
        cursor = conn.cursor()

        # 1. Wallets table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS wallets (
                wallet_type TEXT PRIMARY KEY,
                cash REAL NOT NULL CHECK(cash >= 0)
            )
            """
        )

        # 2. Holdings table (with peak_price for dynamic trailing stops)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS holdings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wallet_type TEXT NOT NULL,
                ticker TEXT NOT NULL,
                quantity REAL NOT NULL CHECK(quantity >= 0),
                avg_buy_price REAL NOT NULL CHECK(avg_buy_price >= 0),
                peak_price REAL,
                stop_loss REAL,
                take_profit REAL,
                UNIQUE(wallet_type, ticker),
                FOREIGN KEY(wallet_type) REFERENCES wallets(wallet_type)
            )
            """
        )

        # Migration: ensure peak_price exists on existing databases
        cursor.execute("PRAGMA table_info(holdings)")
        columns = [row[1] for row in cursor.fetchall()]
        if "peak_price" not in columns:
            cursor.execute("ALTER TABLE holdings ADD COLUMN peak_price REAL")
            logger.info("Migrated holdings table: added peak_price column.")

        # 3. Trade ledger table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                wallet_type TEXT NOT NULL,
                ticker TEXT NOT NULL,
                action TEXT NOT NULL CHECK(action IN ('BUY', 'SELL', 'RESET', 'STOP_LOSS', 'TAKE_PROFIT')),
                quantity REAL NOT NULL CHECK(quantity >= 0),
                price REAL NOT NULL CHECK(price >= 0),
                realized_pnl REAL DEFAULT 0.0,
                FOREIGN KEY(wallet_type) REFERENCES wallets(wallet_type)
            )
            """
        )

        # 4. Portfolio snapshots table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                wallet_type TEXT NOT NULL,
                net_worth REAL NOT NULL,
                FOREIGN KEY(wallet_type) REFERENCES wallets(wallet_type)
            )
            """
        )

        # Additive migrations preserve existing wallet and ledger history.
        for table, additions in {
            'trade_ledger': {'execution_source': "TEXT DEFAULT 'legacy'",
                             'execution_label': "TEXT DEFAULT 'Legacy fill (source not recorded)'",
                             'quote_timestamp': 'TEXT'},
            'portfolio_snapshots': {'reason': "TEXT NOT NULL DEFAULT 'event'"},
        }.items():
            present = {row[1] for row in cursor.execute(f'PRAGMA table_info({table})')}
            for name, declaration in additions.items():
                if name not in present:
                    cursor.execute(f'ALTER TABLE {table} ADD COLUMN {name} {declaration}')
        cursor.execute('CREATE INDEX IF NOT EXISTS snapshot_wallet_reason ON portfolio_snapshots(wallet_type, reason, id)')

        # Seed initial wallets if not present
        now_iso = datetime.now(timezone.utc).isoformat()
        for wallet_type in config.VALID_WALLET_TYPES:
            cursor.execute(
                "SELECT cash FROM wallets WHERE wallet_type = ?",
                (wallet_type,)
            )
            row = cursor.fetchone()
            if row is None:
                cursor.execute(
                    "INSERT INTO wallets (wallet_type, cash) VALUES (?, ?)",
                    (wallet_type, config.STARTING_CASH)
                )
                cursor.execute(
                    "INSERT INTO portfolio_snapshots (timestamp, wallet_type, net_worth) VALUES (?, ?, ?)",
                    (now_iso, wallet_type, config.STARTING_CASH)
                )
                logger.info(f"Seeded {wallet_type} wallet with ${config.STARTING_CASH:,.2f}")


def get_wallet(wallet_type: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve wallet state by wallet_type ('human' or 'ai')."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT wallet_type, cash FROM wallets WHERE wallet_type = ?", (wallet_type,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_all_wallets(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieve all wallets."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT wallet_type, cash FROM wallets ORDER BY wallet_type")
        return [dict(r) for r in cursor.fetchall()]


def get_holdings(wallet_type: str, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieve all active stock holdings for a specific wallet."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT wallet_type, ticker, quantity, avg_buy_price, peak_price, stop_loss, take_profit
            FROM holdings
            WHERE wallet_type = ? AND quantity > 0
            ORDER BY ticker ASC
            """,
            (wallet_type,)
        )
        return [dict(r) for r in cursor.fetchall()]


def get_holding_for_ticker(wallet_type: str, ticker: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve holding record for a specific wallet and ticker."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT wallet_type, ticker, quantity, avg_buy_price, peak_price, stop_loss, take_profit
            FROM holdings
            WHERE wallet_type = ? AND ticker = ? AND quantity > 0
            """,
            (wallet_type, ticker.strip().upper())
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def update_holding_peak_price(wallet_type: str, ticker: str, new_peak_price: float, db_path: Optional[Path] = None) -> None:
    """Update peak price reached for a holding (used for trailing stop-loss)."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE holdings
            SET peak_price = MAX(COALESCE(peak_price, avg_buy_price), ?)
            WHERE wallet_type = ? AND ticker = ?
            """,
            (new_peak_price, wallet_type, ticker.strip().upper())
        )


def get_trade_ledger(wallet_type: Optional[str] = None, limit: int = 100, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieve recent trades from trade ledger."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        if wallet_type:
            cursor.execute(
                """
                SELECT id, timestamp, wallet_type, ticker, action, quantity, price, realized_pnl, execution_source, execution_label, quote_timestamp
                FROM trade_ledger
                WHERE wallet_type = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (wallet_type, limit)
            )
        else:
            cursor.execute(
                """
                SELECT id, timestamp, wallet_type, ticker, action, quantity, price, realized_pnl, execution_source, execution_label, quote_timestamp
                FROM trade_ledger
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,)
            )
        return [dict(r) for r in cursor.fetchall()]


def record_snapshot(wallet_type: str, net_worth: float, timestamp: Optional[str] = None,
                    db_path: Optional[Path] = None, *, reason: str = 'event',
                    min_interval_seconds: float = 0) -> int:
    """Persist every event; throttle refresh writes atomically across processes."""
    stamp = datetime.fromisoformat(timestamp) if timestamp else datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    with get_connection(db_path) as conn:
        conn.execute('BEGIN IMMEDIATE')
        if min_interval_seconds > 0:
            previous = conn.execute(
                'SELECT timestamp FROM portfolio_snapshots WHERE wallet_type=? AND reason=? ORDER BY id DESC LIMIT 1',
                (wallet_type, reason)).fetchone()
            if previous:
                previous_time = datetime.fromisoformat(previous['timestamp'])
                if previous_time.tzinfo is None:
                    previous_time = previous_time.replace(tzinfo=timezone.utc)
                if (stamp-previous_time).total_seconds() < min_interval_seconds:
                    return 0
        cursor = conn.execute(
            'INSERT INTO portfolio_snapshots(timestamp,wallet_type,net_worth,reason) VALUES(?,?,?,?)',
            (stamp.isoformat(), wallet_type, float(net_worth), reason))
        return cursor.lastrowid or 0


def get_portfolio_snapshots(wallet_type: str, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieve portfolio snapshots for a wallet, ordered chronologically."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, timestamp, wallet_type, net_worth, reason
            FROM portfolio_snapshots
            WHERE wallet_type = ?
            ORDER BY id ASC
            """,
            (wallet_type,)
        )
        return [dict(r) for r in cursor.fetchall()]


def reset_wallets(starting_cash: float = config.STARTING_CASH, db_path: Optional[Path] = None) -> None:
    """Reset both human and AI wallets, clearing holdings, trade ledger, and snapshots."""
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM holdings")
        cursor.execute("DELETE FROM trade_ledger")
        cursor.execute("DELETE FROM portfolio_snapshots")

        for wallet_type in config.VALID_WALLET_TYPES:
            cursor.execute(
                "INSERT INTO wallets (wallet_type, cash) VALUES (?, ?) "
                "ON CONFLICT(wallet_type) DO UPDATE SET cash = excluded.cash",
                (wallet_type, starting_cash)
            )
            cursor.execute(
                "INSERT INTO portfolio_snapshots (timestamp, wallet_type, net_worth, reason) VALUES (?, ?, ?, 'reset')",
                (now_iso, wallet_type, starting_cash)
            )
            cursor.execute(
                "INSERT INTO trade_ledger (timestamp, wallet_type, ticker, action, quantity, price, realized_pnl) "
                "VALUES (?, ?, 'SYSTEM', 'RESET', 0, 0, 0)",
                (now_iso, wallet_type)
            )
        logger.info(f"Reset all wallets to ${starting_cash:,.2f}")

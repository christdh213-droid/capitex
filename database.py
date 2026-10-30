"""CAPITEX single-wallet WAL storage; prototype adapters remain isolated."""
from contextlib import contextmanager
from pathlib import Path
import sqlite3
import config
import legacy_database as _legacy

# Compatibility adapters for the existing app; new core uses get_connection.
get_legacy_connection = _legacy.get_connection
get_wallet = _legacy.get_wallet
get_all_wallets = _legacy.get_all_wallets
get_holdings = _legacy.get_holdings
get_holding_for_ticker = _legacy.get_holding_for_ticker
update_holding_peak_price = _legacy.update_holding_peak_price
get_trade_ledger = _legacy.get_trade_ledger
record_snapshot = _legacy.record_snapshot
get_portfolio_snapshots = _legacy.get_portfolio_snapshots
reset_wallets = _legacy.reset_wallets


@contextmanager
def get_connection(db_path: Path | None = None):
    """Commit on success, roll back on error, close on every exit."""
    path = Path(db_path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path | None = None):
    """Idempotent initialization; never reset an existing CAPITEX balance."""
    with get_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        statements = (
            """CREATE TABLE IF NOT EXISTS wallet (
                id INTEGER PRIMARY KEY CHECK(id=1),
                cash REAL NOT NULL DEFAULT 100000.0 CHECK(cash>=0),
                starting_cash REAL NOT NULL DEFAULT 100000.0 CHECK(starting_cash>=0))""",
            """CREATE TABLE IF NOT EXISTS holdings (
                ticker TEXT PRIMARY KEY, name TEXT NOT NULL,
                quantity REAL NOT NULL CHECK(quantity>=0),
                avg_buy_price REAL NOT NULL CHECK(avg_buy_price>=0),
                peak_price REAL, stop_loss_pct REAL, take_profit_pct REAL)""",
            """CREATE TABLE IF NOT EXISTS trade_ledger (
                id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, ticker TEXT NOT NULL,
                action TEXT NOT NULL, quantity REAL NOT NULL CHECK(quantity>=0),
                price REAL NOT NULL CHECK(price>=0), total_value REAL NOT NULL,
                realized_pnl REAL NOT NULL DEFAULT 0)""",
            """CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL,
                net_worth REAL NOT NULL, cash REAL NOT NULL, invested_value REAL NOT NULL)""",
        )
        for statement in statements:
            conn.execute(statement)
        for table, additions in {
            'trade_ledger': {'execution_source': "TEXT DEFAULT 'legacy'",
                             'execution_label': "TEXT DEFAULT 'Historical fill'",
                             'quote_timestamp': 'TEXT', 'client_order_id': 'TEXT',
                             'order_fingerprint': 'TEXT'},
            'portfolio_snapshots': {'reason': "TEXT NOT NULL DEFAULT 'event'"},
        }.items():
            present = {row[1] for row in conn.execute(f'PRAGMA table_info({table})')}
            for name, declaration in additions.items():
                if name not in present:
                    conn.execute(f'ALTER TABLE {table} ADD COLUMN {name} {declaration}')
        conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS capitex_order_id ON trade_ledger(client_order_id) WHERE client_order_id IS NOT NULL')
        conn.execute('CREATE INDEX IF NOT EXISTS capitex_snapshot_reason ON portfolio_snapshots(reason, id)')
        conn.execute("INSERT OR IGNORE INTO wallet(id,cash,starting_cash) VALUES(1,?,?)",
                     (config.STARTING_CASH, config.STARTING_CASH))
        if not conn.execute('SELECT 1 FROM portfolio_snapshots LIMIT 1').fetchone():
            from datetime import datetime, timezone
            wallet = conn.execute('SELECT cash FROM wallet WHERE id=1').fetchone()
            held = conn.execute('SELECT COALESCE(SUM(quantity*avg_buy_price),0) FROM holdings').fetchone()[0]
            conn.execute("INSERT INTO portfolio_snapshots(timestamp,net_worth,cash,invested_value,reason) VALUES(?,?,?,?, 'initial')",
                         (datetime.now(timezone.utc).isoformat(), wallet['cash']+held, wallet['cash'], held))

    if db_path is None:
        _legacy.init_db()


def get_capitex_wallet(db_path: Path | None = None):
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM wallet WHERE id=1").fetchone()
        return dict(row) if row else None

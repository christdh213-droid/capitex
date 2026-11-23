"""One persistent UTC timestamp convention for CAPITEX."""
from datetime import datetime, timezone
import pandas as pd

def now_utc():
    return datetime.now(timezone.utc)

def utc_z(value=None):
    stamp=pd.Timestamp(now_utc() if value is None else value)
    if pd.isna(stamp):raise ValueError('Invalid timestamp')
    stamp=stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp.tz_convert('UTC')
    return stamp.strftime('%Y-%m-%dT%H:%M:%SZ')

def normalize_existing(conn):
    """Canonicalise timestamp text without touching financial values or IDs."""
    for table, in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall():
        columns=[r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')
                 if r[1] in ('timestamp','quote_timestamp','opened_at','updated_at','issued_at','data_timestamp',
                             'target_timestamp','actual_timestamp','graded_at','published_at','fetched_at','generated_at','expires_at','prepared_at')]
        for column in columns:
            for rowid,value in conn.execute(f'SELECT rowid,"{column}" FROM "{table}" WHERE "{column}" IS NOT NULL').fetchall():
                try:canonical=utc_z(value)
                except (ValueError,TypeError):continue
                if value!=canonical:conn.execute(f'UPDATE "{table}" SET "{column}"=? WHERE rowid=?',(canonical,rowid))

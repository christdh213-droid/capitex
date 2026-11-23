"""Upgrade contracts: injected clocks/quotes; temporary populated SQLite files."""
import json,sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
import pytest
import database,config
from market_calendar import targets,completed_history,session_close
from forecast_log import log_forecasts
from time_utils import utc_z

@pytest.fixture
def db(tmp_path,monkeypatch):
    monkeypatch.setattr(config,'DB_PATH',tmp_path/'wallet.db')
    monkeypatch.setattr(config,'LEGACY_DB_PATH',tmp_path/'legacy.db')
    database.init_db()
    return tmp_path/'wallet.db'

def fake_forecast(bundle,steps,current_price,ticker):
    path=[current_price*(1+.01*i) for i in range(steps+1)]
    return dict(ticker=ticker,horizon=steps,horizon_label=str(steps),current_price=current_price,
                outlook='bullish',expected_change_pct=steps,confidence=50.,median_target_price=path[-1],
                corridor={'median':path,'lower':[p*.9 if i else p for i,p in enumerate(path)],
                          'upper':[p*1.1 if i else p for i,p in enumerate(path)]})

def test_migrate_populated_twice(db):
    with database.get_connection() as c:
        c.execute('UPDATE wallet SET cash=12345 WHERE id=1')
        c.execute("INSERT INTO holdings VALUES('AAPL','Apple',2,100,110,NULL,NULL)")
        c.execute("INSERT INTO short_positions(symbol,quantity,avg_entry_price,locked_collateral,opened_at,updated_at) VALUES('BTC-USD',1,100,100,'2026-01-01T00:00:00+00:00','2026-01-01T00:00:00+00:00')")
        c.execute("INSERT INTO trade_ledger(timestamp,ticker,action,quantity,price,total_value) VALUES('2026-01-01T00:00:00+00:00','AAPL','BUY',2,100,200)")
    copy=db.with_name('populated_copy.db')
    with sqlite3.connect(db) as source,sqlite3.connect(copy) as target:source.backup(target)
    database.init_db(copy);database.init_db(copy)
    with database.get_connection(copy) as c:
        assert c.execute('SELECT cash FROM wallet').fetchone()[0]==12345
        assert c.execute('SELECT quantity FROM holdings').fetchone()[0]==2
        assert c.execute('SELECT locked_collateral FROM short_positions').fetchone()[0]==100
        assert c.execute('SELECT timestamp FROM trade_ledger').fetchone()[0]=='2026-01-01T00:00:00Z'
        assert c.execute('PRAGMA journal_mode').fetchone()[0]=='wal'
        assert c.execute('PRAGMA busy_timeout').fetchone()[0]==5000

def test_holiday_and_partial_bars():
    assert targets('AAPL','2026-11-25T21:00:00Z',1)==['2026-11-27T18:00:00Z']
    assert targets('AAPL','2026-11-27T18:00:00Z',1)==['2026-11-30T21:00:00Z']
    hist=pd.DataFrame({'Close':[100,101]},index=pd.to_datetime(['2026-11-23','2026-11-24'],utc=True))
    assert len(completed_history('AAPL',hist,'2026-11-24T15:00:00Z'))==1
    assert utc_z(session_close('BTC-USD','2026-11-23'))=='2026-11-23T23:59:59Z'

def test_stored_winner_and_all_horizon_dedup(db):
    hist=pd.DataFrame({'Close':[100.]},index=pd.to_datetime(['2026-11-25'],utc=True))
    bundle=SimpleNamespace(metadata={'fingerprint':'frozen-v1'})
    first,_=log_forecasts('AAPL',bundle,hist,fake_forecast,now=pd.Timestamp('2026-11-26T10:00Z'))
    def forbidden(*args,**kwargs):raise AssertionError('Duplicate inference')
    second,_=log_forecasts('AAPL',bundle,hist,forbidden,now=pd.Timestamp('2026-11-26T11:00Z'))
    assert {k:r['id'] for k,r in first.items()}=={k:r['id'] for k,r in second.items()}
    assert first==second
    assert first['1']['target_timestamp']=='2026-11-27T18:00:00Z'
    with database.get_connection() as c:assert c.execute('SELECT COUNT(*) FROM forecast_logs').fetchone()[0]==3

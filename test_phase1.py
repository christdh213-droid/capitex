"""Phase 1 verification: offline contracts, optional live provider and startup smoke.

Run: python test_phase1.py [--live] [--smoke]
"""
import argparse
import importlib
import io
import json
from pathlib import Path
import py_compile
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.request import urlopen
import numpy as np
import pandas as pd
import config
import database
import data_loader as dl
import branding


def fixture(freq='h'):
    index = pd.date_range('2026-06-01', '2026-10-27 12:00', freq=freq, tz='UTC')
    close = np.linspace(100, 200, len(index))
    return pd.DataFrame(dict(Open=close, High=close+1, Low=close-1,
                             Close=close, Volume=np.full(len(index), 100)), index=index)


class Phase1Tests(unittest.TestCase):
    def tearDown(self):
        dl.get_live_quote.clear()
        dl._lookback_history.clear()
        dl.load_market_data.clear()

    def test_database_schema_persistence_and_rollback(self):
        with tempfile.TemporaryDirectory(dir=config.DATA_DIR) as tmp:
            db = Path(tmp)/'test.db'
            database.init_db(db)
            with database.get_connection(db) as conn:
                self.assertEqual(conn.execute('PRAGMA journal_mode').fetchone()[0], 'wal')
                self.assertEqual(conn.execute('PRAGMA busy_timeout').fetchone()[0], 5000)
                for table in ('wallet','holdings','trade_ledger','portfolio_snapshots'):
                    self.assertTrue(conn.execute(f'PRAGMA table_info({table})').fetchall())
                conn.execute('UPDATE wallet SET cash=123 WHERE id=1')
            database.init_db(db)
            self.assertEqual(database.get_capitex_wallet(db)['cash'], 123)
            with self.assertRaises(RuntimeError):
                with database.get_connection(db) as conn:
                    conn.execute('UPDATE wallet SET cash=0 WHERE id=1')
                    raise RuntimeError('rollback')
            self.assertEqual(database.get_capitex_wallet(db)['cash'], 123)

    def test_csv_bytes_cleaning(self):
        raw = b'Date,open,high,low,close,volume\n2026-01-02T10:00:00+05:30,10,12,9,11,0\n2026-01-01,9,11,8,10,100\n2026-01-01,9,11,8,,100\nbad,1,2,1,1,10\n'
        df, report = dl.load_from_csv(raw)
        self.assertEqual(len(df), 2)
        self.assertEqual(str(df.index.tz), 'UTC')
        self.assertEqual(df.index[-1].hour, 4)
        self.assertEqual(report['duplicates_removed'], 1)
        self.assertEqual(report['invalid_dates'], 1)
        self.assertEqual(report['nan_after'], 0)
        self.assertGreater(report['imputed_count'], 0)

    def test_bad_csv(self):
        for source in (b'Date,Close\n2026-01-01,1', b'Close\n2', b'Date,Open,High,Low,Close,Volume\nbad,1,2,1,1,1'):
            with self.assertRaises(ValueError):
                dl.load_from_csv(source)

    def test_multiindex_and_invalid_prices(self):
        df = fixture().tail(3)
        df.iloc[0, df.columns.get_loc('Close')] = -1
        df.columns = pd.MultiIndex.from_product([df.columns, ['AAPL']])
        result, report = dl.clean_stock_data(df)
        self.assertEqual(len(result), 2)
        self.assertEqual(report['invalid_bars'], 1)

    def test_all_90_offsets_and_weekend(self):
        now = pd.Timestamp('2026-10-27 12:30', tz='UTC')
        df = fixture()
        df = df.loc[df.index.dayofweek < 5]
        with patch.object(dl, '_utc_now', return_value=now), patch.object(dl, '_lookback_history', return_value=(df,'1h')), patch.object(dl, 'get_live_quote', return_value={'price':210.0}):
            for days in range(1,91):
                result = dl.get_90d_history_and_lookback('AAPL', days)
                target = now-pd.Timedelta(days=days)
                actual = pd.Timestamp(result['historical_timestamp'])
                self.assertEqual(result['requested_timestamp'], target.isoformat())
                self.assertLessEqual(actual+pd.Timedelta(hours=1), target)
                self.assertEqual(result['historical_price'], df.loc[actual,'Close'])
                self.assertAlmostEqual(result['change_pct'], (210/df.loc[actual,'Close']-1)*100)
                self.assertIn(actual.isoformat(), [b['timestamp'] for b in result['history']])
            sunday = dl.get_90d_history_and_lookback('AAPL', 2)
            self.assertEqual(pd.Timestamp(sunday['historical_timestamp']).dayofweek, 4)

    def test_invalid_offsets(self):
        for value in (0,91,-1,1.2,'44',True,None):
            with self.assertRaises(ValueError):
                dl.get_90d_history_and_lookback('AAPL',value)

    def test_insufficient_history(self):
        with patch.object(dl, '_lookback_history', return_value=(fixture().tail(1),'1h')):
            with self.assertRaises(dl.MarketDataError):
                dl.get_90d_history_and_lookback('AAPL',90)

    def test_daily_fallback_is_labelled(self):
        now = pd.Timestamp('2026-10-27 12:30', tz='UTC')
        with patch.object(dl, '_utc_now', return_value=now), patch.object(dl, '_lookback_history', return_value=(fixture('D'),'1d')), patch.object(dl, 'get_live_quote', return_value={'price':210.0}):
            r=dl.get_90d_history_and_lookback('AAPL',44)
            self.assertEqual(r['interval'],'1d')
            self.assertLessEqual(pd.Timestamp(r['historical_available_at']),pd.Timestamp(r['requested_timestamp']))

    def test_cache_copy_clear_and_errors(self):
        calls=[]
        @dl._cached(30)
        def sample(key):
            calls.append(key)
            if key=='fail':
                raise ValueError('failed')
            return {'a':[]}
        a=sample('ok'); a['a'].append(1)
        self.assertEqual(sample('ok'), {'a':[]})
        self.assertEqual(len(calls),1)
        for _ in range(2):
            with self.assertRaises(ValueError): sample('fail')
        self.assertEqual(calls.count('fail'),2)
        sample.clear(); sample('ok')
        self.assertEqual(calls.count('ok'),2)

    def test_quote_change_and_session_status(self):
        now=pd.Timestamp('2026-10-27 12:30',tz='UTC')
        class Provider:
            def history(self,**kwargs): return fixture().tail(250)
            def get_history_metadata(self):
                return {'currentTradingPeriod':{'regular':{'start':int((now-pd.Timedelta(hours=1)).timestamp()),'end':int((now+pd.Timedelta(hours=5)).timestamp())}}}
        with patch.object(dl,'_utc_now',return_value=now), patch.object(dl.yf,'Ticker',return_value=Provider()), patch.object(dl,'_history',return_value=fixture('D')):
            for ticker in ('AAPL','BTC-USD'):
                q=dl.get_live_quote(ticker)
                self.assertGreater(q['price'],0)
                self.assertTrue(q['market_open'])
                self.assertEqual(q['change_basis'],'rolling_24h' if ticker.endswith('-USD') else 'previous_session_close')
                self.assertAlmostEqual(q['change_24h_pct'],(q['price']/q['previous_close']-1)*100)

    def test_quote_outage_raises(self):
        with patch.object(dl.yf,'Ticker',side_effect=RuntimeError('offline')):
            with self.assertRaises(RuntimeError): dl.get_live_quote('AAPL')

    def test_provider_metadata_timestamps(self):
        expected = pd.Timestamp('2026-10-27 12:30', tz='UTC')
        self.assertEqual(dl._provider_timestamp(int(expected.timestamp())), expected)
        self.assertEqual(dl._provider_timestamp(expected), expected)
        self.assertEqual(dl._provider_timestamp(expected.tz_convert('America/New_York')), expected)

    def test_cache_expires_after_30_seconds(self):
        calls=[]
        @dl._cached(30)
        def sample():
            calls.append(1)
            return len(calls)
        with patch.object(dl, 'monotonic', return_value=0):
            self.assertEqual(sample(), 1)
        with patch.object(dl, 'monotonic', return_value=29):
            self.assertEqual(sample(), 1)
        with patch.object(dl, 'monotonic', return_value=30):
            self.assertEqual(sample(), 2)

    def test_offline_brand_assets(self):
        for ticker, asset in config.ASSET_UNIVERSE.items():
            self.assertTrue(Path(asset['logo_fallback']).is_file())
            self.assertIn('data:image/svg+xml;base64,',branding.asset_logo_source(ticker,False))
        self.assertTrue(config.LOGO_PATH.is_file())
        self.assertTrue(config.BANNER_PATH.is_file())
        with patch.object(branding.requests,'get',side_effect=branding.requests.ConnectionError('offline')):
            self.assertIn('data:image/svg+xml;base64,',branding.asset_logo_source('AAPL',True))


def live_checks():
    evidence=[]
    for ticker in ('AAPL','BTC-USD'):
        quote=dl.get_live_quote(ticker)
        assert quote['price']>0
        for days in range(1,91):
            r=dl.get_90d_history_and_lookback(ticker,days)
            assert r['historical_price']>0 and len(r['history'])>0
            assert pd.Timestamp(r['historical_available_at'])<=pd.Timestamp(r['requested_timestamp'])
        evidence.append(dict(ticker=ticker,quote=quote,offsets_verified=90,
                             bars=len(r['history']),interval=r['interval']))
    (config.DATA_DIR/'phase1_live_results.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print('LIVE PASS: AAPL and BTC-USD quotes; all 180 calendar-day lookbacks.')


def _free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        return sock.getsockname()[1]


def smoke_checks(log_prefix='phase1'):
    processes=[]
    logs=[]
    try:
        for name,args,url_suffix in (
            ('backend',['uvicorn','main:app','--host','127.0.0.1'], '/api/health'),
            ('frontend',['streamlit','run','app.py','--server.headless=true','--server.address=127.0.0.1'], '/_stcore/health')):
            port=_free_port()
            command=[sys.executable,'-m',*args, *(['--port',str(port)] if name=='backend' else [f'--server.port={port}'])]
            log=(config.DATA_DIR/f'{log_prefix}_{name}_smoke.log').open('w',encoding='utf-8')
            logs.append(log)
            process=subprocess.Popen(command,cwd=config.BASE_DIR,stdout=log,stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            processes.append(process)
            deadline=time.monotonic()+75
            while time.monotonic()<deadline:
                if process.poll() is not None:
                    raise RuntimeError(f'{name} exited; see smoke log.')
                try:
                    with urlopen(f'http://127.0.0.1:{port}{url_suffix}',timeout=1) as response:
                        if response.status==200: break
                except OSError:
                    time.sleep(.3)
            else:
                raise RuntimeError(f'{name} startup timeout')
            print(f'SMOKE PASS: {name} health HTTP 200')
        # Runs the actual Streamlit script, beyond just its web server health.
        from streamlit.testing.v1 import AppTest
        with patch.object(dl,'get_live_price',return_value=dict(price=200.0,previous_close=198.0,is_fallback=True,status_message='Smoke fixture',timestamp='2026-10-27T12:00:00Z',currency='USD')):
            app=AppTest.from_file(str(config.BASE_DIR/'app.py')).run(timeout=75)
            assert not app.exception, [e.message for e in app.exception]
        print('SMOKE PASS: actual Streamlit script rendered without exceptions')
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=8)
                except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
        for log in logs: log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    options=parser.parse_args()
    files=sorted(config.BASE_DIR.rglob('*.py'))
    for path in files:
        subprocess.run([sys.executable,'-m','py_compile',str(path)],check=True,capture_output=True)
    print(f'COMPILE PASS: python -m py_compile on {len(files)} Python files')
    # Compilation cannot detect missing imports; check imports independently.
    for name in ('config','database','data_loader','branding','indicators','lstm_model','reasoning_engine','trading_engine','chart_plotter','backtester','main','run'):
        importlib.import_module(name)
    print('IMPORT PASS: application modules')
    outcome=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Phase1Tests))
    if not outcome.wasSuccessful(): sys.exit(1)
    if options.live: live_checks()
    if options.smoke: smoke_checks()

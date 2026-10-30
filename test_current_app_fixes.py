"""Regression tests for current-app fixes. Uses isolated databases, never live orders."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
import config
import database
import legacy_database
import data_loader
import trading_engine as engine
import main
import chart_plotter
from formatting import signed_money, local_timestamp, performance_text


class CurrentAppFixTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=config.DATA_DIR)
        self.patches=[patch.object(config,'LEGACY_DB_PATH',Path(self.temp.name)/'wallets.db'),
                      patch.object(config,'DB_PATH',Path(self.temp.name)/'core.db'),
                      patch.object(engine,'get_cached_live_price',return_value=63000.)]
        for p in self.patches: p.start()
        database.init_db()
        self.quote=patch.object(data_loader,'get_live_quote',return_value={
            'price':63000.,'timestamp':'2026-10-27T04:00:00+00:00','is_fallback':False})
        self.live=self.quote.start()

    def tearDown(self):
        self.quote.stop()
        for p in reversed(self.patches): p.stop()
        self.temp.cleanup()

    def rows(self,sql,args=()):
        with database.get_legacy_connection() as conn:
            return [dict(r) for r in conn.execute(sql,args)]

    def test_btc_human_and_ai_routes_share_live_fill_and_ledger(self):
        with TestClient(main.app) as client:
            human=client.post('/trade/execute',json={'wallet_type':'human','ticker':'BTC-USD',
                'action':'BUY','quantity':.01,'price':50000.}).json()
            self.assertEqual(human['trade']['price'],63000.)
            forecast={'signal':'BUY','predicted_return_pct':1.,'current_close':50000.}
            with patch.object(main,'predict_asset',return_value=forecast):
                ai=client.post('/trade/ai-step',json={'ticker':'BTC-USD','tranche_pct':10}).json()
            self.assertTrue(ai['success'])
            self.assertEqual(ai['price'],63000.)
            self.assertEqual(ai['signal_reference_price'],50000.)
            self.assertAlmostEqual(ai['trade']['quantity'],10000./63000.)
            self.assertAlmostEqual(ai['trade']['cost'],10000.)
            self.live.return_value['price']=65000.
            human_sell=client.post('/trade/execute',json={'wallet_type':'human','ticker':'BTC-USD',
                'action':'SELL','quantity':.01,'price':1.}).json()
            forecast['signal']='SELL'
            with patch.object(main,'predict_asset',return_value=forecast):
                ai_sell=client.post('/trade/ai-step',json={'ticker':'BTC-USD'}).json()
            self.assertEqual(human_sell['trade']['price'],65000.)
            self.assertEqual(ai_sell['price'],65000.)
        fills=self.rows('SELECT wallet_type,action,price,execution_source FROM trade_ledger ORDER BY id')
        self.assertEqual([r['price'] for r in fills],[63000.,63000.,65000.,65000.])
        self.assertEqual({r['wallet_type'] for r in fills},{'human','ai'})
        self.assertTrue(all(r['execution_source']=='live_quote' for r in fills))
        self.assertEqual(self.live.call_count,4)
        for call in self.live.call_args_list:
            self.assertEqual(call.args[0],'BTC-USD')
            self.assertTrue(call.kwargs['force_refresh'])
        for wallet in ('human','ai'):
            self.assertEqual(len(self.rows("SELECT id FROM portfolio_snapshots WHERE wallet_type=? AND reason='trade'",(wallet,))),2)

    def test_btc_fallback_is_recorded_and_labeled(self):
        self.live.side_effect=ConnectionError('quote unavailable')
        history=pd.DataFrame({'Close':[61000.]},index=pd.DatetimeIndex(['2026-10-26'],tz='UTC'))
        with patch.object(data_loader,'_history',return_value=history):
            ok,message,receipt=engine.execute_buy('human','BTC-USD',.01,current_price=1.)
        self.assertTrue(ok)
        self.assertEqual(receipt['price'],61000.)
        self.assertTrue(receipt['is_fallback'])
        self.assertIn('FALLBACK',message)
        row=database.get_trade_ledger('human')[0]
        self.assertEqual(row['price'],61000.)
        self.assertEqual(row['execution_source'],'last_close')
        self.assertIn('FALLBACK',row['execution_label'])

    def test_missing_quotes_reject_order_without_debit(self):
        self.live.side_effect=ConnectionError('offline')
        with patch.object(data_loader,'_history',side_effect=ValueError('offline')):
            ok,_,_=engine.execute_buy('human','BTC-USD',.01,1.)
        self.assertFalse(ok)
        self.assertEqual(database.get_wallet('human')['cash'],100000.)
        self.assertFalse(database.get_trade_ledger('human'))

    def test_execution_bypasses_display_cache(self):
        calls=[]
        @data_loader._cached(30)
        def quote(*,force_refresh=False):
            calls.append(1)
            return len(calls)
        self.assertEqual(quote(),1)
        self.assertEqual(quote(),1)
        self.assertEqual(quote(force_refresh=True),2)
        self.assertEqual(quote(force_refresh=True),3)

    def test_refresh_throttle_is_per_wallet_and_cross_connection(self):
        base=datetime(2026,10,27,tzinfo=timezone.utc)
        def record(wallet,seconds):
            return database.record_snapshot(wallet,100000.,(base+timedelta(seconds=seconds)).isoformat(),reason='refresh',min_interval_seconds=30)
        self.assertGreater(record('human',0),0)
        self.assertEqual(record('human',29),0)
        self.assertGreater(record('human',30),0)
        self.assertGreater(record('ai',0),0)
        with ThreadPoolExecutor(max_workers=4) as pool:
            writes=list(pool.map(lambda _:record('ai',30),range(8)))
        self.assertEqual(sum(bool(v) for v in writes),1)

    def test_every_risk_check_and_reset_has_snapshot(self):
        engine.check_risk_triggers('human')
        engine.check_risk_triggers('human')
        self.assertEqual(len(self.rows("SELECT id FROM portfolio_snapshots WHERE wallet_type='human' AND reason='risk'")),2)
        database.reset_wallets()
        for wallet in ('human','ai'):
            snaps=database.get_portfolio_snapshots(wallet)
            self.assertEqual(len(snaps),1)
            self.assertEqual(snaps[0]['reason'],'reset')
            self.assertEqual(snaps[0]['net_worth'],100000.)

    def test_refresh_updates_peak_without_risk_button(self):
        engine.execute_buy('human','BTC-USD',.01,1.)
        with patch.object(engine,'get_cached_live_price',return_value=65000.):
            summary=engine.get_portfolio_summary('human')
        self.assertEqual(summary['holdings'][0]['peak_price'],65000.)
        with patch.object(engine,'get_cached_live_price',return_value=64000.):
            summary=engine.get_portfolio_summary('human')
        database.update_holding_peak_price('human','BTC-USD',1.)
        self.assertEqual(database.get_holding_for_ticker('human','BTC-USD')['peak_price'],65000.)
        self.assertEqual(summary['holdings'][0]['peak_price'],65000.)
        self.assertEqual(len(self.rows("SELECT id FROM portfolio_snapshots WHERE wallet_type='human' AND reason='refresh'")),1)

    def test_no_closed_stats_and_reset_is_not_a_trade(self):
        database.reset_wallets()
        summary=engine.get_portfolio_summary('human',snapshot_reason=None)
        self.assertIsNone(summary['win_rate_pct'])
        self.assertIsNone(summary['win_loss_ratio'])
        self.assertEqual(summary['total_trades'],0)
        self.assertEqual(performance_text(summary)[0],'—')
        self.assertIn('Win: —',performance_text(summary)[1])
        engine.execute_buy('human','BTC-USD',.01,1.)
        summary=engine.get_portfolio_summary('human',snapshot_reason=None)
        self.assertEqual(summary['total_trades'],1)
        self.assertEqual(summary['total_closed_trades'],0)
        self.live.return_value['price']=64000.
        engine.execute_sell('human','BTC-USD',.01,1.)
        summary=engine.get_portfolio_summary('human',snapshot_reason=None)
        self.assertEqual(summary['total_trades'],2)
        self.assertEqual(summary['total_closed_trades'],1)
        self.assertEqual(summary['win_rate_pct'],100.)
        with database.get_legacy_connection() as conn:
            conn.execute('DELETE FROM portfolio_snapshots')
        summary=engine.get_portfolio_summary('human',snapshot_reason=None)
        self.assertIsNone(summary['max_drawdown_pct'])
        self.assertNotIn('MDD',performance_text(summary)[1])

    def test_chart_padding_precision_and_short_history(self):
        snap=lambda i,v:{'timestamp':f'2026-10-27T01:00:0{i}+00:00','net_worth':v}
        short=[snap(0,100000.),snap(1,100013.1)]
        fig=chart_plotter.plot_arena_net_worth_history(short,short)
        self.assertTrue(any('Not enough history yet' in a.text for a in fig.layout.annotations))
        self.assertEqual(len(fig.data),0)
        self.assertEqual(fig.layout.yaxis.tickformat,'$,.2f')
        low,high=fig.layout.yaxis.range
        self.assertLessEqual(low,99500.)
        self.assertGreaterEqual(high,100500.)
        fig=chart_plotter.plot_arena_net_worth_history(short+[snap(2,100020.)],[])
        self.assertEqual(len(fig.data),1)

    def test_currency_and_local_timestamp_format(self):
        self.assertEqual(signed_money(13.1),'+$13.10')
        self.assertEqual(signed_money(-13.1),'-$13.10')
        self.assertEqual(local_timestamp('2026-10-26T17:50:30.460943+00:00'),'2026-10-26 23:20:30 IST')

    def test_streamlit_renders_without_phase_tracker_or_zero_win_stats(self):
        from streamlit.testing.v1 import AppTest
        quote=dict(price=63000.,previous_close=62000.,timestamp='2026-10-27T04:00:00Z',
                   currency='USD',is_fallback=False,status_message='Test quote')
        with patch.object(data_loader,'get_live_price',return_value=quote), patch('requests.get',side_effect=ConnectionError('test offline')):
            app=AppTest.from_file(str(config.BASE_DIR/'legacy_app.py')).run(timeout=60)
        self.assertFalse(app.exception,[e.message for e in app.exception])
        self.assertTrue(any(t.value=='CAPITEX' for t in app.sidebar.title))
        self.assertFalse(any('Phase 1' in str(t.value) for t in app.success))
        self.assertEqual(sum(m.value=='—' for m in app.metric),2)
        self.assertTrue(all('Win: 0.0%' not in str(m.delta) for m in app.metric))


if __name__=='__main__':
    import subprocess,sys
    files=sorted(config.BASE_DIR.rglob('*.py'))
    for file in files:
        subprocess.run([sys.executable,'-m','py_compile',str(file)],check=True,capture_output=True)
    print(f'COMPILE PASS: all {len(files)} Python files',flush=True)
    unittest.main(verbosity=2)

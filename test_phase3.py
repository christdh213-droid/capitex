"""Phase 3 wallet/API/UI tests. All orders use temporary databases and mocked quotes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit
import uuid
import json
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
import config
import database
import branding
import data_loader
import trading_engine as engine
import main


class WalletTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=config.DATA_DIR)
        self.prices={t:(63000. if t=='BTC-USD' else 2500. if t=='ETH-USD' else 200.) for t in config.DEFAULT_TICKERS}
        original=branding.asset_logo_source
        def quote(ticker,**kwargs):
            price=self.prices.get(ticker,200.)
            return dict(ticker=ticker,price=price,previous_close=price/1.01,change_24h=price-price/1.01,
                        change_24h_pct=1.,change_basis='rolling_24h' if ticker.endswith('-USD') else 'previous_session_close',
                        timestamp=pd.Timestamp.now(tz='UTC').isoformat(),market_status='open',market_open=True,
                        is_fallback=False,is_stale=False,interval='1m',currency='USD')
        def history(ticker):
            times=pd.date_range(pd.Timestamp.now(tz='UTC').floor('h')-pd.Timedelta(days=120),periods=2881,freq='h')
            values=np.linspace(self.prices.get(ticker,200)*.9,self.prices.get(ticker,200),len(times))
            return pd.DataFrame(dict(Open=values,High=values+1,Low=values-1,Close=values,Volume=np.full(len(times),1000.)),index=times),'1h'
        self.patches=[patch.object(config,'DB_PATH',Path(self.temp.name)/'capitex.db'),
            patch.object(config,'LEGACY_DB_PATH',Path(self.temp.name)/'prototype.db'),
            patch.object(data_loader,'get_live_quote',side_effect=quote),
            patch.object(data_loader,'_lookback_history',side_effect=history),
            patch.object(branding,'asset_logo_source',side_effect=lambda t,try_remote=True:original(t,False))]
        for p in self.patches:p.start()
        self.client=TestClient(main.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None,None,None)
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()

    def order(self,**changes):
        payload={'ticker':'BTC-USD','action':'BUY','amount':1000.,'client_order_id':str(uuid.uuid4())}
        payload.update(changes)
        return self.client.post('/api/trade',json=payload)

    def test_wallet_home_schema_and_empty_stats(self):
        wallet=self.client.get('/api/wallet').json()
        self.assertEqual(wallet['cash'],100000.)
        self.assertEqual(wallet['net_worth'],100000.)
        self.assertEqual(len(wallet['watchlist']),10)
        self.assertEqual(wallet['holdings'],[])
        self.assertIsNone(wallet['win_rate_pct'])
        self.assertIsNone(wallet['win_loss_ratio'])
        self.assertEqual(wallet['change_period'],'since_start')
        self.assertTrue(all(row['logo'].startswith('data:image/') for row in wallet['watchlist']))
        json.dumps(wallet,allow_nan=False)

    def test_buy_dollars_sell_quantity_and_ledger_fills(self):
        buy=self.order().json()['trade']
        self.assertEqual(buy['price'],63000.)
        self.assertLessEqual(buy['total_value'],1000.)
        wallet=self.client.get('/api/wallet').json()
        self.assertAlmostEqual(wallet['net_worth'],100000.,places=5)
        self.assertEqual(len(wallet['holdings']),1)
        self.prices['BTC-USD']=65000.
        sold=self.order(action='SELL',amount=None,quantity=buy['quantity']).json()['trade']
        self.assertEqual(sold['price'],65000.)
        self.assertGreater(sold['realized_pnl'],0)
        self.assertEqual(sold['execution_source'],'live_quote')
        wallet=self.client.get('/api/wallet').json()
        self.assertEqual(wallet['holdings'],[])
        self.assertEqual(wallet['total_closed_trades'],1)
        self.assertEqual(wallet['win_rate_pct'],100.)
        self.assertEqual(wallet['total_trades'],2)
        self.assertEqual(len([s for s in wallet['snapshots'] if s['reason']=='trade']),2)

    def test_quantity_buy_dollar_sell_and_weighted_average(self):
        self.order(amount=None,quantity=.1)
        self.prices['BTC-USD']=65000.
        self.order(amount=None,quantity=.1)
        wallet=self.client.get('/api/wallet').json()
        self.assertAlmostEqual(wallet['holdings'][0]['avg_buy_price'],64000.)
        self.assertAlmostEqual(wallet['holdings'][0]['quantity'],.2)
        sale=self.order(action='SELL',amount=650.).json()['trade']
        self.assertAlmostEqual(sale['quantity'],.01)
        self.assertAlmostEqual(sale['total_value'],650.)

    def test_max_cash_does_not_overdraw(self):
        receipt=self.order(amount=100000.).json()['trade']
        self.assertGreaterEqual(receipt['cash'],0)
        self.assertLess(receipt['cash'],.00001)
        rejected=self.order(amount=1.)
        self.assertEqual(rejected.status_code,400)

    def test_invalid_and_insufficient_orders_are_atomic(self):
        for details in ({'amount':-1},{'amount':0},{'amount':200000},
                        {'quantity':.1},{'amount':None},{'action':'HACK'},
                        {'ticker':'../unsafe'},{'amount':None,'quantity':0}):
            with self.subTest(details=details):
                self.assertIn(self.order(**details).status_code,(400,422))
        self.assertEqual(self.order(action='SELL',amount=None,quantity=1).status_code,400)
        wallet=self.client.get('/api/wallet').json()
        self.assertEqual(wallet['cash'],100000.)
        self.assertEqual(wallet['total_trades'],0)

    def test_duplicate_retry_is_one_fill_and_conflicting_id_rejected(self):
        key=str(uuid.uuid4())
        first=self.order(client_order_id=key).json()['trade']
        second=self.order(client_order_id=key).json()['trade']
        self.assertEqual(first['id'],second['id'])
        self.assertTrue(second['replayed'])
        self.assertEqual(self.order(client_order_id=key,amount=2000.).status_code,400)
        self.assertEqual(self.client.get('/api/wallet').json()['total_trades'],1)

    def test_parallel_orders_cannot_overspend(self):
        def buy(_):
            try:
                return engine.execute_capitex_trade('BTC-USD','BUY',amount=60000.)
            except ValueError: return None
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(buy,range(2)))
        self.assertEqual(sum(r is not None for r in results),1)
        wallet=self.client.get('/api/wallet').json()
        self.assertGreaterEqual(wallet['cash'],0)
        self.assertEqual(wallet['total_trades'],1)

    def test_asset_exact_lookback_validation_and_holding(self):
        self.order()
        for days in (1,44,90):
            result=self.client.get(f'/api/asset/BTC-USD?days_ago={days}')
            self.assertEqual(result.status_code,200)
            asset=result.json()
            self.assertEqual(asset['days_ago'],days)
            self.assertLessEqual(pd.Timestamp(asset['historical_available_at']),pd.Timestamp(asset['requested_timestamp']))
            self.assertIsNotNone(asset['holding'])
            self.assertGreater(len(asset['history']),100)
        for days in (0,91,'1.5'):
            self.assertEqual(self.client.get(f'/api/asset/AAPL?days_ago={days}').status_code,422)

    def test_refresh_peaks_and_throttle(self):
        self.order()
        self.prices['BTC-USD']=67000.
        a=self.client.get('/api/wallet').json()
        self.prices['BTC-USD']=64000.
        b=self.client.get('/api/wallet').json()
        self.assertEqual(a['holdings'][0]['peak_price'],67000.)
        self.assertEqual(b['holdings'][0]['peak_price'],67000.)
        self.assertEqual(len([s for s in b['snapshots'] if s['reason']=='refresh']),1)

    def test_risk_and_reset_snapshot_and_legacy_isolation(self):
        self.order(stop_loss_pct=3.)
        self.client.get('/api/wallet')
        self.prices['BTC-USD']=60000.
        risk=self.client.post('/api/wallet/risk-check').json()
        self.assertEqual(risk['triggered_count'],1)
        history=self.client.get('/api/wallet').json()
        self.assertTrue(any(s['reason']=='risk' for s in history['snapshots']))
        self.assertEqual(history['trade_ledger'][0]['action'],'STOP_LOSS')
        self.assertEqual(self.client.post('/api/wallet/reset').status_code,200)
        wallet=self.client.get('/api/wallet').json()
        self.assertEqual(wallet['cash'],100000.)
        self.assertEqual(wallet['holdings'],[])
        self.assertEqual(wallet['total_trades'],0)
        self.assertEqual(wallet['snapshots'][0]['reason'],'reset')
        self.assertEqual(database.get_wallet('human')['cash'],100000.)

    def test_partial_market_outage_is_explicit(self):
        self.order()
        def fail(ticker):
            asset=engine.asset_metadata(ticker)
            return dict(asset,price=None,available=False,change_24h=None,change_24h_pct=None)
        with patch.object(engine,'_market_asset',side_effect=fail):
            wallet=self.client.get('/api/wallet').json()
        self.assertTrue(wallet['valuation_estimated'])
        self.assertEqual(wallet['holdings'][0]['valuation_source'],'cost_estimate')
        self.assertTrue(all(row['price'] is None for row in wallet['watchlist']))
        self.assertAlmostEqual(wallet['net_worth'],100000.,places=5)

    def test_prediction_route_passes_all_horizons_to_phase2_engine(self):
        history=pd.DataFrame({'Close':[200.]},index=pd.DatetimeIndex(['2026-10-01'],tz='UTC'))
        def forecast(bundle,horizon,**kwargs):
            return {'outlook':'bullish','expected_change_pct':1.,'confidence':50.,
                    'median_target_price':202.,'horizon_label':str(horizon),
                    'corridor':{'median':[200.,202.],'lower':[200.,190.],'upper':[200.,210.]}}
        with patch.object(main,'get_capitex_forecast_bundle',return_value=(object(),history)), \
             patch.object(main.lstm_model,'forecast_multihorizon',side_effect=forecast) as called, \
             patch.object(main.reasoning_engine,'build_reasoning',return_value={'sentiment':{'score':50},'catalyst_tags':[]}):
            for h in ('24h','7d','30d'):
                r=self.client.get(f'/api/predict/AAPL?horizon={h}')
                self.assertEqual(r.status_code,200)
                self.assertIn('hero',r.json());self.assertIn('why',r.json())
            self.assertEqual([c.args[1] for c in called.call_args_list],[1,7,30])
        self.assertEqual(self.client.get('/api/predict/AAPL?horizon=60d').status_code,400)

    def test_full_streamlit_home_asset_lookback_and_order(self):
        from streamlit.testing.v1 import AppTest
        def request(method,url,json=None,**kwargs):
            parts=urlsplit(url)
            return self.client.request(method,parts.path+('?' + parts.query if parts.query else ''),json=json)
        with patch('requests.request',side_effect=request):
            page=AppTest.from_file(str(config.BASE_DIR/'app.py')).run(timeout=60)
            self.assertFalse(page.exception,[e.message for e in page.exception])
            button=next(b for b in page.button if b.key=='open_market_BTC-USD')
            button.click().run(timeout=60)
            self.assertFalse(page.exception,[e.message for e in page.exception])
            self.assertEqual(page.session_state['view'],'asset')
            self.assertEqual(page.slider(key='lookback_slider').value,44)
            self.assertEqual(page.number_input(key='lookback_exact').value,44)
            self.assertEqual(page.session_state['timeframe'],'90D')
            page.number_input(key='lookback_exact').set_value(90).run(timeout=60)
            self.assertFalse(page.exception,[e.message for e in page.exception])
            self.assertEqual(page.session_state['lookback_slider'],90)
            page.button(key='buy_action').click().run(timeout=60)
            self.assertFalse(page.exception,[e.message for e in page.exception])
            page.number_input(key='order_size').set_value(500.).run(timeout=60)
            page.button(key='execute_order').click().run(timeout=60)
            self.assertFalse(page.exception,[e.message for e in page.exception])
            self.assertEqual(self.client.get('/api/wallet').json()['total_trades'],1)
            self.assertTrue(any('BUY complete' in s.value for s in page.success))
            page.button(key='sell_action').click().run(timeout=60)
            page.button(key='fill_MAX').click().run(timeout=60)
            self.assertFalse(page.exception,[e.message for e in page.exception])
            self.assertEqual(page.session_state['order_unit'],'Share / Token Quantity')
            page.button(key='execute_order').click().run(timeout=60)
            self.assertFalse(page.exception,[e.message for e in page.exception])
            self.assertEqual(self.client.get('/api/wallet').json()['holdings'],[])


if __name__=='__main__':
    import subprocess,sys
    files=list(config.BASE_DIR.rglob('*.py'))
    for file in files: subprocess.run([sys.executable,'-m','py_compile',str(file)],check=True,capture_output=True)
    print(f'COMPILE PASS: all {len(files)} Python files',flush=True)
    unittest.main(verbosity=2)

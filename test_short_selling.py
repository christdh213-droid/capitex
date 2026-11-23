"""Short accounting/API regression tests using isolated databases and BTC-USD quotes."""
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import trading_engine as engine
from test_phase3 import WalletTests

class ShortTests(WalletTests):
    def short(self,qty=1,cover=False):
        return self.client.post('/api/short/cover' if cover else '/api/short/open',json={'symbol':'BTC-USD','quantity':qty})
    def wallet(self):return self.client.get('/api/wallet').json()
    def test_short_equity_and_profit(self):
        self.prices['BTC-USD']=100
        self.assertEqual(self.short(10).status_code,200)
        w=self.wallet();self.assertEqual(w['total_equity'],100000);self.assertEqual(w['cash'],99000);self.assertEqual(w['total_locked_collateral'],1000)
        self.prices['BTC-USD']=80
        r=self.short(10,True).json()['trade'];self.assertEqual(r['realized_pnl'],200)
        self.assertEqual(self.wallet()['cash'],100200)
    def test_short_loss(self):
        self.prices['BTC-USD']=100;self.short(10);self.prices['BTC-USD']=120
        self.assertEqual(self.short(10,True).json()['trade']['realized_pnl'],-200)
        self.assertEqual(self.wallet()['cash'],99800)
    def test_weighted_and_partial(self):
        self.prices['BTC-USD']=100;self.short(10);self.prices['BTC-USD']=120;self.short(10)
        r=self.wallet()['short_positions'][0];self.assertEqual(r['avg_entry_price'],110);self.assertEqual(r['locked_collateral'],2200)
        self.prices['BTC-USD']=90;self.short(5,True)
        w=self.wallet();r=w['short_positions'][0]
        self.assertEqual(r['quantity'],15);self.assertEqual(r['locked_collateral'],1650);self.assertEqual(r['avg_entry_price'],110)
        self.assertEqual(w['cash'],98450);self.assertEqual(w['net_worth'],100400)
    def test_validation_and_liquidation(self):
        self.prices['BTC-USD']=100
        self.assertEqual(self.short(1001).status_code,400)
        self.assertEqual(self.short(0).status_code,422)
        self.short(10)
        self.assertEqual(self.short(11,True).status_code,400)
        self.prices['BTC-USD']=210
        w=self.wallet();self.assertEqual(w['short_positions'],[]);self.assertEqual(w['net_worth'],99000)
        self.assertEqual(w['trade_ledger'][0]['action'],'SHORT_LIQUIDATED');self.assertEqual(w['trade_ledger'][0]['realized_pnl'],-1000)
        self.assertEqual(w['total_closed_trades'],1)
    def test_cover_at_threshold_and_reset(self):
        self.prices['BTC-USD']=100;self.short(10);self.prices['BTC-USD']=200
        self.assertEqual(self.short(1,True).json()['trade']['action'],'SHORT_LIQUIDATED')
        self.assertEqual(self.wallet()['short_positions'],[])
        self.prices['BTC-USD']=100;self.short(1);self.client.post('/api/wallet/reset')
        self.assertEqual(self.wallet()['short_positions'],[]);self.assertEqual(self.wallet()['cash'],100000)
    def test_fallback_rejected_and_idempotent(self):
        with patch.object(engine,'get_execution_price',return_value={'price':100,'is_fallback':True,'execution_source':'last_close'}):
            self.assertEqual(self.short().status_code,400)
        payload={'ticker':'BTC-USD','action':'SHORT_OPEN','quantity':.1,'client_order_id':'test-short-retry'}
        self.assertEqual(self.client.post('/api/trade',json=payload).status_code,200)
        self.assertTrue(self.client.post('/api/trade',json=payload).json()['trade']['replayed'])
        self.assertAlmostEqual(self.wallet()['short_positions'][0]['quantity'],.1)

    def test_short_ui_roundtrip(self):
        from streamlit.testing.v1 import AppTest
        from urllib.parse import urlsplit
        def request(method,url,json=None,**kwargs):
            parts=urlsplit(url)
            return self.client.request(method,parts.path+('?' + parts.query if parts.query else ''),json=json)
        self.prices['BTC-USD']=100
        with patch('requests.request',side_effect=request):
            page=AppTest.from_file(str(__import__('config').BASE_DIR/'app.py')).run(timeout=60)
            page.button(key='open_market_BTC-USD').click().run(timeout=60)
            page.button(key='SHORT_OPEN').click().run(timeout=60)
            page.number_input(key='order_size').set_value(1000.).run(timeout=60)
            page.button(key='execute_order').click().run(timeout=60)
            self.assertFalse(page.exception)
            self.assertEqual(self.wallet()['short_positions'][0]['quantity'],10)
            self.prices['BTC-USD']=80
            page.button(key='SHORT_COVER').click().run(timeout=60)
            page.button(key='fill_MAX').click().run(timeout=60)
            page.button(key='execute_order').click().run(timeout=60)
            self.assertFalse(page.exception)
            self.assertEqual(self.wallet()['short_positions'],[])
            self.assertEqual(self.wallet()['cash'],100200)

if __name__=='__main__':unittest.main()

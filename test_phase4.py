"""Phase 4 forecast chart, route and consumer flow verification. No real trades."""
import copy
import json
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
import config
import main
import chart_plotter
import test_phase3 as fixtures


def forecast(bundle,steps,current_price,ticker):
    path=np.linspace(current_price,current_price*1.043,steps+1)
    spread=current_price*.02*np.sqrt(np.arange(steps+1))
    return dict(ticker=ticker,horizon=steps,horizon_label={1:'Next 24 Hours',7:'7 Days',30:'30 Days'}[steps],
        horizon_basis='calendar_days' if ticker.endswith('-USD') else 'trading_sessions',
        current_price=current_price,outlook='bullish',expected_change_pct=4.3,confidence=82.,
        median_target_price=float(path[-1]),interval_level=.8,
        corridor=dict(median=path.tolist(),lower=(path-spread).tolist(),upper=(path+spread).tolist()))


WHY={'sentiment':{'score':68.,'label':'Greed','headlines_used':2},
    'technical_summary':{'tags':[{'label':'RSI: Neutral (54.1)','tone':'neutral'},
        {'label':'MACD: Bullish Crossover','tone':'bullish'}]},
    'catalyst_tags':[{'label':'Volume Surge (+180%)','tone':'caution'}]}


class PredictorTests(unittest.TestCase):
    def setUp(self):
        fixtures.WalletTests.setUp(self)
        history=pd.DataFrame({'Close':np.linspace(170,200,90)},
            index=pd.date_range(end=pd.Timestamp.now(tz='UTC')-pd.Timedelta(days=2),periods=90))
        self.prediction_patches=[
            patch.object(main,'get_capitex_forecast_bundle',return_value=(object(),history)),
            patch.object(main.lstm_model,'forecast_multihorizon',side_effect=forecast),
            patch.object(main.reasoning_engine,'build_reasoning',return_value=copy.deepcopy(WHY))]
        for p in self.prediction_patches:p.start()

    def tearDown(self):
        for p in reversed(self.prediction_patches):p.stop()
        fixtures.WalletTests.tearDown(self)

    def test_route_live_anchor_future_dates_all_horizons(self):
        for ticker in ('AAPL','BTC-USD'):
            for horizon,steps in [('24h',1),('7d',7),('30d',30)]:
                result=self.client.get(f'/api/predict/{ticker}?horizon={horizon}')
                self.assertEqual(result.status_code,200)
                p=result.json();c=p['corridor']
                self.assertEqual(len(p['history']),60)
                self.assertEqual(len(c['timestamps']),steps+1)
                self.assertEqual(c['timestamps'][0],p['quote']['timestamp'])
                self.assertTrue(all(pd.Timestamp(d)>pd.Timestamp(c['timestamps'][0]) for d in c['timestamps'][1:]))
                if ticker=='AAPL':self.assertTrue(all(pd.Timestamp(d).weekday()<5 for d in c['timestamps'][1:]))
                for k in ('median','lower','upper'):self.assertEqual(c[k][0],p['current_price'])
                self.assertEqual(p['hero']['confidence'],82.)
                json.dumps(p,allow_nan=False)

    def test_chart_funnel_target_and_no_technical_clutter(self):
        for horizon in ('24h','7d','30d'):
            p=self.client.get(f'/api/predict/BTC-USD?horizon={horizon}').json()
            fig=chart_plotter.plot_prediction_corridor(p)
            self.assertEqual(sum(t.fill=='tonexty' for t in fig.data),3)
            self.assertEqual(fig.data[-1].y[0],p['median_target_price'])
            self.assertIn('Target: $',fig.layout.annotations[0].text)
            self.assertEqual(fig.data[-2].name,'Target glow')
            self.assertFalse(any(any(s in (t.name or '') for s in ('RSI','SMA','MACD','EMA')) for t in fig.data))
            self.assertEqual(fig.layout.yaxis.tickformat,'$,.2f')
        bad=copy.deepcopy(p);bad['corridor']['lower'][0]=0
        with self.assertRaises(ValueError):chart_plotter.plot_prediction_corridor(bad)
        bad=copy.deepcopy(p);bad['corridor']['timestamps'].pop()
        with self.assertRaises(ValueError):chart_plotter.plot_prediction_corridor(bad)

    def request(self,method,url,json=None,**kwargs):
        parts=urlsplit(url)
        return self.client.request(method,parts.path+('?' + parts.query if parts.query else ''),json=json)

    def test_predictor_horizons_reasoning_and_trade_bridge(self):
        with patch('requests.request',side_effect=self.request):
            page=AppTest.from_file(str(config.BASE_DIR/'app.py')).run(timeout=60)
            page.session_state['main_navigation']='🧠 AI Predictor'
            page.run(timeout=60)
            self.assertFalse(page.exception,[e.message for e in page.exception])
            self.assertEqual(page.expander[0].label,'🔍 View AI Reasoning Factors')
            self.assertFalse(page.expander[0].proto.expanded)
            self.assertEqual(page.session_state['prediction_horizon'],'7 Days')
            content=' '.join(m.value for m in page.markdown)
            self.assertIn('Bullish Outlook',content);self.assertIn('82%',content)
            self.assertIn('Greed',content);self.assertIn('Volume Surge',content)
            page.selectbox(key='prediction_asset').select('BTC-USD').run(timeout=60)
            for label in ('Next 24 Hours','7 Days','30 Days'):
                page.session_state['prediction_horizon']=label
                page.run(timeout=60)
                self.assertFalse(page.exception,[e.message for e in page.exception])
                self.assertEqual(len(page.get('plotly_chart')),1)
            page.button(key='trade_prediction').click().run(timeout=60)
            self.assertFalse(page.exception,[e.message for e in page.exception])
            self.assertEqual(page.session_state['main_navigation'],'👛 Wallet')
            self.assertEqual(page.session_state['asset'],'BTC-USD')
            self.assertEqual(page.session_state['order_action'],'BUY')
            self.assertTrue(any(b.key=='execute_order' for b in page.button))
            self.assertEqual(self.client.get('/api/wallet').json()['total_trades'],0)

    def test_outage_leaves_controls_and_retry(self):
        with patch('requests.request',side_effect=self.request),patch.object(main,'get_capitex_forecast_bundle',side_effect=main.data_loader.MarketDataError('offline')):
            page=AppTest.from_file(str(config.BASE_DIR/'app.py')).run(timeout=60)
            page.session_state['main_navigation']='🧠 AI Predictor';page.run(timeout=60)
            self.assertFalse(page.exception)
            self.assertTrue(page.error)
            self.assertTrue(page.button(key='refresh_prediction'))
            self.assertEqual(page.selectbox(key='prediction_asset').value,'AAPL')

    def test_refresh_cache_and_empty_catalysts(self):
        why=copy.deepcopy(WHY);why['catalyst_tags']=[];why['sentiment']['headlines_used']=0
        with patch('requests.request',side_effect=self.request),patch.object(main.reasoning_engine,'build_reasoning',return_value=why) as called:
            page=AppTest.from_file(str(config.BASE_DIR/'app.py')).run(timeout=60)
            page.session_state['main_navigation']='🧠 AI Predictor';page.run(timeout=60)
            count=called.call_count
            page.run(timeout=60)
            self.assertEqual(called.call_count,count)
            self.assertTrue(any('No verified near-term catalysts' in c.value for c in page.caption))
            page.button(key='refresh_prediction').click().run(timeout=60)
            self.assertEqual(called.call_count,count+1)
            self.assertFalse(page.exception)

    def test_bearish_neutral_cards_and_clamped_ring(self):
        for outlook,move,label in [('bearish',-.04,'▼ Bearish Outlook'),('neutral',0.,'◆ Neutral Outlook')]:
            def alternate(bundle,steps,current_price,ticker):
                p=forecast(bundle,steps,current_price,ticker)
                path=np.linspace(current_price,current_price*(1+move),steps+1)
                width=current_price*.01*np.sqrt(np.arange(steps+1))
                p.update(outlook=outlook,expected_change_pct=move*100,confidence=110.,median_target_price=float(path[-1]))
                p['corridor']=dict(median=path.tolist(),lower=(path-width).tolist(),upper=(path+width).tolist())
                return p
            with patch('requests.request',side_effect=self.request),patch.object(main.lstm_model,'forecast_multihorizon',side_effect=alternate):
                page=AppTest.from_file(str(config.BASE_DIR/'app.py')).run(timeout=60)
                page.session_state['main_navigation']='🧠 AI Predictor';page.run(timeout=60)
                self.assertFalse(page.exception)
                content=' '.join(m.value for m in page.markdown)
                self.assertIn(label,content);self.assertIn('99%',content)


if __name__=='__main__':
    import subprocess,sys
    files=list(config.BASE_DIR.rglob('*.py'))
    for file in files:subprocess.run([sys.executable,'-m','py_compile',str(file)],check=True,capture_output=True)
    print(f'COMPILE PASS: all {len(files)} Python files',flush=True)
    unittest.main(verbosity=2)

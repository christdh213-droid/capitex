"""Phase 2 verification. Real small LSTM training on a labelled synthetic fixture.

python test_phase2.py [--live] [--smoke]
Live mode additionally trains on current Yahoo AAPL daily data and fetches context.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import config
import data_loader
import indicators
import lstm_model as lm
import reasoning_engine as re


def fixture(n=620):
    rng=np.random.default_rng(81)
    dates=pd.bdate_range('2023-01-02',periods=n,tz='UTC')
    returns=.0004+.003*np.sin(np.arange(n)/7)+rng.normal(0,.009,n)
    close=100*np.exp(np.cumsum(returns))
    return pd.DataFrame(dict(Open=close,High=close*1.01,Low=close*.99,Close=close,
                             Volume=rng.integers(1000,2000,n)),index=dates)


class FeatureReasoningTests(unittest.TestCase):
    def test_wilder_rsi_known_reference_and_flat_series(self):
        prices=pd.Series([44.34,44.09,44.15,43.61,44.33,44.83,45.10,45.42,
                          45.84,46.08,45.89,46.03,45.61,46.28,46.28,46.00])
        self.assertAlmostEqual(indicators.compute_rsi(prices).iloc[14],70.464,places=2)
        flat=pd.Series(np.full(60,100.0))
        self.assertEqual(indicators.compute_rsi(flat).iloc[-1],50)
        self.assertTrue((indicators.compute_rsi(pd.Series(np.arange(60.0))).dropna()==100).all())

    def test_features_are_causal_and_dimensionless(self):
        frame=fixture()
        features=indicators.stationary_features(frame)
        revised=frame.copy(); revised.iloc[-1,revised.columns.get_loc('Close')]*=2
        pd.testing.assert_frame_equal(indicators.stationary_features(revised).iloc[:-1],features.iloc[:-1])
        scaled=frame.copy()
        for c in ('Open','High','Low','Close'): scaled[c]*=10
        np.testing.assert_allclose(indicators.stationary_features(scaled),features,atol=1e-12)
        self.assertEqual(list(features.columns),list(indicators.FEATURE_COLUMNS))

    def test_purged_labels_and_training_only_scalers(self):
        frame=fixture()
        ds=lm.prepare_multihorizon_dataset(frame,20)
        self.assertLess(ds.origins['train'][-1]+30,ds.calibration_split)
        self.assertGreaterEqual(ds.origins['calibration'][0]+1,ds.calibration_split)
        self.assertLess(ds.origins['calibration'][-1]+30,ds.split)
        self.assertGreaterEqual(ds.origins['test'][0]+1,ds.split)
        np.testing.assert_allclose(ds.x_scaler.mean_,ds.features.iloc[:ds.calibration_split].mean())
        revised=frame.copy()
        revised.loc[ds.features.index[ds.split]:,'Close']*=3
        changed=lm.prepare_multihorizon_dataset(revised,20)
        np.testing.assert_array_equal(ds.X['train'],changed.X['train'])
        np.testing.assert_array_equal(ds.y['train'],changed.y['train'])
        np.testing.assert_array_equal(ds.x_scaler.mean_,changed.x_scaler.mean_)
        self.assertEqual(lm._model_fingerprint(ds,'TEST',2,32,.001)[0],
                         lm._model_fingerprint(changed,'TEST',2,32,.001)[0])
        revised=frame.copy(); revised.iloc[40,revised.columns.get_loc('Close')]*=1.01
        changed=lm.prepare_multihorizon_dataset(revised,20)
        self.assertNotEqual(lm._model_fingerprint(ds,'TEST',2,32,.001)[0],
                            lm._model_fingerprint(changed,'TEST',2,32,.001)[0])

    def test_target_alignment(self):
        ds=lm.prepare_multihorizon_dataset(fixture(),20)
        origin=ds.origins['train'][0]
        target=lm._unscale_returns(ds,ds.y['train'][0])
        np.testing.assert_allclose(target,ds.raw_returns[origin+1:origin+31],rtol=1e-6,atol=1e-8)
        np.testing.assert_allclose(ds.X['train'][0],ds.scaled_features[origin-19:origin+1])

    def test_bad_model_inputs(self):
        with self.assertRaises(ValueError): lm.prepare_multihorizon_dataset(fixture(90))
        for value in (True,3,1.5):
            with self.assertRaises(ValueError): lm.prepare_multihorizon_dataset(fixture(),value)
        for value in (True,'2d',0,90):
            with self.assertRaises(ValueError): lm.parse_horizon(value)
        for value in ('24 Hours','Next 24 Hours','1d',1): self.assertEqual(lm.parse_horizon(value),1)
        frame=fixture(); frame.iloc[-1,frame.columns.get_loc('Close')]=np.nan
        with self.assertRaises(ValueError): indicators.stationary_features(frame)
        with self.assertRaises(ValueError): indicators.stationary_features(fixture().iloc[::-1])

    def test_indicator_tags_and_raw_values(self):
        summary=indicators.technical_summary(fixture())
        for key in ('SMA_20','SMA_50','EMA_12','EMA_26','RSI_14','MACD_Hist'):
            self.assertIsInstance(summary['values'][key],float)
        self.assertGreaterEqual(len(summary['tags']),4)
        self.assertTrue(all(tag['color'].startswith('#') for tag in summary['tags']))
        json.dumps(summary,allow_nan=False)

    def test_news_recency_dedup_and_negation(self):
        now=pd.Timestamp('2026-10-27 12:00',tz='UTC')
        title='Profits beat forecasts as growth surges'
        news=[{'content':{'title':title,'pubDate':now.isoformat()}},
              {'title':title,'providerPublishTime':int(now.timestamp())},
              {'title':'Outdated crash','providerPublishTime':int((now-pd.Timedelta(days=10)).timestamp())},
              {'title':'Future surge','providerPublishTime':int((now+pd.Timedelta(days=1)).timestamp())},
              {'title':'Undated headline'}]
        records=re._recent_headlines(news,now)
        self.assertEqual(len(records),1)
        self.assertGreater(records[0]['polarity'],0)
        self.assertLess(re.lexical_polarity('not strong growth'),0)
        self.assertGreater(re.lexical_polarity('no losses'),0)

    def test_reasoning_earnings_volume_fed_and_json(self):
        frame=fixture()
        frame.iloc[-1,frame.columns.get_loc('Volume')]=10000
        now=pd.Timestamp('2026-10-27 12:00',tz='UTC')
        calendar={'Earnings Date':[pd.Timestamp('2026-10-29')]}
        news=[{'content':{'title':'Company profits beat forecasts','pubDate':now.isoformat()}}]
        result=re.build_reasoning('AAPL',frame,news=news,calendar=calendar,as_of=now)
        self.assertTrue(0<=result['sentiment']['score']<=100)
        self.assertEqual(result['sentiment']['headlines_used'],1)
        tags={t['id']:t for t in result['catalyst_tags']}
        self.assertIn('earnings',tags); self.assertIn('volume_surge',tags); self.assertIn('fed_window',tags)
        self.assertEqual(tags['earnings']['days_away'],2)
        self.assertIn('federalreserve.gov',tags['fed_window']['source'])
        self.assertAlmostEqual(sum(c['weight'] for c in result['sentiment']['components']),1)
        json.dumps(result,allow_nan=False)

    def test_missing_context_is_explicit_and_weights_renormalize(self):
        result=re.build_reasoning('AAPL',fixture(),fetch_context=False,as_of='2026-10-27')
        self.assertEqual(result['sentiment']['headlines_used'],0)
        self.assertFalse(result['sentiment']['components'][0]['available'])
        self.assertEqual(result['source_status']['news'],'not_requested')
        self.assertAlmostEqual(sum(c['weight'] for c in result['sentiment']['components']),1)
        frame=fixture(); frame['Volume']=0
        result=re.build_reasoning('AAPL',frame,fetch_context=False,as_of='2026-10-27')
        self.assertIsNone(result['sentiment']['volume_ratio'])
        self.assertFalse(any(t['id']=='volume_surge' for t in result['catalyst_tags']))
        json.dumps(result,allow_nan=False)

    def test_context_outage_does_not_fabricate_news(self):
        class Provider:
            def get_news(self,**kwargs): raise ConnectionError('offline')
            def get_calendar(self): raise ConnectionError('offline')
        re.fetch_market_context.clear()
        with patch.object(re.yf,'Ticker',return_value=Provider()):
            data=re.fetch_market_context('AAPL')
        self.assertEqual(data['source_status']['news'],'unavailable')
        self.assertEqual(data['source_status']['calendar'],'unavailable')
        self.assertEqual(data['news'],[])
        re.fetch_market_context.clear()

    def test_52week_high_and_no_fake_macro_dates(self):
        frame=fixture()
        frame['Close']=np.linspace(100,250,len(frame)); frame['High']=frame.Close
        result=re.build_reasoning('BTC-USD',frame,news=[],calendar={'Earnings Date':['2026-10-29']},as_of='2026-10-27')
        self.assertTrue(any(t['id']=='high_test' for t in result['catalyst_tags']))
        self.assertFalse(any(t['id']=='earnings' for t in result['catalyst_tags']))
        tags,status=re._macro_tags(pd.Timestamp('2031-01-01',tz='UTC'))
        self.assertEqual(tags,[])
        self.assertEqual(status['status'],'outside_verified_schedule')

    def test_calendar_dataframe_and_past_events(self):
        cal=pd.DataFrame({'Value':[['2026-10-29','2026-10-30']]},index=['Earnings Date'])
        self.assertEqual(re._earnings_dates(cal),[pd.Timestamp('2026-10-29').date(),pd.Timestamp('2026-10-30').date()])
        result=re.build_reasoning('AAPL',fixture(),news=[],calendar={'Earnings Date':['2026-10-01']},as_of='2026-10-27')
        self.assertFalse(any(t['id']=='earnings' for t in result['catalyst_tags']))


class LSTMIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(dir=config.DATA_DIR)
        cls.frame=fixture()
        cls.bundle=lm.train_multihorizon_model(cls.frame,'PHASE2_TEST',window_size=20,
                                               epochs=2,batch_size=64,cache_dir=cls.temp.name)
    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_actual_training_and_three_corridors(self):
        bundle=self.bundle
        self.assertFalse(bundle.metadata['loaded_from_cache'])
        self.assertEqual(bundle.metadata['epochs_trained'],2)
        self.assertTrue(Path(bundle.metadata['cache_path']).is_file())
        output=[]
        for horizon in ('24 Hours','7 Days','30 Days'):
            result=lm.forecast_multihorizon(bundle,horizon,current_price=220,ticker='AAPL')
            n=lm.parse_horizon(horizon)
            corridor=result['corridor']
            self.assertEqual(len(corridor['median']),n+1)
            self.assertEqual(corridor['median'][0],220)
            self.assertEqual(corridor['lower'][0],220)
            self.assertEqual(corridor['upper'][0],220)
            for lower,median,upper in zip(corridor['lower'],corridor['median'],corridor['upper']):
                self.assertTrue(0<lower<=median<=upper)
            widths=np.log(np.array(corridor['upper'])/np.array(corridor['lower']))
            self.assertTrue((np.diff(widths)>=-1e-12).all())
            self.assertTrue(0<=result['confidence']<=99)
            self.assertEqual(result['mc_passes'],30)
            for field in ('rmse','mae','directional_accuracy','naive_rmse','test_interval_coverage_pct'):
                self.assertTrue(np.isfinite(result['metrics'][field]))
            self.assertFalse(result['confidence_is_probability'])
            json.dumps(result,allow_nan=False)
            output.append(result)
        report={'input':'deterministic synthetic test fixture, not market data',
                'training_epochs':2,'horizons':output}
        (config.DATA_DIR/'phase2_fixture_results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')

    def test_cache_reload_and_dropout_variability(self):
        cached=lm.train_multihorizon_model(self.frame,'PHASE2_TEST',window_size=20,
                                          epochs=2,batch_size=64,cache_dir=self.temp.name)
        self.assertTrue(cached.metadata['loaded_from_cache'])
        np.testing.assert_allclose(cached.dataset.x_scaler.mean_,self.bundle.dataset.x_scaler.mean_)
        window=cached.dataset.scaled_features[-20:][None,...]
        deterministic=[lm._predict_returns(cached.model,cached.dataset,window) for _ in range(2)]
        np.testing.assert_array_equal(*deterministic)
        passes=np.stack([lm._predict_returns(cached.model,cached.dataset,window,True) for _ in range(5)])
        self.assertGreater(float(np.max(np.std(passes,axis=0))),0)

    def test_invalid_anchor_and_horizon_rejected(self):
        for anchor in (0,-1,np.nan,np.inf):
            with self.assertRaises(ValueError): lm.forecast_multihorizon(self.bundle,current_price=anchor)
        with self.assertRaises(ValueError): lm.forecast_multihorizon(self.bundle,'60d')


def live_checks():
    frame=data_loader.load_market_data('AAPL','5y')
    bundle=lm.train_multihorizon_model(frame,'AAPL',window_size=30,epochs=3,batch_size=64)
    quote=data_loader.get_live_quote('AAPL')
    forecasts=[lm.forecast_multihorizon(bundle,h,current_price=quote['price'],ticker='AAPL') for h in ('24h','7d','30d')]
    reasoning=re.build_reasoning('AAPL',frame)
    for result in forecasts:
        assert 0<result['lower_target_price']<=result['median_target_price']<=result['upper_target_price']
        assert 0<=result['confidence']<=99
    report={'input':'Yahoo Finance daily AAPL history','training_epochs':3,
            'history_rows':len(frame),'quote':quote,'forecasts':forecasts,'reasoning':reasoning}
    (config.DATA_DIR/'phase2_live_results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('LIVE PASS: real AAPL LSTM training, 24h/7d/30d corridors, sentiment and catalysts')
    print('Context source status:',reasoning['source_status'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    options=parser.parse_args()
    files=sorted(config.BASE_DIR.rglob('*.py'))
    for path in files:
        subprocess.run([sys.executable,'-m','py_compile',str(path)],check=True,capture_output=True)
    print(f'COMPILE PASS: python -m py_compile on {len(files)} Python files',flush=True)
    import main
    import chart_plotter
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(FeatureReasoningTests),
                              unittest.defaultTestLoader.loadTestsFromTestCase(LSTMIntegrationTests)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful(): sys.exit(1)
    if options.live: live_checks()
    if options.smoke:
        from test_phase1 import smoke_checks
        smoke_checks('phase2')

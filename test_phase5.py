"""Phase 5 inspector, CSV, exports and launcher lifecycle regression checks."""
import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock,patch
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
import academic
import chart_plotter
import config
import database
import main
import run
import test_phase4 as fixtures
import test_phase3 as wallet_fixtures


def sample_history():
    values=200*np.exp(np.cumsum(np.random.default_rng(42).normal(.0003,.012,500)))
    return pd.DataFrame({'Open':values,'High':values*1.01,'Low':values*.99,'Close':values,
        'Volume':np.full(500,10000.)},index=pd.date_range('2023-01-01',periods=500,tz='UTC'))


class AcademicTests(unittest.TestCase):
    def setUp(self):
        fixtures.PredictorTests.setUp(self)
        self.history=sample_history()
        self.csv=self.history.rename_axis('Date').to_csv()
        self.market_patch=patch.object(main.data_loader,'load_market_data',return_value=self.history)
        self.market_patch.start()

    def tearDown(self):
        self.market_patch.stop()
        fixtures.PredictorTests.tearDown(self)

    request=fixtures.PredictorTests.request

    def test_technical_route_chart_and_raw_indicators(self):
        r=self.client.get('/api/inspect/AAPL')
        self.assertEqual(r.status_code,200)
        records=r.json()['history'];self.assertEqual(len(records),252)
        self.assertTrue({'SMA_20','SMA_50','EMA_12','EMA_26','RSI_14','MACD','MACD_Signal'}.issubset(records[-1]))
        fig=chart_plotter.plot_academic_chart(records,'AAPL')
        self.assertTrue(any(t.type=='candlestick' for t in fig.data))
        self.assertEqual(len([k for k in fig.layout if k in ('yaxis','yaxis2','yaxis3')]),3)
        self.assertTrue(any(t.name=='SMA 50' for t in fig.data))
        json.dumps(r.json(),allow_nan=False)

    def csv_forecast(self,bundle,horizon,**kwargs):
        p=fixtures.forecast(bundle,main.lstm_model.parse_horizon(horizon),float(self.history.Close.iloc[-1]),kwargs['ticker'])
        p['anchor_source']='last_historical_close'
        p['corridor']['timestamps']=[d.isoformat() for d in pd.date_range(self.history.index[-1],periods=len(p['corridor']['median']))]
        p['metrics']={'rmse':4.,'mae':3.,'directional_accuracy':56.,'naive_rmse':5.,'naive_mae':4.,'naive_directional_accuracy':0.,'beats_baseline':True,'test_samples':90}
        return p

    def test_csv_parser_forecast_and_separate_cache_identity(self):
        with patch.object(main.lstm_model,'train_multihorizon_model',return_value=object()) as train,patch.object(main.lstm_model,'forecast_multihorizon',side_effect=self.csv_forecast):
            for horizon in ('24h','7d','30d'):
                r=self.client.post('/api/inspect/csv',json={'csv_text':self.csv,'horizon':horizon})
                self.assertEqual(r.status_code,200,r.text)
                p=r.json();self.assertEqual(p['rows'],500)
                self.assertTrue(train.call_args.args[1].startswith('CSV_'))
                self.assertEqual(p['forecast']['anchor_source'],'last_historical_close')
                self.assertEqual(p['source'],'uploaded_csv_last_close')
                json.dumps(p,allow_nan=False)
            self.assertEqual(self.client.get('/api/wallet').json()['total_trades'],0)

    def test_invalid_csv_and_limits_do_not_train(self):
        with patch.object(main.lstm_model,'train_multihorizon_model') as train:
            for text in ('hello,world\n1,2','Date,Open,High,Low,Close,Volume\ninvalid,1,2,1,1,4'):
                self.assertEqual(self.client.post('/api/inspect/csv',json={'csv_text':text}).status_code,400)
            self.assertEqual(self.client.post('/api/inspect/csv',json={'csv_text':'x'*2_000_001}).status_code,422)
            self.assertEqual(self.client.post('/api/inspect/csv',json={'csv_text':self.csv,'horizon':'60d'}).status_code,422)
            self.assertEqual(train.call_count,0)

    def test_complete_ledger_and_prediction_csv_roundtrip(self):
        with database.get_connection() as conn:
            for i in range(205):
                conn.execute("INSERT INTO trade_ledger(timestamp,ticker,action,quantity,price,total_value,realized_pnl,execution_source,execution_label) VALUES(?,?,?,?,?,?,?,?,?)",('2026-10-30T10:20:30.123456+00:00','BTC-USD','BUY',.01,63000.,630.,0.,'live_quote','Live quote'))
        rows=self.client.get('/api/wallet/ledger').json()['rows']
        self.assertEqual(len(rows),205)
        ledger=pd.read_csv(io.BytesIO(academic.ledger_csv(rows)))
        self.assertEqual(len(ledger),205)
        self.assertEqual(ledger.timestamp_ist.iloc[0],'2026-10-30 15:50:30 IST')
        p=self.client.get('/api/predict/AAPL?horizon=7d').json()
        csv=pd.read_csv(io.BytesIO(academic.predictions_csv(p)))
        self.assertEqual(len(csv),8)
        self.assertEqual(csv.median_price.iloc[-1],round(p['median_target_price'],8))

    def test_collapsed_inspector_load_chart_and_downloads(self):
        with patch('requests.request',side_effect=self.request):
            page=AppTest.from_file(str(config.BASE_DIR/'app.py')).run(timeout=60)
            page.session_state['main_navigation']='🧠 AI Predictor';page.run(timeout=60)
            self.assertFalse(page.exception,[e.message for e in page.exception])
            inspector=next(e for e in page.expander if e.label.startswith('🎓'))
            self.assertFalse(inspector.proto.expanded)
            self.assertEqual(len(page.get('plotly_chart')),1)
            self.assertEqual(len(page.get('download_button')),2)
            page.button(key='load_technical').click().run(timeout=60)
            self.assertFalse(page.exception,[e.message for e in page.exception])
            self.assertEqual(len(page.get('plotly_chart')),2)

    def test_csv_uploader_to_result_and_export(self):
        upload=io.BytesIO(self.csv.encode())
        with patch('requests.request',side_effect=self.request),patch('streamlit.file_uploader',return_value=upload),patch.object(main.lstm_model,'train_multihorizon_model',return_value=object()),patch.object(main.lstm_model,'forecast_multihorizon',side_effect=self.csv_forecast):
            page=AppTest.from_file(str(config.BASE_DIR/'app.py')).run(timeout=60)
            page.session_state['main_navigation']='🧠 AI Predictor';page.run(timeout=60)
            page.button(key='analyze_csv').click().run(timeout=60)
            self.assertFalse(page.exception,[e.message for e in page.exception])
            self.assertEqual(page.session_state['csv_inspection']['rows'],500)
            self.assertTrue(any('CSV simulation' in i.value for i in page.info))
            self.assertEqual(len(page.get('download_button')),3)

    def test_legacy_routes_remain_available_with_isolated_wallets(self):
        next_day={'current_close':200.,'predicted_next_close':202.,'predicted_return_pct':1.,'signal':'BUY','reason':'test signal'}
        with patch.object(main.data_loader,'load_from_yfinance',return_value=(self.history,{})),patch.object(main.lstm_model,'train_or_load_lstm',return_value=(object(),True,{})),patch.object(main.lstm_model,'predict_next_day',return_value=next_day),patch.object(main.backtester,'run_single_ticker_backtest',return_value={'ticker':'AAPL'}),patch.object(main.backtester,'run_bulk_backtest',return_value=([{'ticker':'AAPL'}],[])):
            for path in ('/api/health','/api/price/AAPL','/portfolio/human','/portfolio/ai','/openapi.json','/docs'):
                self.assertEqual(self.client.get(path).status_code,200,path)
            for path,body in [('/api/predict',{'ticker':'AAPL','epochs':5}),('/trade/execute',{'wallet_type':'human','ticker':'BTC-USD','action':'BUY','quantity':.01}),('/trade/ai-step',{'ticker':'AAPL'}),('/trade/check-risk/human',{}),('/trade/check-risk/ai',{}),('/api/backtest/ticker',{'ticker':'AAPL'}),('/api/backtest/run',{'tickers':['AAPL']}),('/portfolio/reset',{})]:
                r=self.client.post(path,json=body);self.assertEqual(r.status_code,200,(path,r.text))


class LauncherTests(unittest.TestCase):
    def test_owned_services_cleanup_on_ctrl_c(self):
        backend=Mock(pid=123);frontend=Mock(pid=456)
        backend.poll.return_value=None;frontend.poll.return_value=None
        with tempfile.TemporaryDirectory(dir=config.DATA_DIR) as directory,patch.object(run,'PROJECT_DIR',Path(directory)),patch.object(run,'is_backend_ready',return_value=False),patch.object(run,'port_available',return_value=True),patch.object(run,'wait_ready'),patch.object(run.subprocess,'Popen',side_effect=[backend,frontend]),patch.object(run.signal,'signal'),patch.object(run.time,'sleep',side_effect=KeyboardInterrupt):
            self.assertEqual(run.main(),0)
            backend.terminate.assert_called_once();frontend.terminate.assert_called_once()

    def test_reused_services_are_not_terminated(self):
        with tempfile.TemporaryDirectory(dir=config.DATA_DIR) as directory,patch.object(run,'PROJECT_DIR',Path(directory)),patch.object(run,'is_backend_ready',return_value=True),patch.object(run.subprocess,'Popen') as launch,patch.object(run.signal,'signal'),patch.object(run.time,'sleep',side_effect=KeyboardInterrupt):
            self.assertEqual(run.main(),0);launch.assert_not_called()

    def test_start_failure_cleans_backend_and_returns_failure(self):
        backend=Mock(pid=123);backend.poll.return_value=None
        with patch.object(run,'is_backend_ready',return_value=False),patch.object(run,'port_available',side_effect=[True,False]),patch.object(run,'wait_ready'),patch.object(run.subprocess,'Popen',return_value=backend),patch.object(run.signal,'signal'):
            self.assertEqual(run.main(),1);backend.terminate.assert_called_once()


if __name__=='__main__':
    import sys
    sources=list(config.BASE_DIR.glob('*.py'))
    for source in sources:subprocess.run([sys.executable,'-m','py_compile',str(source)],check=True,capture_output=True)
    print(f'COMPILE PASS: all {len(sources)} Python sources',flush=True)
    unittest.main(verbosity=2)

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
    with sqlite3.connect(copy) as target:target.execute('ALTER TABLE trade_ledger DROP COLUMN thesis_snapshot_json')
    with sqlite3.connect(copy) as target:target.execute('ALTER TABLE context_summary_cache DROP COLUMN technical_summary_text')
    database.init_db(copy);database.init_db(copy)
    with database.get_connection(copy) as c:
        assert c.execute('SELECT cash FROM wallet').fetchone()[0]==12345
        assert c.execute('SELECT quantity FROM holdings').fetchone()[0]==2
        assert c.execute('SELECT locked_collateral FROM short_positions').fetchone()[0]==100
        assert c.execute('SELECT timestamp FROM trade_ledger').fetchone()[0]=='2026-01-01T00:00:00Z'
        assert c.execute('PRAGMA journal_mode').fetchone()[0]=='wal'
        assert c.execute('PRAGMA busy_timeout').fetchone()[0]==5000
        assert 'thesis_snapshot_json' in {r[1] for r in c.execute('PRAGMA table_info(trade_ledger)')}

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

def news_fixture(symbol='BTC-USD',number=1):
    name='Bitcoin' if symbol=='BTC-USD' else 'Solana' if symbol=='SOL-USD' else 'Apple'
    words=['earnings rise 5%','guidance review','product launch','regulation change','rates pressure']
    return [{'content':{'title':name+' '+words[i%len(words)],'summary':'Provider snippet only.',
                       'pubDate':'2026-11-23T09:00:00Z','provider':{'displayName':'Fixture publisher'},
                       'canonicalUrl':{'url':f'https://example.test/story/{i}'},'relatedTickers':[symbol]}} for i in range(number)]

def test_news_payload_shapes_relevance_and_dedup(db):
    import market_context as mc
    now=pd.Timestamp('2026-11-23T12:00Z')
    nested=news_fixture()
    legacy=[{'title':'Bitcoin earnings rise 5%','link':'https://example.test/story/0?utm_source=copy',
             'publisher':'Fixture','providerPublishTime':int(pd.Timestamp('2026-11-23T09:00Z').timestamp())}]
    unrelated=[{'title':'Solar panels launch','url':'https://example.test/no','datetime':int(now.timestamp())}]
    rows=mc.get_articles('BTC-USD',now,[('nested',lambda *_:nested),('legacy',lambda *_:legacy)])
    assert len(rows)==1
    assert rows[0]['publisher']=='Fixture' or rows[0]['publisher']=='Fixture publisher'
    assert len(mc.parse_articles('SOL-USD',unrelated,'fake',now))==0
    assert len(mc.parse_articles('SOL-USD',news_fixture('SOL-USD'),'fake',now))==1
    assert len(mc.parse_articles('AAPL',news_fixture('AAPL'),'fake',now))==1

@pytest.mark.parametrize('count',[0,1,2,4])
def test_news_ttl_and_summary_retention(db,count):
    import market_context as mc
    now=pd.Timestamp('2026-11-23T12:00Z');calls=[]
    def fetch(*args):calls.append(1);return news_fixture(number=count)
    providers=[('fixture',fetch)]
    a=mc.get_context('BTC-USD',now,providers,llm_call=lambda _:None)
    b=mc.get_context('BTC-USD',now+pd.Timedelta(minutes=10),providers,llm_call=lambda _:None)
    assert len(calls)==1 and a['context_summary_id']==b['context_summary_id']
    assert a['headline_count']==count and len(a['bullets'])==count
    c=mc.get_context('BTC-USD',now+pd.Timedelta(minutes=31),providers,llm_call=lambda _:None)
    assert len(calls)==2 and c['context_summary_id']!=a['context_summary_id']
    with database.get_connection() as conn:
        old=conn.execute('SELECT * FROM context_summary_cache WHERE id=?',(a['context_summary_id'],)).fetchone()
        assert old['is_stale']==1 and json.loads(old['bullets_json'])==a['bullets']

@pytest.mark.parametrize('mode',['bad_id','number','fabricated_fact','timeout','extra_url'])
def test_llm_grounding_failures_fall_back(db,mode):
    import market_context as mc
    def llm(articles):
        if mode=='timeout':raise TimeoutError('simulated')
        row=articles[0];event=row['title']
        b={'article_id':row['id'],'verified_event':event,'possible_implication':'An interpretation.'}
        if mode=='bad_id':b['article_id']=99999
        if mode=='number':b['verified_event']=event+' 99%'
        if mode=='fabricated_fact':b['verified_event']='The CEO was fired'
        if mode=='extra_url':b['url']='https://attacker.test'
        return {'bullets':[b],'counter_thesis':[]}
    result=mc.get_context('BTC-USD',pd.Timestamp('2026-11-23T12:00Z'),[('fixture',lambda *_:news_fixture())],llm)
    assert result['model_used']=='deterministic'
    assert result['bullets'][0]['verified_event']=='Bitcoin earnings rise 5%'
    assert result['bullets'][0]['url']=='https://example.test/story/0'

def test_valid_llm_hydrates_only_db_fields(db):
    import market_context as mc
    def llm(articles):return {'bullets':[{'article_id':articles[0]['id'],'verified_event':articles[0]['title'],
                             'possible_implication':'Future demand remains uncertain.'}],'counter_thesis':[]}
    result=mc.get_context('BTC-USD',pd.Timestamp('2026-11-23T12:00Z'),[('fixture',lambda *_:news_fixture())],llm,
                          signals=[{'label':'RSI: Neutral (50)','tone':'neutral'}])
    assert result['model_used']!='deterministic'
    assert result['bullets'][0]['publisher']=='Fixture publisher'
    assert result['counter_thesis']['signals'][0]['label']=='RSI: Neutral (50)'

def test_context_show_more_ui():
    from test_phase3 import WalletTests
    from test_phase4 import forecast,WHY
    from streamlit.testing.v1 import AppTest
    from urllib.parse import urlsplit
    import main,copy
    env=WalletTests();env.setUp()
    context={'context_summary_id':777,'headline_count':4,'bullets':[
        {'article_id':i,'verified_event':f'Bitcoin reported event {i}','possible_implication':'',
         'publisher':'Fixture','published_at':'2026-11-23T09:00:00Z','url':f'https://example.test/{i}'} for i in range(4)],'counter_thesis':{}}
    history=pd.DataFrame({'Close':[100.]*90},index=pd.date_range('2026-06-01',periods=90,tz='UTC'))
    def request(method,url,json=None,**kwargs):
        parts=urlsplit(url);return env.client.request(method,parts.path+('?' + parts.query if parts.query else ''),json=json)
    try:
        with patch('requests.request',side_effect=request),patch.object(main,'get_capitex_forecast_bundle',return_value=(object(),history)),\
             patch.object(main.lstm_model,'forecast_multihorizon',side_effect=forecast),\
             patch.object(main.reasoning_engine,'build_reasoning',return_value=copy.deepcopy(WHY)),\
             patch.object(main.market_context,'get_articles',return_value=[]),\
             patch.object(main.market_context,'get_context',return_value=context):
            page=AppTest.from_file(str(config.BASE_DIR/'app.py')).run(timeout=60)
            page.session_state['main_navigation']='🧠 AI Predictor';page.run(timeout=60)
            assert not page.exception
            assert len(page.text)==3
            raw=next(item for item in page.expander if item.label=='Show raw technical indicators')
            assert not raw.proto.expanded
            assert any('broader price trend' in item.value for item in page.markdown)
            page.toggle(key='context_show_more_777').set_value(True).run(timeout=60)
            assert len(page.text)==4 and not page.exception
    finally:env.tearDown()

def test_analyzer_endpoint_deduplicates_ids():
    from test_phase3 import WalletTests
    from test_phase4 import forecast,WHY
    import main,copy
    env=WalletTests();env.setUp()
    history=pd.DataFrame({'Close':[100.]},index=pd.to_datetime(['2026-11-25'],utc=True))
    try:
        with patch.object(main,'get_capitex_forecast_bundle',return_value=(object(),history)),\
             patch.object(main.lstm_model,'forecast_multihorizon',side_effect=forecast),\
             patch.object(main.reasoning_engine,'build_reasoning',return_value=copy.deepcopy(WHY)),\
             patch.object(main.market_context,'get_articles',return_value=[]),\
             patch.object(main.forecast_log,'now_utc',return_value=pd.Timestamp('2026-11-26T12:00Z')):
            # Endpoint history is filtered again using its own injected clock.
            with patch.object(main,'completed_history',side_effect=lambda t,h:h):
                first=env.client.get('/api/predict/AAPL?horizon=7d')
                second=env.client.get('/api/predict/AAPL?horizon=7d')
            assert first.status_code==200,first.text
            assert first.json()['forecast_log_ids']==second.json()['forecast_log_ids']
            assert first.json()['corridor']==second.json()['corridor']
            with database.get_connection() as conn:
                rows=conn.execute('SELECT * FROM forecast_logs').fetchall()
                assert len(rows)==3
                assert next(r for r in rows if r['horizon']==1)['target_timestamp']=='2026-11-27T18:00:00Z'
    finally:env.tearDown()

def test_grader_loop_does_not_block_event_loop(monkeypatch):
    import asyncio,threading,forecast_grader as grader
    started=threading.Event();release=threading.Event()
    def blocking():
        started.set();assert release.wait(5)
    monkeypatch.setattr(grader,'grade_pending',blocking)
    async def check():
        task=asyncio.create_task(grader.grader_loop())
        try:
            for _ in range(100):
                if started.is_set():break
                await asyncio.sleep(.005)
            assert started.is_set()
            assert await asyncio.wait_for(asyncio.sleep(.01,result='responsive'),.5)=='responsive'
        finally:
            release.set();task.cancel()
            try:await task
            except asyncio.CancelledError:pass
    asyncio.run(check())

def add_log(symbol='AAPL',target='2026-11-27T18:00:00Z',origin='2026-11-25T21:00:00Z',horizon=1,**extra):
    from market_calendar import asset_type
    row=dict(symbol=symbol,asset_type=asset_type(symbol),model_version='fixture',issued_at=origin,data_timestamp=origin,
        horizon=horizon,starting_price=100.,target_timestamp=target,predicted_price=110.,band_low=90.,band_high=120.,
        persistence_predicted_price=100.,status='PENDING',payload_json='{}')
    row.update(extra)
    with database.get_connection() as conn:
        return conn.execute('INSERT INTO forecast_logs('+','.join(row)+') VALUES('+','.join('?' for _ in row)+')',list(row.values())).lastrowid

def logged(rowid):
    with database.get_connection() as conn:return dict(conn.execute('SELECT * FROM forecast_logs WHERE id=?',(rowid,)).fetchone())

def test_grader_friday_close_grouping_idempotency_and_no_transaction(db):
    from forecast_grader import grade_pending
    friday=add_log();monday=add_log(target='2026-11-30T21:00:00Z',horizon=7)
    calls=[]
    def provider(*args):
        calls.append(args)
        # Acquiring a write lock here proves no grader connection holds a transaction.
        with database.get_connection() as conn:conn.execute('BEGIN IMMEDIATE');conn.execute('UPDATE wallet SET cash=cash')
        return pd.DataFrame({'Close':[99.,75.],'Stock Splits':[0.,0.]},index=pd.to_datetime(['2026-11-27','2026-11-30'],utc=True))
    assert grade_pending(pd.Timestamp('2026-11-30T22:00Z'),provider)['graded']==2
    assert len(calls)==1
    assert logged(friday)['actual_price']==99 and logged(friday)['actual_timestamp']=='2026-11-27T18:00:00Z'
    assert logged(monday)['actual_price']==75
    assert grade_pending(pd.Timestamp('2026-11-30T23:00Z'),provider)['graded']==0 and len(calls)==1

def test_grader_mid_session_and_buffer_never_grade_partial(db):
    from forecast_grader import grade_pending
    rowid=add_log()
    def forbidden(*args):raise AssertionError('Too early to fetch')
    assert grade_pending(pd.Timestamp('2026-11-27T17:00Z'),forbidden)['graded']==0
    assert grade_pending(pd.Timestamp('2026-11-27T18:29Z'),forbidden)['graded']==0
    frame=pd.DataFrame({'Close':[105.]},index=pd.to_datetime(['2026-11-27T15:00:00Z']))
    assert grade_pending(pd.Timestamp('2026-11-27T18:31Z'),lambda *_:frame)['graded']==0
    assert logged(rowid)['status']=='PENDING'

def test_crypto_day_end_and_grace(db):
    from forecast_grader import grade_pending
    rowid=add_log('BTC-USD','2026-11-23T23:59:59Z','2026-11-22T23:59:59Z')
    calls=[]
    def provider(*args):calls.append(1);return pd.DataFrame({'Close':[101.]},index=pd.to_datetime(['2026-11-23'],utc=True))
    grade_pending(pd.Timestamp('2026-11-23T23:59:59Z'),provider)
    assert not calls and logged(rowid)['status']=='PENDING'
    grade_pending(pd.Timestamp('2026-11-24T00:00:01Z'),provider)
    assert logged(rowid)['actual_timestamp']=='2026-11-23T23:59:59Z'
    missing=add_log('SOL-USD','2026-11-23T23:59:59Z','2026-11-22T23:59:59Z')
    grade_pending(pd.Timestamp('2026-11-25T23:59:58Z'),lambda *_:pd.DataFrame())
    assert logged(missing)['status']=='PENDING'
    grade_pending(pd.Timestamp('2026-11-26T00:00:00Z'),lambda *_:pd.DataFrame())
    assert logged(missing)['status']=='UNAVAILABLE'

def test_split_and_stock_missing_grace(db):
    from forecast_grader import grade_pending
    split=add_log(origin='2026-11-24T21:00:00Z')
    frame=pd.DataFrame({'Close':[50.,55.],'Stock Splits':[2.,0.]},index=pd.to_datetime(['2026-11-25','2026-11-27'],utc=True))
    grade_pending(pd.Timestamp('2026-11-27T19:00Z'),lambda *_:frame)
    assert logged(split)['unavailable_reason']=='stock_split_in_window'
    missing=add_log('NVDA')
    grade_pending(pd.Timestamp('2026-12-01T18:00Z'),lambda *_:pd.DataFrame())
    assert logged(missing)['status']=='PENDING'
    grade_pending(pd.Timestamp('2026-12-03T18:00Z'),lambda *_:pd.DataFrame())
    assert logged(missing)['status']=='UNAVAILABLE'

def test_report_scale_ties_zero_and_identical_baseline_set(db):
    from report_card import get_report
    add_log(status='GRADED',actual_price=110.)
    add_log(horizon=7,status='GRADED',actual_price=100.)
    add_log('BTC-USD',status='GRADED',actual_price=0.)
    pooled=get_report()
    assert 'mae' not in pooled['lstm'] and 'rmse' not in pooled['naive_persistence']
    assert pooled['graded']==3 and pooled['directional_n']==2 and pooled['low_sample']
    assert pooled['lstm']['mape_n']==2
    one=get_report('AAPL')
    assert one['lstm']['mae']==5 and one['naive_persistence']['mae']==5
    assert one['directional_n']==1 and one['directional_accuracy_pct']==100
    assert one['interval_coverage_pct']==100

def execution(price):
    return dict(price=price,execution_source='live_quote',execution_label='Injected live quote',
                quote_timestamp='2026-11-23T12:00:00Z',is_fallback=False)

@pytest.mark.parametrize('action',['BUY','SHORT_OPEN'])
def test_thesis_atomic_validation_and_immutable_snapshot(db,monkeypatch,action):
    from fastapi.testclient import TestClient
    import main,thesis,trading_engine as engine,market_context as mc
    monkeypatch.setattr(engine,'get_execution_price',lambda _:execution(100))
    monkeypatch.setattr(engine,'_market_assets',lambda _: {})
    rowid=add_log('BTC-USD');foreign=add_log('AAPL')
    context=mc.get_context('BTC-USD',pd.Timestamp('2026-11-23T12:00Z'),
        [('fixture',lambda *_:news_fixture())],llm_call=lambda _:None)
    other=mc.get_context('AAPL',pd.Timestamp('2026-11-23T12:00Z'),
        [('fixture',lambda *_:news_fixture('AAPL'))],llm_call=lambda _:None)
    real_build=thesis.build
    def checked_build(conn,*args):
        assert conn.in_transaction
        return real_build(conn,*args)
    monkeypatch.setattr(thesis,'build',checked_build)
    with TestClient(main.app) as client:
        base=dict(ticker='BTC-USD',action=action,quantity=1,forecast_log_ids={'1':rowid},
            selected_horizon=1,context_summary_id=context['context_summary_id'],prepared_at='2026-11-23T12:00:00+00:00')
        for bad in ({'forecast_log_ids':[foreign]},{'forecast_log_ids':[999999]},
                    {'context_summary_id':99999},{'context_summary_id':other['context_summary_id']},
                    {'selected_horizon':30},{'forecast_log_ids':{'7':rowid}}):
            response=client.post('/api/trade',json={**base,**bad})
            assert response.status_code==400,response.text
        assert database.get_capitex_wallet()['cash']==100000
        response=client.post('/api/trade',json={**base,'client_order_id':'thesis-valid-once'})
        assert response.status_code==200,response.text
        trade=response.json()['trade'];saved=trade['thesis_snapshot_json'];snapshot=json.loads(saved)
        assert snapshot['bullets']==context['bullets'] and snapshot['forecasts'][0]['id']==rowid
        assert snapshot['technical_summary_text']==context['technical_summary_text']
        assert snapshot['prepared_at']=='2026-11-23T12:00:00Z' and snapshot['executed_at']==trade['timestamp']
        with database.get_connection() as conn:
            conn.execute("UPDATE context_summary_cache SET bullets_json='[]',is_stale=1")
            conn.execute("UPDATE context_summary_cache SET technical_summary_text='Changed later'")
            conn.execute('UPDATE forecast_logs SET predicted_price=999 WHERE id=?',(rowid,))
        replay=client.post('/api/trade',json={**base,'client_order_id':'thesis-valid-once'}).json()['trade']
        assert replay['replayed'] and replay['thesis_snapshot_json']==saved
        changed=client.post('/api/trade',json={**base,'client_order_id':'thesis-valid-once','prepared_at':'2026-11-23T12:01:00Z'})
        assert changed.status_code==400
        with database.get_connection() as conn:
            assert conn.execute('SELECT thesis_snapshot_json FROM trade_ledger WHERE id=?',(trade['id'],)).fetchone()[0]==saved

@pytest.mark.parametrize('move',[-30,0,50,100,120])
def test_short_preview_matches_actual_to_cent(db,monkeypatch,move):
    from fastapi.testclient import TestClient
    from decimal import Decimal
    import main,trading_engine as engine
    quote=[100.037];qty=3.33333333
    monkeypatch.setattr(engine,'get_execution_price',lambda _:execution(quote[0]))
    monkeypatch.setattr(engine,'_market_assets',lambda _: {})
    with TestClient(main.app) as client:
        before=database.get_capitex_wallet()['cash']
        response=client.post('/api/short-preview',json=dict(symbol='BTC-USD',quantity=qty,entry_price=quote[0],pct_move=move))
        assert response.status_code==200,response.text
        preview=response.json()
        assert database.get_capitex_wallet()['cash']==before # preview has no trading side effects
        opened=client.post('/api/short/open',json=dict(symbol='BTC-USD',quantity=qty))
        assert opened.status_code==200,opened.text
        with database.get_connection() as conn:
            assert conn.execute('SELECT locked_collateral FROM short_positions').fetchone()[0]==preview['required_collateral']
        quote[0]=preview['simulated_price']
        result=client.post('/api/short/cover',json=dict(symbol='BTC-USD',quantity=qty)).json()['trade']
        assert Decimal(str(result['realized_pnl']))==Decimal(str(preview['projected_unrealized_pnl']))
        assert database.get_capitex_wallet()['cash']==preview['projected_total_equity']
        assert preview['would_liquidate']==(move>=100)
        assert (result['action']=='SHORT_LIQUIDATED')==(move>=100)

def test_decimal_partial_cover_and_refresh_liquidation(db,monkeypatch):
    import trading_engine as engine
    from short_accounting import calculate
    monkeypatch.setattr(engine,'get_execution_price',lambda _:execution(10.015))
    monkeypatch.setattr(engine,'_market_assets',lambda _: {})
    engine.execute_capitex_short('BTC-USD','SHORT_OPEN',quantity=3)
    with database.get_connection() as conn:row=dict(conn.execute('SELECT * FROM short_positions').fetchone())
    expected=calculate('SHORT_COVER',1,9.117,row)
    monkeypatch.setattr(engine,'get_execution_price',lambda _:execution(9.117))
    actual=engine.execute_capitex_short('BTC-USD','SHORT_COVER',quantity=1)
    assert actual['realized_pnl']==float(expected['realized_pnl'])
    with database.get_connection() as conn:
        remaining=dict(conn.execute('SELECT * FROM short_positions').fetchone())
        assert remaining['locked_collateral']==float(expected['locked_collateral'])
        cash=conn.execute('SELECT cash FROM wallet').fetchone()[0]
        engine._liquidate_shorts(conn,{'BTC-USD':25})
        assert not conn.execute('SELECT 1 FROM short_positions').fetchone()
        assert conn.execute('SELECT cash FROM wallet').fetchone()[0]==cash
        assert conn.execute('SELECT realized_pnl FROM trade_ledger ORDER BY id DESC').fetchone()[0]==-remaining['locked_collateral']

def test_prepare_short_order_ui_flow():
    from streamlit.testing.v1 import AppTest
    from test_phase3 import WalletTests
    from test_phase4 import forecast,WHY
    from urllib.parse import urlsplit
    import main,copy
    env=WalletTests();env.setUp()
    history=pd.DataFrame({'Close':[100.]*90},index=pd.date_range('2026-06-01',periods=90,tz='UTC'))
    def request(method,url,json=None,**kwargs):
        parts=urlsplit(url);return env.client.request(method,parts.path+('?' + parts.query if parts.query else ''),json=json)
    try:
        with patch('requests.request',side_effect=request),patch.object(main,'get_capitex_forecast_bundle',return_value=(object(),history)),\
             patch.object(main.lstm_model,'forecast_multihorizon',side_effect=forecast),\
             patch.object(main.reasoning_engine,'build_reasoning',return_value=copy.deepcopy(WHY)),\
             patch.object(main.market_context,'get_articles',return_value=[]):
            page=AppTest.from_file(str(config.BASE_DIR/'app.py')).run(timeout=60)
            page.session_state['main_navigation']='🧠 AI Predictor';page.run(timeout=60)
            page.button(key='prepare_short_order').click().run(timeout=60)
            assert not page.exception
            assert page.session_state['main_navigation']=='👛 Wallet'
            assert page.session_state['order_action']=='SHORT_OPEN'
            linked=page.session_state['pending_thesis']
            assert linked['symbol']=='AAPL' and linked['trade_side']=='SHORT_OPEN'
            assert linked['selected_horizon']==7 and set(linked['forecast_log_ids'])=={'1','7','30'}
            assert linked['context_summary_id'] is not None
            assert page.slider(key='short_preview_move').value==0
            page.number_input(key='order_size').set_value(100.).run(timeout=60)
            page.slider(key='short_preview_move').set_value(100).run(timeout=60)
            assert not page.exception
            assert any('reaches liquidation' in item.value for item in page.warning)
            page.button(key='execute_order').click().run(timeout=60)
            assert not page.exception
            with database.get_connection() as conn:
                row=conn.execute("SELECT * FROM trade_ledger WHERE action='SHORT_OPEN'").fetchone()
                assert json.loads(row['thesis_snapshot_json'])['selected_horizon']==7
    finally:env.tearDown()

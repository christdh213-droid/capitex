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
            page.toggle(key='context_show_more_777').set_value(True).run(timeout=60)
            assert len(page.text)==4 and not page.exception
    finally:env.tearDown()

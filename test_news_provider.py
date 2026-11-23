"""Yahoo Search/RSS fixtures only; no live network or credentials."""
from io import BytesIO
from types import SimpleNamespace
from urllib.parse import urlsplit,parse_qs
import pandas as pd
import pytest
import database,market_context as mc,news_provider as provider
from test_upgrade import db

NOW=pd.Timestamp('2026-11-23T12:00:00Z')

@pytest.fixture
def rss_xml():
    return b'''<?xml version="1.0"?><rss version="2.0"><channel>
      <item><title>Bitcoin product launch</title><link>https://example.test/btc?utm_source=rss</link>
      <description><![CDATA[<p>Bitcoin network update &amp; product details.</p>]]></description>
      <pubDate>Mon, 23 Nov 2026 09:00:00 GMT</pubDate><source>Fixture Wire</source></item>
      <item><title>Bitcoin product launch</title><link>https://example.test/btc</link>
      <description><![CDATA[<p>Bitcoin network update &amp; product details.</p>]]></description>
      <pubDate>Mon, 23 Nov 2026 09:00:00 GMT</pubDate><source>Fixture Wire</source></item>
      <item><title>Unrelated solar launch</title><link>https://example.test/solar</link>
      <pubDate>Mon, 23 Nov 2026 09:00:00 GMT</pubDate></item>
      <item><title>Bitcoin old event</title><link>https://example.test/old</link>
      <pubDate>Sun, 01 Nov 2026 09:00:00 GMT</pubDate></item>
      <item><title>Bitcoin future event</title><link>https://example.test/future</link>
      <pubDate>Tue, 24 Nov 2026 09:00:00 GMT</pubDate></item>
      <item><title>Bitcoin malformed date</title><link>https://example.test/bad</link><pubDate>bad</pubDate></item>
    </channel></rss>'''

@pytest.mark.parametrize('symbol,queries,title',[('AAPL',['AAPL','Apple'],'Apple earnings outlook'),
                                              ('BTC-USD',['BTC-USD','Bitcoin'],'Bitcoin product launch')])
def test_search_zero_key_and_cached_filtering(db,monkeypatch,symbol,queries,title):
    calls=[]
    monkeypatch.delenv('FINNHUB_API_KEY',raising=False)
    def search(query,**kwargs):
        calls.append(query)
        assert kwargs==dict(max_results=0,news_count=10,timeout=5)
        return SimpleNamespace(news=[dict(title=title,link='https://example.test/story',publisher='Fixture',
            providerPublishTime=int((NOW-pd.Timedelta(hours=1)).timestamp()),relatedTickers=[symbol])])
    monkeypatch.setattr(provider.yf,'Search',search)
    rss_queries=[];google_queries=[]
    monkeypatch.setattr(provider,'rss_news',lambda query:rss_queries.append(query) or [])
    monkeypatch.setattr(provider,'google_news',lambda *args:google_queries.append(args) or [])
    first=mc.get_articles(symbol,NOW)
    second=mc.get_articles(symbol,NOW+pd.Timedelta(minutes=14))
    assert sorted(calls)==sorted(queries) and len(first)==1 and first==second
    assert sorted(rss_queries)==sorted(queries) and google_queries==[(symbol,queries[-1])]
    with database.get_connection() as conn:
        assert conn.execute('SELECT COUNT(*) FROM news_articles_cache').fetchone()[0]==1

@pytest.mark.parametrize('mode',['empty','error'])
def test_rss_fallback_recency_relevance_dedup_and_ttl(db,monkeypatch,rss_xml,mode):
    search_calls=[];rss_calls=[]
    monkeypatch.delenv('FINNHUB_API_KEY',raising=False)
    def search(query,**kwargs):
        search_calls.append(query)
        if mode=='error':raise ConnectionError('fixture Search outage')
        return SimpleNamespace(news=[])
    def urlopen(req,timeout):
        rss_calls.append(req.full_url)
        assert timeout==5
        parts=urlsplit(req.full_url);params=parse_qs(parts.query)
        if parts.hostname=='feeds.finance.yahoo.com':
            assert params['s'][0] in ('BTC-USD','Bitcoin')
            assert params['region']==['US'] and params['lang']==['en-US']
        else:
            assert parts.hostname=='news.google.com'
            assert params['q']==['"Bitcoin" OR "BTC-USD" when:7d']
            assert params['hl']==['en-US'] and params['gl']==['US'] and params['ceid']==['US:en']
        return BytesIO(rss_xml)
    monkeypatch.setattr(provider.yf,'Search',search)
    monkeypatch.setattr(provider.request,'urlopen',urlopen)
    rows=mc.get_articles('BTC-USD',NOW)
    again=mc.get_articles('BTC-USD',NOW+pd.Timedelta(minutes=1))
    assert sorted(search_calls)==['BTC-USD','Bitcoin'] and len(rss_calls)==3
    assert len(rows)==1 and rows==again
    assert rows[0]['title']=='Bitcoin product launch'
    assert rows[0]['snippet']=='Bitcoin network update & product details.'
    assert rows[0]['url']=='https://example.test/btc'
    assert rows[0]['publisher']=='Fixture Wire'
    assert rows[0]['published_at']=='2026-11-23T09:00:00Z'
    assert rows[0]['fetched_at']=='2026-11-23T12:00:00Z'

def test_rss_failure_is_safe_and_empty_state_is_cached(db,monkeypatch):
    monkeypatch.setattr(provider.yf,'Search',lambda *a,**k:SimpleNamespace(news=[]))
    calls=[]
    def malformed(req,timeout):calls.append(1);return BytesIO(b'not xml')
    monkeypatch.setattr(provider.request,'urlopen',malformed)
    assert mc.get_articles('AAPL',NOW)==[]
    assert mc.get_articles('AAPL',NOW+pd.Timedelta(minutes=1))==[]
    assert len(calls)==3
    with database.get_connection() as conn:
        assert 'ParseError' in conn.execute('SELECT errors_json FROM news_fetch_state').fetchone()[0]

def test_related_reasoning_fetcher_uses_shared_provider(monkeypatch):
    import reasoning_engine as reasoning
    item=dict(title='Apple revenue update',pubDate='2026-11-23T09:00:00Z')
    monkeypatch.setattr(provider,'yahoo_news',lambda symbol:[item])
    monkeypatch.setattr(reasoning.yf,'Ticker',lambda _:SimpleNamespace(get_calendar=lambda:{}))
    reasoning.fetch_market_context.clear()
    try:
        result=reasoning.fetch_market_context('AAPL')
        assert result['news']==[item] and result['source_status']['news']=='available'
    finally:reasoning.fetch_market_context.clear()

def story(title,snippet='',minutes=1,path='story'):
    return dict(title=title,summary=snippet,relatedTickers=['AAPL','NVDA'],publisher='Fixture',
        link='https://example.test/'+path,providerPublishTime=int((NOW-pd.Timedelta(minutes=minutes)).timestamp()))

@pytest.mark.parametrize('title,snippet',[
    ('Why Is HPE Stock Up Today?','Hewlett Packard Enterprise raises guidance.'),
    ('Trump calls for Nvidia to reinstate Jensen Huang','Nvidia shares rise.'),
    ('Pineapple production rises','Pineapple growers expand.'),
    ('Why Is HPE Stock Up Today?','Analysts also compare Apple valuations.'),
])
def test_related_tickers_cannot_admit_foreign_or_substring_news(title,snippet):
    assert mc.parse_articles('AAPL',[story(title,snippet)],'fixture',NOW)==[]

@pytest.mark.parametrize('symbol,title',[
    ('AAPL','Tim Cook discusses the next launch'),('AAPL','iPhone demand improves'),
    ('NVDA','Jensen Huang addresses policy changes'),('NVDA','Blackwell shipments expand'),
    ('MSFT','Satya Nadella announces a product update'),('SOL-USD','Solana network upgrade arrives'),
])
def test_key_entity_aliases_are_direct_evidence(symbol,title):
    assert len(mc.parse_articles(symbol,[story(title)],'fixture',NOW))==1

def test_title_rank_recency_and_roundup_demotion(db):
    items=[story('Markets end mixed','Nvidia suppliers face uncertainty.',minutes=1,path='market'),
           story('Jensen Huang discusses export policy',minutes=120,path='jensen'),
           story('Blackwell launch draws attention',minutes=60,path='blackwell'),
           story('3 stocks to buy: Nvidia and Apple',minutes=2,path='roundup'),
           story('Apple gains while Nvidia suppliers report','Nvidia is mentioned in a broader story.',minutes=3,path='other')]
    rows=mc.get_articles('NVDA',NOW,[('fixture',lambda *_:items)])
    assert [r['url'].rsplit('/',1)[-1] for r in rows]==['blackwell','jensen','roundup','other','market']
    assert rows[1]['title'].startswith('Jensen Huang')

def test_old_cached_metadata_noise_is_rejected_without_refetch(db):
    # Simulate an article inserted by the old, permissive relevance gate.
    row=mc.parse_articles('NVDA',[story('Jensen Huang discusses exports')],'fixture',NOW)[0]
    row['symbol']='AAPL'
    with database.get_connection() as conn:
        conn.execute('INSERT INTO news_articles_cache('+','.join(row)+') VALUES('+','.join('?' for _ in row)+')',list(row.values()))
        conn.execute('INSERT INTO news_fetch_state VALUES(?,?,?)',('AAPL',NOW.strftime('%Y-%m-%dT%H:%M:%SZ'),'[]'))
    def forbidden(*args):pytest.fail('Fresh cache should not refetch')
    assert mc.get_articles('AAPL',NOW,[('fixture',forbidden)])==[]

def test_unknown_company_uses_cached_ticker_metadata(monkeypatch):
    import news_entities as entities
    calls=[]
    def info():calls.append(1);return dict(shortName='Example Robotics Inc.')
    monkeypatch.setattr(provider.yf,'Ticker',lambda _:SimpleNamespace(get_info=info))
    entities.company_name.cache_clear()
    try:
        assert entities.company_name('ZZZQ')=='Example Robotics'
        assert entities.relevance('ZZZQ','Example Robotics announces a product','')==0
        assert calls==[1]
    finally:entities.company_name.cache_clear()

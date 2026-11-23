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

@pytest.mark.parametrize('symbol,queries,title',[('AAPL',['AAPL'],'Apple earnings outlook'),
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
    monkeypatch.setattr(provider,'rss_news',lambda _:pytest.fail('RSS not needed when Search succeeds'))
    first=mc.get_articles(symbol,NOW)
    second=mc.get_articles(symbol,NOW+pd.Timedelta(minutes=14))
    assert calls==queries and len(first)==1 and first==second
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
        assert urlsplit(req.full_url).hostname=='feeds.finance.yahoo.com'
        assert parse_qs(urlsplit(req.full_url).query)==dict(s=['BTC-USD'],region=['US'],lang=['en-US'])
        return BytesIO(rss_xml)
    monkeypatch.setattr(provider.yf,'Search',search)
    monkeypatch.setattr(provider.request,'urlopen',urlopen)
    rows=mc.get_articles('BTC-USD',NOW)
    again=mc.get_articles('BTC-USD',NOW+pd.Timedelta(minutes=1))
    assert search_calls==['BTC-USD','Bitcoin'] and len(rss_calls)==1
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
    assert len(calls)==1
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

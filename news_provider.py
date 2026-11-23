"""Zero-key Yahoo Search news, with a bounded public RSS fallback (no article I/O)."""
from datetime import timezone
from email.utils import parsedate_to_datetime
from html import unescape
import re
from urllib import request
from urllib.parse import urlencode
from xml.etree import ElementTree
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
from news_entities import company_name

MAX_RSS_BYTES=2_000_000

def parse_rss(payload):
    """Return provider metadata in the same flat shape as Yahoo Search news."""
    root=ElementTree.fromstring(payload)
    articles=[]
    for item in root.findall('./channel/item'):
        title=item.findtext('title');link=item.findtext('link');published=item.findtext('pubDate')
        if not title or not link or not published:continue
        try:
            stamp=parsedate_to_datetime(published)
            stamp=stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp.astimezone(timezone.utc)
        except (ValueError,TypeError,OverflowError):continue
        snippet=unescape(re.sub(r'<[^>]*>','',item.findtext('description') or ''))
        articles.append(dict(title=unescape(title),link=link,summary=snippet,
            pubDate=stamp.strftime('%Y-%m-%dT%H:%M:%SZ'),
            publisher=item.findtext('source') or 'Yahoo Finance'))
    return articles

def _rss_url(url):
    req=request.Request(url,headers={'User-Agent':'CAPITEX/1.0','Accept':'application/rss+xml, application/xml'})
    with request.urlopen(req,timeout=5) as response:
        payload=response.read(MAX_RSS_BYTES+1)
    if len(payload)>MAX_RSS_BYTES:raise ValueError('Yahoo headline RSS exceeds size limit.')
    return parse_rss(payload)

def rss_news(query):
    return _rss_url('https://feeds.finance.yahoo.com/rss/2.0/headline?'+urlencode(dict(s=query,region='US',lang='en-US')))

def google_news(symbol,name):
    query=f'"{name}" OR "{symbol}" when:7d'
    return _rss_url('https://news.google.com/rss/search?'+urlencode(dict(q=query,hl='en-US',gl='US',ceid='US:en')))

def search_news(query):
    return yf.Search(query,max_results=0,news_count=10,timeout=5).news

def yahoo_news(symbol,now=None):
    name=company_name(symbol)
    queries=list(dict.fromkeys([symbol,name]))
    jobs=[(fetch,(query,)) for fetch in (search_news,rss_news) for query in queries]
    jobs.append((google_news,(symbol,name)))
    articles=[];failures=[]
    # Independently query every source, even when Search succeeds. Bound the fan-out.
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(fetch,*args) for fetch,args in jobs]
        for future in futures:
            try:
                news=future.result()
                if isinstance(news,list):articles.extend(item for item in news if isinstance(item,dict))
            except Exception as exc:failures.append(exc)
    if not articles and failures:raise failures[-1]
    return articles

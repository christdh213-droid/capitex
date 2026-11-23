"""Zero-key Yahoo Search news, with a bounded public RSS fallback (no article I/O)."""
from datetime import timezone
from email.utils import parsedate_to_datetime
from html import unescape
import re
from urllib import request
from urllib.parse import urlencode
from xml.etree import ElementTree
import yfinance as yf
import config

COIN_NAMES={'BTC-USD':'Bitcoin','ETH-USD':'Ethereum','SOL-USD':'Solana'}
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

def rss_news(symbol):
    url='https://feeds.finance.yahoo.com/rss/2.0/headline?'+urlencode(dict(s=symbol,region='US',lang='en-US'))
    req=request.Request(url,headers={'User-Agent':'CAPITEX/1.0','Accept':'application/rss+xml, application/xml'})
    with request.urlopen(req,timeout=5) as response:
        payload=response.read(MAX_RSS_BYTES+1)
    if len(payload)>MAX_RSS_BYTES:raise ValueError('Yahoo headline RSS exceeds size limit.')
    return parse_rss(payload)

def yahoo_news(symbol,now=None):
    queries=[symbol]
    if symbol.endswith('-USD'):
        name=COIN_NAMES.get(symbol) or config.ASSET_UNIVERSE.get(symbol,{}).get('name')
        if name and name!=symbol:queries.append(name)
    articles=[]
    for query in queries:
        try:
            news=yf.Search(query,max_results=0,news_count=10,timeout=5).news
            if isinstance(news,list):articles.extend(item for item in news if isinstance(item,dict))
        except Exception:
            # Try the coin alias and then RSS; never fetch linked article pages.
            continue
    return articles if articles else rss_news(symbol)

"""Shared asset queries and text-only entity relevance."""
from functools import lru_cache
import re
import config

ENTITIES={
    'AAPL':('Apple','iPhone','Tim Cook'),
    'NVDA':('Nvidia','Jensen Huang','GeForce','Blackwell'),
    'MSFT':('Microsoft','Satya Nadella','Azure'),
    'TSLA':('Tesla','Elon Musk'),
    'AMZN':('Amazon','Andy Jassy','Amazon Web Services'),
    'GOOGL':('Alphabet','Google','Sundar Pichai'),
    'GOOG':('Alphabet','Google','Sundar Pichai'),
    'META':('Meta Platforms','Facebook','Instagram','Mark Zuckerberg'),
    'BTC-USD':('Bitcoin','BTC'),
    'ETH-USD':('Ethereum','ETH','Vitalik Buterin'),
    'SOL-USD':('Solana','SOL'),
    'HPE':('Hewlett Packard Enterprise','Hewlett-Packard Enterprise'),
}

def clean_name(name):
    return re.sub(r'(?:,?\s+(?:Inc\.?|Corp\.?|Corporation|Ltd\.?|Limited|plc|Class [A-Z]))+$','',name,flags=re.I).strip()

@lru_cache(maxsize=256)
def company_name(symbol):
    configured=config.ASSET_UNIVERSE.get(symbol,{}).get('name')
    if configured:return clean_name(configured)
    if symbol in ENTITIES:return ENTITIES[symbol][0]
    try:
        import yfinance as yf
        info=yf.Ticker(symbol).get_info()
        name=info.get('shortName') or info.get('longName')
        if name:return clean_name(name)
    except Exception:pass
    return symbol

def aliases(symbol):
    return tuple(dict.fromkeys([symbol,company_name(symbol),*ENTITIES.get(symbol,())]))

def matches(text,terms):
    return any(len(term)>1 and re.search(r'(?<!\w)'+re.escape(term)+r'(?!\w)',text,re.I) for term in terms)

def relevance(symbol,title,snippet):
    """0 direct title, 1 roundup title, 2 snippet-only, None excluded."""
    terms=aliases(symbol)
    if matches(title,terms):
        roundup=re.search(r'\b(?:top\s+\d+|\d+\s+(?:best\s+)?stocks|stocks?\s+to\s+(?:buy|watch)|roundup|round-up|stock market today|market movers)\b',title,re.I)
        def first(entities):
            found=[re.search(r'(?<!\w)'+re.escape(a)+r'(?!\w)',title,re.I) for a in entities]
            return min((m.start() for m in found if m),default=len(title)+1)
        own_position=first(terms)
        other_primary=any(first((other,*entities))<own_position for other,entities in ENTITIES.items()
            if other!=symbol and not ({other,symbol}<= {'GOOG','GOOGL'}))
        return 1 if roundup or other_primary else 0
    if not matches(snippet,terms):return None
    for other,entities in ENTITIES.items():
        if other==symbol or (other in ('GOOG','GOOGL') and symbol in ('GOOG','GOOGL')):continue
        if matches(title,(other,*entities)):return None
    if re.search(r'\b[A-Z][A-Z0-9.\-]{1,9}\s+(?:[Ss]tock|[Ss]hares)\b',title):return None
    return 2

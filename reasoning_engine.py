"""Transparent market sentiment and evidence-backed catalyst tags for CAPITEX.

This lexical/technical composite is not an official Fear & Greed index and is not
an LSTM attribution. Missing provider inputs are disclosed and weights renormalized.
"""
from datetime import date, datetime
import json
import math
import re
import numpy as np
import pandas as pd
import yfinance as yf
import config
from data_loader import _cached, _symbol
import indicators

_POSITIVE={'beat':1.2,'beats':1.2,'growth':.8,'gain':.8,'gains':.8,'surge':1.0,
           'surges':1.0,'rally':1.0,'record':.6,'upgrade':1.2,'upgrades':1.2,
           'profit':.6,'profits':.6,'bullish':1.0,'approval':.8,'outperform':1.1,
           'strong':.6,'breakthrough':.8,'expansion':.7,'rises':.7,'soars':1.0}
_NEGATIVE={'miss':-1.2,'misses':-1.2,'loss':-.8,'losses':-.8,'fall':-.8,
           'falls':-.8,'drop':-.8,'drops':-.8,'slump':-1.0,'crash':-1.2,
           'downgrade':-1.2,'downgrades':-1.2,'bearish':-1.0,'fraud':-1.2,
           'lawsuit':-.9,'weak':-.6,'decline':-.8,'cuts':-.6,'plunges':-1.0,
           'bankruptcy':-1.5,'warning':-.7,'hack':-1.2,'outflow':-.7}
_NEGATORS={'not','no','never','without','isnt','wasnt','didnt'}
_UNSET=object()


def _timestamp(value):
    if isinstance(value,(int,float,np.integer,np.floating)):
        if not np.isfinite(value): raise ValueError('Invalid timestamp')
        return pd.Timestamp(value,unit='ms' if abs(value)>1e11 else 's',tz='UTC')
    stamp=pd.Timestamp(value)
    if pd.isna(stamp): raise ValueError('Invalid timestamp')
    return stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp.tz_convert('UTC')


def lexical_polarity(headline):
    """Small auditable finance lexicon, with three-token local negation handling."""
    tokens=re.findall(r"[a-z]+",str(headline).lower().replace("'",''))
    total=0.0
    matches=0
    for i,token in enumerate(tokens):
        weight=_POSITIVE.get(token,_NEGATIVE.get(token,0))
        if weight:
            if any(t in _NEGATORS for t in tokens[max(0,i-3):i]): weight=-weight
            total+=weight
            matches+=1
    return float(math.tanh(total/max(math.sqrt(matches),1)))


def _recent_headlines(news, as_of):
    records=[]
    seen=set()
    for item in news if isinstance(news,list) else []:
        if not isinstance(item,dict): continue
        content=item.get('content') if isinstance(item.get('content'),dict) else item
        title=content.get('title')
        published=content.get('pubDate') or content.get('displayTime') or content.get('providerPublishTime') or item.get('providerPublishTime')
        if not isinstance(title,str) or not title.strip() or published is None: continue
        try: stamp=_timestamp(published)
        except (ValueError,TypeError,OverflowError): continue
        if not as_of-pd.Timedelta(days=7)<=stamp<=as_of: continue
        key=title.strip().casefold()
        if key in seen: continue
        seen.add(key)
        records.append({'headline':title.strip(),'timestamp':stamp.isoformat(),
                        'polarity':lexical_polarity(title)})
    return sorted(records,key=lambda r:r['timestamp'],reverse=True)[:20]


@_cached(config.REASONING_CACHE_TTL)
def fetch_market_context(ticker):
    """Best-effort yfinance news/calendar fetch; preserve each source's status."""
    ticker=_symbol(ticker)
    provider=yf.Ticker(ticker)
    result={'news':[],'calendar':{},'source_status':{},
            'fetched_at':pd.Timestamp.now(tz='UTC').isoformat()}
    try:
        news=provider.get_news(count=20)
        if not isinstance(news,list): raise ValueError('Unexpected news response')
        result['news']=news
        result['source_status']['news']='available' if news else 'empty'
    except Exception as exc:
        result['source_status']['news']='unavailable'
        result['source_status']['news_error']=type(exc).__name__
    if config.ASSET_UNIVERSE.get(ticker,{}).get('kind')=='crypto' or ticker.endswith('-USD'):
        result['source_status']['calendar']='not_applicable_crypto'
    else:
        try:
            calendar=provider.get_calendar()
            if not isinstance(calendar,(dict,pd.DataFrame)):
                raise ValueError('Unexpected calendar response')
            result['calendar']=calendar
            result['source_status']['calendar']='available' if len(calendar) else 'empty'
        except Exception as exc:
            result['source_status']['calendar']='unavailable'
            result['source_status']['calendar_error']=type(exc).__name__
    return result


def _earnings_dates(calendar):
    if isinstance(calendar,pd.DataFrame):
        found=[]
        for index,row in calendar.iterrows():
            if 'earn' in str(index).lower() and 'date' in str(index).lower():
                found.extend(row.tolist())
        for column in calendar.columns:
            if 'earn' in str(column).lower() and 'date' in str(column).lower():
                found.extend(calendar[column].tolist())
        candidates=found
    elif isinstance(calendar,dict):
        candidates=[value for key,value in calendar.items()
                    if 'earn' in str(key).lower() and 'date' in str(key).lower()]
    else: return []
    flat=[]
    def unpack(value):
        if isinstance(value,(list,tuple,np.ndarray,pd.Series)):
            for nested in value: unpack(nested)
        elif isinstance(value,dict):
            for nested in value.values(): unpack(nested)
        else:
            try: flat.append(_timestamp(value))
            except (TypeError,ValueError,OverflowError): pass
    for value in candidates: unpack(value)
    return sorted(set(stamp.date() for stamp in flat))


def _volume_ratio(frame):
    if 'Volume' not in frame or len(frame)<21: return None
    volume=pd.to_numeric(frame.Volume,errors='coerce')
    prior=volume.iloc[-21:-1]
    if not np.isfinite(prior).all() or (prior<0).any() or prior.mean()<=0:
        return None
    latest=float(volume.iloc[-1])
    if not np.isfinite(latest) or latest<0: return None
    return latest/float(prior.mean())


def _macro_tags(as_of, schedule=None):
    if schedule is None:
        try:
            schedule=json.loads((config.DATA_DIR/'fomc_schedule.json').read_text(encoding='utf-8-sig'))
        except (OSError,ValueError):
            return [],{'status':'unavailable'}
    today=as_of.tz_convert('America/New_York').date()
    if today.year not in schedule.get('covered_years',[]):
        return [],{'status':'outside_verified_schedule','source':schedule.get('source')}
    tags=[]
    for start,end in schedule.get('meetings',[]):
        start,end=date.fromisoformat(start),date.fromisoformat(end)
        away=(start-today).days
        if today<=end and away<=7:
            tags.append({'id':'fed_window','label':'Fed Meeting Window' if away<=0 else f'Fed Meeting - {away} Days',
                         'tone':'caution','event_date':start.isoformat(),'end_date':end.isoformat(),
                         'days_away':max(0,away),'source':schedule.get('source'),
                         'schedule_verified_on':schedule.get('verified_on'),'tentative':True})
    return tags,{'status':'verified_schedule_snapshot','source':schedule.get('source'),
                 'verified_on':schedule.get('verified_on'),'covered_years':schedule.get('covered_years')}


def build_reasoning(ticker, df, *, news=_UNSET, calendar=_UNSET, as_of=None,
                    fetch_context=True, macro_schedule=None):
    """Return sentiment, technical pills and catalyst evidence, all JSON-safe.

    Inject news/calendar for reproducible academic evaluation. Current news is never
    presented as causal evidence for a historical forecast or as an LSTM attribution.
    """
    ticker=_symbol(ticker)
    frame=indicators.validate_daily_prices(df)
    if len(frame)<21: raise ValueError('Reasoning needs at least 21 daily bars.')
    now=_timestamp(as_of if as_of is not None else pd.Timestamp.now(tz='UTC'))
    last=_timestamp(frame.index[-1])
    if last>now: raise ValueError('Price data cannot end after reasoning as_of.')
    status={}
    if fetch_context and (news is _UNSET or calendar is _UNSET):
        context=fetch_market_context(ticker)
        status=dict(context['source_status'])
    else: context={}
    if news is _UNSET: news=context.get('news',[])
    else: status['news']='provided'
    if calendar is _UNSET: calendar=context.get('calendar',{})
    else: status['calendar']='provided'
    status.setdefault('news','not_requested')
    status.setdefault('calendar','not_requested')
    headlines=_recent_headlines(news,now)
    technical=indicators.technical_summary(frame)
    returns=np.log(frame.Close/frame.Close.shift(1))
    momentum=float(frame.Close.iloc[-1]/frame.Close.iloc[-6]-1)
    vol=float(returns.tail(20).std(ddof=1))
    scale=max(vol*math.sqrt(5),.005)
    momentum_score=float(50+50*math.tanh(math.log1p(momentum)/scale))
    rsi=technical['values']['RSI_14']
    ratio=_volume_ratio(frame)
    volume_score=None if ratio is None else float(50+50*math.tanh((ratio-1)*np.sign(momentum)))
    news_score=None if not headlines else float(50+50*np.mean([h['polarity'] for h in headlines]))
    inputs=[('headlines',news_score,.30),('momentum_5d',momentum_score,.30),
            ('rsi_regime',rsi,.20),('volume_ratio',volume_score,.20)]
    weight=sum(w for _,score,w in inputs if score is not None)
    score=float(np.clip(sum(value*w for _,value,w in inputs if value is not None)/weight,0,100))
    label=('Extreme Fear' if score<20 else 'Fear' if score<40 else 'Neutral' if score<=60
           else 'Greed' if score<80 else 'Extreme Greed')
    components=[{'factor':name,'score':None if value is None else round(value,2),
                 'weight':w/weight if value is not None else 0.0,'available':value is not None}
                for name,value,w in inputs]
    catalysts=[]
    today=now.tz_convert('America/New_York').date()
    crypto=config.ASSET_UNIVERSE.get(ticker,{}).get('kind')=='crypto' or ticker.endswith('-USD')
    if not crypto:
        dates=[d for d in _earnings_dates(calendar) if 0<=(d-today).days<=30]
        if dates:
            earliest=dates[0]
            days=(earliest-today).days
            catalysts.append({'id':'earnings','label':f'Earnings Window - {days} Days',
                              'tone':'caution','event_date':earliest.isoformat(),
                              'latest_estimated_date':dates[-1].isoformat(),
                              'days_away':days,'source':'yfinance calendar',
                              'note':'Estimated earnings date; no confirmed call time supplied.'})
    if ratio is not None and ratio>=2:
        catalysts.append({'id':'volume_surge','label':f'Volume Surge (+{(ratio-1)*100:.0f}%)',
                          'tone':'caution','ratio':ratio,'source':'latest volume / prior 20-bar average'})
    index=pd.to_datetime(frame.index,utc=True)
    trailing=frame.loc[index>=last-pd.Timedelta(days=365)]
    if index[0]<=last-pd.Timedelta(days=350):
        high=pd.to_numeric(trailing.get('High',trailing.Close),errors='coerce')
        high=float(high.replace([np.inf,-np.inf],np.nan).max())
        if np.isfinite(high) and high>0 and float(frame.Close.iloc[-1])>=.98*high:
            catalysts.append({'id':'high_test','label':'52-Week High Test','tone':'bullish',
                              'high':high,'source':'trailing 365 calendar days of available price bars'})
    macro,macro_status=_macro_tags(now,macro_schedule)
    catalysts.extend(macro)
    status['macro_calendar']=macro_status
    return {'ticker':ticker,'as_of':now.isoformat(),'price_as_of':last.isoformat(),
            'event_timezone':'America/New_York','source_status':status,
            'sentiment':{'score':round(score,1),'label':label,'components':components,
                         'momentum_5d_pct':momentum*100,'volume_ratio':ratio,
                         'headlines_used':len(headlines),'headlines':headlines,
                         'method':'CAPITEX lexical + momentum + RSI + directional volume heuristic',
                         'is_official_fear_greed_index':False},
            'technical_summary':technical,'catalyst_tags':catalysts,
            'limitations':['Lexical headline scoring can miss sarcasm, ambiguity and financial context.',
                          'Factors describe market context; they are not causal LSTM feature attributions.',
                          'Missing inputs are omitted and available weights are renormalized.',
                          'News/calendar sources and the Fed schedule may be incomplete or delayed.']}

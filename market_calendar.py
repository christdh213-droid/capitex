"""Daily-bar close conventions; official XNYS sessions for US listings."""
from functools import lru_cache
import pandas as pd
import exchange_calendars as xc
from time_utils import utc_z,now_utc

def asset_type(symbol):
    return 'crypto' if symbol.endswith('-USD') else 'stock'

@lru_cache(maxsize=12)
def us_calendar(year):
    return xc.get_calendar('XNYS',start=f'{year-2}-01-01',end=f'{year+3}-12-31')

@lru_cache(maxsize=50000)
def session_close(symbol,day):
    day=pd.Timestamp(day).date()
    if asset_type(symbol)=='crypto':return pd.Timestamp(f'{day}T23:59:59Z')
    if '.' not in symbol:
        cal=us_calendar(day.year);label=pd.Timestamp(day)
        return cal.session_close(label) if cal.is_session(label) else None
    # Non-US suffixes use weekday schedules; actual bar publication gates grading.
    if day.weekday()>=5:return None
    zones={'.NS':('Asia/Kolkata','15:30'),'.BO':('Asia/Kolkata','15:30'),
           '.L':('Europe/London','16:30'),'.TO':('America/Toronto','16:00')}
    zone,clock=next((v for suffix,v in zones.items() if symbol.endswith(suffix)),('UTC','23:59:59'))
    return pd.Timestamp(f'{day} {clock}',tz=zone).tz_convert('UTC')

def bar_day(symbol,stamp):
    stamp=pd.Timestamp(stamp)
    # Daily provider timestamps are exchange-local midnight. Preserve their date.
    return stamp.date()

def completed_history(symbol,history,now=None):
    now=pd.Timestamp(now or now_utc())
    now=now.tz_localize('UTC') if now.tzinfo is None else now.tz_convert('UTC')
    mask=[]
    for stamp in history.index:
        close=session_close(symbol,bar_day(symbol,stamp))
        mask.append(close is not None and close<=now)
    result=history.loc[mask].copy()
    if result.empty:raise ValueError('No completed daily bar is available.')
    return result

def targets(symbol,data_timestamp,horizon):
    current=pd.Timestamp(data_timestamp).date();result=[]
    while len(result)<horizon:
        current+=pd.Timedelta(days=1)
        close=session_close(symbol,current)
        if close is not None:result.append(utc_z(close))
    return result

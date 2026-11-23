"""Target-bar grading with injectable clock/provider and no network-held DB locks."""
import asyncio,logging,math,os
from datetime import timedelta
from threading import Lock
import pandas as pd
import yfinance as yf
import database
from time_utils import now_utc,utc_z
from market_calendar import bar_day

log=logging.getLogger(__name__)
_RUN_LOCK=Lock()

def historical_provider(symbol,start,end):
    """One unadjusted daily fetch includes split actions for the entire group."""
    return yf.Ticker(symbol).history(start=str(start),end=str(end),interval='1d',
        auto_adjust=False,actions=True,timeout=12,raise_errors=True)

def eligible(row,now):
    target=pd.Timestamp(row['target_timestamp'])
    return target.date()<now.date() if row['asset_type']=='crypto' else now>=target+timedelta(minutes=30)

def resolve(row,frame,now,error=None):
    target=pd.Timestamp(row['target_timestamp']);origin=pd.Timestamp(row['data_timestamp'])
    grace=timedelta(hours=48) if row['asset_type']=='crypto' else timedelta(days=5)
    if not eligible(row,now):return None
    if frame is not None and not frame.empty:
        if row['asset_type']=='stock' and 'Stock Splits' in frame:
            for stamp,split in frame['Stock Splits'].items():
                if origin.date()<bar_day(row['symbol'],stamp)<=target.date() and pd.notna(split) and float(split)!=0:
                    return {'status':'UNAVAILABLE','unavailable_reason':'stock_split_in_window'}
        candidates=[]
        for stamp,bar in frame.iterrows():
            stamp=pd.Timestamp(stamp)
            # Provider contract is daily OHLCV. Reject apparent intraday fixtures.
            daily=(stamp.hour==0 and stamp.minute==0) or (stamp.tzinfo is not None and row['asset_type']=='stock' and
                    stamp.tz_convert('America/New_York').hour==0 and stamp.minute==0)
            if daily and bar_day(row['symbol'],stamp)==target.date():
                try:price=float(bar['Close'])
                except (TypeError,ValueError,KeyError):continue
                if math.isfinite(price) and price>0:candidates.append(price)
        if len(candidates)==1:
            return {'status':'GRADED','actual_price':candidates[0],'actual_timestamp':utc_z(target),'unavailable_reason':None}
    if now>=target+grace:
        return {'status':'UNAVAILABLE','unavailable_reason':'provider_unavailable_after_grace' if error else 'target_bar_missing_after_grace'}
    return None

def grade_pending(now=None,provider=None):
    now=pd.Timestamp(now or now_utc());provider=provider or historical_provider
    if not _RUN_LOCK.acquire(blocking=False):return {'graded':0,'unavailable':0,'busy':True}
    counts={'graded':0,'unavailable':0}
    try:
        with database.get_connection() as conn:
            rows=[dict(r) for r in conn.execute("SELECT * FROM forecast_logs WHERE status='PENDING' AND target_timestamp<=?",(utc_z(now),))]
        groups={}
        for row in rows:
            if eligible(row,now):groups.setdefault(row['symbol'],[]).append(row)
        for symbol,group in groups.items():
            start=min(pd.Timestamp(r['data_timestamp']).date() for r in group)
            end=max(pd.Timestamp(r['target_timestamp']).date() for r in group)+timedelta(days=2)
            error=None
            # All connections are CLOSED before this blocking provider operation.
            try:frame=provider(symbol,start,end)
            except Exception as exc:frame=None;error=type(exc).__name__
            decisions=[(row,resolve(row,frame,now,error)) for row in group]
            with database.get_connection() as conn:
                for row,decision in decisions:
                    if decision is None:continue
                    cols={**decision,'graded_at':utc_z(now)}
                    cursor=conn.execute('UPDATE forecast_logs SET '+','.join(k+'=?' for k in cols)+" WHERE id=? AND status='PENDING'",
                                        [*cols.values(),row['id']])
                    counts['graded' if decision['status']=='GRADED' else 'unavailable']+=cursor.rowcount
        return counts
    finally:_RUN_LOCK.release()

async def grader_loop():
    try:interval=max(1,float(os.getenv('CAPITEX_GRADER_INTERVAL_SEC','900')))
    except ValueError:interval=900
    while True:
        try:await asyncio.to_thread(grade_pending)
        except asyncio.CancelledError:raise
        except Exception:log.exception('Forecast grading deferred after an error')
        await asyncio.sleep(interval)

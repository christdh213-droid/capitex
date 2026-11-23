"""Persist immutable forecast payloads, selecting the winner after IGNORE."""
import hashlib,json,math
import database
from time_utils import utc_z,now_utc
from market_calendar import asset_type,session_close,bar_day,targets,completed_history

def model_version(bundle):
    metadata=getattr(bundle,'metadata',{})
    # Fingerprint represents trained weights/config; fixture fallback is deterministic.
    return str(metadata.get('fingerprint') or hashlib.sha256(b'capitex-return-path-v1-frozen').hexdigest())[:24]

def log_forecasts(symbol,bundle,history,infer,now=None,horizons=(1,7,30)):
    now=now or now_utc();history=completed_history(symbol,history,now)
    data_ts=utc_z(session_close(symbol,bar_day(symbol,history.index[-1])))
    anchor=float(history.Close.iloc[-1]);version=model_version(bundle)
    results={}
    for horizon in horizons:
        with database.get_connection() as conn:
            old=conn.execute('SELECT * FROM forecast_logs WHERE symbol=? AND horizon=? AND data_timestamp=? AND model_version=?',
                             (symbol,horizon,data_ts,version)).fetchone()
        if old:results[str(horizon)]=dict(old);continue
        payload=infer(bundle,horizon,current_price=anchor,ticker=symbol)
        corridor=payload['corridor']
        if any(len(corridor[k])!=horizon+1 for k in ('median','lower','upper')):
            raise ValueError('Forecast corridor length does not match its horizon.')
        timestamps=[data_ts]+targets(symbol,data_ts,horizon)
        corridor.update(timestamps=timestamps,calendar='official_XNYS' if asset_type(symbol)=='stock' and '.' not in symbol else 'UTC_daily' if asset_type(symbol)=='crypto' else 'weekday_bar_availability')
        predicted=float(corridor['median'][-1]);low=float(corridor['lower'][-1]);high=float(corridor['upper'][-1])
        if not all(math.isfinite(v) and v>0 for v in (anchor,low,predicted,high)) or not low<=predicted<=high:
            raise ValueError('Invalid forecast prices.')
        move=(predicted/anchor-1)*100
        payload.update(current_price=anchor,median_target_price=predicted,lower_target_price=low,
                       upper_target_price=high,expected_change_pct=move,outlook='bullish' if move>.25 else 'bearish' if move<-.25 else 'neutral',
                       forecast_as_of=data_ts,anchor_source='unadjusted_completed_daily_close')
        with database.get_connection() as conn:
            conn.execute('''INSERT OR IGNORE INTO forecast_logs(symbol,asset_type,model_version,issued_at,data_timestamp,horizon,
                starting_price,target_timestamp,predicted_price,band_low,band_high,persistence_predicted_price,status,payload_json)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,'PENDING',?)''',
                (symbol,asset_type(symbol),version,utc_z(now),data_ts,horizon,anchor,timestamps[-1],predicted,low,high,anchor,json.dumps(payload,allow_nan=False)))
            results[str(horizon)]=dict(conn.execute('SELECT * FROM forecast_logs WHERE symbol=? AND horizon=? AND data_timestamp=? AND model_version=?',
                      (symbol,horizon,data_ts,version)).fetchone())
    return results,history

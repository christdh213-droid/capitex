"""Academic inspection and CSV exports, separated from the consumer forecast UI."""
import json
import pandas as pd
import indicators
from formatting import local_timestamp


def technical_records(history):
    frame=indicators.add_all_indicators(history,sma_period=20,ema_period=12)
    for name,period,kind in [('SMA_50',50,'sma'),('EMA_26',26,'ema')]:
        frame[name]=(indicators.compute_sma if kind=='sma' else indicators.compute_ema)(frame.Close,period)
    frame=frame.tail(252).copy()
    frame.index.name='timestamp'
    return json.loads(frame.reset_index().to_json(orient='records',date_format='iso'))


def predictions_csv(payload):
    corridor=payload['corridor']
    frame=pd.DataFrame({'timestamp_utc':corridor['timestamps'],
        'median_price':corridor['median'],'lower_80_price':corridor['lower'],'upper_80_price':corridor['upper']})
    frame.insert(0,'ticker',payload.get('ticker','CUSTOM'))
    frame['horizon_steps']=payload.get('horizon',len(frame)-1)
    frame['confidence_score_pct']=payload.get('confidence',payload.get('hero',{}).get('confidence'))
    frame['anchor_source']=payload.get('anchor_source','provided_live_price')
    return frame.to_csv(index=False,float_format='%.8f').encode('utf-8-sig')


def ledger_csv(rows):
    columns=['id','timestamp_ist','ticker','action','quantity','price','total_value','realized_pnl',
        'execution_source','execution_label','quote_timestamp']
    normalized=[dict(row,timestamp_ist=local_timestamp(row['timestamp'])) for row in rows]
    return pd.DataFrame(normalized,columns=columns).to_csv(index=False,float_format='%.8f').encode('utf-8-sig')


def records_frame(records):
    frame=pd.DataFrame(records)
    if frame.empty:raise ValueError('No technical history is available.')
    return frame.set_index(pd.to_datetime(frame.pop('timestamp'),utc=True))

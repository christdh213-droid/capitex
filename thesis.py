"""Build immutable evidence snapshots only from persisted server-owned rows."""
import json
from time_utils import utc_z

def metadata(forecast_log_ids=None,selected_horizon=None,context_summary_id=None,prepared_at=None):
    values=dict(forecast_log_ids=forecast_log_ids,selected_horizon=selected_horizon,
                context_summary_id=context_summary_id,prepared_at=prepared_at)
    return values if any(v is not None for v in values.values()) else None

def build(conn,symbol,meta,executed_at):
    """Caller holds the SAME write transaction that inserts the ledger record."""
    if meta is None:return None
    supplied=meta.get('forecast_log_ids')
    if supplied is None:supplied=[]
    if not isinstance(supplied,(list,dict)):raise ValueError('Forecast IDs must be a list or horizon map.')
    items=list(supplied.items()) if isinstance(supplied,dict) else [(None,v) for v in supplied]
    if len(items)>30:raise ValueError('Too many forecast IDs.')
    forecasts=[];seen=set()
    for horizon,identifier in items:
        if isinstance(identifier,bool) or not isinstance(identifier,int) or identifier<=0:
            raise ValueError('Invalid forecast ID.')
        row=conn.execute('SELECT * FROM forecast_logs WHERE id=?',(identifier,)).fetchone()
        if row is None or row['symbol']!=symbol:raise ValueError('Forecast ID is missing or belongs to another symbol.')
        if horizon is not None and str(horizon)!=str(row['horizon']):raise ValueError('Forecast horizon map does not match its row.')
        if identifier in seen:continue
        seen.add(identifier)
        data=dict(row);data['forecast']=json.loads(data.pop('payload_json'))
        forecasts.append(data)
    if forecasts:
        runs={(r['data_timestamp'],r['model_version']) for r in forecasts}
        if len(runs)!=1:raise ValueError('Forecast IDs must belong to the same completed-bar model run.')
        origin,version=next(iter(runs))
        # Expand IDs to every persisted horizon in that exact run, server-side.
        forecasts=[]
        for row in conn.execute('SELECT * FROM forecast_logs WHERE symbol=? AND data_timestamp=? AND model_version=? ORDER BY horizon',
                                (symbol,origin,version)):
            data=dict(row);data['forecast']=json.loads(data.pop('payload_json'));forecasts.append(data)
    selected=meta.get('selected_horizon')
    if selected is not None and (isinstance(selected,bool) or selected not in (1,7,30) or
                                not any(r['horizon']==selected for r in forecasts)):
        raise ValueError('Selected horizon must reference a supplied forecast.')
    summary_id=meta.get('context_summary_id');bullets=[];counter={}
    if summary_id is not None:
        if isinstance(summary_id,bool) or not isinstance(summary_id,int) or summary_id<=0:raise ValueError('Invalid context summary ID.')
        row=conn.execute('SELECT * FROM context_summary_cache WHERE id=?',(summary_id,)).fetchone()
        if row is None or row['symbol']!=symbol:raise ValueError('Context summary ID is missing or belongs to another symbol.')
        bullets=json.loads(row['bullets_json']);counter=json.loads(row['counter_thesis_json'])
    try:prepared=utc_z(meta['prepared_at']) if meta.get('prepared_at') else executed_at
    except Exception as exc:raise ValueError('Invalid prepared_at timestamp.') from exc
    snapshot=dict(symbol=symbol,prepared_at=prepared,executed_at=executed_at,
        context_summary_id=summary_id,bullets=bullets,counter_thesis=counter,
        forecasts=forecasts,selected_horizon=selected)
    return json.dumps(snapshot,sort_keys=True,allow_nan=False)

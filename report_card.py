"""Forward-only prediction outcomes; comparable scale and explicit denominators."""
import math
import database

def _metrics(rows,key,dollars):
    valid=[r for r in rows if r['actual_price'] is not None and math.isfinite(float(r['actual_price']))]
    errors=[float(r[key])-float(r['actual_price']) for r in valid]
    percentage=[abs((float(r[key])-float(r['actual_price']))/float(r['actual_price']))*100 for r in valid if r['actual_price']!=0]
    result={'mape':sum(percentage)/len(percentage) if percentage else None,'mape_n':len(percentage)}
    if dollars:
        result.update(mae=sum(abs(v) for v in errors)/len(errors) if errors else None,
                      rmse=math.sqrt(sum(v*v for v in errors)/len(errors)) if errors else None)
    return result

def get_report(symbol=None,asset_type=None,horizon=None):
    where=[];params=[]
    for column,value in [('symbol',symbol),('asset_type',asset_type),('horizon',horizon)]:
        if value is not None:where.append(column+'=?');params.append(value)
    with database.get_connection() as conn:
        rows=[dict(r) for r in conn.execute('SELECT * FROM forecast_logs'+(' WHERE '+' AND '.join(where) if where else '')+' ORDER BY issued_at DESC,id DESC',params)]
    graded=[r for r in rows if r['status']=='GRADED' and r['actual_price'] is not None]
    directed=[r for r in graded if r['actual_price']!=r['starting_price']]
    correct=sum((r['predicted_price']-r['starting_price'])*(r['actual_price']-r['starting_price'])>0 for r in directed)
    coverage=sum(r['band_low']<=r['actual_price']<=r['band_high'] for r in graded)
    clean=[]
    for row in rows:
        row.pop('payload_json',None)
        row['in_band']=row['band_low']<=row['actual_price']<=row['band_high'] if row['status']=='GRADED' and row['actual_price'] is not None else None
        clean.append(row)
    return {'graded':len(graded),'pending':sum(r['status']=='PENDING' for r in rows),'unavailable':sum(r['status']=='UNAVAILABLE' for r in rows),
        'lstm':_metrics(graded,'predicted_price',bool(symbol)),
        'naive_persistence':_metrics(graded,'persistence_predicted_price',bool(symbol)),
        'directional_accuracy_pct':100*correct/len(directed) if directed else None,'directional_n':len(directed),
        'interval_coverage_pct':100*coverage/len(graded) if graded else None,
        'interval_label':'Observed coverage of the estimated 80% interval',
        'low_sample':len(graded)<30,'sample_note':'Fewer than 30 graded forecasts; results are preliminary.' if len(graded)<30 else '',
        'overlap_note':'Overlapping multi-day forecasts issued on consecutive days reduce the effective independent sample size.',
        'dollar_metrics_visible':bool(symbol),'rows':clean}

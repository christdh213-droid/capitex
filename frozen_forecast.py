"""Inference from existing CAPITEX weights only; never train or write models."""
import json
from pathlib import Path
import numpy as np
import config,lstm_model as lm

def load_frozen(history,symbol):
    candidates=[]
    for manifest in Path(config.MODELS_DIR).glob('capitex_*.json'):
        try:
            info=json.loads(manifest.read_text(encoding='utf-8'))
            if info['parameters']['ticker']==symbol and manifest.with_suffix('.keras').exists():
                candidates.append((manifest.stat().st_mtime,manifest,info))
        except (ValueError,KeyError,OSError):continue
    if not candidates:raise ValueError('No existing trained model for this asset. This upgrade does not train models.')
    _,manifest,info=max(candidates,key=lambda r:r[0])
    dataset=lm.prepare_multihorizon_dataset(history,info['parameters']['window'])
    for scaler,prefix in [(dataset.x_scaler,'x'),(dataset.y_scaler,'y')]:
        scaler.mean_=np.asarray(info[f'{prefix}_scaler_mean']);scaler.scale_=np.asarray(info[f'{prefix}_scaler_scale']);scaler.var_=scaler.scale_**2
    dataset.scaled_features=dataset.x_scaler.transform(dataset.features).astype(np.float32)
    ys=dataset.y_scaler.transform(dataset.features[['log_return']]).reshape(-1)
    for part,origins in dataset.origins.items():
        dataset.X[part]=np.stack([dataset.scaled_features[i-dataset.window_size+1:i+1] for i in origins])
        for row,origin in enumerate(origins):
            length=min(30,len(ys)-origin-1);dataset.y[part][row,:length]=ys[origin+1:origin+1+length]
    model=lm.load_model(manifest.with_suffix('.keras'),compile=False)
    sigma,multipliers,calibration=lm._calibrate_intervals(model,dataset)
    metrics=lm._evaluate_multihorizon(model,dataset,sigma,multipliers)
    return lm.ForecastBundle(model,dataset,metrics,sigma,multipliers,dict(info,loaded_from_cache=True,cache_path=str(manifest.with_suffix('.keras'))),calibration)

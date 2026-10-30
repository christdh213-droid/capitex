"""CAPITEX forecasting: multi-feature return paths and calibrated uncertainty.

New APIs: prepare_multihorizon_dataset, train_multihorizon_model,
forecast_multihorizon. Prototype single-feature APIs below remain compatible
until the dedicated CAPITEX API/UI integration in Phases 3 and 4.

The legacy next-day engine also predicts log returns rather than price levels.
Return modeling alone does not guarantee accuracy or prevent persistence behavior.

Features:
1. Stationary log-return transformation: r_t = ln(Close_t / Close_{t-1}).
2. Chronological dataset splitting and sliding window construction.
3. MinMaxScaler fitted strictly on training returns only (no lookahead leakage).
4. Predicted close reconstructed dynamically: Close_hat_t = Close_{t-1}^{actual} * exp(r_hat_t).
5. Naive persistence baseline benchmark ('tomorrow = today').
6. Mean prediction bias tracking to monitor systematic drift.
7. Evaluator reporting RMSE, MAE, and Directional Accuracy % against 50% baseline.
8. Algorithmic BUY/SELL/HOLD signal generator with indicator confirmation.
"""

from datetime import date, datetime
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

import config

logger = logging.getLogger(__name__)


def set_reproducible_seeds(seed: int = config.RANDOM_SEED) -> None:
    """Fix random seeds across Python, NumPy, and TensorFlow for determinism."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class StreamlitTrainingCallback(Callback):
    """Custom Keras training callback updating Streamlit progress and live metrics."""

    def __init__(
        self,
        total_epochs: int,
        progress_bar: Optional[Any] = None,
        status_text: Optional[Any] = None,
        epoch_callback: Optional[Callable[[int, float, float], None]] = None,
    ) -> None:
        super().__init__()
        self.total_epochs = max(1, total_epochs)
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.epoch_callback = epoch_callback
        self.loss_history: List[float] = []
        self.val_loss_history: List[float] = []

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs or {}
        loss = float(logs.get("loss", 0.0))
        val_loss = float(logs.get("val_loss", loss))

        self.loss_history.append(loss)
        self.val_loss_history.append(val_loss)

        progress_pct = min(1.0, (epoch + 1) / self.total_epochs)

        if self.progress_bar is not None:
            self.progress_bar.progress(progress_pct)

        if self.status_text is not None:
            self.status_text.markdown(
                f"**Training Progress:** Epoch **{epoch + 1}/{self.total_epochs}** "
                f"| Loss: `{loss:.6f}` | Val Loss: `{val_loss:.6f}`"
            )

        if self.epoch_callback is not None:
            self.epoch_callback(epoch + 1, loss, val_loss)


def prepare_lstm_dataset(
    df: pd.DataFrame,
    window_size: int = config.DEFAULT_WINDOW_SIZE,
    train_split: float = config.DEFAULT_TRAIN_SPLIT_RATIO,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    MinMaxScaler,
    pd.DatetimeIndex,
    pd.DatetimeIndex,
    np.ndarray,
    np.ndarray,
]:
    """Prepare chronological sliding windows of LOG RETURNS, scaling on train split only.

    Formulation:
    - Close prices: P_0, P_1, ..., P_N
    - Log returns: r_t = ln(P_t / P_{t-1}) for t = 1 ... N
    - Scaler fitted ONLY on r_train (first int(len(r) * train_split) elements)
    - Reconstructed predicted close: P_hat_t = P_{t-1} * exp(r_hat_t)

    Returns:
        X_train: (N_train, window_size, 1)
        y_train: (N_train,)
        X_test: (N_test, window_size, 1)
        y_test: (N_test,)
        scaler: Fitted MinMaxScaler instance on train log returns
        train_dates: Target dates for training split
        test_dates: Target dates for test split
        test_prev_actual: Actual close of the immediately preceding day P_{t-1}
        test_actual_close: Actual close of the target test day P_t
    """
    close_vals = df["Close"].values.astype(np.float64)
    if len(close_vals) < window_size + 20:
        raise ValueError(
            f"Insufficient data bars ({len(close_vals)}) for window size {window_size}. "
            f"Requires at least {window_size + 20} bars."
        )

    # 1. Compute stationary log returns: r_t = ln(P_t / P_{t-1})
    log_returns = np.log(close_vals[1:] / close_vals[:-1]).reshape(-1, 1)
    ret_dates = df.index[1:]
    prev_close_series = close_vals[:-1]
    act_close_series = close_vals[1:]

    total_returns = len(log_returns)
    split_idx = int(total_returns * train_split)

    if split_idx <= window_size:
        raise ValueError(
            f"Train split ({split_idx}) must exceed window size ({window_size}). "
            "Expand date range or decrease window size."
        )

    # 2. Fit StandardScaler on training split ONLY (zero-centered for log returns)
    train_raw = log_returns[:split_idx]
    test_raw = log_returns[split_idx:]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled = scaler.transform(test_raw)

    full_scaled = np.vstack([train_scaled, test_scaled])

    # 3. Build training sliding windows
    X_train_list, y_train_list = [], []
    for i in range(window_size, split_idx):
        X_train_list.append(full_scaled[i - window_size : i, 0])
        y_train_list.append(full_scaled[i, 0])

    X_train = np.array(X_train_list, dtype=np.float32).reshape((-1, window_size, 1))
    y_train = np.array(y_train_list, dtype=np.float32)
    train_dates = ret_dates[window_size:split_idx]

    # 4. Build test sliding windows (incorporating trailing window from train)
    X_test_list, y_test_list = [], []
    for i in range(split_idx, len(full_scaled)):
        X_test_list.append(full_scaled[i - window_size : i, 0])
        y_test_list.append(full_scaled[i, 0])

    X_test = np.array(X_test_list, dtype=np.float32).reshape((-1, window_size, 1))
    y_test = np.array(y_test_list, dtype=np.float32)
    test_dates = ret_dates[split_idx:]

    # 5. Price references for test split
    test_prev_actual = prev_close_series[split_idx:]
    test_actual_close = act_close_series[split_idx:]

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        scaler,
        train_dates,
        test_dates,
        test_prev_actual,
        test_actual_close,
    )


def get_model_cache_path(
    ticker: str,
    start_date: Any,
    end_date: Any,
    window_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> Path:
    """Derive deterministic cache path using SHA-256 hash of training parameters."""
    start_str = str(start_date)
    end_str = str(end_date)
    clean_ticker = ticker.strip().upper()

    key_repr = (
        f"logret_std_{clean_ticker}_{start_str}_{end_str}_w{window_size}_ep{epochs}_"
        f"b{batch_size}_lr{learning_rate:.6f}"
    )
    hash_digest = hashlib.sha256(key_repr.encode("utf-8")).hexdigest()[:16]
    filename = f"lstm_{clean_ticker}_{hash_digest}.keras"
    return config.MODELS_DIR / filename


def build_lstm_network(
    window_size: int = config.DEFAULT_WINDOW_SIZE,
    learning_rate: float = config.DEFAULT_LEARNING_RATE,
) -> Sequential:
    """Construct two-layer Sequential LSTM architecture with tanh activation for log-return regression."""
    model = Sequential(
        [
            Input(shape=(window_size, 1)),
            LSTM(config.LSTM_UNITS_L1, return_sequences=True),
            Dropout(config.LSTM_DROPOUT),
            LSTM(config.LSTM_UNITS_L2, return_sequences=False),
            Dropout(config.LSTM_DROPOUT),
            Dense(config.LSTM_DENSE_UNITS, activation="tanh"),
            Dense(1),
        ]
    )
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mean_squared_error")
    return model


def train_or_load_lstm(
    ticker: str,
    start_date: Any,
    end_date: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    window_size: int = config.DEFAULT_WINDOW_SIZE,
    epochs: int = config.DEFAULT_EPOCHS,
    batch_size: int = config.DEFAULT_BATCH_SIZE,
    learning_rate: float = config.DEFAULT_LEARNING_RATE,
    force_retrain: bool = False,
    callback: Optional[Callback] = None,
) -> Tuple[Sequential, bool, Dict[str, Any]]:
    """Load existing cached .keras model or train new LSTM and serialize to disk."""
    set_reproducible_seeds()
    cache_path = get_model_cache_path(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        window_size=window_size,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    if cache_path.exists() and not force_retrain:
        try:
            logger.info(f"Loading cached LSTM model: {cache_path}")
            model = load_model(str(cache_path))
            info = {
                "cache_path": str(cache_path),
                "loaded_from_cache": True,
                "epochs_trained": epochs,
                "final_loss": 0.0,
            }
            return model, True, info
        except Exception as load_err:
            logger.warning(f"Failed to load cached model ({load_err}). Retraining fresh model.")

    logger.info(f"Training fresh LSTM model for {ticker} ({epochs} epochs)...")
    model = build_lstm_network(window_size=window_size, learning_rate=learning_rate)
    callbacks_list = [callback] if callback is not None else []

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks_list,
        verbose=0,
        shuffle=False,
    )

    try:
        model.save(str(cache_path))
        logger.info(f"Serialized trained LSTM model to {cache_path}")
    except Exception as save_err:
        logger.warning(f"Could not persist model to disk: {save_err}")

    final_loss = float(history.history["loss"][-1]) if history.history.get("loss") else 0.0
    val_loss = float(history.history["val_loss"][-1]) if history.history.get("val_loss") else final_loss

    info = {
        "cache_path": str(cache_path),
        "loaded_from_cache": False,
        "epochs_trained": epochs,
        "final_loss": final_loss,
        "final_val_loss": val_loss,
        "loss_curve": history.history.get("loss", []),
        "val_loss_curve": history.history.get("val_loss", []),
    }
    return model, False, info


def evaluate_forecast_with_baseline(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    prev_actual: np.ndarray,
    pred_log_returns: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate LSTM predictions against a Naive Baseline ('tomorrow close = today close').

    Computes:
    - LSTM RMSE & MAE ($)
    - Naive Baseline RMSE & MAE ($)
    - Directional Accuracy % for both LSTM and baseline
    - Mean Prediction Bias % (average predicted log return to monitor systematic drift)
    - Benchmark comparison label ('Beats Baseline' vs 'Loses to Baseline')
    """
    if len(y_actual) == 0:
        return {}

    # 1. LSTM Metrics
    rmse = float(np.sqrt(np.mean((y_actual - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_actual - y_pred)))

    actual_diff = y_actual - prev_actual
    pred_diff = y_pred - prev_actual

    actual_dir = np.sign(actual_diff)
    pred_dir = np.sign(pred_diff)
    dir_acc = float(np.mean(actual_dir == pred_dir) * 100.0)

    # 2. Naive Baseline Metrics (tomorrow = today => pred_diff = 0)
    naive_pred = prev_actual
    naive_rmse = float(np.sqrt(np.mean((y_actual - naive_pred) ** 2)))
    naive_mae = float(np.mean(np.abs(y_actual - naive_pred)))
    naive_dir_acc = float(np.mean(actual_dir == 0) * 100.0)

    # 3. Mean Prediction Bias (systematic drift check)
    mean_bias_pct = float(np.mean(pred_log_returns) * 100.0)

    # 4. Comparison Label
    beats_rmse = rmse <= naive_rmse
    rmse_diff = rmse - naive_rmse
    baseline_label = (
        f"Beats Naive Baseline (RMSE ${abs(rmse_diff):.2f} lower)"
        if beats_rmse
        else f"Loses to Naive Baseline (RMSE ${abs(rmse_diff):.2f} higher)"
    )

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "directional_accuracy": round(dir_acc, 2),
        "naive_rmse": round(naive_rmse, 4),
        "naive_mae": round(naive_mae, 4),
        "naive_directional_accuracy": round(naive_dir_acc, 2),
        "mean_prediction_bias_pct": round(mean_bias_pct, 4),
        "beats_baseline": beats_rmse,
        "baseline_label": baseline_label,
        "n_test": len(y_actual),
    }


def generate_trading_signals(
    df: pd.DataFrame,
    test_dates: pd.DatetimeIndex,
    actual_prices: np.ndarray,
    predicted_prices: np.ndarray,
    prev_actual_prices: np.ndarray,
    pred_log_returns: np.ndarray,
    threshold_pct: float = config.DEFAULT_SIGNAL_THRESHOLD_PCT,
    rsi_oversold: float = config.DEFAULT_RSI_OVERSOLD,
    rsi_overbought: float = config.DEFAULT_RSI_OVERBOUGHT,
    rsi_period: int = config.DEFAULT_RSI_PERIOD,
) -> pd.DataFrame:
    """Generate rule-based BUY, SELL, or HOLD signals on the test split.

    Signal Rules:
    - BUY: Predicted next close exceeds current close by more than X%
           AND (RSI < oversold OR a bullish MACD crossover occurred).
    - SELL: Predicted next close is below current close by more than X%
            AND (RSI > overbought OR a bearish MACD crossover occurred).
    - Otherwise: HOLD.
    """
    signals: List[str] = []
    reasons: List[str] = []
    pred_returns: List[float] = []

    rsi_col = f"RSI_{rsi_period}"
    has_rsi = rsi_col in df.columns
    has_macd = "MACD" in df.columns and "MACD_Signal" in df.columns

    for i, date_idx in enumerate(test_dates):
        act_p = actual_prices[i]
        curr_p = prev_actual_prices[i]
        pred_p = predicted_prices[i]

        # Predicted percentage return: (exp(r) - 1) * 100
        pred_ret_pct = (np.exp(pred_log_returns[i]) - 1.0) * 100.0
        pred_returns.append(pred_ret_pct)

        # Indicator state on day date_idx
        rsi_val = float(df.loc[date_idx, rsi_col]) if has_rsi and not np.isnan(df.loc[date_idx, rsi_col]) else 50.0

        bullish_macd = False
        bearish_macd = False
        if has_macd:
            try:
                curr_loc = df.index.get_loc(date_idx)
                if isinstance(curr_loc, int) and curr_loc > 0:
                    curr_macd = float(df["MACD"].iloc[curr_loc])
                    curr_sig = float(df["MACD_Signal"].iloc[curr_loc])
                    prev_macd = float(df["MACD"].iloc[curr_loc - 1])
                    prev_sig = float(df["MACD_Signal"].iloc[curr_loc - 1])

                    # Bullish: MACD crosses above signal line or MACD > signal with upward momentum
                    bullish_macd = (curr_macd >= curr_sig and prev_macd < prev_sig) or (curr_macd >= curr_sig)
                    # Bearish: MACD crosses below signal line or MACD < signal with downward momentum
                    bearish_macd = (curr_macd <= curr_sig and prev_macd > prev_sig) or (curr_macd <= curr_sig)
            except Exception:
                pass

        if pred_ret_pct > threshold_pct and (rsi_val < rsi_oversold or bullish_macd):
            signals.append("BUY")
            trigger = "RSI Oversold" if rsi_val < rsi_oversold else "Bullish MACD"
            reasons.append(f"Predicted +{pred_ret_pct:.3f}% (> {threshold_pct}%) & {trigger}")
        elif pred_ret_pct < -threshold_pct and (rsi_val > rsi_overbought or bearish_macd):
            signals.append("SELL")
            trigger = "RSI Overbought" if rsi_val > rsi_overbought else "Bearish MACD"
            reasons.append(f"Predicted {pred_ret_pct:.3f}% (< -{threshold_pct}%) & {trigger}")
        else:
            signals.append("HOLD")
            reasons.append(f"Predicted return ({pred_ret_pct:+.3f}%) within bounds or neutral confirmation")

    sig_df = pd.DataFrame(
        {
            "Actual_Close": actual_prices,
            "Predicted_Close": predicted_prices,
            "Current_Close": prev_actual_prices,
            "Predicted_Return_Pct": pred_returns,
            "RSI": [float(df.loc[d, rsi_col]) if has_rsi else np.nan for d in test_dates],
            "MACD": [float(df.loc[d, "MACD"]) if has_macd else np.nan for d in test_dates],
            "MACD_Signal": [float(df.loc[d, "MACD_Signal"]) if has_macd else np.nan for d in test_dates],
            "Signal": signals,
            "Reason": reasons,
        },
        index=test_dates,
    )
    sig_df.index.name = "Date"
    return sig_df


def predict_next_day(
    model: Sequential,
    df: pd.DataFrame,
    scaler: MinMaxScaler,
    window_size: int = config.DEFAULT_WINDOW_SIZE,
    threshold_pct: float = config.DEFAULT_SIGNAL_THRESHOLD_PCT,
    rsi_oversold: float = config.DEFAULT_RSI_OVERSOLD,
    rsi_overbought: float = config.DEFAULT_RSI_OVERBOUGHT,
    rsi_period: int = config.DEFAULT_RSI_PERIOD,
) -> Dict[str, Any]:
    """Forecast tomorrow's next-day close price and generate live signal."""
    close_vals = df["Close"].values.astype(np.float64)
    if len(close_vals) < window_size + 1:
        raise ValueError("Insufficient data to predict next day.")

    # Recent log returns: r_t = ln(P_t / P_{t-1})
    log_returns = np.log(close_vals[1:] / close_vals[:-1]).reshape(-1, 1)
    recent_log_ret = log_returns[-window_size:]
    recent_scaled = scaler.transform(recent_log_ret)

    X_input = np.reshape(recent_scaled, (1, window_size, 1)).astype(np.float32)

    pred_scaled = model.predict(X_input, verbose=0)
    pred_log_ret = float(scaler.inverse_transform(pred_scaled)[0, 0])

    last_close = float(close_vals[-1])
    # Reconstruct predicted close dynamically
    pred_price = float(last_close * np.exp(pred_log_ret))
    pred_return_pct = float((np.exp(pred_log_ret) - 1.0) * 100.0)

    rsi_col = f"RSI_{rsi_period}"
    rsi_val = float(df[rsi_col].iloc[-1]) if rsi_col in df.columns else 50.0

    bullish_macd = False
    bearish_macd = False
    if "MACD" in df.columns and "MACD_Signal" in df.columns and len(df) >= 2:
        curr_m = float(df["MACD"].iloc[-1])
        curr_s = float(df["MACD_Signal"].iloc[-1])
        prev_m = float(df["MACD"].iloc[-2])
        prev_s = float(df["MACD_Signal"].iloc[-2])
        bullish_macd = (curr_m >= curr_s and prev_m < prev_s) or (curr_m >= curr_s)
        bearish_macd = (curr_m <= curr_s and prev_m > prev_s) or (curr_m <= curr_s)

    if pred_return_pct > threshold_pct and (rsi_val < rsi_oversold or bullish_macd):
        signal = "BUY"
        reason = f"Predicted +{pred_return_pct:.3f}% gain with confirmation"
    elif pred_return_pct < -threshold_pct and (rsi_val > rsi_overbought or bearish_macd):
        signal = "SELL"
        reason = f"Predicted {pred_return_pct:.3f}% loss with confirmation"
    else:
        signal = "HOLD"
        reason = "Predicted move within neutral threshold or neutral indicators"

    return {
        "current_close": last_close,
        "predicted_next_close": round(pred_price, 2),
        "predicted_return_pct": round(pred_return_pct, 3),
        "signal": signal,
        "reason": reason,
        "rsi": round(rsi_val, 2),
    }


# =============================================================================
# CAPITEX multi-feature, direct multi-horizon engine. The functions above remain
# compatible with the prototype until its API/UI is replaced in Phases 3 and 4.
# =============================================================================
from dataclasses import dataclass
import json
import math
import uuid
from threading import RLock
import indicators as _indicators

_MODEL_LOCK = RLock()


@dataclass
class MultiHorizonDataset:
    features: pd.DataFrame
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    scaled_features: np.ndarray
    raw_returns: np.ndarray
    split: int
    calibration_split: int
    window_size: int
    origins: dict
    X: dict
    y: dict
    prices: np.ndarray


@dataclass
class ForecastBundle:
    model: Any
    dataset: MultiHorizonDataset
    metrics: dict
    residual_sigma: float
    interval_multipliers: np.ndarray
    metadata: dict
    calibration: dict


def parse_horizon(horizon) -> int:
    if isinstance(horizon, bool):
        raise ValueError('Horizon must be 24h/1d, 7d, or 30d.')
    if isinstance(horizon, (int, np.integer)) and horizon in (1, 7, 30):
        return int(horizon)
    aliases = {'24 hours':'24h', 'next 24 hours':'24h', '7 days':'7d', '30 days':'30d'}
    value = str(horizon).strip().lower()
    value = aliases.get(value, value)
    if value not in config.FORECAST_HORIZONS:
        raise ValueError('Horizon must be 24h/1d, 7d, or 30d.')
    return config.FORECAST_HORIZONS[value]


def prepare_multihorizon_dataset(df, window_size=config.DEFAULT_WINDOW_SIZE):
    """80/20 chronological outer split; purged internal calibration block.

    Each origin sees only features through that origin. Training/calibration labels
    remain entirely inside their respective blocks. Final test origins can use
    earlier observed test features, as in a walk-forward evaluation, never labels
    later than that origin. Both scalers fit only the internal fit block.
    """
    if isinstance(window_size, bool) or not isinstance(window_size, (int, np.integer)) or window_size < 5:
        raise ValueError('window_size must be an integer >= 5.')
    frame = _indicators.validate_daily_prices(df)
    features = _indicators.stationary_features(frame)
    n=len(features)
    split=int(n*.8)
    calibration_split=int(split*(1-config.FORECAST_CALIBRATION_FRACTION))
    maximum=config.FORECAST_MAX_HORIZON
    origins={
        'train':np.arange(window_size-1, calibration_split-maximum),
        'calibration':np.arange(calibration_split-1, split-maximum),
        'test':np.arange(split-1, n-1),
    }
    if (len(origins['train'])<30 or len(origins['calibration'])<20 or
            n-split < maximum+10):
        raise ValueError('Insufficient daily history: provide at least 400 daily bars '
                         '(more may be needed for a large window).')
    x_scaler=StandardScaler().fit(features.iloc[:calibration_split])
    y_scaler=StandardScaler().fit(features[['log_return']].iloc[:calibration_split])
    scaled=x_scaler.transform(features).astype(np.float32)
    raw_returns=features.log_return.to_numpy(dtype=float)
    scaled_y=y_scaler.transform(features[['log_return']]).reshape(-1)
    X, y={}, {}
    for part, indices in origins.items():
        X[part]=np.stack([scaled[i-window_size+1:i+1] for i in indices])
        y[part]=np.full((len(indices), maximum), np.nan, dtype=np.float32)
        for row, origin in enumerate(indices):
            length=min(maximum,n-origin-1)
            y[part][row,:length]=scaled_y[origin+1:origin+1+length]
    return MultiHorizonDataset(features,x_scaler,y_scaler,scaled,raw_returns,split,
                               calibration_split,window_size,origins,X,y,
                               frame.loc[features.index,'Close'].to_numpy(dtype=float))


def build_multihorizon_network(window_size=config.DEFAULT_WINDOW_SIZE,
                               learning_rate=config.DEFAULT_LEARNING_RATE):
    """One 30-return head avoids recursive feature fabrication and price shifting."""
    model=Sequential([
        Input(shape=(window_size,len(_indicators.FEATURE_COLUMNS))),
        LSTM(config.LSTM_UNITS_L1,return_sequences=True),
        Dropout(config.LSTM_DROPOUT),
        LSTM(config.LSTM_UNITS_L2),
        Dropout(config.LSTM_DROPOUT),
        Dense(config.LSTM_DENSE_UNITS,activation='tanh'),
        Dense(config.FORECAST_MAX_HORIZON),
    ],name='capitex_return_path')
    model.compile(optimizer=Adam(learning_rate=learning_rate,clipnorm=1.0),loss='mse')
    return model


def _unscale_returns(dataset, values):
    return (np.asarray(values,dtype=float)*float(dataset.y_scaler.scale_[0]) +
            float(dataset.y_scaler.mean_[0]))


def _predict_returns(model, dataset, windows, stochastic=False, batch_size=256):
    # Direct calls avoid tf.data worker-pool creation on desktop Windows.
    outputs=[]
    for begin in range(0,len(windows),batch_size):
        raw=model(windows[begin:begin+batch_size],training=stochastic).numpy()
        if not np.isfinite(raw).all():
            raise ValueError('LSTM produced nonfinite returns.')
        outputs.append(raw)
    return _unscale_returns(dataset,np.concatenate(outputs,axis=0))


def _train_chronologically(model, dataset, epochs, batch_size, callback=None):
    """Mini-batch optimization without shuffling or looking at final test labels."""
    history=[]
    X,y=dataset.X['train'],dataset.y['train']
    if callback is not None:
        callback.set_model(model)
        callback.set_params({'epochs':epochs,'steps':math.ceil(len(X)/batch_size)})
        callback.on_train_begin({})
    for epoch in range(epochs):
        model.reset_metrics()
        if callback is not None: callback.on_epoch_begin(epoch,{})
        losses=[]
        for start in range(0,len(X),batch_size):
            end=start+batch_size
            losses.append(float(model.train_on_batch(X[start:end],y[start:end])))
        history.append(losses[-1])
        if callback is not None: callback.on_epoch_end(epoch,{'loss':history[-1]})
        if getattr(model,'stop_training',False): break
    if callback is not None: callback.on_train_end({'loss':history[-1]})
    return history


def _evaluate_multihorizon(model, dataset, residual_sigma, interval_multipliers):
    """Out-of-sample target-price errors for each horizon, plus persistence baseline."""
    predictions=_predict_returns(model,dataset,dataset.X['test'])
    stochastic=np.cumsum(np.stack([
        _predict_returns(model,dataset,dataset.X['test'],True)
        for _ in range(config.FORECAST_MC_PASSES)]),axis=2)
    mc_centers=np.median(stochastic,axis=0)
    time=np.arange(1,config.FORECAST_MAX_HORIZON+1)
    mc_widths=np.maximum.accumulate(
        np.sqrt(np.var(stochastic,axis=0,ddof=1)+residual_sigma**2*time)*interval_multipliers,
        axis=1)
    origins=dataset.origins['test']
    output={}
    for horizon in (1,7,30):
        valid=origins+horizon < len(dataset.prices)
        selected=origins[valid]
        reference=dataset.prices[selected]
        actual=dataset.prices[selected+horizon]
        predicted_returns=np.sum(predictions[valid,:horizon],axis=1)
        predicted=reference*np.exp(predicted_returns)
        if not np.isfinite(predicted).all():
            raise ValueError('LSTM price reconstruction overflowed.')
        actual_returns=np.log(actual/reference)
        error=predicted-actual
        naive_error=reference-actual
        return_rmse=float(np.sqrt(np.mean((predicted_returns-actual_returns)**2)))
        realized_scale=max(float(np.sqrt(np.mean(actual_returns**2))),1e-8)
        rmse=float(np.sqrt(np.mean(error**2)))
        naive_rmse=float(np.sqrt(np.mean(naive_error**2)))
        mc_center=mc_centers[valid,horizon-1]
        width=mc_widths[valid,horizon-1]
        coverage=float(np.mean(np.abs(actual_returns-mc_center)<=width)*100)
        output[str(horizon)]={
            'rmse':rmse,'mae':float(np.mean(np.abs(error))),
            'directional_accuracy':float(np.mean(np.sign(predicted_returns)==np.sign(actual_returns))*100),
            'naive_rmse':naive_rmse,'naive_mae':float(np.mean(np.abs(naive_error))),
            'naive_directional_accuracy':float(np.mean(actual_returns==0)*100),
            'beats_baseline':bool(rmse<naive_rmse),'test_samples':int(valid.sum()),
            'log_return_rmse':return_rmse,'normalized_test_error':return_rmse/realized_scale,
            'test_interval_coverage_pct':coverage,
            'evaluation':'walk_forward_origin_to_horizon_target_price',
        }
    return output


def _calibrate_intervals(model, dataset):
    """Pre-test residual volatility plus independent MC dropout dispersion.

    Empirical 80th-percentile absolute standardized path errors set conservative
    horizon-specific multipliers. Overlapping windows imply no coverage guarantee.
    """
    deterministic=_predict_returns(model,dataset,dataset.X['calibration'])
    actual=_unscale_returns(dataset,dataset.y['calibration'])
    errors=actual-deterministic
    # One-step residuals have distinct outcome dates; do not pool overlapping paths.
    sigma=max(float(np.std(errors[:,0],ddof=1)),1e-8)
    passes=np.stack([_predict_returns(model,dataset,dataset.X['calibration'],True)
                     for _ in range(config.FORECAST_MC_PASSES)])
    cumulative_passes=np.cumsum(passes,axis=2)
    median=np.median(cumulative_passes,axis=0)
    variance=np.var(cumulative_passes,axis=0,ddof=1)
    time=np.arange(1,config.FORECAST_MAX_HORIZON+1)
    scale=np.sqrt(variance+sigma*sigma*time)
    scores=np.abs(np.cumsum(actual,axis=1)-median)/scale
    n=len(scores)
    rank=min(n,math.ceil((n+1)*config.FORECAST_INTERVAL_LEVEL))-1
    # 1.28155 is the central 80% normal reference multiplier.
    multipliers=np.maximum(1.2815515655446004,np.sort(scores,axis=0)[rank])
    width=scale*multipliers
    coverage=np.mean(np.abs(np.cumsum(actual,axis=1)-median)<=width,axis=0)
    return sigma,multipliers,{
        'source':'internal_pre_test_calibration_block',
        'samples':n,'mc_passes':config.FORECAST_MC_PASSES,
        'nominal_coverage':config.FORECAST_INTERVAL_LEVEL,
        'observed_calibration_coverage':{str(h):float(coverage[h-1]) for h in (1,7,30)},
        'method':'MC_dropout_plus_residual_sigma_sqrt_t_with_empirical_path_error_multiplier',
        'coverage_guaranteed':False,
    }


def _model_fingerprint(dataset, ticker, epochs, batch_size, learning_rate):
    parameters={
        'version':config.FORECAST_CACHE_VERSION,'ticker':str(ticker).strip().upper(),
        'window':dataset.window_size,'epochs':epochs,'batch_size':batch_size,
        'learning_rate':learning_rate,'features':list(_indicators.FEATURE_COLUMNS),
        'outer_train_rows':dataset.split,'calibration_split':dataset.calibration_split,
        'units':[config.LSTM_UNITS_L1,config.LSTM_UNITS_L2,config.LSTM_DENSE_UNITS],
        'dropout':config.LSTM_DROPOUT,'max_horizon':config.FORECAST_MAX_HORIZON,
        'seed':config.RANDOM_SEED,'tensorflow':tf.__version__,
    }
    # Test values never affect training/cache selection. Training revisions do.
    content=pd.util.hash_pandas_object(dataset.features.iloc[:dataset.split],index=True).values.tobytes()
    digest=hashlib.sha256(json.dumps(parameters,sort_keys=True).encode()+content).hexdigest()
    return digest,parameters


def train_multihorizon_model(df, ticker='CUSTOM', *, window_size=config.DEFAULT_WINDOW_SIZE,
                             epochs=config.DEFAULT_EPOCHS, batch_size=config.DEFAULT_BATCH_SIZE,
                             learning_rate=config.DEFAULT_LEARNING_RATE, force_retrain=False,
                             cache_dir=None, callback=None):
    """Train/load the new CAPITEX model; return evaluation and calibration bundle.

    JSON manifests contain only parameters and numeric provenance (no pickle).
    Cache keys hash training content, model shape and training parameters. Files
    are atomically replaced. All final test metrics are recomputed on current data.
    """
    if isinstance(epochs,bool) or not isinstance(epochs,int) or not 1<=epochs<=500:
        raise ValueError('epochs must be an integer from 1 to 500.')
    if isinstance(batch_size,bool) or not isinstance(batch_size,int) or not 1<=batch_size<=1024:
        raise ValueError('batch_size must be an integer from 1 to 1024.')
    if not np.isfinite(learning_rate) or not 0<learning_rate<=.1:
        raise ValueError('learning_rate must be finite and in (0,0.1].')
    dataset=prepare_multihorizon_dataset(df,window_size)
    digest,parameters=_model_fingerprint(dataset,ticker,epochs,batch_size,learning_rate)
    folder=Path(cache_dir or config.MODELS_DIR)
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'capitex_{digest[:24]}.keras'
    manifest=path.with_suffix('.json')
    cached=False
    with _MODEL_LOCK:
        set_reproducible_seeds()
        info={}
        if path.exists() and manifest.exists() and not force_retrain:
            try:
                info=json.loads(manifest.read_text(encoding='utf-8'))
                if info['fingerprint']!=digest:
                    raise ValueError('Manifest mismatch')
                model=load_model(path,compile=False)
                if tuple(model.input_shape[1:])!=(window_size,len(_indicators.FEATURE_COLUMNS)) or model.output_shape[-1]!=30:
                    raise ValueError('Cached model shape mismatch')
                cached=True
            except Exception as exc:
                logger.warning('CAPITEX cache invalid; training afresh: %s',exc)
        if not cached:
            model=build_multihorizon_network(window_size,learning_rate)
            losses=_train_chronologically(model,dataset,epochs,batch_size,callback)
            info={'fingerprint':digest,'parameters':parameters,'epochs_trained':len(losses),
                  'loss_curve':losses,'final_loss':losses[-1],
                  'scalers_fit_through':dataset.features.index[dataset.calibration_split-1].isoformat(),
                  'x_scaler_mean':dataset.x_scaler.mean_.tolist(),
                  'x_scaler_scale':dataset.x_scaler.scale_.tolist(),
                  'y_scaler_mean':dataset.y_scaler.mean_.tolist(),
                  'y_scaler_scale':dataset.y_scaler.scale_.tolist()}
            temporary=folder/f'.capitex_{uuid.uuid4().hex}.keras'
            temporary_json=folder/f'.capitex_{uuid.uuid4().hex}.json'
            try:
                model.save(temporary)
                temporary_json.write_text(json.dumps(info,indent=2,allow_nan=False),encoding='utf-8')
                os.replace(temporary,path)
                os.replace(temporary_json,manifest)
            finally:
                temporary.unlink(missing_ok=True)
                temporary_json.unlink(missing_ok=True)
        sigma,multipliers,calibration=_calibrate_intervals(model,dataset)
        metrics=_evaluate_multihorizon(model,dataset,sigma,multipliers)
    metadata=dict(info,loaded_from_cache=cached,cache_path=str(path),
                  feature_columns=list(_indicators.FEATURE_COLUMNS),
                  train_test_split=[.8,.2],
                  fit_samples=len(dataset.origins['train']),
                  test_start=dataset.features.index[dataset.split].isoformat(),
                  training_end=dataset.features.index[dataset.calibration_split-1].isoformat(),
                  calibration_start=dataset.features.index[dataset.calibration_split].isoformat(),
                  calibration_end=dataset.features.index[dataset.split-1].isoformat(),
                  history_as_of=dataset.features.index[-1].isoformat())
    return ForecastBundle(model,dataset,metrics,sigma,multipliers,metadata,calibration)


def forecast_multihorizon(bundle, horizon='7d', *, current_price=None, ticker='CUSTOM'):
    """30 stochastic forward passes; price path and widening central 80% corridor.

    Confidence is a directional-agreement score damped by normalized held-out
    error, not a calibrated event probability. Median/bounds start at the same
    supplied live price (or last close when none is supplied).
    """
    steps=parse_horizon(horizon)
    dataset=bundle.dataset
    anchor=float(dataset.prices[-1] if current_price is None else current_price)
    if not np.isfinite(anchor) or anchor<=0:
        raise ValueError('current_price must be finite and positive.')
    windows=dataset.scaled_features[-dataset.window_size:][None,...]
    with _MODEL_LOCK:
        passes=np.stack([_predict_returns(bundle.model,dataset,windows,True)[0]
                         for _ in range(config.FORECAST_MC_PASSES)])
    paths=np.cumsum(passes[:,:steps],axis=1)
    center=np.median(paths,axis=0)
    time=np.arange(1,steps+1)
    combined=np.sqrt(np.var(paths,axis=0,ddof=1)+bundle.residual_sigma**2*time)
    widths=np.maximum.accumulate(combined*bundle.interval_multipliers[:steps])
    with np.errstate(over='raise',invalid='raise'):
        median=anchor*np.exp(center)
        lower=anchor*np.exp(center-widths)
        upper=anchor*np.exp(center+widths)
    if not np.isfinite(np.r_[median,lower,upper]).all() or (lower<=0).any():
        raise ValueError('Forecast produced an invalid price corridor.')
    move=float((median[-1]/anchor-1)*100)
    sign=np.sign(center[-1])
    agreement=float(np.mean(np.sign(paths[:,-1])==sign)) if sign else 0.0
    metrics=bundle.metrics[str(steps)]
    confidence=float(np.clip(99*agreement/(1+metrics['normalized_test_error']),0,99))
    last_date=pd.Timestamp(dataset.features.index[-1])
    crypto=config.ASSET_UNIVERSE.get(str(ticker).upper(),{}).get('kind')=='crypto'
    if crypto:
        future=pd.date_range(last_date+pd.Timedelta(days=1),periods=steps,freq='D')
    else:
        future=pd.bdate_range(last_date+pd.Timedelta(days=1),periods=steps)
    return {
        'ticker':str(ticker).strip().upper(),'horizon':steps,
        'horizon_label':{1:'Next 24 Hours (1 Trading Day)',7:'7 Days',30:'30 Days'}[steps],
        'horizon_basis':'calendar_days' if crypto else 'trading_sessions',
        'current_price':anchor,'anchor_source':'provided_live_price' if current_price is not None else 'last_historical_close',
        'historical_as_of':last_date.isoformat(),
        'median_target_price':float(median[-1]),'lower_target_price':float(lower[-1]),
        'upper_target_price':float(upper[-1]),'expected_change_pct':move,
        'outlook':'bullish' if move>.25 else 'bearish' if move<-.25 else 'neutral',
        'confidence':round(confidence,2),'directional_agreement':agreement,
        'confidence_method':'99 * MC_directional_agreement / (1 + normalized_heldout_error)',
        'confidence_is_probability':False,'interval_level':config.FORECAST_INTERVAL_LEVEL,
        'mc_passes':config.FORECAST_MC_PASSES,'residual_sigma':bundle.residual_sigma,
        'corridor':{'steps':list(range(steps+1)),
                    'timestamps':[last_date.isoformat()]+[d.isoformat() for d in future],
                    'median':[anchor]+median.tolist(),'lower':[anchor]+lower.tolist(),
                    'upper':[anchor]+upper.tolist(),
                    'calendar':'daily' if crypto else 'weekday_estimate_exchange_holidays_not_adjusted'},
        'metrics':metrics,'all_horizon_metrics':bundle.metrics,
        'calibration':bundle.calibration,'model':bundle.metadata,
        'limitations':['Estimated 80% corridor; future coverage is not guaranteed.',
                      'Confidence is an error-adjusted model score, not a success probability.',
                      'Return modeling does not ensure the LSTM beats a persistence baseline.'],
    }

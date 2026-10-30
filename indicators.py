"""Technical Indicator Engine for quantitative time-series analysis.

Calculates Simple Moving Average (SMA), Exponential Moving Average (EMA),
Relative Strength Index (RSI) using J. Welles Wilder's smoothing algorithm,
and Moving Average Convergence Divergence (MACD).

Built strictly using Pandas and NumPy only.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd


def compute_sma(series: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Simple Moving Average (SMA).

    Args:
        series: Pandas Series of asset prices (typically Close).
        period: Window size in trading bars (must be >= 1).

    Returns:
        Pandas Series with the SMA values. The initial (period - 1) bars
        are populated with NaN to cleanly reflect the warm-up period.
    """
    if period < 1:
        raise ValueError(f"SMA period must be >= 1, received: {period}")
    return series.rolling(window=period, min_periods=period).mean().rename(f"SMA_{period}")


def compute_ema(series: pd.Series, period: int = 50) -> pd.Series:
    """Calculate Exponential Moving Average (EMA).

    Uses standard smoothing multiplier alpha = 2 / (period + 1).

    Args:
        series: Pandas Series of asset prices (typically Close).
        period: Span/window size for exponential weighting.

    Returns:
        Pandas Series containing the EMA values.
    """
    if period < 1:
        raise ValueError(f"EMA period must be >= 1, received: {period}")
    return series.ewm(span=period, min_periods=period, adjust=False).mean().rename(f"EMA_{period}")


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI) using J. Welles Wilder's smoothing.

    Follows Wilder's original 1978 formulation:
    1. Computes bar-to-bar price changes (diff).
    2. Separates gains and losses.
    3. Seeds the initial average gain/loss using the simple arithmetic mean of the first `period` bars.
    4. Recursively updates subsequent bars:
       AvgGain_t = (AvgGain_{t-1} * (period - 1) + Gain_t) / period
       AvgLoss_t = (AvgLoss_{t-1} * (period - 1) + Loss_t) / period
    5. Computes Relative Strength RS = AvgGain / AvgLoss, and RSI = 100 - (100 / (1 + RS)).

    Args:
        series: Price series (typically Close).
        period: Lookback window (default 14).

    Returns:
        Pandas Series containing RSI values bounded in [0, 100]. Warm-up bars are NaN.
    """
    if period < 1:
        raise ValueError(f"RSI period must be >= 1, received: {period}")

    delta = series.diff()
    gain = delta.clip(lower=0).to_numpy(dtype=np.float64)
    loss = (-delta.clip(upper=0)).to_numpy(dtype=np.float64)

    n = len(series)
    avg_gain = np.full(n, np.nan, dtype=np.float64)
    avg_loss = np.full(n, np.nan, dtype=np.float64)
    rsi_vals = np.full(n, np.nan, dtype=np.float64)

    if n <= period:
        return pd.Series(rsi_vals, index=series.index, name=f"RSI_{period}")

    # Seed initial averages with simple arithmetic mean of the first 'period' bars (excluding index 0 diff)
    initial_gain = np.nanmean(gain[1 : period + 1])
    initial_loss = np.nanmean(loss[1 : period + 1])
    avg_gain[period] = initial_gain
    avg_loss[period] = initial_loss

    # Wilder recursive exponential accumulation
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

    # Calculate RSI values
    for i in range(period, n):
        if np.isnan(avg_gain[i]) or np.isnan(avg_loss[i]):
            continue
        if avg_loss[i] == 0.0:
            rsi_vals[i] = 100.0 if avg_gain[i] > 0.0 else 50.0
        elif avg_gain[i] == 0.0:
            rsi_vals[i] = 0.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi_vals[i] = 100.0 - (100.0 / (1.0 + rs))

    return pd.Series(rsi_vals, index=series.index, name=f"RSI_{period}")


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Moving Average Convergence Divergence (MACD).

    Args:
        series: Price series (typically Close).
        fast: Fast EMA period (default 12).
        slow: Slow EMA period (default 26).
        signal: Signal line EMA period (default 9).

    Returns:
        Tuple of (macd_line, signal_line, histogram).
    """
    if fast >= slow:
        raise ValueError(f"MACD fast period ({fast}) must be strictly less than slow period ({slow})")
    if signal < 1:
        raise ValueError(f"MACD signal period must be >= 1, received: {signal}")

    fast_ema = series.ewm(span=fast, min_periods=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, min_periods=slow, adjust=False).mean()

    macd_line = (fast_ema - slow_ema).rename("MACD_Line")
    signal_line = (
        macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean().rename("MACD_Signal")
    )
    histogram = (macd_line - signal_line).rename("MACD_Hist")

    return macd_line, signal_line, histogram


def add_all_indicators(
    df: pd.DataFrame,
    sma_period: int = 20,
    ema_period: int = 50,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    dropna: bool = False,
) -> pd.DataFrame:
    """Compute and append SMA, EMA, RSI, and MACD components to an OHLCV DataFrame.

    Handles NaN warm-up periods cleanly:
    - If dropna=False (default), retains all dates so candlestick charts and price history
      are unclipped, with indicator lines starting after their respective warm-up intervals.
    - If dropna=True, drops any leading rows that have NaN values in the computed indicators.

    Args:
        df: Cleaned OHLCV DataFrame with standard 'Close' column.
        sma_period: SMA lookback period.
        ema_period: EMA lookback period.
        rsi_period: RSI lookback period.
        macd_fast: MACD fast EMA span.
        macd_slow: MACD slow EMA span.
        macd_signal: MACD signal line EMA span.
        dropna: Whether to drop warm-up rows containing NaNs.

    Returns:
        DataFrame copy with appended indicator columns.
    """
    if df.empty or "Close" not in df.columns:
        raise ValueError("Input DataFrame must be non-empty and contain a 'Close' column.")

    result_df = df.copy()

    # 1. Simple Moving Average
    sma_col = f"SMA_{sma_period}"
    result_df[sma_col] = compute_sma(result_df["Close"], period=sma_period)

    # 2. Exponential Moving Average
    ema_col = f"EMA_{ema_period}"
    result_df[ema_col] = compute_ema(result_df["Close"], period=ema_period)

    # Always expose the CAPITEX set alongside configurable prototype columns.
    for period in (20, 50):
        result_df[f"SMA_{period}"] = compute_sma(result_df["Close"], period)
    for period in (12, 20, 26):
        result_df[f"EMA_{period}"] = compute_ema(result_df["Close"], period)

    # 3. Relative Strength Index (Wilder)
    rsi_col = f"RSI_{rsi_period}"
    result_df[rsi_col] = compute_rsi(result_df["Close"], period=rsi_period)

    # 4. MACD Line, Signal, Histogram
    macd_line, signal_line, histogram = compute_macd(
        result_df["Close"], fast=macd_fast, slow=macd_slow, signal=macd_signal
    )
    result_df["MACD"] = macd_line
    result_df["MACD_Signal"] = signal_line
    result_df["MACD_Hist"] = histogram

    if dropna:
        result_df = result_df.dropna(subset=[sma_col, ema_col, rsi_col, "MACD", "MACD_Signal"])

    return result_df


def analyze_indicator_signals(
    df: pd.DataFrame,
    rsi_period: int = 14,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    sma_period: int = 20,
    ema_period: int = 50,
) -> Dict[str, Any]:
    """Extract latest technical readings and classify crossover/momentum conditions.

    Returns:
        Dictionary containing current values, status labels, and bullish/bearish flags.
    """
    if df.empty:
        return {}

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest

    close = float(latest["Close"])
    sma_col = f"SMA_{sma_period}"
    ema_col = f"EMA_{ema_period}"
    rsi_col = f"RSI_{rsi_period}"

    sma_val = float(latest[sma_col]) if sma_col in latest and not np.isnan(latest[sma_col]) else None
    ema_val = float(latest[ema_col]) if ema_col in latest and not np.isnan(latest[ema_col]) else None
    rsi_val = float(latest[rsi_col]) if rsi_col in latest and not np.isnan(latest[rsi_col]) else None

    macd_curr = float(latest["MACD"]) if "MACD" in latest and not np.isnan(latest["MACD"]) else None
    sig_curr = float(latest["MACD_Signal"]) if "MACD_Signal" in latest and not np.isnan(latest["MACD_Signal"]) else None
    hist_curr = float(latest["MACD_Hist"]) if "MACD_Hist" in latest and not np.isnan(latest["MACD_Hist"]) else None

    macd_prev = float(prev["MACD"]) if "MACD" in prev and not np.isnan(prev["MACD"]) else None
    sig_prev = float(prev["MACD_Signal"]) if "MACD_Signal" in prev and not np.isnan(prev["MACD_Signal"]) else None

    # RSI classification
    if rsi_val is not None:
        if rsi_val < rsi_oversold:
            rsi_status = "Oversold (Bullish Reversal Potential)"
            rsi_tone = "bullish"
        elif rsi_val > rsi_overbought:
            rsi_status = "Overbought (Bearish Exhaustion Potential)"
            rsi_tone = "bearish"
        else:
            rsi_status = "Neutral"
            rsi_tone = "neutral"
    else:
        rsi_status = "Warm-up"
        rsi_tone = "neutral"

    # MACD Crossover detection
    macd_crossover = "Neutral"
    if macd_curr is not None and sig_curr is not None and macd_prev is not None and sig_prev is not None:
        if macd_prev <= sig_prev and macd_curr > sig_curr:
            macd_crossover = "Bullish Crossover (MACD crossed above Signal)"
        elif macd_prev >= sig_prev and macd_curr < sig_curr:
            macd_crossover = "Bearish Crossover (MACD crossed below Signal)"
        elif macd_curr > sig_curr:
            macd_crossover = "Bullish Momentum (MACD above Signal)"
        else:
            macd_crossover = "Bearish Momentum (MACD below Signal)"

    # Trend alignment
    trend_desc = []
    if sma_val is not None:
        trend_desc.append("Above SMA" if close >= sma_val else "Below SMA")
    if ema_val is not None:
        trend_desc.append("Above EMA" if close >= ema_val else "Below EMA")
    trend_status = ", ".join(trend_desc) if trend_desc else "Insufficient bars"

    return {
        "close": close,
        "sma_val": sma_val,
        "ema_val": ema_val,
        "rsi_val": rsi_val,
        "rsi_status": rsi_status,
        "rsi_tone": rsi_tone,
        "macd_line": macd_curr,
        "macd_signal": sig_curr,
        "macd_hist": hist_curr,
        "macd_crossover": macd_crossover,
        "trend_status": trend_status,
    }


FEATURE_COLUMNS = ('log_return', 'rsi_normalized', 'macd_hist_normalized',
                   'price_to_ema20', 'volatility_10d')


def validate_daily_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Reject ambiguous/invalid model inputs instead of silently imputing them."""
    if df.empty or 'Close' not in df:
        raise ValueError('A nonempty daily dataset with Close is required.')
    frame = df.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError('A daily DatetimeIndex is required.')
    if frame.index.hasnans or not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError('Dates must be valid, unique and chronological.')
    if frame.index.normalize().duplicated().any():
        raise ValueError('Use daily bars: intraday rows must be aggregated first.')
    values = pd.to_numeric(frame.Close, errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Close must contain only finite, strictly positive prices.')
    frame['Close'] = values
    return frame


def stationary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal, dimensionless return/regime features; no future filling or scaling."""
    frame = validate_daily_prices(df)
    close = frame.Close
    returns = np.log(close / close.shift(1))
    _, _, hist = compute_macd(close, 12, 26, 9)
    features = pd.DataFrame({
        'log_return': returns,
        'rsi_normalized': compute_rsi(close, 14) / 100.0,
        'macd_hist_normalized': hist / close,
        'price_to_ema20': close / compute_ema(close, 20) - 1.0,
        'volatility_10d': returns.rolling(10, min_periods=10).std(ddof=1),
    }, index=frame.index)
    return features.replace([np.inf, -np.inf], np.nan).dropna()


def technical_summary(df: pd.DataFrame) -> dict:
    """Human-readable colored tags and raw readings for the collapsed Why drawer."""
    frame = add_all_indicators(validate_daily_prices(df))
    last = frame.iloc[-1]
    prev = frame.iloc[-2] if len(frame) > 1 else last
    def number(name):
        value = last.get(name, np.nan)
        return float(value) if np.isfinite(value) else None
    def pill(label, tone='neutral'):
        return {'label': label, 'tone': tone,
                'color': {'bullish': '#14F195', 'bearish': '#FF5C5C',
                          'neutral': '#AB9FF2', 'caution': '#F5BE62'}[tone]}
    tags=[]
    rsi=number('RSI_14')
    if rsi is None:
        tags.append(pill('RSI: Warming Up'))
    else:
        state = 'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'
        tags.append(pill(f'RSI: {state} ({rsi:.1f})', 'caution' if state != 'Neutral' else 'neutral'))
    hist=number('MACD_Hist')
    if hist is not None:
        previous=float(prev.get('MACD_Hist', np.nan))
        if np.isfinite(previous) and previous <= 0 < hist:
            state='Bullish Crossover'
        elif np.isfinite(previous) and previous >= 0 > hist:
            state='Bearish Crossover'
        else:
            state='Bullish Momentum' if hist > 0 else 'Bearish Momentum' if hist < 0 else 'Neutral'
        tags.append(pill(f'MACD: {state}', 'bullish' if hist>0 else 'bearish' if hist<0 else 'neutral'))
    sma=number('SMA_50')
    if sma is not None:
        above=float(last.Close)>=sma
        tags.append(pill(f'Trend: {"Above" if above else "Below"} 50-Day SMA', 'bullish' if above else 'bearish'))
    if number('EMA_12') is not None and number('EMA_26') is not None:
        gap=float(last.EMA_12-last.EMA_26)
        oldgap=float(prev.EMA_12-prev.EMA_26)
        crossed=np.isfinite(oldgap) and ((oldgap<=0<gap) or (oldgap>=0>gap))
        state=('Bullish' if gap>0 else 'Bearish') + (' Crossover' if crossed else ' Alignment')
        tags.append(pill(f'Moving Average: {state}', 'bullish' if gap>0 else 'bearish'))
    vol=np.log(frame.Close/frame.Close.shift(1)).rolling(10).std()
    recent=vol.dropna().tail(252)
    latest_vol=float(recent.iloc[-1]) if len(recent) else None
    if len(recent)>=30:
        state='Low Squeeze' if latest_vol<=recent.quantile(.2) else 'Elevated' if latest_vol>=recent.quantile(.8) else 'Normal'
        tags.append(pill(f'Volatility: {state}', 'caution' if state=='Elevated' else 'neutral'))
    return {'tags':tags, 'values':{name:number(name) for name in
            ('Close','SMA_20','SMA_50','EMA_12','EMA_20','EMA_26','RSI_14','MACD','MACD_Signal','MACD_Hist')},
            'daily_volatility':latest_vol, 'as_of':frame.index[-1].isoformat()}

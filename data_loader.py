"""CAPITEX market data: clean UTC bars, bounded caches, honest as-of lookbacks.

All provider errors are explicit. No synthetic prices are used by this module.
"""
from copy import deepcopy
from datetime import date, datetime, timezone
from functools import wraps
import io
import re
from threading import RLock
from time import monotonic

import numpy as np
import pandas as pd
import yfinance as yf
import config


class MarketDataError(RuntimeError):
    """The provider returned no usable observation."""


def _symbol(ticker):
    value = str(ticker).strip().upper()
    if not re.fullmatch(r'[A-Z0-9^][A-Z0-9.^=\-]{0,24}', value):
        raise ValueError('Invalid ticker symbol.')
    return value


def _utc_now():
    return pd.Timestamp.now(tz='UTC')


def _provider_timestamp(value):
    """yfinance metadata can contain Unix seconds or already-aware timestamps."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return pd.Timestamp(value, unit='s', tz='UTC')
    stamp = pd.Timestamp(value)
    return stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp.tz_convert('UTC')


def _cached(ttl):
    """Bounded TTL cache with striped request locks; execution can bypass it."""
    def decorate(fn):
        cache, lock = {}, RLock()
        stripes = [RLock() for _ in range(32)]
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if kwargs.get('force_refresh', False):
                return fn(*args, **kwargs)
            key = (args, tuple(sorted(kwargs.items())))
            with stripes[hash(key) % len(stripes)]:
                with lock:
                    item = cache.get(key)
                    if item is not None and monotonic()-item[0] < ttl:
                        return deepcopy(item[1])
                result = fn(*args, **kwargs)
                with lock:
                    if len(cache) >= 128:
                        cache.pop(next(iter(cache)))
                    cache[key] = (monotonic(), deepcopy(result))
                return result
        def clear():
            with lock: cache.clear()
        wrapped.clear = clear
        return wrapped
    return decorate


def _standardize_ohlcv_dataframe(frame):
    df = frame.copy()
    if isinstance(df.columns, pd.MultiIndex):
        matches = [i for i in range(df.columns.nlevels)
                   if 'Close' in df.columns.get_level_values(i)]
        if not matches:
            raise ValueError('No OHLCV column level found.')
        df.columns = df.columns.get_level_values(matches[0])
    df = df.rename(columns={c: str(c).strip().title() for c in df.columns})
    if df.columns.duplicated().any():
        raise ValueError('Ambiguous duplicate columns; provide one asset per dataset.')
    missing = set(config.REQUIRED_PRICE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f'Missing required price columns: {sorted(missing)}')
    if isinstance(df.index, pd.RangeIndex):
        raise ValueError('A Date or Timestamp index is required.')
    # Aware timestamps are converted (not stripped); naive CSV timestamps mean UTC.
    df.index = pd.to_datetime(df.index, errors='coerce', utc=True, format='mixed')
    df.index.name = 'Date'
    for col in config.REQUIRED_PRICE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return df[list(config.REQUIRED_PRICE_COLUMNS)]


def clean_stock_data(frame):
    initial = len(frame)
    df = _standardize_ohlcv_dataframe(frame)
    invalid_dates = int(df.index.isna().sum())
    df = df.loc[~df.index.isna()]
    duplicates = int(df.index.duplicated(keep='last').sum())
    df = df.loc[~df.index.duplicated(keep='last')].sort_index()
    nan_before = int(df.isna().sum().sum())
    df = df.ffill().bfill()
    imputed = nan_before - int(df.isna().sum().sum())
    df = df.dropna()
    prices = df[['Open', 'High', 'Low', 'Close']]
    valid = (prices > 0).all(axis=1) & (df.Volume >= 0)
    valid &= df.High >= prices.max(axis=1)
    valid &= df.Low <= prices.min(axis=1)
    invalid_bars = int((~valid).sum())
    df = df.loc[valid].copy()
    report = dict(initial_rows=initial, final_rows=len(df), duplicates_removed=duplicates,
                  invalid_dates=invalid_dates, invalid_bars=invalid_bars,
                  nan_before=nan_before, nan_after=int(df.isna().sum().sum()),
                  imputed_count=imputed, dropped_rows=initial - duplicates - len(df),
                  timezone='UTC', imputation='forward-fill then backward-fill')
    return df, report


def load_from_csv(uploaded_file):
    source = io.BytesIO(uploaded_file) if isinstance(uploaded_file, bytes) else uploaded_file
    try:
        raw = pd.read_csv(source)
    except Exception as exc:
        raise ValueError(f'Failed to parse CSV: {exc}') from exc
    date_col = next((c for c in raw if str(c).strip().lower() in
                     ('date', 'timestamp', 'datetime', 'time')), None)
    if date_col is None:
        raise ValueError('CSV requires Date or Timestamp plus Open, High, Low, Close, Volume.')
    raw = raw.set_index(date_col)
    cleaned, report = clean_stock_data(raw)
    if cleaned.empty:
        raise ValueError('CSV has no usable OHLCV rows after cleaning.')
    return cleaned, report


def _history(ticker, **kwargs):
    try:
        raw = yf.Ticker(ticker).history(auto_adjust=False, actions=False,
                                         timeout=12, raise_errors=True, **kwargs)
        df, report = clean_stock_data(raw)
        if df.empty:
            raise ValueError('No valid bars.')
        df.attrs['cleaning_report'] = report
        return df
    except Exception as exc:
        raise MarketDataError(f'{ticker}: market history unavailable ({exc})') from exc


@_cached(config.HISTORICAL_CACHE_TTL)
def load_market_data(ticker, period='5y'):
    """Return a cleaned daily OHLCV DataFrame; audit report in df.attrs."""
    return _history(_symbol(ticker), period=period, interval='1d')


@_cached(config.HISTORICAL_CACHE_TTL)
def load_from_yfinance(ticker, start, end):
    """Prototype-compatible tuple contract."""
    df = _history(_symbol(ticker), start=str(start), end=str(end), interval='1d')
    return df, df.attrs['cleaning_report']


@_cached(config.LIVE_PRICE_CACHE_TTL)
def get_live_quote(ticker, *, force_refresh=False):
    """Latest provider bar; stocks compare previous close, crypto rolling 24 hours.

    Quotes can be delayed by Yahoo. Market status uses provider session boundaries,
    including holidays/early closes; unknown is never guessed to be open.
    """
    ticker = _symbol(ticker)
    now = _utc_now()
    asset = yf.Ticker(ticker)
    interval = '1m'
    try:
        raw = asset.history(period='5d', interval=interval, auto_adjust=False,
                            actions=False, timeout=12, raise_errors=True)
        bars, _ = clean_stock_data(raw)
        if bars.empty:
            raise ValueError('Empty intraday history')
    except Exception:
        interval = '1d'
        bars = _history(ticker, period='1mo', interval='1d')
    bars = bars.loc[bars.index <= now]
    if bars.empty:
        raise MarketDataError(f'{ticker}: no observation at or before current time.')
    observed = bars.index[-1]
    price = float(bars.Close.iloc[-1])
    crypto = config.ASSET_UNIVERSE.get(ticker, {}).get('kind') == 'crypto' or ticker.endswith('-USD')
    if crypto:
        prior = bars.loc[bars.index <= now - pd.Timedelta(days=1)]
        basis = 'rolling_24h'
    else:
        daily = _history(ticker, period='1mo', interval='1d')
        # Daily index is exchange midnight converted to UTC. Previous session must
        # precede the latest intraday session, even when fetched on a weekend.
        prior = daily.loc[daily.index.normalize() < observed.normalize()]
        basis = 'previous_session_close'
    if prior.empty:
        raise MarketDataError(f'{ticker}: no prior reference price.')
    reference = float(prior.Close.iloc[-1])
    market_open, status = (True, 'open_24_7') if crypto else (None, 'unknown')
    if not crypto:
        try:
            regular = asset.get_history_metadata()['currentTradingPeriod']['regular']
            start = _provider_timestamp(regular['start'])
            end = _provider_timestamp(regular['end'])
            market_open = bool(start <= now < end)
            status = 'open' if market_open else 'closed'
        except (KeyError, TypeError, ValueError, AttributeError):
            pass
    age = max(0.0, (now - observed).total_seconds())
    return dict(ticker=ticker, price=price, previous_close=reference,
                change_24h=price-reference, change_24h_pct=(price/reference-1)*100,
                change_basis=basis, reference_timestamp=prior.index[-1].isoformat(),
                timestamp=observed.isoformat(), fetched_at=now.isoformat(),
                market_open=market_open, market_status=status, currency='USD',
                interval=interval, source='Yahoo Finance via yfinance',
                is_fallback=interval == '1d', is_delayed=True, age_seconds=age,
                is_stale=age > (900 if crypto or market_open else 4*86400),
                status_message='Latest available provider bar; quotes may be delayed.')


def get_live_price(ticker):
    """Compatibility adapter: callers must check price > 0 before using it."""
    try:
        return get_live_quote(ticker)
    except (MarketDataError, ValueError) as exc:
        return dict(price=0.0, previous_close=None, is_fallback=True,
                    status_message=str(exc), timestamp=datetime.now(timezone.utc).isoformat(),
                    currency='USD')


@_cached(config.HISTORICAL_CACHE_TTL)
def _lookback_history(ticker):
    # Hourly data is available beyond the 60-day limit of finer Yahoo intervals.
    try:
        return _history(ticker, period='6mo', interval='1h'), '1h'
    except MarketDataError:
        return _history(ticker, period='6mo', interval='1d'), '1d'


def get_90d_history_and_lookback(ticker, days_ago: int):
    """Exact integer calendar-day offset, resolved to latest available prior bar.

    Never interpolate a nonexistent market price or choose a future observation.
    Both requested timestamp and actual bar timestamp are returned. Daily fallback
    is labelled and uses the prior completed session (daily labels mark day start).
    """
    if isinstance(days_ago, bool) or not isinstance(days_ago, (int, np.integer)) or not 1 <= days_ago <= 90:
        raise ValueError('days_ago must be an integer from 1 to 90.')
    ticker = _symbol(ticker)
    now = _utc_now()
    target = now - pd.Timedelta(days=int(days_ago))
    bars, interval = _lookback_history(ticker)
    # Hourly closes become observable after the labelled bar's start.
    available_at = bars.index + pd.Timedelta(hours=1 if interval == '1h' else 24)
    eligible = bars.loc[available_at <= target]
    if eligible.empty:
        raise MarketDataError(f'{ticker}: insufficient history for {days_ago} days ago.')
    actual = eligible.index[-1]
    historical_price = float(eligible.Close.iloc[-1])
    quote = get_live_quote(ticker)
    visible = bars.loc[(bars.index >= now-pd.Timedelta(days=90)) & (bars.index <= now)]
    # Include selected prior bar if a weekend lookback at day 90 precedes window.
    chart = pd.concat([eligible.tail(1), visible]).sort_index()
    chart = chart.loc[~chart.index.duplicated()]
    history = [dict(timestamp=ts.isoformat(), **{c.lower(): float(row[c])
               for c in config.REQUIRED_PRICE_COLUMNS}) for ts, row in chart.iterrows()]
    return dict(ticker=ticker, days_ago=int(days_ago), requested_timestamp=target.isoformat(),
                historical_timestamp=actual.isoformat(), historical_price=historical_price,
                historical_available_at=(actual+pd.Timedelta(hours=1 if interval=='1h' else 24)).isoformat(),
                exact_timestamp_match=actual == target, resolution='last_completed_bar_at_or_before_target',
                interval=interval, current_price=quote['price'], quote=quote,
                change=quote['price']-historical_price,
                change_pct=(quote['price']/historical_price-1)*100, history=history)

"""Stock Market Trend Analyzer and Price Predictor.

Phase 1: Architecture, SQLite Setup, and Data Pipeline.
Phase 2: Technical Indicator Engine & Synchronized Multi-Row Interactive Charting.
Phase 3: LSTM Forecasting Engine & Algorithmic Trading Signal Generator.

Features:
- Persistent SQLite database initialization and wallet seeding
- Live and historical data ingestion from yfinance REST API or user CSV
- Automated data cleaning, gap imputation, and deduplication reporting
- Pure Pandas/NumPy Technical Indicators: SMA, EMA, Wilder RSI, MACD
- 3-Row synchronized interactive Plotly candlestick chart + Matplotlib fallback
- Sequential Keras LSTM network predicting next-day close
- Chronological train/test split with MinMaxScaler fitted strictly on training data
- Model caching to disk (.keras) using SHA-256 parameter hash with force-retrain toggle
- Live training progress bar and loss callback
- Rigorous metrics: RMSE, MAE, Directional Accuracy %
- Hybrid signal generator combining LSTM expected returns with RSI and MACD crossovers
- Actual vs. Predicted price chart overlaid with BUY (green up-triangle) and SELL (red down-triangle) markers
"""

from datetime import date, datetime, timedelta, timezone
import logging
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

import backtester
import chart_plotter
import config
import database
import data_loader
import indicators
import lstm_model
import trading_engine
from formatting import signed_money, local_timestamp, performance_text

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# FastAPI Backend Communication Layer
# ==============================================================================
FASTAPI_BASE_URL = getattr(config, "FASTAPI_BASE_URL", "http://127.0.0.1:8000")


def get_fastapi_health() -> Tuple[bool, Dict[str, Any]]:
    """Check if the FastAPI backend server is accessible and database is operational."""
    try:
        resp = requests.get(f"{FASTAPI_BASE_URL}/api/health", timeout=1.5)
        if resp.status_code == 200:
            return True, resp.json()
        return False, {"error": f"HTTP {resp.status_code}"}
    except Exception as exc:
        return False, {"error": str(exc)}


def api_get(endpoint: str, timeout: float = 3.0) -> Tuple[bool, Any]:
    """Execute GET request against FastAPI backend with error handling."""
    try:
        resp = requests.get(f"{FASTAPI_BASE_URL}{endpoint}", timeout=timeout)
        if resp.status_code == 200:
            return True, resp.json()
        try:
            err = resp.json().get("detail", resp.text)
        except Exception:
            err = resp.text
        return False, err
    except Exception as exc:
        return False, str(exc)


def api_post(endpoint: str, payload: Optional[Dict[str, Any]] = None, timeout: float = 15.0) -> Tuple[bool, Any]:
    """Execute POST request against FastAPI backend with error handling."""
    try:
        resp = requests.post(f"{FASTAPI_BASE_URL}{endpoint}", json=payload or {}, timeout=timeout)
        if resp.status_code == 200:
            return True, resp.json()
        try:
            err = resp.json().get("detail", resp.text)
        except Exception:
            err = resp.text
        return False, err
    except Exception as exc:
        return False, str(exc)


# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Phase 1 branding: bundled vector assets work without a network connection.
st.logo(str(config.LOGO_PATH), size="large")
st.image(str(config.BANNER_PATH), width="stretch")

# Custom CSS for modern financial dashboard styling
st.markdown(
    """
    <style>

    /* Responsive rows: wrap whole cards and words, never individual characters. */
    [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; align-items: stretch; }
    [data-testid="stColumn"] {
        flex: 1 1 16rem !important;
        min-width: min(100%, 16rem) !important;
        max-width: 100% !important;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *,
    [data-testid="stMetricDelta"], [data-testid="stMetricDelta"] *,
    [data-testid="stWidgetLabel"] *, .stButton button p {
        white-space: normal !important;
        word-break: normal !important;
        overflow-wrap: normal !important;
        text-overflow: clip !important;
        overflow: visible !important;
        height: auto !important;
    }
    [data-testid="stMetricLabel"] { min-height: 1.5rem; }
    [data-testid="stMetricValue"] { font-size: clamp(1.1rem, 2vw, 1.4rem) !important; }
    [data-testid="stMetricValue"] * { white-space: nowrap !important; }
    [data-testid="stMetric"] { box-sizing: border-box; width: 100%; }
    .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        word-break: normal !important; overflow-wrap: normal !important;
    }
    .block-container { padding-bottom: 5rem; }

    /* Metric Card Styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.40rem !important;
        white-space: nowrap !important;
        overflow: visible !important;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(28, 33, 40, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        min-width: 0 !important;
        overflow: visible !important;
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(66, 153, 225, 0.4);
        transition: all 0.2s ease-in-out;
    }
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge-live {
        background-color: rgba(46, 160, 67, 0.18);
        color: #3fb950;
        border: 1px solid rgba(63, 185, 80, 0.4);
    }
    .badge-fallback {
        background-color: rgba(210, 153, 34, 0.18);
        color: #d29922;
        border: 1px solid rgba(210, 153, 34, 0.4);
    }
    .badge-db {
        background-color: rgba(56, 139, 253, 0.18);
        color: #58a6ff;
        border: 1px solid rgba(56, 139, 253, 0.4);
    }
    .badge-bullish {
        background-color: rgba(38, 166, 154, 0.18);
        color: #26a69a;
        border: 1px solid rgba(38, 166, 154, 0.4);
    }
    .badge-bearish {
        background-color: rgba(239, 83, 80, 0.18);
        color: #ef5350;
        border: 1px solid rgba(239, 83, 80, 0.4);
    }
    .badge-neutral {
        background-color: rgba(139, 148, 158, 0.18);
        color: #8b949e;
        border: 1px solid rgba(139, 148, 158, 0.4);
    }
    /* Action Signals */
    .signal-buy {
        background-color: rgba(0, 230, 118, 0.2);
        color: #00e676;
        border: 1px solid #00e676;
        font-weight: bold;
        padding: 4px 12px;
        border-radius: 6px;
        display: inline-block;
    }
    .signal-sell {
        background-color: rgba(255, 23, 68, 0.2);
        color: #ff1744;
        border: 1px solid #ff1744;
        font-weight: bold;
        padding: 4px 12px;
        border-radius: 6px;
        display: inline-block;
    }
    .signal-hold {
        background-color: rgba(144, 164, 174, 0.2);
        color: #b0bec5;
        border: 1px solid #90a4ae;
        font-weight: bold;
        padding: 4px 12px;
        border-radius: 6px;
        display: inline-block;
    }
    /* Footer Styling */
    .app-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: rgba(13, 17, 23, 0.95);
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        color: #8b949e;
        text-align: center;
        padding: 8px 16px;
        font-size: 0.85rem;
        z-index: 999;
        backdrop-filter: blur(8px);
    }
    /* Prevent content overlap with persistent footer */
    .main .block-container {
        padding-bottom: 75px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Database Initialization (Safe on every rerun)
# ==============================================================================
try:
    database.init_db()
    wallets = database.get_all_wallets()
    human_cash = next((w["cash"] for w in wallets if w["wallet_type"] == config.WALLET_TYPE_HUMAN), 0.0)
    ai_cash = next((w["cash"] for w in wallets if w["wallet_type"] == config.WALLET_TYPE_AI), 0.0)
    db_status_ok = True
except Exception as db_err:
    logger.error(f"Failed to initialize database: {db_err}")
    st.error(f"Database Initialization Warning: {db_err}")
    db_status_ok = False
    human_cash = 0.0
    ai_cash = 0.0

# ==============================================================================
# Session State Management
# ==============================================================================
# Data & Core State
if "stock_df" not in st.session_state:
    st.session_state.stock_df = None
if "cleaning_report" not in st.session_state:
    st.session_state.cleaning_report = None
if "active_ticker" not in st.session_state:
    st.session_state.active_ticker = config.DEFAULT_TICKER
if "data_source" not in st.session_state:
    st.session_state.data_source = "yfinance"
if "start_date" not in st.session_state:
    st.session_state.start_date = date.today() - timedelta(days=config.DEFAULT_LOOKBACK_YEARS * 365)
if "end_date" not in st.session_state:
    st.session_state.end_date = date.today()

# Technical Indicator State
if "sma_period" not in st.session_state:
    st.session_state.sma_period = config.DEFAULT_SMA_PERIOD
if "ema_period" not in st.session_state:
    st.session_state.ema_period = config.DEFAULT_EMA_PERIOD
if "rsi_period" not in st.session_state:
    st.session_state.rsi_period = config.DEFAULT_RSI_PERIOD
if "rsi_oversold" not in st.session_state:
    st.session_state.rsi_oversold = config.DEFAULT_RSI_OVERSOLD
if "rsi_overbought" not in st.session_state:
    st.session_state.rsi_overbought = config.DEFAULT_RSI_OVERBOUGHT
if "macd_fast" not in st.session_state:
    st.session_state.macd_fast = config.DEFAULT_MACD_FAST
if "macd_slow" not in st.session_state:
    st.session_state.macd_slow = config.DEFAULT_MACD_SLOW
if "macd_signal" not in st.session_state:
    st.session_state.macd_signal = config.DEFAULT_MACD_SIGNAL
if "chart_engine" not in st.session_state:
    st.session_state.chart_engine = "Plotly (Interactive)"

# Phase 3 LSTM & Signal State
if "lstm_window_size" not in st.session_state:
    st.session_state.lstm_window_size = config.DEFAULT_WINDOW_SIZE
if "lstm_epochs" not in st.session_state:
    st.session_state.lstm_epochs = config.DEFAULT_EPOCHS
if "lstm_batch_size" not in st.session_state:
    st.session_state.lstm_batch_size = config.DEFAULT_BATCH_SIZE
if "lstm_learning_rate" not in st.session_state:
    st.session_state.lstm_learning_rate = config.DEFAULT_LEARNING_RATE
if "signal_threshold_pct" not in st.session_state:
    st.session_state.signal_threshold_pct = config.DEFAULT_SIGNAL_THRESHOLD_PCT
if "lstm_results" not in st.session_state:
    st.session_state.lstm_results = None


# ==============================================================================
# Sidebar: Parameters & Controls
# ==============================================================================
with st.sidebar:
    st.title("CAPITEX")
    st.caption("Markets. Clarity. Possibility.")

    # 1. Data Ingestion
    st.markdown("---")
    st.subheader("1. Data Ingestion Source")

    source_choice = st.radio(
        "Select Pipeline Mode:",
        options=["Yahoo Finance (REST API)", "CSV File Upload"],
        index=0 if st.session_state.data_source == "yfinance" else 1,
        help="Fetch historical OHLCV from Yahoo Finance REST API or ingest an arbitrary local CSV.",
    )

    if source_choice == "Yahoo Finance (REST API)":
        st.session_state.data_source = "yfinance"

        ticker_mode = st.selectbox(
            "Ticker Universe:",
            options=config.DEFAULT_TICKERS + ["Custom Ticker..."],
            index=0,
        )
        if ticker_mode == "Custom Ticker...":
            chosen_ticker = st.text_input(
                "Enter Symbol (e.g. NVDA, SPY, QQQ):",
                value="NVDA",
            ).strip().upper()
        else:
            chosen_ticker = ticker_mode

        st.markdown("**Date Presets:**")
        col_p1, col_p2, col_p3 = st.columns(3)
        if col_p1.button("1 Year", use_container_width=True):
            st.session_state.start_date = date.today() - timedelta(days=365)
            st.session_state.end_date = date.today()
        if col_p2.button("2 Years", use_container_width=True):
            st.session_state.start_date = date.today() - timedelta(days=730)
            st.session_state.end_date = date.today()
        if col_p3.button("5 Years", use_container_width=True):
            st.session_state.start_date = date.today() - timedelta(days=1825)
            st.session_state.end_date = date.today()

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            start_date_input = st.date_input(
                "Start Date",
                value=st.session_state.start_date,
                max_value=date.today(),
            )
        with col_d2:
            end_date_input = st.date_input(
                "End Date",
                value=st.session_state.end_date,
                max_value=date.today(),
            )

        st.session_state.start_date = start_date_input
        st.session_state.end_date = end_date_input

        fetch_clicked = st.button("📥 Fetch & Clean Data", type="primary", use_container_width=True)

        if fetch_clicked or st.session_state.stock_df is None:
            if start_date_input >= end_date_input:
                st.error("Start Date must precede End Date.")
            else:
                with st.spinner(f"Fetching {chosen_ticker} from Yahoo Finance..."):
                    try:
                        df, report = data_loader.load_from_yfinance(
                            chosen_ticker,
                            start=start_date_input,
                            end=end_date_input,
                        )
                        st.session_state.stock_df = df
                        st.session_state.cleaning_report = report
                        st.session_state.active_ticker = chosen_ticker
                        st.session_state.lstm_results = None  # Reset stale model predictions
                        st.success(f"Loaded {len(df)} bars for {chosen_ticker}")
                    except Exception as err:
                        st.error(f"Data Fetch Error: {err}")
                        logger.exception("Error loading yfinance data")

    else:
        st.session_state.data_source = "csv"
        uploaded_file = st.file_uploader(
            "Upload OHLCV CSV File",
            type=["csv"],
            help="CSV must contain Date, Open, High, Low, Close, Volume columns.",
        )

        use_sample = st.checkbox("Or use bundled sample dataset (AAPL 6-Month)", value=uploaded_file is None)

        if st.button("📥 Process & Clean CSV", type="primary", use_container_width=True):
            target_data = None
            if uploaded_file is not None:
                target_data = uploaded_file
                st.session_state.active_ticker = uploaded_file.name.split(".")[0].upper()
            elif use_sample:
                sample_path = config.DATA_DIR / "sample_stock_data.csv"
                if sample_path.exists():
                    target_data = str(sample_path)
                    st.session_state.active_ticker = "AAPL (Sample)"
                else:
                    st.error("Bundled sample CSV not found on disk.")

            if target_data is not None:
                with st.spinner("Parsing and cleaning CSV dataset..."):
                    try:
                        df, report = data_loader.load_from_csv(target_data)
                        st.session_state.stock_df = df
                        st.session_state.cleaning_report = report
                        st.session_state.lstm_results = None
                        st.success(f"Parsed {len(df)} records from CSV.")
                    except Exception as csv_err:
                        st.error(f"CSV Ingestion Error: {csv_err}")
                        logger.exception("Error processing CSV file")

    # 2. Technical Indicators Controls (Phase 2)
    st.markdown("---")
    st.subheader("2. Technical Indicators (Module 2)")

    with st.expander("⚙️ Indicator Parameters & Visuals", expanded=False):
        sma_p = st.slider("SMA Period", min_value=5, max_value=200, value=st.session_state.sma_period, step=1)
        ema_p = st.slider("EMA Period", min_value=5, max_value=200, value=st.session_state.ema_period, step=1)
        rsi_p = st.slider("RSI Period", min_value=2, max_value=50, value=st.session_state.rsi_period, step=1)

        col_rsi_t1, col_rsi_t2 = st.columns(2)
        with col_rsi_t1:
            rsi_os = st.number_input(
                "Oversold", min_value=10.0, max_value=45.0, value=float(st.session_state.rsi_oversold), step=1.0
            )
        with col_rsi_t2:
            rsi_ob = st.number_input(
                "Overbought", min_value=55.0, max_value=90.0, value=float(st.session_state.rsi_overbought), step=1.0
            )

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            macd_f = st.number_input("Fast", min_value=2, max_value=50, value=st.session_state.macd_fast, step=1)
        with col_m2:
            macd_s = st.number_input("Slow", min_value=5, max_value=100, value=st.session_state.macd_slow, step=1)
        with col_m3:
            macd_sig = st.number_input("Signal", min_value=2, max_value=30, value=st.session_state.macd_signal, step=1)

        st.session_state.sma_period = int(sma_p)
        st.session_state.ema_period = int(ema_p)
        st.session_state.rsi_period = int(rsi_p)
        st.session_state.rsi_oversold = float(rsi_os)
        st.session_state.rsi_overbought = float(rsi_ob)
        st.session_state.macd_fast = int(macd_f)
        st.session_state.macd_slow = int(macd_s)
        st.session_state.macd_signal = int(macd_sig)

        chart_choice = st.radio(
            "Renderer Engine:",
            options=["Plotly (Interactive)", "Matplotlib (Static Fallback)"],
            index=0 if st.session_state.chart_engine == "Plotly (Interactive)" else 1,
        )
        st.session_state.chart_engine = chart_choice

    # 3. LSTM Hyperparameters Controls (Phase 3)
    st.markdown("---")
    st.subheader("3. LSTM Forecasting & Signals (Module 3)")

    with st.expander("🧠 LSTM Model Hyperparameters", expanded=True):
        win_size = st.slider(
            "Window Size (Lookback bars):",
            min_value=10,
            max_value=120,
            value=st.session_state.lstm_window_size,
            step=5,
            help="Number of consecutive preceding trading bars fed into the LSTM.",
        )
        epochs_cnt = st.slider(
            "Training Epochs:",
            min_value=5,
            max_value=100,
            value=st.session_state.lstm_epochs,
            step=5,
            help="Total passes over the training dataset.",
        )
        batch_sz = st.selectbox(
            "Batch Size:",
            options=[16, 32, 64, 128],
            index=[16, 32, 64, 128].index(st.session_state.lstm_batch_size)
            if st.session_state.lstm_batch_size in [16, 32, 64, 128]
            else 1,
        )
        lr_val = st.selectbox(
            "Learning Rate (Adam):",
            options=[0.0005, 0.001, 0.002, 0.005],
            index=[0.0005, 0.001, 0.002, 0.005].index(st.session_state.lstm_learning_rate)
            if st.session_state.lstm_learning_rate in [0.0005, 0.001, 0.002, 0.005]
            else 1,
        )
        thresh_pct = st.slider(
            "Signal Move Threshold (%):",
            min_value=0.01,
            max_value=1.00,
            value=float(st.session_state.signal_threshold_pct),
            step=0.01,
            help="Minimum predicted next-day return magnitude required to trigger BUY/SELL.",
        )
        force_retrain_chk = st.checkbox("Force Retrain (Bypass .keras Cache)", value=False)

        st.session_state.lstm_window_size = int(win_size)
        st.session_state.lstm_epochs = int(epochs_cnt)
        st.session_state.lstm_batch_size = int(batch_sz)
        st.session_state.lstm_learning_rate = float(lr_val)
        st.session_state.signal_threshold_pct = float(thresh_pct)

        train_model_btn = st.button("🚀 Train / Load LSTM Model", type="primary", use_container_width=True)

    # 4. Database Status
    st.markdown("---")
    st.subheader("Database Status")
    if db_status_ok:
        st.markdown(
            f'<div class="status-badge badge-db">SQLite: market_sim.db (Active)</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"👤 Human Wallet: **${human_cash:,.2f}**")
        st.caption(f"🤖 AI Wallet: **${ai_cash:,.2f}**")
    else:
        st.error("Database connection failed.")


# ==============================================================================
# Main Dashboard Header & Progress Roadmap
# ==============================================================================
st.title(f"{config.APP_ICON} {config.APP_TITLE}")
st.caption("Explore markets, understand forecasts, and practice virtual trading.")

st.markdown("---")

# Active Ticker Live Quote Header
current_ticker = st.session_state.active_ticker.split()[0]
live_quote = data_loader.get_live_price(current_ticker)

header_col1, header_col2, header_col3, header_col4 = st.columns([1.5, 1, 1, 1])

with header_col1:
    st.subheader(f"Active Symbol: **{st.session_state.active_ticker}**")
    if live_quote.get("is_fallback"):
        st.markdown(
            f'<span class="status-badge badge-fallback">● {live_quote.get("status_message", "Fallback Price")}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span class="status-badge badge-live">● {live_quote.get("status_message", "Live Stream")}</span>',
            unsafe_allow_html=True,
        )

with header_col2:
    price_val = live_quote.get("price", 0.0)
    prev_close = live_quote.get("previous_close")
    delta_str = None
    if prev_close and prev_close > 0:
        chg = price_val - prev_close
        pct = (chg / prev_close) * 100
        delta_str = f"{chg:+.2f} ({pct:+.2f}%)"

    st.metric(
        label=f"Current Price ({live_quote.get('currency', 'USD')})",
        value=f"${price_val:,.2f}",
        delta=delta_str,
    )

with header_col3:
    if st.session_state.stock_df is not None:
        total_bars = len(st.session_state.stock_df)
        st.metric(label="Cleaned Bars", value=f"{total_bars:,} rows")
    else:
        st.metric(label="Data Status", value="Awaiting Data")

with header_col4:
    if st.session_state.stock_df is not None and not st.session_state.stock_df.empty:
        start_ts = st.session_state.stock_df.index[0].strftime("%Y-%m-%d")
        end_ts = st.session_state.stock_df.index[-1].strftime("%Y-%m-%d")
        st.metric(label="Historical Span", value=f"{start_ts}", delta=f"to {end_ts}")
    else:
        st.metric(label="Date Range", value="N/A")


# ==============================================================================
# Technical Indicator Computation (Phase 2 Engine)
# ==============================================================================
df_with_indicators = None
tech_signals: Dict[str, Any] = {}

if st.session_state.stock_df is not None and not st.session_state.stock_df.empty:
    try:
        df_with_indicators = indicators.add_all_indicators(
            st.session_state.stock_df,
            sma_period=st.session_state.sma_period,
            ema_period=st.session_state.ema_period,
            rsi_period=st.session_state.rsi_period,
            macd_fast=st.session_state.macd_fast,
            macd_slow=st.session_state.macd_slow,
            macd_signal=st.session_state.macd_signal,
            dropna=False,
        )
        tech_signals = indicators.analyze_indicator_signals(
            df_with_indicators,
            rsi_period=st.session_state.rsi_period,
            rsi_oversold=st.session_state.rsi_oversold,
            rsi_overbought=st.session_state.rsi_overbought,
            sma_period=st.session_state.sma_period,
            ema_period=st.session_state.ema_period,
        )
    except Exception as ind_err:
        st.error(f"Error computing technical indicators: {ind_err}")
        logger.exception("Failed to compute technical indicators")


# ==============================================================================
# LSTM Training & Evaluation Execution
# ==============================================================================
def run_lstm_pipeline(force_retrain: bool = False) -> None:
    """Execute LSTM training or cache loading and compute metrics/signals."""
    if df_with_indicators is None or df_with_indicators.empty:
        st.error("Cannot train LSTM: No data loaded.")
        return

    try:
        w_size = st.session_state.lstm_window_size
        e_epochs = st.session_state.lstm_epochs
        b_size = st.session_state.lstm_batch_size
        l_rate = st.session_state.lstm_learning_rate
        t_thresh = st.session_state.signal_threshold_pct

        # 1. Prepare Dataset (train/test split on log returns, scaler on train only, sliding windows)
        (
            X_train,
            y_train,
            X_test,
            y_test,
            scaler,
            train_dates,
            test_dates,
            test_prev_actual,
            test_actual_close,
        ) = lstm_model.prepare_lstm_dataset(
            df_with_indicators,
            window_size=w_size,
            train_split=config.DEFAULT_TRAIN_SPLIT_RATIO,
        )

        # 2. Setup training progress containers
        train_container = st.container()
        with train_container:
            status_box = st.empty()
            pbar = st.progress(0.0)

        cb = lstm_model.StreamlitTrainingCallback(
            total_epochs=e_epochs,
            progress_bar=pbar,
            status_text=status_box,
        )

        # 3. Train or Load Model
        model, was_cached, info = lstm_model.train_or_load_lstm(
            ticker=st.session_state.active_ticker,
            start_date=df_with_indicators.index[0].date(),
            end_date=df_with_indicators.index[-1].date(),
            X_train=X_train,
            y_train=y_train,
            window_size=w_size,
            epochs=e_epochs,
            batch_size=b_size,
            learning_rate=l_rate,
            force_retrain=force_retrain,
            callback=cb,
        )

        pbar.progress(1.0)
        status_box.empty()

        # 4. Generate Test Predictions (Inverse Transformed Log Returns & Reconstructed Close)
        pred_scaled = model.predict(X_test, verbose=0)
        pred_log_ret = scaler.inverse_transform(pred_scaled).flatten()
        # Reconstruct predicted close as: previous_actual_close * exp(predicted_return)
        y_test_pred = test_prev_actual * np.exp(pred_log_ret)
        y_test_actual = test_actual_close

        # 5. Evaluate Metrics against Naive Baseline & Track Prediction Bias
        metrics = lstm_model.evaluate_forecast_with_baseline(
            y_actual=y_test_actual,
            y_pred=y_test_pred,
            prev_actual=test_prev_actual,
            pred_log_returns=pred_log_ret,
        )

        # 6. Generate Algorithmic Signals on Test Split
        signals_df = lstm_model.generate_trading_signals(
            df=df_with_indicators,
            test_dates=test_dates,
            actual_prices=y_test_actual,
            predicted_prices=y_test_pred,
            prev_actual_prices=test_prev_actual,
            pred_log_returns=pred_log_ret,
            threshold_pct=t_thresh,
            rsi_oversold=st.session_state.rsi_oversold,
            rsi_overbought=st.session_state.rsi_overbought,
            rsi_period=st.session_state.rsi_period,
        )

        # 7. Predict Next Trading Day
        next_day_forecast = lstm_model.predict_next_day(
            model=model,
            df=df_with_indicators,
            scaler=scaler,
            window_size=w_size,
            threshold_pct=t_thresh,
            rsi_oversold=st.session_state.rsi_oversold,
            rsi_overbought=st.session_state.rsi_overbought,
            rsi_period=st.session_state.rsi_period,
        )

        st.session_state.lstm_results = {
            "model": model,
            "was_cached": was_cached,
            "info": info,
            "metrics": metrics,
            "test_dates": test_dates,
            "y_test_actual": y_test_actual,
            "y_test_pred": y_test_pred,
            "signals_df": signals_df,
            "next_day_forecast": next_day_forecast,
        }

        if was_cached:
            st.success(f"Loaded pre-trained model from disk cache: {info['cache_path']}")
        else:
            st.success(f"Trained fresh LSTM model ({e_epochs} epochs) and saved to cache.")

    except Exception as exc:
        st.error(f"LSTM Training / Execution Error: {exc}")
        logger.exception("Error executing LSTM pipeline")


# Check if user clicked train button in sidebar
if "train_model_btn" in locals() and train_model_btn:
    run_lstm_pipeline(force_retrain=force_retrain_chk)


# ==============================================================================
# Main Dashboard Tabs
# ==============================================================================
tab_arena, tab_backtest, tab_charts, tab_lstm, tab_data, tab_stats, tab_cleaning = st.tabs(
    [
        "⚔️ AI vs. Human Arena",
        "🔬 Bulk CSV Strategy Backtester",
        "📈 Interactive Technical Chart",
        "🤖 LSTM Forecasting & Signal Generator",
        "📋 Cleaned OHLCV & Indicators Table",
        "📊 Descriptive Statistics",
        "🧹 Data Cleaning Pipeline Report",
    ]
)

# ------------------------------------------------------------------------------
# TAB 1: AI VS. HUMAN PAPER TRADING ARENA (Phase 4)
# ------------------------------------------------------------------------------
with tab_arena:
    st.subheader("⚔️ AI vs. Human Paper Trading Arena")
    st.caption("Real-time virtual trading arena powered by FastAPI, SQLite WAL persistence, and Dynamic Trailing Stops.")

    # 1. Backend Connectivity & Health Check
    backend_online, health_data = get_fastapi_health()
    top_badge_col, top_btn_col = st.columns([2.5, 1.5])

    with top_badge_col:
        if backend_online:
            st.markdown(
                f'<span class="status-badge badge-live">● FastAPI Backend Online ({FASTAPI_BASE_URL})</span> '
                f'<span class="status-badge badge-db">● SQLite WAL Mode (Concurrency Safe)</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<span class="status-badge badge-bearish">● FastAPI Backend Offline ({FASTAPI_BASE_URL})</span> '
                f'<span class="status-badge badge-fallback">● Launch via `python run.py` to route orders through FastAPI</span>',
                unsafe_allow_html=True,
            )

    with top_btn_col:
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            refresh_arena = st.button("🔄 Refresh Arena", use_container_width=True)
        with btn_c2:
            reset_expander = st.expander("⚠️ Reset Wallets", expanded=False)
            with reset_expander:
                st.caption("Reset both Human and AI wallets back to $100,000 cash and clear ledger.")
                confirm_reset = st.checkbox("Confirm wipe", key="confirm_reset_wallets")
                if st.button("Proceed Reset", type="secondary", disabled=not confirm_reset, use_container_width=True):
                    if backend_online:
                        ok, r_res = api_post("/portfolio/reset")
                        if ok:
                            st.success(r_res.get("message", "Wallets reset successfully!"))
                            st.rerun()
                        else:
                            st.error(f"Reset failed: {r_res}")
                    else:
                        database.reset_wallets(starting_cash=config.STARTING_CASH)
                        st.success("Wallets reset directly in database.")
                        st.rerun()

    # 2. Portfolio Data Retrieval
    if backend_online:
        ok_h, h_port = api_get("/portfolio/human")
        ok_a, a_port = api_get("/portfolio/ai")
        if not ok_h or not isinstance(h_port, dict):
            h_port = trading_engine.get_portfolio_summary("human")
        if not ok_a or not isinstance(a_port, dict):
            a_port = trading_engine.get_portfolio_summary("ai")
    else:
        h_port = trading_engine.get_portfolio_summary("human")
        a_port = trading_engine.get_portfolio_summary("ai")

    human_nw = float(h_port.get("net_worth", config.STARTING_CASH))
    human_cash = float(h_port.get("cash", config.STARTING_CASH))
    human_unrealized = float(h_port.get("unrealized_pnl", 0.0))
    human_roi = float(h_port.get("benchmark_roi_pct", 0.0))
    human_holdings = h_port.get("holdings", [])
    human_snaps = h_port.get("snapshots", [])

    ai_nw = float(a_port.get("net_worth", config.STARTING_CASH))
    ai_cash = float(a_port.get("cash", config.STARTING_CASH))
    ai_unrealized = float(a_port.get("unrealized_pnl", 0.0))
    ai_roi = float(a_port.get("benchmark_roi_pct", 0.0))
    ai_holdings = a_port.get("holdings", [])
    ai_snaps = a_port.get("snapshots", [])

    human_performance, human_details = performance_text(h_port)
    ai_performance, ai_details = performance_text(a_port)

    # 3. Side-by-Side Comparison Metric Cards (3x2 Grid for full number visibility)
    st.markdown("#### 📊 Real-Time Portfolio Performance Metrics")
    m_r1_c1, m_r1_c2, m_r1_c3 = st.columns(3)

    h_delta_nw = human_nw - config.STARTING_CASH
    ai_delta_nw = ai_nw - config.STARTING_CASH

    with m_r1_c1:
        st.metric(
            label="👤 Human Net Worth",
            value=f"${human_nw:,.2f}",
            delta=f"{signed_money(h_delta_nw)} ({human_roi:+.2f}%)",
        )
    with m_r1_c2:
        st.metric(
            label="👤 Human Cash",
            value=f"${human_cash:,.2f}",
            delta=f"Unrealized: {signed_money(human_unrealized)}",
        )
    with m_r1_c3:
        st.metric(
            label="👤 Human Performance",
            value=human_performance,
            delta=human_details,
            delta_color="off",
        )

    m_r2_c1, m_r2_c2, m_r2_c3 = st.columns(3)
    with m_r2_c1:
        st.metric(
            label="🤖 AI Bot Net Worth",
            value=f"${ai_nw:,.2f}",
            delta=f"{signed_money(ai_delta_nw)} ({ai_roi:+.2f}%)",
        )
    with m_r2_c2:
        st.metric(
            label="🤖 AI Bot Cash",
            value=f"${ai_cash:,.2f}",
            delta=f"Unrealized: {signed_money(ai_unrealized)}",
        )
    with m_r2_c3:
        st.metric(
            label="🤖 AI Bot Performance",
            value=ai_performance,
            delta=ai_details,
            delta_color="off",
        )

    # 4. Net-Worth-Over-Time Plotly Chart
    st.markdown("---")
    nw_fig = chart_plotter.plot_arena_net_worth_history(human_snaps, ai_snaps)
    st.plotly_chart(nw_fig, use_container_width=True)

    # 5. Split-Screen Layout: Left = Human, Right = AI
    st.markdown("---")
    col_human, col_ai = st.columns(2)

    # --- LEFT COLUMN: HUMAN PORTFOLIO & ORDER DESK ---
    with col_human:
        st.markdown("### 👤 Human Portfolio & Execution Desk")

        with st.form("human_order_form"):
            st.markdown("##### 📝 Place Manual Paper Order")
            h_order_ticker = st.selectbox(
                "Select Asset:",
                options=config.DEFAULT_TICKERS + ["Custom..."],
                index=0,
                key="human_ticker_select",
            )
            if h_order_ticker == "Custom...":
                h_target_ticker = st.text_input("Enter Ticker Symbol:", value="NVDA", key="human_custom_ticker").strip().upper()
            else:
                h_target_ticker = h_order_ticker

            # Get quote for price calculation
            h_live_p = trading_engine.get_cached_live_price(h_target_ticker)
            if h_live_p <= 0:
                h_quote = data_loader.get_live_price(h_target_ticker)
                h_live_p = float(h_quote.get("price", 0.0))

            st.caption(f"Market Price: **\\${h_live_p:,.2f}** | Available Cash: **\\${human_cash:,.2f}**")

            h_action = st.radio("Order Type:", options=["BUY", "SELL"], horizontal=True, key="human_action_radio")
            h_qty = st.number_input("Shares Quantity:", min_value=0.01, max_value=100000.0, value=10.0, step=1.0, key="human_qty_input")

            h_est_cost = h_qty * h_live_p
            if h_action == "BUY":
                st.caption(f"Estimated Purchase Outlay: **\\${h_est_cost:,.2f}**")
            else:
                st.caption(f"Estimated Gross Proceeds: **\\${h_est_cost:,.2f}**")

            col_sl, col_tp = st.columns(2)
            with col_sl:
                h_sl_pct = st.slider(
                    "Dynamic Trailing Stop (%):",
                    min_value=0.5,
                    max_value=25.0,
                    value=float(config.DEFAULT_STOP_LOSS_PCT),
                    step=0.5,
                    key="human_sl_slider",
                    help="Auto-liquidates position if price drops below (peak_price * (1 - stop_loss_pct)).",
                )
            with col_tp:
                h_tp_pct = st.slider(
                    "Take-Profit Target (%):",
                    min_value=1.0,
                    max_value=50.0,
                    value=float(config.DEFAULT_TAKE_PROFIT_PCT),
                    step=0.5,
                    key="human_tp_slider",
                    help="Auto-liquidates position if price gains above (avg_buy_price * (1 + take_profit_pct)).",
                )

            h_submit_btn = st.form_submit_button("🚀 Submit Order", type="primary", use_container_width=True)

        if h_submit_btn:
            if not backend_online:
                st.error("FastAPI backend is offline. Please launch via `python run.py` to route orders through the REST API.")
            else:
                order_payload = {
                    "wallet_type": "human",
                    "ticker": h_target_ticker,
                    "action": h_action,
                    "quantity": float(h_qty),
                    "stop_loss_pct": float(h_sl_pct),
                    "take_profit_pct": float(h_tp_pct),
                }
                with st.spinner(f"Submitting {h_action} order for {h_qty} {h_target_ticker} via FastAPI..."):
                    ok, res = api_post("/trade/execute", order_payload)
                    if ok:
                        st.success(f"✅ {res.get('message', 'Trade executed successfully!')}")
                        st.rerun()
                    else:
                        st.error(f"❌ Trade Rejected: {res}")

        # Human Risk Evaluation Trigger
        if st.button("🛡️ Check Human Risk Triggers", use_container_width=True, key="human_risk_btn"):
            if backend_online:
                ok, risk_res = api_post("/trade/check-risk/human")
                if ok:
                    count = risk_res.get("triggered_count", 0)
                    if count > 0:
                        st.warning(f"⚠️ {count} risk trigger(s) activated! Positions closed.")
                    else:
                        st.info("✅ All Human holdings within acceptable risk boundaries.")
                    st.rerun()
                else:
                    st.error(f"Risk evaluation error: {risk_res}")
            else:
                st.warning("Backend offline; start via `python run.py`.")

        # Human Active Holdings Table
        st.markdown("##### 💼 Active Human Holdings")
        if human_holdings:
            h_df = pd.DataFrame(human_holdings)
            h_disp_cols = [
                "ticker", "quantity", "avg_buy_price", "current_price",
                "peak_price", "market_value", "unrealized_pnl", "trailing_stop_price", "take_profit_price"
            ]
            h_df_disp = h_df[[c for c in h_disp_cols if c in h_df.columns]].copy()
            h_df_disp.columns = [
                "Symbol", "Shares", "Avg Buy ($)", "Live ($)",
                "Peak ($)", "Market Val ($)", "Unrealized ($)", "Trail Stop ($)", "Take Profit ($)"
            ]
            st.dataframe(h_df_disp.style.format({
                "Shares": "{:,.4f}",
                "Avg Buy ($)": "${:,.2f}",
                "Live ($)": "${:,.2f}",
                "Peak ($)": "${:,.2f}",
                "Market Val ($)": "${:,.2f}",
                "Unrealized ($)": signed_money,
                "Trail Stop ($)": "${:,.2f}",
                "Take Profit ($)": "${:,.2f}",
            }), use_container_width=True)
        else:
            st.info("No open holdings in Human portfolio. Place a BUY order above.")

    # --- RIGHT COLUMN: AI BOT PORTFOLIO & CONTROLS ---
    with col_ai:
        st.markdown("### 🤖 Autonomous AI Agent Desk")

        st.markdown("##### ⚙️ AI Bot Execution Engine")
        ai_target_ticker = st.selectbox(
            "Target Asset for AI Trading Step:",
            options=config.DEFAULT_TICKERS + ["Custom..."],
            index=0,
            key="ai_ticker_select",
        )
        if ai_target_ticker == "Custom...":
            ai_target_ticker = st.text_input("Enter Ticker:", value="AAPL", key="ai_custom_ticker").strip().upper()

        ai_live_p = trading_engine.get_cached_live_price(ai_target_ticker)
        if ai_live_p <= 0:
            ai_quote = data_loader.get_live_price(ai_target_ticker)
            ai_live_p = float(ai_quote.get("price", 0.0))

        st.caption(f"Market Price: **\\${ai_live_p:,.2f}** | Available Cash: **\\${ai_cash:,.2f}**")

        ai_tranche_pct = st.slider(
            "Tranche Sizing (% of cash per BUY signal):",
            min_value=2.0,
            max_value=50.0,
            value=float(config.DEFAULT_AI_POSITION_PCT),
            step=1.0,
            help="AI allocates this fraction of available cash on each BUY signal (scaled tranche buying).",
            key="ai_tranche_slider",
        )

        col_ai_sl, col_ai_tp = st.columns(2)
        with col_ai_sl:
            ai_sl_pct = st.slider(
                "AI Trailing Stop (%):",
                min_value=0.5,
                max_value=25.0,
                value=float(config.DEFAULT_STOP_LOSS_PCT),
                step=0.5,
                key="ai_sl_slider",
            )
        with col_ai_tp:
            ai_tp_pct = st.slider(
                "AI Take-Profit (%):",
                min_value=1.0,
                max_value=50.0,
                value=float(config.DEFAULT_TAKE_PROFIT_PCT),
                step=0.5,
                key="ai_tp_slider",
            )

        # AI Action Buttons
        ai_btn_col1, ai_btn_col2 = st.columns(2)
        with ai_btn_col1:
            run_ai_step_btn = st.button("🤖 Run AI Step Now", type="primary", use_container_width=True)
        with ai_btn_col2:
            check_ai_risk_btn = st.button("🛡️ Check AI Risk Triggers", use_container_width=True)

        if run_ai_step_btn:
            if not backend_online:
                st.error("FastAPI backend is offline. Launch via `python run.py` to run AI steps.")
            else:
                ai_step_payload = {
                    "ticker": ai_target_ticker,
                    "tranche_pct": float(ai_tranche_pct),
                    "threshold_pct": float(st.session_state.signal_threshold_pct),
                    "stop_loss_pct": float(ai_sl_pct),
                    "take_profit_pct": float(ai_tp_pct),
                }
                with st.spinner(f"Evaluating LSTM forecast & executing AI step for {ai_target_ticker}..."):
                    ok, res = api_post("/trade/ai-step", ai_step_payload)
                    if ok:
                        sig = res.get("signal", "UNKNOWN")
                        sig_badge = f'<span class="signal-{"buy" if sig=="BUY" else "sell" if sig=="SELL" else "hold"}">{sig}</span>'
                        st.markdown(f"**AI Signal Evaluated:** {sig_badge}", unsafe_allow_html=True)
                        st.success(f"🤖 {res.get('message', 'AI step completed.')}")
                        st.rerun()
                    else:
                        st.error(f"❌ AI Step Error: {res}")

        if check_ai_risk_btn:
            if backend_online:
                ok, risk_res = api_post("/trade/check-risk/ai")
                if ok:
                    count = risk_res.get("triggered_count", 0)
                    if count > 0:
                        st.warning(f"⚠️ {count} AI risk trigger(s) activated! Positions closed.")
                    else:
                        st.info("✅ All AI positions within risk boundaries.")
                    st.rerun()
                else:
                    st.error(f"Risk evaluation error: {risk_res}")
            else:
                st.warning("Backend offline; start via `python run.py`.")

        # AI Active Holdings Table
        st.markdown("##### 💼 Active AI Bot Holdings")
        if ai_holdings:
            ai_df = pd.DataFrame(ai_holdings)
            ai_disp_cols = [
                "ticker", "quantity", "avg_buy_price", "current_price",
                "peak_price", "market_value", "unrealized_pnl", "trailing_stop_price", "take_profit_price"
            ]
            ai_df_disp = ai_df[[c for c in ai_disp_cols if c in ai_df.columns]].copy()
            ai_df_disp.columns = [
                "Symbol", "Shares", "Avg Buy ($)", "Live ($)",
                "Peak ($)", "Market Val ($)", "Unrealized ($)", "Trail Stop ($)", "Take Profit ($)"
            ]
            st.dataframe(ai_df_disp.style.format({
                "Shares": "{:,.4f}",
                "Avg Buy ($)": "${:,.2f}",
                "Live ($)": "${:,.2f}",
                "Peak ($)": "${:,.2f}",
                "Market Val ($)": "${:,.2f}",
                "Unrealized ($)": signed_money,
                "Trail Stop ($)": "${:,.2f}",
                "Take Profit ($)": "${:,.2f}",
            }), use_container_width=True)
        else:
            st.info("No open holdings in AI portfolio. Run an AI step above.")

    # 6. Full Trade Ledger Table (Combined / Filterable)
    st.markdown("---")
    st.markdown("#### 📜 Unified Trade Ledger")

    ledger_filter = st.radio(
        "Ledger Filter:",
        options=["All Wallets", "Human Only", "AI Only"],
        horizontal=True,
        key="ledger_filter_radio",
    )

    w_filter = None
    if ledger_filter == "Human Only":
        w_filter = "human"
    elif ledger_filter == "AI Only":
        w_filter = "ai"

    ledger_rows = database.get_trade_ledger(wallet_type=w_filter, limit=100)
    if ledger_rows:
        ledger_df = pd.DataFrame(ledger_rows)
        ledger_df["timestamp"] = ledger_df["timestamp"].map(local_timestamp)
        ledger_disp_cols = [c for c in ["timestamp", "wallet_type", "ticker", "action", "quantity", "price", "realized_pnl", "execution_label"] if c in ledger_df.columns]
        ledger_df_disp = ledger_df[ledger_disp_cols].copy()
        ledger_df_disp.columns = ["Timestamp (IST)", "Wallet", "Symbol", "Action", "Shares", "Exec Price ($)", "Realized P&L ($)", "Fill Source"][:len(ledger_disp_cols)]
        st.dataframe(
            ledger_df_disp.style.format({
                "Shares": "{:,.4f}",
                "Exec Price ($)": "${:,.2f}",
                "Realized P&L ($)": signed_money,
            }),
            use_container_width=True,
        )

        # Full Trade Ledger Download Button
        full_ledger_all = database.get_trade_ledger(limit=5000)
        if full_ledger_all:
            full_df = pd.DataFrame(full_ledger_all)
            full_df["timestamp"] = full_df["timestamp"].map(local_timestamp)
            st.download_button(
                label="📥 Download Full Trade Ledger CSV",
                data=full_df.to_csv(index=False).encode("utf-8"),
                file_name="trade_ledger_export.csv",
                mime="text/csv",
                key="download_full_ledger_btn",
                use_container_width=True,
            )

    else:
        st.info("Trade ledger is currently empty. Execute paper trades or run an AI step to populate.")

# ------------------------------------------------------------------------------
# TAB 2: BULK CSV STRATEGY BACKTESTER (Phase 5)
# ------------------------------------------------------------------------------
with tab_backtest:
    st.subheader("🔬 Bulk CSV Strategy Backtester & Alpha Benchmark")
    st.caption(
        "Walk-forward out-of-sample backtesting simulator: stress-tests the autonomous AI trading strategy "
        "across a multi-ticker universe against the passive Buy & Hold benchmark with zero look-ahead bias."
    )

    # State initialization
    if "backtest_results" not in st.session_state:
        st.session_state.backtest_results = None
    if "backtest_skipped" not in st.session_state:
        st.session_state.backtest_skipped = None
    if "backtest_tickers_text" not in st.session_state:
        st.session_state.backtest_tickers_text = "AAPL, MSFT, NVDA, GOOGL, AMZN, TSLA, META"

    # Controls Section
    with st.expander("⚙️ Backtest Universe & Simulation Parameters", expanded=True):
        bt_col1, bt_col2 = st.columns([1.2, 1])

        with bt_col1:
            st.markdown("##### 📁 Ingest Ticker Universe")
            upload_c1, upload_c2 = st.columns([1.5, 1])
            with upload_c1:
                uploaded_csv = st.file_uploader(
                    "Upload Tickers CSV (with 'Ticker' or 'Symbol' column):",
                    type=["csv"],
                    key="backtest_csv_uploader",
                )
            with upload_c2:
                st.write("")
                st.write("")
                if st.button("📂 Load Sample CSV", help="Loads data/sample_tickers.csv (AAPL, MSFT, NVDA, GOOGL, AMZN, TSLA, META)", use_container_width=True):
                    try:
                        sample_df = pd.read_csv("data/sample_tickers.csv")
                        col_name = "Ticker" if "Ticker" in sample_df.columns else sample_df.columns[0]
                        sample_tickers = sample_df[col_name].dropna().astype(str).str.strip().tolist()
                        st.session_state.backtest_tickers_text = ", ".join(sample_tickers)
                        st.success(f"Loaded {len(sample_tickers)} benchmark symbols from data/sample_tickers.csv")
                    except Exception as exc:
                        st.error(f"Error loading sample CSV: {exc}")

            if uploaded_csv is not None:
                try:
                    user_df = pd.read_csv(uploaded_csv)
                    cand_cols = [c for c in user_df.columns if c.strip().lower() in ("ticker", "symbol", "tickers", "symbols")]
                    target_col = cand_cols[0] if cand_cols else user_df.columns[0]
                    parsed_tickers = user_df[target_col].dropna().astype(str).str.strip().tolist()
                    st.session_state.backtest_tickers_text = ", ".join(parsed_tickers)
                    st.caption(f"Loaded {len(parsed_tickers)} tickers from uploaded CSV ('{target_col}' column).")
                except Exception as up_err:
                    st.error(f"Error parsing uploaded CSV: {up_err}")

            raw_input_text = st.text_area(
                "Active Ticker Universe (comma-separated):",
                value=st.session_state.backtest_tickers_text,
                height=75,
                key="backtest_text_area",
            )
            st.session_state.backtest_tickers_text = raw_input_text

        with bt_col2:
            st.markdown("##### ⚙️ Strategy & LSTM Hyperparameters")
            bt_lookback = st.slider("Historical Lookback (Years):", min_value=1, max_value=10, value=5, step=1)
            bt_epochs = st.slider(
                "Bulk LSTM Epochs:",
                min_value=5,
                max_value=30,
                value=10,
                step=5,
                help="Fast-training parameter for multi-ticker batches; pre-cached .keras models are automatically loaded.",
            )
            b_sl1, b_sl2, b_sl3 = st.columns(3)
            with b_sl1:
                bt_tranche = st.slider("Tranche (%):", 2.0, 50.0, float(config.DEFAULT_AI_POSITION_PCT), 1.0, help="% cash per BUY")
            with b_sl2:
                bt_sl = st.slider("Trail Stop (%):", 0.5, 20.0, float(config.DEFAULT_STOP_LOSS_PCT), 0.5, help="Trailing stop from peak")
            with b_sl3:
                bt_tp = st.slider("Target (%):", 1.0, 50.0, float(config.DEFAULT_TAKE_PROFIT_PCT), 0.5, help="Take-profit target")

        run_bt_btn = st.button("🚀 Run Walk-Forward Bulk Backtest", type="primary", use_container_width=True)

    if run_bt_btn:
        active_tickers = [t.strip().upper() for t in st.session_state.backtest_tickers_text.split(",") if t.strip()]
        if not active_tickers:
            st.warning("Please specify at least one valid ticker symbol.")
        else:
            total_cnt = len(active_tickers)
            results_list = []
            skipped_list = []

            prog_container = st.container()
            with prog_container:
                bt_pbar = st.progress(0.0, text=f"Preparing walk-forward backtest for {total_cnt} tickers...")
                status_box = st.empty()

            backend_on, _ = get_fastapi_health()

            for idx, t_sym in enumerate(active_tickers):
                pct = idx / total_cnt
                bt_pbar.progress(pct, text=f"Evaluating {t_sym} ({idx + 1}/{total_cnt})...")
                status_box.info(f"⏳ Ingesting data & simulating walk-forward strategy for {t_sym}...")

                try:
                    if backend_on:
                        payload = {
                            "ticker": t_sym,
                            "lookback_years": int(bt_lookback),
                            "epochs": int(bt_epochs),
                            "tranche_pct": float(bt_tranche),
                            "stop_loss_pct": float(bt_sl),
                            "take_profit_pct": float(bt_tp),
                        }
                        ok, api_resp = api_post("/api/backtest/ticker", payload, timeout=120.0)
                        if ok and isinstance(api_resp, dict) and api_resp.get("success"):
                            results_list.append(api_resp["result"])
                        else:
                            # Fallback to direct computation
                            res_direct = backtester.run_single_ticker_backtest(
                                ticker=t_sym,
                                lookback_years=int(bt_lookback),
                                epochs=int(bt_epochs),
                                tranche_pct=float(bt_tranche),
                                stop_loss_pct=float(bt_sl),
                                take_profit_pct=float(bt_tp),
                            )
                            results_list.append(res_direct)
                    else:
                        res_direct = backtester.run_single_ticker_backtest(
                            ticker=t_sym,
                            lookback_years=int(bt_lookback),
                            epochs=int(bt_epochs),
                            tranche_pct=float(bt_tranche),
                            stop_loss_pct=float(bt_sl),
                            take_profit_pct=float(bt_tp),
                        )
                        results_list.append(res_direct)
                except Exception as exc:
                    logger.warning(f"Backtest error on {t_sym}: {exc}")
                    skipped_list.append({"ticker": t_sym, "reason": str(exc)})

            bt_pbar.progress(1.0, text="✅ Bulk backtesting complete!")
            status_box.empty()

            # Sort results by Strategy ROI % descending and assign Rank
            results_list.sort(key=lambda x: x["strategy_roi_pct"], reverse=True)
            for rk, item in enumerate(results_list, start=1):
                item["rank"] = rk

            st.session_state.backtest_results = results_list
            st.session_state.backtest_skipped = skipped_list
            st.rerun()

    # Results Section
    if st.session_state.backtest_results:
        results = st.session_state.backtest_results
        skipped = st.session_state.backtest_skipped or []

        st.markdown("---")
        st.subheader("🏆 Multi-Ticker Walk-Forward Leaderboard & Benchmark")

        # 1. Top Summary Metric Cards
        strat_rois = [r["strategy_roi_pct"] for r in results]
        bh_rois = [r["buy_and_hold_roi_pct"] for r in results]
        win_rates = [r["win_rate_pct"] for r in results]

        avg_strat_roi = float(np.mean(strat_rois)) if strat_rois else 0.0
        avg_bh_roi = float(np.mean(bh_rois)) if bh_rois else 0.0
        avg_alpha = avg_strat_roi - avg_bh_roi
        avg_win_rate = float(np.mean(win_rates)) if win_rates else 0.0
        best_performer = results[0] if results else None
        outperforming_count = sum(1 for r in results if r["alpha_pct"] > 0)

        sum_c1, sum_c2, sum_c3, sum_c4 = st.columns(4)
        with sum_c1:
            if best_performer:
                st.metric(
                    label="🥇 Best Performing Ticker",
                    value=f"{best_performer['ticker']}",
                    delta=f"{best_performer['strategy_roi_pct']:+.2f}% Strategy ROI",
                )
        with sum_c2:
            st.metric(
                label="📊 Average Strategy vs B&H",
                value=f"{avg_strat_roi:+.2f}%",
                delta=f"Alpha: {avg_alpha:+.2f}% (B&H: {avg_bh_roi:+.2f}%)",
            )
        with sum_c3:
            st.metric(
                label="🎯 Average Trade Win Rate",
                value=f"{avg_win_rate:.1f}%",
                delta=f"{outperforming_count}/{len(results)} beat passive benchmark",
                delta_color="off",
            )
        with sum_c4:
            st.metric(
                label="📈 Strategies Evaluated",
                value=f"{len(results)} Tickers",
                delta=f"{len(skipped)} skipped/delisted" if skipped else "100% complete",
                delta_color="off" if not skipped else "inverse",
            )

        # 2. Plotly Grouped Bar Chart
        st.markdown("---")
        bar_fig = chart_plotter.plot_backtest_leaderboard_chart(results)
        st.plotly_chart(bar_fig, use_container_width=True)

        # 3. Sortable Leaderboard Table
        st.markdown("##### 📋 Strategy Outperformance Leaderboard")
        leaderboard_df = pd.DataFrame([
            {
                "Rank": r["rank"],
                "Ticker": r["ticker"],
                "Strategy ROI (%)": r["strategy_roi_pct"],
                "Buy & Hold ROI (%)": r["buy_and_hold_roi_pct"],
                "Outperformance / Alpha (%)": r["alpha_pct"],
                "Win Rate (%)": r["win_rate_pct"],
                "Max Drawdown (%)": r["max_drawdown_pct"],
                "Total Trades": r["total_trades"],
                "RMSE ($)": r["rmse"],
                "Directional Acc (%)": r["directional_accuracy_pct"],
            }
            for r in results
        ])

        st.dataframe(
            leaderboard_df.style.format({
                "Strategy ROI (%)": "{:+.2f}%",
                "Buy & Hold ROI (%)": "{:+.2f}%",
                "Outperformance / Alpha (%)": "{:+.2f}%",
                "Win Rate (%)": "{:.1f}%",
                "Max Drawdown (%)": "{:.2f}%",
                "Total Trades": "{:d}",
                "RMSE ($)": "${:.2f}",
                "Directional Acc (%)": "{:.1f}%",
            }),
            use_container_width=True,
        )

        # 4. Download Leaderboard CSV Button
        lb_csv_bytes = leaderboard_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Backtest Leaderboard CSV",
            data=lb_csv_bytes,
            file_name="backtest_leaderboard.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_leaderboard_btn",
        )

        # 5. Skipped / Failed Tickers Collapsible Log
        if skipped:
            with st.expander(f"⚠️ Skipped / Failed Symbols ({len(skipped)})", expanded=False):
                st.dataframe(pd.DataFrame(skipped), use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: TECHNICAL INDICATORS & CANDLESTICK CHART (Phase 2)
# ------------------------------------------------------------------------------
with tab_charts:
    if df_with_indicators is not None and not df_with_indicators.empty:
        st.subheader(f"Technical Analysis Telemetry: {st.session_state.active_ticker}")

        card1, card2, card3 = st.columns(3)
        with card1:
            sma_val = tech_signals.get("sma_val")
            ema_val = tech_signals.get("ema_val")
            close_val = tech_signals.get("close", 0.0)
            trend_txt = tech_signals.get("trend_status", "N/A")

            sma_disp = f"${sma_val:.2f}" if sma_val else "Warm-up"
            ema_disp = f"${ema_val:.2f}" if ema_val else "Warm-up"

            st.metric(
                label=f"SMA ({st.session_state.sma_period}) & EMA ({st.session_state.ema_period})",
                value=sma_disp,
                delta=f"EMA: {ema_disp}",
            )
            is_bullish_trend = sma_val and close_val >= sma_val
            badge_class = "badge-bullish" if is_bullish_trend else "badge-bearish"
            st.markdown(f'<span class="status-badge {badge_class}">{trend_txt}</span>', unsafe_allow_html=True)

        with card2:
            rsi_val = tech_signals.get("rsi_val")
            rsi_status = tech_signals.get("rsi_status", "N/A")
            rsi_tone = tech_signals.get("rsi_tone", "neutral")
            rsi_disp = f"{rsi_val:.2f}" if rsi_val else "Warm-up"

            st.metric(
                label=f"Wilder RSI ({st.session_state.rsi_period})",
                value=rsi_disp,
                delta=f"Range: {int(st.session_state.rsi_oversold)} - {int(st.session_state.rsi_overbought)}",
            )
            rsi_badge = (
                "badge-bullish"
                if rsi_tone == "bullish"
                else ("badge-bearish" if rsi_tone == "bearish" else "badge-neutral")
            )
            st.markdown(f'<span class="status-badge {rsi_badge}">{rsi_status}</span>', unsafe_allow_html=True)

        with card3:
            macd_l = tech_signals.get("macd_line")
            macd_s = tech_signals.get("macd_signal")
            crossover_txt = tech_signals.get("macd_crossover", "Neutral")
            macd_disp = f"{macd_l:.2f}" if macd_l else "Warm-up"
            sig_disp = f"Signal: {macd_s:.2f}" if macd_s else "Warm-up"

            st.metric(
                label=f"MACD ({st.session_state.macd_fast}, {st.session_state.macd_slow}, {st.session_state.macd_signal})",
                value=macd_disp,
                delta=sig_disp,
            )
            is_bull_macd = macd_l and macd_s and macd_l >= macd_s
            macd_badge = "badge-bullish" if is_bull_macd else "badge-bearish"
            st.markdown(f'<span class="status-badge {macd_badge}">{crossover_txt}</span>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.session_state.chart_engine == "Plotly (Interactive)":
            fig_plotly = chart_plotter.plot_technical_chart_plotly(
                df_with_indicators,
                ticker=st.session_state.active_ticker,
                sma_period=st.session_state.sma_period,
                ema_period=st.session_state.ema_period,
                rsi_period=st.session_state.rsi_period,
                rsi_oversold=st.session_state.rsi_oversold,
                rsi_overbought=st.session_state.rsi_overbought,
                macd_fast=st.session_state.macd_fast,
                macd_slow=st.session_state.macd_slow,
                macd_signal=st.session_state.macd_signal,
            )
            st.plotly_chart(fig_plotly, use_container_width=True)
        else:
            fig_mpl = chart_plotter.plot_technical_chart_matplotlib(
                df_with_indicators,
                ticker=st.session_state.active_ticker,
                sma_period=st.session_state.sma_period,
                ema_period=st.session_state.ema_period,
                rsi_period=st.session_state.rsi_period,
                rsi_oversold=st.session_state.rsi_oversold,
                rsi_overbought=st.session_state.rsi_overbought,
                macd_fast=st.session_state.macd_fast,
                macd_slow=st.session_state.macd_slow,
                macd_signal=st.session_state.macd_signal,
            )
            st.pyplot(fig_mpl)
            plt.close(fig_mpl)

        with st.expander("📚 Indicator Equations & Trading Methodology", expanded=False):
            st.markdown(
                f"""
                - **SMA ({st.session_state.sma_period})**: Moving average over {st.session_state.sma_period} bars.
                - **EMA ({st.session_state.ema_period})**: Exponential moving average over {st.session_state.ema_period} bars.
                - **Wilder RSI ({st.session_state.rsi_period})**: J. Welles Wilder Jr. algorithm with threshold bounds [{int(st.session_state.rsi_oversold)}, {int(st.session_state.rsi_overbought)}].
                - **MACD ({st.session_state.macd_fast}, {st.session_state.macd_slow}, {st.session_state.macd_signal})**: MACD Line, Signal Line, and Momentum Histogram.
                """
            )
    else:
        st.info("Select a ticker in the sidebar or upload a CSV file to render technical charts.")


# ------------------------------------------------------------------------------
# TAB 2: LSTM FORECASTING ENGINE & SIGNAL GENERATOR (Phase 3 Deliverable)
# ------------------------------------------------------------------------------
with tab_lstm:
    st.subheader(f"LSTM Neural Forecasting & Signal Generation: {st.session_state.active_ticker}")

    if st.session_state.lstm_results is None:
        st.info(
            "💡 **No trained LSTM model loaded in session.** "
            "Click the button below or in the sidebar to train a fresh neural network or load the cached `.keras` model."
        )
        if st.button("🚀 Train / Load LSTM Model Now", type="primary"):
            run_lstm_pipeline(force_retrain=False)
            st.rerun()

    else:
        res = st.session_state.lstm_results
        metrics = res["metrics"]
        next_day = res["next_day_forecast"]
        signals_df = res["signals_df"]
        info = res["info"]

        # Cache Status Header Badge
        cache_status_text = "Disk Cached (.keras)" if res["was_cached"] else "Newly Trained"
        st.markdown(
            f'<div class="status-badge badge-db">Model: {cache_status_text} | Window: {st.session_state.lstm_window_size} bars | Epochs: {st.session_state.lstm_epochs} | LR: {st.session_state.lstm_learning_rate}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # 1. Live Next-Day Forecast Hero Card
        st.markdown("### 🔮 Live Next-Day Forecast & Actionable Signal")
        sig_val = next_day["signal"]
        sig_class = (
            "signal-buy"
            if sig_val == "BUY"
            else ("signal-sell" if sig_val == "SELL" else "signal-hold")
        )

        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            st.metric(
                label="Latest Known Close",
                value=f"${next_day['current_close']:.2f}",
            )
        with f_col2:
            st.metric(
                label="Predicted Tomorrow Close",
                value=f"${next_day['predicted_next_close']:.2f}",
                delta=f"{next_day['predicted_return_pct']:+.2f}%",
            )
        with f_col3:
            st.markdown(f"**Recommended Signal:**")
            st.markdown(f'<span class="{sig_class}">{sig_val}</span>', unsafe_allow_html=True)
            st.caption(f"Threshold: ±{st.session_state.signal_threshold_pct}%")
        with f_col4:
            st.markdown("**Trigger Reason:**")
            st.caption(next_day["reason"])

        st.markdown("---")

        # 2. Evaluation Metrics Cards (RMSE, MAE, Directional Accuracy %, Naive Baseline & Mean Bias)
        st.markdown("### 📊 Test Split Evaluator Metrics vs. Naive Baseline")

        beats_base = metrics.get("beats_baseline", False)
        base_badge_class = "badge-bullish" if beats_base else "badge-bearish"
        base_label = metrics.get("baseline_label", "Naive Persistence Baseline")
        st.markdown(
            f'<div style="margin-bottom: 12px;"><span class="status-badge {base_badge_class}"><b>Benchmark:</b> {base_label}</span></div>',
            unsafe_allow_html=True,
        )

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)

        with m_col1:
            st.metric(
                label="RMSE (LSTM vs. Naive)",
                value=f"${metrics['rmse']:.2f}",
                delta=f"Naive: ${metrics.get('naive_rmse', 0.0):.2f}",
                delta_color="off",
                help="Root Mean Squared Error of LSTM vs. Naive persistence baseline ('tomorrow close = today close').",
            )
        with m_col2:
            st.metric(
                label="MAE (LSTM vs. Naive)",
                value=f"${metrics['mae']:.2f}",
                delta=f"Naive: ${metrics.get('naive_mae', 0.0):.2f}",
                delta_color="off",
                help="Mean Absolute Error in dollars of LSTM vs. Naive persistence baseline.",
            )
        with m_col3:
            dir_acc = metrics["directional_accuracy"]
            n_bars = metrics.get("n_test", len(signals_df))
            st.metric(
                label="Directional Accuracy %",
                value=f"{dir_acc:.1f}%",
                delta=f"vs 50% baseline, n={n_bars} test bars (low statistical power)",
                delta_color="off",
                help="Share of test days where predicted direction (predicted close vs previous day's actual close) matches actual direction.",
            )
        with m_col4:
            bias_val = metrics.get("mean_prediction_bias_pct", 0.0)
            buy_cnt = int((signals_df["Signal"] == "BUY").sum())
            sell_cnt = int((signals_df["Signal"] == "SELL").sum())
            hold_cnt = int((signals_df["Signal"] == "HOLD").sum())
            st.metric(
                label="Mean Prediction Bias",
                value=f"{bias_val:+.4f}%",
                delta=f"{buy_cnt} BUY / {sell_cnt} SELL / {hold_cnt} HOLD",
                delta_color="off",
                help="Average predicted daily return. Values close to 0.0000% confirm the model does not suffer from systematic upward or downward drift.",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. Actual vs. Predicted Chart with Buy/Sell Markers
        st.markdown("### 📈 Test Period: Actual vs. Predicted Price with Signals")
        if st.session_state.chart_engine == "Plotly (Interactive)":
            fig_fc = chart_plotter.plot_forecast_and_signals_plotly(
                dates=res["test_dates"],
                actual_prices=res["y_test_actual"],
                predicted_prices=res["y_test_pred"],
                signals_df=signals_df,
                ticker=st.session_state.active_ticker,
            )
            st.plotly_chart(fig_fc, use_container_width=True)
        else:
            fig_fc_mpl = chart_plotter.plot_forecast_and_signals_matplotlib(
                dates=res["test_dates"],
                actual_prices=res["y_test_actual"],
                predicted_prices=res["y_test_pred"],
                signals_df=signals_df,
                ticker=st.session_state.active_ticker,
            )
            st.pyplot(fig_fc_mpl)
            plt.close(fig_fc_mpl)

        # 4. Recent Signals Table
        st.markdown("### 📋 Test Split Algorithmic Signals Ledger")
        st.caption("Walk-forward signals generated sequentially without lookahead bias:")

        # Show table with recent signals first
        display_signals = signals_df.sort_index(ascending=False).copy()
        display_signals["Date"] = display_signals.index.strftime("%Y-%m-%d")

        st.dataframe(
            display_signals[
                [
                    "Date",
                    "Signal",
                    "Actual_Close",
                    "Predicted_Close",
                    "Predicted_Return_Pct",
                    "RSI",
                    "MACD",
                    "Reason",
                ]
            ].style.format(
                {
                    "Actual_Close": "${:.2f}",
                    "Predicted_Close": "${:.2f}",
                    "Predicted_Return_Pct": "{:+.2f}%",
                    "RSI": "{:.2f}",
                    "MACD": "{:.3f}",
                }
            ),
            use_container_width=True,
            height=320,
        )

        csv_sig = signals_df.to_csv().encode("utf-8")
        st.download_button(
            label="⬇️ Download Full Signals Ledger CSV",
            data=csv_sig,
            file_name=f"{current_ticker}_lstm_signals_{date.today().isoformat()}.csv",
            mime="text/csv",
        )


# ------------------------------------------------------------------------------
# TAB 3: CLEANED DATASET & INDICATORS TABLE
# ------------------------------------------------------------------------------
with tab_data:
    if df_with_indicators is not None and not df_with_indicators.empty:
        st.subheader(f"OHLCV + Technical Indicators Table: {st.session_state.active_ticker}")

        format_dict = {
            "Open": "{:.2f}",
            "High": "{:.2f}",
            "Low": "{:.2f}",
            "Close": "{:.2f}",
            "Volume": "{:,.0f}",
            f"SMA_{st.session_state.sma_period}": "{:.2f}",
            f"EMA_{st.session_state.ema_period}": "{:.2f}",
            f"RSI_{st.session_state.rsi_period}": "{:.2f}",
            "MACD": "{:.3f}",
            "MACD_Signal": "{:.3f}",
            "MACD_Hist": "{:.3f}",
        }
        active_format = {k: v for k, v in format_dict.items() if k in df_with_indicators.columns}

        st.dataframe(
            df_with_indicators.style.format(active_format, na_rep="NaN"),
            use_container_width=True,
            height=400,
        )

        csv_download = df_with_indicators.to_csv().encode("utf-8")
        st.download_button(
            label=f"⬇️ Download {st.session_state.active_ticker} Data & Indicators CSV",
            data=csv_download,
            file_name=f"{current_ticker}_indicators_{date.today().isoformat()}.csv",
            mime="text/csv",
        )
    else:
        st.info("No data available.")


# ------------------------------------------------------------------------------
# TAB 4: DESCRIPTIVE STATISTICS
# ------------------------------------------------------------------------------
with tab_stats:
    if df_with_indicators is not None and not df_with_indicators.empty:
        st.subheader("Statistical Properties of OHLCV & Technical Indicators")
        stat_cols = [
            c
            for c in df_with_indicators.columns
            if c
            in (
                list(config.REQUIRED_PRICE_COLUMNS)
                + [
                    f"SMA_{st.session_state.sma_period}",
                    f"EMA_{st.session_state.ema_period}",
                    f"RSI_{st.session_state.rsi_period}",
                    "MACD",
                    "MACD_Signal",
                    "MACD_Hist",
                ]
            )
        ]
        stats_table = df_with_indicators[stat_cols].describe().T
        st.dataframe(
            stats_table.style.format(
                {
                    "mean": "{:,.2f}",
                    "std": "{:,.2f}",
                    "min": "{:,.2f}",
                    "25%": "{:,.2f}",
                    "50%": "{:,.2f}",
                    "75%": "{:,.2f}",
                    "max": "{:,.2f}",
                }
            ),
            use_container_width=True,
        )
    else:
        st.info("No data available.")


# ------------------------------------------------------------------------------
# TAB 5: DATA CLEANING PIPELINE REPORT
# ------------------------------------------------------------------------------
with tab_cleaning:
    if st.session_state.cleaning_report is not None:
        report = st.session_state.cleaning_report
        st.subheader("Data Quality & Cleaning Pipeline Verification")

        rep_col1, rep_col2, rep_col3, rep_col4, rep_col5 = st.columns(5)
        with rep_col1:
            st.metric(
                label="1. Raw Records Ingested",
                value=f"{report.get('initial_rows', 0):,}",
            )
        with rep_col2:
            dups = report.get("duplicates_removed", 0)
            st.metric(
                label="2. Duplicate Dates Dropped",
                value=f"{dups:,}",
                delta="-0 duplicates" if dups == 0 else f"-{dups} dropped",
                delta_color="normal" if dups == 0 else "inverse",
            )
        with rep_col3:
            imputed = report.get("imputed_count", 0)
            st.metric(
                label="3. Missing Gaps Imputed",
                value=f"{imputed:,}",
                delta="Clean (0 gaps)" if imputed == 0 else f"+{imputed} fixed",
            )
        with rep_col4:
            dropped = report.get("dropped_rows", 0)
            st.metric(
                label="4. Unresolved Rows Dropped",
                value=f"{dropped:,}",
                delta="-0 rows" if dropped == 0 else f"-{dropped} dropped",
                delta_color="normal" if dropped == 0 else "inverse",
            )
        with rep_col5:
            final_rows = report.get("final_rows", 0)
            st.metric(
                label="5. Clean Production Bars",
                value=f"{final_rows:,}",
                delta="100% Complete",
            )
    else:
        st.info("Load data to view the cleaning verification report.")


# ==============================================================================
# Persistent Footer (Required on every page)
# ==============================================================================
st.markdown(
    f'<div class="app-footer">⚖️ {config.FOOTER_TEXT}</div>',
    unsafe_allow_html=True,
)

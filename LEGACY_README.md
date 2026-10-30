# CAPITEX

**Current upgrade stage: Phase 4.** The CAPITEX wallet and AI Predictor are available at
<http://localhost:8501>; API documentation is at <http://localhost:8000/docs>.
Install the dependencies with `python -m pip install -r requirements.txt`, then start
both services with `python run.py`.

See [Phase 1 setup and data contracts](PHASE1.md),
[Phase 2 forecasting and reasoning](PHASE2.md), and
[Phase 3 wallet and API contracts](PHASE3.md), and
[Phase 4 Predictor and verification](PHASE4.md).
Choose **AI Predictor** to view the outlook, confidence ring, price corridor and
collapsed reasoning drawer for any of the ten default markets.
The new wallet uses `capitex.db` and starts with $100,000 in virtual cash. Existing
human/AI wallets remain in `market_sim.db`, accessible through `legacy_app.py` and
the preserved legacy API routes. No existing trades are migrated or deleted.

The documentation below describes that preserved prototype, including its original
phase numbering; it does not describe the current CAPITEX delivery gates.

## Original prototype documentation

A production-grade quantitative financial analysis, algorithmic backtesting, and virtual paper-trading platform built with Python, Streamlit, FastAPI, Pandas, Plotly, scikit-learn, TensorFlow/Keras LSTM, and SQLite WAL.

> **Compliance Notice:** Simulated trading for educational purposes only. Not financial advice.

---

## 🏗️ Architecture Overview

The application features a decoupled, production-ready full-stack architecture:

```
python capstone/
│
├── run.py                 # Cross-platform unified launcher (FastAPI + Streamlit + graceful shutdown)
├── main.py                # High-performance FastAPI REST backend server
├── app.py                 # Streamlit interactive frontend dashboard
├── config.py              # Central configuration constants, paths, and hyperparameters
├── database.py            # SQLite WAL context manager, schema, and persistent state management
├── data_loader.py         # Data fetching (yfinance REST API / CSV), imputation, cleaning, and live quotes
├── indicators.py          # Technical indicators engine (SMA, EMA, RSI, MACD)
├── lstm_model.py          # Log-return LSTM neural network, model caching (.keras), and signal generation
├── trading_engine.py      # Atomic execution engine, dynamic trailing stop-loss, and portfolio analytics
├── backtester.py          # Walk-forward multi-ticker strategy simulator and alpha benchmark
├── requirements.txt       # Pinned production dependencies
├── README.md              # Complete project documentation
│
├── data/                  # Market data and sample universes
│   ├── sample_stock_data.csv
│   └── sample_tickers.csv
├── models/                # Serialized trained Keras LSTM models (.keras SHA-256 cache)
└── market_sim.db          # Persistent SQLite WAL database (wallets, holdings, ledger, snapshots)
```

---

## 🌟 Key Features Across All 5 Phases

### Phase 1: Architecture, SQLite Setup & Data Pipeline
- Persistent SQLite database (`market_sim.db`) with `wallets`, `holdings`, `trade_ledger`, and `portfolio_snapshots`.
- Seeded virtual currency wallets with `$100,000.00` starting cash for both Human and AI agents.
- Automated data ingestion from Yahoo Finance REST API (`yfinance`) or user-uploaded CSV files.
- Resilient time-series cleaning: deduplication, forward-fill + backward-fill gap imputation, and multi-index flattening.
- Real-time quote fetcher with 60-second caching and market-close fallback.

### Phase 2: Technical Indicators & Synchronized Multi-Row Charting
- Pure Pandas/NumPy technical indicators: SMA, EMA, Wilder's RSI, and MACD (12, 26, 9).
- Clean handling of warmup periods without trimming price history.
- 3-row synchronized interactive Plotly candlestick chart with SMA/EMA overlays, RSI oscillator with 70/30 thresholds, and MACD momentum histogram.
- Static Matplotlib dark-themed fallback renderer.

### Phase 3: LSTM Neural Forecasting & Algorithmic Signals
- Stationary log-return target formulation: $r_t = \ln(Close_t / Close_{t-1})$.
- Chronological 80/20 train/test split with `StandardScaler` fitted strictly on training data (zero look-ahead bias).
- Reconstructed price forecast: $\hat{Close}_t = Close_{t-1}^{\text{actual}} \times \exp(\hat{r}_t)$.
- Two-layer Keras LSTM network with `tanh` activations for symmetric positive/negative return propagation.
- Deterministic model caching via SHA-256 parameter hashes to `.keras` files with a force-retrain toggle.
- Rigorous forecast evaluation against Naive Persistence Baseline ("tomorrow = today") reporting RMSE, MAE, Directional Accuracy %, and Mean Prediction Bias ($\text{mean}(\hat{r}_t)$).
- Multi-indicator signal generator generating balanced BUY, SELL, and HOLD recommendations.

### Phase 4: FastAPI REST Backend, SQLite WAL & AI vs. Human Arena
- **FastAPI REST API (`main.py`)**: Endpoints for `/api/health`, `/api/price/{ticker}`, `/api/predict`, `/trade/execute`, `/trade/ai-step`, `/trade/check-risk/{wallet_type}`, `/portfolio/{wallet_type}`, and `/portfolio/reset`.
- **SQLite Write-Ahead Logging (`WAL`)**: Enabled `PRAGMA journal_mode=WAL;` and `PRAGMA busy_timeout=5000;` on every connection to allow non-blocking concurrent reads and writes between FastAPI and Streamlit.
- **Atomic Trade Engine (`trading_engine.py`)**: Cash/share validation, weighted average price recomputation, and net-worth snapshot logging in single atomic transactions.
- **Dynamic Trailing Stop-Loss & Take-Profit**: Positions continuously update `peak_price = max(peak_price, live_price)`. Automatic liquidation occurs when price drops below $\text{peak\_price} \times (1 - \text{stop\_loss\_pct})$ or reaches $\text{avg\_buy\_price} \times (1 + \text{take\_profit\_pct})$.
- **AI vs. Human Arena Tab**: Split-screen execution desk with manual order routing for humans, scaled tranche buying for the AI bot (default 10% cash up to 40% position cap), side-by-side metric cards, Plotly net-worth trajectory chart, and unified trade ledger.

### Phase 5: Bulk CSV Strategy Backtester & Alpha Benchmark
- **Walk-Forward Simulator (`backtester.py`)**: Out-of-sample walk-forward backtesting simulator with zero look-ahead bias.
- **Benchmark Comparison**: Computes passive Buy & Hold Benchmark ROI % over the exact same test window and determines strategy Alpha.
- **Multi-Ticker Ingestion**: Supports arbitrary CSV file uploads or one-click loading of `data/sample_tickers.csv` (AAPL, MSFT, NVDA, GOOGL, AMZN, TSLA, META).
- **FastAPI Backtest Endpoints**: Exposes `/api/backtest/ticker` and `/api/backtest/run`. Streamlit invokes per-ticker endpoints in a live loop to update the progress bar incrementally.
- **Performance Leaderboard**: Sortable leaderboard showing Rank, Ticker, Strategy ROI %, Buy & Hold ROI %, Alpha %, Win Rate %, Max Drawdown %, Total Trades, RMSE, and Directional Accuracy %.
- **Comparative Visualizations**: Plotly grouped bar chart comparing Strategy ROI % vs. Buy & Hold ROI % across the universe.
- **CSV Data Export**: Instant download buttons for the Backtest Leaderboard CSV and the complete SQLite Trade Ledger CSV.

---

## 🚀 Quickstart Guide

### 1. Installation
Clone or navigate to the repository directory and install the pinned dependencies:

```bash
cd "python capstone"
pip install -r requirements.txt
```

### 2. Launch the Full Stack (Recommended)
Launch both the FastAPI REST backend and Streamlit frontend concurrently using the unified launcher:

```bash
python run.py
```

The launcher will:
1. Boot the FastAPI backend server on `http://127.0.0.1:8000`.
2. Poll the health endpoint until operational.
3. Print the interactive Swagger documentation URL.
4. Launch the Streamlit frontend dashboard on `http://localhost:8501`.
5. Intercept `Ctrl+C` to cleanly shut down both servers.

### 3. Service URLs
- **Streamlit Web Application:** [http://localhost:8501](http://localhost:8501)
- **FastAPI Interactive Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **FastAPI Alternative ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **FastAPI Health Status:** [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

---

## 🧪 Testing & Verification

Run the automated test suites to verify backend endpoints and execution mechanics:

```bash
# Verify Python syntax and byte-compilation across all modules
python -m py_compile app.py main.py run.py trading_engine.py database.py config.py data_loader.py indicators.py chart_plotter.py lstm_model.py backtester.py

# Verify FastAPI REST endpoints (Health, Price cache, Orders, AI steps, Risk triggers)
python test_phase4.py

# Verify Dynamic Trailing Stop-Loss and partial sell accounting
python test_trailing_stop.py
```

---

## ⚖️ Disclaimer
*This software is developed strictly for educational and research demonstrations. It does not constitute investment or financial advice. Virtual trading is conducted entirely with simulated paper currency.*

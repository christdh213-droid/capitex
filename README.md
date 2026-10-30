# CAPITEX

CAPITEX is a local stock and cryptocurrency analyzer, calibrated LSTM forecaster,
and virtual-currency trading app. It pairs a FastAPI backend with a Streamlit
interface, SQLite WAL storage, and offline-safe CAPITEX branding.

## Setup

Use **Python 3.13** for the exact pinned dependencies in this repository. The source
uses Python 3.10+ syntax, but the supplied package versions were verified on Python
3.13; older Python installations may require compatible dependency versions.

From the project folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run.py
```

If PowerShell activation is restricted, run `.\.venv\Scripts\python.exe run.py`
after installing dependencies with that interpreter. No API keys are required.
Market quotes, history and news require internet access. The CPU version of
TensorFlow works on Windows; a GPU is optional.

- App: <http://localhost:8501>
- API documentation: <http://localhost:8000/docs>
- Backend health: <http://localhost:8000/api/health>

The launcher waits for the backend, prints the documentation URL, then starts
Streamlit and checks readiness. Open the app link in your browser. **Ctrl+C** stops
both processes started by this launcher. Healthy existing services are reused and
are not stopped by a later launcher. Foreign or unready services occupying ports
8000/8501 produce an actionable error. Both services bind to localhost only.

To run the services separately, use two terminals:

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

## Wallet

The wallet starts with **$100,000 in virtual cash**. Select an asset from Explore
markets to open its chart, check an exact 1–90 calendar-day lookback, and buy or sell
by dollar amount or share/token quantity. Quick-fill buttons provide 25%, 50% and
MAX sizing. The order card estimates cost; the receipt and ledger record the actual
fill from a fresh execution-time quote. A historical-close fallback is explicitly
labeled. Retried requests with the same order ID do not create duplicate trades.

Cash, holdings, ledger entries and event snapshots commit atomically. Refreshes
update trailing peaks and record at most one refresh snapshot per 30 seconds.
History displays net worth, performance and local-time ledger entries. Risk checks
are explicit; position protection is not an unattended background trading service.
Reset clears only the CAPITEX virtual wallet after the confirmation checkbox.

Closed-market lookbacks show the last completed hourly bar at or before the
requested time, with both requested and observed timestamps. They do not invent a
tick for a closed market. Quotes may be delayed.

## AI Predictor

Choose a stock or crypto asset, then select **Next 24 Hours**, **7 Days**, or
**30 Days**. The screen shows a directional outlook, model certainty ring, and
median target within an estimated 80% price corridor. Stocks use trading sessions
and estimated weekday dates (exchange holidays are not adjusted); crypto uses
calendar days. The corridor starts at the displayed quote and timestamp.

**View AI Reasoning Factors** starts collapsed and reveals the CAPITEX sentiment
blend, technical tags and available catalysts. These are market context, not causal
explanations of individual LSTM decisions. Missing news and calendar inputs are
omitted and do not produce fabricated catalysts.

Models use stationary features, chronological 80/20 evaluation and training-only
scaling. Monte Carlo dropout and held-out residual calibration inform the corridor.
Certainty is an error-adjusted score capped at 99%, not a success probability. The
model may fail to beat the naive persistence baseline. First forecasts can take a
minute or more on a CPU; subsequent requests reuse model caches. Browser-session
outlooks are retained for five minutes; Refresh outlook requests a new inference.

## Academic inspector and CSV

At the bottom of the Predictor, expand **Academic & Quantitative Inspector
(RMSE / Raw Indicators / CSV Batch)**. It contains:

- Test RMSE, MAE and directional accuracy with the naive baseline comparison.
- A technical chart loaded on demand: candlesticks, SMA 20/50, EMA 12/26,
  Wilder RSI 14 and MACD 12/26/9, plus raw readings.
- Prediction CSV and complete trade-ledger CSV downloads. The ledger includes
  RESET audit rows; those rows are excluded from win/loss and trade statistics.
- A CSV uploader for custom daily datasets, with separate evaluation, forecast
  corridor, technical chart and prediction download. CSV analysis never places trades
  or changes the consumer market forecast.

CSV columns: `Date` (or `Timestamp`), `Open`, `High`, `Low`, `Close`, `Volume`.
Use at least **400 cleaned daily observations**, no more than **10,000 rows**, and
at most **2 MB**. Input is normalized to UTC; timestamps without a zone mean UTC.
Cleaning removes duplicate timestamps and invalid bars and fills missing values
forward then backward; the inspector reports this cleaning. Intraday observations
must be aggregated into daily bars before upload. A CSV forecast starts at its last
historical close and is clearly labeled as a dataset simulation, not a live quote.

CSV results stay in the Streamlit session. Dataset-specific model caches are saved
in `models/`; raw uploaded CSVs are not written to disk by the upload endpoint.
Prices are exported numerically without currency symbols. Ledger timestamps are
shown in IST without microseconds; forecast timestamps are exported in UTC.
A reproducible **synthetic** 500-row dataset is supplied at
`data/sample_daily_ohlcv.csv` for offline upload testing. It is not market history.
The older `sample_stock_data.csv` is preserved but is too short for this model's
train/calibration/test split.

## Storage and project layout

| File / folder | Purpose |
| --- | --- |
| `app.py`, `assets/` | Consumer UI, styles, Canva branding and local logo badges |
| `main.py`, `run.py` | REST API and unified launcher |
| `database.py`, `trading_engine.py` | WAL storage and shared-price atomic execution |
| `data_loader.py`, `indicators.py` | Quotes, CSV cleaning and technical indicators |
| `lstm_model.py`, `reasoning_engine.py` | Forecasting, calibration and market context |
| `chart_plotter.py`, `academic.py` | Consumer/technical charts and CSV export helpers |
| `capitex.db` | Current CAPITEX wallet and ledger |
| `market_sim.db`, `legacy_app.py` | Preserved prototype human/AI wallets and UI |
| `models/`, `data/` | Model caches, samples, logs and verification evidence |

The new CAPITEX wallet is separate from the original human/AI wallets. Existing
prototype trades are preserved rather than migrated or deleted. To open that UI,
run `python -m streamlit run legacy_app.py --server.port 8502` while the backend is
running. Historical prototype documentation is in `LEGACY_README.md`.

## API

Swagger lists current and compatibility routes. CAPITEX routes:

| Method | Path |
| --- | --- |
| GET | `/api/health` |
| GET | `/api/wallet` |
| GET | `/api/asset/{ticker}?days_ago=44` |
| POST | `/api/trade` |
| POST | `/api/wallet/reset` |
| POST | `/api/wallet/risk-check` |
| GET | `/api/predict/{ticker}?horizon=7d` |
| GET | `/api/inspect/{ticker}` |
| POST | `/api/inspect/csv` |
| GET | `/api/wallet/ledger` |

An order accepts one of `amount` or `quantity`, with `ticker`, `action` (BUY/SELL),
and an optional unique `client_order_id`. The CSV endpoint accepts JSON with
`csv_text` and a `horizon` of `24h`, `7d` or `30d`.

## Verification and troubleshooting

```powershell
python -m unittest test_phase1.Phase1Tests test_phase2.FeatureReasoningTests test_phase2.LSTMIntegrationTests test_current_app_fixes.CurrentAppFixTests test_phase3 test_phase4 test_phase5
python test_phase5.py
```

The second command also runs `python -m py_compile` over the project Python files.
The tests use isolated wallets; no test orders or resets are applied to your wallet.
Phase-specific contracts and evidence are documented in `PHASE1.md` through
`PHASE5.md`, with logs and smoke-test results under `data/`.

If data is unavailable, use Refresh or try another ticker; CSV inspection remains
available from the academic drawer. Inspect launcher output for startup errors.
If another app occupies ports 8000 or 8501, stop that app before launching CAPITEX.
Do not remove a database while services are running. To back up WAL databases,
stop both services first or use SQLite's backup API.

**CAPITEX — Simulated trading for educational purposes only. Not financial advice.**

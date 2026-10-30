# CAPITEX Phase 5 — Final delivery

The five requested phases are implemented. `python run.py` starts the backend,
waits for a healthy CAPITEX database response, prints the Swagger URL, and starts
Streamlit on localhost. Startup failures clean up already-started children.
Ctrl+C stops only services owned by the launcher; reused services are preserved.
Both the clean startup and actual Ctrl+C child-process cleanup were verified.
The app was then restarted in the background for continued use.

## Academic drawer

At the bottom of AI Predictor, open **Academic & Quantitative Inspector
(RMSE / Raw Indicators / CSV Batch)**. It starts collapsed and contains the selected
horizon's test RMSE, MAE, directional accuracy and naive baseline comparison.
**Load technical chart** fetches the latest 252 observations with indicators already
computed over the full history: candlesticks, SMA 20/50, EMA 12/26, Wilder RSI 14,
MACD, signal line and histogram. A raw readings table is included.

**Download predictions CSV** exports all forecast points, including the quote
anchor, UTC dates, lower/median/upper prices, model score and anchor source.
**Download trade ledger CSV** exports every ledger row, without the wallet screen's
200-row preview limit. Times are formatted in IST without microseconds. RESET
audit rows remain in exports but do not count toward performance statistics.
Both download buttons were exercised in the real browser.

The uploader accepts a daily OHLCV CSV (Date/Timestamp, Open, High, Low, Close,
Volume): at least 400 cleaned daily rows, up to 10,000 rows and 2 MB. Uploads are
analyzed through the same calibrated model with isolated dataset-specific cache
identities. They show their own metrics, corridor, technical chart, cleaning report
and prediction download. The result is labeled as a simulation anchored to the
dataset's last close. No live quote is substituted, no wallet is modified, and the
consumer market outlook is not replaced. Raw upload content is not saved by the
endpoint; learned model caches remain in models/.

Use `data/sample_daily_ohlcv.csv`, a clearly synthetic 500-row dataset, to try the
full flow. The older short sample remains preserved. Real CSV training/evaluation
passed with 500 cleaned rows and 88 held-out origins.

## Verification

- All 24 project Python files pass `python -m py_compile`.
- All 69 tests pass in the final complete regression run.
- The complete five-phase regression suite covers data cleaning/lookbacks, actual
  model training/cache reload, reasoning, execution and snapshots, wallet APIs/UI,
  Predictor navigation, CSV/API/UI/export flows and launcher lifecycle.
- Compatibility API routes are tested with temporary wallets and mocked expensive
  legacy training/backtests. This checks route wiring without changing user funds.
- Live backend and Streamlit health returned 200. Apple and Bitcoin forecasts
  passed at all three horizons, and market inspection, wallet, ledger and lookback
  routes returned 200. Temporary Yahoo failures were observed earlier and recovered;
  the final recheck passed without outages. Failures remain explicit in the UI.
- Browser checks exercised both main sub-parts, a real file upload and analysis,
  technical chart, actual prediction/ledger downloads, default-collapsed drawers,
  and layouts at 1280px and 800px. No horizontal page overflow or broken logos were
  found. The educational footer is rendered on both sections and CSV results.
- Verification placed no real user-wallet trades and did not reset existing wallets.

Evidence: `data/phase5_full_tests.log`, `data/phase5_tests.log`,
`data/phase5_live_results.json`, `data/phase5_csv_live_result.json`,
`data/phase5_launcher_started.json`, and `data/phase5_*.jpg`.

See README.md for complete setup, operation, limits, API paths and troubleshooting.

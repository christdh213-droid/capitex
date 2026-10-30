# CAPITEX — Phase 2

The multi-horizon forecasting and market-context engines are implemented. The prototype
API/UI continues to use its compatible single-feature functions until the CAPITEX routes
and interfaces are introduced in Phases 3 and 4. No new trading routes or UI screens are
part of this phase. Stop until the user explicitly says **continue**.

## Verification

```powershell
python test_phase2.py
python test_phase2.py --live --smoke
```

The script compiles every Python file with `python -m py_compile`. Its 15 tests check
Wilder RSI, causal/dimensionless features, train-only scaling, purged target alignment,
cache invalidation, real Keras training/reload, stochastic dropout, all three forecast
corridors, reasoning tags, missing-source behavior and JSON serialization. The offline
integration test trains for two epochs on an explicitly labelled synthetic fixture.
Live mode trains a separate three-epoch model on current five-year Yahoo AAPL history,
uses its latest quote as the forecast anchor, and retrieves news/calendar context.
These short runs verify functionality, not production forecasting accuracy. Normal
training defaults to 25 epochs, with 60 daily bars per input window.

Smoke mode starts real FastAPI/Streamlit processes, probes both health routes, runs the
actual Streamlit page with a quote fixture, and cleans up its temporary processes.
Logs use `data/phase2_*_smoke.log`. Live evidence and forecast payloads are stored in
`data/phase2_live_results.json`; synthetic evidence is `data/phase2_fixture_results.json`.
Keras may use the operating system temporary directory when serializing `.keras` files;
restrictive execution sandboxes must permit that directory as well as the project folder.

## Engine use

```python
import data_loader
import lstm_model
import reasoning_engine

history = data_loader.load_market_data("AAPL", period="5y")
bundle = lstm_model.train_multihorizon_model(history, ticker="AAPL")
quote = data_loader.get_live_quote("AAPL")
prediction = lstm_model.forecast_multihorizon(
    bundle, horizon="7d", current_price=quote["price"], ticker="AAPL"
)
reasoning = reasoning_engine.build_reasoning("AAPL", history)
```

Horizon aliases: `24h`, `1d`, `24 Hours`, `Next 24 Hours`, `7d`, `7 Days`, `30d`,
`30 Days`, or integers 1/7/30. Crypto steps represent calendar days; stock steps
represent trading sessions. Future stock timestamp labels exclude weekends but are
explicitly labelled weekday estimates, without exchange-holiday adjustments. The
model targets daily observations, so a 30-session stock horizon is not 30 calendar days.

Provide at least 400 valid daily bars (more for a large window). The bundled Phase 1
sample CSV has only 188 rows and is intentionally rejected for this model. Inputs must
have unique, chronological daily timestamps and finite positive Close prices. No
backward filling is performed inside the model. The Phase 1 cleaning pipeline's separate
backward-fill behavior should be considered when using imported imputed datasets.

## Modeling and leakage controls

- Causal features: daily log return, RSI/100, MACD histogram/Close, Close/EMA20 − 1,
  and 10-bar rolling return standard deviation. These are dimensionless return/regime
  features; statistical stationarity is not guaranteed for real financial series.
- An 80/20 chronological outer split reserves the last 20% for evaluation. Within the
  first 80%, the last 20% is reserved for interval calibration. Both StandardScalers fit
  only the internal fitting block (roughly the first 64% of feature rows). Training and
  calibration origins with 30-step labels crossing a block boundary are excluded.
- A two-layer LSTM with dropout predicts 30 daily log returns directly. A linear head
  avoids price-range clipping; there is no recursive feeding of predicted OHLCV/features.
  Training uses chronological mini-batches without shuffling or test-driven early stopping.
- At inference, 30 calls with `training=True` retain dropout. Cumulative predicted log
  returns reconstruct median price paths from the supplied live price. All corridor
  arrays include step zero, anchored exactly to that price. Without a live anchor the
  historical last Close is used and labelled.
- Interval variance combines cumulative MC dispersion and pre-test residual variance
  multiplied by time (`sigma * sqrt(t)` in standard-deviation terms). An empirical
  absolute standardized path-error quantile on the calibration block sets each horizon's
  multiplier, with the central 80% normal multiplier as a conservative floor. Log-widths
  cannot narrow as the horizon grows. Zero-width is used only at the starting anchor.
- Evaluation reports origin-to-target-price RMSE, MAE, directional accuracy, persistence
  baseline RMSE/MAE/direction, sample counts and observed held-out interval coverage
  for 1, 7 and 30 steps. Final test origins include earlier observed test history but
  never future information. Multi-step windows overlap; sample counts are not counts
  of independent observations. Persistence direction is zero, not an arbitrary 50% score.
- Confidence is `99 * MC directional agreement / (1 + normalized held-out return RMSE)`,
  clipped to 0–99. This is an error-adjusted heuristic score, **not a statistically
  calibrated probability of a correct direction**. The payload says so explicitly.
  An estimated 80% corridor likewise does not guarantee 80% future coverage. Realized
  test coverage is exposed so conservative or poorly calibrated intervals remain visible.
- Modeling returns prevents a mechanical shifted-price target construction, but cannot
  guarantee a profitable forecast or improvement over a random walk. Baseline comparison
  remains visible even when the LSTM loses.

## Persistence

New caches use `models/capitex_<training-content-hash>.keras` plus a JSON manifest.
The key includes training feature content and timestamps, architecture, feature schema,
window, optimizer rate, seed, TensorFlow version and training settings. Revised training
data invalidates the cache; changing held-out values cannot influence model selection.
Scaler statistics are recorded as JSON; no untrusted pickle is loaded. Corrupt or
mismatched caches retrain. Writes replace temporary files atomically. Calibration and
test metrics are recomputed when a bundle is loaded. In-process TensorFlow work is
serialized; independent app processes can still perform their own training.

## Reasoning defaults

`indicators.technical_summary(history)` provides colored RSI/MACD/trend/EMA/volatility
tags plus raw SMA20/50, EMA12/20/26, Wilder RSI14 and MACD12/26/9 values. Warm-up values
are null rather than fabricated readings. Configurable prototype indicator columns remain
compatible with existing chart code.

`reasoning_engine.build_reasoning(...)` blends seven-day headline lexical polarity (30%),
five-bar momentum (30%), RSI regime (20%) and directional volume ratio (20%) into a
0–100 CAPITEX Fear & Greed heuristic. It is not an official external Fear & Greed index.
Missing factors receive zero weight and available weights are renormalized. News is
deduplicated, timestamp-filtered and screened with a small auditable lexicon and local
negation rules. Headlines can be ambiguous; this is not an NLP language model.

The output includes source availability, effective component weights, timestamps and
headline evidence. `news=` and `calendar=` injection allow reproducible tests;
`fetch_context=False` performs no news/calendar requests. News/calendar responses are
cached for 15 minutes. The live verification returned no headlines from Yahoo, which is
disclosed rather than replaced with invented news.

Catalyst tags use estimated yfinance earnings dates (labelled **Earnings Window**, since
these do not establish an earnings-call time), a volume surge relative to the preceding
20 bars, and a 52-week high test from trailing calendar-year price data when history is
long enough. Crypto never receives corporate earnings tags. Fed tags use a local
2026–2027 schedule snapshot verified from the Federal Reserve; no dates are invented
outside those years. Refresh the snapshot if the published tentative schedule changes.
Event-day offsets use the US market's America/New_York date, recorded in the payload.
Reasoning describes market context; it does not claim causal attribution to the LSTM.

Primary references: [Keras Dropout](https://keras.io/api/layers/regularization_layers/dropout/),
[scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html),
[yfinance news API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.get_news.html),
[Federal Reserve meeting calendar](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm).

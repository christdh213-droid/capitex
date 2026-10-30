# CAPITEX — Phase 1

Phase 1 is the branding, database and data-engine foundation. The existing prototype UI,
trading routes and LSTM remain available for compatibility. They are not the new Phase 2–5
implementations. Stop here until the user explicitly says **continue**.

## Run and verify

The supplied dependency pins match the installed Python 3.13 environment. Source code uses
Python 3.10+ syntax, but these exact package versions have only been verified on 3.13.

```powershell
python -m pip install -r requirements.txt
python test_phase1.py
python test_phase1.py --live --smoke
python run.py
```

The test script runs `python -m py_compile` against every project Python file, checks
module imports separately, tests database WAL/persistence/rollback, CSV validation,
timezone conversion, cache expiration and all 90 offsets with deterministic fixtures.
`--live` verifies Apple and Bitcoin quotes and all 180 lookbacks against Yahoo Finance,
then writes `data/phase1_live_results.json`. `--smoke` starts real FastAPI and Streamlit
processes on free local ports, probes their health routes, runs the Streamlit page script
with an explicit quote fixture, and terminates only the processes it started. Startup
logs are in `data/phase1_*_smoke.log`. Internet access is required for live data.

## Contracts and defaults

- `config.ASSET_UNIVERSE`: seven major US equities and Bitcoin, Ethereum and Solana.
  All are quoted in USD. Publisher/project-hosted image URLs are optional sources; some
  provide only favicons. They are not guaranteed high-resolution circular portraits.
  Every asset has a bundled scalable circular SVG badge. `branding.asset_logo_html`
  validates fetched image types and substitutes the badge if the remote request fails.
- `database.init_db()` creates `capitex.db` with `wallet`, `holdings`, `trade_ledger`,
  and `portfolio_snapshots`. Connections use WAL and a 5,000 ms busy timeout.
  The singleton wallet begins with 100,000 virtual USD. Initialization is idempotent.
- Prototype human/AI wallets remain in `market_sim.db`. `legacy_database.py` preserves
  their implementation; adapters and the old trading engine explicitly use that database.
  There is no automatic migration of existing balances and no clearing of user positions.
  The new wallet/trading API will be implemented in Phase 3.
- `load_market_data(ticker, period="5y")` returns a daily OHLCV DataFrame; its cleaning
  report is in `df.attrs["cleaning_report"]`. `load_from_csv(file_or_bytes)` and the old
  `load_from_yfinance(ticker, start, end)` return `(DataFrame, report)`.
- CSV accepts Date/Timestamp/Datetime/Time and case-insensitive OHLCV headers. Naive
  timestamps mean UTC; aware timestamps are converted to UTC. The pipeline sorts,
  removes invalid dates, keeps the last duplicate, converts numeric columns, replaces
  infinities, forward-fills then backward-fills, drops unresolved values and rejects
  nonpositive or inconsistent price bars and negative volume. Zero volume is valid.
  Backward filling is requested cleaning behavior; imported imputed datasets should be
  inspected before being used for academic evaluation.
- `get_live_quote(ticker)` returns the latest available one-minute provider bar, its
  actual timestamp, fetch timestamp, age, reference price, USD/% change and market status.
  Successful results are cached for 30 seconds in each process. Stock change uses the
  previous trading session close; crypto uses the latest available observation at/before
  24 hours ago. Yahoo data may be delayed; this is not an exchange tick feed. Session
  status comes from provider regular-session boundaries; missing metadata means unknown.
  Daily fallback is labelled. A total outage raises `MarketDataError`, never a fake price.
  The old `get_live_price` adapter retains the prototype zero-price/error-message contract
  only for callers that already validate `price > 0`.
- `get_90d_history_and_lookback(ticker, days_ago)` accepts integers 1–90, meaning
  **calendar days**, with target `current UTC time minus N days`. Returns hourly OHLCV
  bars over the last 90 days, the live quote, requested timestamp, actual historical bar
  timestamp, historical availability timestamp, price and dollar/% comparison.
  Prices use **the last completed bar available at/before the target**; hourly bar labels
  are start times. Weekends/holidays resolve backward to a traded observation. No price
  is invented for a closed market. `exact_timestamp_match` is explicit. Daily fallback
  is conservative: a daily bar becomes available 24 hours after its timestamp. The chart
  also includes the selected older observation if day 90 falls outside the visible window.
  All timestamps are ISO 8601 UTC. Historical data cache TTL is one hour; changing N
  reuses the same history and quote caches. No future observation is selected.

## Branding

Generated with the connected Canva plugin:

- Logo: [Open generated image](https://www.canva.com/M/MAHZI-Ihnfc?utm_source=OC-AaBlKrqBJJNF&utm_campaign=agent_connector_create_image_asset_opened)
- Banner: [Open generated image](https://www.canva.com/M/MAHZI_TaOuw?utm_source=OC-AaBlKrqBJJNF&utm_campaign=agent_connector_create_image_asset_opened)

The original Canva images are 1264×1264 and 1776×896. Their 200px previews are stored
as `assets/canva_logo_preview.png` and `assets/canva_banner_preview.png`. The app uses
matching, hand-authored, resolution-independent `assets/capitex_logo.svg` and
`assets/capitex_banner.svg` in the sidebar and header, so it works offline. These are
fallback artwork, not claimed Canva SVG exports. Provenance is in `assets/brand_manifest.json`.

Provider API reference: [yfinance Ticker.history](https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.history.html).

CAPITEX — Simulated trading for educational purposes only. Not financial advice.

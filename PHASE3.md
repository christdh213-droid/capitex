# CAPITEX Phase 3 — Wallet and REST API

Phase 3 delivers the Phantom-inspired wallet home, selectable market cards with
publisher logos and offline badges, an interactive asset chart, exact 1–90 day
lookback controls, and a Buy/Sell order card. The AI Predictor navigation destination
is a placeholder until Phase 4. No Phase 4 consumer prediction screen is implemented.

## Start and test

```powershell
python -m pip install -r requirements.txt
python run.py
```

- Frontend: http://localhost:8501
- Backend documentation: http://localhost:8000/docs
- Health: http://localhost:8000/api/health

Select Bitcoin from Explore markets. Set the exact lookback to 44 or 90; the slider,
chart period and dated comparison stay synchronized. Buy or sell using either USD
or quantity, with 25%, 50% and MAX sizing. The displayed cost is an estimate; the
receipt and ledger show the actual execution price and its source. History shows
portfolio snapshots and the local-time ledger. Reset requires an explicit checkbox.

## REST contracts

| Method | Route | Behavior |
| --- | --- | --- |
| GET | `/api/health` | Service readiness |
| GET | `/api/wallet` | Cash, holdings, values, performance, watchlist, ledger and snapshots |
| GET | `/api/asset/BTC-USD?days_ago=44` | Quote, 90-day hourly history, requested/observed timestamps, comparison and holding |
| POST | `/api/trade` | Atomic BUY/SELL with one of `amount` or `quantity` |
| POST | `/api/wallet/reset` | Reset only the new CAPITEX wallet to $100,000 |
| POST | `/api/wallet/risk-check` | Update peaks, evaluate protections and record a snapshot |
| GET | `/api/predict/AAPL?horizon=7d` | Phase 2 forecast, hero fields and reasoning payload |

Example order body:

```json
{"ticker":"BTC-USD","action":"BUY","amount":500,"client_order_id":"my-unique-order-1"}
```

Reuse an order ID only when retrying the same request. A replay returns the original
fill without another order; conflicting details are rejected. Protection percentages
are optional. Invalid sizes, insufficient cash and overselling leave the wallet unchanged.

## Defaults and execution details

- The singleton CAPITEX wallet lives in `capitex.db`. The original human and AI
  wallets remain untouched in `market_sim.db`; `legacy_app.py` retains their UI.
- The new engine and both legacy wallets use `get_execution_price()` in
  `trading_engine.py`, which bypasses the display quote cache at execution. The
  historical-close fallback is explicitly labeled and stored with the fill price,
  quote timestamp and source in the ledger.
- Cash, holding changes, ledger entries and event snapshots commit together under
  `BEGIN IMMEDIATE`. WAL and the busy timeout remain enabled. Quotes are fetched
  before opening write transactions; cash and quantities are checked inside them.
- The screen refreshes every 30 seconds. Refreshes update holding peaks and persist
  at most one refresh snapshot per 30 seconds. Trade, risk and reset snapshots are
  event records and are not throttled. Missing live marks are labeled estimated at
  cost; those refreshes do not create misleading net-worth history.
- With fewer than three snapshots the chart says “Not enough history yet.” Its
  y-axis has at least ±0.5% padding around starting cash and two-decimal ticks.
  Empty closed-trade statistics are dashes. RESET records are excluded from trade
  and win/loss counts.
- The exact lookback means N calendar days before now. For weekends, market closures
  and intrabar times, the chart uses the last completed hourly bar at or before the
  requested timestamp and displays both timestamps. It does not invent an exact tick.
- The headline change is labeled “Since start” until a true 24-hour portfolio
  comparison exists. Asset changes identify their rolling-24h or prior-close basis.
- A first prediction trains the Phase 2 model (25 epochs); subsequent requests reuse
  its cache. Confidence is the model's calibrated score, not a promised probability.

## Verification

`python test_phase3.py` compiles all Python sources and runs 13 isolated integration
tests, including Bitcoin dollar/quantity buys and sells, full MAX liquidation,
weighted cost, concurrent overspend rejection, retry idempotency, snapshots, peaks,
risk/reset isolation, unavailable quotes, lookback bounds, prediction route wiring,
and a complete Streamlit order flow. Temporary databases keep test trades separate
from the user's wallets.

The 40 earlier data, forecasting, reasoning and current-app-fix regression tests
also passed. Read-only live checks verified all ten market quotes, Apple and Bitcoin
lookbacks, and valid 24-hour/7-day/30-day forecast corridors using the real 25-epoch
model. Results are in `data/phase3_live_results.json`. Browser checks cover 1280px
and 800px layouts; screenshots are saved under `data/phase3_*.jpg`.

Consumer prediction cards/corridor/reasoning belong to Phase 4. The academic drawer,
CSV batch UI and final launcher/documentation polish remain reserved for Phase 5.

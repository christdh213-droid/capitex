# CAPITEX Phase 4 — AI Predictor

The second navigation destination now provides the consumer prediction screen.
The wallet, its database, and the legacy human/AI wallets retain their behavior.
The Phase 5 academic drawer, CSV tools and launcher polish are not included.

## Try it

Keep the running services open, or launch with `python run.py`.
Open http://localhost:8501 and choose **AI Predictor**.

1. Choose a stock or cryptocurrency, then select **Next 24 Hours**, **7 Days**, or
   **30 Days** in the target horizon card.
2. View the directional badge and expected move, SVG certainty ring, and median
   target. The main chart shows recent prices and an estimated 80% corridor with
   a glowing, labeled target node.
3. Open **View AI Reasoning Factors** to see the fear-to-greed gauge, technical
   summary pills and verified catalyst tags. This drawer starts collapsed.
4. Select **Trade [TICKER] in CAPITEX Wallet →** to open that asset's wallet
   screen with its Buy order card ready. It never places an order automatically;
   both Buy and Sell remain available on the asset screen.

## Design and defaults

- Both the market and horizon controls remain available if a data source fails.
  A first forecast may require training; the request timeout allows up to three
  minutes. The existing model cache is reused. Each browser session retains its
  completed ticker/horizon outlook for five minutes; **Refresh outlook** clears
  that session cache and requests a new inference. It does not force model training.
- Forecast dates start at the quote timestamp, and the first lower, median and
  upper points all equal that quote's price. Recent daily history is joined to the
  same point. The chart uses 7, 20 or 45 recent observations for the three horizons.
- Stock horizons represent 1, 7 or 30 trading sessions; weekday dates are estimates
  and do not adjust exchange holidays. Crypto horizons use calendar days. The
  horizon card labels this distinction. A historical/delayed fallback quote receives
  a warning instead of being presented as a fresh tick.
- The outer funnel is the Phase 2 calibrated 80% interval. Nested translucent fills
  provide the gradient; they are decorative and do not claim additional interval
  levels. The model score is capped at 99% and is explicitly described as a score,
  rather than a success probability.
- Fear/greed is the existing CAPITEX blend, not an official sentiment index. The
  drawer reports the headline count and states that missing sources are omitted.
  Empty catalyst lists say that no verified near-term catalysts were found. Market
  context is distinguished from causal attribution of the LSTM's predictions.
- Colors follow the existing obsidian/violet/emerald theme, with coral bearish and
  amber neutral states. Layout and pills wrap at narrower widths. No raw indicator
  plots or academic metrics are added to this consumer screen.

## Verification

`python test_phase4.py` compiles every Python file and runs six tests covering the
API's live-price/time anchor, stock and crypto calendars, all three horizons,
funnel/target geometry and invalid paths, full Predictor navigation, collapsed
reasoning, the trade bridge, source outages, refresh caching, empty catalysts,
and bearish/neutral display states. All trades remain in isolated test wallets.
The 13 Phase 3 wallet regression tests and 40 earlier data, forecasting, reasoning,
and current-app-fix tests also pass: 59 tests in total. All 23 Python files compile.

Real API checks passed for Apple and Bitcoin across all three horizons, with
positive ordered corridors and 60 historical observations in each payload.
Both backend and frontend health checks returned 200. See
`data/phase4_live_results.json` and `data/phase4_tests.log`.
Browser verification covers 1280px and 800px, the actual horizon selectors,
reasoning expansion, and opening the matching wallet asset without executing a trade.

# Current-app corrections — no phase advancement

## Shared fill-price proof

```text
Human: POST /trade/execute (main.execute_trade)
       -> trading_engine.execute_buy / execute_sell

AI:    POST /trade/ai-step (main.ai_trade_step)
       -> trading_engine.run_ai_trading_step
       -> trading_engine.execute_buy / execute_sell

Both:  -> trading_engine.get_execution_price(ticker)
       -> data_loader.get_live_quote(ticker, force_refresh=True)
       -> actual fill written to trade_ledger.price inside the wallet transaction
```

`get_execution_price` bypasses the UI/marking caches. The compatibility argument
`current_price` in buy/sell is ignored; clients and the forecast's historical close
cannot set the fill. AI cash allocations are converted to quantity using that same
resolved price, so a stale reference cannot change the intended tranche amount.
The AI API's `price` field now reports its actual fill; `signal_reference_price`
separately identifies the historical input to the signal. HOLD has no fill price.

If a fresh quote is unavailable, the resolver fetches the latest available daily
close. Its receipt, success message and ledger **Fill Source** explicitly say
`FALLBACK: last historical close`. If neither quote nor historical close is usable,
the order is rejected without debiting the wallet. Both BUY and SELL records include
the fill's source, label and provider timestamp. Existing historical ledger fills
retain their original price and are labelled as legacy records.

`test_current_app_fixes.py` exercises both HTTP routes with BTC-USD: a stale
50,000 reference/client price fills at a mocked fresh 63,000 quote, and both sell
paths fill at a fresh 65,000 quote. It checks the cash allocation, ledger price,
wallet identity and forced-refresh calls. Separate cases verify labelled fallback,
quote failure without a debit, and bypass of a primed display cache. Tests use
temporary databases; they never place orders in the running user wallets.

## Portfolio and display corrections

- Trades write snapshots in the same transaction as the fill. Every risk check
  writes a snapshot, including checks with no triggered trades. Reset writes a new
  starting snapshot in its reset transaction. Refresh snapshots are limited to one
  per 30 seconds **per wallet**, checked under a SQLite write lock across processes.
  Trade/risk/reset event snapshots bypass the refresh throttle.
- Every portfolio refresh raises `peak_price` to the greater of its stored value
  and the current mark. The database update itself uses MAX, so a later lower quote
  or concurrent refresh cannot reduce the peak. This does not automatically execute
  a risk order; the existing risk-check action remains the execution trigger.
- The net-worth chart draws a wallet's trajectory only after three snapshots;
  otherwise it shows **Not enough history yet**. Its range includes at least
  starting cash ±0.5%, with additional padding for actual observations, and its
  dollar ticks have two decimal places.
- No closed trades means **—** for W/L and **Win: —**. Drawdown is null/hidden
  without snapshots. RESET records remain visible for audit but are excluded from
  closed-trade, win/loss and total-trade statistics.
- Sidebar branding is CAPITEX with **Markets. Clarity. Possibility.** The developer
  phase tracker is removed. Signed P&L uses +$13.10 / -$13.10. Ledger display and
  CSV timestamps use Asia/Kolkata local time (IST), without microseconds; stored
  timestamps remain UTC.
- Metric rows wrap whole cards. Labels wrap at word boundaries with no ellipsis
  or character-by-character breaking. Browser inspection at 1280px and 800px found
  no visible metric or label width overflow. Screenshots are saved in
  `data/fixes_layout_1280.jpg` and `data/fixes_layout_800.jpg`.

## Verify

```powershell
python test_current_app_fixes.py
python test_phase1.py
```

The targeted script runs `python -m py_compile` on every Python file, then tests
both wallet routes, fallbacks, event snapshots, concurrent refresh throttling,
peak updates, statistics, chart bounds, formatting and a full Streamlit script
render. The existing services remain at http://localhost:8501 and
http://localhost:8000/docs. The tracked backend was restarted to load the fixes;
the frontend page was rerun without resetting user balances or positions.

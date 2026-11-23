# CAPITEX upgrade decisions

## Prerequisite and baseline
- Branch: `capitex-upgrade`. Existing long/short source was checkpointed in `53fda9a`; unrelated reports, presentation artifacts and historical fixture output are excluded from upgrade commits.
- The manual `test_trailing_stop.py` is excluded from pytest because it resets a running real wallet. Isolated long/short regression tests remain enabled.
- Three pre-existing synthetic-training integration cases are retained but skipped in pytest to honour the no-retraining scope. No model architecture or trained production files are changed.
- Pytest blocks both requests and curl_cffi HTTP, disables the background grader and uses temporary databases/fake providers. Streamlit TestClient HTTP stays in process.
- Baseline: 99 passed, 3 training-only tests skipped, 24 subtests passed (pytest, 88.66 s). Existing BTC-USD short tests characterize collateral, partial covers, weighting and liquidation before P4.

## P2: persisted completed-bar forecasts
- Canonical persistent time is UTC `YYYY-MM-DDTHH:MM:SSZ`. Existing timestamp text is normalized without changing IDs/financial values; new trade/snapshot writes use the same formatter.
- Horizons are integer 1, 7, 30 (API map keys are their string representations). Every Analyzer request ensures all three exist. Extra `payload_json` preserves the entire winning corridor/score, not just its final point.
- US unsuffixed symbols use XNYS sessions, DST and early closes via exchange-calendars. `.NS`/`.BO` use weekday 15:30 Asia/Kolkata; `.L` 16:30 Europe/London; `.TO` 16:00 America/Toronto. Other suffixed symbols conservatively use weekday UTC day-end; actual bar publication gates grading.
- Crypto UTC daily D closes at D 23:59:59Z. Starting price is the unadjusted last completed daily Close, not a live intraday quote. UI execution quotes remain separate.
- Model version is the first 24 characters of the trained manifest fingerprint. Inference-only fixtures use a deterministic config fingerprint. Existing weights and original scaler arrays are loaded; no model training/rebuilding occurs. Residual calibration/evaluation uses the unchanged existing functions. Missing trained weights return a clear error.
- The old alternate-outlook UI test clears its temporary forecast log between independent mock models; ordinary refresh tests assert persistence/deduplication.
- P2 verification: 102 passed, 3 skipped, 24 subtests passed (full offline pytest, 61.11 s). Existing long/short wallet tests remain green.

## P1: cached provider evidence
- News comes only from provider titles/snippets, never fetched article pages. Optional Finnhub and zero-key yfinance cover stock and crypto aliases; only relevant articles published within the last seven days are candidates.
- `news_fetch_state` is an additional table so empty/error results also respect the 15-minute refetch TTL. Canonical URLs and 90%-similar normalized titles deduplicate candidates; original cache rows remain available.
- Summaries expire after 30 minutes, are marked stale, and are never deleted. Evidence hash is SHA-256 of sorted candidate content hashes plus `capitex-grounded-v1`.
- Optional LLM uses `OPENAI_API_KEY`, `CAPITEX_LLM_MODEL` (default `gpt-4o-mini`), one request and a 10-second timeout. No local transformer or additional LLM SDK is loaded.
- Code validates allowed article IDs, schema and numeric support. As a stricter grounding rule, verified events must be exact provider excerpts. URLs, publishers and publication dates are hydrated exclusively from SQLite; LLM attempts to provide these fields are rejected.
- Deterministic fallback quotes the headline and uses cautious keyword-category interpretations, or omits an implication. Counter-thesis uses cited articles and/or the stored computed technical tags; absence of evidence is stated explicitly.
- UI shows up to three bullets initially, an ID-keyed session-state Show more toggle, source links and explicit interpretation labels. News is context, not LSTM attribution. API returns `context_summary_id`.
- P1 verification: 114 passed, 3 skipped, 24 subtests passed (full offline pytest, 66.42 s), including Show more AppTest.

## P3: target-aware grading and report card
- Startup catch-up and periodic grading run in a background asyncio task; blocking provider work runs in `asyncio.to_thread`. The default interval is 900 seconds. `CAPITEX_DISABLE_BACKGROUND_GRADER=1` prevents task creation during tests. Wallet refresh does not run the grader.
- One unadjusted daily OHLCV/actions fetch covers each symbol's due forecasts, with every SQLite connection closed before provider I/O. Conditional PENDING-only updates make repeated grading idempotent.
- Stocks wait 30 minutes after the target session close; crypto waits until the next UTC date. Only the exact target-date daily bar is accepted, never a live or later quote. Split events in the forecast window make the outcome unavailable. Missing bars remain pending for five calendar days (stocks) or 48 hours (crypto).
- Actual timestamps represent the target session close under the same calendar convention as issuance. Apparent intraday bars are rejected.
- Report-card dollar MAE/RMSE require an explicit single-symbol filter. Pooled views expose MAPE, excluding zero actual prices from its denominator. Both models use the identical graded set. Actual flat ties are excluded from directional accuracy; a flat prediction against a non-flat actual counts as incorrect.
- Coverage is labelled `Observed coverage of the estimated 80% interval`. Preliminary samples below 30 and overlapping forecast dependence are disclosed. Historical holdout metrics remain separate from this forward report card.
- P3 verification: 119 passed, 3 skipped, 24 subtests passed (full offline pytest, 128.30 s).

## P4: immutable order theses and Decimal short previews
- Prepared orders submit IDs, selected horizon and preparation time only. The server rejects arbitrary snapshot fields, missing/foreign-symbol IDs, incorrect horizon maps and mixed model/bar runs. It expands valid run IDs to all persisted horizons. Optional thesis metadata is included in idempotency fingerprints.
- Snapshot assembly and ledger persistence occur inside the trade's existing BEGIN IMMEDIATE transaction. Saved hydrated bullets, counter-evidence, forecasts and UTC preparation/execution times are independent of later summary edits, expiry or forecast grading. Old ledger rows receive a nullable `thesis_snapshot_json` column; financial records are never reset by migration.
- Prepare buttons use callbacks before ticket widgets instantiate. Navigation to another asset clears the prepared thesis; a thesis attaches only to its original symbol and trade side. History renders saved values, not current news.
- `short_accounting.calculate` is the single pure Decimal source for opens, proportional covers, liquidation and preview. Cash flows, released collateral and realized P&L round half up to cents; market prices and weighted entry retain precision so a +100% scenario reaches the exact 2x trigger. Dust below 0.00000001 closes fully as before. No borrowing fees, leverage increase or sale proceeds credited to available cash are introduced.
- The preview is read-only and uses the merged same-symbol short if one exists. Simulated ROI/P&L describe that merged position; equity also includes existing long positions and other shorts. Other asset quotes are held constant; unavailable quotes use labelled cost/entry estimates. A long position in the simulated symbol moves with that symbol. Preview input price is hypothetical; execution still fetches a fresh live quote.
- UI slider is -30% to +100%, initialized to zero and held in session state. API accepts up to +120% to check beyond-trigger capped losses. Preview cash requirements are validated against available cash.
- P4 verification: 130 passed, 3 skipped, 24 subtests passed (full offline pytest, 102.97 s). Tests cover missing/foreign IDs, atomic immutable snapshots for long and short orders, idempotency metadata, cent parity at -30/0/+50/+100/+120%, partial covers/refresh liquidation, and the prepared-short Streamlit flow through execution. Additional tests confirm Analyzer endpoint deduplication across Thanksgiving and a responsive async loop during blocking grading.
- Startup smoke compiled all 40 root Python files, started FastAPI and Streamlit on temporary ports, verified health and HTTP 200 report-card/short-preview routes, and stopped both test services. It used an isolated temporary wallet, disabled background grading, and made no live provider requests or production trades/resets. Existing production servers were not restarted.
- README documents the new routes, environment variables and frozen-model requirement. Live provider credentials/connectivity were not validated by offline tests; missing keys use deterministic/yfinance fallback and unavailable news is disclosed.

## Zero-key Yahoo news retrieval fix
- Both market context and the older reasoning fetcher share `news_provider.yahoo_news`. It uses `yf.Search` with zero quote results and ten news items per query; crypto searches both its ticker and coin name. Empty/error Search results fall back to Yahoo's public headline RSS using only standard-library urllib and ElementTree.
- Requests have five-second timeouts; RSS reads are capped at 2 MB. RSS titles/snippets, publication dates, source labels and links feed the existing relevance, seven-day recency, deduplication and SQLite TTL pipeline. No linked article pages are fetched and no dependency is added.
- Pytest now blocks urllib HTTP as well as requests/curl_cffi. Search and RSS fixtures exercise success, empty/error fallback, malformed XML, crypto aliases, relevance/recency/deduplication and cache reuse. Related reasoning tests inject the shared provider.
- A separate live probe outside pytest returned ten recent relevant Apple items and ten Bitcoin items without Finnhub or LLM calls. Only `news_fetch_state` is cleared for immediate refresh; articles, summaries, forecasts and all wallet/history records are retained.
- Verification: 136 passed, 3 skipped, 24 subtests passed (full offline pytest, 112.43 s); all 42 root Python files compiled.

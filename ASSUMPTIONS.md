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

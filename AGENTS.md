# CAPITEX Repository Guardrails
- **Stack:** FastAPI backend + Streamlit frontend + SQLite (WAL mode, `PRAGMA busy_timeout = 5000`).
- **UI Theme:** Preserve the existing visual theme, dark cards, typography, pills, and layout. Do NOT redesign the UI.
- **ML & Product Scope:** Do NOT retrain/rebuild the LSTM, do NOT load local transformers (e.g., FinBERT), do NOT build Market Replay mode yet (keep schemas clean for it), and do NOT scrape article web pages (store provider titles/snippets only).
- **Database Safety:** All migrations must be non-destructive (`CREATE TABLE IF NOT EXISTS` and safe column checks). Never wipe balances, holdings, short positions, or history.
- **Testing:** Zero live network calls in `pytest`. Use fixtures, injected clocks, fake OHLCV providers, and disable background tasks during tests.
- **Dependencies:** Keep new packages minimal and update `requirements.txt` if any are added.

# CAPITEX upgrade decisions

## Prerequisite and baseline
- Branch: `capitex-upgrade`. Existing long/short source was checkpointed in `53fda9a`; unrelated reports, presentation artifacts and historical fixture output are excluded from upgrade commits.
- The manual `test_trailing_stop.py` is excluded from pytest because it resets a running real wallet. Isolated long/short regression tests remain enabled.
- Three pre-existing synthetic-training integration cases are retained but skipped in pytest to honour the no-retraining scope. No model architecture or trained production files are changed.
- Pytest blocks both requests and curl_cffi HTTP, disables the background grader and uses temporary databases/fake providers. Streamlit TestClient HTTP stays in process.
- Baseline: 99 passed, 3 training-only tests skipped, 24 subtests passed (pytest, 88.66 s). Existing BTC-USD short tests characterize collateral, partial covers, weighting and liquidation before P4.

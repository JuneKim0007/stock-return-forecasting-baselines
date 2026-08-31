# Refactor backlog

Surveyed 2026-08-31 · scope `src/` · 20 files
Baseline: tests 64 green
Previous: 46 green (post R1–R8) · 39 green (first survey)

R1–R8 closed. Defects 1, 2, 3, 5 and a newly-found Defect 6 fixed 2026-09-01.
R9–R14 remain open; R15 dropped.

**The organising fact.** The pipeline was migrated from a unified rolling window
`W` to per-model lookbacks. `src/evaluate.py` was migrated; its readers were not.
Most findings here are that migration's debt, and the five defects in
`## Defects

Found during the sweep. **A bug is not a smell** — these were recorded rather
than scheduled as refactors, and fixed separately on 2026-09-01.

### Defect 1 · analysis stage matched nothing · analysis/runner.py:25 — FIXED
`_PRED_RE` required `<ticker>_<window>_<model>.csv`; the pipeline writes
`<ticker>_<model>.csv`. Verified 0/9 matches, so `analyse_test_run` persisted no
rows and rendered no dotplots — while still returning a DB path and reporting no
error. Masked twice: `runner.py` wrapped the call in `except Exception →
warning`, and `tests/test_analysis.py` fabricated the legacy `AAA_60_naive.csv`.

Fixed at the root rather than by patching the regex: `window` recorded the
retired unified rolling window, was derivable from `model`, and was **never
read** — yet sat in the PRIMARY KEY of both analysis tables. It is now dropped
from `analysis/db.py`, `analysis/persist.py` and `analysis/runner.py`.
Verified end-to-end: a 2-ticker × 9-model run now persists 18 summary rows and
4,680 per-step rows, and writes all four dotplots. Previously zero of each.

Schema note: `CREATE TABLE IF NOT EXISTS` cannot alter a database written
before this change, and every insert would fail on the dropped NOT NULL column.
`init_analysis_schema` now detects that and raises `LegacyAnalysisSchemaError`
naming the file and the fix, rather than silently dropping the caller's rows.

### Defect 2 · per-pair figure discovery matched nothing · plots.py:89 — FIXED
Same stale pattern. `discover_predictions` returned `{}` for every real run, so
`render_all` rendered nothing. Its key is now `ticker` rather than
`(ticker, window)`.

### Defect 3 · `python -m src.plots` crashed · plots.py:424 — FIXED
`_load_metrics` sorted on a `window` column the runner does not write —
reproduced as `KeyError: 'window'`. `window` is gone from `_load_metrics`,
`plot_metric_by_model_tier` (which faceted rows=window; now one panel per tier,
averaged across tickers) and `plot_ensemble_vs_best`. The CLI now runs to
completion and writes 14 figures on the verification fixture.

### Defect 4 · 6 of 9 models rendered identical grey · plots.py:64 — FIXED
Closed 2026-08-31 by R5. Verified: 0/9 fall back, 9 distinct colours.
`tests/test_model_registry.py` pins it against recurrence.

### Defect 5 · `--refresh-cache` was a silent no-op · runner.py:381 — FIXED
The flag reached `manifest.json` and stopped; `load_returns` implements
`refresh` correctly and was never passed it, so a run recorded as
cache-refreshed had not refreshed. Now threaded through `_backtest_tier` and
`_run_one_ticker_into_tier`.

### Defect 6 · two of five figure types never rendered · plots.py — FIXED
**Found while verifying the fixes above, not during the original sweep.**
`_rolling_error_frame` capped its smoothing span at
`min(ROLLING_ERROR_WINDOW, window)`. With the unified window retired both
callers pass no real window, so the span collapsed to 0 and `rolling(0)` raised
`ValueError`. The runner's blanket `except Exception: continue` swallowed it, so
`rolling_rmse` and `rolling_mae` were **absent from every run the project has
ever made** and nothing said so.

The span is now capped by the number of steps available, which is what the cap
was for. The runner's swallow stays non-fatal but now logs at warning level —
the silence is what let this hide. Verified: a real run now produces 5 of 5
per-pair figure kinds, up from 3.

## Follow-ups

**Regenerate `assets/img/*.png`.** Ten committed figures, all embedded in
`README.md`, were rendered before R5 and show six models in the same grey. They
cannot be regenerated without a full pipeline run against yfinance. Until then
the README figures disagree with what the code now produces.

**Characterisation test for `run_eval`** — the prerequisite R9 and R10 are
blocked on. `src/rolling.py` still has no direct test, and the only end-to-end
caller passes a single `NaiveModel`, so three of four dispatch branches remain
unexercised.

**The nominal `window` argument on per-pair renderers.** The five registered
renderers still take a `window` they use only for the figure title and
filename, and both call sites pass 0, producing names like
`rolling_rmse_AAA_0.png`. Cosmetic, not broken, so it was left out of a bug fix
— removing it renames every per-pair figure the runner writes.

# Refactor backlog

Surveyed 2026-08-31 · scope `src/` · 20 files
Baseline: tests 46 green · 3,896 lines · ~985 prose lines (25%)
Previous survey 2026-08-31: 39 green · 4,271 lines · 1,167 prose lines (27%)

R1–R8 closed on branch `refactor/backlog-r1-r8`. R9–R14 remain open; R15 dropped.

**The organising fact.** The pipeline was migrated from a unified rolling window
`W` to per-model lookbacks. `src/evaluate.py` was migrated; its readers were not.
Most findings here are that migration's debt, and the five defects in
`## Defects` are the same migration's casualties.

```
evaluate.py writes ──> AAPL_ma30.csv   (ticker_model)
      ┌───────────────────┼────────────────────┐
      ▼                   ▼                    ▼
summary._PRED_RE     analysis._PRED_RE    plots._PREDICTION_FILENAME
 (ticker_model)      (ticker_W_model)      (ticker_W_model)
   9/9 match ✅         0/9 match ❌          0/9 match ❌
```

---

## Open

### R9 · Switch statements · rolling.py:71-91 · run_eval
status   blocked — needs a characterisation test first
evidence 4-way `if/elif` on `getattr(m, "kind")` inside the hot double loop, plus a
         nested `isinstance(m, ExpandingMeanModel)` type check. Cross-refs the
         `kind: str` type code on the model classes (primitive obsession) and the
         refused bequest below.
remedy   Replace Conditional with Polymorphism — **not yet**
safety   **RECONSIDER.** `src/rolling.py` has no direct tests. Its only end-to-end
         caller, `test_runner.py:112`, passes `models_factory=lambda: [NaiveModel()]`
         — so 3 of the 4 branches (`global`, `expanding`, `windowed`), the
         `ExpandingMeanModel.set_state` O(1) running-sum path at :68, and the
         `t - L < 0` NaN guard are executed by **no test in the suite**. There is no
         proof method available. Refactoring this on the strength of a green suite
         would be refactoring untested code.
         The expanding branch also carries a real O(1)-vs-O(t) optimisation that a
         naive polymorphic rewrite would silently undo.
next     write a characterisation test for `run_eval` covering all four kinds, in
         its own commit; then re-evaluate this item
blocked  the characterisation test
first seen 2026-08-31

### R10 · Refused bequest · models.py:197-259
status   open — no remedy decided
evidence `NaiveModel`, `GlobalMeanModel` and `ExpandingMeanModel` implement the
         `Forecaster.fit(y)` contract, but the engine bypasses it: naive is inlined
         as `y_full[t-1]` (rolling.py:75), expanding is driven through `set_state`
         (:80), an extra method on neither the ABC nor `ForecasterProtocol`. The
         declared abstraction is not the one used.
remedy   undecided — the honest fix is the same one R9 is waiting on
blocked  R9
first seen 2026-08-31

### R11 · Feature envy · summary.py:75 · analysis/compute.py:30 · evaluate.py:112
status   open — no remedy decided
evidence RMSE/MAE are computed three independent times over the same predictions:
         once into `metrics.csv`, once by `_per_ticker_metrics`, once by
         `compute_summary`. Three implementations of one piece of knowledge.
remedy   undecided — the right consolidation depends on whether the analysis stage
         survives (see Defect 1)
blocked  Defect 1
first seen 2026-08-31

### R12 · Inappropriate intimacy · plots.py:142 · figure_dirs
status   open — no remedy decided
evidence Context manager mutates `src.config.FIGURES_DIR` / `FIGURES_DATA_DIR`
         module globals and restores them in `finally`. Its own docstring justifies
         this: *"The pipeline runs tier-by-tier sequentially in a single process, so
         this scoped mutation is safe."*
remedy   undecided. The docstring is evidence the shape is **deliberate** — a
         recorded constraint, not an oversight. Threading an output directory
         through every renderer is the alternative, and it is more code. Left open
         rather than dropped only because that constraint is undefended: nothing
         fails if the pipeline is ever parallelised.
first seen 2026-08-31

### R13 · Divergent change · plots.py (552 lines, was 699)
status   open — no remedy decided
evidence Two jobs remain with different reasons to change: per-pair figure
         renderers (consumed by `runner`) and a standalone legacy CLI
         (`render_all`, `main`) that is currently broken — see Defects 2 and 3.
         R5 removed the third (palette/ordering) by moving it to `src.models`.
remedy   Extract Class, but the split depends on what survives the defect fixes
blocked  Defects 2 & 3
first seen 2026-08-31

### R14 · Long parameter list · selection.py:207, data.py:331, storage/db.py:148
status   open — no remedy decided
evidence `select_tickers_for_tier` 10 params / 117 lines · `load_returns` 8 / 92 ·
         `put_history` 7 / 80. `(period_start, period_end)` travels as a pair
         through 19 sites across three modules.
remedy   candidate: Introduce Parameter Object for the date-window pair. Not decided
         — `select_tickers_for_tier`'s length is mostly injected test seams
         (`history_loader`, `current_price_fn`), which are doing their job.
first seen 2026-08-31

---

## Done

### R1 · Dead code · runner.py
closed 2026-08-31 — 167 lines removed across 5 zero-call-site symbols
(`_individual_figure`, `_build_model_dict`, `_ensemble_predictions`,
`_save_predictions_csv`, `_DEFAULT_*`) plus the imports they orphaned.

### R2 · Speculative generality · models.py + plots.py
closed 2026-08-31 — deleted `_MODEL_REGISTRY`, `register_model`,
`model_factory`, `_populate_registry`, `register_per_pair_figure` and
`PerPairRendererProtocol`. The surviving `_PER_PAIR_FIGURE_REGISTRY` dict is
still iterated by the runner and kept; only its unused registration API went.
models.py 638 → 516, plots.py 699 → 552.

### R3 · Dead code · plots.py, summary.py, runner.py
closed 2026-08-31 — `PredKey` and the `windows` parameter deleted;
`_HISTOGRAM_EXCLUDE` wired to the two call sites that were re-typing its
literal. `run_test` 11 → 10 params. Also removed `_build_long_predictions`
(dead since the initial commit) and three unused imports in `rolling.py`.

### R4 · Duplicate code · _setup_logger
closed 2026-08-31 — one `get_logger(name)` in the new `src/logging_setup.py`;
the logger name is now the parameter that was the only difference. The
models.py config-block twin died with R2.

### R5 · Shotgun surgery · 6 model lists
closed 2026-08-31 — one registry in `src.models`, **derived from
`default_models()`** rather than hand-listed, so a new lookback cannot produce
a model with no row. `MODEL_ORDER`, `MODEL_COLORS`, `MODEL_LINESTYLES`,
`ENSEMBLE_NAME`, `ENSEMBLE_CHILDREN`, `color_for`, `linestyle_for` and
`ordered_models` now have exactly one definition each.
**Behaviour changed as approved:** grey fallbacks went 6/9 → 0/9, nine distinct
colours. Closes Defect 4. `assets/img/*.png` are now stale — see Follow-ups.

### R6 · Duplicate code · 3 ordering functions
closed 2026-08-31 — `summary._ordered_models` and `dotplot._models_in_order`
deleted; both modules now import from `src.models`, not from `src.plots`, so no
analysis module depends on a renderer. `ordered_models` takes any iterable of
names, covering all three call-site shapes (dict, list, Series).

### R7 · Long method · run_test
closed 2026-08-31 — 210 → 65 lines via eight named helpers
(`_resolve_active_tiers`, `_select_per_tier`, `_make_run_tree`,
`_enforce_arma_budget`, `_backtest_tier`, `_write_run_tables`,
`_write_manifest`, `_render_cross_tier_outputs`). Proved by golden-tree diff:
39/39 output entries byte-identical before and after.

### R8 · Comments · models.py
closed 2026-08-31 — module docstring, `Forecaster`, `ForecasterProtocol`,
`ARMAModel` and `MovingAverageModel` reduced to the enforceable contract; the
LSP advocacy and the empty `# Ensemble` section header dropped. Repo prose
1,167 → ~985 lines (27% → 25%). Also fixed four stale contracts found while
reading: the `EnsembleModel` references in models.py (no such class), the
runner docstring's `src.evaluate._run_one_ticker` (no such function), the
plots discovery claim (now points at Defects 2–3), and
`test_integration.py`'s instruction not to import a module that has existed
since the initial commit.

---

## Dropped

### R15 · Data clumps · src/analysis/ · (test_run, tier, ticker, window, model)
dropped 2026-08-31 — the clump is real (16 sites, `persist_per_step` takes 7
positional params), but it lives entirely in the analysis stage, which currently
matches 0 of 9 real prediction files and persists nothing (Defect 1). Wrapping a
dead stage's parameters in an object is a step whose need cannot be measured. The
`window` field is itself a residue of the retired unified-`W` design; once the
stage is revived the likely change is deleting that column, not encapsulating it.
Re-open if Defect 1 is fixed and the clump survives.

---

## Refused

_(none)_

---

## Defects

Found during the sweep. **A bug is not a smell** — these are recorded so they are
not lost, and are explicitly *not* scheduled as refactors. Fix them elsewhere.

### Defect 1 · analysis stage matches nothing · analysis/runner.py:25
`_PRED_RE` requires `<ticker>_<window>_<model>.csv`; the pipeline writes
`<ticker>_<model>.csv`. Verified 0/9 matches. `analyse_test_run` therefore persists
no rows and renders no dotplots. Masked twice over: `runner.py:579` wraps the call
in `except Exception → logger.warning`, and `tests/test_analysis.py:136` fabricates
the legacy `AAA_60_naive.csv`, so the suite is green against a contract the
pipeline abandoned.

### Defect 2 · per-pair figure discovery matches nothing · plots.py:89
`_PREDICTION_FILENAME` carries the same stale 3-part pattern. `discover_predictions()`
returns `{}`, so `render_all()` renders no per-pair figures.

### Defect 3 · `python -m src.plots` crashes · plots.py:424
`_load_metrics` sorts on a `window` column. The runner writes
`tier,ticker,model,rmse,mae,n`. Reproduced: `KeyError: 'window'`.

### Defect 4 · 6 of 9 models render identical grey · plots.py:64 — FIXED
Closed 2026-08-31 by R5. Verified: 0/9 models resolve to the fallback, 9
distinct colours. `tests/test_model_registry.py` pins it against recurrence.

### Defect 5 · `--refresh-cache` is a silent no-op · runner.py:381
The flag is recorded into `manifest.json` (:562) and never reaches
`load_returns(refresh=…)`. A run recorded as cache-refreshed did not refresh.


---

## Follow-ups

**Regenerate `assets/img/*.png`.** Ten committed figures, all embedded in
`README.md`, were rendered before R5 and show six models in the same grey. They
cannot be regenerated without a full pipeline run against yfinance. Until then
the README figures disagree with what the code now produces.

**Characterisation test for `run_eval`** — the prerequisite R9 and R10 are
blocked on. `src/rolling.py` still has no direct test, and the only end-to-end
caller passes a single `NaiveModel`, so three of four dispatch branches remain
unexercised.

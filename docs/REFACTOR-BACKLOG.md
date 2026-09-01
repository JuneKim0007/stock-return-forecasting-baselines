# Refactor backlog

Surveyed 2026-08-31 · scope `src/` · 20 files
Baseline: tests 102 green · `src/` 3,874 lines · `tests/` 2,760
Duplication: 49 repeated 4-line blocks → 24 (src 20→7, tests 29→17)
Previous: 89 · 90 · 81 · 64 · 46 · 39

R1–R8 closed. Defects 1, 2, 3, 5 and a newly-found Defect 6 fixed 2026-09-01.
R9 re-evaluated with its blocker cleared and **refused**; R10 dropped as a
consequence. R11 and R13 closed; R12 and R14 dropped with reasons. **Nothing remains open.**

**The organising fact.** The pipeline was migrated from a unified rolling window
`W` to per-model lookbacks. `src/evaluate.py` was migrated; its readers were not.
Most findings here are that migration's debt, and the five defects in
`

## Open

_(nothing open — see Done, Dropped and Refused below.)_

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

### R11 · Feature envy · metrics computed three times — and three ways
closed 2026-09-01 — the finding understated it. The three implementations did
not merely duplicate logic, they disagreed: `evaluate` and `analysis.compute`
propagated `nan`, `summary` masked it. Measured on one 5-step series with a
single unpredictable step, the same (ticker, model) was reported as `rmse=nan`
in `metrics.csv`, `0.0132` in `summary_tier1.csv`, and `nan` in `analysis.db`.

That made it a defect rather than a smell, so it went to the user as a
measurement-semantics decision. Answer: **NaN-tolerant everywhere** — score the
finite pairs, report the survivor count in `n`. This is the contract
`src/models.py` already documented (*"callers must tolerate it ... one nan child
does not poison the result"*) and which `evaluate.py` violated two lines below
its own `np.nanmean` call.

The policy now lives in `src/metrics.py` alone, with `n_scored` as its companion
so callers report coverage rather than hiding it. `rmse`/`mae` raise when
nothing is finite, keeping "nothing could be scored" distinct from "the model
was perfect". `evaluate` skips such a model with a warning instead of crashing
the ticker — a gap the new tests caught during the fix.

No output changes under the shipped config: verified a full run scores all 260
steps for all 18 (ticker, model) pairs with no NaN, and `metrics.csv` now
provably agrees with `summary_tier1.csv`. Mutation-checked: removing the mask
fails 6 tests.

### R13 · Divergent change · plots.py
closed 2026-09-01 — the finding was right that the file had more than one job,
but wrong about the remedy. Investigating it showed the second job was not
merely separable, it was **dead**: the standalone CLI read
`results/predictions/` and `results/metrics.csv`, the layout the pipeline
abandoned when it moved to `results/test_runs/test_<ts>/`. Verified on a fresh
clone — `python -m src.plots` exits 0 and reports "no artifacts written". A
third casualty of the same migration as Defects 1–3, one level further out.

So Extract Class was the wrong tool; deletion was right. Removed `render_all`,
`main`, `_summarize`, `plot_metric_by_model_tier`, `plot_ensemble_vs_best`,
`_load_metrics`, `discover_predictions`, `_PREDICTION_FILENAME` and
`_AGGREGATE_METRICS` — 211 lines. `plots.py` 557 → 346, nine functions, one
job: the per-(ticker, model) figures the runner drives.

Nothing was lost with it. `plot_metric_by_model_tier` duplicated
`summary._plot_summary_bars`, which is strictly better (it adds min/max error
bars). `plot_ensemble_vs_best` answered a question `summary_overall.csv` and
`score_histogram.png` already answer, and which the README reports from those.
`discover_predictions` was the dead twin of the live
`summary._scan_predictions` — an unrecorded duplication this closes. The dead
config constants went too: `DATA_DIR` (zero readers), `RESULTS_DIR` and
`PREDICTIONS_DIR` (read only by the deleted half).

**This partly reverses the previous commit.** Defects 2 and 3 fixed
`discover_predictions` and `_load_metrics`, which are now deleted. Those were
real bugs and the fixes were correct, but the prior question — does this code
have a job at all — went unasked. Fixing it first was wasted work, not wrong
work.

Proved by golden-tree diff: the runner's full output, all 73 entries, is
byte-identical before and after.

### F1 · Unreproducible README figure · assets/img/best_predictor_vs_price.png
closed 2026-09-01 — not a backlog smell but a reproducibility gap found while
closing R13: the README publishes this figure as a result and nothing in `src/`
produced it. Every other README figure traces to `src/summary.py`.

Two problems, both fixed. **No generator:** added
`summary._plot_error_vs_price`, wired into `summarise_overall`, plus the
plumbing it needs — `storage.get_mean_prices` reads back the mean the cache
already stores, and the runner carries it into `ticker_tested.csv`, so the
summary stage gains no database dependency. Verified end to end: a run over 17
synthetic tickers with a planted `price ** -0.23` scaling recovers a slope of
−0.21.

**The caption described a different figure.** It read *"Best causal model vs.
mean stock price"* and claimed the figure shows `expanding` winning uniformly
across tiers. That claim is true but shown by the score histogram above it; the
figure plots error magnitude against price. Caption rewritten to describe what
is drawn; the winner claim moved up to the histogram that supports it.

The published figure drew RMSE and volatility as two side-by-side panels with
identical points and identical slopes. That is not a plotting bug: predicting
the central tendency makes RMSE the sample standard deviation, verified equal to
machine precision, so the panels were one quantity drawn twice. The generator
draws one panel and states the identity on the axis label.

A dtype fragility surfaced during the end-to-end check and not in the unit
tests, whose tickers were all alphabetic: a numeric-looking symbol is read from
CSV as `int64` but parsed from a filename as `str`, so the join failed. Both
sides are now normalised, with a regression test.

### F2 · Deduplication sweep · src/
closed 2026-09-01 — prompted by the observation that the campaign had grown the
tree rather than shrunk it. Measured first: a 4-line-block detector found 20
repeated sequences. Each candidate was tested against "do these two sites change
for the same reason?" before unifying; all six passed.

* **The prediction filename was stated three times** — an f-string in the writer
  and a regex in each of two readers — and the two regexes had already drifted
  (greedy vs non-greedy). `evaluate.prediction_filename` /
  `parse_prediction_filename` now state it once, next to the code that writes it.
* **`open_analysis_db` duplicated `open_db`** byte for byte. It delegates now;
  the schemas stay separate, because those do change for different reasons.
* **Figure persistence was repeated nine times** across `summary.py` and
  `analysis/dotplot.py`, with a "no data" placeholder branch four more times.
  Both are now `plots.save_figure` / `plots.save_placeholder`.
* **The per-model line loop appeared three times** in `plots.py`; extracted as
  `_draw_model_lines`, with weight and opacity as the parameters that genuinely
  varied.
* **The shortfall warning appeared twice** in `selection.py`; the cause is the
  parameter.

That reduced duplication but not size — the shared helpers cost what the
duplication had. The size came from what the sweep exposed next: **`build_dataset`
has no callers anywhere**, in `src/` or `tests/`, and with it went
`_write_ticker_csv`, `_adf_pvalue`, `classify_tier` and the `adfuller` import —
126 lines. An earlier dead-code sweep had missed it because the module docstring
mentions the name, which the detector counted as a use.

`classify_tier` was worth removing for a second reason: it defined the tier
bands as 0–10 / 10–100 / 100+, while `config.TIERS` — what `selection.py`
actually enforces — uses 0–30 / 30–100 / 100+. A $20 stock was `tier2` by one
and `tier1` by the other. There is now one definition of where a tier begins.

Result in `src/`: 20 repeated blocks → 7, 3,997 → 3,874 lines.

**`tests/` was worse than `src/` and mostly my doing** — 29 repeated blocks
across 83 sites against `src/`'s 20. The prediction-CSV writer existed in three
files and the AR(1) generator in two; `test_selection.py` repeated the same
eight-line call block eleven times. `tests/helpers.py` now holds what several
modules genuinely share — the on-disk prediction schema and the synthetic series
— while each test keeps the data strategy that is its own. A local `_select`
helper collapses the selection call sites. Eight unused imports went with it.
`tests/` 29 blocks → 17.

Both halves show the same thing: **deduplication cuts duplication, not size** —
the shared helpers cost roughly what the duplicated code did. Size came from
deleting what nothing called. Worth remembering the next time a line count is
the goal.

Proved behaviour-preserving by golden-tree diff, all 74 entries including every
figure's byte size.

### F3 · Look-ahead leak in the naive branch · rolling.py
closed 2026-09-02 — the engine evaluated `y_full[t - 1]` unguarded, and numpy
resolves `y_full[-1]` to the *last* element of the series. At `t = 0` the
forecast for the first observation was therefore the final one: a look-ahead
leak that scores as a perfect prediction of a future the model cannot have seen.

Found while writing the characterisation tests (F-series above) and pinned there
rather than fixed, because pinning is what made the fix safe to make afterwards.

The fix needed no new policy. `run_eval`'s windowed branch already answers the
same question — not enough history — with NaN, at `t - L < 0`. The naive branch
now answers it the same way, so one condition has one answer.

Unreachable under the shipped config, where 100 warm-up trading days precede the
test window and `t` is never 0: verified by golden-tree diff, all 74 output
entries identical. It becomes reachable the moment anyone shortens `START_DATE`
or widens the test window, and it composes correctly with the metrics policy
from R11 — the unscoreable step is dropped from `n` rather than scored as a hit.

---

## Dropped

### R10 · Refused bequest · models.py:197-259
dropped 2026-09-01 — the finding is real: the engine bypasses `Forecaster.fit`
for `naive` (inlined as `y_full[t-1]`) and drives `expanding` through
`set_state`, a method on neither the ABC nor `ForecasterProtocol`. But the only
honest remedy is the polymorphic rewrite refused as R9, for reasons that apply
here unchanged — widening the interface so every model can be driven uniformly
costs more than the inconsistency does.

What was reachable has been taken: the branch no longer names a concrete class,
so the engine depends on the interface even where it does not use it uniformly.
The characterisation tests pin the bypass explicitly, making it documented
behaviour rather than a surprise. Re-open only alongside R9.

### R12 · Inappropriate intimacy · plots.py · figure_dirs
dropped 2026-09-01 — the evidence supports the current design. The context
manager mutates `config.FIGURES_DIR` / `FIGURES_DATA_DIR` and restores them in
`finally`, and its own docstring records why that is safe: *"The pipeline runs
tier-by-tier sequentially in a single process."* That is a stated constraint,
not an oversight, and it is currently true.

The alternative — threading an output directory through `save_fig_and_data` and
the three renderers — would change the registry contract
`(model_dict, ticker, window)` that `src.runner` drives, to remove a mutation
that is scoped, restored on exception, and covered by
`tests/test_plots_figure_dirs.py`. More churn than the finding is worth.

Caveat recorded rather than acted on: nothing *enforces* the single-process
constraint. If the pipeline is ever parallelised across tiers, this breaks
silently and figures land in the wrong tier's directory. Re-open then, not
before.

### R14 · Long parameter list · selection.py, data.py, storage/db.py
dropped 2026-09-01 — the counts are inflated by optional arguments, and the
evidence supports the current design.

`select_tickers_for_tier` reads as 10 parameters but is **4 required plus
`seed`** at its one production call site; the other five (`max_attempts`,
`history_loader`, `current_price_fn`, `period_start`, `period_end`) are
injected seams with working defaults, used by the ten tests that keep this
module off the network. That is dependency injection doing its job — the exact
shape `check-safety-refactoring` lists as a smell that is really an answer.
`load_returns` is the same pattern: 8 parameters, called as
`load_returns(ticker, conn=conn, refresh=refresh)`.

`put_history`'s seven are the columns of the record it writes; wrapping them in
an object used at one call site adds a class and removes nothing.

The one sub-finding with real substance — `(period_start, period_end)` travelling
as a pair through 19 sites — still fails the test for Introduce Parameter Object:
the pair is only ever passed through. No behaviour attaches to it as a unit, so
the object would have no methods and would exist to shorten signatures that are
not the problem.

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

### R9 · Switch statements · rolling.py · run_eval
refused 2026-09-01 — the blocker was cleared and the answer changed anyway.

`tests/test_rolling_characterisation.py` now covers all four dispatch branches,
both expanding paths, both guards, aliasing, ordering, the empty-input guard and
duplicate-name collapse — 17 tests, mutation-verified 8/8. The *provable* check
that produced the original RECONSIDER now passes. The refusal moved to a check
that could not be reached before: does it fix more than it breaks.

**It does not.** The measurable benefit is that adding a new `kind` would edit
one file instead of two. Measured against history: **5 kinds at the initial
commit, 5 today** — no new kind in four months. That is precisely the case
`refactor-principles` reserves: *open/closed only when a new variant is actually
expected; otherwise it is YAGNI.*

The cost is concrete. The four branches are not symmetric, so polymorphism needs
a **wider** interface than `fit(window) + predict_one()`:

| branch | would need |
|---|---|
| `naive` | `y_full` and `t` — it bypasses the model object entirely today |
| `global` | nothing; already fitted |
| `expanding` | the engine's running sum and `train_start` |
| `windowed` | the window plus the `t - L < 0` guard, moved into each model |

Every model would take a context carrying `y_full`, `t`, `train_start` and
running state, when most need only a window. The coupling does not disappear —
it relocates into a parameter object and stops being visible. A legible 20-line
switch is the better shape here. Do not re-open without a real second variant.

**The smaller change was applied.** The nested `isinstance(m, ExpandingMeanModel)`
is gone; the branch now asks `callable(getattr(m, "set_state", None))`.
`src/rolling.py` no longer imports any concrete model class — which is what
`Forecaster`'s docstring already claimed (*"`src.rolling` and `src.evaluate`
depend on this type, never on a concrete model"*) and the code contradicted.

One behaviour difference, pinned in
`test_expanding_routing_is_decided_by_set_state_not_by_class`: a model exposing
`set_state` without subclassing `ExpandingMeanModel` now takes the fast path
where it previously re-fitted. No such model exists here and the forecasts are
identical either way; only the route differs.

---

## Defects

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

**`assets/img/*.png` still predate the palette fix.** See above — the ten
committed figures need a live yfinance run to regenerate.

**`metrics.csv` is write-only.** The runner produces it and nothing in the
pipeline reads it back — `summary` and `analysis` both work from the prediction
CSVs. That is legitimate for a deliverable, and is noted only so the next
reader does not go looking for the consumer.

**The nominal `window` argument on per-pair renderers.** The five registered
renderers still take a `window` they use only for the figure title and
filename, and both call sites pass 0, producing names like
`rolling_rmse_AAA_0.png`. Cosmetic, not broken, so it was left out of a bug fix
— removing it renames every per-pair figure the runner writes.

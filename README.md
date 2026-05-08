# Forecasting Stock Returns: Simple Baselines vs. ARMA

A small reproducible measurement pipeline that benchmarks classical time-series
forecasters on **daily log returns** of US equities, sliced into three
price-based tiers.

## What it does

For each ticker, every model produces a one-step-ahead forecast for every
trading day in the test period (`2023-01-01 → 2023-12-31`). The pipeline
scores RMSE / MAE per `(ticker, model)`, summarises per-tier and across
tiers, and renders the figures used in `REPORT.tex`.

## Models

| name        | what it predicts at step `t`                              |
|-------------|-----------------------------------------------------------|
| `naive`     | yesterday: `y[t-1]`                                       |
| `global`    | mean of the entire sample (test set included). **Future-leaking benchmark only.** |
| `expanding` | mean of every observation seen so far                     |
| `ma30`      | mean of the past 30 returns                               |
| `ma60`      | mean of the past 60 returns                               |
| `ma90`      | mean of the past 90 returns                               |
| `arma60`    | ARMA(p,q) AIC-fit on the past 60 returns                  |
| `arma90`    | ARMA(p,q) AIC-fit on the past 90 returns                  |
| `ensemble`  | per-step mean of `expanding`, `ma30`, `ma60`, `ma90`, `arma60`, `arma90` (excludes `naive` and `global`) |

ARMA grid: `p, q ∈ {0..2}`, AIC re-search every 40 steps with cached order in
between.

## Tiers (Russell 3000 universe)

A stock is assigned to a tier purely by its mean adjusted price over the
sample window — no outlier filter:

| tier   | label  | mean band       |
|--------|--------|-----------------|
| tier1  | small  | `< $30`         |
| tier2  | medium | `[$30, $100]`   |
| tier3  | large  | `> $100`        |

The pipeline stochastically samples 30 tickers per tier from
`data/universe/russell3000.txt` (configurable via `--target` / `--seed`).

## Forecasting protocol

- Test period (scored): `2023-01-01 → 2023-12-31`, ≈ 252 trading days.
- Burn-in: data downloaded from `2022-08-15` so even the 90-day-lookback
  models have a full window on test-day 1.
- Each model uses only its own past returns (half-open slice `y[t-L:t]`).
  `global` is the deliberate exception — included as a benchmark.

## Run

```bash
python -m pytest tests/ -x --tb=short            # 39 tests, ~45s
python -m src.runner --seed 42 --force           # full run, ~25 min
```

CLI flags:

| flag                | meaning                                           |
|---------------------|---------------------------------------------------|
| `--target N`        | per-tier target (default 30)                      |
| `--seed S`          | reproducible ticker sampling                      |
| `--refresh-cache`   | bypass the SQLite price cache                     |
| `--tiers tier1,…`   | restrict to a subset of tiers                     |
| `--force`           | bypass the 1-hour ARMA budget gate                |

## Outputs

```
results/test_runs/test_<UTC_TIMESTAMP>/
├── tier1/                  # 30 tickers, mean < $30
│   ├── individual/         # per-ticker per-pair figures
│   ├── grouped/            # cumulative_tier1.png, summary_tier1.{png,csv}
│   ├── predictions/        # raw <TICKER>_<MODEL>.csv
│   ├── analysis/           # per-stock dotplots
│   └── tier1_tickers.txt   # ticker list
├── tier2/                  # 30 tickers, $30 ≤ mean ≤ $100
├── tier3/                  # 30 tickers, mean > $100
├── analysis/
│   ├── cumulative_overall.png
│   ├── summary_overall.{png,csv}
│   ├── score_histogram.{png,csv}
│   └── all_tiers_dotplot_*.png
├── metrics.csv             # tier, ticker, model, rmse, mae, n
├── ticker_tested.csv       # combined tier + ticker
└── manifest.json           # config snapshot, seed, runtime
```

The SQLite price cache lives at `ticker_data/cache.db`; reuse across runs is
automatic.

## Headline result

Pooled across all 90 tickers (30 per tier):

| model       | mean RMSE | win count |
|-------------|-----------|-----------|
| `global`    | 0.0237    | (excluded — benchmark) |
| `expanding` | 0.0238    | **81 / 90** |
| `ma90`      | 0.0239    | 1 |
| `ma60`      | 0.0239    | 8 |
| `ensemble`  | 0.0239    | (excluded — meta) |
| `ma30`      | 0.0241    | 0 |
| `arma90`    | 0.0243    | 0 |
| `arma60`    | 0.0245    | 0 |
| `naive`     | 0.0335    | (excluded — trivial) |

The empirical `naive / baseline` RMSE ratio = `0.0335 / 0.0237` = **1.414**,
matching the theoretical √2 to four significant figures. The expanding mean
is the single best causal forecaster; ARMA does not pay for its complexity.

## Repository layout

```
src/
├── config.py        # tier specs, dates, ARMA grid, lookbacks
├── data.py          # cache-aware load_returns
├── models.py        # Naive / Global / Expanding / MA / ARMA
├── rolling.py       # run_eval (per-model lookback driver)
├── evaluate.py      # run_one_ticker_eval + ARMA cost estimator
├── runner.py        # CLI entry, orchestration, ticker selection
├── selection.py     # per-tier sampling
├── summary.py       # per-tier / overall summaries + score histogram
├── plots.py         # per-pair figure registry
├── analysis/        # per-stock dotplot module
└── storage/db.py    # SQLite ticker price cache

tests/                # pytest suite (39 tests)
data/universe/        # Russell 3000 ticker list
results/test_runs/    # one directory per pipeline run
ticker_data/cache.db  # shared SQLite price cache
REPORT.tex            # full write-up matching this README
```

## Requirements

```
yfinance, pandas, numpy, statsmodels, scipy, matplotlib, pytest
```

`pip install -r requirements.txt` and you're set.

# Peak-hour pipeline — v4 handoff

Picking this up on another machine. Everything below was produced on
2026-08-27 from a full end-to-end run of
`Peak_Hours_Complete_Pipeline_v1.ipynb`.

## What v4 changed

Three evaluation-integrity fixes. No new modelling.

1. **Horizon-matched validation.** v3's walk-forward CV handed test rows real
   `rtm_lag_1/2/4` and real DAM, so it measured a 15-minute-ahead problem
   while the pipeline ships a 10-to-40-day-ahead declaration. Folds now carry
   the production lead gap and are scored under the declaration information
   set. New cell: *SECTION 19b*. The CV and the production forecast call the
   same transform, so they cannot drift apart again.
2. **Disjoint fit / calibrate / test.** v3 estimated the conformal margin on
   a slice and reported coverage on that same slice. Coverage is now measured
   on a third slice untouched by fitting and by margin estimation.
3. **Net load got the same treatment.** The recursive-compounding fix v3
   claimed was applied to RTM only. Net load now uses an exogenous-only
   feature set (57 -> 29 features) and predicts the month in one pass instead
   of 3,744 recursive steps.

Plus: city weather weights anchored on measured state demand instead of
LightGBM split importance, `DECLARATION_LEAD_DAYS` made a first-class config
value, and a value-capture scorecard for scoring declared windows.

## The honest numbers

| Evaluation | MAE (Rs/MWh) | Cap-hit F1 |
|---|---|---|
| 1-block-ahead (v3's published number) | 360 | 0.922 |
| declaration horizon, exogenous | 1133 | 0.316 |
| declaration horizon, frozen | 1178 | 0.412 |
| baseline: DAM price | 1385 | 0.000 |
| baseline: climatology | 1455 | 0.000 |

Net load: MAE 612 MW / MAPE 1.53% at 1-block-ahead vs **3,942 MW / 9.42%** at
the declaration horizon.

Conformal: circular calibration-slice coverage 90.0%, **held-out coverage
80.9%** against a 90% target, mean interval width 6,776 Rs/MWh, per-block
coverage 12%-100%.

Test slice (2026-05 to 2026-08): cap-hit precision 0.118, recall 0.069, F1
0.087 against a 20.8% base rate; MAE 2,412 Rs/MWh. Much worse than the CV
folds, which all sit in 2025 — the Jan-2026 market coupling looks like a
genuine regime break.

## The finding that matters most

Value capture against the ex-post optimal window, last 12 months:

| Method | mean | std | min |
|---|---|---|---|
| baseline: fixed 18:30-21:30 | 87.3% | 10.3 | 71.1% |
| **baseline: last year's optimum** | **99.8%** | **0.4** | **98.9%** |

Declaring last year's same-month optimal window captures 99.1-100% of
achievable value in every month tested. The whole forecasting stack is
competing for ~0.2%. Frame the paper around how little forecast skill the
declaration decision needs, not around price-forecast accuracy.

## TODO on the LAN machine

1. **State demand — blocking the city weights.**
   The cache holds only 2 days (01-01-2023, 01-01-2024), both New Year's Day,
   so the current weights are winter-biased and flagged PROVISIONAL in
   `CITY_WEIGHT_PROVENANCE`.
   ```
   cd scada-cache && python update_cache.py --source state_demand
   ```
   Needs >= 90 distinct days (`STATE_DEMAND_MIN_DAYS`) before the warning
   clears. Alternatively drop a `State_Demand_Weights.csv` (columns
   `State,Mean_MW`) next to the notebook to pin the weights to published CEA
   figures — that path overrides the cache.

2. **Grid frequency.**
   `NR_FRIENDLY_HEADERS` in `scada-cache/scada_cache/tags.py` never mapped a
   frequency column, so the parser dropped it and `cache/nr/*.parquet` has
   only Hydro/Thermal/Solar/Wind/Load. Five candidate header spellings have
   been added ("Frequency", "NR Frequency", "Freq", "FREQ", "Grid Frequency").
   Re-parse:
   ```
   cd scada-cache && rm -rf cache/nr && python update_cache.py --source nr
   ```
   Check the real header in a raw `Stack1Minute\DD-MM-YYYY.csv` row 13 if none
   match, and add it. **Note this rewrites ~69 MB of parquet, so the commit
   diff will be large.**

   `load_frequency_15min()` and `frequency_corroboration()` in *SECTION 23b*
   are already wired up and return block minimum plus fraction of minutes
   below 49.90 Hz — do not use the block mean, it washes out excursions.
   Frequency is all-India (one synchronous grid), so it measures national
   rather than NR-local scarcity. Use it as independent corroboration, not as
   the primary evaluation.

3. **`Previous_Declarations.csv`.**
   Drop it next to the notebook with columns `Month` (YYYY-MM) and
   `Peak_Hours` ("18:30-21:30" or "18:30-20:30, 21:45-22:45"). *SECTION 23b*
   picks it up automatically and scores each declaration against the ex-post
   optimum. The parser was verified against the existing block lists.

4. **Re-run the CV once.**
   `CAP_HIT_THRESHOLD_GRID` was widened to `arange(0.01, 0.96, 0.02)` AFTER
   the recorded run, because the F1 optimum landed on the old grid's bottom
   edge (0.05) — a truncated optimum, not a chosen one. Only that number
   changes. The CV takes ~40 min.

5. **Check the net-load day shape.**
   The new exogenous model puts the evening peak at 18:45 / 70,100 MW; the old
   recursive one put it at 22:15 / 78,640 MW. `PEAK_START_BLOCK` moved 73 ->
   72. The declared window is unchanged at 18:45-21:45, but the exogenous
   model may be under-representing the late-evening ramp.

## Still open (from the pre-v4 review)

- No model-replay backtest yet: the pipeline's own declarations are still not
  scored against that 99.8% baseline. This is the experiment the paper turns on.
- Score weights (0.70/0.08/0.15/0.07, `robust_blend`'s seven coefficients, and
  the rest) remain hand-set with no ablation.
- Selection methods disagree by up to 3 hours; nothing justifies
  `recency_weighted` as the default.
- Sample weights include a magnitude term keyed on the target, so the fitted
  quantiles are quantiles of a reweighted distribution. Worth stating
  alongside the pinball losses.
- Single `random_state=42`, no seed variation, no CIs on any metric.
- Gap-fill in the net-load cell uses `shift(-96)` (tomorrow's same block) on
  ~2.6% of days; those rows are target-contaminated and should be excluded
  from evaluation.

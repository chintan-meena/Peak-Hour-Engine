# Peak_Hour_Engine

NRLDC peak-hour declaration engine. Regulation requires RLDC to declare next
month's peak hours ~10 days before the month starts (7-day regulatory floor).
The long-term goal (user's) is a transparent, scientific, publishable
methodology for choosing those hours instead of ad hoc human judgement —
this repo, the pilot, and an intended IEEE-PES-style operational paper are
all in service of that one goal.

**This is now a standalone repository**, migrated out of the
`power-applications` monorepo (`entitlement-monitor` branch, commits
`b532bb9`..`7b1bcb1`, plus an earlier squashed snapshot on that repo's
`peak-hour-engine` branch) on 2026-09-22. Fully self-contained: scada-cache's
code is copied in under `scada-cache/`, and all data (market prices,
declaration history, SCADA parquet) resolves via `powertools_common` to the
shared OneDrive PowerTools area — no sibling checkout needed on any machine.

**Session note (2026-09-22):** two Claude Code sessions had independently
been working this project for several days without knowing about each other
(see git log below for the overlap). One has taken over as the sole active
thread going forward — if you're picking this up fresh, check for other
active/idle sessions on this repo before assuming you're the only one
touching it.

## Where the real logic actually lives (read this first)

The **declare engine** (`peak_hours/engine/`) is still an empty stub — this
is Segment 2 of the plan below. The real forecasting + declaration decision
currently runs in the *separate, not-yet-migrated* legacy project
`ML_Peak_Hour_Declaration` (its own repo/checkout,
`Peak_Hours_Complete_Pipeline_v1.ipynb`, 51 cells: LightGBM net-load + RTM
price forecasts, then a weighted block-scoring/selection step with ~20
hand-set `Peak_Score` coefficients). That project is being read from for
reference and ported piece by piece — it is not otherwise touched.

What *is* real here today: `peak_hours/io` (market cache, declarations),
`peak_hours/benchmarking` (`ex_post_optimal` — length-matched value-capture
scoring; `replay_scorecard` — multi-candidate side-by-side scoring, supports
in-progress months), `peak_hours/provenance` (run history/fingerprinting).

## The core finding everything else is built around

At the real 10-40 day declaration horizon, the legacy notebook's own honest
metrics collapse (cap-hit F1 0.92 at 1-block-ahead → 0.32-0.41 in CV → 0.087
out-of-time; RTM MAE 360 → ~1150 Rs/MWh). Meanwhile:

- **Thermal**: declaring last year's actual optimal window for the same
  calendar month captures **98.34% mean** of ex-post-optimal value (measured
  here, `peak_hours.benchmarking.ex_post_optimal`, 39 months) — vs 90.69%
  for NRPC/RLDC's own real declarations, and vs only 92.80% for the legacy
  pipeline's own gate-constrained optimum.
- **Hydro**: same story, 98.55% mean, and `hydro_peak_model.py`'s own
  zero-fitted-parameter design (RTM price alone, ranked over M-12/M-24,
  fixed 3hr window) already *measurably outperforms* the hand-weighted score
  out-of-sample (98.5% vs 82.8% value capture, its own ablation).
- **Live, not just backtested**: September 2026's frozen (Aug-22 cutoff)
  model recommendation, replayed via `replay_scorecard.py` against 17/30 real
  elapsed days, landed on **the same window** the zero-parameter last-year
  baseline would have picked, both ~99.1% value capture — first real
  (non-backtest) evidence that the ~20 hand-set `Peak_Score` coefficients may
  be buying nothing over a trivial baseline.

**Working hypothesis, agreed with the user**: default every parameter to
zero/near-zero; a weight is only kept if it proves a measured out-of-sample
improvement via ablation. "No arbitrary human 'feels like' declarations."
This is the thesis the whole migration plan below tests and, so far, confirms.

## Migration plan (merged — engineering + methodology audit, one sequence)

This merges what were previously two separate framings: a software-engineering
migration plan (unify thermal/hydro into one engine, package properly) and a
"parked" methodology audit (strip unjustified `Peak_Score` weights stage by
stage). They're the same goal from two angles — proceed as one sequence,
review checkpoint after each segment, don't skip ahead.

- **Segment 0 — [DONE].** Confirm the simplification thesis before building
  anything: thermal 98.34%, hydro 98.55% (both measured here), September's
  live 99.1% (measured by the other session, `replay_scorecard.py`).
- **Segment 1 — [DONE].** Data-access layer: `io/market_cache.py`,
  `io/declarations.py`, `benchmarking/ex_post_optimal.py`,
  `provenance/registry.py` — ported from the legacy scripts, verified
  byte-for-byte, now fully OneDrive/self-contained (no sibling checkout).
- **Segment 2 — core engine primitives.** `engine/windows.py` (candidate
  enumeration — retires 3 near-duplicate implementations already in
  `ex_post_optimal.build_candidates`, `hydro_peak_model.py`, the notebook's
  inline builder), `engine/gates.py` (evening/morning ramp gate), `engine/
  scoring.py`, `engine/information_set.py` (pluggable lookback:
  same_month_2yr vs forecast_horizon — **this is where "Segment: information
  set / lead time" from the audit side lands**), `engine/aggregation.py` (the
  7 existing monthly aggregation methods, kept selectable for advisory
  scoring only), `engine/select.py` (orchestration).
- **Segment 3 — hydro end-to-end.** Wire `hydro.yaml` through the Segment 2
  engine. Already near-zero-parameter — this is the easy, low-risk proof
  that generalizing the engine doesn't change behavior. Verify against the
  documented 98.40% backtest (99.17% Apr-Sep, 97.58% Oct-Mar, 13/18
  winter-morning correct) within ±0.5pp; pin as regression test.
- **Segment 4 — thermal end-to-end, audited stage by stage.** This is where
  the methodology audit is the actual work, not a formality: for each stage
  — net-load forecast, RTM price forecast, city/weather weighting,
  block/window scoring (`Peak_Score`'s ~20 coefficients, `robust_blend`,
  `PEAK_SELECTION_METHOD`), window-shape/split decision — default to
  zero/near-zero and only keep a parameter that proves itself via
  out-of-sample ablation, exactly like `hydro_peak_model.py` already did for
  its own scoring choice. Backport morning-window candidate admissibility
  (19/72 real historical thermal declarations use one; today's thermal
  candidate builder can't express it). Verify against the known
  `Monthly_Peak_Hours_2026-09.csv` output, explaining any deltas.
- **Segment 5 — seasonal-baseline primary method + advisory wrapper.**
  `engine/seasonal_baseline.py` (promotes `ex_post_optimal`'s existing
  `LastYearOpt_%` side-computation into the actual production declaration),
  `engine/advisory.py` (divergence flag / confidence metadata / bounded
  tie-break). Use `replay_scorecard.py`'s multi-candidate side-by-side
  scoring as the comparison mechanism, including for in-progress months —
  this is how the October 2026 live declaration gets used as an active test
  case (see below), not just history. **Explicit sign-off required before
  this becomes the default** — the single biggest behavior change in the
  whole redesign.
- **Segment 6 — ML advisory stack, two candidates, last on purpose.**
  (a) Port the legacy LightGBM net-load + hurdle-quantile RTM ensemble
  notebook stack. (b) The Hugging Face `amazon/chronos-2` zero-shot pilot
  (see below) — first result: beat a 7-day-seasonal-naive baseline by 32.7%
  MAE at a 4-day horizon, not yet tested at the real 10-40 day horizon or
  against the seasonal-baseline method itself. Bench both against Segment 5's
  primary method; keep whichever, if either, earns a place as advisory.
- **Segment 7 — CLI.** `declare` / `backtest` / `benchmark` / `register`
  subcommands (a minimal `benchmark`/`register` version already exists in
  `peak_hours/cli.py`).
- **Segment 8 — cutover + archive.** Point real declaration work at this
  engine; retire the legacy notebook's role in production (archive it in the
  `ML_Peak_Hour_Declaration` project, don't touch this repo for that).

**Reconciled note on `hydro_peak_model.py`**: its zero-fitted-parameter
design is the existence proof for the whole "prove a weight before keeping
it" principle. Segment 4's thermal audit should explicitly benchmark against
it, not just against the hand-weighted `Peak_Score`.

## Live production state (check before assuming this is just historical)

- Legacy notebook has `PEAK_MONTH="2026-10"` committed (10-day lead
  unchanged) — the actual next declaration RLDC needs to file. Needs real
  SCADA/market data through the Sep-21 cutoff (should exist by now, given
  today's date — check before treating a run as final).
- **User has approved using the live October declaration cycle as an active
  test case for this development work** (compute/compare candidate
  recommendations for it as part of Segment 5/6 testing) — not off-limits,
  but never write anything that could be mistaken for the actual filed
  declaration without saying so explicitly.
- September's frozen (Aug-22 cutoff) recommendation, scored against 17/30
  real elapsed days: 99.1% value capture, identical window to the
  zero-parameter last-year-optimum baseline. Still need the actually-filed
  September window from the user to complete the three-way comparison
  (model vs. baseline vs. what RLDC actually filed).

## The Hugging Face pilot (Chronos-2)

`amazon/chronos-2` (120M params, Apache-2.0, native covariate + multivariate
support, ~1024-step/10.7-day native horizon at 15-min blocks) is being piloted
as a zero-shot candidate for Segment 6's advisory role, per the user's
explicit "no arbitrary human numbers" principle — a pretrained model needs
no hand-engineered features at all. First result (scratch script, not yet in
this package): 4-day-ahead zero-shot forecast beat a 7-day-seasonal-naive
baseline by 32.7% (MAE 3150.8 vs 4679.9 MW, MAPE 5.35% vs 7.89%). Requires
`truststore.inject_into_ssl()` before any huggingface_hub network call on
this machine (TLS-inspecting office proxy, same fix as `wbes`/`iex`).

Needed before this counts as real evidence for the paper: test at the actual
10-40 day declaration horizon (via hourly-granularity forecast +
shape-disaggregation back to 15-min blocks, to avoid the recursive-compounding
trap the legacy notebook's v3→v4 fix already discovered), and benchmark
against the seasonal-baseline method, not just a naive baseline.

## Data layout — everything is OneDrive-backed via `powertools_common`

`D:\power-libraries` is on `PYTHONPATH` (machine-wide env var), so
`from powertools_common import cache_dir, reports_dir` works directly, and
`import scada_cache` works once `peak_hours.paths.SCADA_DIR` is on `sys.path`
(done automatically by code that needs it). `cache_dir(name)` /
`reports_dir(program, *sub)` resolve to the OneDrive PowerTools area —
`%OneDrive%` on Windows, `~/Library/CloudStorage/...` on macOS — so the same
code works unmodified on the MacBook, reading whatever was last synced.

| What | Path | Notes |
|---|---|---|
| RTM/DAM prices | `cache_dir("peak_hours")/{RTM,DAM}_Prices_2022_2026.csv` | refreshed via IEX; LAN/credential-gated |
| Declarations | `cache_dir("peak_hours")/Previous_Declarations*.csv` | built from `.xlsx` via `peak_hours.io.declarations` |
| peak_hours cache dir | `cache_dir("peak_hours")` | == `peak_hours.paths.MARKET_CACHE_DIR` |
| Pipeline outputs | `reports_dir("Peak_Hour_Engine", "monthly_peak_pipeline_outputs")` | == `peak_hours.paths.PIPELINE_ARTIFACTS`; shared, gets overwritten by every notebook run |
| SCADA parquet cache | `scada-cache/scada_cache/config.py` → `$POWERTOOLS_HOME/cache/scada` | code lives in this repo now; data stays OneDrive |
| NRPC actual-filed peak hours | `cache_dir("peakhrs")` | weekly `peakHr-<start>-<end>.csv` block grids; manually-downloaded NOC zips, LAN/GUI-gated, can go stale |

**Known sync gap (partially worked around, not fixed):** on this Windows
machine, `scada-cache/scada_cache/config.py`'s own `powertools_home()` does
**not** check the OneDrive env vars the way `powertools_common.paths.
powertools_home()` does — it resolves to a local-only
`C:\Users\chintan\PowerTools\cache\scada`, separate from the OneDrive copy.
Every time `scada-cache/update_cache.py` is run, manually copy the refreshed
`cache/scada/nr/*.parquet` + `_manifest.json` into the OneDrive copy, or
other machines/checkouts keep seeing a stale snapshot.

## Running things

```
pip install -e ".[dev,forecasting]"      # forecasting extra only needed to run the legacy notebook's stack
python -m pytest tests/unit -q           # no network/LAN needed

python -m peak_hours.benchmarking.ex_post_optimal          # score real declarations vs ex-post optimum
python -m peak_hours.benchmarking.replay_scorecard 2026-09 \
    --candidate "model (frozen Aug-22)=18:45-21:45"          # multi-candidate / partial-month scorer
```

## Running the legacy notebook (still holds the real declare logic)

Lives in the separate `ML_Peak_Hour_Declaration` project, not this repo.
`Peak_Hours_Complete_Pipeline_v1.ipynb`, cell 2: `PEAK_MONTH` +
`DECLARATION_LEAD_DAYS` are the single source of truth. Requires LAN (IEX
API, `\\...\scadashare`, Open-Meteo) — cannot run on the MacBook.

```
python -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3600 \
    --output Peak_Hours_Complete_Pipeline_v1.EXECUTED.ipynb \
    Peak_Hours_Complete_Pipeline_v1.ipynb
```

**Shared-artifact collision**: every run overwrites
`monthly_peak_pipeline_outputs/{Net_Load_15min,Net_Load_Forecast,
Monthly_Peak_Hours_<month>,...}` in place. If a run's output needs to survive
the next run, copy it into a dated subfolder immediately.

Credentials: `iex` reads `load_secrets("iex", ...)` → env var → OS keychain →
`~/.power_tools/iex.json` (per-machine, **not** OneDrive-synced).

## Out of scope unless the user explicitly asks

- Renaming the `NRPC_%` column `hydro_peak_model.py` writes into
  `Hydro_Model_Backtest.csv` (its own output contract in the *other*
  project, bigger blast radius).
- Bringing `D:\Peak_Hours_Declaration` (a separate, retired notebook, no
  `.git`) under version control, or porting its heuristics here.
- Any git-history rewrite.
- Writing anything that could be mistaken for the actual filed October 2026
  declaration without saying so explicitly.

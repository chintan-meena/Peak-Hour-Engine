# Peak_Hour_Engine

NRLDC peak-hour declaration engine. Regulation requires RLDC to declare next
month's peak hours ~10 days before the month starts (7-day regulatory floor).
The long-term goal (user's) is a transparent, scientific, publishable
methodology for choosing those hours instead of ad hoc human judgement —
this repo, the pilot, and an intended IEEE-PES-style operational paper are
all in service of that one goal.

**Who declares: NRLDC, not NRPC.** The peak hours this engine chooses are
declared by the Northern Regional Load Despatch Centre — the system operator.
NRPC, the Northern Regional Power Committee, is a separate coordinating body
and does not issue the declaration. The legacy *code* says "NRPC" in ~25
places — `hydro_peak_model.py`'s `NRPC_%`/`NRPC_Freq_%`/`NRPC_Morning`
columns and its print labels, plus `run_registry.py` — all of it a
misattribution of the declaring authority. Corrected throughout `peak_hours/`
and this file; left as-is in `legacy/`, which is a frozen record. The
methodology write-up (`legacy/docs/Peak_Hour_Declaration.docx`) is clean —
it says "RLDC" 6 times and never "NRPC" — so the paper should tighten that
to **NRLDC** for specificity, not fix an error.

**This is now a standalone repository**, migrated out of the
`power-applications` monorepo (`entitlement-monitor` branch, commits
`b532bb9`..`7b1bcb1`, plus an earlier squashed snapshot on that repo's
`peak-hour-engine` branch) on 2026-09-22. Fully self-contained — **a clone of
this repo plus the shared OneDrive data area is the whole dependency set, on
either machine**:

- scada-cache's code is copied in under `scada-cache/`;
- the entire legacy `ML_Peak_Hour_Declaration` project is imported under
  `legacy/` as frozen reference (see `legacy/README.md`), so no migration
  segment needs another repo checked out to read from;
- `powertools_common` from `power-libraries` is used when importable but is
  no longer required — `peak_hours/_powertools_fallback.py` resolves the same
  OneDrive paths when it isn't, pinned to the real library by
  `tests/unit/test_powertools_fallback.py` so the copy can't drift;
- all data (market prices, declaration history, SCADA parquet) resolves to
  the shared OneDrive PowerTools area, found automatically on Windows and
  macOS alike.

**Physical location**: `D:\power-research\Peak_Hour_Engine\` — deliberately
outside `D:\power-applications\` (a peer of it, like `D:\power-libraries\`),
also 2026-09-22. The original, still-live `ML_Peak_Hour_Declaration` remains at
`D:\power-applications\ML_Peak_Hour_Declaration\` and keeps running until
Segment 8 cuts over; it is no longer *read* from, though — use `legacy/`, which
is the same code frozen at 2026-09-22. A stale, now-empty
`D:\power-applications\Peak_Hour_Engine\` may still exist if a leftover
process (see session note below) was holding it open at move time — safe to
delete once nothing has it open, contains nothing.

**Session note (2026-09-22):** two Claude Code sessions had independently
been working this project for several days without knowing about each other
(see git log below for the overlap). One has taken over as the sole active
thread going forward — if you're picking this up fresh, check for other
active/idle sessions on this repo before assuming you're the only one
touching it.

## Where the real logic actually lives (read this first)

`peak_hours/engine/` now has its Segment 2 primitives (`windows.py`,
`gates.py`, `scoring.py`, `information_set.py`, `aggregation.py`,
`select.py`) but **nothing is wired to real data yet** — no `hydro.yaml`/
`thermal.yaml` config, no `seasonal_baseline.py`, no CLI `declare` command.
The real forecasting + declaration decision still runs in the *not-yet-migrated*
legacy pipeline, now imported here as
`legacy/notebooks/Peak_Hours_Complete_Pipeline_v1.py` (the 51-cell notebook
converted to plain Python, outputs stripped: LightGBM net-load + RTM price
forecasts, then a weighted block-scoring/selection step with ~20 hand-set
`Peak_Score` coefficients — see `SECTION 23`). Port from `legacy/`, not from
`D:\power-applications\ML_Peak_Hour_Declaration\`: same code, but `legacy/` is
frozen, travels with the repo, and exists on the MacBook.

What *is* real here today: `peak_hours/io` (market cache, declarations),
`peak_hours/benchmarking` (`ex_post_optimal` — length-matched value-capture
scoring; `replay_scorecard` — multi-candidate side-by-side scoring, supports
in-progress months), `peak_hours/provenance` (run history/fingerprinting),
`peak_hours/engine` (candidate windows, ramp gates, RTM-only scoring, the 6
generic monthly-aggregation methods, same-month-2yr information set, argmax
selection — all pure functions, unit-tested against synthetic/hand-computed
cases, not yet plugged into `benchmarking` or real market data).

## The core finding everything else is built around

At the real 10-40 day declaration horizon, the legacy notebook's own honest
metrics collapse (cap-hit F1 0.92 at 1-block-ahead → 0.32-0.41 in CV → 0.087
out-of-time; RTM MAE 360 → ~1150 Rs/MWh). Meanwhile:

- **Thermal**: declaring last year's actual optimal window for the same
  calendar month captures **98.34% mean** of ex-post-optimal value (measured
  here, `peak_hours.benchmarking.ex_post_optimal`, 39 months) — vs 90.69%
  for NRLDC's own real declarations, and vs only 92.80% for the legacy
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
- **Segment 2 — [DONE].** `engine/windows.py` (candidate
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

## Data layout — everything is OneDrive-backed

`peak_hours.paths` is the one place that answers "where does this live". It
imports `cache_dir`/`reports_dir` from `powertools_common` when that's
importable (`D:\power-libraries` is on `PYTHONPATH` here via a machine-wide
env var) and from `peak_hours/_powertools_fallback.py` when it isn't, so a
bare clone works either way. `import scada_cache` works once
`peak_hours.paths.SCADA_DIR` is on `sys.path` (done automatically by code that
needs it). `cache_dir(name)` / `reports_dir(program, *sub)` resolve to the
OneDrive PowerTools area — `%OneDrive%` on Windows,
`~/Library/CloudStorage/...` on macOS — so the same code works unmodified on
the MacBook, reading whatever was last synced.

**Don't add a third copy of this path logic.** Two already diverged once (see
the scada-cache fix below) and cost a silently-invisible cache. The fallback
is allowed to exist only because `tests/unit/test_powertools_fallback.py`
asserts it resolves identically to the real library whenever both are present.

| What | Path | Notes |
|---|---|---|
| RTM/DAM prices | `cache_dir("peak_hours")/{RTM,DAM}_Prices_2022_2026.csv` | refreshed via IEX; LAN/credential-gated |
| Declarations | `cache_dir("peak_hours")/Previous_Declarations*.csv` | built from `.xlsx` via `peak_hours.io.declarations` |
| peak_hours cache dir | `cache_dir("peak_hours")` | == `peak_hours.paths.MARKET_CACHE_DIR` |
| Pipeline outputs | `reports_dir("Peak_Hour_Engine", "monthly_peak_pipeline_outputs")` | == `peak_hours.paths.PIPELINE_ARTIFACTS`; shared, gets overwritten by every notebook run |
| SCADA parquet cache | `scada-cache/scada_cache/config.py` → `$POWERTOOLS_HOME/cache/scada` | code lives in this repo now; data stays OneDrive |
| NRLDC actual-filed peak hours | `cache_dir("peakhrs")` | weekly `peakHr-<start>-<end>.csv` block grids; manually-downloaded NOC zips, LAN/GUI-gated, can go stale |

**Fixed 2026-09-22 (was: known sync gap, "partially worked around, not
fixed")**: `scada-cache/scada_cache/config.py`'s own `powertools_home()`
used to skip the OneDrive-detection step `powertools_common.paths.
powertools_home()` does, resolving instead to a plain `~/PowerTools` --
which on this Windows machine was a real local-only folder
(`C:\Users\chintan\PowerTools\cache\scada`), not a symlink into OneDrive,
confirmed empirically. That made the SCADA parquet cache invisible to any
other machine, MacBook included. It now delegates to `powertools_common.
paths.powertools_home()` directly, same as everything else in this package.
Verified end-to-end: `load_frequency()` returns real data (147,648 rows)
through the corrected path, which already had the full synced cache
(`nr/2022..2026.parquet`) sitting there -- the data was fine, only the code
was looking in the wrong place. `scada-cache/update_cache.py` (the raw LAN
ingest, `\\...\scadashare`) is unaffected and still Windows/LAN-only, as it
has to be -- it's the one path in this whole package that genuinely needs
LAN access, and it's never on the read path `declare`/`benchmark`/`register`
use.

**MacBook readiness, audited 2026-09-22** (not yet literally run there, but
every dependency path traced): `declare`/`benchmark`/`register`/tests need
zero LAN access -- only `market_cache.refresh()` (IEX) and `scada-cache/
update_cache.py` (`\\...\scadashare`) are LAN-gated, and neither is on the
read path. No hardcoded Windows path literals in `peak_hours/` itself (`grep`
confirmed clean); `market_cache.py`'s `_IEX_ROOTS` already lists a macOS
candidate (`~/Development/power-libraries`) alongside the Windows one.

**Updated 2026-09-22 (same day), after "make it a complete separate project"**:
the last two things a fresh MacBook checkout needed from elsewhere are gone.
`powertools_common` is now optional (fallback above), and the legacy project
is imported under `legacy/`. What remains is OneDrive signed in and synced, so
`onedrive_root()` finds `~/Library/CloudStorage/OneDrive-*` — data, not code.
Verified by importing `peak_hours.paths` and `scada_cache.config` with
`powertools_common` blocked at `sys.meta_path`: all four roots still resolved
to the correct OneDrive locations. Still not literally run on the MacBook —
that remains an open item, not a claimed-verified one.

## Running things

```
pip install -e ".[dev,forecasting]"      # forecasting extra only needed to run the legacy notebook's stack
python -m pytest tests/unit -q           # no network/LAN needed

python -m peak_hours.benchmarking.ex_post_optimal          # score real declarations vs ex-post optimum
python -m peak_hours.benchmarking.replay_scorecard 2026-09 \
    --candidate "model (frozen Aug-22)=18:45-21:45"          # multi-candidate / partial-month scorer
```

## Running the legacy notebook (still holds the real declare logic)

To *read* it, use `legacy/notebooks/Peak_Hours_Complete_Pipeline_v1.py` in this
repo. To *run* it you still need the original `.ipynb` in the
`ML_Peak_Hour_Declaration` project — `legacy/` is a code-only, outputs-stripped
conversion for reference, and the notebook needs a live kernel anyway.
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

- Rewriting the `NRPC_%` column *in the legacy code itself*
  (`legacy/scripts/hydro_peak_model.py` and the `Hydro_Model_Backtest.csv`
  files already on OneDrive). The name is wrong — see "Who declares" — but
  `legacy/` is a frozen snapshot and rewriting it destroys its value as a
  faithful record of what produced the published numbers. `registry.py` reads
  `NRLDC_%` first and falls back to `NRPC_%`, so the correction lands without
  a migration; the legacy files age out on their own at Segment 8.
- Bringing `D:\Peak_Hours_Declaration` (a separate, retired notebook, no
  `.git`) under version control, or porting its heuristics here.
- Any git-history rewrite.
- Writing anything that could be mistaken for the actual filed October 2026
  declaration without saying so explicitly.

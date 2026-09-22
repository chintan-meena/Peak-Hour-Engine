# Peak_Hour_Engine

A redesign of NRLDC's peak-hour declaration tool: a unified, mature engine
for declaring the daily peak hours (thermal and hydro) that Indian grid
regulation requires be filed ~10 days ahead of each month, replacing an
ad-hoc, hand-weighted scoring pipeline with a methodology where every
parameter has to earn its place via out-of-sample evidence.

**Status:** early — Segment 1 (data access) is done and verified; the
declare engine itself (`peak_hours/engine/`) is still a stub. See
[`CLAUDE.md`](CLAUDE.md) for the full migration plan, current state, and the
empirical findings this project is built around.

## The core finding

Declaring last year's actual optimal window for the same calendar month
captures **~98–99% of the theoretical maximum value** — confirmed both by
backtest (39 months, thermal and hydro) and live in production (September
2026's frozen recommendation scored 99.1% against real elapsed days). The
hand-weighted scoring model it's meant to replace does not measurably beat
this. This project exists to prove that out properly, build a system around
it, and document it well enough to publish.

## Quickstart

```bash
pip install -e ".[dev]"
python -m pytest tests/unit -q

python -m peak_hours.benchmarking.ex_post_optimal
python -m peak_hours.benchmarking.replay_scorecard 2026-09 \
    --candidate "model (frozen Aug-22)=18:45-21:45"
```

Data (market prices, declaration history, SCADA readings) is read from a
shared OneDrive cache via `powertools_common` — nothing to configure locally
beyond having that on `PYTHONPATH`. See `CLAUDE.md` for the full data layout.

## Layout

```
peak_hours/
  io/            market price cache, declaration-history parser
  engine/        candidate windows, scoring, selection (stub — Segment 2+)
  benchmarking/  ex-post-optimal scoring, multi-candidate replay scorecard
  provenance/    run history / fingerprinting, for reproducibility
  forecasting/   the ML advisory stack (LightGBM + a Hugging Face pilot) — Segment 6
scada-cache/     self-contained SCADA parquet cache reader (code only; data is OneDrive)
configs/         thermal.yaml / hydro.yaml engine profiles (Segment 2+)
tests/           unit tests (no network needed) + regression tests (need the SCADA cache)
```

## Private repository

This repo contains references to real grid market data and internal
infrastructure paths and is intentionally private. Treat it accordingly.

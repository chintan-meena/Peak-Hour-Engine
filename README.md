# Peak_Hour_Engine

A redesign of NRLDC's peak-hour declaration tool: a unified, mature engine
for declaring the daily peak hours (thermal and hydro) that Indian grid
regulation requires be filed ~10 days ahead of each month, replacing an
ad-hoc, hand-weighted scoring pipeline with a methodology where every
parameter has to earn its place via out-of-sample evidence.

**Status:** early — Segments 1 (data access) and 2 (engine primitives) are
done and verified; nothing is wired to real data through the new engine yet.
See [`CLAUDE.md`](CLAUDE.md) for the full migration plan, current state, and
the empirical findings this project is built around.

This is a standalone project: a clone plus the shared OneDrive data area is
everything it needs. The tool it replaces is imported whole under
[`legacy/`](legacy/README.md) as frozen reference, so no segment of the
migration requires another repository to be checked out.

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

Data (market prices, declaration history, SCADA readings) is read from the
shared OneDrive PowerTools area, found automatically on both Windows and
macOS — nothing to configure. If the `power-libraries` checkout happens to be
on `PYTHONPATH`, its `powertools_common` is used; if not, a vendored copy of
the same path resolution takes over, so a bare clone still works. See
`CLAUDE.md` for the full data layout.

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
legacy/          the original ML_Peak_Hour_Declaration project, frozen as reference
```

## Private repository

This repo contains references to real grid market data and internal
infrastructure paths and is intentionally private. Treat it accordingly.

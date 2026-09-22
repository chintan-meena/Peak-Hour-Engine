# `legacy/` — the original ML_Peak_Hour_Declaration project, frozen

This is the complete source of the tool this engine replaces, imported so that
Peak_Hour_Engine is a self-contained project: every remaining migration segment
can be worked from inside this repo alone, on any machine, with no checkout of
`power-applications` present.

**Nothing in here is imported by `peak_hours/` and nothing in here runs as part
of the engine.** It is reference material — read it, port from it, cite it in
the paper. The code here is a snapshot as of 2026-09-22; the live, still-running
original stays at `D:\power-applications\ML_Peak_Hour_Declaration\` on the
Windows machine until Segment 8 cuts over.

## Why this is a copy and not a submodule

The original is tracked inside the `power-applications` repo, which carries
~40 other unrelated NRLDC tools. Depending on it would mean this engine only
builds where that repo is also cloned — the exact coupling that makes "run it
on the MacBook" a research project rather than a `git clone`.

## No `.ipynb` files

The original's logic lived in eight Jupyter notebooks. They are imported here
as plain `.py` — code cells in order, markdown cells as comments, **stored
outputs stripped**. That keeps the whole engine a pure-Python project and, as a
side effect, shrinks 7 MB of notebooks to 1 MB of readable code (the difference
was almost entirely base64 PNGs and dumped DataFrames).

They are for *reading*, not executing: cell order, globals, and the `iex`
LAN-only import all assume a live notebook kernel.

## Layout

| Path | What it is |
|---|---|
| `scripts/` | The nine standalone `.py` programs — the real source of truth for everything not in a notebook |
| `notebooks/` | The eight pipeline notebooks, converted to `.py` (see above) |
| `docs/` | `HANDOFF_v4.md` (the best "why" document in the project), `SETUP_MACBOOK.md`, the methodology write-up in `.docx`/`.html` that the paper extends, and scada-cache's own README |
| `figures/` | `Paper_Figures/*.png` as generated on 2026-09-22 |
| `data/` | Two small committed CSVs. The bulk data is **not** here — see below |

## Where the data is (deliberately not in `legacy/`)

Prices, declarations, the SCADA parquet cache and the generated monthly outputs
are not copied. They already live in the shared OneDrive PowerTools area and
resolve automatically through `peak_hours.paths` on both machines:

| Data | Resolves to |
|---|---|
| RTM/DAM prices, `Previous_Declarations*` | `$POWERTOOLS_HOME/cache/peak_hours/` |
| SCADA parquet (net load, solar, wind, frequency) | `$POWERTOOLS_HOME/cache/scada/` |
| Generated declarations, benchmarks, run history | `$POWERTOOLS_HOME/reports/Peak_Hour_Engine/` |

Committing them would duplicate ~90 MB into git and, worse, create a second
copy that drifts from the one the live pipeline writes.

## What has already been ported out of here

| Legacy source | Now lives at | Segment |
|---|---|---|
| `scripts/market_cache.py` | `peak_hours/io/market_cache.py` | 1 |
| `scripts/build_previous_declarations.py` | `peak_hours/io/declarations.py` | 1 |
| `scripts/benchmark_declarations.py` | `peak_hours/benchmarking/ex_post_optimal.py` | 1 |
| `scripts/run_registry.py` | `peak_hours/provenance/registry.py` | 1 |
| `scripts/hydro_peak_model.py` (primitives) | `peak_hours/engine/{windows,gates,scoring,information_set,select}.py` | 2 |
| `notebooks/…v1.py` SECTION 23 (aggregators only) | `peak_hours/engine/aggregation.py` | 2 |

Still to port: the thermal `Peak_Score` weights and composite methods
(SECTION 23, Segment 4 — held back deliberately, pending the weighting audit),
the LightGBM/RTM forecasting stack (Segment 6), reporting (`figures.py`), and
the GUI (deferred).

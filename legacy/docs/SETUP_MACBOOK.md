# Continuing on the MacBook

Everything that can only be obtained on the NRLDC LAN is committed, so a clone
works with no network at all. Read this before the first run.

## 1. Dependencies

```bash
pip install pandas numpy pyarrow requests lightgbm holidays matplotlib \
            nbformat nbclient ipykernel tkcalendar
```

`pyarrow` is required — the SCADA cache is Parquet.

## 2. The one external dependency

The notebook does `from iex import get_trade_data`. That library lives outside
this repo, at `D:\power-libraries\iex` on the Windows machine. Clone it
alongside and put it on the path, **or** skip it: `hydro_peak_model.py`,
`benchmark_declarations.py` and `build_previous_declarations.py` do not need
it — they read the committed price CSVs via `market_cache.py` instead. Only
the v1 notebook needs `iex`, and only to refresh those CSVs.

## 3. What works offline immediately

| Data | Where | Off-LAN? |
|---|---|---|
| NR net load, solar, wind, **frequency** | `$POWERTOOLS_HOME/cache/scada/nr/` | OneDrive |
| State-wise demand | `$POWERTOOLS_HOME/cache/scada/state_demand/` | OneDrive |
| RTM / DAM block prices + volumes | `RTM_Prices_2022_2026.csv`, `DAM_Prices_2022_2026.csv` | committed |
| Weather, 40 stations | `monthly_peak_pipeline_outputs/Weather_*.csv` | committed |
| Declarations, 72 months | `Previous_Declarations*.csv` + `.xlsx` | committed |

The SCADA parquet cache is **not** in the repo. It lives in the shared
PowerTools area — `$POWERTOOLS_HOME/cache/scada`, defaulting to
`~/PowerTools/cache/scada` — beside `cache/wbes` and `cache/iex` from
power-libraries, and syncs between the two machines through OneDrive. On this
MacBook `~/PowerTools` is a symlink to
`~/Library/CloudStorage/OneDrive-grid-india.in/PowerTools`; on the Windows PC
point `POWERTOOLS_HOME` at the same OneDrive folder (or symlink it likewise).
Set `SCADA_CACHE_DIR` to override for a single run.

Verify the caches loaded:

```bash
cd scada-cache && python update_cache.py --status
```

Off the LAN it prints `raw folder not reachable ... using existing cache` and
still reports the committed rows. That message is expected, not an error.

## 4. Run the model

**Use `run_pipeline.py`. It is the only entry point that records what it did.**

```bash
python run_pipeline.py                   # notebook + hydro + benchmark + record
python run_pipeline.py --skip notebook   # everything that runs off-LAN (~5 s)
python run_pipeline.py --list            # what the steps are
```

Off the LAN, skip the notebook step — it is the only one that needs `iex`.
The other three run from the committed caches and price CSVs.

Every run appends a row to `monthly_peak_pipeline_outputs/Run_History.csv`
and copies that run's small result CSVs to `runs/<run_id>/`, so a later run
never destroys an earlier run's numbers:

```bash
python run_registry.py --history    # every run, with its headline scores
python run_registry.py --compare    # diff the last two runs
```

The history stores a content hash of every input alongside the scores. If two
runs share a commit and all fingerprints and still differ, the pipeline is
nondeterministic — `--compare` says so in as many words. This exists because
three runs on 2026-08-27 gave three different declarations for 2026-09 and
nothing on disk recorded why.

The individual pieces still run on their own:

```bash
python hydro_peak_model.py                  # 37-month replay backtest
python hydro_peak_model.py --declare 2026-09
python benchmark_declarations.py            # declarations vs ex-post optimum
```

Expected backtest, so you can tell at a glance whether the caches came across
intact — **98.40% overall, 99.17% Apr–Sep, 97.58% Oct–Mar, morning 13/18**.
Different numbers mean a cache did not travel.

## 5. Back on the LAN

```bash
cd scada-cache && python update_cache.py     # appends new days only
python market_cache.py --refresh             # re-pull IEX prices
```

Both are incremental — the SCADA manifest re-parses only new or changed files.

## Two parser bugs fixed in `scada-cache` this session

Both were silent — they produced plausible-looking data rather than errors, so
neither would have surfaced without checking null counts.

1. **Duplicate column names.** `tags.py` maps five spellings of "Frequency" to
   one clean name, and `parser.py` emitted the column once per alias. The
   duplicate labels then broke the Parquet merge with
   `InvalidIndexError: Reindexing only valid with uniquely valued Index`.
   Fixed by de-duplicating before reindex.

2. **Excel-serialised measurements.** Parts of the 5-minute export write a
   reading of `7184.42` as the date `29-10-1909 04:48:00`. The old
   `to_numeric(errors="coerce")` turned every one into NaN, which deleted
   **83% of the UP series — everything after 08:00**, so both peaks were
   missing while the column still looked populated. Values are now recovered
   from the Excel serial; the recovered points join continuously onto the
   numeric ones.

The `nr` source was checked after the fix and has **zero nulls** across all
2,126,016 rows, so it was never affected.

## Known state

- `Previous_Declarations.csv` is **hydro** (12 blocks). Thermal is 16 blocks;
  regenerate with `python build_previous_declarations.py --peak thermal`.
- `hydro_peak_model.py` is standalone. The `DECLARATION_TYPE` switch discussed
  for the v1 notebook is **not implemented** — the notebook still emits a
  12-block window scored on thermal-market economics.
- `scada-cache/cache/nr.BACKUP/` (69 MB) is git-ignored. Delete it once you
  trust the rebuilt cache.
- Open items from `HANDOFF_v4.md`: #4 (re-run CV with the widened threshold
  grid) and #5 (net-load day shape) are still open. #1 (state demand), #2
  (frequency) and #3 (declarations file) are done.

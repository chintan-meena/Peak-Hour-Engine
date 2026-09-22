# scada-cache

A small, fast, cross-platform cache over the NRLDC **Stack** CSV exports on
`\\192.168.50.247\scadashare`. It parses the messy raw files once into
Parquet, then loads them instantly, and only re-parses files that are new or
changed. The cache is partitioned by year so it can be committed to git and
pulled down already warm when you move between the MacBook and the Windows PC.

Two sources are wired up out of the box:

- **`state_demand`** — the nine state drawals from `Stack 5minute`
  (Punjab, Haryana, Rajasthan, Delhi, UP, Uttarakhand, Himachal, JK, Chandigarh).
- **`nr`** — the NR-level series from `Stack1Minute`
  (NR Load / Solar / Wind / Hydro / Thermal), from which `Net_Load` is derived.

## Why it's built this way

The raw files are awkward: about twelve rows of calendar junk at the top, a
friendly-header row, a machine-tag row, then the data. Worse, the two families
label their columns differently — the 5-minute state file leaves the friendly
headers blank and only carries the state series as machine tags like
`!COMPANIES!PSTCL!REPOT_PS!LINE!PSE_DRW!P.MvMoment`, while the 1-minute file
uses readable headers like `NR Load`. And the 5-minute file stacks each day
twice with different values. All of that is handled once in `parser.py`; the
rest of the code (and your notebook) only ever sees a clean, `Datetime`-indexed
frame. Two quirks worth knowing: the date always comes from the **filename**
(`DD-MM-YYYY.csv`), and the duplicated 5-minute block is collapsed to its first
occurrence, so one file maps to exactly one day.

## Layout

```
scada-cache/
├─ scada_cache/          # the package
│  ├─ tags.py            # column definitions (edit here to add a series)
│  ├─ parser.py          # raw Stack CSV -> tidy DataFrame
│  ├─ config.py          # picks the right path for THIS operating system
│  └─ cache.py           # incremental parquet cache + load helpers
├─ update_cache.py       # CLI: refresh the cache, or show status
├─ config.example.json   # committed template
├─ config.json           # your paths (committed; same on both machines)
└─ config.local.json     # optional per-machine override (git-ignored)
```

The parsed cache is **not** inside this folder. It lives in the shared
PowerTools area, git-ignored, and syncs between machines via OneDrive:

```
$POWERTOOLS_HOME/cache/scada/    # default: ~/PowerTools/cache/scada
├─ nr/2023.parquet
└─ state_demand/2023.parquet
```

## One-time setup on each machine

```bash
git clone <your-repo-url> scada-cache
cd scada-cache
pip install -r requirements.txt
cp config.example.json config.json      # only needed the first time in the repo
```

Then make sure the network share is reachable:

- **Windows** — the UNC paths in `config.json`
  (`\\192.168.50.247\scadashare\Stack 5minute`) work as-is once you can browse
  to the machine in Explorer.
- **macOS** — in Finder use *Go → Connect to Server* →
  `smb://192.168.50.247/scadashare`. It mounts under `/Volumes/scadashare`,
  which is what the `darwin` paths in `config.json` already point to.
- **Linux** — mount the share wherever you like and update the `linux` path.

If a source ever lives somewhere non-standard on one machine only, put it in a
git-ignored `config.local.json` (same shape as `config.json`) or set an env var
like `SCADA_STATE_DEMAND_DIR=/some/path` for a single run — no code change.

## Everyday use

Refresh the cache (only new/changed files are parsed):

```bash
python update_cache.py                 # both sources
python update_cache.py --source nr     # just one
python update_cache.py --status        # show what's cached, ingest nothing
```

Read it from Python:

```python
import scada_cache as sc

sc.update_all()                                   # no-op if the share is offline
states = sc.load("state_demand",                  # -> Datetime-indexed, 9 cols
                 start="2023-01-01", end="2023-12-31")
net    = sc.load_net_load(freq="15min")           # Net_Load = NR_Load - Solar - Wind
```

`load()` reads only the year partitions your date range touches, so slices are
cheap even as history grows.

## Using it from `Peak_Hours_Complete_Pipeline_v9_no_hydro.ipynb`

The cache reproduces the pipeline's own net-load definition, so it can replace
the raw-folder scan in the net-load cell:

```python
import scada_cache as sc
sc.update_source("nr")                            # pick up any new demand files
net_load_raw = (sc.load_net_load(freq="15min")
                  .loc[START_TS:END_TS]
                  .reset_index())                 # columns: Datetime, Net_Load
```

And to bring in the state demands you wanted to add as features:

```python
states = sc.load("state_demand").resample("15min").mean().loc[START_TS:END_TS]
# join on the 15-min Datetime index alongside the weather features
```

That gives you real per-state MW to weight or blend with the weather feature
importances — a far closer proxy to load than raw population.

## Switching machines

The parsed cache (Parquet, partitioned by year) is **not** tracked by git — it
is bulk derived data, and committing it added roughly 170 MB to the repository
before this was changed. It lives in the shared PowerTools area instead:

    $POWERTOOLS_HOME/cache/scada        # default: ~/PowerTools/cache/scada

On both machines that path is a OneDrive folder, so it sits beside `cache/wbes`
and `cache/iex` from power-libraries and follows the same rule those already
do: *"Cache files are not tracked by Git"* (power-libraries/README.md). On the
MacBook `~/PowerTools` is a symlink into
`~/Library/CloudStorage/OneDrive-grid-india.in/PowerTools`; on the Windows PC
set `POWERTOOLS_HOME` to the same OneDrive folder.

So on the second machine you `git pull` for code, and OneDrive has already
delivered the cache — everything loads instantly, no network share needed to
*read*. When you're back on the LAN, `python update_cache.py` appends any new
days and rewrites only the affected year's parquet; OneDrive syncs it across.
Nothing to commit.

Resolution order is `SCADA_CACHE_DIR` env var, then an absolute `cache_dir` in
config, then `$POWERTOOLS_HOME/cache/scada` (see `scada_cache/config.py`).

## Adding another series or state later

Add one line to the relevant map in `scada_cache/tags.py` (a tag substring for
a 5-minute series, or the exact friendly header for a 1-minute one), delete the
affected `$POWERTOOLS_HOME/cache/scada/<source>/` folder so it rebuilds, and run `update_cache.py`.

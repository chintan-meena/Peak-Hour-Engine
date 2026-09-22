"""Fast, incremental, git-friendly cache over the raw Stack CSVs.

Design goals, in order:

1. **Loads instantly.** The cache is Parquet, so ``load("state_demand")`` is a
   single columnar read — no re-parsing hundreds of messy CSVs every run.

2. **Only touches new files.** A per-source manifest records each ingested
   file's size and mtime. On update we glob the raw folder and parse *only*
   files that are new or changed; everything else is skipped. If the network
   share isn't mounted (e.g. you're on the laptop off-LAN), update is a no-op
   and the committed cache still loads.

3. **Git-friendly commits.** The cache is partitioned by year
   (``cache/state_demand/2023.parquet`` etc.). Ingesting a new day rewrites
   only that year's file, so a commit is a small diff, not a whole-history
   binary churn. This is what lets you commit the cache and pull it down
   already-warm on the other machine.

Layout on disk::

    cache/
      state_demand/
        _manifest.json
        2022.parquet
        2023.parquet
      nr/
        _manifest.json
        ...
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config as cfg_mod
from .parser import parse_stack_csv

_MANIFEST = "_manifest.json"


# --------------------------------------------------------------------------- #
# Manifest helpers
# --------------------------------------------------------------------------- #
def _manifest_path(source_dir: Path) -> Path:
    return source_dir / _MANIFEST


def _load_manifest(source_dir: Path) -> dict:
    path = _manifest_path(source_dir)
    if path.exists():
        return json.loads(path.read_text())
    return {"files": {}}


def _save_manifest(source_dir: Path, manifest: dict) -> None:
    _manifest_path(source_dir).write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _file_signature(path: Path) -> dict:
    stat = path.stat()
    return {"size": stat.st_size, "mtime": int(stat.st_mtime)}


def _is_unchanged(manifest: dict, path: Path) -> bool:
    prev = manifest["files"].get(path.name)
    if not prev:
        return False
    sig = _file_signature(path)
    return prev.get("size") == sig["size"] and prev.get("mtime") == sig["mtime"]


# --------------------------------------------------------------------------- #
# Parquet partition helpers
# --------------------------------------------------------------------------- #
def _year_path(source_dir: Path, year: int) -> Path:
    return source_dir / f"{year}.parquet"


def _write_year(source_dir: Path, year: int, new_rows: pd.DataFrame) -> None:
    """Merge ``new_rows`` into that year's parquet (new data wins on conflict)."""
    path = _year_path(source_dir, year)
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_rows])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new_rows.sort_index()
    combined.to_parquet(path, compression="snappy")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def update_source(name: str, cfg: dict | None = None, verbose: bool = True) -> dict:
    """Ingest any new/changed raw files for one source into its cache.

    Returns a small summary dict. Never raises just because the share is
    missing — that is reported and treated as "nothing to do".
    """
    cfg = cfg or cfg_mod.load_config()
    spec = cfg_mod.source_spec(cfg, name)
    profile = spec["profile"]

    source_cache = cfg_mod.cache_dir(cfg) / name
    source_cache.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(source_cache)

    raw_dir = cfg_mod.resolve_source_dir(cfg, name)
    if raw_dir is None or not Path(raw_dir).exists():
        if verbose:
            where = raw_dir if raw_dir else f"(no path set for {cfg_mod.current_os_key()})"
            print(f"[{name}] raw folder not reachable: {where} — using existing cache.")
        return {"source": name, "reachable": False, "ingested": 0, "skipped": 0}

    raw_dir = Path(raw_dir)
    csv_files = sorted(raw_dir.glob("*.csv"))
    if verbose:
        print(f"[{name}] scanning {raw_dir} — {len(csv_files)} csv file(s) present")

    by_year: dict[int, list[pd.DataFrame]] = {}
    ingested, skipped, failed = 0, 0, 0

    for file in csv_files:
        if _is_unchanged(manifest, file):
            skipped += 1
            continue
        try:
            frame = parse_stack_csv(file, profile)
        except Exception as exc:  # noqa: BLE001 - one bad file shouldn't stop the batch
            failed += 1
            if verbose:
                print(f"  ! {file.name}: {exc}")
            continue

        if not frame.empty:
            for year, chunk in frame.groupby(frame.index.year):
                by_year.setdefault(int(year), []).append(chunk)
        manifest["files"][file.name] = _file_signature(file)
        manifest["files"][file.name]["rows"] = int(len(frame))
        # Date-formatted cells the parser had to repair or refuse. Recorded per
        # file so a source that starts emitting them is visible immediately
        # rather than after someone notices a hole in a series.
        recovery = {k: v for k, v in (frame.attrs.get("recovery") or {}).items() if v}
        if recovery:
            manifest["files"][file.name]["recovery"] = recovery
        ingested += 1
        if verbose:
            note = "  [" + ", ".join(f"{k} {v}" for k, v in sorted(recovery.items())) + "]" if recovery else ""
            print(f"  + {file.name}: {len(frame)} rows{note}")

    for year, chunks in by_year.items():
        _write_year(source_cache, year, pd.concat(chunks))

    _save_manifest(source_cache, manifest)
    if verbose:
        print(
            f"[{name}] ingested {ingested}, skipped {skipped}"
            + (f", failed {failed}" if failed else "")
            + f" — cache at {source_cache}"
        )
    return {
        "source": name,
        "reachable": True,
        "ingested": ingested,
        "skipped": skipped,
        "failed": failed,
        "years_written": sorted(by_year),
    }


def update_all(cfg: dict | None = None, verbose: bool = True) -> list[dict]:
    """Update every source declared in the config."""
    cfg = cfg or cfg_mod.load_config()
    return [update_source(name, cfg, verbose) for name in cfg_mod.source_names(cfg)]


def load(name: str, start=None, end=None, columns=None, cfg: dict | None = None) -> pd.DataFrame:
    """Load a source from cache as one ``Datetime``-indexed DataFrame.

    ``start``/``end`` are inclusive and accept anything ``pd.Timestamp`` does.
    Reads only the parquet partitions that overlap the requested year range.
    """
    cfg = cfg or cfg_mod.load_config()
    source_cache = cfg_mod.cache_dir(cfg) / name
    if not source_cache.exists():
        raise FileNotFoundError(
            f"No cache for {name!r} at {source_cache}. Run update_source({name!r}) first."
        )

    parts = sorted(source_cache.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"Cache dir {source_cache} has no parquet partitions yet.")

    start_ts = pd.Timestamp(start) if start is not None else None
    end_ts = pd.Timestamp(end) if end is not None else None

    frames = []
    for part in parts:
        year = int(part.stem)
        if start_ts is not None and year < start_ts.year:
            continue
        if end_ts is not None and year > end_ts.year:
            continue
        frames.append(pd.read_parquet(part))

    if not frames:
        empty = pd.read_parquet(parts[0]).iloc[0:0]
        return empty

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if start_ts is not None:
        df = df[df.index >= start_ts]
    if end_ts is not None:
        df = df[df.index <= end_ts]
    if columns is not None:
        df = df[[c for c in columns if c in df.columns]]
    return df


def load_net_load(start=None, end=None, freq="15min", cfg: dict | None = None) -> pd.DataFrame:
    """Convenience: NR net load = NR_Load - NR_Solar - NR_Wind, resampled.

    Mirrors the v9 pipeline's definition so this cache can drop straight in as
    the source of ``Net_Load``.
    """
    nr = load("nr", start=start, end=end, cfg=cfg)
    for col in ("NR_Load", "NR_Solar", "NR_Wind"):
        if col not in nr.columns:
            raise KeyError(f"'nr' cache is missing {col!r}; got {list(nr.columns)}")
    net = (nr["NR_Load"] - nr["NR_Solar"] - nr["NR_Wind"]).rename("Net_Load").to_frame()
    if freq:
        net = net.resample(freq).mean()
    return net


def cache_status(cfg: dict | None = None) -> pd.DataFrame:
    """One-row-per-source summary of what is currently cached on this machine."""
    cfg = cfg or cfg_mod.load_config()
    rows = []
    for name in cfg_mod.source_names(cfg):
        source_cache = cfg_mod.cache_dir(cfg) / name
        parts = sorted(source_cache.glob("*.parquet")) if source_cache.exists() else []
        manifest = _load_manifest(source_cache) if source_cache.exists() else {"files": {}}
        if parts:
            spans = [pd.read_parquet(p).index for p in parts]
            idx = spans[0].append(spans[1:]) if len(spans) > 1 else spans[0]
            rows.append(
                {
                    "source": name,
                    "files_ingested": len(manifest["files"]),
                    "partitions": len(parts),
                    "rows": int(len(idx)),
                    "start": idx.min(),
                    "end": idx.max(),
                }
            )
        else:
            rows.append(
                {
                    "source": name,
                    "files_ingested": 0,
                    "partitions": 0,
                    "rows": 0,
                    "start": None,
                    "end": None,
                }
            )
    return pd.DataFrame(rows)

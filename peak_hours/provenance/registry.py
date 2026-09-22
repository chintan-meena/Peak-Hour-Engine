#!/usr/bin/env python3
"""Append-only history of pipeline runs, so past scores stay comparable.

Ported from ML_Peak_Hour_Declaration/run_registry.py (logic unchanged) as
part of the Peak_Hour_Engine redesign, Segment 1.

Why this exists
---------------
Three runs on 2026-08-27 produced three different declarations for 2026-09
with nothing on disk recording which code or which input data produced which
number. So every run now stamps itself: a row in Run_History.csv carries the
headline scores, the git commit, whether the tree was dirty, and a
fingerprint of each input file.

This reads harvested metrics from the shared OneDrive output area
(peak_hours.paths.PIPELINE_ARTIFACTS) but writes its own history/archive into
peak_hours.paths.ENGINE_ARTIFACTS -- local run provenance doesn't need to
sync across machines the way finished pipeline output does.

    python -m peak_hours.provenance.registry --label "baseline v4"
    python -m peak_hours.provenance.registry --history
    python -m peak_hours.provenance.registry --compare
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from peak_hours.paths import (
    ENGINE_ARTIFACTS,
    ENGINE_DIR,
    MARKET_CACHE_DIR,
    PIPELINE_ARTIFACTS,
    SCADA_DIR,
)

RUNS_DIR = ENGINE_ARTIFACTS / "runs"
HISTORY = ENGINE_ARTIFACTS / "Run_History.csv"

# Inputs whose content decides the scores. A change in any of these is a
# legitimate reason for the numbers to move; a change in none of them is not.
# Each is (base dir, path relative to it) -- the price/declaration CSVs live
# in the shared OneDrive cache, scada-cache's manifests live in this repo's
# own scada-cache/ (copied in for self-containment).
TRACKED_INPUTS = [
    (MARKET_CACHE_DIR, "RTM_Prices_2022_2026.csv"),
    (MARKET_CACHE_DIR, "DAM_Prices_2022_2026.csv"),
    (MARKET_CACHE_DIR, "Previous_Declarations.csv"),
    (MARKET_CACHE_DIR, "Previous_Declarations_Hydro.csv"),
    (SCADA_DIR, "cache/nr/_manifest.json"),
    (SCADA_DIR, "cache/state_demand/_manifest.json"),
]

# Result files small enough to keep one copy of per run.
ARCHIVE_MAX_BYTES = 2_000_000
ARCHIVE_SKIP_PREFIXES = ("Weather_", "Net_Load_", "RTM_Forecast_", "Future_Weather_")


# ── provenance ──────────────────────────────────────────────────────────────
def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ENGINE_DIR, capture_output=True,
                              text=True, timeout=15).stdout.strip()
    except Exception:
        return ""


def git_state() -> tuple[str, bool]:
    commit = _git("rev-parse", "--short", "HEAD") or "unknown"
    dirty = bool(_git("status", "--porcelain"))
    return commit, dirty


def fingerprint(base: Path, rel: str) -> str:
    """Short content hash, so a touched-but-unchanged file does not look new."""
    p = base / rel
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


# ── metric harvest ──────────────────────────────────────────────────────────
def _one(path: Path, where, col: str, default=None):
    """Pull a single cell out of a result CSV without exploding if it moved."""
    if not path.exists():
        return default
    try:
        d = pd.read_csv(path)
        if callable(where):
            d = d[where(d)]
        if d.empty or col not in d.columns:
            return default
        return d[col].iloc[0]
    except Exception:
        return default


def _latest(pattern: str) -> Path | None:
    hits = sorted(PIPELINE_ARTIFACTS.glob(pattern))
    return hits[-1] if hits else None


def harvest() -> dict:
    """Everything worth comparing between two runs, in one flat row.

    Reads from PIPELINE_ARTIFACTS (whatever the pipeline has produced there)
    -- read-only, never written to.
    """
    m: dict = {}

    bench = PIPELINE_ARTIFACTS / "Declaration_Benchmark.csv"
    if bench.exists():
        d = pd.read_csv(bench)
        m["RLDC_Capture_mean"] = round(float(d["Declared_%"].mean()), 2)
        if "LastYearOpt_%" in d.columns:
            m["LastYearOpt_Capture_mean"] = round(float(d["LastYearOpt_%"].dropna().mean()), 2)

    hyd = PIPELINE_ARTIFACTS / "Hydro_Model_Backtest.csv"
    if hyd.exists():
        d = pd.read_csv(hyd)
        m["Hydro_Model_Capture_mean"] = round(float(d["Model_%"].mean()), 2)
        declared = _declared_capture_column(d)
        if declared is not None:
            m["Hydro_RLDC_Capture_mean"] = round(float(d[declared].mean()), 2)

    return m


#: The column holding "what was actually declared, scored" in
#: Hydro_Model_Backtest.csv. Peak hours are declared by **NRLDC**, but the
#: legacy hydro model labelled this column `NRPC_%` -- NRPC is the regional
#: *committee*, a different body that does not issue the declaration. Files
#: already written to OneDrive carry the old name, so read either: the correct
#: name first, the legacy one as a fallback. Anything this engine *writes*
#: uses NRLDC.
_DECLARED_CAPTURE_COLUMNS = ("NRLDC_%", "NRPC_%")


def _declared_capture_column(d: pd.DataFrame) -> str | None:
    """First declared-capture column present, or None if the file predates both."""
    for name in _DECLARED_CAPTURE_COLUMNS:
        if name in d.columns:
            return name
    return None


# ── archive + append ────────────────────────────────────────────────────────
def archive(run_id: str) -> int:
    dest = RUNS_DIR / run_id
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(PIPELINE_ARTIFACTS.glob("*.csv")):
        if f.name.startswith(ARCHIVE_SKIP_PREFIXES) or f.name == HISTORY.name:
            continue
        if f.stat().st_size > ARCHIVE_MAX_BYTES:
            continue
        shutil.copy2(f, dest / f.name)
        n += 1
    return n


def register(label: str, note: str = "") -> pd.Series:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    commit, dirty = git_state()
    row = {"Run_ID": run_id,
           "Timestamp_UTC": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "Label": label, "Git_Commit": commit, "Git_Dirty": dirty}
    row.update(harvest())
    for base, rel in TRACKED_INPUTS:
        row[f"fp:{Path(rel).name}"] = fingerprint(base, rel)
    row["Note"] = note

    n = archive(run_id)
    hist = pd.read_csv(HISTORY) if HISTORY.exists() else pd.DataFrame()
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    hist.to_csv(HISTORY, index=False)
    print(f"Recorded run {run_id} ({label}) — {n} result file(s) archived to "
          f"{RUNS_DIR / run_id}")
    return pd.Series(row)


# ── reporting ───────────────────────────────────────────────────────────────
KEY_METRICS = ["RLDC_Capture_mean", "LastYearOpt_Capture_mean",
               "Hydro_Model_Capture_mean", "Hydro_RLDC_Capture_mean"]


def show_history(n: int = 12) -> None:
    if not HISTORY.exists():
        sys.exit(f"No history yet at {HISTORY}. Run --label first.")
    h = pd.read_csv(HISTORY).tail(n)
    cols = ["Run_ID", "Label", "Git_Commit", "Git_Dirty"] + [c for c in KEY_METRICS if c in h.columns]
    print(h[cols].to_string(index=False))


def compare() -> None:
    if not HISTORY.exists():
        sys.exit(f"No history yet at {HISTORY}.")
    h = pd.read_csv(HISTORY)
    if len(h) < 2:
        sys.exit("Need at least two recorded runs to compare.")
    a, b = h.iloc[-2], h.iloc[-1]
    print(f"{a.Run_ID} ({a.Label})  ->  {b.Run_ID} ({b.Label})\n")

    fps = [c for c in h.columns if c.startswith("fp:")]
    changed = [c[3:] for c in fps if a.get(c) != b.get(c)]
    print("inputs changed: " + (", ".join(changed) if changed else "NONE"))
    if not changed and a.get("Git_Commit") == b.get("Git_Commit"):
        print("  -> same code, same inputs: any metric change below is nondeterminism.")
    print()
    for c in KEY_METRICS:
        if c not in h.columns:
            continue
        x, y = a.get(c), b.get(c)
        if pd.isna(x) and pd.isna(y):
            continue
        flag = "" if str(x) == str(y) else "   <-- changed"
        print(f"  {c:28s} {str(x):>22s}  ->  {str(y):>22s}{flag}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", default=None, help="record the current outputs as a run")
    p.add_argument("--note", default="", help="free-text note stored with the run")
    p.add_argument("--history", action="store_true", help="print recorded runs")
    p.add_argument("--compare", action="store_true", help="diff the last two runs")
    a = p.parse_args()

    if a.history:
        show_history()
    elif a.compare:
        compare()
    elif a.label is not None:
        register(a.label, a.note)
    else:
        p.print_help()


if __name__ == "__main__":
    main()

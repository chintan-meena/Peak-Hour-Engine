#!/usr/bin/env python3
"""Append-only history of pipeline runs, so past scores stay comparable.

Why this exists
---------------
Three runs on 2026-08-27 produced three different declarations for 2026-09
(18:45-21:45 score 44.78, then a split window score 48.43, then 18:45-21:45
score 46.82). Nothing on disk recorded which code or which input data produced
which number: the notebook was saved at 19:16 while its outputs were written at
20:15, and the price CSVs had been refreshed at 19:12 in between. The
differences were explainable, but only by forensics after the fact.

So every run now stamps itself. A row in ``Run_History.csv`` carries the
headline scores, the git commit, whether the tree was dirty, and a fingerprint
of each input file. Two runs with the same fingerprints and commit must produce
the same scores; if they do not, the pipeline is nondeterministic and the
history is what proves it.

The small result CSVs are also copied into ``runs/<run_id>/`` so a past run's
numbers survive the next run overwriting them. Bulk data (the cleaned net-load
series, weather, per-block forecasts) is deliberately not archived - it is
reproducible from the caches and would dominate the repo.

    python run_registry.py --label "baseline v4"    # record current outputs
    python run_registry.py --history                # show past runs
    python run_registry.py --compare                # diff the last two runs
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
from powertools_common import cache_dir, reports_dir

PROJECT_DIR = Path(__file__).resolve().parent
MARKET_CACHE_DIR = cache_dir("peak_hours")
ARTIFACTS = reports_dir("Peak_Hour_Engine", "monthly_peak_pipeline_outputs")
RUNS_DIR = ARTIFACTS / "runs"
HISTORY = ARTIFACTS / "Run_History.csv"

# Inputs whose content decides the scores. A change in any of these is a
# legitimate reason for the numbers to move; a change in none of them is not.
# Each is (base dir, path relative to it) -- price/declaration CSVs live in
# the shared OneDrive cache, everything else stays relative to this checkout.
TRACKED_INPUTS = [
    (MARKET_CACHE_DIR, "RTM_Prices_2022_2026.csv"),
    (MARKET_CACHE_DIR, "DAM_Prices_2022_2026.csv"),
    (MARKET_CACHE_DIR, "Previous_Declarations.csv"),
    (MARKET_CACHE_DIR, "Previous_Declarations_Hydro.csv"),
    (PROJECT_DIR, "scada-cache/cache/nr/_manifest.json"),
    (PROJECT_DIR, "scada-cache/cache/state_demand/_manifest.json"),
    (PROJECT_DIR, "Peak_Hours_Complete_Pipeline_v1.ipynb"),
    (PROJECT_DIR, "hydro_peak_model.py"),
    (PROJECT_DIR, "benchmark_declarations.py"),
]

# Result files small enough to keep one copy of per run.
ARCHIVE_MAX_BYTES = 2_000_000
ARCHIVE_SKIP_PREFIXES = ("Weather_", "Net_Load_", "RTM_Forecast_", "Future_Weather_")


# ── provenance ──────────────────────────────────────────────────────────────
def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=PROJECT_DIR, capture_output=True,
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
    hits = sorted(ARTIFACTS.glob(pattern))
    return hits[-1] if hits else None


def late_evening_bias() -> float | None:
    """Forecast minus actual net load over 21:45-23:00, same calendar month.

    The exogenous net-load model decays after 19:15 while the measured evening
    is a plateau to ~23:00. That single defect pushes the declared window
    early, so it is tracked as a number rather than rediscovered each time.
    Negative = the forecast is under-calling the late evening.
    """
    fc_path = ARTIFACTS / "Net_Load_Forecast.csv"
    if not fc_path.exists():
        return None
    try:
        fc = pd.read_csv(fc_path, parse_dates=["Datetime"])
        col = next(c for c in fc.columns if "Net_Load" in c and c != "Datetime")
        month = fc["Datetime"].dt.to_period("M").mode().iloc[0]

        sys.path.insert(0, str(PROJECT_DIR / "scada-cache"))
        import scada_cache as sc
        nr = sc.load("nr")
        nl = (nr["NR_Load"] - nr["NR_Solar"] - nr["NR_Wind"]).resample("15min").mean()
        nl = nl[(nl > 10_000) & (nl < 100_000)]

        # Same calendar month in prior years - the only honest comparator.
        hist = nl[nl.index.month == month.month]
        hist = hist[hist.index.year < month.year]
        if hist.empty:
            return None
        blk = lambda idx: idx.hour * 4 + idx.minute // 15 + 1
        late = range(88, 93)                                  # 21:45-23:00
        h = hist.groupby(blk(hist.index)).mean().reindex(late).mean()
        f = fc.assign(B=blk(fc["Datetime"].dt)).groupby("B")[col].mean().reindex(late).mean()
        return round(float(f - h), 1)
    except Exception:
        return None


def harvest() -> dict:
    """Everything worth comparing between two runs, in one flat row."""
    m: dict = {}

    decl = _latest("Monthly_Peak_Hours_*.csv")
    if decl is not None:
        d = pd.read_csv(decl)
        m["Peak_Month"] = d["Month"].iloc[0]
        m["Declared"] = d["Peak_Hours"].iloc[0]
        m["Pattern"] = d.get("Pattern", pd.Series([None])).iloc[0]
        m["Score"] = round(float(d["Score"].iloc[0]), 4)
        m["Selection_Method"] = d.get("Selection_Method", pd.Series([None])).iloc[0]

    cv = _latest("RTM_v4_CV_Summary_*.csv")
    if cv is not None:
        pick = lambda key: (lambda d: d["Evaluation"].str.contains(key, case=False, na=False))
        m["RTM_MAE_declhorizon"] = _one(cv, pick("declaration horizon \\(exogenous"), "MAE")
        m["RTM_F1_declhorizon"] = _one(cv, pick("declaration horizon \\(exogenous"), "CapHit_F1@0.5")
        m["RTM_MAE_1block"] = _one(cv, pick("1-block-ahead"), "MAE")

    hm = _latest("RTM_v4_HonestMetrics_*.csv")
    if hm is not None:
        h = pd.read_csv(hm)
        for src, dst in (("Test_Coverage_Honest", "Conformal_Coverage"),
                         ("Test_Mean_Interval_Width", "Conformal_Width"),
                         ("Test_MAE", "RTM_Test_MAE"),
                         ("Cap_Hit_Threshold", "CapHit_Threshold"),
                         ("City_Weight_Provenance", "Weight_Provenance")):
            if src in h.columns:
                m[dst] = h[src].iloc[0]

    nb = _latest("NetLoad_Backtest_*.csv")
    if nb is not None:
        d = pd.read_csv(nb)
        sel = d[d["Evaluation"].str.contains("declaration horizon", na=False)]
        if len(sel):
            m["NetLoad_MAE_declhorizon"] = round(float(sel["MAE"].mean()), 1)
            m["NetLoad_MAPE_declhorizon"] = round(float(sel["MAPE_%"].mean()), 3)

    m["LateEvening_Bias_MW"] = late_evening_bias()

    bench = ARTIFACTS / "Declaration_Benchmark.csv"
    if bench.exists():
        d = pd.read_csv(bench)
        m["NRPC_Capture_mean"] = round(float(d["Declared_%"].mean()), 2)
        if "LastYearOpt_%" in d.columns:
            m["LastYearOpt_Capture_mean"] = round(float(d["LastYearOpt_%"].dropna().mean()), 2)

    hyd = ARTIFACTS / "Hydro_Model_Backtest.csv"
    if hyd.exists():
        d = pd.read_csv(hyd)
        m["Hydro_Model_Capture_mean"] = round(float(d["Model_%"].mean()), 2)
        m["Hydro_NRPC_Capture_mean"] = round(float(d["NRPC_%"].mean()), 2)
        # The non-circular pair: no objective tried so far optimises frequency,
        # so this is the comparison that survives review. Tracked alongside
        # value capture precisely so a change that trades one for the other is
        # visible instead of looking like a plain regression.
        if "Model_Freq_%" in d.columns:
            m["Hydro_Model_Freq_mean"] = round(float(d["Model_Freq_%"].mean()), 2)
            m["Hydro_NRPC_Freq_mean"] = round(float(d["NRPC_Freq_%"].mean()), 2)
            m["Hydro_Freq_Wins"] = f"{int((d['Model_Freq_%'] > d['NRPC_Freq_%']).sum())}/{len(d)}"

    sweep = _latest("RTM_v4_CapHitThresholdSweep_*.csv")
    if sweep is not None:
        d = pd.read_csv(sweep).sort_values("Threshold")
        best = d.loc[d["F1"].idxmax(), "Threshold"]
        m["CapHit_F1_Optimum"] = best
        # An optimum sitting on the grid edge is not a chosen operating point.
        m["CapHit_Optimum_OnEdge"] = bool(best in (d["Threshold"].min(), d["Threshold"].max()))
    return m


# ── archive + append ────────────────────────────────────────────────────────
def archive(run_id: str) -> int:
    dest = RUNS_DIR / run_id
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(ARTIFACTS.glob("*.csv")):
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
KEY_METRICS = ["Declared", "Score", "RTM_MAE_declhorizon", "RTM_F1_declhorizon",
               "RTM_Test_MAE", "Conformal_Coverage", "NetLoad_MAE_declhorizon",
               "LateEvening_Bias_MW", "NRPC_Capture_mean", "LastYearOpt_Capture_mean",
               "Hydro_Model_Capture_mean", "Hydro_Model_Freq_mean", "Hydro_NRPC_Freq_mean",
               "Hydro_Freq_Wins", "CapHit_F1_Optimum", "CapHit_Optimum_OnEdge"]


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

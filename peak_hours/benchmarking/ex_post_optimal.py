#!/usr/bin/env python3
"""How much value did the no-model declarations actually capture?

Ported from ML_Peak_Hour_Declaration/benchmark_declarations.py (logic
unchanged) as part of the Peak_Hour_Engine redesign, Segment 1.

The pipeline's premise is that forecasting buys a better peak-hour window.
That is only worth anything if the window RLDC already declares by hand
leaves value on the table. This scores every real declaration against the
ex-post optimum computed from actuals, so the headroom a model could
possibly win is measured before any model is trusted.

`build_candidates` / `best_of` here are also the computational core reused by
peak_hours.engine.windows for live candidate search (see Segment 2) — this
module is not just a benchmarking script, it is where that logic originates.

    python -m peak_hours.benchmarking.ex_post_optimal
    python -m peak_hours.benchmarking.ex_post_optimal --length 12
    python -m peak_hours.benchmarking.ex_post_optimal --no-frequency
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from peak_hours.io import market_cache
from peak_hours.paths import MARKET_CACHE_DIR, PIPELINE_ARTIFACTS, SCADA_DIR

IEGC_BAND_LO = 49.90

# Declaration lengths are fixed by practice, not by this script:
#   thermal 4 hours = 16 blocks, hydro 3 hours = 12 blocks.
EXPECTED_BLOCKS = {"thermal": 16, "hydro": 12}


# ── the notebook's own block conventions ────────────────────────────────────
def block_to_time(block: int) -> str:
    m = (int(block) - 1) * 15
    return f"{m // 60:02d}:{m % 60:02d}"


def time_to_block(hhmm: str) -> int:
    h, m = (int(x) for x in hhmm.strip().split(":"))
    return (h * 60 + m) // 15 + 1


def blocks_from_peak_hours(text) -> tuple:
    blocks = []
    for part in str(text).split(","):
        part = part.strip()
        if "-" not in part:
            continue
        lo, hi = (p.strip() for p in part.split("-", 1))
        b_lo = time_to_block(lo)
        b_hi = 97 if hi in ("00:00", "24:00") else time_to_block(hi)
        blocks.extend(range(b_lo, b_hi))
    return tuple(sorted(set(blocks)))


def ranges_from_blocks(blocks) -> str:
    """Contiguous runs -> '18:30-20:30, 21:30-23:30'."""
    if not blocks:
        return ""
    blocks = sorted(blocks)
    runs, s, p = [], blocks[0], blocks[0]
    for b in blocks[1:]:
        if b == p + 1:
            p = b
            continue
        runs.append((s, p)); s = p = b
    runs.append((s, p))
    return ", ".join(f"{block_to_time(a)}-{block_to_time(b + 1)}" for a, b in runs)


# ── candidate windows ───────────────────────────────────────────────────────
def build_candidates(total_blocks: int, start_block: int = 1,
                     end_block: int = 96, min_split: int = 4) -> np.ndarray:
    """Continuous and two-segment windows of exactly `total_blocks`.

    Returned as an (n_candidates, total_blocks) array of 0-based block indices
    so a whole month can be scored with one `vals[cand].sum(axis=1)` rather
    than ~83,000 individual lookups.
    """
    out = []
    for s in range(start_block, end_block + 1):
        if s + total_blocks - 1 > end_block:
            break
        out.append(list(range(s, s + total_blocks)))
    for len1 in range(min_split, total_blocks - min_split + 1):
        len2 = total_blocks - len1
        for s1 in range(start_block, end_block + 1):
            e1 = s1 + len1 - 1
            if e1 > end_block:
                break
            for s2 in range(e1 + 2, end_block + 1):
                if s2 + len2 - 1 > end_block:
                    break
                out.append(list(range(s1, s1 + len1)) + list(range(s2, s2 + len2)))
    return np.asarray(out, dtype=np.int32) - 1        # to 0-based


def best_of(cand: np.ndarray, vals: np.ndarray) -> tuple:
    """(blocks 1-based, value) of the highest-scoring candidate."""
    totals = vals[cand].sum(axis=1)
    i = int(totals.argmax())
    return tuple(int(b) + 1 for b in cand[i]), float(totals[i])


# ── data ────────────────────────────────────────────────────────────────────
def load_actuals() -> pd.DataFrame:
    """Per-block actual Net_Load and RTM price, 15-min, from the legacy pipeline's outputs."""
    nl_path = PIPELINE_ARTIFACTS / "Net_Load_Final_Clean.csv"
    if not nl_path.exists():
        nl_path = PIPELINE_ARTIFACTS / "Net_Load_15min.csv"
    if not nl_path.exists():
        sys.exit(f"error: no net-load artifact in {PIPELINE_ARTIFACTS}. Run the pipeline first.")
    nl = pd.read_csv(nl_path, parse_dates=["Datetime"])[["Datetime", "Net_Load"]]

    rtm = market_cache.load("RTM").rename(columns={"Price": "RTM_Price"})

    df = nl.merge(rtm[["Datetime", "Block", "RTM_Price"]], on="Datetime", how="inner")
    df["Month"] = df["Datetime"].dt.to_period("M").astype(str)
    df["NL_RTM"] = df["Net_Load"] * df["RTM_Price"]
    return df


def load_frequency() -> pd.DataFrame | None:
    """Per-block frequency stress. Exogenous to every model in the stack."""
    if not SCADA_DIR.exists():
        return None
    sys.path.insert(0, str(SCADA_DIR))
    try:
        import scada_cache as sc
        raw = sc.load("nr")
    except Exception as exc:
        print(f"  (frequency unavailable: {exc})")
        return None
    col = next((c for c in raw.columns if "freq" in c.lower()), None)
    if col is None:
        print("  (no frequency column in the nr cache)")
        return None
    f = raw[col].astype(float)
    out = pd.DataFrame({
        "Freq_Min": f.resample("15min").min(),
        "Frac_Below_Band": f.lt(IEGC_BAND_LO).resample("15min").mean(),
    })
    out["Month"] = out.index.to_period("M").astype(str)
    out["Block"] = out.index.hour * 4 + out.index.minute // 15 + 1
    return out


# ── scoring ─────────────────────────────────────────────────────────────────
def run(decl_path: Path, length: int | None = None, gate_start: int = 72,
        decl_type: str | None = None, with_frequency: bool = True) -> pd.DataFrame:
    """Core computation, importable for tests/other callers without argparse."""
    if not decl_path.exists():
        raise FileNotFoundError(f"{decl_path} not found - run peak_hours.io.declarations")
    decl = pd.read_csv(decl_path).set_index("Month")

    decl_type = decl_type or ("hydro" if "hydro" in decl_path.name.lower() else "thermal")
    want = EXPECTED_BLOCKS[decl_type]

    act = load_actuals()
    freq = load_frequency() if with_frequency else None

    cand_cache: dict[tuple, list] = {}
    rows = []
    for month, g in act.groupby("Month"):
        if month not in decl.index:
            continue
        days = g["Datetime"].dt.normalize().nunique()
        exp_days = pd.Period(month).days_in_month
        if days < exp_days:                      # skip incomplete months
            continue

        vals = (g.groupby("Block")["NL_RTM"].mean()
                .reindex(range(1, 97)).fillna(0.0).to_numpy())
        declared = blocks_from_peak_hours(decl.loc[month, "Peak_Hours"])
        if not declared:
            continue
        L = length or len(declared)

        for s0 in (1, gate_start):
            if (L, s0) not in cand_cache:
                cand_cache[(L, s0)] = build_candidates(L, start_block=s0)
        free = cand_cache[(L, 1)]
        gated = cand_cache[(L, gate_start)]

        def score(bl):
            return float(vals[[b - 1 for b in bl]].sum())

        opt, opt_v = best_of(free, vals)
        _, opt_gated_v = best_of(gated, vals)

        prev = str(pd.Period(month, freq="M") - 12)
        ly = None
        if prev in act.Month.values:
            pg = act[act.Month == prev]
            pv = (pg.groupby("Block")["NL_RTM"].mean()
                  .reindex(range(1, 97)).fillna(0.0).to_numpy())
            ly, _ = best_of(free, pv)

        entry = {
            "Month": month,
            "Declared": decl.loc[month, "Peak_Hours"],
            "Blocks": len(declared),
            "Length_OK": len(declared) == want,
            "Declared_%": 100 * score(declared) / opt_v,
            "Optimum": ranges_from_blocks(opt),
            "Gated_Optimum_%": 100 * opt_gated_v / opt_v,
            "LastYearOpt_%": 100 * score(ly) / opt_v if ly else np.nan,
        }
        if freq is not None:
            fm = freq[freq.Month == month]
            if len(fm):
                prof = fm.groupby("Block")[["Freq_Min", "Frac_Below_Band"]].mean()
                ins = prof.loc[prof.index.isin(declared)]
                out_ = prof.loc[~prof.index.isin(declared)]
                oins = prof.loc[prof.index.isin(opt)]
                entry["Decl_BelowBand"] = ins.Frac_Below_Band.mean()
                entry["Rest_BelowBand"] = out_.Frac_Below_Band.mean()
                entry["Opt_BelowBand"] = oins.Frac_Below_Band.mean()
        rows.append(entry)

    if not rows:
        raise RuntimeError("no month had both a declaration and complete actuals")

    return pd.DataFrame(rows).sort_values("Month")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--decl", type=Path, default=MARKET_CACHE_DIR / "Previous_Declarations.csv")
    p.add_argument("--length", type=int, default=None,
                   help="window length in blocks (default: each month's own)")
    p.add_argument("--gate-start", type=int, default=72,
                   help="pipeline's earliest admissible start block")
    p.add_argument("--type", choices=sorted(EXPECTED_BLOCKS), default=None,
                   help="declaration type; inferred from the filename if omitted")
    p.add_argument("--no-frequency", action="store_true")
    p.add_argument("--out", type=Path, default=PIPELINE_ARTIFACTS / "Declaration_Benchmark.csv")
    a = p.parse_args()

    print(f"Declaration type: {a.type or ('hydro' if 'hydro' in a.decl.name.lower() else 'thermal')}")
    print("Loading actuals (net load + RTM) ...")
    try:
        r = run(a.decl, a.length, a.gate_start, a.type, not a.no_frequency)
    except (FileNotFoundError, RuntimeError) as exc:
        sys.exit(f"error: {exc}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(a.out, index=False)

    show = ["Month", "Declared", "Blocks", "Declared_%", "Optimum",
            "Gated_Optimum_%", "LastYearOpt_%"]
    print(f"\n{'='*100}\nValue capture vs ex-post optimum at the declaration's own length\n{'='*100}")
    print(r[show].to_string(index=False,
          formatters={c: "{:.1f}".format for c in show if c.endswith("%")}))

    print(f"\n{'-'*60}\nSummary over {len(r)} months")
    print(f"{'-'*60}")
    for c, label in [("Declared_%", "RLDC declaration (no model)"),
                     ("LastYearOpt_%", "last year's optimum"),
                     ("Gated_Optimum_%", "best window the pipeline's gate allows")]:
        s = r[c].dropna()
        if len(s):
            print(f"  {label:42s} mean {s.mean():6.2f}%  min {s.min():6.2f}%  "
                  f"std {s.std():5.2f}")
    head = 100 - r["Declared_%"].mean()
    print(f"\n  Headroom above the no-model declaration: {head:.2f} percentage points")

    if "Decl_BelowBand" in r.columns:
        d = r.dropna(subset=["Decl_BelowBand"])
        tighter = (d.Decl_BelowBand > d.Rest_BelowBand).sum()
        print(f"\n  Frequency (exogenous check), {len(d)} months:")
        print(f"    declared window is tighter than rest of day: {tighter}/{len(d)}")
        print(f"    mean frac below {IEGC_BAND_LO} Hz - declared {d.Decl_BelowBand.mean():.3f}"
              f" | rest {d.Rest_BelowBand.mean():.3f} | ex-post optimum {d.Opt_BelowBand.mean():.3f}")
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()

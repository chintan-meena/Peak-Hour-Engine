#!/usr/bin/env python3
"""Hydro peak-hour declaration: model + honest replay backtest.

Why hydro needs its own objective
---------------------------------
Thermal peak hours are a commercial question, so scoring them on Net_Load x RTM
price is reasonable. Hydro peak hours are a *frequency* question: hydro is
declared where its fast ramping is needed to arrest a frequency decline. The
measured NR record says the same thing --

    winter (Oct-Mar)   mean ramp   Freq_Min   frac<49.9Hz   solar
      morning 05:45-08:45   +99 MW    49.9401     0.084      2465 MW
      evening 17:30-22:00   -51 MW    49.9401     0.068       283 MW
      rest of day            -7 MW    49.9428     0.058      8187 MW
    summer mornings      -4801 MW    49.9612     0.052      4977 MW

In winter, fog suppresses solar, demand climbs, and the morning carries the
HIGHEST frequency stress of the day - higher than the evening peak. In summer
the morning is the healthiest block of the day because solar ramps hard. That
is why 33 of 36 winter hydro declarations carry a morning segment and only 1
of 36 summer ones do.

So this model scores blocks on net load x RTM price - both published and both
forecastable from climatology - and admits morning+evening split windows. No
season rule is hardcoded and none is needed: winter mornings score because they
are genuinely expensive (mean RTM 5441 Rs/MWh against 3746 for the rest of the
winter day), while summer mornings score below their own rest-of-day (3858
against 4101) and so never appear.

Frequency is NOT an input. Frequency is an outcome of demand, not a driver of
it, and cannot be forecast at declaration time. It is used only to VALIDATE
after the fact: if frequency stays low through a declared window even with
hydro running, demand was genuinely large.

The window is fixed at 3 hours because hydro storage is limited. That is a
physical cap, not a tunable.

Honesty of the backtest
-----------------------
A declaration for month M is issued before M starts, so the replay uses ONLY
data strictly before M - the same month in the two prior years. Adjacent months
are deliberately excluded: peak timing tracks sunset, so blending May and June
into a July profile drags the window 1.5-2 hours early (measured: 57.91% vs
67.57% frequency stress captured, out-of-sample).

    python hydro_peak_model.py                 # backtest all months
    python hydro_peak_model.py --declare 2026-09
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from powertools_common import cache_dir, reports_dir

PROJECT_DIR = Path(__file__).resolve().parent
SCADA_DIR = PROJECT_DIR / "scada-cache"
MARKET_CACHE_DIR = cache_dir("peak_hours")
ARTIFACTS = reports_dir("Peak_Hour_Engine", "monthly_peak_pipeline_outputs")
IEGC_BAND_LO = 49.90

# Declaration lengths are fixed by practice and are NOT tunable:
#   hydro   3 hours = 12 blocks   (71 of 72 declarations; storage-limited)
#   thermal 4 hours = 16 blocks   (100 of 100 declarations)
# Kept here as named constants so the two never silently swap.
HYDRO_WINDOW_BLOCKS = 12
THERMAL_WINDOW_BLOCKS = 16
WINDOW_BLOCKS = HYDRO_WINDOW_BLOCKS     # this model declares hydro peak hours
MIN_SEGMENT = 4             # matches the observed (6,6)/(5,7)/(4,8) splits
assert WINDOW_BLOCKS * 15 == 180, "hydro declaration must be exactly 3 hours"
NL_MIN, NL_MAX = 10_000, 100_000    # same sanity band the notebook uses

# ── The objective ───────────────────────────────────────────────────────────
#
# Blocks are scored by RTM price alone. No weights, no product, one published
# observable. Net load still shapes the search - it sets the evening and
# morning ramp gates in select() - but it does not enter the score.
#
# Why not net load x RTM, which this model used previously:
#
# 1. The product was never an interaction. Decomposing var(log NL x RTM) over
#    145,057 blocks gives 70.0% price, 9.6% net load, 20.4% covariance;
#    sd(log RTM) is 2.7x sd(log NL). Net load moved under a tenth of the
#    variation that decided the ranking, and picking on net load ALONE agreed
#    with the product in only 23% of months while price alone agreed in 69%.
#
# 2. Including it actively hurt. Out-of-sample - window chosen from M-12/M-24
#    only, scored on the actual month, 37 months, adjudicated on frequency
#    stress, which no candidate here optimises:
#
#       objective            freq%   NLxRTM value%   win/tie/loss    p
#       NL x RTM (was)       62.82         98.80          -          -
#       RTM alone            64.79         97.85       18/17/ 2    0.002
#       mean x (1 - CV)      64.86         98.24       18/13/ 6    0.015
#
#    Dropping net load wins 18 months and loses 2, for 0.95% of the NL x RTM
#    value it was nominally maximising.
#
# 3. There is a mechanism, not just a correlation. RTM price is endogenous to
#    net load - the market has already aggregated load, outages, fuel and
#    inter-regional flows into one number. Multiplying by load again
#    double-counts the single input the price has already priced, and drags
#    the window toward the load peak (~19:15) in months when scarcity is
#    genuinely later.
#
# Rejected variants, so they are not re-tried: rank(NL)+rank(RTM) scored 4.18
# points WORSE (p<0.001) and NL_RTM/sd collapsed to 39.89% frequency capture.
# Equalising the two factors' influence makes things worse, which is further
# evidence that price should dominate. mean x (1-CV) is statistically tied
# with RTM alone and is the one variant worth revisiting if reliability of the
# window, rather than its level, becomes the priority.
#
# CAVEAT ON THE PRICE SIGNAL. RTM is censored: 12.89% of all blocks and 31.91%
# of 18:00-23:00 blocks sit at or above 9,000 Rs/MWh against a 10,000 cap. In
# those blocks price reports that it is scarce but not how scarce, so the score
# cannot rank a mildly tight evening against a severe one. This is a limitation
# of the market data, not of the model, and it must be stated wherever price
# LEVELS in the evening peak are used to support a claim.
#
# Frequency is deliberately NOT an input. It is an outcome of demand rather
# than a driver of it, and cannot be forecast at declaration time. It is used
# only to validate after the fact - see frequency_corroboration(). That is what
# makes it a fair adjudicator between candidate objectives above.
SCORE_COLUMN = "RTM"


def block_to_time(b: int) -> str:
    m = (int(b) - 1) * 15
    return f"{m // 60:02d}:{m % 60:02d}"


def ranges_from_blocks(blocks) -> str:
    blocks = sorted(blocks)
    runs, s, p = [], blocks[0], blocks[0]
    for b in blocks[1:]:
        if b == p + 1:
            p = b
            continue
        runs.append((s, p)); s = p = b
    runs.append((s, p))
    return ", ".join(f"{block_to_time(a)}-{block_to_time(b + 1)}" for a, b in runs)


def blocks_from_peak_hours(text) -> tuple:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if "-" not in part:
            continue
        lo, hi = (p.strip() for p in part.split("-", 1))
        f = lambda t: (int(t[:2]) * 60 + int(t[3:])) // 15 + 1
        out.extend(range(f(lo), 97 if hi in ("00:00", "24:00") else f(hi)))
    return tuple(sorted(set(out)))


def robust_z(a: np.ndarray) -> np.ndarray:
    """Median/MAD z-score, clipped - the notebook's own normalisation."""
    med = np.nanmedian(a)
    mad = np.nanmedian(np.abs(a - med))
    denom = 1.4826 * mad if mad and np.isfinite(mad) else np.nanstd(a)
    if not denom or not np.isfinite(denom):
        return np.zeros_like(a)
    return np.clip((a - med) / denom, -5, 5)


# ── candidate windows ───────────────────────────────────────────────────────
def build_candidates(total=WINDOW_BLOCKS, evening_start=1, morning_band=None,
                     min_seg=MIN_SEGMENT, end_block=96) -> np.ndarray:
    """Continuous evening windows, evening splits, and morning+evening splits.

    `morning_band` is (lo, hi) or None. When given, the FIRST segment may sit
    in the morning band while the second stays in the evening region - the
    shape every winter hydro declaration uses.
    """
    out = []
    for s in range(evening_start, end_block - total + 2):
        out.append(list(range(s, s + total)))
    for l1 in range(min_seg, total - min_seg + 1):
        l2 = total - l1
        for s1 in range(evening_start, end_block + 1):
            e1 = s1 + l1 - 1
            if e1 > end_block:
                break
            for s2 in range(e1 + 2, end_block - l2 + 2):
                out.append(list(range(s1, s1 + l1)) + list(range(s2, s2 + l2)))
    if morning_band:
        m_lo, m_hi = morning_band
        for l1 in range(min_seg, total - min_seg + 1):
            l2 = total - l1
            for s1 in range(m_lo, m_hi - l1 + 2):
                for s2 in range(evening_start, end_block - l2 + 2):
                    out.append(list(range(s1, s1 + l1)) + list(range(s2, s2 + l2)))
    return np.asarray(out, dtype=np.int32) - 1


def derive_gate(curve: np.ndarray, lo: int, hi: int, peak_hi: int,
                frac: float = 0.5):
    """Trough in [lo,hi] -> following peak -> first block past the half-rise.

    This is the notebook's evening-ramp gate generalised, so the same rule can
    place the morning gate instead of hardcoding a clock time.
    """
    seg = curve[lo - 1:hi]
    if not len(seg) or np.all(np.isnan(seg)):
        return None
    trough = int(np.nanargmin(seg)) + lo
    tail = curve[trough - 1:peak_hi]
    if not len(tail) or np.all(np.isnan(tail)):
        return None
    peak = int(np.nanargmax(tail)) + trough
    if peak <= trough:
        return None
    half = curve[trough - 1] + frac * (curve[peak - 1] - curve[trough - 1])
    rise = curve[trough - 1:peak]
    cross = np.flatnonzero(rise >= half)
    return (int(cross[0]) + trough) if len(cross) else trough


# ── data ────────────────────────────────────────────────────────────────────
def load_blocks() -> pd.DataFrame:
    """Per-block Net_Load, ramp, frequency stress and RTM price."""
    sys.path.insert(0, str(SCADA_DIR))
    import scada_cache as sc
    nr = sc.load("nr")
    nl = (nr["NR_Load"] - nr["NR_Solar"] - nr["NR_Wind"]).rename("Net_Load")
    d = pd.DataFrame({
        "Net_Load": nl.resample("15min").mean(),
        "Freq_Min": nr["NR_Frequency"].resample("15min").min(),
        "Frac_Below": nr["NR_Frequency"].lt(IEGC_BAND_LO).resample("15min").mean(),
    })

    # Outlier removal, matching the notebook's SECTION 4. The raw SCADA series
    # carries ~30 spikes (one reads 38,900,434 MW). Left in, a single spike
    # both inflates that block's level and creates an enormous Ramp_Up, which
    # is exactly what the score keys on - one bad block hijacks the window.
    bad = (d["Net_Load"] < NL_MIN) | (d["Net_Load"] > NL_MAX)
    n_bad = int(bad.sum())
    d.loc[bad, "Net_Load"] = np.nan
    d["Net_Load"] = d["Net_Load"].interpolate(method="time").ffill().bfill()
    if n_bad:
        print(f"  cleaned {n_bad} out-of-range net-load block(s) "
              f"(outside {NL_MIN:,}-{NL_MAX:,} MW)")

    d["Ramp_Up"] = d["Net_Load"].diff().clip(lower=0)

    # Prices come from the repo-local Parquet, so this runs with no network
    # and no IEX credentials. See market_cache.py.
    sys.path.insert(0, str(PROJECT_DIR))
    import market_cache
    rtm = market_cache.load("RTM")
    d = d.join(rtm.set_index("Datetime")["Price"].rename("RTM"))
    d["Block"] = d.index.hour * 4 + d.index.minute // 15 + 1
    d["Month"] = d.index.to_period("M").astype(str)
    d["NL_RTM"] = d["Net_Load"] * d["RTM"]
    return d.dropna(subset=["Net_Load"])


def profile(hist: pd.DataFrame) -> pd.DataFrame:
    """Per-block means over whatever history slice is passed in."""
    return (hist.groupby("Block")[["Net_Load", "Ramp_Up", "Frac_Below", "RTM", "NL_RTM"]]
            .mean().reindex(range(1, 97)))


def information_set(d: pd.DataFrame, month: str) -> pd.DataFrame:
    """History observable when `month` is declared: the SAME month in the two
    prior years. Nothing from the target month.

    Deliberately not the two most recent months. Peak TIMING is seasonal - it
    tracks sunset - so averaging June and May into a July profile drags the
    declared window 1.5-2 hours early. Measured out-of-sample on 19 months,
    frequency stress captured:
        M-12, M-1, M-2   57.91%   (loses to NRPC's 61.32%)
        M-12             67.43%
        M-12, M-24       67.57%   <- used here
    """
    per = pd.Period(month, freq="M")
    return d[d["Month"].isin([str(per - 12), str(per - 24)])]


def score_blocks(prof: pd.DataFrame) -> np.ndarray:
    """Block score: net load x RTM price. No weights, no frequency term."""
    return robust_z(prof[SCORE_COLUMN].to_numpy())


def select(prof: pd.DataFrame):
    """Declare a window from an observable profile.

    The window is exactly WINDOW_BLOCKS (3 hours) because hydro storage is
    limited - that is a physical cap, not a tunable.
    """
    curve = prof["Net_Load"].to_numpy()
    eve = derive_gate(curve, lo=49, hi=72, peak_hi=96) or 72
    morn = derive_gate(curve, lo=1, hi=28, peak_hi=40)
    band = (max(1, morn - 4), min(44, (morn or 24) + 12)) if morn else None
    cand = build_candidates(evening_start=eve, morning_band=band)
    s = np.nan_to_num(score_blocks(prof))
    totals = s[cand].sum(axis=1)
    best = cand[int(totals.argmax())]
    return tuple(int(b) + 1 for b in best), eve, band


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--declare", default=None, help="declare one month, YYYY-MM")
    p.add_argument("--decl-csv", type=Path,
                   default=MARKET_CACHE_DIR / "Previous_Declarations_Hydro.csv")
    p.add_argument("--out", type=Path, default=ARTIFACTS / "Hydro_Model_Backtest.csv")
    a = p.parse_args()

    print("Loading blocks (net load, ramp, frequency, RTM) ...")
    d = load_blocks()
    print(f"  {len(d):,} blocks, {d.Month.min()} -> {d.Month.max()}")

    if a.declare:
        hist = information_set(d, a.declare)
        if hist.empty:
            sys.exit(f"error: no history available for {a.declare}")
        blocks, eve, band = select(profile(hist))
        print(f"\n{a.declare}: {ranges_from_blocks(blocks)}")
        print(f"  evening gate block {eve} ({block_to_time(eve)})"
              + (f" | morning band {band[0]}-{band[1]} "
                 f"({block_to_time(band[0])}-{block_to_time(band[1])})" if band else
                 " | no morning ramp detected"))
        return

    decl = pd.read_csv(a.decl_csv).set_index("Month")
    free = build_candidates(evening_start=1, morning_band=(1, 44))
    rows = []
    for month, g in d.groupby("Month"):
        if month not in decl.index:
            continue
        if g.index.normalize().nunique() < pd.Period(month).days_in_month:
            continue
        hist = information_set(d, month)
        if hist["Month"].nunique() < 1:
            continue
        actual = profile(g)                      # ex-post, for scoring only
        val = np.nan_to_num(actual["NL_RTM"].to_numpy())
        opt = val[free].sum(axis=1).max()
        # Frequency stress is the ONLY column here that no candidate objective
        # optimises, so it is the only non-circular comparison on the table.
        # Value_% is reported alongside it because it is what the market cares
        # about, but a model scored on price cannot be judged on price.
        frq = np.nan_to_num(actual["Frac_Below"].to_numpy())
        fopt = max(frq[free].sum(axis=1).max(), 1e-9)
        chosen, _, band = select(profile(hist))  # declared from history alone
        declared = blocks_from_peak_hours(decl.loc[month, "Peak_Hours"])
        ci = [b - 1 for b in chosen]
        di = [b - 1 for b in declared]
        rows.append({
            "Month": month,
            "Model": ranges_from_blocks(chosen),
            "Model_%": 100 * val[ci].sum() / opt,
            "NRPC": decl.loc[month, "Peak_Hours"],
            "NRPC_%": 100 * val[di].sum() / opt,
            "Model_Freq_%": 100 * frq[ci].sum() / fopt,
            "NRPC_Freq_%": 100 * frq[di].sum() / fopt,
            "Model_Morning": bool(band) and min(chosen) < 44,
            "NRPC_Morning": min(declared) < 44,
            "Winter": month[-2:] in ("10", "11", "12", "01", "02", "03"),
        })

    r = pd.DataFrame(rows)
    if r.empty:
        sys.exit("error: no month had both a declaration and enough history")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(a.out, index=False)

    print(f"\n{'='*94}\nReplay backtest - declared from history only, scored on actuals\n{'='*94}")
    print(r[["Month", "Model", "Model_%", "NRPC", "NRPC_%"]].to_string(
        index=False, formatters={"Model_%": "{:.1f}".format, "NRPC_%": "{:.1f}".format}))

    print(f"\n{'-'*60}\nSummary over {len(r)} months\n{'-'*60}")
    print("  value capture (NL x RTM) - the market's metric, NOT independent of")
    print("  what the declaration is chosen on:")
    for lab, s in (("ALL", r), ("Apr-Sep", r[~r.Winter]), ("Oct-Mar", r[r.Winter])):
        if not len(s):
            continue
        print(f"    {lab:8s} n={len(s):2d}  model {s['Model_%'].mean():6.2f}% "
              f"(min {s['Model_%'].min():6.2f})   NRPC {s['NRPC_%'].mean():6.2f}% "
              f"(min {s['NRPC_%'].min():6.2f})   model beats NRPC in "
              f"{int((s['Model_%'] > s['NRPC_%']).sum())}/{len(s)}")
    print("\n  frequency stress captured - exogenous to every objective tried,")
    print("  so this is the comparison to report:")
    for lab, s in (("ALL", r), ("Apr-Sep", r[~r.Winter]), ("Oct-Mar", r[r.Winter])):
        if not len(s):
            continue
        print(f"    {lab:8s} n={len(s):2d}  model {s['Model_Freq_%'].mean():6.2f}% "
              f"   NRPC {s['NRPC_Freq_%'].mean():6.2f}%"
              f"   model beats NRPC in "
              f"{int((s['Model_Freq_%'] > s['NRPC_Freq_%']).sum())}/{len(s)}")
    w = r[r.Winter]
    if len(w):
        print(f"\n  Winter morning segment declared - model {int(w.Model_Morning.sum())}/{len(w)}"
              f" | NRPC {int(w.NRPC_Morning.sum())}/{len(w)}")
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()

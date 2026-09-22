#!/usr/bin/env python3
"""Turn Previous_Declarations.xlsx into the CSV the notebook scores against.

Previous_Declarations.xlsx is the authoritative record of RLDC peak-hour
declarations. This script converts it to Previous_Declarations.csv in the shape
SECTION 23b expects (Month, Peak_Hours) and reports any month it cannot fill,
so those can be added to the xlsx by hand and the script re-run.

    python build_previous_declarations.py
    python build_previous_declarations.py --peak hydro
    python build_previous_declarations.py --from 2022-07 --to 2026-08

What it has to reconcile
------------------------
* Two time spellings. Newer rows read "18:15 to 20:15 and 21:30 to 23:30";
  rows up to 2024-10 read "1815 to 2115 and 2200 to 2300". Both parse.
* One or two segments per declaration.
* Rows are DATE RANGES, not months, and mid-month revisions overlap earlier
  rows (e.g. 2026-04-01..30 is superseded from 2026-04-03). Days are resolved
  by Date of Declaration, latest wins, then each month takes the window that
  covers the most of its days.
* Thermal and Hydro are separate declarations. Thermal is the default here:
  the pipeline scores Net_Load x RTM price, which is the thermal peak. Hydro
  is carried alongside for reference.

Nothing is invented. A month with no covering row is reported as MISSING and
left out of the CSV rather than guessed at.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from powertools_common import cache_dir

PROJECT_DIR = Path(__file__).resolve().parent
MARKET_CACHE_DIR = cache_dir("peak_hours")
XLSX = MARKET_CACHE_DIR / "Previous_Declarations.xlsx"
CSV = MARKET_CACHE_DIR / "Previous_Declarations.csv"

# "18:15", "1815", "630" -> (hh, mm). Anchored so it cannot straddle two times.
_TIME = re.compile(r"\b(\d{1,2}):?(\d{2})\b")
VALID_MIN = {0, 15, 30, 45}


def parse_ranges(text) -> list[tuple[str, str]]:
    """'18:15 to 20:15 and 21:30 to 23:30' -> [('18:15','20:15'), ...]."""
    out: list[tuple[str, str]] = []
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return out
    for part in re.split(r"\s+and\s+|\s*&\s*", str(text), flags=re.I):
        toks = _TIME.findall(part)
        if len(toks) != 2:
            continue
        pair = []
        for h, m in toks:
            h, m = int(h), int(m)
            if not (0 <= h <= 24 and m in VALID_MIN):
                pair = []
                break
            pair.append(f"{h:02d}:{m:02d}")
        if len(pair) == 2 and pair[0] != pair[1]:
            out.append((pair[0], pair[1]))
    return out


def ranges_to_text(ranges) -> str:
    """Render in the notebook's own format: 'HH:MM-HH:MM, HH:MM-HH:MM'."""
    return ", ".join(f"{a}-{b}" for a, b in ranges)


def n_blocks(ranges) -> int:
    tot = 0
    for a, b in ranges:
        ah, am = (int(x) for x in a.split(":"))
        bh, bm = (int(x) for x in b.split(":"))
        tot += ((bh * 60 + bm) - (ah * 60 + am)) // 15
    return tot


def resolve_days(df: pd.DataFrame, column: str) -> dict:
    """day -> declared window text, latest Date of Declaration winning."""
    df = df.sort_values("Date of Declaration", kind="stable")
    by_day: dict[pd.Timestamp, str] = {}
    unparsed = []
    for _, r in df.iterrows():
        rng = parse_ranges(r[column])
        if not rng:
            unparsed.append((r["S.No"], r[column]))
            continue
        text = ranges_to_text(rng)
        for day in pd.date_range(r["Start_Date"], r["End_Date"], freq="D"):
            by_day[day] = text          # later declaration overwrites earlier
    return by_day, unparsed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xlsx", type=Path, default=XLSX)
    p.add_argument("--out", type=Path, default=CSV)
    p.add_argument("--peak", choices=["thermal", "hydro"], default="thermal",
                   help="which declaration to write as Peak_Hours")
    p.add_argument("--from", dest="frm", default=None, help="first month YYYY-MM")
    p.add_argument("--to", dest="to", default=None, help="last month YYYY-MM")
    a = p.parse_args()

    if not a.xlsx.exists():
        sys.exit(f"error: {a.xlsx} not found")

    df = pd.read_excel(a.xlsx)
    need = {"Start_Date", "End_Date", "Thermal Peak", "Hydro Peak",
            "Date of Declaration"}
    missing_cols = need - set(df.columns)
    if missing_cols:
        sys.exit(f"error: {a.xlsx.name} is missing columns {sorted(missing_cols)}")
    for c in ("Start_Date", "End_Date", "Date of Declaration"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    bad_dates = df[df[["Start_Date", "End_Date"]].isna().any(axis=1)]
    if len(bad_dates):
        print(f"! {len(bad_dates)} row(s) with unreadable dates, skipped: "
              f"{bad_dates['S.No'].tolist()}")
        df = df.drop(bad_dates.index)

    col = "Thermal Peak" if a.peak == "thermal" else "Hydro Peak"
    other = "Hydro Peak" if a.peak == "thermal" else "Thermal Peak"
    days, unparsed = resolve_days(df, col)
    days_other, _ = resolve_days(df, other)

    if unparsed:
        print(f"! {len(unparsed)} row(s) whose {col!r} could not be parsed:")
        for sno, txt in unparsed[:10]:
            print(f"    S.No {sno}: {txt!r}")

    if not days:
        sys.exit("error: nothing parsed - check the time format in the xlsx")

    lo = pd.Period(min(days), freq="M")
    hi = pd.Period(max(days), freq="M")
    if a.frm:
        lo = max(lo, pd.Period(a.frm, freq="M"))
    if a.to:
        hi = min(hi, pd.Period(a.to, freq="M"))

    rows, missing, partial = [], [], []
    for per in pd.period_range(lo, hi, freq="M"):
        month_days = pd.date_range(per.start_time, per.end_time.normalize(), freq="D")
        have = {d: days[d] for d in month_days if d in days}
        if not have:
            missing.append(str(per))
            continue
        counts = pd.Series(list(have.values())).value_counts()
        text = counts.index[0]
        rng = [tuple(s.split("-")) for s in text.split(", ")]
        cover = len(have) / len(month_days)
        if cover < 1.0:
            partial.append((str(per), len(have), len(month_days)))
        oth = pd.Series([days_other[d] for d in month_days if d in days_other])
        rows.append({
            "Month": str(per),
            "Peak_Hours": text,
            "Blocks": n_blocks(rng),
            "Segments": len(rng),
            "Has_Morning": any(int(s.split(":")[0]) < 12 for s, _ in rng),
            "Days_On_This_Window": int(counts.iloc[0]),
            "Days_Declared": len(have),
            "Days_In_Month": len(month_days),
            "Revisions_In_Month": int(len(counts)),
            f"{other.split()[0]}_Peak_Hours": oth.value_counts().index[0] if len(oth) else "",
        })

    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)

    print(f"\nWrote {a.out.name}: {len(out)} months "
          f"({out.Month.min()} -> {out.Month.max()}) from {a.peak} declarations")
    print(f"  block counts   : {out.Blocks.value_counts().to_dict()}")
    print(f"  with a morning segment: {int(out.Has_Morning.sum())} of {len(out)}")
    print(f"  months revised mid-month: {int((out.Revisions_In_Month > 1).sum())}")

    if partial:
        print(f"\n  {len(partial)} month(s) NOT fully covered by any declaration:")
        for m, h, t in partial:
            print(f"    {m}: {h}/{t} days declared")

    if missing:
        print(f"\n!! {len(missing)} MISSING month(s) - no row in the xlsx covers them.")
        print("   Add them to Previous_Declarations.xlsx and re-run this script:")
        for m in missing:
            print(f"     {m}")
    else:
        print(f"\n  No missing months between {lo} and {hi}.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Turn Previous_Declarations.xlsx into the CSV the engine scores against.

Ported from ML_Peak_Hour_Declaration/build_previous_declarations.py (logic
unchanged) as part of the Peak_Hour_Engine redesign, Segment 1. The xlsx
lives in the shared OneDrive PowerTools area (peak_hours.paths.MARKET_CACHE_DIR)
as the single source of truth for both machines; this module reads it in place.

Previous_Declarations.xlsx is the authoritative record of RLDC peak-hour
declarations.

    python -m peak_hours.io.declarations
    python -m peak_hours.io.declarations --peak hydro
    python -m peak_hours.io.declarations --from 2022-07 --to 2026-08

What it has to reconcile
------------------------
* Two time spellings. Newer rows read "18:15 to 20:15 and 21:30 to 23:30";
  rows up to 2024-10 read "1815 to 2115 and 2200 to 2300". Both parse.
* One or two segments per declaration.
* Rows are DATE RANGES, not months, and mid-month revisions overlap earlier
  rows. Days are resolved by Date of Declaration, latest wins, then each
  month takes the window that covers the most of its days.
* Thermal and Hydro are separate declarations. Thermal is the default here.

Nothing is invented. A month with no covering row is reported as MISSING and
left out of the CSV rather than guessed at.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from peak_hours.paths import MARKET_CACHE_DIR

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
    """Render as 'HH:MM-HH:MM, HH:MM-HH:MM'."""
    return ", ".join(f"{a}-{b}" for a, b in ranges)


def n_blocks(ranges) -> int:
    tot = 0
    for a, b in ranges:
        ah, am = (int(x) for x in a.split(":"))
        bh, bm = (int(x) for x in b.split(":"))
        tot += ((bh * 60 + bm) - (ah * 60 + am)) // 15
    return tot


def resolve_days(df: pd.DataFrame, column: str) -> tuple[dict, list]:
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


def build(xlsx: Path = XLSX, peak: str = "thermal",
          frm: str | None = None, to: str | None = None) -> pd.DataFrame:
    """Core conversion, importable for tests without going through argparse/CLI."""
    if not xlsx.exists():
        raise FileNotFoundError(f"{xlsx} not found")

    df = pd.read_excel(xlsx)
    need = {"Start_Date", "End_Date", "Thermal Peak", "Hydro Peak",
            "Date of Declaration"}
    missing_cols = need - set(df.columns)
    if missing_cols:
        raise ValueError(f"{xlsx.name} is missing columns {sorted(missing_cols)}")
    for c in ("Start_Date", "End_Date", "Date of Declaration"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    bad_dates = df[df[["Start_Date", "End_Date"]].isna().any(axis=1)]
    if len(bad_dates):
        df = df.drop(bad_dates.index)

    col = "Thermal Peak" if peak == "thermal" else "Hydro Peak"
    other = "Hydro Peak" if peak == "thermal" else "Thermal Peak"
    days, unparsed = resolve_days(df, col)
    days_other, _ = resolve_days(df, other)

    if not days:
        raise ValueError("nothing parsed - check the time format in the xlsx")

    lo = pd.Period(min(days), freq="M")
    hi = pd.Period(max(days), freq="M")
    if frm:
        lo = max(lo, pd.Period(frm, freq="M"))
    if to:
        hi = min(hi, pd.Period(to, freq="M"))

    rows = []
    for per in pd.period_range(lo, hi, freq="M"):
        month_days = pd.date_range(per.start_time, per.end_time.normalize(), freq="D")
        have = {d: days[d] for d in month_days if d in days}
        if not have:
            continue
        counts = pd.Series(list(have.values())).value_counts()
        text = counts.index[0]
        rng = [tuple(s.split("-")) for s in text.split(", ")]
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

    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xlsx", type=Path, default=XLSX)
    p.add_argument("--out", type=Path, default=CSV)
    p.add_argument("--peak", choices=["thermal", "hydro"], default="thermal")
    p.add_argument("--from", dest="frm", default=None, help="first month YYYY-MM")
    p.add_argument("--to", dest="to", default=None, help="last month YYYY-MM")
    a = p.parse_args()

    try:
        out = build(a.xlsx, a.peak, a.frm, a.to)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"error: {exc}")

    out.to_csv(a.out, index=False)
    print(f"\nWrote {a.out.name}: {len(out)} months "
          f"({out.Month.min()} -> {out.Month.max()}) from {a.peak} declarations")
    print(f"  block counts   : {out.Blocks.value_counts().to_dict()}")
    print(f"  with a morning segment: {int(out.Has_Morning.sum())} of {len(out)}")


if __name__ == "__main__":
    main()

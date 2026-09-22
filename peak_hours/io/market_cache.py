#!/usr/bin/env python3
"""Offline accessor for IEX block prices and volumes.

Ported from ML_Peak_Hour_Declaration/market_cache.py (logic unchanged) as
part of the Peak_Hour_Engine redesign — see Segment 1 of the migration plan.
The CSVs it reads (RTM_Prices_2022_2026.csv, DAM_Prices_2022_2026.csv) live in
the shared OneDrive PowerTools area via peak_hours.paths.MARKET_CACHE_DIR, so
both machines see the same files without either duplicating them.

    from peak_hours.io import market_cache
    rtm = market_cache.load("RTM")        # Datetime, Block, Price
    full = market_cache.load_full("RTM")  # every column, prices + volumes

Refresh on a machine that can reach IEX (or that has the raw payload cache --
neither the volumes nor the prices need a network round-trip for any period
already downloaded):

    python -m peak_hours.io.market_cache --refresh
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from peak_hours.paths import MARKET_CACHE_DIR

AREAS = ["N1", "N2", "N3"]          # NR price areas, averaged - as the notebook does

# [SCOPE] Kept apart deliberately: these two groups answer different questions
# and merging them into one "volumes" bucket is how the geography gets lost.
ALL_INDIA_BID_COLS = ["Purchase_Bid_MW", "Sell_Bid_MW", "MCV_MW"]
NR_VOLUME_COLS = [
    "NR_Cleared_Buy_MW",
    "NR_Cleared_Sell_MW",
    "NR_Net_Buy_MW",
    "NR_Price_Spread",
] + [f"{a}_Cleared_{side}_MW" for a in AREAS for side in ("Buy", "Sell")]

# Candidate roots for the `iex` library, for the refresh path only. Usually
# already importable via PYTHONPATH; these are the fallbacks.
_IEX_ROOTS = [
    Path.home() / "Development" / "power-libraries",
    Path(r"D:\power-libraries"),
]


def csv_path(market: str) -> Path:
    return MARKET_CACHE_DIR / f"{market.upper()}_Prices_2022_2026.csv"


def load_full(market: str = "RTM") -> pd.DataFrame:
    """Every column the export carries, with a Datetime added."""
    p = csv_path(market)
    if not p.exists():
        raise FileNotFoundError(
            f"{p.name} not found in {MARKET_CACHE_DIR}. On a machine with IEX access run:\n"
            f"    python -m peak_hours.io.market_cache --refresh"
        )
    df = pd.read_csv(p)
    df["Block"] = pd.to_numeric(df["Block"], errors="coerce").astype("int16")
    df["Datetime"] = (pd.to_datetime(df["Date"])
                      + pd.to_timedelta((df["Block"] - 1) * 15, unit="min"))
    return df.sort_values("Datetime").reset_index(drop=True)


def load(market: str = "RTM", start=None, end=None) -> pd.DataFrame:
    """Long-format block prices: Datetime, Block, Price (NR areas averaged)."""
    df = load_full(market)
    col = f"{market.upper()}_Price"
    if col in df.columns:
        price = pd.to_numeric(df[col], errors="coerce")
    else:                                     # fall back to averaging the areas
        have = [a for a in AREAS if a in df.columns]
        price = df[have].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    out = pd.DataFrame({"Datetime": df["Datetime"], "Block": df["Block"],
                        "Price": price})
    if start is not None:
        out = out[out["Datetime"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["Datetime"] <= pd.Timestamp(end)]
    return out.reset_index(drop=True)


def load_nr_volumes(market: str = "RTM") -> pd.DataFrame:
    """Just the NR cleared-volume columns, with a Datetime.

    Convenience for code that wants the region's own position without having
    to remember which of load_full()'s columns are national."""
    df = load_full(market)
    have = [c for c in NR_VOLUME_COLS if c in df.columns]
    if not have:
        raise KeyError(
            f"{csv_path(market).name} predates the NR-volume columns. Run:\n"
            f"    python -m peak_hours.io.market_cache --refresh"
        )
    return df[["Datetime", "Date", "Block"] + have]


def _import_iex():
    for root in _IEX_ROOTS:
        if root.exists() and str(root) not in sys.path:
            sys.path.insert(0, str(root))
    try:
        import iex
    except ImportError as exc:
        raise RuntimeError(
            "iex library not importable - refresh only works on a machine that has it"
        ) from exc
    for name in ("get_trade_data", "get_market_volumes", "get_region_volumes"):
        if not hasattr(iex, name):
            raise RuntimeError(
                f"the installed iex library has no {name}(); it predates the "
                f"area-volume extractor this module needs"
            )
    return iex


def refresh(market: str, start="2022-07-01", end=None) -> Path:
    """Re-pull from IEX and rewrite the CSV with prices AND both volume sets."""
    iex = _import_iex()

    market = market.upper()
    start = pd.Timestamp(start)
    end = pd.Timestamp(end or pd.Timestamp.today())
    d1, d2 = start.strftime("%d-%m-%Y"), end.strftime("%d-%m-%Y")

    # ---- NR area prices, pivoted 96-wide -> long ----------------------------
    raw = iex.get_trade_data(d1, d2, market)
    cols = [str(i) for i in range(1, 97)]
    nr = raw[raw["Area"].isin(AREAS)]
    long = nr.melt(id_vars=["Date", "Area"], value_vars=cols,
                   var_name="Block", value_name="Price")
    long["Block"] = pd.to_numeric(long["Block"], errors="coerce").astype("int16")
    long["Price"] = pd.to_numeric(long["Price"], errors="coerce")
    wide = (long.pivot_table(index=["Date", "Block"], columns="Area",
                             values="Price", aggfunc="first")
            .reset_index().rename_axis(None, axis=1))
    for a in AREAS:
        if a not in wide.columns:
            wide[a] = pd.NA
    wide["Market"] = market
    wide[f"{market}_Price"] = wide[AREAS].mean(axis=1)
    wide["Date"] = pd.to_datetime(wide["Date"])

    # ---- All-India bid stacks -------------------------------------------------
    bids = iex.get_market_volumes(d1, d2, market).rename(columns={
        "Buy_Volume": "Purchase_Bid_MW",
        "Sell_Volume": "Sell_Bid_MW",
        "Cleared_Volume": "MCV_MW",
    })
    bids["Date"] = pd.to_datetime(bids["Date"])
    bids["Block"] = pd.to_numeric(bids["Block"], errors="coerce").astype("int16")
    wide = wide.merge(bids[["Date", "Block"] + ALL_INDIA_BID_COLS],
                      on=["Date", "Block"], how="left")

    # ---- NR cleared volumes -----------------------------------------------------
    region = iex.get_region_volumes(d1, d2, market, areas=AREAS, prefix="NR")
    region["Date"] = pd.to_datetime(region["Date"])
    region["Block"] = pd.to_numeric(region["Block"], errors="coerce").astype("int16")
    keep = ["Date", "Block"] + [c for c in NR_VOLUME_COLS if c in region.columns]
    wide = wide.merge(region[keep], on=["Date", "Block"], how="left")

    ordered = (["Date", "Block", "Market"] + AREAS + [f"{market}_Price"]
               + ALL_INDIA_BID_COLS
               + [c for c in NR_VOLUME_COLS if c in wide.columns])
    wide = wide[ordered].sort_values(["Date", "Block"]).reset_index(drop=True)
    wide["Date"] = wide["Date"].dt.strftime("%Y-%m-%d")

    p = csv_path(market)
    wide.to_csv(p, index=False)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--markets", default="RTM,DAM")
    ap.add_argument("--start", default="2022-07-01")
    ap.add_argument("--end", default=None)
    a = ap.parse_args()

    for m in [x.strip().upper() for x in a.markets.split(",") if x.strip()]:
        if a.refresh:
            print(f"{m}: refreshing from IEX (overwrites the CSV)")
            p = refresh(m, a.start, a.end)
            df = pd.read_csv(p)
            print(f"  wrote {p.name}: {df.shape[0]:,} rows x {df.shape[1]} cols")
        else:
            try:
                df = load(m)
                full = load_full(m)
                nr_have = [c for c in NR_VOLUME_COLS if c in full.columns]
                print(f"{m}: {len(df):,} block-prices  "
                      f"{df.Datetime.min().date()} -> {df.Datetime.max().date()}  "
                      f"({csv_path(m).name})")
                print(f"    NR volume columns: "
                      f"{len(nr_have)}{' (none - run --refresh)' if not nr_have else ''}")
            except FileNotFoundError as e:
                print(e)


if __name__ == "__main__":
    main()

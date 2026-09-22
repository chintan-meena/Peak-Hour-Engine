#!/usr/bin/env python3
r"""Regenerate City_Weather_Feature_Importance.csv — standalone.

WHAT THIS FILE IS
-----------------
`City_Weather_Feature_Importance.csv` is the weighting table that
Peak_Hours_Complete_Pipeline_v1.ipynb reads at startup to blend 40 weather
stations into one composite Temp/Humidity signal:

    CITY_WEIGHTS[city] = Total_Importance / sum(Total_Importance)

v1 only *consumes* the file; it was produced by the v5 notebooks. This script
replaces that derivation so the v5 notebooks can be deleted.

WHAT IT DOES DIFFERENTLY FROM THE v5 NOTEBOOK CELL
--------------------------------------------------
The v5 cell got these importances as a byproduct of the *production* net-load
model, which made it far more expensive than the task requires:

  1. It fit the model TWICE — once on a train split only to print MAE/RMSE,
     then again on full data. Only the second fit's importances reach the CSV.
     This script fits once.
  2. Production settings (700 trees, depth 10, all ~137k rows, 447 features).
     Importance *ranking* stabilizes long before that, so the default here is
     a cheaper probe: fewer/shallower trees on a row subsample. Use --full to
     reproduce production settings exactly.
  3. It built state-level composites (Weighted_Temp, LoadActual state shares).
     Those columns are excluded from the model's feature list and are not used
     for the weather lags either, so they cannot affect the result — dropped.
     That removes the LoadActual/ and iex_client dependencies entirely.

Everything that *does* affect the numbers is ported verbatim: the raw-demand
parsing, the net-load cleaning, the 447-column feature engineering, the
per-city importance aggregation, and LightGBM's default split-based
importance_type.

WHERE NET LOAD COMES FROM
-------------------------
Preferred source is the scada-cache/ subproject, which parses the NRLDC Stack
exports off \\192.168.50.247\scadashare once into per-year Parquet and defines
Net_Load = NR_Load - NR_Solar - NR_Wind — the same definition as the notebooks:

    import scada_cache as sc
    sc.load_net_load(freq="15min")

By default the script refreshes that cache from the share first
(--update-cache yes). The ingest is incremental — a manifest records each
file's size and mtime, so only new or changed days are parsed. The first run
is therefore slow (it parses the whole share, ~1500 files) and every run after
that is quick. When the share is unreachable the update is a no-op and
whatever is already cached still loads, so this is safe off-LAN.

Pass --update-cache no to skip it and use the cache exactly as it stands.

If the cache still cannot cover the requested window, the script falls back to
scanning a folder of raw DD-MM-YYYY.csv demand files with the notebook's own
parser. Control the source with --net-load-source.

Note the cache committed to git is only a seed (the 'nr' source holds a single
day), so the first run on a new clone does the full ingest.

CACHING
-------
  * Weather -> monthly_peak_pipeline_outputs/Weather_<City>_Hourly.csv
    This is the SAME path and format v1 uses, so the two share one cache —
    whichever runs first pays for the 40 downloads.
  * Net load -> scada-cache's own Parquet when that is the source. Only the
    raw-folder path writes Net_Load_15min.csv, so a stale CSV can never
    shadow fresher SCADA data.

USAGE
-----
    python build_city_weather_importance.py                  # fast probe
    python build_city_weather_importance.py --full           # production settings
    python build_city_weather_importance.py --net-load-source raw \
        --demand-folder ../NR_DEMAND_TEMP

If a previous City_Weather_Feature_Importance.csv exists, the script prints the
Spearman rank correlation between the old and new weights — use that to confirm
the fast defaults agree with whatever produced your current file.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_DIR = Path(__file__).resolve().parent
SCADA_DIR = PROJECT_DIR / "scada-cache"

# ── Defaults mirror Peak_Hours_Complete_Pipeline_v1.ipynb ────────────────────
DEFAULT_START = "2022-07-03"
DEFAULT_END = "2026-08-22"
NL_MIN, NL_MAX = 10_000, 100_000
RAW_EXPECTED_COLS = {"HRS", "NR Load", "NR Solar", "NR Wind"}

# ── The 40 stations, grouped by state (State column comes from this) ─────────
# (city, lat, lon) — intra-state weights and IMD ids from v5 are not used here.
WEATHER_STATIONS: dict[str, list[tuple[str, float, float]]] = {
    "Punjab": [("Ludhiana", 30.90, 75.86), ("Amritsar", 31.63, 74.87),
               ("Patiala", 30.33, 76.40), ("Jalandhar", 31.33, 75.58),
               ("Bathinda", 30.21, 74.94)],
    "Haryana": [("Hisar", 29.15, 75.72), ("Gurugram", 28.46, 77.03),
                ("Faridabad", 28.41, 77.31), ("Ambala", 30.38, 76.78),
                ("Rohtak", 28.90, 76.58), ("Panipat", 29.39, 76.97)],
    "Rajasthan": [("Jaipur", 26.82, 75.80), ("Jodhpur", 26.24, 73.02),
                  ("Kota", 25.17, 75.85), ("Udaipur", 24.58, 73.71),
                  ("Bikaner", 28.01, 73.31), ("Ajmer", 26.45, 74.64),
                  ("Sriganganagar", 29.92, 73.88)],
    "Delhi": [("Palam", 28.57, 77.10), ("Safdarjung", 28.59, 77.21),
              ("LodiBhawan", 28.59, 77.22)],
    "UP": [("Lucknow", 26.85, 80.95), ("Agra", 27.18, 78.01),
           ("Kanpur", 26.47, 80.33), ("Varanasi", 25.32, 82.97),
           ("Prayagraj", 25.45, 81.84), ("Noida", 28.54, 77.39),
           ("Gorakhpur", 26.75, 83.37), ("Meerut", 28.98, 77.72),
           ("Bareilly", 28.36, 79.41)],
    "Uttarakhand": [("Dehradun", 30.32, 78.03), ("Haridwar", 29.97, 78.17),
                    ("Roorkee", 29.87, 77.89)],
    "HP": [("Shimla", 31.10, 77.17), ("Dharamsala", 32.22, 76.32),
           ("Mandi", 31.71, 76.93)],
    "J&K Ladakh": [("Jammu", 32.74, 74.87), ("Srinagar", 34.08, 74.80),
                   ("Leh", 34.15, 77.58)],
    "Chd": [("Chandigarh", 30.74, 76.79)],
}

ALL_CITIES = [c for stns in WEATHER_STATIONS.values() for (c, _la, _lo) in stns]
CITY_STATE = {c: s for s, stns in WEATHER_STATIONS.items() for (c, _la, _lo) in stns}
CITY_COORDS = {c: (la, lo) for stns in WEATHER_STATIONS.values() for (c, la, lo) in stns}


# ════════════════════════════════════════════════════════════════════════════
# Net load
# ════════════════════════════════════════════════════════════════════════════
def _files_in_range(folder: Path, start_ts, end_ts):
    """Daily demand files are named DD-MM-YYYY.csv; keep those inside the range."""
    dated, skipped_bad = [], 0
    for f in sorted(folder.glob("*.csv")):
        try:
            fd = pd.to_datetime(f.stem, format="%d-%m-%Y").normalize()
        except Exception:
            skipped_bad += 1
            continue
        if start_ts.normalize() <= fd <= end_ts.normalize():
            dated.append((f, fd))
    if skipped_bad:
        print(f"  skipped {skipped_bad} file(s) with unparseable names")
    return dated


def build_net_load_from_raw(folder: Path, start_ts, end_ts) -> pd.DataFrame:
    """Port of the v1/v5 raw parser: header on row 13, data from row 15."""
    if not folder.exists():
        raise FileNotFoundError(
            f"Demand folder not found: {folder}\n"
            f"Pass the right one with --demand-folder."
        )
    dated = _files_in_range(folder, start_ts, end_ts)
    if not dated:
        raise ValueError(f"No demand CSVs in {folder} between "
                         f"{start_ts.date()} and {end_ts.date()}")
    print(f"  parsing {len(dated)} daily demand files …")

    parts, errors = [], []
    for file, file_date in dated:
        try:
            raw = pd.read_csv(file, header=None)
            cols = raw.iloc[13].astype(str).str.strip().tolist()
            d = raw.iloc[15:].copy()
            d.columns = (pd.Series(cols).astype(str)
                         .str.strip().str.replace("\n", " ", regex=False).tolist())
            missing = RAW_EXPECTED_COLS - set(d.columns)
            if missing:
                errors.append(f"{file.name} (missing {missing})")
                continue
            d["HRS"] = d["HRS"].astype(str).str.strip()
            d["Datetime"] = pd.to_datetime(
                file_date.strftime("%Y-%m-%d") + " " + d["HRS"],
                format="%Y-%m-%d %H:%M:%S", errors="coerce",
            )
            d = d.dropna(subset=["Datetime"])
            for c in ("NR Load", "NR Solar", "NR Wind"):
                d[c] = pd.to_numeric(d[c], errors="coerce")
            d["Net_Load"] = d["NR Load"] - d["NR Solar"] - d["NR Wind"]
            parts.append(d[["Datetime", "Net_Load"]].dropna())
        except Exception as e:
            errors.append(f"{file.name} ({e})")

    if errors:
        print(f"  {len(errors)} file(s) had problems; first few: {errors[:3]}")
    if not parts:
        raise ValueError("No valid rows extracted from demand files.")
    return pd.concat(parts, ignore_index=True).sort_values("Datetime").dropna()


def clean_net_load(df15: pd.DataFrame) -> pd.DataFrame:
    """Outlier replacement, ported from v5 SECTION 4.

    The original looped over every row checking `isna()`; only NaN rows do any
    work, so iterating just the NaN positions in ascending order is equivalent
    (later fills still see earlier fills) and much faster.
    """
    df = df15.copy()
    invalid = (df["Net_Load"] < NL_MIN) | (df["Net_Load"] > NL_MAX)
    print(f"  outlier rows replaced: {int(invalid.sum())}")
    df.loc[invalid, "Net_Load"] = np.nan

    season, window = 96 * 7, 96
    col = df.columns.get_loc("Net_Load")
    values = df["Net_Load"]
    nan_positions = np.flatnonzero(values.isna().to_numpy())
    for i in nan_positions:
        if i < season + window:
            continue
        curr_avg = df["Net_Load"].iloc[i - window:i].mean()
        week_avg = df["Net_Load"].iloc[i - season - window:i - season].mean()
        week_same = df["Net_Load"].iloc[i - season]
        if pd.notna(curr_avg) and pd.notna(week_avg) and pd.notna(week_same) and week_avg != 0:
            df.iloc[i, col] = week_same * (curr_avg / week_avg)

    df["Net_Load"] = df["Net_Load"].interpolate(method="time").ffill().bfill()
    return df


def _finalise_net_load(d15: pd.DataFrame) -> pd.DataFrame:
    """Gap filling + outlier cleaning, identical for every source."""
    # Same block yesterday / tomorrow averaged, then time interpolation.
    d15["Net_Load"] = d15["Net_Load"].fillna(
        (d15["Net_Load"].shift(96) + d15["Net_Load"].shift(-96)) / 2
    )
    d15["Net_Load"] = d15["Net_Load"].interpolate(method="time").ffill().bfill()
    return clean_net_load(d15)


def _coverage_ok(idx: pd.DatetimeIndex, start_ts, end_ts) -> bool:
    """True if the index spans the requested window (allowing a 1-day margin)."""
    if len(idx) == 0:
        return False
    return idx.min() <= start_ts and idx.max() >= end_ts - pd.Timedelta(days=1)


def _update_scada_cache(sc) -> None:
    """Incrementally ingest new days from the SCADA share.

    Only the 'nr' source is refreshed — it is the only one this script reads.
    Use scada-cache/update_cache.py for a full refresh of every source.
    """
    n_known = 0
    try:
        manifest = SCADA_DIR / "cache" / "nr" / "_manifest.json"
        if manifest.exists():
            import json
            n_known = len(json.loads(manifest.read_text()).get("files", {}))
    except Exception:
        pass
    if n_known < 50:
        print(f"  cache holds {n_known} day(s) — the first ingest parses the "
              f"whole share and takes a while; later runs only add new days.")

    print("  updating SCADA cache (incremental; no-op if the share is offline) …")
    try:
        s = sc.update_source("nr")
    except Exception as e:                      # never fail the run over this
        print(f"  update failed ({e}) — using the cache as it stands")
        return
    if not s.get("reachable"):
        print("  share unreachable — using the cache as it stands")
    else:
        msg = f"  ingested {s.get('ingested', 0)} new/changed file(s), skipped {s.get('skipped', 0)}"
        if s.get("failed"):
            msg += f", failed {s['failed']}"
        print(msg)


def load_net_load_scada(start_ts, end_ts, update: bool = True):
    """Net_Load from the scada-cache subproject, or None if unusable.

    Returns None (rather than raising) so `--net-load-source auto` can fall
    back to the raw-folder scan.
    """
    if not SCADA_DIR.exists():
        print(f"  scada-cache not found at {SCADA_DIR}")
        return None
    if str(SCADA_DIR) not in sys.path:
        sys.path.insert(0, str(SCADA_DIR))
    try:
        import scada_cache as sc
    except ImportError as e:
        print(f"  scada-cache not importable ({e}) — needs: "
              f"pip install -r {SCADA_DIR / 'requirements.txt'}")
        return None

    if update:
        _update_scada_cache(sc)

    try:
        net = sc.load_net_load(start=start_ts, end=end_ts, freq="15min")
    except (FileNotFoundError, KeyError) as e:
        print(f"  scada-cache unusable: {e}")
        return None

    if _coverage_ok(net.index, start_ts, end_ts):
        print(f"  net load: scada-cache ({len(net):,} rows, "
              f"{net.index.min().date()} -> {net.index.max().date()})")
        return net

    have = (f"{net.index.min().date()} -> {net.index.max().date()}"
            if len(net) else "empty")
    print(f"  net load: scada-cache covers {have}, need "
          f"{start_ts.date()} -> {end_ts.date()}")
    if not update:
        print("  (--update-cache is 'no'; re-run with 'yes' to pull new days "
              "from the share)")
    return None


def load_net_load(source: str, demand_folder: Path, cache_dir: Path,
                  start_ts, end_ts, refresh: bool,
                  update_cache: bool = True) -> pd.DataFrame:
    """15-min Net_Load indexed by Datetime, from SCADA cache or raw CSVs."""
    if source in ("auto", "scada"):
        net = load_net_load_scada(start_ts, end_ts, update=update_cache)
        if net is not None:
            return _finalise_net_load(net.loc[start_ts:end_ts].copy())
        if source == "scada":
            raise ValueError(
                "scada-cache could not supply the requested range. Run this on "
                "the LAN with --update-cache yes, or use --net-load-source raw."
            )
        print("  falling back to raw demand folder")

    # Raw-folder path keeps its own CSV cache; the SCADA path does not, so a
    # stale CSV can never shadow fresher Parquet.
    cache = cache_dir / "Net_Load_15min.csv"
    if cache.exists() and not refresh:
        nl = pd.read_csv(cache, parse_dates=["Datetime"]).set_index("Datetime")
        if _coverage_ok(nl.index, start_ts, end_ts):
            print(f"  net load: Net_Load_15min.csv cache hit ({len(nl):,} rows)")
            return nl.loc[start_ts:end_ts]
        print("  net load: CSV cache does not cover requested range — rebuilding")

    raw = build_net_load_from_raw(demand_folder, start_ts, end_ts)
    raw["Datetime"] = pd.to_datetime(raw["Datetime"], errors="coerce")
    d15 = raw.set_index("Datetime").resample("15min").mean().loc[start_ts:end_ts]
    d15 = _finalise_net_load(d15)

    cache_dir.mkdir(parents=True, exist_ok=True)
    d15.reset_index().to_csv(cache, index=False)
    print(f"  net load: built from raw and cached ({len(d15):,} rows) -> {cache.name}")
    return d15


# ════════════════════════════════════════════════════════════════════════════
# Weather
# ════════════════════════════════════════════════════════════════════════════
def download_weather(city: str, lat: float, lon: float, start: str, end: str,
                     retries: int = 3) -> pd.DataFrame:
    url = ("https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={lat}&longitude={lon}"
           f"&start_date={start}&end_date={end}"
           "&hourly=temperature_2m,relative_humidity_2m"
           "&timezone=Asia/Kolkata")
    last = None
    for attempt in range(retries):
        try:
            j = requests.get(url, timeout=120).json()
            if "hourly" not in j:
                raise ValueError(f"unexpected response: {str(j)[:200]}")
            return pd.DataFrame({
                "Datetime": pd.to_datetime(j["hourly"]["time"]),
                f"{city}_Temp": j["hourly"]["temperature_2m"],
                f"{city}_Humidity": j["hourly"]["relative_humidity_2m"],
            })
        except Exception as e:                       # transient / rate limit
            last = e
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"weather download failed for {city}: {last}")


def load_weather(cache_dir: Path, start: str, end: str, start_ts, end_ts,
                 refresh: bool) -> pd.DataFrame:
    """Wide 15-min frame of <City>_Temp / <City>_Humidity for all 40 stations.

    Uses v1's cache path and format (Weather_<City>_Hourly.csv) so this script
    and the notebook share one cache. v1's filename does not encode the date
    range, so a cache that does not cover the requested window is re-fetched.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames, downloaded = [], 0

    for city, (lat, lon) in CITY_COORDS.items():
        cache = cache_dir / f"Weather_{city}_Hourly.csv"
        wx = None
        if cache.exists() and not refresh:
            try:
                cached = pd.read_csv(cache, parse_dates=["Datetime"])
                covers = (cached["Datetime"].min() <= start_ts
                          and cached["Datetime"].max() >= end_ts - pd.Timedelta(days=1))
                if covers:
                    wx = cached
            except Exception as e:
                print(f"  {city}: unreadable cache ({e}) — refetching")
        if wx is None:
            wx = download_weather(city, lat, lon, start, end)
            wx.to_csv(cache, index=False)
            downloaded += 1

        frames.append(wx.set_index("Datetime").resample("15min").ffill().loc[start_ts:end_ts])

    print(f"  weather: {len(frames)} stations "
          f"({downloaded} downloaded, {len(frames) - downloaded} from cache)")
    wide = pd.concat(frames, axis=1).ffill().bfill()
    return wide


# ════════════════════════════════════════════════════════════════════════════
# Feature engineering — verbatim port of v5 add_net_load_features
# ════════════════════════════════════════════════════════════════════════════
def rothfusz_heat_index_c(temp_c, rh_pct):
    """NWS Rothfusz regression heat index; C in, C out."""
    T = temp_c * 9.0 / 5.0 + 32.0
    R = rh_pct
    c1, c2, c3, c4 = -42.379, 2.04901523, 10.14333127, -0.22475541
    c5, c6, c7, c8, c9 = -6.83783e-3, -5.481717e-2, 1.22874e-3, 8.5282e-4, -1.99e-6
    HI_f = (c1 + c2 * T + c3 * R + c4 * T * R
            + c5 * T**2 + c6 * R**2
            + c7 * (T**2) * R + c8 * T * (R**2)
            + c9 * (T**2) * (R**2))
    return (HI_f - 32.0) * 5.0 / 9.0


def add_net_load_features(frame: pd.DataFrame):
    """Returns (frame_with_features, feature_names).

    Column set and ordering match the v5 notebook exactly (447 features for
    40 cities). Columns are assembled into a dict and concatenated once rather
    than inserted one at a time, which avoids repeated DataFrame reallocation.
    """
    out = frame.copy().sort_index()
    new: dict[str, pd.Series] = {}
    idx = out.index

    # Calendar
    new["hour"] = pd.Series(idx.hour, index=idx)
    new["minute"] = pd.Series(idx.minute, index=idx)
    new["dayofweek"] = pd.Series(idx.dayofweek, index=idx)
    new["month"] = pd.Series(idx.month, index=idx)
    new["quarter"] = pd.Series(idx.quarter, index=idx)
    new["block"] = pd.Series(idx.hour * 4 + idx.minute // 15, index=idx)
    new["is_evening"] = ((new["hour"] >= 18) & (new["hour"] <= 23)).astype(int)
    new["hour_sin"] = pd.Series(np.sin(2 * np.pi * idx.hour / 24), index=idx)
    new["hour_cos"] = pd.Series(np.cos(2 * np.pi * idx.hour / 24), index=idx)
    new["dow_sin"] = pd.Series(np.sin(2 * np.pi * idx.dayofweek / 7), index=idx)
    new["dow_cos"] = pd.Series(np.cos(2 * np.pi * idx.dayofweek / 7), index=idx)
    new["block_sin"] = np.sin(2 * np.pi * new["block"] / 96)
    new["block_cos"] = np.cos(2 * np.pi * new["block"] / 96)

    # Per-city derived weather (raw Temp/Humidity are already columns of `out`)
    for city in ALL_CITIES:
        t_col, h_col = f"{city}_Temp", f"{city}_Humidity"
        if t_col not in out.columns or h_col not in out.columns:
            continue
        new[f"{city}_cooling"] = (out[t_col] - 22).clip(lower=0)
        new[f"{city}_heating"] = (18 - out[t_col]).clip(lower=0)
        new[f"{city}_heat_idx"] = rothfusz_heat_index_c(out[t_col], out[h_col])
        new[f"{city}_temp_hum"] = out[t_col] * out[h_col]

    # Net-load lags (0-filled, with a missingness flag each)
    for lag in [1, 2, 4, 8, 96, 192, 672, 1344, 364 * 96]:
        shifted = out["Net_Load"].shift(lag)
        new[f"lag_{lag}"] = shifted.fillna(0)
        new[f"lag_{lag}_missing"] = shifted.isna().astype("int8")

    # Per-city weather lags
    for lag in [96, 672]:
        for city in ALL_CITIES:
            if f"{city}_Temp" in out.columns:
                new[f"{city}_temp_lag_{lag}"] = out[f"{city}_Temp"].shift(lag)
            if f"{city}_Humidity" in out.columns:
                new[f"{city}_hum_lag_{lag}"] = out[f"{city}_Humidity"].shift(lag)

    # Rolling net-load statistics (window ends at the previous row)
    prev = out["Net_Load"].shift(1)
    for w in [96, 672, 1344, 364 * 96]:
        rmean, rstd = prev.rolling(w).mean(), prev.rolling(w).std()
        new[f"rolling_mean_{w}"] = rmean.fillna(0)
        new[f"rolling_std_{w}"] = rstd.fillna(0)
        new[f"rolling_mean_{w}_missing"] = rmean.isna().astype("int8")
        new[f"rolling_std_{w}_missing"] = rstd.isna().astype("int8")

    out = pd.concat([out, pd.DataFrame(new, index=idx)], axis=1)
    out = out.ffill().bfill()

    # State-level / composite columns are never model inputs. They are not
    # built by this script, but the exclusion is kept so the port stays exact.
    composite = ({"Weighted_Temp", "Weighted_Humidity"}
                 | {f"{s}_{suf}" for s in WEATHER_STATIONS
                    for suf in ("Temp", "Humidity", "cooling", "heating",
                                "heat_idx", "temp_hum")})
    features = [c for c in out.columns if c != "Net_Load" and c not in composite]
    return out, features


# ════════════════════════════════════════════════════════════════════════════
# Model + importance
# ════════════════════════════════════════════════════════════════════════════
def make_model(n_estimators, max_depth, learning_rate, max_bin, importance_type, seed):
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        sys.exit(
            "LightGBM is required (the CSV records LightGBM feature importances).\n"
            "Install it with:  pip install lightgbm"
        )
    return LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, max_bin=max_bin,
        subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
        reg_alpha=0.5, reg_lambda=3.0,
        importance_type=importance_type,
        random_state=seed, n_jobs=-1, verbose=-1,
    )


def city_importance_table(model, features) -> pd.DataFrame:
    total = model.feature_importances_.sum() or 1.0
    feat_imp = dict(zip(features, model.feature_importances_ / total))
    rows = []
    for city in ALL_CITIES:
        city_cols = [f for f in features if f.startswith(city + "_")]
        imp = sum(feat_imp.get(c, 0.0) for c in city_cols)
        rows.append({
            "City": city,
            "State": CITY_STATE.get(city, "Unknown"),
            "Total_Importance": round(imp, 6),
            "Num_Features": len(city_cols),
            "Recommend": "KEEP" if imp >= 0.001 else "CONSIDER DROPPING",
        })
    return pd.DataFrame(rows).sort_values("Total_Importance", ascending=False)


def compare_with_previous(new_df: pd.DataFrame, path: Path) -> None:
    """Spearman rank correlation against the file being replaced."""
    try:
        old = pd.read_csv(path)[["City", "Total_Importance"]]
    except Exception:
        return
    merged = old.merge(new_df[["City", "Total_Importance"]],
                       on="City", suffixes=("_old", "_new"))
    if len(merged) < 3:
        return
    rho = merged["Total_Importance_old"].corr(merged["Total_Importance_new"],
                                              method="spearman")
    print(f"\n  rank correlation vs previous CSV: rho = {rho:.4f} "
          f"(over {len(merged)} cities)")
    if rho < 0.9:
        print("  ! Below 0.90 — consider raising --sample-frac / --trees, "
              "or run --full to compare against production settings.")


# ════════════════════════════════════════════════════════════════════════════
def main() -> None:
    p = argparse.ArgumentParser(
        description="Regenerate City_Weather_Feature_Importance.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--net-load-source", default="auto", choices=["auto", "scada", "raw"],
                   help="'auto' tries scada-cache then falls back to --demand-folder")
    p.add_argument("--demand-folder", type=Path, default=PROJECT_DIR / "NR_DEMAND_TEMP",
                   help="folder of DD-MM-YYYY.csv raw demand files (fallback source)")
    p.add_argument("--cache-dir", type=Path, default=PROJECT_DIR / "monthly_peak_pipeline_outputs",
                   help="shared weather / net-load cache (same one v1 uses)")
    p.add_argument("--out", type=Path, default=PROJECT_DIR / "City_Weather_Feature_Importance.csv",
                   help="written where v1 reads it")
    p.add_argument("--sample-frac", type=float, default=0.35,
                   help="fraction of rows to fit on (1.0 = all)")
    p.add_argument("--trees", type=int, default=200)
    p.add_argument("--depth", type=int, default=7)
    p.add_argument("--learning-rate", type=float, default=0.08)
    p.add_argument("--max-bin", type=int, default=63, help="lower = faster histograms")
    p.add_argument("--importance-type", default="split", choices=["split", "gain"],
                   help="'split' matches how the existing CSV was produced")
    p.add_argument("--full", action="store_true",
                   help="production settings: all rows, 700 trees, depth 10, lr 0.04")
    p.add_argument("--update-cache", choices=["yes", "no"], default="yes",
                   help="refresh the SCADA cache from the share before reading; "
                        "'no' uses the cache exactly as it stands")
    p.add_argument("--refresh-weather", action="store_true")
    p.add_argument("--refresh-net-load", action="store_true",
                   help="rebuild Net_Load_15min.csv (raw-folder path only)")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    if a.full:
        a.sample_frac, a.trees, a.depth, a.learning_rate, a.max_bin = 1.0, 700, 10, 0.04, 255

    start_ts = pd.Timestamp(a.start)
    end_ts = pd.Timestamp(a.end) + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
    t_start = time.time()

    print(f"Window: {a.start} -> {a.end}")
    print("\n[1/4] Net load")
    nl = load_net_load(a.net_load_source, a.demand_folder, a.cache_dir,
                       start_ts, end_ts, a.refresh_net_load,
                       update_cache=(a.update_cache == "yes"))

    print("\n[2/4] Weather")
    wx = load_weather(a.cache_dir, a.start, a.end, start_ts, end_ts, a.refresh_weather)

    print("\n[3/4] Features")
    merged = nl[["Net_Load"]].join(wx, how="left").ffill().bfill()
    model_df, features = add_net_load_features(merged)
    X, y = model_df[features], model_df["Net_Load"]
    print(f"  {len(X):,} rows x {len(features)} features")

    print("\n[4/4] Model")
    if 0 < a.sample_frac < 1.0:
        rng = np.random.default_rng(a.seed)
        take = np.sort(rng.choice(len(X), size=int(len(X) * a.sample_frac), replace=False))
        X_fit, y_fit = X.iloc[take], y.iloc[take]
    else:
        X_fit, y_fit = X, y
    print(f"  fitting on {len(X_fit):,} rows | trees={a.trees} depth={a.depth} "
          f"max_bin={a.max_bin} importance={a.importance_type}")

    model = make_model(a.trees, a.depth, a.learning_rate, a.max_bin,
                       a.importance_type, a.seed)
    t_fit = time.time()
    model.fit(X_fit, y_fit)
    print(f"  fit took {time.time() - t_fit:.1f}s")

    table = city_importance_table(model, features)
    if a.out.exists():
        compare_with_previous(table, a.out)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out, index=False)
    print(f"\nWrote {a.out}  ({time.time() - t_start:.1f}s total)")
    print(table.head(10).to_string(index=False))

    dropping = table[table["Recommend"] == "CONSIDER DROPPING"]["City"].tolist()
    print(f"\nBelow 0.001 threshold: {dropping or 'none'}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        sys.exit(f"error: {exc}")
    except KeyboardInterrupt:
        sys.exit("\ninterrupted")

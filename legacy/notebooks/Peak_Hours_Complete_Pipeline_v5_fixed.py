"""Reference dump of Peak_Hours_Complete_Pipeline_v5_fixed.ipynb -- code cells only, outputs stripped.

Frozen copy for reading while porting; not meant to be executed as-is.
"""

# --------------------------------------------------------------------------
# # Peak Hours Complete Pipeline — v5 (city-only weather fix)
#
# **RTM Forecast Improvements applied:**
#
# | Tag | Change |
# |-----|--------|
# | IMP-1 | Quantile loss (α=0.85) RTM model blended with mean model (55/45) |
# | IMP-2 | P90 market factor profiles instead of median in future frame |
# | IMP-3 | Night-peak and evening-ramp binary features + night×lag interaction |
# | IMP-4 | Sample-weight high-price training rows (weight ∝ RTM / mean, cap 5×) |
# | IMP-5 | Block-level P10 price floor for output clipping |
# | IMP-6 | Recursive lag blending with historical P75 for day 3+ to cap error |
# | **FIX-1** | **Net-load model uses ONLY per-city Temp & Humidity — no composite Weighted_Temp / per-state averages as direct features; composite kept for lags only** |

# --------------------------------------------------------------------------
# ## Setup & Configuration

# ==========================================================================
# [cell 2]
# ==========================================================================
from pathlib import Path
from datetime import datetime, timedelta
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PROJECT_DIR = Path.cwd()
IEX_CLIENT_ROOT = PROJECT_DIR
DEMAND_FOLDER = PROJECT_DIR / "NR_DEMAND_TEMP"

if str(IEX_CLIENT_ROOT) not in sys.path:
    sys.path.insert(0, str(IEX_CLIENT_ROOT))

from iex_client import get_trade_data

# ── Configuration ─────────────────────────────────────────────────────────────
START_DATE = "2022-07-03"
PEAK_MONTH = "2026-06"       # month to declare peak hours for, YYYY-MM

_peak_period  = pd.Period(PEAK_MONTH, freq="M")
_prev_period  = _peak_period - 1
END_DATE      = str(_prev_period.end_time.date())
print(f"Auto-derived END_DATE (last day of month before PEAK_MONTH): {END_DATE}")

FORECAST_DAYS         = 35
TEST_DAYS             = 92
NL_MIN                = 10_000
NL_MAX                = 100_000
WEATHER_FORECAST_DAYS = 16
REFRESH_MARKET_DATA   = False
HISTORY_TAIL          = 1_344

# [IMP-1] Quantile blend ratio: 55% mean + 45% quantile
RTM_MEAN_WEIGHT     = 0.55
RTM_QUANTILE_WEIGHT = 0.45
RTM_QUANTILE_ALPHA  = 0.85   # target the 85th percentile

# ── State-level weather stations (Open-Meteo archive / forecast API) ───────────
# Each entry: (station_name, lat, lon, intra_state_weight, imd_id)
# intra_state_weight: fraction of this station's contribution to its state's avg
# State-to-NR weights are computed dynamically from LoadActual MW data (Section 3).
WEATHER_STATIONS = {
    "Punjab":     [("Ludhiana",      30.90, 75.86, 0.28, "42027"),
                   ("Amritsar",      31.63, 74.87, 0.22, "42021"),
                   ("Patiala",       30.33, 76.40, 0.18, "42101"),
                   ("Jalandhar",     31.33, 75.58, 0.16, "42024"),
                   ("Bathinda",      30.21, 74.94, 0.16, "42130")],
    "Haryana":    [("Hisar",         29.15, 75.72, 0.18, "42161"),
                   ("Gurugram",      28.46, 77.03, 0.25, "42163"),
                   ("Faridabad",     28.41, 77.31, 0.20, "42164"),
                   ("Ambala",        30.38, 76.78, 0.15, "42022"),
                   ("Rohtak",        28.90, 76.58, 0.12, "42165"),
                   ("Panipat",       29.39, 76.97, 0.10, "42162")],
    "Rajasthan":  [("Jaipur",        26.82, 75.80, 0.25, "42360"),
                   ("Jodhpur",       26.24, 73.02, 0.18, "42339"),
                   ("Kota",          25.17, 75.85, 0.15, "42462"),
                   ("Udaipur",       24.58, 73.71, 0.12, "42461"),
                   ("Bikaner",       28.01, 73.31, 0.12, "42232"),
                   ("Ajmer",         26.45, 74.64, 0.10, "42361"),
                   ("Sriganganagar", 29.92, 73.88, 0.08, "42136")],
    "Delhi":      [("Palam",         28.57, 77.10, 0.42, "42182"),
                   ("Safdarjung",    28.59, 77.21, 0.35, "42183"),
                   ("LodiBhawan",    28.59, 77.22, 0.23, "42184")],
    "UP":         [("Lucknow",       26.85, 80.95, 0.20, "42275"),
                   ("Agra",          27.18, 78.01, 0.12, "42181"),
                   ("Kanpur",        26.47, 80.33, 0.15, "42276"),
                   ("Varanasi",      25.32, 82.97, 0.10, "42379"),
                   ("Prayagraj",     25.45, 81.84, 0.10, "42316"),
                   ("Noida",         28.54, 77.39, 0.13, "42182"),
                   ("Gorakhpur",     26.75, 83.37, 0.08, "42376"),
                   ("Meerut",        28.98, 77.72, 0.07, "42073"),
                   ("Bareilly",      28.36, 79.41, 0.05, "42173")],
    "Uttarakhand":[("Dehradun",      30.32, 78.03, 0.55, "42208"),
                   ("Haridwar",      29.97, 78.17, 0.25, "42209"),
                   ("Roorkee",       29.87, 77.89, 0.20, "42210")],
    "HP":         [("Shimla",        31.10, 77.17, 0.45, "42153"),
                   ("Dharamsala",    32.22, 76.32, 0.30, "42103"),
                   ("Mandi",         31.71, 76.93, 0.25, "42143")],
    "J&K Ladakh": [("Jammu",         32.74, 74.87, 0.40, "42054"),
                   ("Srinagar",      34.08, 74.80, 0.40, "42028"),
                   ("Leh",           34.15, 77.58, 0.20, "42091")],
    "Chd":        [("Chandigarh",    30.74, 76.79, 1.00, "42031")],
}

# Validate intra-state weights sum to 1.0 per state
for _state, _stns in WEATHER_STATIONS.items():
    _w = sum(s[3] for s in _stns)
    assert abs(_w - 1.0) < 1e-6, f"{_state} weights sum to {_w:.4f}"

# Folder containing LoadActual CSVs (place combined_Demand_data*.csv files here)
LOAD_ACTUAL_FOLDER = PROJECT_DIR / "LoadActual"

# Maps WEATHER_STATIONS keys → column names in LoadActual CSVs
STATE_LOAD_COL_MAP = {
    "Punjab":     "Punjab",
    "Haryana":    "Haryana",
    "Rajasthan":  "Rajasthan",
    "Delhi":      "Delhi",
    "UP":         "UP",
    "Uttarakhand":"Uttarakhand",
    "HP":         "HP",
    "J&K Ladakh": "J&K",
    "Chd":        "CHD",
}
NR_TOTAL_COL = "Total NR"

# Long-run mean shares (Apr 2023–Mar 2025 actuals) — used as fallback
FALLBACK_STATE_SHARES = {
    "Punjab":     0.145,
    "Haryana":    0.135,
    "Rajasthan":  0.229,
    "Delhi":      0.072,
    "UP":         0.316,
    "Uttarakhand":0.032,
    "HP":         0.025,
    "J&K Ladakh": 0.043,
    "Chd":        0.003,
}
_fb_sum = sum(FALLBACK_STATE_SHARES.values())
FALLBACK_STATE_SHARES = {k: v / _fb_sum for k, v in FALLBACK_STATE_SHARES.items()}

_n_stations = sum(len(v) for v in WEATHER_STATIONS.values())
print(f"Weather stations configured: {_n_stations} stations across {len(WEATHER_STATIONS)} states")
print(f"States: {', '.join(WEATHER_STATIONS.keys())}")


ARTIFACT_DIR = PROJECT_DIR / "monthly_peak_pipeline_outputs"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

START_TS  = pd.Timestamp(START_DATE)
END_TS    = pd.Timestamp(END_DATE) + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
TEST_SIZE = TEST_DAYS * 96

print("Project:", PROJECT_DIR)
print("Net-load window:", START_DATE, "to", END_DATE)
print("Peak month:", PEAK_MONTH)
print("Artifacts:", ARTIFACT_DIR)


# --------------------------------------------------------------------------
# ## SECTION 1 – Utility helpers

# ==========================================================================
# [cell 4]
# ==========================================================================
def block_to_time(block):
    minutes = (int(block) - 1) * 15
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def add_datetime_from_date_block(df, date_col="Date", block_col="Block"):
    out = df.copy()
    out[date_col]  = pd.to_datetime(out[date_col], dayfirst=True, errors="coerce")
    out[block_col] = pd.to_numeric(out[block_col], errors="coerce").astype("Int64")
    out["Datetime"] = out[date_col] + pd.to_timedelta(
        (out[block_col].astype(int) - 1) * 15, unit="min"
    )
    return out


def robust_z(s):
    s    = pd.Series(s, dtype=float)
    med  = s.median()
    mad  = (s - med).abs().median()
    denom = 1.4826 * mad if mad and np.isfinite(mad) else s.std()
    if not denom or not np.isfinite(denom):
        return pd.Series(0.0, index=s.index)
    return ((s - med) / denom).clip(-5, 5)


def month_bounds(month_yyyy_mm):
    start = pd.Period(month_yyyy_mm, freq="M").start_time
    end   = pd.Period(month_yyyy_mm, freq="M").end_time.floor("15min")
    return start, end


# --------------------------------------------------------------------------
# ## SECTION 2 – Net-load: raw build, resample, clean (strict date-range filtering)

# ==========================================================================
# [cell 6]
# ==========================================================================
RAW_EXPECTED_COLS = {"HRS", "NR Load", "NR Solar", "NR Wind"}


def build_net_load_from_raw(input_folder=DEMAND_FOLDER):
    """
    Load NR demand data from daily CSV files in `input_folder`.

    Only files whose filename date falls strictly within [START_DATE, END_DATE]
    (both inclusive, date-only comparison) are read.  Any file outside that
    range is logged as skipped — it is never silently ignored.

    File naming convention: DD-MM-YYYY.csv  (e.g. 03-07-2022.csv)
    """
    # ── Derive inclusive date-only bounds from global timestamps ──────────────
    date_start = pd.Timestamp(START_DATE).normalize()   # midnight on START_DATE
    date_end   = pd.Timestamp(END_DATE).normalize()     # midnight on END_DATE

    all_csv = sorted(Path(input_folder).glob("*.csv"))
    if not all_csv:
        raise FileNotFoundError(f"No CSV files found in {input_folder}")

    # ── Classify every file before reading anything ───────────────────────────
    in_range, before_range, after_range, unparseable = [], [], [], []

    for file in all_csv:
        try:
            file_date = pd.to_datetime(file.stem, format="%d-%m-%Y").normalize()
        except Exception:
            unparseable.append(file.name)
            continue

        if file_date < date_start:
            before_range.append(file.name)
        elif file_date > date_end:
            after_range.append(file.name)
        else:
            in_range.append((file, file_date))

    # ── Summary before we start reading ──────────────────────────────────────
    print(f"── NR_DEMAND_TEMP scan ({'DD-MM-YYYY'} filenames) ──")
    print(f"  Date range requested  : {date_start.date()}  →  {date_end.date()}")
    print(f"  Total files in folder : {len(all_csv)}")
    print(f"  Files IN range        : {len(in_range)}")
    print(f"  Skipped (before range): {len(before_range)}"
          + (f"  [{before_range[0]} … {before_range[-1]}]" if before_range else ""))
    print(f"  Skipped (after range) : {len(after_range)}"
          + (f"  [{after_range[0]} … {after_range[-1]}]" if after_range else ""))
    if unparseable:
        print(f"  Skipped (bad name)    : {len(unparseable)}  {unparseable}")
    print()

    if not in_range:
        raise ValueError(
            f"No files found between {date_start.date()} and {date_end.date()}. "
            f"Check START_DATE / END_DATE or the folder path."
        )

    # ── Read only in-range files ───────────────────────────────────────────────
    master_df   = []
    read_ok     = 0
    read_errors = []

    for file, file_date in in_range:
        try:
            raw  = pd.read_csv(file, header=None)
            cols = raw.iloc[13].astype(str).str.strip().tolist()
            df_raw = raw.iloc[15:].copy()
            df_raw.columns = (
                pd.Series(cols).astype(str)
                .str.strip().str.replace("\n", " ", regex=False).tolist()
            )
            missing_cols = RAW_EXPECTED_COLS - set(df_raw.columns)
            if missing_cols:
                read_errors.append(f"{file.name} (missing cols: {missing_cols})")
                continue

            df_raw["HRS"] = df_raw["HRS"].astype(str).str.strip()
            df_raw["Datetime"] = pd.to_datetime(
                file_date.strftime("%Y-%m-%d") + " " + df_raw["HRS"],
                format="%Y-%m-%d %H:%M:%S",
                errors="coerce",
            )
            df_raw = df_raw.dropna(subset=["Datetime"])

            for col in ["NR Load", "NR Solar", "NR Wind"]:
                df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

            df_raw["Net_Load"] = df_raw["NR Load"] - df_raw["NR Solar"] - df_raw["NR Wind"]
            master_df.append(df_raw[["Datetime", "Net_Load"]].dropna())
            read_ok += 1

        except Exception as e:
            read_errors.append(f"{file.name} ({e})")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"  Successfully read     : {read_ok} / {len(in_range)} in-range files")
    if read_errors:
        print(f"  Read errors ({len(read_errors)}):")
        for err in read_errors:
            print(f"    ✗ {err}")

    if not master_df:
        raise ValueError("No valid data extracted from in-range files.")

    result = pd.concat(master_df, ignore_index=True).sort_values("Datetime").dropna()
    print(f"\n✔ Net-load built: {len(result):,} rows"
          f"  |  {result['Datetime'].min()}  →  {result['Datetime'].max()}")
    return result


raw_master = build_net_load_from_raw()

df_raw = raw_master.copy()
df_raw["Datetime"] = pd.to_datetime(df_raw["Datetime"], errors="coerce")
df_raw.set_index("Datetime", inplace=True)

df_15min = df_raw.resample("15min").mean()
df_15min = df_15min.loc[START_TS:END_TS]

df_15min["Net_Load"] = df_15min["Net_Load"].fillna(
    (df_15min["Net_Load"].shift(96) + df_15min["Net_Load"].shift(-96)) / 2
)
df_15min["Net_Load"] = df_15min["Net_Load"].interpolate(method="time").ffill().bfill()

net_load_raw = df_15min.reset_index()
net_load_raw.to_csv(ARTIFACT_DIR / "Net_Load_15min.csv", index=False)
print(
    f"15-min grid: {len(net_load_raw):,} rows | "
    f"{net_load_raw['Datetime'].min()} → {net_load_raw['Datetime'].max()}"
)
print("Missing:", net_load_raw["Net_Load"].isna().sum())


# --------------------------------------------------------------------------
# ## SECTION 3 – Weather: historical download & merge (v5: per-city caching + per-city features)

# ==========================================================================
# [cell 8]
# ==========================================================================
# ══════════════════════════════════════════════════════════════════════════════
# V5-3: Universal local cache helpers
# ══════════════════════════════════════════════════════════════════════════════

import hashlib, os
from pathlib import Path

CACHE_DIR = PROJECT_DIR / "pipeline_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _cache_key(prefix, *parts):
    """Deterministic filename from prefix + variable string parts."""
    raw = "__".join(str(p) for p in parts)
    h   = hashlib.md5(raw.encode()).hexdigest()[:8]
    safe = raw.replace("/", "-").replace(" ", "_")[:60]
    return CACHE_DIR / f"{prefix}__{safe}__{h}.csv"


def cache_load(path):
    """Load a CSV cache file; return DataFrame or None."""
    try:
        if Path(path).exists():
            return pd.read_csv(path, parse_dates=["Datetime"])
    except Exception as e:
        print(f"  [cache] Warning: could not load {path}: {e}")
    return None


def cache_save(df, path):
    """Persist a DataFrame to CSV cache."""
    try:
        df.to_csv(path, index=False)
    except Exception as e:
        print(f"  [cache] Warning: could not save {path}: {e}")


# ── Section 3a: Load state-wise actual MW → dynamic share table ───────────────
def load_state_shares_from_actual():
    """
    Reads all LoadActual CSVs, resamples to 15-min, computes each state's
    fraction of Total NR at every timestep.  Returns a DataFrame indexed by
    Datetime with columns  <State>_share  for each state in WEATHER_STATIONS.
    """
    csv_files = sorted(LOAD_ACTUAL_FOLDER.glob("*.csv"))
    if not csv_files:
        print("  WARNING: No CSVs found in LoadActual folder; using equal state shares.")
        return None

    frames = []
    for fp in csv_files:
        try:
            raw = pd.read_csv(fp)
            if "Date" in raw.columns and "Time" in raw.columns:
                raw["Datetime"] = pd.to_datetime(
                    raw["Date"].astype(str) + " " + raw["Time"].astype(str),
                    dayfirst=True, errors="coerce"
                )
                state_cols = [c for c in STATE_LOAD_COL_MAP.values() if c in raw.columns]
                if NR_TOTAL_COL not in raw.columns or not state_cols:
                    continue
                keep = ["Datetime", NR_TOTAL_COL] + state_cols
                frames.append(raw[keep].dropna(subset=["Datetime"]))
        except Exception as exc:
            print(f"  Skipping {fp.name}: {exc}")

    if not frames:
        print("  WARNING: Could not parse any LoadActual CSVs; using equal state shares.")
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates("Datetime").sort_values("Datetime")
    combined = combined.set_index("Datetime")
    combined = combined.apply(pd.to_numeric, errors="coerce")

    combined_15 = combined.resample("15min").mean()
    combined_15 = combined_15.ffill().bfill()

    share_df = pd.DataFrame(index=combined_15.index)
    for state, load_col in STATE_LOAD_COL_MAP.items():
        if load_col in combined_15.columns:
            share_df[f"{state}_share"] = (
                combined_15[load_col] / combined_15[NR_TOTAL_COL]
            ).clip(0.001, 0.999)

    share_cols = [c for c in share_df.columns]
    row_sums = share_df[share_cols].sum(axis=1).replace(0, np.nan)
    share_df[share_cols] = share_df[share_cols].div(row_sums, axis=0)
    share_df = share_df.ffill().bfill()

    print(f"  Dynamic shares: {len(share_df):,} 15-min rows, "
          f"{share_df.index.min().date()} → {share_df.index.max().date()}")
    return share_df

actual_share_df = load_state_shares_from_actual()

FALLBACK_STATE_SHARES = {
    "Punjab":     0.145,
    "Haryana":    0.135,
    "Rajasthan":  0.229,
    "Delhi":      0.072,
    "UP":         0.316,
    "Uttarakhand":0.032,
    "HP":         0.025,
    "J&K Ladakh": 0.043,
    "Chd":        0.003,
}
_fb_sum = sum(FALLBACK_STATE_SHARES.values())
FALLBACK_STATE_SHARES = {k: v / _fb_sum for k, v in FALLBACK_STATE_SHARES.items()}

def get_state_share(state, dt):
    """Return the state's share of NR net-load at a given Datetime."""
    if actual_share_df is not None:
        col = f"{state}_share"
        if col in actual_share_df.columns:
            idx = actual_share_df.index.get_indexer([dt], method="nearest")[0]
            if 0 <= idx < len(actual_share_df):
                return float(actual_share_df[col].iloc[idx])
    return FALLBACK_STATE_SHARES.get(state, 1 / len(WEATHER_STATIONS))


# ── Section 3b: Download / load historical weather per station (V5-3: cached) ─
def download_weather_station(name, lat, lon, start, end):
    """Fetch hourly Temp + Humidity from Open-Meteo archive."""
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        "&hourly=temperature_2m,relative_humidity_2m"
        "&timezone=Asia/Kolkata"
    )
    j = requests.get(url, timeout=120).json()
    return pd.DataFrame({
        "Datetime":         pd.to_datetime(j["hourly"]["time"]),
        f"{name}_Temp":     j["hourly"]["temperature_2m"],
        f"{name}_Humidity": j["hourly"]["relative_humidity_2m"],
    })


station_weather = {}   # station_name → hourly DataFrame (Datetime-indexed)

for state, stations in WEATHER_STATIONS.items():
    for (stn_name, lat, lon, w, imd_id) in stations:
        # V5-3: check cache first (keyed by station + date range)
        cache_path = _cache_key("hist_wx", stn_name, START_DATE, END_DATE)
        cached = cache_load(cache_path)
        if cached is not None:
            sdf = cached
            print(f"  {stn_name} ({state}): loaded from cache ({len(sdf):,} rows)")
        else:
            print(f"  {stn_name} ({state}): downloading …")
            sdf = download_weather_station(stn_name, lat, lon, START_DATE, END_DATE)
            cache_save(sdf, cache_path)
            print(f"  {stn_name} ({state}): cached ({len(sdf):,} rows)")
        station_weather[stn_name] = sdf.set_index("Datetime")


# ── Section 3c: Build per-state aggregated + per-city columns ────────────────
# V5-1: Keep INDIVIDUAL city columns — do NOT reduce to per-state averages only.
# The net-load model receives each city's Temp & Humidity independently so that
# feature importance / SHAP can reveal which cities matter.

all_state_15 = None
states_list  = list(WEATHER_STATIONS.keys())

# First pass: resample each station to 15-min and attach as city columns
city_dfs = {}
for state, stations in WEATHER_STATIONS.items():
    for (stn_name, lat, lon, w, _imd) in stations:
        sdf = station_weather.get(stn_name)
        if sdf is None:
            continue
        sdf_15 = sdf.resample("15min").ffill().loc[START_TS:END_TS]
        city_dfs[stn_name] = sdf_15   # {City}_Temp, {City}_Humidity columns

# Build wide frame with ALL city columns
all_city_15 = None
for stn_name, cdf in city_dfs.items():
    all_city_15 = cdf if all_city_15 is None else all_city_15.join(cdf, how="outer")

all_city_15 = all_city_15.ffill().bfill()

# Second pass: compute intra-state weighted averages (for composite calc)
state_weather_hourly = {}
for state, stations in WEATHER_STATIONS.items():
    t_parts, h_parts = [], []
    for (stn_name, _lat, _lon, w, _imd) in stations:
        sdf = city_dfs.get(stn_name)
        if sdf is None:
            continue
        t_col = f"{stn_name}_Temp"
        h_col = f"{stn_name}_Humidity"
        t_parts.append(w * sdf[t_col])
        h_parts.append(w * sdf[h_col])
    if not t_parts:
        state_weather_hourly[state] = pd.DataFrame()
        continue
    idx = t_parts[0].index
    state_weather_hourly[state] = pd.DataFrame({
        f"{state}_Temp":     sum(t_parts).reindex(idx),
        f"{state}_Humidity": sum(h_parts).reindex(idx),
    }, index=idx)
    print(f"  {state}: aggregated ({len(state_weather_hourly[state]):,} hourly rows)")

# Add per-state aggregated columns alongside city columns
all_state_15 = None
for state, sdf in state_weather_hourly.items():
    if sdf.empty:
        continue
    sdf_15 = sdf.resample("15min").ffill().loc[START_TS:END_TS]
    all_state_15 = sdf_15 if all_state_15 is None else all_state_15.join(sdf_15, how="outer")

all_state_15 = all_state_15.ffill().bfill()

# Join city columns onto state frame → one wide frame with both granularities
all_weather_15 = all_state_15.join(all_city_15, how="outer").ffill().bfill()


# ── Section 3d: Compute share-weighted composite Temp & Humidity ──────────────
def build_weighted_composites(state_df, share_df_arg, fallback_shares):
    """
    For each timestep: Weighted_Temp = Σ share_s(t) * Temp_s(t)
    Uses actual dynamic shares where available, fallback elsewhere.
    """
    idx    = state_df.index
    w_temp = pd.Series(0.0, index=idx)
    w_hum  = pd.Series(0.0, index=idx)
    total_w = pd.Series(0.0, index=idx)

    for state in states_list:
        t_col = f"{state}_Temp"
        h_col = f"{state}_Humidity"
        if t_col not in state_df.columns:
            continue

        if share_df_arg is not None and f"{state}_share" in share_df_arg.columns:
            share_aligned = (
                share_df_arg[f"{state}_share"]
                .reindex(idx, method="nearest", tolerance=pd.Timedelta("30min"))
                .ffill().bfill()
                .fillna(fallback_shares.get(state, 0.0))
            )
        else:
            share_aligned = pd.Series(fallback_shares.get(state, 0.0), index=idx)

        w_temp  += share_aligned * state_df[t_col]
        w_hum   += share_aligned * state_df[h_col]
        total_w += share_aligned

    w_temp = w_temp / total_w.replace(0, np.nan)
    w_hum  = w_hum  / total_w.replace(0, np.nan)
    return w_temp.ffill().bfill(), w_hum.ffill().bfill()

all_weather_15["Weighted_Temp"], all_weather_15["Weighted_Humidity"] = \
    build_weighted_composites(all_state_15, actual_share_df, FALLBACK_STATE_SHARES)

print(f"\n✔ Composite weather built (dynamic state shares).")

# List all available city columns (V5-1)
city_temp_cols = [f"{s}_Temp"     for s in city_dfs]
city_hum_cols  = [f"{s}_Humidity" for s in city_dfs]
print(f"  Per-city columns: {len(city_temp_cols)} Temp + {len(city_hum_cols)} Humidity")

# Merge onto net_load_df
all_weather_cols = (
    ["Weighted_Temp", "Weighted_Humidity"]
    + [f"{s}_Temp" for s in states_list]
    + [f"{s}_Humidity" for s in states_list]
    + city_temp_cols
    + city_hum_cols
)
# deduplicate (state names that are also single-city states)
all_weather_cols = list(dict.fromkeys(all_weather_cols))

weather_reset  = all_weather_15[[c for c in all_weather_cols if c in all_weather_15.columns]].reset_index().rename(columns={"index": "Datetime"})
net_load_df    = pd.merge(net_load_raw, weather_reset, on="Datetime", how="left")
present_cols   = [c for c in all_weather_cols if c in net_load_df.columns]
net_load_df[present_cols] = net_load_df[present_cols].ffill().bfill()

print(f"Merged weather: {len(net_load_df):,} rows | "
      f"state cols: {[f'{s}_Temp' for s in states_list][:3]}…")
print(f"City cols available: {city_temp_cols[:5]}…")


# --------------------------------------------------------------------------
# ## SECTION 4 – Net-load cleaning (outlier replacement)

# ==========================================================================
# [cell 10]
# ==========================================================================
df = net_load_df.copy()
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.sort_values("Datetime").set_index("Datetime")

invalid = (df["Net_Load"] < NL_MIN) | (df["Net_Load"] > NL_MAX)
print("Outlier rows replaced:", int(invalid.sum()))
df.loc[invalid, "Net_Load"] = np.nan

season = 96 * 7
window = 96
for i in range(season + window, len(df)):
    if pd.isna(df["Net_Load"].iloc[i]):
        curr_avg  = df["Net_Load"].iloc[i - window : i].mean()
        week_avg  = df["Net_Load"].iloc[i - season - window : i - season].mean()
        week_same = df["Net_Load"].iloc[i - season]
        if (
            pd.notna(curr_avg)
            and pd.notna(week_avg)
            and pd.notna(week_same)
            and week_avg != 0
        ):
            df.iloc[i, df.columns.get_loc("Net_Load")] = week_same * (curr_avg / week_avg)

df["Net_Load"] = df["Net_Load"].interpolate(method="time").ffill().bfill()
df.reset_index().to_csv(ARTIFACT_DIR / "Net_Load_Final_Clean.csv", index=False)
print(df[["Net_Load", "Weighted_Temp", "Weighted_Humidity"]].describe())


# --------------------------------------------------------------------------
# ## SECTION 5 – Net-load feature engineering (v5: per-city independent weather features)

# ==========================================================================
# [cell 12]
# ==========================================================================
# FIX-1: City-only weather features for net-load model
#
# The model now uses ONLY the individual city Temp & Humidity columns as
# weather inputs (raw + derived signals: cooling, heating, heat_index, temp_hum).
# Composite Weighted_Temp / Weighted_Humidity and per-state averages are NOT
# used as direct features — they only appear as lagged weather signals so the
# model can still capture recent weather history without being biased toward
# any single dominant state.
#
# This makes every city's contribution explicit and learnable, and removes the
# implicit Delhi / UP dominance that comes from share-weighted composites.

ALL_CITIES = [stn_name for state, stns in WEATHER_STATIONS.items()
              for (stn_name, _lat, _lon, _w, _imd) in stns]


def rothfusz_heat_index_c(temp_c, rh_pct):
    """
    NWS Rothfusz regression heat index.
    HI = c1 + c2*T + c3*R + c4*T*R + c5*T^2 + c6*R^2 + c7*T^2*R + c8*T*R^2 + c9*T^2*R^2
    T must be in degrees Fahrenheit, R = relative humidity (%).
    Input/output here are in Celsius to match the rest of the pipeline.
    """
    T = temp_c * 9.0 / 5.0 + 32.0   # C -> F
    R = rh_pct

    c1, c2, c3, c4 = -42.379, 2.04901523, 10.14333127, -0.22475541
    c5, c6, c7, c8, c9 = -6.83783e-3, -5.481717e-2, 1.22874e-3, 8.5282e-4, -1.99e-6

    HI_f = (
        c1 + c2 * T + c3 * R + c4 * T * R
        + c5 * T**2 + c6 * R**2
        + c7 * (T**2) * R + c8 * T * (R**2)
        + c9 * (T**2) * (R**2)
    )
    return (HI_f - 32.0) * 5.0 / 9.0   # F -> C


def add_net_load_features(frame):
    out = frame.copy().sort_index()

    # ── Calendar features ──────────────────────────────────────────────────
    out["hour"]       = out.index.hour
    out["minute"]     = out.index.minute
    out["dayofweek"]  = out.index.dayofweek
    out["month"]      = out.index.month
    out["quarter"]    = out.index.quarter
    out["block"]      = out.index.hour * 4 + out.index.minute // 15
    out["is_evening"] = ((out["hour"] >= 18) & (out["hour"] <= 23)).astype(int)

    out["hour_sin"]  = np.sin(2 * np.pi * out.index.hour / 24)
    out["hour_cos"]  = np.cos(2 * np.pi * out.index.hour / 24)
    out["dow_sin"]   = np.sin(2 * np.pi * out.index.dayofweek / 7)
    out["dow_cos"]   = np.cos(2 * np.pi * out.index.dayofweek / 7)
    out["block_sin"] = np.sin(2 * np.pi * out["block"] / 96)
    out["block_cos"] = np.cos(2 * np.pi * out["block"] / 96)

    # ── FIX-1: Per-CITY weather features (the ONLY direct weather inputs) ──
    # Each city contributes: raw Temp, raw Humidity + four derived signals.
    # No composite or state-aggregate is used as a direct feature here.
    for city in ALL_CITIES:
        t_col = f"{city}_Temp"
        h_col = f"{city}_Humidity"
        if t_col not in out.columns or h_col not in out.columns:
            continue
        out[f"{city}_cooling"]  = (out[t_col] - 22).clip(lower=0)
        out[f"{city}_heating"]  = (18 - out[t_col]).clip(lower=0)
        out[f"{city}_heat_idx"] = rothfusz_heat_index_c(out[t_col], out[h_col])
        out[f"{city}_temp_hum"] = out[t_col] * out[h_col]
        # raw Temp & Humidity columns are already in `out` from the merged df

    # ── Net-load lags ──────────────────────────────────────────────────────
    lags = [1, 2, 4, 8, 96, 192, 672, 1344, 364 * 96]
    for lag in lags:
        out[f"lag_{lag}"]         = out["Net_Load"].shift(lag)
        out[f"lag_{lag}_missing"] = out[f"lag_{lag}"].isna().astype("int8")
        out[f"lag_{lag}"]         = out[f"lag_{lag}"].fillna(0)

    # ── Weather lags: per-city only (lag captures recent weather history) ──
    # We use per-city lags so the model can learn how yesterday's city temp
    # predicts today's load, without polluting direct features with composites.
    weather_lags = [96, 672]
    for lag in weather_lags:
        for city in ALL_CITIES:
            if f"{city}_Temp" in out.columns:
                out[f"{city}_temp_lag_{lag}"] = out[f"{city}_Temp"].shift(lag)
            if f"{city}_Humidity" in out.columns:
                out[f"{city}_hum_lag_{lag}"]  = out[f"{city}_Humidity"].shift(lag)

    # ── Rolling net-load statistics ────────────────────────────────────────
    roll_windows = [96, 672, 1344, 364 * 96]
    for w in roll_windows:
        out[f"rolling_mean_{w}"] = out["Net_Load"].shift(1).rolling(w).mean()
        out[f"rolling_std_{w}"]  = out["Net_Load"].shift(1).rolling(w).std()
        out[f"rolling_mean_{w}_missing"] = out[f"rolling_mean_{w}"].isna().astype("int8")
        out[f"rolling_std_{w}_missing"]  = out[f"rolling_std_{w}"].isna().astype("int8")
        out[f"rolling_mean_{w}"] = out[f"rolling_mean_{w}"].fillna(0)
        out[f"rolling_std_{w}"]  = out[f"rolling_std_{w}"].fillna(0)

    out = out.ffill().bfill()

    # Exclude composite / state-aggregate columns from the feature list.
    # They remain in `out` (needed for lag computation) but are NOT fed to the model.
    composite_cols = (
        {"Weighted_Temp", "Weighted_Humidity"}
        | {f"{s}_Temp"     for s in WEATHER_STATIONS}
        | {f"{s}_Humidity" for s in WEATHER_STATIONS}
        | {f"{s}_cooling"  for s in WEATHER_STATIONS}
        | {f"{s}_heating"  for s in WEATHER_STATIONS}
        | {f"{s}_heat_idx" for s in WEATHER_STATIONS}
        | {f"{s}_temp_hum" for s in WEATHER_STATIONS}
    )

    features = [c for c in out.columns
                if c != "Net_Load" and c not in composite_cols]
    return out, features


model_df, net_features = add_net_load_features(df)
X = model_df[net_features]
y = model_df["Net_Load"]

city_feat_cols = [c for c in net_features
                  if any(c.startswith(city) for city in ALL_CITIES)]
lag_feat_cols  = [c for c in net_features if c.startswith("lag_")]
roll_feat_cols = [c for c in net_features if c.startswith("rolling_")]

print(f"Rows: {len(X)} | Total features: {len(net_features)} | Missing: {int(X.isna().sum().sum())}")
print(f"  City weather features (direct)  : {len(city_feat_cols)}")
print(f"  Net-load lag features           : {len(lag_feat_cols)}")
print(f"  Rolling statistics              : {len(roll_feat_cols)}")
print(f"  Composite / state-avg excluded  : "
      f"{len([c for c in df.columns if c.startswith('Weighted') or '_' in c and c.split('_')[0] in WEATHER_STATIONS])} cols")
print(f"\nFIX-1 active: model uses only per-city Temp & Humidity — "
      f"no Weighted_Temp, no state-level averages as direct inputs.")


# ── City importance utility (unchanged) ───────────────────────────────────────
def prune_cities(model, features, importance_threshold=0.001):
    """
    Identify cities whose total feature importance falls below threshold.
    Usage: cities_to_drop = prune_cities(net_load_model, net_features)
    """
    if not hasattr(model, "feature_importances_"):
        print("  Model does not expose feature_importances_; skipping prune.")
        return []
    imp   = model.feature_importances_
    total = imp.sum() or 1.0
    feat_imp = dict(zip(features, imp / total))
    city_importance = {}
    for city in ALL_CITIES:
        city_cols = [f for f in features if f.startswith(city + "_")]
        city_importance[city] = sum(feat_imp.get(c, 0.0) for c in city_cols)
    low = [c for c, v in city_importance.items() if v < importance_threshold]
    print("City importance summary:")
    for city, val in sorted(city_importance.items(), key=lambda x: -x[1]):
        flag = " ◀ LOW" if val < importance_threshold else ""
        print(f"  {city:20s}: {val:.4f}{flag}")
    print(f"\nCities below threshold ({importance_threshold:.4f}): {low}")
    return low


# --------------------------------------------------------------------------
# ## SECTION 6 – Net-load model: train & evaluate (v5: city importance report)

# ==========================================================================
# [cell 14]
# ==========================================================================
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from lightgbm import LGBMRegressor
    def NetLoadModel():
        return LGBMRegressor(
            objective="regression", max_depth=10, learning_rate=0.04,
            n_estimators=700, subsample=0.9, colsample_bytree=0.9,
            reg_alpha=0.5, reg_lambda=3.0, random_state=42,
            n_jobs=-1, verbose=-1,
        )
except Exception:
    from sklearn.ensemble import HistGradientBoostingRegressor
    def NetLoadModel():
        return HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.05, l2_regularization=0.1, random_state=42
        )

X_train, X_test = X.iloc[:-TEST_SIZE], X.iloc[-TEST_SIZE:]
y_train, y_test = y.iloc[:-TEST_SIZE], y.iloc[-TEST_SIZE:]

eval_model = NetLoadModel()
eval_model.fit(X_train, y_train)
test_pred  = eval_model.predict(X_test)
print({
    "MAE":  round(mean_absolute_error(y_test, test_pred), 2),
    "RMSE": round(float(np.sqrt(mean_squared_error(y_test, test_pred))), 2),
    "R2":   round(r2_score(y_test, test_pred), 4),
})

net_load_model = NetLoadModel()
net_load_model.fit(X, y)
print("Production net-load model trained on full history.")

# V5-1: City contribution analysis — run after production model is trained
print("\n── City-level weather feature importance (V5-1) ──")
cities_to_drop = prune_cities(net_load_model, net_features, importance_threshold=0.001)
if cities_to_drop:
    print(f"\nTip: To prune {cities_to_drop}, remove their columns from WEATHER_STATIONS and re-run.")
else:
    print("All cities contribute above threshold — no pruning recommended.")

# Optional: save city importance to CSV for review
if hasattr(net_load_model, "feature_importances_"):
    imp_total = net_load_model.feature_importances_.sum() or 1.0
    feat_imp  = dict(zip(net_features, net_load_model.feature_importances_ / imp_total))
    city_imp_rows = []
    for city in ALL_CITIES:
        parent_state = next(
            (s for s, stns in WEATHER_STATIONS.items() if any(st[0] == city for st in stns)),
            "Unknown"
        )
        city_cols = [f for f in net_features if f.startswith(city + "_")]
        total_imp = sum(feat_imp.get(c, 0.0) for c in city_cols)
        city_imp_rows.append({
            "City":        city,
            "State":       parent_state,
            "Total_Importance": round(total_imp, 6),
            "Num_Features": len(city_cols),
            "Recommend":   "KEEP" if total_imp >= 0.001 else "CONSIDER DROPPING",
        })
    city_imp_df = pd.DataFrame(city_imp_rows).sort_values("Total_Importance", ascending=False)
    city_imp_df.to_csv(ARTIFACT_DIR / "City_Weather_Feature_Importance.csv", index=False)
    print(f"\nCity importance saved → {ARTIFACT_DIR}/City_Weather_Feature_Importance.csv")
    try:
        display(city_imp_df)
    except Exception:
        print(city_imp_df.to_string(index=False))


# --------------------------------------------------------------------------
# ## SECTION 7 – Net-load forecast for PEAK_MONTH (v5: API 16-day + ML beyond + per-city forecast + caching)

# ==========================================================================
# [cell 16]
# ==========================================================================
# ══════════════════════════════════════════════════════════════════════════════
# V5-2  Weather Forecasting Strategy
# ─────────────────────────────────────────────────────────────────────────────
#  Days  1–16 : Open-Meteo historical-forecast API (free, ~16-day horizon)
#  Days 17+   : Per-city ML weather model trained on historical station data
#               (LightGBM or Ridge; features = calendar + recent lags)
#
# V5-3  All downloaded / computed data is cached locally; re-use on re-runs.
# ══════════════════════════════════════════════════════════════════════════════

DECAY_LAMBDA          = 0.12
API_FORECAST_DAYS     = WEATHER_FORECAST_DAYS   # = 16 from config

forecast_start = pd.Timestamp(END_DATE) + pd.Timedelta(days=1)
month_start, month_end = month_bounds(PEAK_MONTH)
forecast_end   = max(forecast_start + pd.Timedelta(days=FORECAST_DAYS - 1), month_end)

api_cutoff = forecast_start + pd.Timedelta(days=API_FORECAST_DAYS - 1)
print(f"Forecast start         : {forecast_start.date()}")
print(f"Forecast end           : {forecast_end.date()}")
print(f"API weather up to      : {api_cutoff.date()} ({API_FORECAST_DAYS} days)")
print(f"ML weather model from  : {(api_cutoff + pd.Timedelta(days=1)).date()} onward")

all_forecast_idx = pd.date_range(forecast_start, forecast_end, freq="15min")


# ─────────────────────────────────────────────────────────────────────────────
# V5-2a: Historical block-profile helpers (fallback / ML baseline)
# ─────────────────────────────────────────────────────────────────────────────
def city_hist_profile(city_name, target_index):
    """Build median block profile for a single city from historical data."""
    t_col = f"{city_name}_Temp"
    h_col = f"{city_name}_Humidity"
    if t_col not in df.columns:
        # Fall back to state-level
        parent_state = next(
            (s for s, stns in WEATHER_STATIONS.items()
             if any(st[0] == city_name for st in stns)),
            None
        )
        t_col = f"{parent_state}_Temp"     if parent_state else "Weighted_Temp"
        h_col = f"{parent_state}_Humidity" if parent_state else "Weighted_Humidity"

    hist = df[[t_col, h_col]].copy()
    hist["block"] = hist.index.hour * 4 + hist.index.minute // 15
    profile = hist.groupby("block")[[t_col, h_col]].median()

    out = pd.DataFrame(index=target_index)
    out["block"] = out.index.hour * 4 + out.index.minute // 15
    out[f"{city_name}_Temp"]     = out["block"].map(profile[t_col])
    out[f"{city_name}_Humidity"] = out["block"].map(profile[h_col])
    return out.drop(columns="block").ffill().bfill()


def state_hist_profile(state, target_index):
    """Intra-state weighted average of city block profiles."""
    stations = WEATHER_STATIONS[state]
    t_parts, h_parts = [], []
    for (stn_name, _lat, _lon, w, _imd) in stations:
        prof = city_hist_profile(stn_name, target_index)
        t_parts.append(w * prof[f"{stn_name}_Temp"])
        h_parts.append(w * prof[f"{stn_name}_Humidity"])
    out = pd.DataFrame(index=target_index)
    out[f"{state}_Temp"]     = sum(t_parts)
    out[f"{state}_Humidity"] = sum(h_parts)
    return out.ffill().bfill()


# Build historical-profile baseline for ALL cities over full horizon
hist_profile_15 = pd.DataFrame(index=all_forecast_idx)
for state in WEATHER_STATIONS:
    sp = state_hist_profile(state, all_forecast_idx)
    hist_profile_15 = hist_profile_15.join(sp, how="left")

for city in ALL_CITIES:
    cp = city_hist_profile(city, all_forecast_idx)
    hist_profile_15 = hist_profile_15.join(cp, how="left")

hist_profile_15 = hist_profile_15.ffill().bfill()


# ─────────────────────────────────────────────────────────────────────────────
# V5-2b: ML weather forecasting model (days 17+)
# ─────────────────────────────────────────────────────────────────────────────
def build_city_weather_features(series_temp, series_hum, target_dt):
    """
    Feature vector for a city at a forecast datetime:
    calendar + trailing lags from the historical series.
    """
    row = {
        "hour":      target_dt.hour,
        "minute":    target_dt.minute,
        "dayofweek": target_dt.dayofweek,
        "month":     target_dt.month,
        "hour_sin":  np.sin(2 * np.pi * target_dt.hour / 24),
        "hour_cos":  np.cos(2 * np.pi * target_dt.hour / 24),
        "dow_sin":   np.sin(2 * np.pi * target_dt.dayofweek / 7),
        "dow_cos":   np.cos(2 * np.pi * target_dt.dayofweek / 7),
    }
    for lag in [1, 2, 24, 48, 168, 336]:  # in hours
        lag_steps = lag  # series is hourly
        t_val = series_temp.iloc[-lag_steps] if lag_steps <= len(series_temp) else series_temp.median()
        h_val = series_hum.iloc[-lag_steps]  if lag_steps <= len(series_hum)  else series_hum.median()
        row[f"temp_lag_{lag}h"]  = t_val
        row[f"hum_lag_{lag}h"]   = h_val
    return row


def train_city_weather_model(city_name):
    """Train a lightweight ML model to forecast city Temp & Humidity."""
    t_col  = f"{city_name}_Temp"
    h_col  = f"{city_name}_Humidity"
    sdf    = station_weather.get(city_name)
    if sdf is None or t_col not in sdf.columns:
        return None, None, None

    # Build hourly training data with calendar features + lags
    sdf_h = sdf[[t_col, h_col]].dropna().copy()
    if len(sdf_h) < 500:
        return None, None, None

    rows = []
    step = 1  # every hour
    look_back = 336  # two weeks
    for i in range(look_back, len(sdf_h) - step):
        dt = sdf_h.index[i]
        r  = build_city_weather_features(
            sdf_h[t_col].iloc[:i],
            sdf_h[h_col].iloc[:i],
            dt,
        )
        r["target_temp"] = sdf_h[t_col].iloc[i]
        r["target_hum"]  = sdf_h[h_col].iloc[i]
        rows.append(r)

    train_df = pd.DataFrame(rows).dropna()
    if len(train_df) < 100:
        return None, None, None

    feat_cols = [c for c in train_df.columns if c not in ("target_temp", "target_hum")]

    try:
        from lightgbm import LGBMRegressor
        mdl_t = LGBMRegressor(n_estimators=300, learning_rate=0.05,
                               max_depth=6, random_state=42, verbose=-1, n_jobs=-1)
        mdl_h = LGBMRegressor(n_estimators=300, learning_rate=0.05,
                               max_depth=6, random_state=42, verbose=-1, n_jobs=-1)
    except Exception:
        from sklearn.linear_model import Ridge
        mdl_t = Ridge(alpha=1.0)
        mdl_h = Ridge(alpha=1.0)

    mdl_t.fit(train_df[feat_cols], train_df["target_temp"])
    mdl_h.fit(train_df[feat_cols], train_df["target_hum"])
    return mdl_t, mdl_h, feat_cols


# Train per-city weather models (V5-2)  — cached via pickle
import pickle

city_wx_models = {}   # city_name → (model_temp, model_hum, feat_cols)
for city in ALL_CITIES:
    model_cache = _cache_key("city_wx_model", city, START_DATE, END_DATE).with_suffix(".pkl")
    if model_cache.exists():
        with open(model_cache, "rb") as fh:
            city_wx_models[city] = pickle.load(fh)
        print(f"  {city}: weather model loaded from cache")
    else:
        mdl_t, mdl_h, fcols = train_city_weather_model(city)
        city_wx_models[city] = (mdl_t, mdl_h, fcols)
        with open(model_cache, "wb") as fh:
            pickle.dump((mdl_t, mdl_h, fcols), fh)
        status = "trained + cached" if mdl_t is not None else "fallback (hist profile)"
        print(f"  {city}: weather model {status}")


def forecast_city_weather_ml(city_name, target_dt, running_temp, running_hum):
    """
    Predict Temp & Humidity for city at target_dt using trained ML model.
    running_temp / running_hum: hourly Series up to (not including) target_dt.
    Falls back to historical block profile if model is unavailable.
    """
    mdl_t, mdl_h, feat_cols = city_wx_models.get(city_name, (None, None, None))
    if mdl_t is None:
        prof = hist_profile_15[[f"{city_name}_Temp", f"{city_name}_Humidity"]]
        ts   = target_dt.floor("15min")
        row  = prof.loc[ts] if ts in prof.index else prof.iloc[0]
        return float(row[f"{city_name}_Temp"]), float(row[f"{city_name}_Humidity"])

    feat = build_city_weather_features(running_temp, running_hum, target_dt)
    X_row = pd.DataFrame([feat])[feat_cols].fillna(0)
    return float(mdl_t.predict(X_row)[0]), float(mdl_h.predict(X_row)[0])


# ─────────────────────────────────────────────────────────────────────────────
# V5-2c: API forecast for days 1–API_FORECAST_DAYS (V5-3: cached)
# ─────────────────────────────────────────────────────────────────────────────
def download_forecast_station(stn_name, lat, lon, fstart, fend):
    """Fetch forecast weather from Open-Meteo historical-forecast API."""
    url = (
        "https://historical-forecast-api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,relative_humidity_2m"
        f"&start_date={fstart.date()}&end_date={fend.date()}"
        "&timezone=Asia/Kolkata"
    )
    j = requests.get(url, timeout=120).json()
    return pd.DataFrame({
        "Datetime":          pd.to_datetime(j["hourly"]["time"]),
        f"{stn_name}_Temp":      j["hourly"]["temperature_2m"],
        f"{stn_name}_Humidity":  j["hourly"]["relative_humidity_2m"],
    }).set_index("Datetime")


# Download API forecast data (days 1–16) per city — V5-3 cached
api_station_data = {}   # city → hourly DataFrame (API forecast period only)
api_idx = pd.date_range(forecast_start, api_cutoff, freq="h")

for state, stations in WEATHER_STATIONS.items():
    for (stn_name, lat, lon, w, imd_id) in stations:
        cache = _cache_key("fcast_api", stn_name, str(forecast_start.date()), str(api_cutoff.date()))
        cached = cache_load(cache)
        if cached is not None:
            api_station_data[stn_name] = cached.set_index("Datetime")
            print(f"  {stn_name} ({state}): API forecast loaded from cache")
        else:
            try:
                raw = download_forecast_station(stn_name, lat, lon, forecast_start, api_cutoff)
                raw_reset = raw.reset_index()
                cache_save(raw_reset, cache)
                api_station_data[stn_name] = raw
                print(f"  {stn_name} ({state}): API forecast downloaded & cached")
            except Exception as exc:
                print(f"  {stn_name} ({state}): API error, using hist profile ({exc})")
                api_station_data[stn_name] = None


# ─────────────────────────────────────────────────────────────────────────────
# V5-2d: Assemble full forecast weather (15-min) for all cities
#         Days 1–16  : API data (blended with hist-profile using decay weight)
#         Days 17+   : ML weather model (recursive city-by-city)
# ─────────────────────────────────────────────────────────────────────════════
_api_day_idx = pd.date_range(forecast_start, api_cutoff, freq="D")
_day_weights = {day: float(np.exp(-DECAY_LAMBDA * i)) for i, day in enumerate(_api_day_idx)}
weight_series = pd.Series(
    {ts: _day_weights.get(ts.normalize(), 0.0)
     for ts in pd.date_range(forecast_start, api_cutoff, freq="15min")},
    name="weight",
)

# Build per-city 15-min forecast DataFrames
city_forecast_15 = {}   # city → 15-min DataFrame over full horizon

for state, stations in WEATHER_STATIONS.items():
    for (stn_name, lat, lon, w, imd_id) in stations:
        t_col = f"{stn_name}_Temp"
        h_col = f"{stn_name}_Humidity"

        # -- API portion (days 1–16) ------------------------------------------
        api_raw = api_station_data.get(stn_name)

        if api_raw is not None and t_col in api_raw.columns:
            api_15 = (api_raw[[t_col, h_col]]
                      .resample("15min").ffill()
                      .reindex(pd.date_range(forecast_start, api_cutoff, freq="15min"))
                      .ffill().bfill())
        else:
            api_15 = hist_profile_15[[t_col, h_col]].loc[forecast_start:api_cutoff].copy()

        # Blend API with historical profile
        prof_api = hist_profile_15[[t_col, h_col]].loc[forecast_start:api_cutoff].copy()
        w_aligned = weight_series.reindex(api_15.index, fill_value=0.0)
        blended_api = pd.DataFrame(index=api_15.index)
        for col in [t_col, h_col]:
            valid = api_15[col].notna() & (w_aligned > 0)
            blended_api[col] = prof_api[col].copy()
            blended_api.loc[valid, col] = (
                w_aligned[valid] * api_15.loc[valid, col]
                + (1 - w_aligned[valid]) * prof_api.loc[valid, col]
            )

        # -- ML portion (days 17+) --------------------------------------------
        ml_start    = api_cutoff + pd.Timedelta(minutes=15)
        ml_forecast_idx = pd.date_range(ml_start, forecast_end, freq="15min")

        if len(ml_forecast_idx) == 0:
            city_forecast_15[stn_name] = blended_api
            continue

        # Build running hourly series from historical + API portion
        hist_hourly = station_weather.get(stn_name)
        if hist_hourly is not None and t_col in hist_hourly.columns:
            run_temp = hist_hourly[t_col].copy()
            run_hum  = hist_hourly[h_col].copy()
        else:
            run_temp = df["Weighted_Temp"].resample("h").mean()
            run_hum  = df["Weighted_Humidity"].resample("h").mean()

        # Append API-period values (hourly)
        api_h = api_15[[t_col, h_col]].resample("h").mean()
        run_temp = pd.concat([run_temp, api_h[t_col]]).drop_duplicates()
        run_hum  = pd.concat([run_hum,  api_h[h_col]]).drop_duplicates()

        # Recursive ML forecast at 15-min resolution
        ml_rows_t, ml_rows_h = [], []
        for ts in ml_forecast_idx:
            t_pred, h_pred = forecast_city_weather_ml(stn_name, ts, run_temp, run_hum)
            ml_rows_t.append(t_pred)
            ml_rows_h.append(h_pred)
            # Append to running series (hourly; add only on the hour)
            if ts.minute == 0:
                new_t = pd.Series([t_pred], index=[ts])
                new_h = pd.Series([h_pred], index=[ts])
                run_temp = pd.concat([run_temp, new_t])
                run_hum  = pd.concat([run_hum,  new_h])

        ml_df = pd.DataFrame(
            {t_col: ml_rows_t, h_col: ml_rows_h},
            index=ml_forecast_idx,
        )

        # Concatenate API + ML portions
        city_forecast_15[stn_name] = pd.concat([blended_api, ml_df])

print(f"\n✔ Per-city forecast weather assembled:")
print(f"  API portion  : {forecast_start.date()} → {api_cutoff.date()} (blended with hist profile)")
print(f"  ML portion   : {(api_cutoff + pd.Timedelta(days=1)).date()} → {forecast_end.date()} (city ML model)")


# ─────────────────────────────────────────────────────────────────────────────
# V5-2e: Aggregate city forecasts → per-state + composite
# ─────────────────────────────────────────────────────────────────────────────
future_weather_15 = pd.DataFrame(index=all_forecast_idx)

for state, stations in WEATHER_STATIONS.items():
    t_agg = pd.Series(0.0, index=all_forecast_idx)
    h_agg = pd.Series(0.0, index=all_forecast_idx)
    for (stn_name, _lat, _lon, w, _imd) in stations:
        stn_df = city_forecast_15.get(stn_name)
        t_col  = f"{stn_name}_Temp"
        h_col  = f"{stn_name}_Humidity"
        if stn_df is not None and t_col in stn_df.columns:
            t_agg += w * stn_df[t_col].reindex(all_forecast_idx).ffill().bfill().fillna(0)
            h_agg += w * stn_df[h_col].reindex(all_forecast_idx).ffill().bfill().fillna(0)
    future_weather_15[f"{state}_Temp"]     = t_agg
    future_weather_15[f"{state}_Humidity"] = h_agg

# Add per-city columns to future_weather_15 (V5-1: needed in forecast loop)
for stn_name, stn_df in city_forecast_15.items():
    for suffix in ["Temp", "Humidity"]:
        col = f"{stn_name}_{suffix}"
        if stn_df is not None and col in stn_df.columns:
            future_weather_15[col] = stn_df[col].reindex(all_forecast_idx).ffill().bfill()

future_weather_15 = future_weather_15.ffill().bfill()

# Composite Weighted_Temp / Weighted_Humidity using dynamic state shares
def composite_at_time(ts, fw15_row):
    w_temp, w_hum, total_w = 0.0, 0.0, 0.0
    for state in WEATHER_STATIONS:
        t_col = f"{state}_Temp"
        h_col = f"{state}_Humidity"
        if t_col not in fw15_row.index:
            continue
        share  = get_state_share(state, ts)
        w_temp  += share * fw15_row[t_col]
        w_hum   += share * fw15_row[h_col]
        total_w += share
    if total_w > 0:
        return w_temp / total_w, w_hum / total_w
    return np.nan, np.nan

wt_list, wh_list = [], []
for ts, row in future_weather_15.iterrows():
    wt, wh = composite_at_time(ts, row)
    wt_list.append(wt)
    wh_list.append(wh)

future_weather_15["Weighted_Temp"]     = wt_list
future_weather_15["Weighted_Humidity"] = wh_list
future_weather_15 = future_weather_15.ffill().bfill()

print(f"\nState-wise + per-city forecast weather ready:")
print(f"  Index: {future_weather_15.index.min().date()} → {future_weather_15.index.max().date()}")
print(f"  Columns: {len(future_weather_15.columns)} ({list(future_weather_15.columns)[:5]}…)")


# ─────────────────────────────────────────────────────────────────────────────
# Recursive net-load forecast using per-city + per-state weather
# ─────────────────────────────────────────────────────────────────────────────
future_df = df.copy()
preds = []
times = pd.date_range(forecast_start, forecast_end, freq="15min")

for next_time in times:
    fw_row = future_weather_15.loc[next_time]

    # FIX-1: Inject ALL weather columns (composite + state + city) so that
    # lag computation inside add_net_load_features works correctly.
    # The feature list (net_features) already excludes composite / state-avg
    # columns — they will be computed but NOT passed to the model.
    row = pd.DataFrame(index=[next_time])
    row["Net_Load"]          = np.nan
    row["Weighted_Temp"]     = fw_row["Weighted_Temp"]
    row["Weighted_Humidity"] = fw_row["Weighted_Humidity"]
    for state in WEATHER_STATIONS:
        row[f"{state}_Temp"]     = fw_row.get(f"{state}_Temp",     np.nan)
        row[f"{state}_Humidity"] = fw_row.get(f"{state}_Humidity", np.nan)
    for city in ALL_CITIES:
        row[f"{city}_Temp"]     = fw_row.get(f"{city}_Temp",     np.nan)
        row[f"{city}_Humidity"] = fw_row.get(f"{city}_Humidity", np.nan)

    temp       = pd.concat([future_df, row])
    feat_df, _ = add_net_load_features(temp)

    # Only select features the model was trained on (city-only, no composite)
    row_feats = feat_df.loc[[next_time], [f for f in net_features if f in feat_df.columns]]
    for mf in [f for f in net_features if f not in row_feats.columns]:
        row_feats[mf] = 0.0
    row_feats = row_feats[net_features]

    pred = float(net_load_model.predict(row_feats)[0])
    row["Net_Load"] = pred
    future_df = pd.concat([future_df, row])
    preds.append(pred)

net_forecast = pd.DataFrame({"Datetime": times, "Forecast_Net_Load": preds})
net_forecast.to_csv(ARTIFACT_DIR / "Net_Load_Forecast.csv", index=False)
print(net_forecast["Datetime"].min(), "to", net_forecast["Datetime"].max(), len(net_forecast))


# --------------------------------------------------------------------------
# ## SECTION 8 – Market data: DAM / RTM load or fetch (V5-3: cache-backed)

# ==========================================================================
# [cell 18]
# ==========================================================================
dam_path = PROJECT_DIR / "DAM_Prices_2022_2026.csv"
rtm_path = PROJECT_DIR / "RTM_Prices_2022_2026.csv"
MARKET_SNAPSHOT_COLUMNS = {
    "Purchase_Bid_MW",
    "Sell_Bid_MW",
    "MCV_MW",
    "Final_Scheduled_Volume_MW",
    "MCP_Rs_MWh",
}

# V5-3: Also maintain standardised cache entries for market data
_dam_cache = _cache_key("market_dam", START_DATE, END_DATE)
_rtm_cache = _cache_key("market_rtm", START_DATE, END_DATE)


def _iex_client_date(value):
    return pd.Timestamp(value).strftime("%d-%m-%Y")


def load_market_from_iex_cache(market_name, start, end):
    print(f"{market_name}: loading {start} to {end} via iex_client cache")
    data = get_trade_data(
        _iex_client_date(start),
        _iex_client_date(end),
        market_name,
    )

    price_col = f"{market_name}_Price"

    if {"Date", "Block"}.issubset(data.columns):
        out = data.copy()
        for area in ["N1", "N2", "N3"]:
            if area not in out.columns:
                out[area] = np.nan
            out[area] = pd.to_numeric(out[area], errors="coerce")
        if "NR_Avg_Price" in out.columns:
            out[price_col] = pd.to_numeric(out["NR_Avg_Price"], errors="coerce")
        else:
            out[price_col] = out[["N1", "N2", "N3"]].mean(axis=1)
        if "Market" not in out.columns:
            out["Market"] = market_name
        return out

    block_cols = [str(i) for i in range(1, 97)]
    area_rows  = data[data["Area"].isin(["N1", "N2", "N3"])].copy()
    long = area_rows.melt(
        id_vars=["Date", "Area"],
        value_vars=block_cols,
        var_name="Block",
        value_name="Price",
    )
    long["Block"] = pd.to_numeric(long["Block"], errors="coerce").astype("Int64")
    long["Price"] = pd.to_numeric(long["Price"], errors="coerce")

    out = (
        long.pivot_table(
            index=["Date", "Block"],
            columns="Area",
            values="Price",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for area in ["N1", "N2", "N3"]:
        if area not in out.columns:
            out[area] = np.nan

    out["Market"]  = market_name
    out[price_col] = out[["N1", "N2", "N3"]].mean(axis=1)
    return out[["Date", "Block", "Market", "N1", "N2", "N3", price_col]]


def market_csv_ready(path, price_col):
    if not path.exists():
        return False
    try:
        cols = set(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return False
    return price_col in cols and MARKET_SNAPSHOT_COLUMNS.issubset(cols)


MARKET_END_DATE = pd.Timestamp(END_DATE).date()

_start_ts_market = pd.Timestamp(START_DATE)
_end_ts_market   = pd.Timestamp(END_DATE) + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)

if (
    REFRESH_MARKET_DATA
    or not market_csv_ready(dam_path, "DAM_Price")
    or not market_csv_ready(rtm_path, "RTM_Price")
):
    # V5-3: try pipeline_cache first before hitting IEX
    _dam_cached = cache_load(_dam_cache) if not REFRESH_MARKET_DATA else None
    _rtm_cached = cache_load(_rtm_cache) if not REFRESH_MARKET_DATA else None

    if _dam_cached is not None and _rtm_cached is not None:
        dam = _dam_cached
        rtm = _rtm_cached
        print("Market data loaded from pipeline_cache (no IEX call needed).")
    else:
        dam = load_market_from_iex_cache("DAM", START_DATE, MARKET_END_DATE)
        rtm = load_market_from_iex_cache("RTM", START_DATE, MARKET_END_DATE)
        dam.to_csv(dam_path, index=False)
        rtm.to_csv(rtm_path, index=False)
        # Also save to pipeline_cache
        cache_save(dam, _dam_cache)
        cache_save(rtm, _rtm_cache)
        print("Market data fetched from IEX and cached (project + pipeline_cache).")
else:
    dam = pd.read_csv(dam_path)
    rtm = pd.read_csv(rtm_path)
    print("Market data loaded from project-level CSV cache.")

if "NR_Avg_Price" in dam.columns and "DAM_Price" not in dam.columns:
    dam = dam.rename(columns={"NR_Avg_Price": "DAM_Price"})
if "NR_Avg_Price" in rtm.columns and "RTM_Price" not in rtm.columns:
    rtm = rtm.rename(columns={"NR_Avg_Price": "RTM_Price"})

if "Datetime" not in dam.columns:
    dam = add_datetime_from_date_block(dam)
if "Datetime" not in rtm.columns:
    rtm = add_datetime_from_date_block(rtm)

dam["Datetime"] = pd.to_datetime(dam["Datetime"])
rtm["Datetime"] = pd.to_datetime(rtm["Datetime"])

dam = dam[(dam["Datetime"] >= _start_ts_market) & (dam["Datetime"] <= _end_ts_market)].copy()
rtm = rtm[(rtm["Datetime"] >= _start_ts_market) & (rtm["Datetime"] <= _end_ts_market)].copy()

print(f"Market data clipped: {START_DATE} → {END_DATE} (one month before {PEAK_MONTH})")
print("DAM:", dam.shape, dam["Datetime"].min(), dam["Datetime"].max())
print("RTM:", rtm.shape, rtm["Datetime"].min(), rtm["Datetime"].max())


# --------------------------------------------------------------------------
# ## SECTION 9 – Market pressure features & RTM model data

# ==========================================================================
# [cell 20]
# ==========================================================================
MARKET_FACTOR_COLS = [
    "Purchase_Bid_MW",
    "Sell_Bid_MW",
    "MCV_MW",
    "Final_Scheduled_Volume_MW",
    "MCP_Rs_MWh",
]
RTM_MARKET_FACTOR_COLS = MARKET_FACTOR_COLS
DAM_MARKET_FACTOR_COLS = [f"dam_{col}" for col in MARKET_FACTOR_COLS]

for col in MARKET_FACTOR_COLS:
    if col not in rtm.columns:
        rtm[col] = np.nan
    rtm[col] = pd.to_numeric(rtm[col], errors="coerce")
    if col not in dam.columns:
        dam[col] = np.nan
    dam[col] = pd.to_numeric(dam[col], errors="coerce")

dam_factors = dam[["Datetime"] + MARKET_FACTOR_COLS].rename(
    columns={col: f"dam_{col}" for col in MARKET_FACTOR_COLS}
)
market = pd.merge(
    dam[["Datetime", "Date", "Block", "DAM_Price"]],
    dam_factors,
    on="Datetime",
    how="inner",
)
market = pd.merge(
    market,
    rtm[["Datetime", "RTM_Price"] + RTM_MARKET_FACTOR_COLS],
    on="Datetime",
    how="inner",
)
hist_net = df[["Net_Load"]].reset_index()
market   = pd.merge(market, hist_net, on="Datetime", how="left")
market   = market.sort_values("Datetime")
market["Net_Load"] = market["Net_Load"].interpolate().ffill().bfill()


def add_market_pressure_features(
    frame, purchase_col, sell_col, mcv_col, sched_col, mcp_col, out_prefix=""
):
    out = frame.copy()
    for col in [purchase_col, sell_col, mcv_col, sched_col, mcp_col]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    pfx = f"{out_prefix}_" if out_prefix else ""
    eps = 1e-6
    out[f"{pfx}Bid_Spread_MW"]             = out[purchase_col] - out[sell_col]
    out[f"{pfx}Bid_Ratio"]                 = out[purchase_col] / (out[sell_col].abs() + eps)
    out[f"{pfx}Cleared_Share_of_Purchase"] = out[mcv_col] / (out[purchase_col].abs() + eps)
    out[f"{pfx}Cleared_Share_of_Sell"]     = out[mcv_col] / (out[sell_col].abs() + eps)
    out[f"{pfx}Schedule_MCV_Gap_MW"]       = out[sched_col] - out[mcv_col]
    out[f"{pfx}MCP_Rs_kWh"]                = out[mcp_col] / 1000.0
    return out


def add_all_market_pressure_features(frame):
    out = add_market_pressure_features(
        frame,
        "Purchase_Bid_MW", "Sell_Bid_MW", "MCV_MW",
        "Final_Scheduled_Volume_MW", "MCP_Rs_MWh",
        out_prefix="",
    )
    out = add_market_pressure_features(
        out,
        "dam_Purchase_Bid_MW", "dam_Sell_Bid_MW", "dam_MCV_MW",
        "dam_Final_Scheduled_Volume_MW", "dam_MCP_Rs_MWh",
        out_prefix="dam",
    )
    return out


RTM_MARKET_DERIVED_COLS = [
    "Bid_Spread_MW",
    "Bid_Ratio",
    "Cleared_Share_of_Purchase",
    "Cleared_Share_of_Sell",
    "Schedule_MCV_Gap_MW",
    "MCP_Rs_kWh",
]
DAM_MARKET_DERIVED_COLS  = [f"dam_{col}" for col in RTM_MARKET_DERIVED_COLS]
ALL_MARKET_RAW_COLS      = RTM_MARKET_FACTOR_COLS + DAM_MARKET_FACTOR_COLS
ALL_MARKET_DERIVED_COLS  = RTM_MARKET_DERIVED_COLS + DAM_MARKET_DERIVED_COLS


def add_rtm_features(frame, include_market_factors=True):
    """
    Build RTM model features.
    [IMP-3] Added is_night_demand, is_evening_ramp, is_peak_regime, night_rtm_lag1
            to capture the structurally different midnight–5am and 6pm–11pm regimes.
    """
    out = frame.copy().sort_values("Datetime").reset_index(drop=True)
    out = add_all_market_pressure_features(out)
    out["hour"]      = out["Datetime"].dt.hour
    out["minute"]    = out["Datetime"].dt.minute
    out["dayofweek"] = out["Datetime"].dt.dayofweek
    out["month"]     = out["Datetime"].dt.month

    out["block"]     = out["Block"].astype(int)   # 1-based
    out["block_sin"] = np.sin(2 * np.pi * (out["block"] - 1) / 96)
    out["block_cos"] = np.cos(2 * np.pi * (out["block"] - 1) / 96)
    out["hour_sin"]  = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"]  = np.cos(2 * np.pi * out["hour"] / 24)
    out["net_ramp"]  = out["Net_Load"].diff().fillna(0)
    out["dam_ramp"]  = out["DAM_Price"].diff().fillna(0)

    # [IMP-3] Regime flags: night demand (00:00–04:45) and evening ramp (18:00–22:00)
    out["is_night_demand"] = ((out["Block"] >= 1)  & (out["Block"] <= 20)).astype(int)
    out["is_evening_ramp"] = ((out["Block"] >= 73) & (out["Block"] <= 88)).astype(int)
    out["is_peak_regime"]  = ((out["is_night_demand"] == 1) | (out["is_evening_ramp"] == 1)).astype(int)

    for col in ALL_MARKET_RAW_COLS + ALL_MARKET_DERIVED_COLS:
        out[f"{col}_ramp"] = out[col].diff().fillna(0)

    for lag in [1, 2, 4, 96, 672]:
        out[f"rtm_lag_{lag}"] = out["RTM_Price"].shift(lag)
        out[f"dam_lag_{lag}"] = out["DAM_Price"].shift(lag)
        if include_market_factors:
            for col in ALL_MARKET_RAW_COLS + ALL_MARKET_DERIVED_COLS:
                out[f"{col}_lag_{lag}"] = out[col].shift(lag)

    # [IMP-3] Night-price stickiness interaction feature
    out["night_rtm_lag1"] = out["rtm_lag_1"] * out["is_night_demand"]

    for w in [96, 672]:
        out[f"rtm_roll_mean_{w}"] = out["RTM_Price"].shift(1).rolling(w).mean()
        out[f"dam_roll_mean_{w}"] = out["DAM_Price"].shift(1).rolling(w).mean()
        if include_market_factors:
            for col in ALL_MARKET_RAW_COLS + ALL_MARKET_DERIVED_COLS:
                out[f"{col}_roll_mean_{w}"] = out[col].shift(1).rolling(w).mean()

    out = out.ffill().bfill()

    features = [
        "Block", "DAM_Price", "Net_Load", "hour", "minute", "dayofweek", "month",
        "block_sin", "block_cos", "hour_sin", "hour_cos", "net_ramp", "dam_ramp",
        # [IMP-3] new regime features
        "is_night_demand", "is_evening_ramp", "is_peak_regime", "night_rtm_lag1",
    ] + [c for c in out.columns if c.startswith(("rtm_lag_", "dam_lag_", "rtm_roll_", "dam_roll_"))]
    if include_market_factors:
        market_features = ALL_MARKET_RAW_COLS + ALL_MARKET_DERIVED_COLS
        market_features += [
            c for c in out.columns
            if c.endswith("_ramp")
            or any(
                c.startswith(f"{col}_lag_") or c.startswith(f"{col}_roll_mean_")
                for col in ALL_MARKET_RAW_COLS + ALL_MARKET_DERIVED_COLS
            )
        ]
        features += [c for c in market_features if c in out.columns]
    features = list(dict.fromkeys(features))
    return out, features


# --------------------------------------------------------------------------
# ## SECTION 10 – RTM models: mean + quantile (v2)

# ==========================================================================
# [cell 22]
# ==========================================================================
try:
    from lightgbm import LGBMRegressor

    def RTMModel():
        """Mean (MSE) model — same as v1."""
        return LGBMRegressor(
            objective="regression", max_depth=8, learning_rate=0.04,
            n_estimators=600, subsample=0.9, colsample_bytree=0.9,
            reg_alpha=0.5, reg_lambda=3.0, random_state=42,
            n_jobs=-1, verbose=-1,
        )

    # [IMP-1] Quantile model targeting P85 to capture tail spikes
    def RTMModelQuantile(alpha=RTM_QUANTILE_ALPHA):
        return LGBMRegressor(
            objective="quantile", alpha=alpha,
            max_depth=8, learning_rate=0.04,
            n_estimators=600, subsample=0.9, colsample_bytree=0.9,
            reg_alpha=0.5, reg_lambda=3.0, random_state=42,
            n_jobs=-1, verbose=-1,
        )

except Exception:
    from sklearn.ensemble import HistGradientBoostingRegressor

    def RTMModel():
        return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=42)

    def RTMModelQuantile(alpha=RTM_QUANTILE_ALPHA):
        return HistGradientBoostingRegressor(
            loss="quantile", quantile=alpha,
            max_iter=500, learning_rate=0.05, random_state=42,
        )


baseline_rtm_df, baseline_rtm_features = add_rtm_features(market, include_market_factors=False)
rtm_model_df,    rtm_features           = add_rtm_features(market, include_market_factors=True)
train          = rtm_model_df.dropna(subset=["RTM_Price"])
baseline_train = baseline_rtm_df.loc[train.index].copy()

split = int(len(train) * 0.8)

# ── [IMP-4] Compute sample weights — emphasise high-price rows ────────────────
mean_rtm_price   = train["RTM_Price"].mean()
sample_weights   = (train["RTM_Price"] / mean_rtm_price).clip(1.0, 5.0).values
train_sw         = sample_weights[:split]   # weights for the training portion

# Baseline eval (no sample weights, no market factors — kept identical to v1)
baseline_eval = RTMModel()
baseline_eval.fit(
    baseline_train.iloc[:split][baseline_rtm_features],
    baseline_train.iloc[:split]["RTM_Price"],
)
baseline_pred_eval = baseline_eval.predict(baseline_train.iloc[split:][baseline_rtm_features])

# Enhanced mean model with sample weights
rtm_eval_mean = RTMModel()
rtm_eval_mean.fit(
    train.iloc[:split][rtm_features],
    train.iloc[:split]["RTM_Price"],
    sample_weight=train_sw,
)
rtm_pred_mean = rtm_eval_mean.predict(train.iloc[split:][rtm_features])

# [IMP-1] Quantile model eval
rtm_eval_q = RTMModelQuantile()
rtm_eval_q.fit(
    train.iloc[:split][rtm_features],
    train.iloc[:split]["RTM_Price"],
    sample_weight=train_sw,
)
rtm_pred_q = rtm_eval_q.predict(train.iloc[split:][rtm_features])

# Blended eval prediction
rtm_pred_blended = RTM_MEAN_WEIGHT * rtm_pred_mean + RTM_QUANTILE_WEIGHT * rtm_pred_q

rtm_metric_rows = [
    {
        "Model":    "Baseline",
        "Features": len(baseline_rtm_features),
        "RTM_MAE":  mean_absolute_error(baseline_train.iloc[split:]["RTM_Price"], baseline_pred_eval),
        "RTM_RMSE": float(np.sqrt(mean_squared_error(baseline_train.iloc[split:]["RTM_Price"], baseline_pred_eval))),
    },
    {
        "Model":    "Enhanced_Mean",
        "Features": len(rtm_features),
        "RTM_MAE":  mean_absolute_error(train.iloc[split:]["RTM_Price"], rtm_pred_mean),
        "RTM_RMSE": float(np.sqrt(mean_squared_error(train.iloc[split:]["RTM_Price"], rtm_pred_mean))),
    },
    {
        "Model":    f"Enhanced_Q{int(RTM_QUANTILE_ALPHA*100)}",
        "Features": len(rtm_features),
        "RTM_MAE":  mean_absolute_error(train.iloc[split:]["RTM_Price"], rtm_pred_q),
        "RTM_RMSE": float(np.sqrt(mean_squared_error(train.iloc[split:]["RTM_Price"], rtm_pred_q))),
    },
    {
        "Model":    f"Blended_{int(RTM_MEAN_WEIGHT*100)}mean_{int(RTM_QUANTILE_WEIGHT*100)}q",
        "Features": len(rtm_features),
        "RTM_MAE":  mean_absolute_error(train.iloc[split:]["RTM_Price"], rtm_pred_blended),
        "RTM_RMSE": float(np.sqrt(mean_squared_error(train.iloc[split:]["RTM_Price"], rtm_pred_blended))),
    },
]

rtm_metrics = pd.DataFrame(rtm_metric_rows)
rtm_metrics["MAE_Improvement_vs_Baseline"]  = rtm_metrics.loc[0, "RTM_MAE"]  - rtm_metrics["RTM_MAE"]
rtm_metrics["RMSE_Improvement_vs_Baseline"] = rtm_metrics.loc[0, "RTM_RMSE"] - rtm_metrics["RTM_RMSE"]
rtm_metrics[["RTM_MAE", "RTM_RMSE", "MAE_Improvement_vs_Baseline", "RMSE_Improvement_vs_Baseline"]] = (
    rtm_metrics[["RTM_MAE", "RTM_RMSE", "MAE_Improvement_vs_Baseline", "RMSE_Improvement_vs_Baseline"]].round(2)
)
rtm_metrics.to_csv(ARTIFACT_DIR / f"RTM_Model_Metrics_{PEAK_MONTH}.csv", index=False)
print(rtm_metrics)

# ── Production models trained on full history with sample weights ─────────────
rtm_model_mean = RTMModel()
rtm_model_mean.fit(train[rtm_features], train["RTM_Price"], sample_weight=sample_weights)

rtm_model_q = RTMModelQuantile()
rtm_model_q.fit(train[rtm_features], train["RTM_Price"], sample_weight=sample_weights)

print("Production RTM models (mean + quantile blend) trained.")

# [IMP-5] Block-level P10 floor for output clipping
rtm_block_floor = (
    train.groupby("Block")["RTM_Price"]
    .quantile(0.10)
    .to_dict()
)

# [IMP-6] Block-level P75 anchor for lag blending on day 3+
rtm_block_p75 = (
    train.groupby("Block")["RTM_Price"]
    .quantile(0.75)
    .to_dict()
)

print(f"Block floor (P10) computed for {len(rtm_block_floor)} blocks.")
print(f"Block P75 anchor computed for {len(rtm_block_p75)} blocks.")


# --------------------------------------------------------------------------
# ## SECTION 11 – Future market frame + recursive RTM forecast (v2)

# ==========================================================================
# [cell 24]
# ==========================================================================
def build_future_market_frame():
    month_start, month_end = month_bounds(PEAK_MONTH)
    times = pd.date_range(month_start, month_end, freq="15min")
    out   = pd.DataFrame({"Datetime": times})
    out["Date"]  = out["Datetime"].dt.normalize()
    out["Block"] = out["Datetime"].dt.hour * 4 + out["Datetime"].dt.minute // 15 + 1  # 1-based

    known_dam = dam[["Datetime", "DAM_Price"]].dropna()
    out = out.merge(known_dam, on="Datetime", how="left")
    recent_cutoff = known_dam["Datetime"].max() - pd.Timedelta(days=30)
    dam_profile = (
        add_datetime_from_date_block(dam).query("Datetime >= @recent_cutoff")
        .groupby("Block")["DAM_Price"].median()
    )
    out["DAM_Price"] = out["DAM_Price"].fillna(out["Block"].map(dam_profile))
    out["DAM_Price"] = out["DAM_Price"].ffill().bfill()

    known_dam_factors = dam[["Datetime", "Block"] + MARKET_FACTOR_COLS].copy()
    known_dam_factors = known_dam_factors.dropna(subset=MARKET_FACTOR_COLS, how="all")
    if not known_dam_factors.empty:
        recent_dam_cutoff = known_dam_factors["Datetime"].max() - pd.Timedelta(days=30)
        # [IMP-2] Use P90 instead of median for forward market factor profiles
        dam_factor_profile = (
            known_dam_factors.query("Datetime >= @recent_dam_cutoff")
            .groupby("Block")[MARKET_FACTOR_COLS].quantile(0.90)
        )
        for col in MARKET_FACTOR_COLS:
            out[f"dam_{col}"] = out["Block"].map(dam_factor_profile[col])
            out[f"dam_{col}"] = out[f"dam_{col}"].fillna(known_dam_factors[col].quantile(0.90))
            out[f"dam_{col}"] = out[f"dam_{col}"].ffill().bfill()
    else:
        for col in MARKET_FACTOR_COLS:
            out[f"dam_{col}"] = np.nan

    known_rtm_factors = rtm[["Datetime", "Block"] + RTM_MARKET_FACTOR_COLS].copy()
    known_rtm_factors = known_rtm_factors.dropna(subset=RTM_MARKET_FACTOR_COLS, how="all")
    if not known_rtm_factors.empty:
        recent_rtm_cutoff = known_rtm_factors["Datetime"].max() - pd.Timedelta(days=30)
        # [IMP-2] Use P90 for RTM factor profiles as well
        rtm_factor_profile = (
            known_rtm_factors.query("Datetime >= @recent_rtm_cutoff")
            .groupby("Block")[RTM_MARKET_FACTOR_COLS].quantile(0.90)
        )
        for col in RTM_MARKET_FACTOR_COLS:
            out[col] = out["Block"].map(rtm_factor_profile[col])
            out[col] = out[col].fillna(known_rtm_factors[col].quantile(0.90))
            out[col] = out[col].ffill().bfill()
    else:
        for col in RTM_MARKET_FACTOR_COLS:
            out[col] = np.nan

    month_net = net_forecast[["Datetime", "Forecast_Net_Load"]].rename(
        columns={"Forecast_Net_Load": "Net_Load"}
    )
    hist_net = df[["Net_Load"]].reset_index()
    out = out.merge(hist_net, on="Datetime", how="left", suffixes=("", "_hist"))
    out = out.merge(month_net, on="Datetime", how="left", suffixes=("", "_forecast"))
    out["Net_Load"] = out["Net_Load"].fillna(out.get("Net_Load_forecast"))
    out = out.drop(columns=[c for c in ["Net_Load_forecast"] if c in out.columns])
    out["Net_Load"] = out["Net_Load"].interpolate().ffill().bfill()
    return out


future_market = build_future_market_frame()

history = market[
    ["Datetime", "Date", "Block", "DAM_Price", "Net_Load", "RTM_Price"]
    + RTM_MARKET_FACTOR_COLS + DAM_MARKET_FACTOR_COLS
].copy()

rtm_forecasts = []
for _, base_row in future_market.iterrows():
    current_row = {
        "Datetime":  base_row["Datetime"],
        "Date":      base_row["Date"],
        "Block":     base_row["Block"],
        "DAM_Price": base_row["DAM_Price"],
        "Net_Load":  base_row["Net_Load"],
        "RTM_Price": np.nan,
    }
    for col in RTM_MARKET_FACTOR_COLS:
        current_row[col] = base_row[col]
    for col in DAM_MARKET_FACTOR_COLS:
        current_row[col] = base_row[col]

    current = pd.DataFrame([current_row])
    temp = pd.concat([history.tail(HISTORY_TAIL), current], ignore_index=True)
    temp_feat, _ = add_rtm_features(temp)
    row_features = temp_feat.tail(1)[rtm_features]

    # [IMP-1] Blended prediction: mean + quantile
    pred_mean = float(rtm_model_mean.predict(row_features)[0])
    pred_q    = float(rtm_model_q.predict(row_features)[0])
    pred_raw  = RTM_MEAN_WEIGHT * pred_mean + RTM_QUANTILE_WEIGHT * pred_q

    # [IMP-5] Apply block-level P10 floor and global 10,000 ceiling
    block_floor = rtm_block_floor.get(int(base_row["Block"]), 0.0)
    pred = float(np.clip(pred_raw, block_floor, 10_000))

    # [IMP-6] Lag blending: for day 3+ blend with historical P75 to prevent
    #         compounding under-estimation from recursive forecast lags
    day_offset = (base_row["Datetime"] - month_start).days
    if day_offset >= 3:
        alpha_blend = min(0.45, 0.08 * (day_offset - 2))  # ramp 0 → 0.45 over ~8 days
        hist_anchor = rtm_block_p75.get(int(base_row["Block"]), pred)
        pred = (1 - alpha_blend) * pred + alpha_blend * hist_anchor

    current["RTM_Price"] = pred
    history = pd.concat([history, current], ignore_index=True)
    rtm_forecasts.append(pred)

future_market["Forecast_RTM_Price"] = rtm_forecasts
future_market.to_csv(ARTIFACT_DIR / f"RTM_Forecast_{PEAK_MONTH}.csv", index=False)
print(future_market[["Datetime", "Forecast_RTM_Price"]].head())
print(f"Forecast RTM — mean: {np.mean(rtm_forecasts):.0f}  P90: {np.percentile(rtm_forecasts, 90):.0f}  max: {np.max(rtm_forecasts):.0f}")


# --------------------------------------------------------------------------
# ## SECTION 12 – Peak-hour scoring & selection

# ==========================================================================
# [cell 26]
# ==========================================================================
PEAK_START_BLOCK         = 1
PEAK_END_BLOCK           = 96
PEAK_NET_LOAD_PERCENTILE = 0.90
PEAK_WEIGHT_NL_RTM       = 0.35
PEAK_WEIGHT_NET_LOAD     = 0.10
PEAK_WEIGHT_RTM          = 0.45
PEAK_WEIGHT_DAM_MARKET   = 0.10

assert abs(PEAK_WEIGHT_NL_RTM + PEAK_WEIGHT_NET_LOAD + PEAK_WEIGHT_RTM + PEAK_WEIGHT_DAM_MARKET - 1.0) < 1e-9


def _build_candidate_windows(total_blocks=12, min_split_blocks=4):
    valid_blocks = range(PEAK_START_BLOCK, PEAK_END_BLOCK + 1)
    candidates   = []

    for start_block in valid_blocks:
        end_block = start_block + total_blocks - 1
        if end_block > PEAK_END_BLOCK:
            break
        blocks = tuple(range(start_block, start_block + total_blocks))
        candidates.append(("continuous", [(start_block, total_blocks)], blocks))

    for len1 in range(min_split_blocks, total_blocks - min_split_blocks + 1):
        len2 = total_blocks - len1
        for s1 in valid_blocks:
            e1 = s1 + len1 - 1
            if e1 > PEAK_END_BLOCK:
                break
            for s2 in range(e1 + 2, PEAK_END_BLOCK + 1):
                e2 = s2 + len2 - 1
                if e2 > PEAK_END_BLOCK:
                    break
                blocks = tuple(list(range(s1, s1 + len1)) + list(range(s2, s2 + len2)))
                candidates.append(("split", [(s1, len1), (s2, len2)], blocks))

    if not candidates:
        raise ValueError("No valid candidate windows found for the full day.")
    return candidates


_CANDIDATE_WINDOWS = _build_candidate_windows()
print(f"Candidate windows cached: {len(_CANDIDATE_WINDOWS)}")


def candidate_windows():
    return _CANDIDATE_WINDOWS


def ranges_from_windows(windows):
    ranges = []
    for start_block, length in windows:
        start      = block_to_time(start_block)
        end_block  = start_block + length
        end_minutes = (end_block - 1) * 15
        end = f"{(end_minutes // 60) % 24:02d}:{end_minutes % 60:02d}"
        ranges.append(f"{start}-{end}")
    return ranges


def historical_seasonal_block_score(target_month):
    hist = market.copy()
    hist = add_all_market_pressure_features(hist)
    hist["Datetime"] = pd.to_datetime(hist["Datetime"])
    target_month_no  = pd.Period(target_month, freq="M").month
    seasonal = hist[hist["Datetime"].dt.month == target_month_no].copy()
    if seasonal.empty:
        seasonal = hist.copy()

    nl_threshold = seasonal["Net_Load"].quantile(PEAK_NET_LOAD_PERCENTILE)
    seasonal["Net_Load_Excess"]      = (seasonal["Net_Load"] - nl_threshold).clip(lower=0)
    seasonal["Net_Ramp"]             = seasonal["Net_Load"].diff()
    seasonal["Net_Ramp_TopDecile"]   = (
        seasonal["Net_Ramp"].where(seasonal["Net_Load_Excess"] > 0, 0).clip(lower=0)
    )
    seasonal["RTM_Ramp"]             = seasonal["RTM_Price"].diff()
    seasonal["NL_RTM_Interaction"]   = seasonal["Net_Load"] * seasonal["RTM_Price"]
    seasonal["Historical_Block_Score"] = (
        0.20 * robust_z(seasonal["NL_RTM_Interaction"])
        + 0.15 * robust_z(seasonal["Net_Load_Excess"])
        + 0.05 * robust_z(seasonal["Net_Ramp_TopDecile"])
        + 0.30 * robust_z(seasonal["RTM_Price"])
        + 0.12 * robust_z(seasonal["RTM_Ramp"].clip(lower=0))
        + 0.06 * robust_z(seasonal["Purchase_Bid_MW"])
        + 0.04 * robust_z(seasonal["dam_Purchase_Bid_MW"])
        + 0.04 * robust_z(seasonal["dam_Sell_Bid_MW"])
        + 0.04 * robust_z(seasonal["dam_MCP_Rs_kWh"])
    )
    score = seasonal.groupby("Block")["Historical_Block_Score"].mean().reindex(range(1, 97))
    return score.fillna(score.median()).fillna(0)


def _normalize_block_score(score):
    score = pd.Series(score, index=range(1, 97), dtype=float).replace([np.inf, -np.inf], np.nan)
    score = score.fillna(score.median()).fillna(0)
    return robust_z(score)


def build_block_score_methods(scored_month):
    scored = scored_month.copy()
    scored["DateOnly"] = pd.to_datetime(scored["Datetime"]).dt.date

    grouped    = scored.groupby("Block")
    mean_score   = grouped["Peak_Score"].mean().reindex(range(1, 97))
    median_score = grouped["Peak_Score"].median().reindex(range(1, 97))
    p75_score    = grouped["Peak_Score"].quantile(0.75).reindex(range(1, 97))
    p90_score    = grouped["Peak_Score"].quantile(0.90).reindex(range(1, 97))

    daily_top = scored.copy()
    daily_top["Daily_Rank"] = daily_top.groupby("DateOnly")["Peak_Score"].rank(
        method="first", ascending=False
    )
    top_frequency = (
        daily_top[daily_top["Daily_Rank"] <= 12]
        .groupby("Block").size()
        .reindex(range(1, 97), fill_value=0)
        / max(scored["DateOnly"].nunique(), 1)
    )

    rtm_pressure = (
        0.40 * _normalize_block_score(grouped["RTM_Peak_Score"].mean().reindex(range(1, 97)))
        + 0.30 * _normalize_block_score(grouped["DAM_Market_Pressure_Score"].mean().reindex(range(1, 97)))
        + 0.20 * _normalize_block_score(grouped["Purchase_Volume_Score"].mean().reindex(range(1, 97)))
        + 0.10 * _normalize_block_score(grouped["MCP_Pressure_Score"].mean().reindex(range(1, 97)))
    )
    seasonal_score = grouped["Seasonal_Block_Score"].mean().reindex(range(1, 97))

    methods = {
        "mean":               _normalize_block_score(mean_score),
        "median":             _normalize_block_score(median_score),
        "p75":                _normalize_block_score(p75_score),
        "p90":                _normalize_block_score(p90_score),
        "daily_top_frequency": _normalize_block_score(top_frequency),
        "rtm_pressure":       _normalize_block_score(rtm_pressure),
    }
    methods["robust_blend"] = (
        0.22 * methods["mean"]
        + 0.13 * methods["median"]
        + 0.18 * methods["p75"]
        + 0.17 * methods["p90"]
        + 0.15 * methods["daily_top_frequency"]
        + 0.10 * methods["rtm_pressure"]
        + 0.05 * _normalize_block_score(seasonal_score)
    )
    methods["robust_blend"] = _normalize_block_score(methods["robust_blend"])
    return methods


def select_window_from_block_score(score_by_block):
    best = None
    for pattern, windows, blocks in _CANDIDATE_WINDOWS:
        score = float(score_by_block.loc[list(blocks)].sum())
        if best is None or score > best["Score"]:
            best = {
                "Pattern": "3hr continuous" if pattern == "continuous" else "split",
                "Windows": windows,
                "Blocks":  blocks,
                "Score":   score,
            }
    best["Peak_Hours"]      = ", ".join(ranges_from_windows(best["Windows"]))
    best["Selected_Blocks"] = ", ".join(map(str, best["Blocks"]))
    return best


def select_monthly_peak_hours(scored_month, selection_method="robust_blend"):
    score_methods = build_block_score_methods(scored_month)
    if selection_method not in score_methods:
        raise ValueError(f"Unknown selection method: {selection_method}")

    comparisons = []
    for method_name, method_score in score_methods.items():
        method_best = select_window_from_block_score(method_score)
        comparisons.append({
            "Method":          method_name,
            "Peak_Hours":      method_best["Peak_Hours"],
            "Pattern":         method_best["Pattern"],
            "Score":           round(method_best["Score"], 4),
            "Selected_Blocks": method_best["Selected_Blocks"],
        })

    best = select_window_from_block_score(score_methods[selection_method])
    best["Selection_Method"] = selection_method
    comparison_df = pd.DataFrame(comparisons).sort_values("Method").reset_index(drop=True)
    return best, score_methods[selection_method], comparison_df


# ── Score the forecast month ──────────────────────────────────────────────────
month_start, month_end = month_bounds(PEAK_MONTH)
peak_base = future_market[
    (future_market["Datetime"] >= month_start) & (future_market["Datetime"] <= month_end)
].copy()
peak_base = add_all_market_pressure_features(peak_base)
peak_base["Net_Ramp"]              = peak_base["Net_Load"].diff()
peak_base["RTM_Ramp"]              = peak_base["Forecast_RTM_Price"].diff()
peak_base["Bid_Spread_Ramp"]       = peak_base["Bid_Spread_MW"].diff()
peak_base["Purchase_Bid_Ramp"]     = peak_base["Purchase_Bid_MW"].diff()
peak_base["Sell_Bid_Ramp"]         = peak_base["Sell_Bid_MW"].diff()
peak_base["MCV_Ramp"]              = peak_base["MCV_MW"].diff()
peak_base["Scheduled_Volume_Ramp"] = peak_base["Final_Scheduled_Volume_MW"].diff()
peak_base["MCP_Ramp"]              = peak_base["MCP_Rs_kWh"].diff()
peak_base["dam_Purchase_Bid_Ramp"] = peak_base["dam_Purchase_Bid_MW"].diff()
peak_base["dam_Sell_Bid_Ramp"]     = peak_base["dam_Sell_Bid_MW"].diff()
peak_base["dam_MCP_Ramp"]          = peak_base["dam_MCP_Rs_kWh"].diff()
peak_base["Seasonal_Block_Score"]  = peak_base["Block"].map(historical_seasonal_block_score(PEAK_MONTH))

net_load_threshold              = peak_base["Net_Load"].quantile(PEAK_NET_LOAD_PERCENTILE)
peak_base["Net_Load_Excess"]    = (peak_base["Net_Load"] - net_load_threshold).clip(lower=0)
peak_base["Net_Ramp_TopDecile"] = (
    peak_base["Net_Ramp"].where(peak_base["Net_Load_Excess"] > 0, 0).clip(lower=0)
)
peak_base["Net_Load_Peak_Score"] = (
    0.73 * robust_z(peak_base["Net_Load_Excess"])
    + 0.27 * robust_z(peak_base["Net_Ramp_TopDecile"])
)
peak_base["RTM_Peak_Score"] = (
    0.72 * robust_z(peak_base["Forecast_RTM_Price"])
    + 0.28 * robust_z(peak_base["RTM_Ramp"].clip(lower=0))
)
peak_base["NL_RTM_Interaction"] = peak_base["Net_Load"] * peak_base["Forecast_RTM_Price"]

peak_base["DAM_Purchase_Score"] = (
    0.75 * robust_z(peak_base["dam_Purchase_Bid_MW"])
    + 0.25 * robust_z(peak_base["dam_Purchase_Bid_Ramp"].clip(lower=0))
)
peak_base["DAM_Sell_Score"] = (
    0.65 * robust_z(peak_base["dam_Sell_Bid_MW"])
    + 0.35 * robust_z(peak_base["dam_Sell_Bid_Ramp"].clip(lower=0))
)
peak_base["DAM_MCP_Score"] = (
    0.70 * robust_z(peak_base["dam_MCP_Rs_kWh"])
    + 0.30 * robust_z(peak_base["dam_MCP_Ramp"].clip(lower=0))
)
peak_base["DAM_Market_Pressure_Score"] = (
    0.35 * robust_z(peak_base["DAM_Purchase_Score"])
    + 0.25 * robust_z(peak_base["DAM_Sell_Score"])
    + 0.25 * robust_z(peak_base["dam_MCV_MW"])
    + 0.15 * robust_z(peak_base["DAM_MCP_Score"])
)
peak_base["Purchase_Volume_Score"] = (
    0.75 * robust_z(peak_base["Purchase_Bid_MW"])
    + 0.25 * robust_z(peak_base["Purchase_Bid_Ramp"].clip(lower=0))
)
peak_base["Sell_Bid_Score"] = (
    0.65 * robust_z(peak_base["Sell_Bid_MW"])
    + 0.35 * robust_z(peak_base["Sell_Bid_Ramp"].clip(lower=0))
)
peak_base["Cleared_Volume_Score"] = (
    0.45 * robust_z(peak_base["MCV_MW"])
    + 0.35 * robust_z(peak_base["Final_Scheduled_Volume_MW"])
    + 0.20 * robust_z(peak_base["Scheduled_Volume_Ramp"].clip(lower=0))
)
peak_base["MCP_Pressure_Score"] = (
    0.70 * robust_z(peak_base["MCP_Rs_kWh"])
    + 0.30 * robust_z(peak_base["MCP_Ramp"].clip(lower=0))
)
peak_base["Market_Tightness_Score"] = (
    0.30 * robust_z(peak_base["Bid_Spread_MW"])
    + 0.25 * robust_z(peak_base["Bid_Ratio"])
    + 0.20 * robust_z(peak_base["Cleared_Share_of_Sell"])
    + 0.15 * robust_z(peak_base["Schedule_MCV_Gap_MW"].abs())
    + 0.10 * robust_z(peak_base["MCP_Pressure_Score"])
)
peak_base["Peak_Score"] = (
    PEAK_WEIGHT_NL_RTM     * robust_z(peak_base["NL_RTM_Interaction"])
    + PEAK_WEIGHT_NET_LOAD * peak_base["Net_Load_Peak_Score"]
    + PEAK_WEIGHT_RTM      * peak_base["RTM_Peak_Score"]
    + PEAK_WEIGHT_DAM_MARKET * robust_z(peak_base["DAM_Market_Pressure_Score"])
)

PEAK_SELECTION_METHOD = "robust_blend"
best, score_by_block, peak_method_comparison = select_monthly_peak_hours(
    peak_base, selection_method=PEAK_SELECTION_METHOD
)

print(
    f"Peak-hour scoring: NL×RTM interaction ({PEAK_WEIGHT_NL_RTM:.0%}), "
    f"net-load top-{int((1 - PEAK_NET_LOAD_PERCENTILE) * 100)}% excess + top-10% ramp "
    f"({PEAK_WEIGHT_NET_LOAD:.0%}), RTM price ({PEAK_WEIGHT_RTM:.0%}), "
    f"DAM market ({PEAK_WEIGHT_DAM_MARKET:.0%}). No time-of-day prior."
)
print(f"  Net-load 90th-percentile threshold: {net_load_threshold:,.0f} MW")
print(f"  Selection method: {best['Selection_Method']}")
print(f"  Peak Hours: {best['Peak_Hours']}")
print(f"  Pattern   : {best['Pattern']}")
try:
    display(peak_method_comparison)
except NameError:
    print(peak_method_comparison.to_string(index=False))

monthly_peak = pd.DataFrame([{
    "Month":            PEAK_MONTH,
    "Peak_Hours":       best["Peak_Hours"],
    "Pattern":          best["Pattern"],
    "Total_Hours":      3.0,
    "Score":            round(best["Score"], 4),
    "Selected_Blocks":  best["Selected_Blocks"],
    "Selection_Method": best["Selection_Method"],
    "Selection_Description": (
        f"NL×RTM interaction ({PEAK_WEIGHT_NL_RTM:.0%}), "
        f"net-load top-{int((1 - PEAK_NET_LOAD_PERCENTILE) * 100)}% excess + top-10% ramp "
        f"({PEAK_WEIGHT_NET_LOAD:.0%}), RTM price ({PEAK_WEIGHT_RTM:.0%}), "
        f"DAM bid/volume/MCP ({PEAK_WEIGHT_DAM_MARKET:.0%}); "
        "no time-of-day prior; robust blend aggregation across the month; "
        f"v2: quantile blend ({int(RTM_MEAN_WEIGHT*100)}% mean + {int(RTM_QUANTILE_WEIGHT*100)}% Q{int(RTM_QUANTILE_ALPHA*100)}), "
        "P90 market inputs, night-peak features, sample weights, block-floor clipping, lag blending"
    ),
}])

monthly_peak.to_excel(ARTIFACT_DIR / f"Monthly_Peak_Hours_{PEAK_MONTH}.xlsx", index=False)
monthly_peak.to_csv(ARTIFACT_DIR / f"Monthly_Peak_Hours_{PEAK_MONTH}.csv", index=False)
peak_method_comparison.to_excel(ARTIFACT_DIR / f"Peak_Method_Comparison_{PEAK_MONTH}.xlsx", index=False)
peak_method_comparison.to_csv(ARTIFACT_DIR / f"Peak_Method_Comparison_{PEAK_MONTH}.csv", index=False)
print(monthly_peak)


# --------------------------------------------------------------------------
# ## SECTION 13 – Daily audit table

# ==========================================================================
# [cell 28]
# ==========================================================================
daily_records   = []
selected_blocks = set(best["Blocks"])
for day, g in peak_base.groupby(peak_base["Datetime"].dt.date):
    selected = g[g["Block"].isin(selected_blocks)]
    daily_records.append({
        "Date":                        day,
        "Peak_Hours":                  best["Peak_Hours"],
        "Avg_Selected_Net_Load_MW":    round(selected["Net_Load"].mean(), 0),
        "Avg_Selected_RTM_Price":      round(selected["Forecast_RTM_Price"].mean(), 2),
        "Avg_Purchase_Bid_MW":         round(selected["Purchase_Bid_MW"].mean(), 2),
        "Avg_Sell_Bid_MW":             round(selected["Sell_Bid_MW"].mean(), 2),
        "Avg_MCV_MW":                  round(selected["MCV_MW"].mean(), 2),
        "Avg_Scheduled_Volume_MW":     round(selected["Final_Scheduled_Volume_MW"].mean(), 2),
        "Avg_Net_Load_Peak_Score":     round(selected["Net_Load_Peak_Score"].mean(), 3),
        "Avg_RTM_Peak_Score":          round(selected["RTM_Peak_Score"].mean(), 3),
        "Avg_NL_RTM_Interaction":      round(selected["NL_RTM_Interaction"].mean(), 0),
        "Avg_DAM_Market_Pressure_Score": round(selected["DAM_Market_Pressure_Score"].mean(), 3),
        "Avg_Purchase_Volume_Score":   round(selected["Purchase_Volume_Score"].mean(), 3),
        "Avg_Sell_Bid_Score":          round(selected["Sell_Bid_Score"].mean(), 3),
        "Avg_Cleared_Volume_Score":    round(selected["Cleared_Volume_Score"].mean(), 3),
        "Avg_MCP_Pressure_Score":      round(selected["MCP_Pressure_Score"].mean(), 3),
        "Avg_Market_Tightness_Score":  round(selected["Market_Tightness_Score"].mean(), 3),
        "Avg_Seasonal_Block_Score":    round(selected["Seasonal_Block_Score"].mean(), 3),
        "Max_Net_Ramp_MW_per_15min":   round(selected["Net_Ramp"].max(), 0),
        "Max_RTM_Ramp_per_15min":      round(selected["RTM_Ramp"].max(), 2),
        "Mean_Peak_Score":             round(selected["Peak_Score"].mean(), 3),
    })

daily_peak_audit = pd.DataFrame(daily_records)
daily_peak_audit.to_excel(
    ARTIFACT_DIR / f"Daily_Audit_For_Monthly_Peak_Hours_{PEAK_MONTH}.xlsx", index=False
)
print(daily_peak_audit.head())


# --------------------------------------------------------------------------
# ## SECTION 14 – Monthly overview chart

# ==========================================================================
# [cell 30]
# ==========================================================================
fig, axes = plt.subplots(
    4, 1, figsize=(17, 15),
    gridspec_kw={"height_ratios": [1.2, 1.1, 1.35, 1.1]},
)
plot_df = peak_base.copy()

axes[0].plot(plot_df["Datetime"], plot_df["Net_Load"], color="steelblue", lw=1)
axes[0].set_title(f"Forecast Net Load – {PEAK_MONTH}")
axes[0].set_ylabel("MW")
axes[0].grid(alpha=0.3)

axes[1].plot(plot_df["Datetime"], plot_df["Forecast_RTM_Price"], color="darkorange", lw=1)
axes[1].set_title("Forecast RTM Price — v2 (Quantile Blend + P90 Inputs + Night Regime Features)")
axes[1].set_ylabel("Rs/MWh")
axes[1].grid(alpha=0.3)

for ax in axes[:2]:
    for day, _ in plot_df.groupby(plot_df["Datetime"].dt.normalize()):
        for start_block, length in best["Windows"]:
            start_ts = day + pd.Timedelta(minutes=(start_block - 1) * 15)
            end_ts   = start_ts + pd.Timedelta(minutes=length * 15)
            ax.axvspan(start_ts, end_ts, color="gold", alpha=0.12)

block_profile = (
    plot_df.groupby("Block")[[
        "Purchase_Bid_MW", "Sell_Bid_MW", "MCV_MW",
        "Final_Scheduled_Volume_MW", "MCP_Rs_kWh",
    ]].mean().reindex(range(1, 97))
)
block_hours = (block_profile.index - 1) / 4

axes[2].fill_between(block_hours, block_profile["Purchase_Bid_MW"].fillna(0).to_numpy(),
                     color="#4f46a3", alpha=0.55, label="Purchase Bid (MW)")
axes[2].fill_between(block_hours, block_profile["Sell_Bid_MW"].fillna(0).to_numpy(),
                     color="gold", alpha=0.55, label="Sell Bid (MW)")
axes[2].plot(block_hours, block_profile["MCV_MW"], color="tomato", lw=1.4, label="MCV (MW)")
axes[2].plot(block_hours, block_profile["Final_Scheduled_Volume_MW"],
             color="limegreen", lw=1.6, label="Scheduled Volume (MW)")
axes2_mcp = axes[2].twinx()
axes2_mcp.plot(block_hours, block_profile["MCP_Rs_kWh"], color="black", lw=1.2, label="MCP (Rs/kWh)")
axes[2].set_title("Average RTM Market Snapshot by 15-Minute Block (P90 profiles in v2)")
axes[2].set_ylabel("MW")
axes2_mcp.set_ylabel("MCP (Rs/kWh)")
axes[2].set_xlim(0, 24)
axes[2].set_xticks(range(0, 25, 1))
axes[2].grid(alpha=0.3)

axes[3].plot(score_by_block.index, score_by_block.values, color="seagreen", marker="o", ms=3)
for b in best["Blocks"]:
    axes[2].axvspan((b - 1) / 4, b / 4, color="gold", alpha=0.18)
    axes[3].axvspan(b - 0.5, b + 0.5, color="gold", alpha=0.35)

market_lines, market_labels = axes[2].get_legend_handles_labels()
mcp_lines,    mcp_labels    = axes2_mcp.get_legend_handles_labels()
axes[2].legend(
    market_lines + mcp_lines, market_labels + mcp_labels,
    loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=5, fontsize=9,
)
axes[3].set_title(
    f"Monthly Peak Score by Block ({best['Selection_Method']}) | "
    f"Selected: {best['Peak_Hours']} | Full-day selection"
)
axes[3].set_xlabel("Block (1=00:00, 49=12:00, 75=18:30, 96=23:45)")
axes[3].set_ylabel("Score")
axes[3].grid(alpha=0.3)

plt.tight_layout()
plot_path = ARTIFACT_DIR / f"Monthly_Peak_Hours_{PEAK_MONTH}.png"
plt.savefig(plot_path, dpi=160, bbox_inches="tight")
plt.show()
print(plot_path)


# --------------------------------------------------------------------------
# ## SECTION 15 – Single-day diagnostic chart

# ==========================================================================
# [cell 32]
# ==========================================================================
PLOT_DATE = pd.Timestamp(PEAK_MONTH + "-01")

day_data = peak_base[peak_base["Datetime"].dt.normalize() == PLOT_DATE].copy()

if day_data.empty:
    print(f"No data found for {PLOT_DATE.date()} in peak_base. "
          f"Choose a date within {PEAK_MONTH}.")
else:
    blocks    = day_data["Block"].values
    times_lbl = [block_to_time(b) for b in blocks]
    net_load  = day_data["Net_Load"].values
    rtm_price = day_data["Forecast_RTM_Price"].values
    nl_rtm    = day_data["NL_RTM_Interaction"].values
    is_peak   = [b in selected_blocks for b in blocks]

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(
        f"Net Load · RTM Price · NL×RTM Interaction — {PLOT_DATE.date()}\n"
        f"Peak month: {PEAK_MONTH}  |  Selected window: {best['Peak_Hours']}",
        fontsize=12, fontweight="bold",
    )

    def shade_peak(ax):
        for i, (b, pk) in enumerate(zip(blocks, is_peak)):
            if pk:
                ax.axvspan(i - 0.5, i + 0.5, color="red", alpha=0.15, zorder=0)

    x          = range(len(blocks))
    tick_every = 8
    tick_pos   = list(range(0, len(blocks), tick_every))
    tick_labs  = [times_lbl[i] for i in tick_pos]

    axes[0].plot(x, net_load, color="steelblue", linewidth=1.8, label="Net Load (MW)")
    axes[0].axhline(net_load_threshold, color="steelblue", linestyle="--", linewidth=1,
                    alpha=0.7, label=f"90th-pct threshold ({net_load_threshold:,.0f} MW)")
    shade_peak(axes[0])
    axes[0].set_ylabel("Net Load (MW)")
    axes[0].grid(alpha=0.3)
    axes[0].set_title("Net Load")
    axes[0].legend(
        handles=[
            plt.Line2D([0], [0], color="steelblue", linewidth=1.8, label="Net Load (MW)"),
            plt.Line2D([0], [0], color="steelblue", linestyle="--", linewidth=1,
                       label=f"90th-pct threshold ({net_load_threshold:,.0f} MW)"),
            Patch(color="red", alpha=0.3, label="Selected peak blocks"),
        ],
        fontsize=8, loc="upper left",
    )

    axes[1].plot(x, rtm_price, color="darkorange", linewidth=1.8,
                 label="Forecast RTM Price (Rs/MWh)")
    shade_peak(axes[1])
    axes[1].set_ylabel("RTM Price (Rs/MWh)")
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].grid(alpha=0.3)
    axes[1].set_title("Forecast RTM Price — v2")

    axes[2].fill_between(x, nl_rtm, color="mediumpurple", alpha=0.55,
                         label="Net Load × RTM Price")
    axes[2].plot(x, nl_rtm, color="mediumpurple", linewidth=1.2)
    shade_peak(axes[2])
    axes[2].set_ylabel("NL × RTM (MW · Rs/MWh)")
    axes[2].legend(fontsize=8, loc="upper left")
    axes[2].grid(alpha=0.3)
    axes[2].set_title("Net Load × RTM Price (interaction signal)")
    axes[2].set_xticks(tick_pos)
    axes[2].set_xticklabels(tick_labs, rotation=45, ha="right", fontsize=8)
    axes[2].set_xlabel("Time (HH:MM)")

    plt.tight_layout()
    diag_path = ARTIFACT_DIR / f"NL_RTM_Diagnostic_{PLOT_DATE.date()}.png"
    plt.savefig(diag_path, dpi=160, bbox_inches="tight")
    plt.show()
    print(f"Saved: {diag_path}")

    summary = day_data[[
        "Block", "Net_Load", "Forecast_RTM_Price", "NL_RTM_Interaction", "Peak_Score"
    ]].copy()
    summary["Is_Peak_Block"] = summary["Block"].isin(selected_blocks)
    summary["Time"]          = summary["Block"].apply(block_to_time)
    summary = summary[[
        "Time", "Block", "Net_Load", "Forecast_RTM_Price",
        "NL_RTM_Interaction", "Peak_Score", "Is_Peak_Block",
    ]].reset_index(drop=True)
    summary.columns = [
        "Time", "Block", "Net_Load_MW", "RTM_Price_Rs_MWh",
        "NL_x_RTM", "Peak_Score", "Is_Peak",
    ]
    try:
        display(summary.style.format({
            "Net_Load_MW":      "{:,.0f}",
            "RTM_Price_Rs_MWh": "{:,.2f}",
            "NL_x_RTM":         "{:,.0f}",
            "Peak_Score":       "{:.4f}",
        }).apply(
            lambda row: ["background-color: #ffe0e0" if row["Is_Peak"] else "" for _ in row],
            axis=1,
        ))
    except Exception:
        print(summary.to_string(index=False))


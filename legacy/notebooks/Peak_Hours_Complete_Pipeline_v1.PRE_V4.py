"""Reference dump of Peak_Hours_Complete_Pipeline_v1.PRE_V4.ipynb -- code cells only, outputs stripped.

Frozen copy for reading while porting; not meant to be executed as-is.
"""

# --------------------------------------------------------------------------
# # Peak Hours Complete Pipeline — v3
# ### Net-load forecasting (v2, unchanged) + RTM forecasting replaced with the
# **Hurdle-Quantile Ensemble with Climatology Decomposition** (see Section 9 below).

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

from iex import get_trade_data, get_market_volumes

# ── Configuration ─────────────────────────────────────────────────────────────
START_DATE = "2022-07-03"
PEAK_MONTH = "2026-09"       # month to declare peak hours for, YYYY-MM
END_DATE = "2026-08-22"  # last day of month before PEAK_MONTH
print(f"Auto-derived END_DATE (last day of month before PEAK_MONTH): {END_DATE}")

FORECAST_DAYS         = 35
TEST_DAYS             = 92
NL_MIN                = 10_000
NL_MAX                = 100_000
WEATHER_FORECAST_DAYS = 16
REFRESH_MARKET_DATA   = False
HISTORY_TAIL          = 1_344

# ── Net-load source ──────────────────────────────────────────────────────────
# "auto"  : SCADA parquet cache (scada-cache/), falling back to NR_DEMAND_TEMP
# "scada" : SCADA cache only - raise if it cannot cover the window
# "raw"   : the original NR_DEMAND_TEMP folder scan
NET_LOAD_SOURCE     = "auto"
UPDATE_SCADA_CACHE  = True   # pull new days off the share first; set False to
                             # use the cache exactly as it stands (much faster
                             # when you are only re-running the model)

# [IMP-1] Quantile blend ratio: 55% mean + 45% quantile (OFF-PEAK blocks only)
RTM_MEAN_WEIGHT     = 0.55
RTM_QUANTILE_WEIGHT = 0.45
RTM_QUANTILE_ALPHA  = 0.95   # target the 95th percentile (was 0.85) — needed to
                              # actually reach the 10,000 cap during spikes

# [SPIKE-FIX] Night (Block 1-20) and evening-ramp (Block 73-88) blocks are
# where RTM historically slams into the price cap. A flat 55/45 blend was
# smoothing those spikes away, so peak-regime blocks get a much heavier
# quantile weighting instead.
RTM_PEAK_MEAN_WEIGHT     = 0.25
RTM_PEAK_QUANTILE_WEIGHT = 0.75

# [SPIKE-FIX] Sample-weight clip for training — raised from 5.0 so near-cap
# historical rows get more influence on the fitted trees.
RTM_SAMPLE_WEIGHT_CAP = 8.0

# ── City weather stations (lat/lon) — same 40 stations used in the v5 model ──
# Flattened across all NR states (Punjab, Haryana, Rajasthan, Delhi, UP,
# Uttarakhand, HP, J&K Ladakh, Chd). Coordinates taken from v5's WEATHER_STATIONS.
CITY_COORDS = {
    "Ludhiana": (30.90, 75.86), "Amritsar": (31.63, 74.87), "Patiala": (30.33, 76.40),
    "Jalandhar": (31.33, 75.58), "Bathinda": (30.21, 74.94),
    "Hisar": (29.15, 75.72), "Gurugram": (28.46, 77.03), "Faridabad": (28.41, 77.31),
    "Ambala": (30.38, 76.78), "Rohtak": (28.90, 76.58), "Panipat": (29.39, 76.97),
    "Jaipur": (26.82, 75.80), "Jodhpur": (26.24, 73.02), "Kota": (25.17, 75.85),
    "Udaipur": (24.58, 73.71), "Bikaner": (28.01, 73.31), "Ajmer": (26.45, 74.64),
    "Sriganganagar": (29.92, 73.88),
    "Palam": (28.57, 77.10), "Safdarjung": (28.59, 77.21), "LodiBhawan": (28.59, 77.22),
    "Lucknow": (26.85, 80.95), "Agra": (27.18, 78.01), "Kanpur": (26.47, 80.33),
    "Varanasi": (25.32, 82.97), "Prayagraj": (25.45, 81.84), "Noida": (28.54, 77.39),
    "Gorakhpur": (26.75, 83.37), "Meerut": (28.98, 77.72), "Bareilly": (28.36, 79.41),
    "Dehradun": (30.32, 78.03), "Haridwar": (29.97, 78.17), "Roorkee": (29.87, 77.89),
    "Shimla": (31.10, 77.17), "Dharamsala": (32.22, 76.32), "Mandi": (31.71, 76.93),
    "Jammu": (32.74, 74.87), "Srinagar": (34.08, 74.80), "Leh": (34.15, 77.58),
    "Chandigarh": (30.74, 76.79),
}

# ── City weights for the composite Temp/Humidity, from v5 feature importance ──
# CITY_WEIGHTS_CSV should sit next to this notebook (or update the path below).
CITY_WEIGHTS_CSV = PROJECT_DIR / "City_Weather_Feature_Importance.csv"
_city_wt_df = pd.read_csv(CITY_WEIGHTS_CSV)
_city_wt_df = _city_wt_df[_city_wt_df["City"].isin(CITY_COORDS)]
_wt_sum = _city_wt_df["Total_Importance"].sum()
CITY_WEIGHTS = dict(zip(_city_wt_df["City"], _city_wt_df["Total_Importance"] / _wt_sum))

_missing_weight_cities = set(CITY_COORDS) - set(CITY_WEIGHTS)
if _missing_weight_cities:
    print("Cities with no weight in CSV (excluded from composite):", _missing_weight_cities)
print(f"Loaded normalized weights for {len(CITY_WEIGHTS)} cities (sum = "
      f"{sum(CITY_WEIGHTS.values()):.4f})")

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


def parse_date_series(s):
    """Parse a Date column that may be ISO (YYYY-MM-DD, how the cached
    DAM/RTM CSVs are written) or day-first (DD-MM-YYYY, how some iex_client
    payloads come back), deciding per value rather than with one global flag.

    [DAYFIRST-BUG] This previously passed dayfirst=True unconditionally. On an
    ISO column that silently turns every date whose day-part exceeds 12 into
    NaT -- 87,742 of 145,150 rows -- and the NaT rows were then dropped by the
    range filter downstream, leaving the market frame with only days 1-12 of
    every month. Everything price-derived (climatology profile, spike/crash
    thresholds, walk-forward CV, conformal calibration, seasonal block score)
    was therefore fitted on that biased subsample.
    """
    s = s.astype(str).str.strip()
    is_iso = s.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if is_iso.any():
        out.loc[is_iso] = pd.to_datetime(s[is_iso], errors="coerce")
    if (~is_iso).any():
        out.loc[~is_iso] = pd.to_datetime(s[~is_iso], dayfirst=True, errors="coerce")
    return out


def add_datetime_from_date_block(df, date_col="Date", block_col="Block"):
    out = df.copy()
    out[date_col]  = parse_date_series(out[date_col])
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
# ## SECTION 2 – Net-load: raw build, resample, clean

# ==========================================================================
# [cell 6]
# ==========================================================================
def _select_files_in_date_range(csv_files, start_ts, end_ts):
    """[DATE-FILTER-1] Only select files whose filename-date falls inside
    [start_ts, end_ts). A file present in the folder but outside that range
    is dropped here -- before it's ever opened/parsed -- instead of being
    read and then discarded later."""
    selected = []
    for file in csv_files:
        try:
            file_date = pd.to_datetime(file.stem, format="%d-%m-%Y")
        except Exception:
            print(f"  Skipping {file.name} (date parse failed)")
            continue
        if start_ts <= file_date < end_ts:
            selected.append((file, file_date))
        else:
            print(f"  Skipping {file.name} (date {file_date.date()} outside "
                  f"[{start_ts.date()}, {end_ts.date()}))")
    return selected

RAW_EXPECTED_COLS = {"HRS", "NR Load", "NR Solar", "NR Wind"}


def build_net_load_from_raw(input_folder=DEMAND_FOLDER, start_ts=None, end_ts=None):
    start_ts = START_TS if start_ts is None else start_ts
    end_ts   = END_TS if end_ts is None else end_ts

    all_csv_files = sorted(Path(input_folder).glob("*.csv"))
    if not all_csv_files:
        raise FileNotFoundError(f"No raw demand CSV files found in {input_folder}")

    dated_files = _select_files_in_date_range(all_csv_files, start_ts, end_ts)
    print(
        f"Selected {len(dated_files)}/{len(all_csv_files)} files in range "
        f"[{start_ts.date()}, {end_ts.date()})"
    )

    master_df = []
    for file, file_date in dated_files:
        print(f"Processing {file.name}")
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
                print(f"  Skipping {file.name}: missing columns {missing_cols}")
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

        except Exception as e:
            print(f"  Error in {file.name}: {e}")

    if not master_df:
        raise ValueError("No valid data extracted.")

    master_df = pd.concat(master_df, ignore_index=True).sort_values("Datetime").dropna()
    print(f"\n✔ Parsed {len(dated_files)} files — {len(master_df):,} rows total")
    return master_df


# ── Net load: SCADA parquet cache first, raw folder as fallback ─────────────
# The 1-minute NRLDC Stack exports on \\192.168.50.247\scadashare are parsed
# once into per-year Parquet by the scada-cache/ subproject, which defines
# Net_Load = NR_Load - NR_Solar - NR_Wind exactly as build_net_load_from_raw()
# above does. Reading Parquet takes seconds instead of re-parsing ~1500 CSVs.
def _net_load_from_scada():
    """15-min Net_Load from scada-cache, or None if it cannot cover the window."""
    scada_dir = PROJECT_DIR / "scada-cache"
    if not scada_dir.exists():
        print("scada-cache not found at", scada_dir)
        return None
    if str(scada_dir) not in sys.path:
        sys.path.insert(0, str(scada_dir))
    try:
        import scada_cache as sc
    except ImportError as exc:
        print("scada-cache not importable:", exc)
        return None

    if UPDATE_SCADA_CACHE:
        print("Updating SCADA cache (incremental; no-op if the share is offline) ...")
        try:
            info = sc.update_source("nr", verbose=False)
            if info.get("reachable"):
                print(f"  ingested {info.get('ingested', 0)}, "
                      f"skipped {info.get('skipped', 0)}")
            else:
                print("  share unreachable - using the cache as it stands")
        except Exception as exc:
            print("  update failed:", exc, "- using the cache as it stands")

    try:
        net = sc.load_net_load(start=START_TS, end=END_TS, freq="15min")
    except Exception as exc:
        print("scada-cache unusable:", exc)
        return None

    if net.empty or net.index.min() > START_TS or        net.index.max() < END_TS - pd.Timedelta(days=1):
        have = f"{net.index.min()} to {net.index.max()}" if len(net) else "empty"
        print(f"scada-cache covers {have}; need {START_TS} to {END_TS}")
        return None

    print(f"Net load from scada-cache: {len(net):,} rows "
          f"({net.index.min().date()} to {net.index.max().date()})")
    return net


_scada_net = _net_load_from_scada() if NET_LOAD_SOURCE in ("auto", "scada") else None

if _scada_net is not None:
    df_15min = _scada_net.loc[START_TS:END_TS].copy()
else:
    if NET_LOAD_SOURCE == "scada":
        raise RuntimeError(
            "NET_LOAD_SOURCE='scada' but the cache could not cover the window. "
            "Run: cd scada-cache && python update_cache.py --source nr"
        )
    print("Falling back to the raw demand folder:", DEMAND_FOLDER)
    raw_master = build_net_load_from_raw()
    df_raw = raw_master.copy()
    df_raw["Datetime"] = pd.to_datetime(df_raw["Datetime"], errors="coerce")
    df_raw.set_index("Datetime", inplace=True)
    df_15min = df_raw.resample("15min").mean().loc[START_TS:END_TS]

# Gap filling unchanged: same block yesterday/tomorrow averaged, then a time
# interpolation. About 2.6% of days are absent from the share; the 5-minute
# export cannot substitute because it carries no NR Load and no NR Wind, so
# those blocks are reconstructed here. The count is printed so it stays visible.
_missing_before = int(df_15min["Net_Load"].isna().sum())

df_15min["Net_Load"] = df_15min["Net_Load"].fillna(
    (df_15min["Net_Load"].shift(96) + df_15min["Net_Load"].shift(-96)) / 2
)
df_15min["Net_Load"] = df_15min["Net_Load"].interpolate(method="time").ffill().bfill()

net_load_raw = df_15min.reset_index()
if "index" in net_load_raw.columns:
    net_load_raw = net_load_raw.rename(columns={"index": "Datetime"})
net_load_raw.to_csv(ARTIFACT_DIR / "Net_Load_15min.csv", index=False)
print(
    f"15-min grid: {len(net_load_raw):,} rows | "
    f"{net_load_raw['Datetime'].min()} to {net_load_raw['Datetime'].max()}"
)
print(f"Reconstructed (interpolated) blocks: {_missing_before:,} "
      f"({_missing_before / max(len(net_load_raw), 1):.2%})")
print("Missing after fill:", int(net_load_raw["Net_Load"].isna().sum()))


# --------------------------------------------------------------------------
# ## SECTION 3 – Weather: historical download & merge

# ==========================================================================
# [cell 8]
# ==========================================================================
def download_weather(city, lat, lon, start, end):
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        "&hourly=temperature_2m,relative_humidity_2m"
        "&timezone=Asia/Kolkata"
    )
    j = requests.get(url, timeout=120).json()
    return pd.DataFrame({
        "Datetime":          pd.to_datetime(j["hourly"]["time"]),
        f"{city}_Temp":      j["hourly"]["temperature_2m"],
        f"{city}_Humidity":  j["hourly"]["relative_humidity_2m"],
    })


# ── Download / load per-city weather (cached), then build the weighted ───────
# composite Temp/Humidity used everywhere downstream, weighted by
# CITY_WEIGHTS (from City_Weather_Feature_Importance.csv, the v5 output).
city_weather_15 = {}
for city, (lat, lon) in CITY_COORDS.items():
    weight = CITY_WEIGHTS.get(city)
    if not weight:
        continue  # city missing from importance table / zero weight
    weather_cache = ARTIFACT_DIR / f"Weather_{city}_Hourly.csv"
    if weather_cache.exists():
        wx = pd.read_csv(weather_cache, parse_dates=["Datetime"])
    else:
        wx = download_weather(city, lat, lon, START_DATE, END_DATE)
        wx.to_csv(weather_cache, index=False)
    city_weather_15[city] = (
        wx.set_index("Datetime")
        .resample("15min").ffill()
        .loc[START_TS:END_TS]
    )

_common_idx = None
for _wdf in city_weather_15.values():
    _common_idx = _wdf.index if _common_idx is None else _common_idx.union(_wdf.index)

_weighted_temp = pd.Series(0.0, index=_common_idx)
_weighted_hum  = pd.Series(0.0, index=_common_idx)
_weight_total  = pd.Series(0.0, index=_common_idx)
for city, _wdf in city_weather_15.items():
    w = CITY_WEIGHTS[city]
    t = _wdf[f"{city}_Temp"].reindex(_common_idx).ffill().bfill()
    h = _wdf[f"{city}_Humidity"].reindex(_common_idx).ffill().bfill()
    _weighted_temp += w * t
    _weighted_hum  += w * h
    _weight_total  += w

weather_15 = pd.DataFrame({
    "Datetime":          _common_idx,
    "Weighted_Temp":     (_weighted_temp / _weight_total).values,
    "Weighted_Humidity": (_weighted_hum  / _weight_total).values,
}).sort_values("Datetime").reset_index(drop=True)

net_load_df = pd.merge(net_load_raw, weather_15, on="Datetime", how="left")
weather_cols = ["Weighted_Temp", "Weighted_Humidity"]
net_load_df[weather_cols] = net_load_df[weather_cols].ffill().bfill()


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
# ## SECTION 5 – Net-load feature engineering

# ==========================================================================
# [cell 12]
# ==========================================================================
def add_net_load_features(frame):
    out = frame.copy().sort_index()

    out["hour"]      = out.index.hour
    out["minute"]    = out.index.minute
    out["dayofweek"] = out.index.dayofweek
    out["month"]     = out.index.month
    out["quarter"]   = out.index.quarter
    out["block"]     = out.index.hour * 4 + out.index.minute // 15  # 0-based
    out["is_evening"] = ((out["hour"] >= 18) & (out["hour"] <= 23)).astype(int)

    out["hour_sin"]  = np.sin(2 * np.pi * out.index.hour / 24)
    out["hour_cos"]  = np.cos(2 * np.pi * out.index.hour / 24)
    out["dow_sin"]   = np.sin(2 * np.pi * out.index.dayofweek / 7)
    out["dow_cos"]   = np.cos(2 * np.pi * out.index.dayofweek / 7)
    out["block_sin"] = np.sin(2 * np.pi * out["block"] / 96)
    out["block_cos"] = np.cos(2 * np.pi * out["block"] / 96)

    out["temp_humidity"]  = out["Weighted_Temp"] * out["Weighted_Humidity"]
    out["cooling_degree"] = (out["Weighted_Temp"] - 22).clip(lower=0)
    out["heating_degree"] = (18 - out["Weighted_Temp"]).clip(lower=0)
    out["heat_index"]     = out["Weighted_Temp"] + 0.1 * out["Weighted_Humidity"]

    lags = [1, 2, 4, 8, 96, 192, 672, 1344, 364 * 96]
    for lag in lags:
        out[f"lag_{lag}"] = out["Net_Load"].shift(lag)
        out[f"lag_{lag}_missing"] = out[f"lag_{lag}"].isna().astype("int8")
        out[f"lag_{lag}"] = out[f"lag_{lag}"].fillna(0)

    weather_lags = [96, 672]
    for lag in weather_lags:
        out[f"temp_lag_{lag}"]     = out["Weighted_Temp"].shift(lag)
        out[f"humidity_lag_{lag}"] = out["Weighted_Humidity"].shift(lag)

    roll_windows = [96, 672, 1344, 364 * 96]
    for w in roll_windows:
        out[f"rolling_mean_{w}"] = out["Net_Load"].shift(1).rolling(w).mean()
        out[f"rolling_std_{w}"]  = out["Net_Load"].shift(1).rolling(w).std()
        out[f"rolling_mean_{w}_missing"] = out[f"rolling_mean_{w}"].isna().astype("int8")
        out[f"rolling_std_{w}_missing"]  = out[f"rolling_std_{w}"].isna().astype("int8")
        out[f"rolling_mean_{w}"] = out[f"rolling_mean_{w}"].fillna(0)
        out[f"rolling_std_{w}"]  = out[f"rolling_std_{w}"].fillna(0)

    out = out.ffill().bfill()
    features = [c for c in out.columns if c != "Net_Load"]
    return out, features


model_df, net_features = add_net_load_features(df)
X = model_df[net_features]
y = model_df["Net_Load"]
print("Rows:", len(X), "Features:", len(net_features), "Missing:", int(X.isna().sum().sum()))


# --------------------------------------------------------------------------
# ## SECTION 6 – Net-load model: train & evaluate

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


# --------------------------------------------------------------------------
# ## SECTION 7 – Net-load forecast for PEAK_MONTH

# ==========================================================================
# [cell 16]
# ==========================================================================
DECAY_LAMBDA = 0.12

forecast_start = pd.Timestamp(END_DATE) + pd.Timedelta(days=1)
month_start, month_end = month_bounds(PEAK_MONTH)
forecast_end   = max(forecast_start + pd.Timedelta(days=FORECAST_DAYS - 1), month_end)

weather_forecast_cutoff = forecast_start + pd.Timedelta(days=WEATHER_FORECAST_DAYS - 1)
print(f"Forecast start        : {forecast_start.date()}")
print(f"Forecast end          : {forecast_end.date()}")
print(f"Weather forecast up to: {weather_forecast_cutoff.date()} ({WEATHER_FORECAST_DAYS} days)")


def weather_profile_fill(target_index):
    hist = df[["Weighted_Temp", "Weighted_Humidity"]].copy()
    hist["block"]   = hist.index.hour * 4 + hist.index.minute // 15
    profile = hist.groupby("block")[["Weighted_Temp", "Weighted_Humidity"]].median()
    out = pd.DataFrame(index=target_index)
    out["block"]         = out.index.hour * 4 + out.index.minute // 15
    out["Weighted_Temp"]    = out["block"].map(profile["Weighted_Temp"])
    out["Weighted_Humidity"] = out["block"].map(profile["Weighted_Humidity"])
    return out.drop(columns="block").ffill().bfill()


_forecast_day_index = pd.date_range(forecast_start, weather_forecast_cutoff, freq="D")
_day_weights = {
    day: float(np.exp(-DECAY_LAMBDA * i))
    for i, day in enumerate(_forecast_day_index)
}

# NOTE: filename is distinct from the original v2 cache ("Future_Weather_...csv")
# so a stale Delhi-only cache from an earlier v2 run is never mistakenly reused.
future_weather_cache = (
    ARTIFACT_DIR / f"Future_Weather_Weighted_{forecast_start.date()}_to_{forecast_end.date()}.csv"
)
_expected_cols = {"Weighted_Temp", "Weighted_Humidity"}
if future_weather_cache.exists():
    future_weather_raw = pd.read_csv(future_weather_cache, parse_dates=["Datetime"])
    if not _expected_cols.issubset(future_weather_raw.columns):
        print(f"Cache at {future_weather_cache} missing {_expected_cols}; re-fetching.")
        future_weather_cache.unlink()
        future_weather_raw = None
else:
    future_weather_raw = None

if future_weather_raw is None:
    # Fetch each city's forecast, then build the same weighted composite
    # (CITY_WEIGHTS) used for the historical Weighted_Temp/Weighted_Humidity.
    city_future_frames = {}
    for city, (lat, lon) in CITY_COORDS.items():
        weight = CITY_WEIGHTS.get(city)
        if not weight:
            continue
        try:
            url = (
                "https://historical-forecast-api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                "&hourly=temperature_2m,relative_humidity_2m"
                f"&start_date={forecast_start.date()}&end_date={weather_forecast_cutoff.date()}"
                "&timezone=Asia/Kolkata"
            )
            j = requests.get(url, timeout=120,verify=False).json()
            city_future_frames[city] = pd.DataFrame({
                "Datetime":          pd.to_datetime(j["hourly"]["time"]),
                f"{city}_Temp":      j["hourly"]["temperature_2m"],
                f"{city}_Humidity":  j["hourly"]["relative_humidity_2m"],
            }).set_index("Datetime")
        except Exception as exc:
            print(f"Future weather API unavailable for {city}; skipping:", exc)

    if city_future_frames:
        _fc_idx = None
        for _wdf in city_future_frames.values():
            _fc_idx = _wdf.index if _fc_idx is None else _fc_idx.union(_wdf.index)
        _w_temp = pd.Series(0.0, index=_fc_idx)
        _w_hum  = pd.Series(0.0, index=_fc_idx)
        _w_tot  = pd.Series(0.0, index=_fc_idx)
        for city, _wdf in city_future_frames.items():
            w = CITY_WEIGHTS[city]
            t = _wdf[f"{city}_Temp"].reindex(_fc_idx).ffill().bfill()
            h = _wdf[f"{city}_Humidity"].reindex(_fc_idx).ffill().bfill()
            _w_temp += w * t
            _w_hum  += w * h
            _w_tot  += w
        future_weather_raw = pd.DataFrame({
            "Datetime":          _fc_idx,
            "Weighted_Temp":     (_w_temp / _w_tot).values,
            "Weighted_Humidity": (_w_hum  / _w_tot).values,
        }).sort_values("Datetime").reset_index(drop=True)
    else:
        print("No city forecasts available; using historical profile for all cities.")
        idx = pd.date_range(forecast_start, weather_forecast_cutoff, freq="1h")
        future_weather_raw = (
            weather_profile_fill(idx).reset_index().rename(columns={"index": "Datetime"})
        )
    future_weather_raw.to_csv(future_weather_cache, index=False)

future_weather_raw["Datetime"] = pd.to_datetime(future_weather_raw["Datetime"])
future_weather_15_actual = (
    future_weather_raw.set_index("Datetime")
    .resample("15min").ffill()
    .reindex(pd.date_range(forecast_start, weather_forecast_cutoff, freq="15min"))
)

all_forecast_idx = pd.date_range(forecast_start, forecast_end, freq="15min")
hist_profile_15  = weather_profile_fill(all_forecast_idx)

weight_series = pd.Series(
    {ts: _day_weights.get(ts.normalize(), 0.0)
     for ts in pd.date_range(forecast_start, weather_forecast_cutoff, freq="15min")},
    name="weight",
)
future_weather_15 = hist_profile_15.copy()

actual_aligned = future_weather_15_actual.reindex(all_forecast_idx)
weight_aligned  = weight_series.reindex(all_forecast_idx, fill_value=0.0)

valid_mask = actual_aligned.notna().all(axis=1) & (weight_aligned > 0)
future_weather_15.loc[valid_mask, "Weighted_Temp"] = (
    weight_aligned[valid_mask] * actual_aligned.loc[valid_mask, "Weighted_Temp"]
    + (1 - weight_aligned[valid_mask]) * hist_profile_15.loc[valid_mask, "Weighted_Temp"]
)
future_weather_15.loc[valid_mask, "Weighted_Humidity"] = (
    weight_aligned[valid_mask] * actual_aligned.loc[valid_mask, "Weighted_Humidity"]
    + (1 - weight_aligned[valid_mask]) * hist_profile_15.loc[valid_mask, "Weighted_Humidity"]
)
future_weather_15 = future_weather_15.ffill().bfill()
print(f"Blended weather ready: {future_weather_15.index.min()} → {future_weather_15.index.max()}")

future_df = df.copy()
preds = []
times = pd.date_range(forecast_start, forecast_end, freq="15min")
for next_time in times:
    row = pd.DataFrame(index=[next_time])
    row["Net_Load"]      = np.nan
    row["Weighted_Temp"]    = future_weather_15.loc[next_time, "Weighted_Temp"]
    row["Weighted_Humidity"] = future_weather_15.loc[next_time, "Weighted_Humidity"]
    temp = pd.concat([future_df, row])
    feat_df, _ = add_net_load_features(temp)
    pred = float(net_load_model.predict(feat_df.loc[[next_time], net_features])[0])
    row["Net_Load"] = pred
    future_df = pd.concat([future_df, row])
    preds.append(pred)
    
net_forecast = pd.DataFrame({"Datetime": times, "Forecast_Net_Load": preds})
net_forecast.to_csv(ARTIFACT_DIR / "Net_Load_Forecast.csv", index=False)
print(net_forecast["Datetime"].min(), "to", net_forecast["Datetime"].max(), len(net_forecast))


# --------------------------------------------------------------------------
# ## SECTION 15 – Market data: DAM / RTM load or fetch

# ==========================================================================
# [cell 18]
# ==========================================================================
dam_path = PROJECT_DIR / "DAM_Prices_2022_2026.csv"
rtm_path = PROJECT_DIR / "RTM_Prices_2022_2026.csv"
# [VOLUMES] The three traded-volume measures the IEX payload actually
# carries. Two columns that used to be listed here are gone for good:
#   Final_Scheduled_Volume_MW -- no scheduled-volume field exists anywhere in
#     the API payload, so this was never anything but NaN.
#   MCP_Rs_MWh -- the All-India clearing price, which is byte-identical to the
#     NR RTM price in 97.3% of blocks (r = 0.992). It IS the target; carrying
#     it as a same-block feature would be straight target leakage.
MARKET_SNAPSHOT_COLUMNS = {
    "Purchase_Bid_MW",
    "Sell_Bid_MW",
    "MCV_MW",
}


def _iex_client_date(value):
    return pd.Timestamp(value).strftime("%d-%m-%Y")


# [API-CHUNK-FIX] A single get_trade_data() call across the full ~4-year
# START_DATE..END_DATE span was silently coming back truncated (~493 days
# worth of rows every time, regardless of the requested range) -- the
# client/API appears to cap how much it returns per call, with no error
# raised. Fetching in bounded chunks and concatenating avoids the cap; this
# matches "fetching directly from the API" working fine when done in
# smaller pulls.
MARKET_FETCH_CHUNK_DAYS = 180


def _date_chunks(start, end, chunk_days=MARKET_FETCH_CHUNK_DAYS):
    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end)
    chunks   = []
    cur = start_ts
    while cur <= end_ts:
        chunk_end = min(cur + pd.Timedelta(days=chunk_days - 1), end_ts)
        chunks.append((cur, chunk_end))
        cur = chunk_end + pd.Timedelta(days=1)
    return chunks


def _parse_one_chunk(data, market_name):
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

    out["Market"]   = market_name
    out[price_col]  = out[["N1", "N2", "N3"]].mean(axis=1)
    return out[["Date", "Block", "Market", "N1", "N2", "N3", price_col]]


def load_market_from_iex_cache(market_name, start, end):
    chunks = _date_chunks(start, end)
    print(
        f"{market_name}: loading {start} to {end} via iex_client cache "
        f"in {len(chunks)} chunk(s) of up to {MARKET_FETCH_CHUNK_DAYS} days each"
    )

    parts = []
    for chunk_start, chunk_end in chunks:
        chunk_data = get_trade_data(
            _iex_client_date(chunk_start),
            _iex_client_date(chunk_end),
            market_name,
        )
        if chunk_data is None or len(chunk_data) == 0:
            print(f"  ! No data returned for {chunk_start.date()} to {chunk_end.date()}")
            continue
        parts.append(_parse_one_chunk(chunk_data, market_name))

    if not parts:
        raise ValueError(f"{market_name}: no data returned for any chunk in {start}..{end}")

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=["Date", "Block"]).reset_index(drop=True)

    # [VOLUMES] get_trade_data() returns area prices only -- the traded
    # volumes sit in the same raw payload and come back via
    # get_market_volumes(), so this costs no extra network round-trip for any
    # period already in the raw cache. Clearing_Price is deliberately NOT
    # mapped across: see the note on MARKET_SNAPSHOT_COLUMNS above.
    volumes = get_market_volumes(
        _iex_client_date(start), _iex_client_date(end), market_name
    ).rename(columns={
        "Buy_Volume":     "Purchase_Bid_MW",
        "Sell_Volume":    "Sell_Bid_MW",
        "Cleared_Volume": "MCV_MW",
    })
    volumes["Block"] = pd.to_numeric(volumes["Block"], errors="coerce").astype("Int64")

    combined["_join_date"] = parse_date_series(combined["Date"])
    combined["Block"] = pd.to_numeric(combined["Block"], errors="coerce").astype("Int64")
    combined = combined.merge(
        volumes[["Date", "Block", "Purchase_Bid_MW", "Sell_Bid_MW", "MCV_MW"]]
        .rename(columns={"Date": "_join_date"}),
        on=["_join_date", "Block"], how="left",
    ).drop(columns="_join_date")

    _vol_missing = combined["MCV_MW"].isna().mean()
    print(f"  volumes merged: {1 - _vol_missing:.1%} of rows have traded-volume data")

    got_days      = parse_date_series(combined["Date"]).dt.normalize().nunique()
    expected_days = (pd.Timestamp(end).normalize() - pd.Timestamp(start).normalize()).days + 1
    if got_days < expected_days:
        print(
            f"  ! {market_name}: got {got_days}/{expected_days} expected calendar "
            f"days after chunked fetch -- some chunks may still be short."
        )
    return combined


def market_csv_ready(path, price_col, required_start=None, required_end=None):
    if not path.exists():
        return False
    try:
        df_head = pd.read_csv(path)
    except Exception:
        return False

    cols = set(df_head.columns)
    if not (price_col in cols and MARKET_SNAPSHOT_COLUMNS.issubset(cols)):
        return False

    # [CACHE-COVERAGE-FIX] Column presence alone doesn't guarantee the cached
    # file actually spans the date range this run needs. A cache written by
    # an earlier run with a narrower START_DATE/END_DATE (or PEAK_MONTH) would
    # otherwise pass this check and get silently clipped down later, producing
    # far fewer rows than the API path. Explicitly verify date coverage here.
    if required_start is not None and required_end is not None:
        if "Date" in df_head.columns:
            # [DAYFIRST-BUG] Was dayfirst=True, which NaT'd every ISO date
            # past the 12th and so reported the cache as missing ~19 days a
            # month -- forcing a full refetch on every single run. Share the
            # one format-aware parser instead.
            cached_dates = parse_date_series(df_head["Date"])
        elif "Datetime" in df_head.columns:
            cached_dates = pd.to_datetime(df_head["Datetime"], errors="coerce")
        else:
            return False
        cached_dates = cached_dates.dropna()
        if cached_dates.empty:
            return False
        req_start = pd.Timestamp(required_start)
        req_end   = pd.Timestamp(required_end)
        if cached_dates.min() > req_start or cached_dates.max() < req_end:
            print(
                f"Cache at {path.name} covers {cached_dates.min().date()} to "
                f"{cached_dates.max().date()}, which does not fully span the "
                f"required {req_start.date()} to {req_end.date()} — refetching."
            )
            return False

        # [CACHE-COMPLETENESS-FIX] Endpoints alone don't prove the cache is
        # gap-free. A file with the right min/max but missing chunks in the
        # middle (e.g. an earlier interrupted/partial API pull) would pass
        # the check above and get silently accepted with far fewer rows than
        # the API path returns. Explicitly check every calendar day in the
        # required range is present.
        present_days  = set(cached_dates.dt.normalize().unique())
        expected_days = set(pd.date_range(req_start.normalize(), req_end.normalize(), freq="D"))
        missing_days  = sorted(expected_days - present_days)
        if missing_days:
            preview = ", ".join(d.strftime("%Y-%m-%d") for d in missing_days[:5])
            more = f" (+{len(missing_days) - 5} more)" if len(missing_days) > 5 else ""
            print(
                f"Cache at {path.name} is missing {len(missing_days)} day(s) "
                f"within the required range: {preview}{more} — refetching."
            )
            return False
    return True


MARKET_END_DATE = pd.Timestamp(END_DATE).date()

_start_ts_market = pd.Timestamp(START_DATE)
_end_ts_market   = pd.Timestamp(END_DATE) + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)

if (
    REFRESH_MARKET_DATA
    or not market_csv_ready(dam_path, "DAM_Price", START_DATE, MARKET_END_DATE)
    or not market_csv_ready(rtm_path, "RTM_Price", START_DATE, MARKET_END_DATE)
):
    dam = load_market_from_iex_cache("DAM", START_DATE, MARKET_END_DATE)
    rtm = load_market_from_iex_cache("RTM", START_DATE, MARKET_END_DATE)
    dam.to_csv(dam_path, index=False)
    rtm.to_csv(rtm_path, index=False)
else:
    dam = pd.read_csv(dam_path)
    rtm = pd.read_csv(rtm_path)

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
# ## SECTION 16 — RTM v3: "Hurdle-Quantile Ensemble with Climatology Decomposition"
#
# RTM prices behave like a **three-regime mixture**: a normal band that tracks
# DAM, a low-tail "crash" regime (near-zero prices during oversupply / mild
# weather / high renewable output), and a high-tail "spike" regime (prices at
# the regulatory cap during scarcity). This replaces v2's single mean +
# single-quantile blend (with hard-coded night/evening weights) with:
#
# 1. **Target decomposition**: `RTM_Price = seasonal_climatology(block, day_type,
#    season) + deviation`, climatology computed leak-free (expanding,
#    `shift(1)` per group).
# 2. **Distributional deviation model**: 5 LightGBM quantile regressors
#    (P05/P25/P50/P75/P95), monotonic-rearranged.
# 3. **Learned regime probabilities**: `P(spike)` / `P(crash)` classifiers
#    replace v2's fixed Block-range blend weights with a continuous, learned
#    mixing weight.
# 4. **Conformal calibration**: split-conformal adjustment of the P05/P95 band
#    against a held-out recent slice, for a verified empirical coverage.
# 5. **Recursive-forecast fix**: past `RTM_AR_FREEZE_DAYS` (default 3 days),
#    self-referential RTM lag/rolling features reset to climatology instead of
#    the model's own prior forecast — only genuinely exogenous drivers
#    (forecast Net_Load, forecast DAM, calendar) steer the long-horizon
#    forecast, eliminating the compounding-error bug in v2's recursive loop.
# 6. **Honest validation**: 5-fold walk-forward (expanding-window) CV instead
#    of one 80/20 split.
#
# *Domain note: CERC introduced DAM market coupling from January 2026, and
# 2024–2025 data shows increasingly frequent price crashes (not just spikes)
# from renewable oversupply — so this design clips historical prices to the current regulatory cap
# (`RTM_REGULATORY_CAP = 10000`) rather than trusting whatever cap applied
# when older rows were recorded, and sample weights are recency-decayed.*

# ==========================================================================
# [cell 20]
# ==========================================================================
# ============================================================================
# SECTION 16 (v3) — Config additions + market pressure features
# ============================================================================
import holidays as _holidays_pkg
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---- v3 config (extends the Section-0 config block; safe to re-run) --------
RTM_QUANTILES        = [0.05, 0.25, 0.50, 0.75, 0.85]   # deviation-model quantiles
                              # [v3.2] Lowered again, 0.90 -> 0.85, alongside the
                              # median (not P90) future market-factor profile fix
                              # in build_future_market_frame -- that P90 profile
                              # was the bigger source of the persistent
                              # above-actual bias (it fed every forecast day an
                              # artificially aggressive "high pressure" signal,
                              # keeping the spike classifier firing constantly).
                              # This quantile is the second, smaller lever on
                              # top of that fix. If forecasts are still running
                              # high after both changes, try 0.75 next.
RTM_SPIKE_PCTL       = 0.97     # (legacy/reference) percentile-based "elevated price" band
RTM_CRASH_PCTL       = 0.03     # data-driven "crash" (near-zero / oversupply) threshold
CAP_HIT_FRACTION     = 0.98     # [v3.3] PRIMARY TARGET: a block counts as a "cap-hit"
                                 # once RTM_Price >= CAP_HIT_FRACTION * RTM_REGULATORY_CAP
                                 # (9800 Rs/MWh by default). This is what "predict when
                                 # price reaches the cap" actually means -- it's a
                                 # tighter, more literal target than the old P97
                                 # percentile band, which could include prices well
                                 # below the cap and dilute what the classifier learns.
RTM_CONFORMAL_ALPHA  = 0.10     # 90% target coverage for the P05/P95 interval
RTM_AR_FREEZE_DAYS   = 3        # beyond this horizon, self-referential RTM lags are
                                 # frozen to climatology instead of recursed forward
RTM_PROFILE_MIN_OBS  = 20        # min obs required before trusting a profile cell
RTM_SUMMER_MONTHS    = {4, 5, 6}  # Apr–Jun: India's AC-driven demand/price season
RTM_REGULATORY_CAP   = 10000.0    # Rs/MWh — current CERC general RTM price cap.
                                   # NOT inferred from data: historical series can
                                   # contain readings from a prior regulatory regime
                                   # (older/looser caps) that don't reflect today's rules,
                                   # so we clip to the current cap rather than take max().

_hol_years = range(pd.Timestamp(START_DATE).year, pd.Timestamp(END_DATE).year + 2)
IN_HOLIDAYS = _holidays_pkg.India(years=_hol_years)

# [VOLUMES] The three volume measures the IEX payload carries, now populated
# with real data by load_market_from_iex_cache(). Previously this list also
# held Final_Scheduled_Volume_MW and MCP_Rs_MWh; both are gone -- the former
# does not exist in the API at all, the latter is the target under another
# name (see MARKET_SNAPSHOT_COLUMNS). Until this fix every column here was
# silently all-NaN, which meant several hundred all-NaN model features and a
# DAM_Market_Pressure_Score that was identically zero.
MARKET_FACTOR_COLS = [
    "Purchase_Bid_MW",
    "Sell_Bid_MW",
    "MCV_MW",
]
RTM_MARKET_FACTOR_COLS = MARKET_FACTOR_COLS
DAM_MARKET_FACTOR_COLS = [f"dam_{col}" for col in MARKET_FACTOR_COLS]

for col in MARKET_FACTOR_COLS:
    for _frame, _label in ((rtm, "RTM"), (dam, "DAM")):
        if col not in _frame.columns:
            raise KeyError(
                f"{_label} frame is missing {col}. The cached CSV predates the "
                "volume fix -- delete DAM_Prices_*.csv / RTM_Prices_*.csv (or "
                "set REFRESH_MARKET_DATA=True) so they are rebuilt with volumes."
            )
        _frame[col] = pd.to_numeric(_frame[col], errors="coerce")

dam_factors = dam[["Datetime"] + MARKET_FACTOR_COLS].rename(
    columns={col: f"dam_{col}" for col in MARKET_FACTOR_COLS}
)
market = pd.merge(
    dam[["Datetime", "Date", "Block", "DAM_Price"]],
    dam_factors, on="Datetime", how="inner",
)
market = pd.merge(
    market, rtm[["Datetime", "RTM_Price"] + RTM_MARKET_FACTOR_COLS],
    on="Datetime", how="inner",
)
hist_net = df[["Net_Load"]].reset_index()
market   = pd.merge(market, hist_net, on="Datetime", how="left")
market   = market.sort_values("Datetime").reset_index(drop=True)
market["Net_Load"] = market["Net_Load"].interpolate().ffill().bfill()

# Cap-aware history: clip to the current regulatory cap so profiles, the
# deviation target, and the spike classifier are all trained against today's
# rules rather than whatever cap happened to apply when older rows were set.
_pre_clip_over_cap = int((market["RTM_Price"] > RTM_REGULATORY_CAP).sum())
market["RTM_Price"] = market["RTM_Price"].clip(upper=RTM_REGULATORY_CAP)
print(f"Clipped {_pre_clip_over_cap} historical RTM rows down to the "
      f"{RTM_REGULATORY_CAP:.0f} Rs/MWh regulatory cap")


def add_market_pressure_features(frame, purchase_col, sell_col, mcv_col, out_prefix=""):
    out = frame.copy()
    for col in [purchase_col, sell_col, mcv_col]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    pfx = f"{out_prefix}_" if out_prefix else ""
    eps = 1e-6
    out[f"{pfx}Bid_Spread_MW"]             = out[purchase_col] - out[sell_col]
    out[f"{pfx}Bid_Ratio"]                 = out[purchase_col] / (out[sell_col].abs() + eps)
    out[f"{pfx}Cleared_Share_of_Purchase"] = out[mcv_col] / (out[purchase_col].abs() + eps)
    out[f"{pfx}Cleared_Share_of_Sell"]     = out[mcv_col] / (out[sell_col].abs() + eps)
    return out


def add_all_market_pressure_features(frame):
    out = add_market_pressure_features(
        frame, "Purchase_Bid_MW", "Sell_Bid_MW", "MCV_MW", out_prefix="",
    )
    out = add_market_pressure_features(
        out, "dam_Purchase_Bid_MW", "dam_Sell_Bid_MW", "dam_MCV_MW", out_prefix="dam",
    )
    return out


RTM_MARKET_DERIVED_COLS = [
    "Bid_Spread_MW", "Bid_Ratio", "Cleared_Share_of_Purchase",
    "Cleared_Share_of_Sell",
]
DAM_MARKET_DERIVED_COLS = [f"dam_{col}" for col in RTM_MARKET_DERIVED_COLS]
ALL_MARKET_RAW_COLS     = RTM_MARKET_FACTOR_COLS + DAM_MARKET_FACTOR_COLS
ALL_MARKET_DERIVED_COLS = RTM_MARKET_DERIVED_COLS + DAM_MARKET_DERIVED_COLS

print(f"Loaded {len(IN_HOLIDAYS)} India holiday dates spanning {list(_hol_years)}")
print("Market frame:", market.shape, market["Datetime"].min(), "→", market["Datetime"].max())


# --------------------------------------------------------------------------
# ### Feature engineering — calendar, DAM↔RTM spread, volatility, real-time tightness

# ==========================================================================
# [cell 22]
# ==========================================================================
# ============================================================================
# SECTION 17 (v3) — Feature engineering: calendar, spread, volatility, regime
# ============================================================================
def _day_type(dt_series):
    """weekday / weekend / holiday — holiday takes priority over weekend."""
    dow = dt_series.dt.dayofweek
    out = np.where(dow >= 5, "weekend", "weekday")
    is_hol = dt_series.dt.normalize().isin(
        pd.to_datetime(list(IN_HOLIDAYS.keys()))
    )
    return np.where(is_hol, "holiday", out)


def _season(dt_series):
    return np.where(dt_series.dt.month.isin(RTM_SUMMER_MONTHS), "summer", "other")


def add_rtm_features_v3(frame, include_market_factors=True):
    """
    Builds the exogenous / structural feature set used for both the
    seasonal-profile lookup and the ML deviation model. Every feature here
    is either (a) calendar/known-in-advance, (b) derived from Net_Load /
    DAM_Price (which are forecastable), or (c) a lag/rolling stat of the
    market itself (only trustworthy for near-horizon rows — see the
    recursive-forecast section for how far-horizon rows get these frozen).
    """
    out = frame.copy().sort_values("Datetime").reset_index(drop=True)
    out = add_all_market_pressure_features(out)

    out["hour"]      = out["Datetime"].dt.hour
    out["minute"]    = out["Datetime"].dt.minute
    out["dayofweek"] = out["Datetime"].dt.dayofweek
    out["month"]     = out["Datetime"].dt.month
    out["day_type"]  = _day_type(out["Datetime"])
    out["season"]    = _season(out["Datetime"])
    out["is_weekend"]  = (out["dayofweek"] >= 5).astype(int)
    out["is_holiday"]  = (out["day_type"] == "holiday").astype(int)
    out["is_summer"]   = (out["season"] == "summer").astype(int)

    out["block"]     = out["Block"].astype(int)
    out["block_sin"] = np.sin(2 * np.pi * (out["block"] - 1) / 96)
    out["block_cos"] = np.cos(2 * np.pi * (out["block"] - 1) / 96)
    out["hour_sin"]  = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"]  = np.cos(2 * np.pi * out["hour"] / 24)
    out["doy_sin"]   = np.sin(2 * np.pi * out["Datetime"].dt.dayofyear / 365.25)
    out["doy_cos"]   = np.cos(2 * np.pi * out["Datetime"].dt.dayofyear / 365.25)

    out["net_ramp"] = out["Net_Load"].diff().fillna(0)
    out["dam_ramp"] = out["DAM_Price"].diff().fillna(0)

    # [v3] Real-time supply/demand tightness: where does *today's* net load
    # sit relative to *today's* own min/max? A much sharper tightness signal
    # than the raw level, and it's exogenous (works off forecasted net load).
    day_grp = out.groupby(out["Datetime"].dt.date)["Net_Load"]
    day_min, day_max = day_grp.transform("min"), day_grp.transform("max")
    out["net_load_intraday_pctl"] = ((out["Net_Load"] - day_min) / (day_max - day_min + 1e-6)).clip(0, 1)

    # [v3] DAM↔RTM structural relationship — RTM tends to track DAM plus a
    # real-time deviation; expressing this explicitly helps the deviation model.
    out["DAM_RTM_Spread"] = out["RTM_Price"] - out["DAM_Price"]
    out["DAM_RTM_Ratio"]  = out["RTM_Price"] / (out["DAM_Price"].abs() + 1e-6)

    # [v3] Regime flags kept from v2 for continuity / interpretability, but
    # the model no longer *hard-codes* its blend weights off these — see the
    # learned spike/crash classifiers below.
    out["is_night_demand"] = ((out["Block"] >= 1)  & (out["Block"] <= 20)).astype(int)
    out["is_evening_ramp"] = ((out["Block"] >= 73) & (out["Block"] <= 88)).astype(int)
    out["is_peak_regime"]  = ((out["is_night_demand"] == 1) | (out["is_evening_ramp"] == 1)).astype(int)

    # NOTE: DAM_RTM_Spread / DAM_RTM_Ratio are intentionally excluded from this
    # ramp loop -- they're a function of the *current* row's RTM_Price, which
    # is exactly what we're trying to predict, so a same-row ramp on them
    # would leak the target. Only their *lagged* versions (spread_lag_*
    # below) are used as model inputs.
    for col in ALL_MARKET_RAW_COLS + ALL_MARKET_DERIVED_COLS:
        out[f"{col}_ramp"] = out[col].diff().fillna(0)

    for lag in [1, 2, 4, 96, 672]:
        out[f"rtm_lag_{lag}"] = out["RTM_Price"].shift(lag)
        out[f"dam_lag_{lag}"] = out["DAM_Price"].shift(lag)
        out[f"spread_lag_{lag}"] = out["DAM_RTM_Spread"].shift(lag)
        if include_market_factors:
            for col in ALL_MARKET_RAW_COLS + ALL_MARKET_DERIVED_COLS:
                out[f"{col}_lag_{lag}"] = out[col].shift(lag)

    out["night_rtm_lag1"] = out["rtm_lag_1"] * out["is_night_demand"]

    for w in [96, 672]:
        out[f"rtm_roll_mean_{w}"] = out["RTM_Price"].shift(1).rolling(w).mean()
        out[f"rtm_roll_std_{w}"]  = out["RTM_Price"].shift(1).rolling(w).std()
        out[f"dam_roll_mean_{w}"] = out["DAM_Price"].shift(1).rolling(w).mean()
        out[f"net_roll_std_{w}"]  = out["Net_Load"].shift(1).rolling(w).std()
        if include_market_factors:
            for col in ALL_MARKET_RAW_COLS + ALL_MARKET_DERIVED_COLS:
                out[f"{col}_roll_mean_{w}"] = out[col].shift(1).rolling(w).mean()

    out = out.ffill().bfill()

    features = [
        "Block", "DAM_Price", "Net_Load", "hour", "minute", "dayofweek", "month",
        "block_sin", "block_cos", "hour_sin", "hour_cos", "doy_sin", "doy_cos",
        "net_ramp", "dam_ramp", "net_load_intraday_pctl",
        "is_night_demand", "is_evening_ramp", "is_peak_regime",
        "is_weekend", "is_holiday", "is_summer", "night_rtm_lag1",
    ] + [c for c in out.columns if c.startswith((
        "rtm_lag_", "dam_lag_", "spread_lag_", "rtm_roll_", "dam_roll_", "net_roll_"
    ))]
    if include_market_factors:
        # [LEAK-FIX] RTM-side volumes are cleared by the very auction that
        # sets RTM_Price, so the same-block value is contemporaneous with the
        # target, not antecedent to it -- and a month out it is unknowable
        # anyway (build_future_market_frame fills it with a 30-day median), so
        # including it would train on real values and predict on climatology.
        # Only its lags and rolling means go in. DAM-side volumes are exempt:
        # DAM clears the day before delivery, so they are genuinely known
        # ahead of the RTM block they describe.
        rtm_contemporaneous = set(RTM_MARKET_FACTOR_COLS + RTM_MARKET_DERIVED_COLS)
        rtm_contemporaneous |= {f"{col}_ramp" for col in rtm_contemporaneous}

        market_features = [
            c for c in ALL_MARKET_RAW_COLS + ALL_MARKET_DERIVED_COLS
            if c not in rtm_contemporaneous
        ]
        market_features += [
            c for c in out.columns
            if c not in rtm_contemporaneous
            and (
                c.endswith("_ramp")
                or any(c.startswith(f"{col}_lag_") or c.startswith(f"{col}_roll_mean_")
                       for col in ALL_MARKET_RAW_COLS + ALL_MARKET_DERIVED_COLS)
            )
        ]
        features += [c for c in market_features if c in out.columns]
    features = list(dict.fromkeys(features))
    return out, features


rtm_model_df, rtm_features = add_rtm_features_v3(market, include_market_factors=True)
train_full = rtm_model_df.dropna(subset=["RTM_Price"]).reset_index(drop=True)
print(f"Feature-engineered training frame: {train_full.shape}, {len(rtm_features)} features")


# --------------------------------------------------------------------------
# ### Leak-free seasonal/climatology profile (replaces the single P75/P95 anchor)

# ==========================================================================
# [cell 24]
# ==========================================================================
# ============================================================================
# SECTION 18 (v3) — Leak-free seasonal/climatology block profile
# ============================================================================
# Instead of a single hard-coded P75/P95 "anchor" (v2), we build a full
# climatology table keyed on (Block, day_type, season) using an *expanding,
# shift(1)* median/quantiles — i.e. at every historical row we only ever use
# data strictly BEFORE that row, so there is zero leakage and the same
# machinery can be reused verbatim for the future forecast month.

PROFILE_KEYS = ["block", "day_type", "season"]
PROFILE_QUANTILES = [0.10, 0.50, 0.90]


def _expanding_leakfree_quantiles(frame, value_col, keys, quantiles):
    """For each row, compute quantile(value_col) over all *prior* rows that
    share the same `keys` group. Returns a dict of new columns."""
    frame = frame.sort_values("Datetime").reset_index(drop=True)
    out_cols = {f"{value_col}_profile_p{int(q*100)}": np.full(len(frame), np.nan) for q in quantiles}
    for _, idx in frame.groupby(keys).groups.items():
        idx = np.sort(idx.values)
        shifted = frame[value_col].values[idx]
        s = pd.Series(shifted)
        for q in quantiles:
            # shift(1) -> never includes the current row; expanding -> only the past
            out_cols[f"{value_col}_profile_p{int(q*100)}"][idx] = (
                s.shift(1).expanding(min_periods=RTM_PROFILE_MIN_OBS).quantile(q).values
            )
    for k, v in out_cols.items():
        frame[k] = v
    return frame


train_full = _expanding_leakfree_quantiles(train_full, "RTM_Price", PROFILE_KEYS, PROFILE_QUANTILES)

# Early rows in a group won't have RTM_PROFILE_MIN_OBS prior observations yet;
# backfill within-group then fall back to the global median so nothing is NaN.
for q in PROFILE_QUANTILES:
    col = f"RTM_Price_profile_p{int(q*100)}"
    train_full[col] = train_full.groupby(PROFILE_KEYS)[col].transform(lambda s: s.bfill())
    train_full[col] = train_full[col].fillna(train_full["RTM_Price"].median())

train_full["seasonal_profile"] = train_full["RTM_Price_profile_p50"]
train_full["deviation_target"] = train_full["RTM_Price"] - train_full["seasonal_profile"]

# Static lookup table (last available expanding value per group) — this is
# what we reuse for every row of the future forecast month, since by
# definition none of that data is available for the group's own expanding
# window yet.
FINAL_PROFILE_TABLE = (
    train_full.sort_values("Datetime")
    .groupby(PROFILE_KEYS)[[f"RTM_Price_profile_p{int(q*100)}" for q in PROFILE_QUANTILES] + ["RTM_Price"]]
    .last()
)
GLOBAL_PROFILE_FALLBACK = {
    f"p{int(q*100)}": train_full["RTM_Price"].quantile(q) for q in PROFILE_QUANTILES
}


def lookup_seasonal_profile(block, day_type, season, q=50):
    key = (block, day_type, season)
    col = f"RTM_Price_profile_p{q}"
    if key in FINAL_PROFILE_TABLE.index:
        val = FINAL_PROFILE_TABLE.loc[key, col]
        if pd.notna(val):
            return float(val)
    return float(GLOBAL_PROFILE_FALLBACK[f"p{q}"])


print("Seasonal profile table built:", FINAL_PROFILE_TABLE.shape[0], "groups")
print(train_full[["Datetime", "RTM_Price", "seasonal_profile", "deviation_target"]].tail())


# --------------------------------------------------------------------------
# ### Regime classifiers (spike/crash) + quantile deviation-model ensemble

# ==========================================================================
# [cell 26]
# ==========================================================================
# ============================================================================
# SECTION 19 (v3) — Regime classifiers + quantile deviation-model ensemble
# ============================================================================
from lightgbm import LGBMRegressor, LGBMClassifier

# [v3.3] Primary target: literal cap-hit events, not a percentile band. This
# is the thing you actually care about predicting ("when does RTM reach the
# cap"), so the classifier is trained directly against it.
CAP_HIT_THRESH = CAP_HIT_FRACTION * RTM_REGULATORY_CAP
SPIKE_THRESH   = CAP_HIT_THRESH   # kept as SPIKE_THRESH so downstream code (Peak
                                   # scoring, forecast output, etc.) needs no renaming
CRASH_THRESH = train_full["RTM_Price"].quantile(RTM_CRASH_PCTL)
PRICE_CAP    = RTM_REGULATORY_CAP   # fixed at the current CERC cap, not train_full.max()
PRICE_FLOOR  = max(0.0, train_full["RTM_Price"].min())

train_full["is_spike"] = (train_full["RTM_Price"] >= SPIKE_THRESH).astype(int)
train_full["is_crash"] = (train_full["RTM_Price"] <= CRASH_THRESH).astype(int)

print(f"Cap-hit threshold ({CAP_HIT_FRACTION:.0%} of the {RTM_REGULATORY_CAP:.0f} cap): "
      f"{SPIKE_THRESH:.0f} Rs/MWh ({train_full['is_spike'].mean()*100:.1f}% of historical rows)")
print(f"Crash threshold (P{int(RTM_CRASH_PCTL*100)}): {CRASH_THRESH:.0f} Rs/MWh "
      f"({train_full['is_crash'].mean()*100:.1f}% of rows)")
print(f"Observed price range: [{PRICE_FLOOR:.0f}, {PRICE_CAP:.0f}]")

_days_old = (train_full["Datetime"].max() - train_full["Datetime"]).dt.total_seconds() / 86400
RECENCY_HALFLIFE_DAYS = 270
recency_weight = 0.5 ** (_days_old / RECENCY_HALFLIFE_DAYS)
mean_rtm_price = train_full["RTM_Price"].mean()
magnitude_weight = (train_full["RTM_Price"] / mean_rtm_price).clip(1.0, 8.0)
sample_weights_full = (recency_weight * magnitude_weight).values


def make_classifier():
    return LGBMClassifier(
        objective="binary", max_depth=6, learning_rate=0.05, n_estimators=400,
        subsample=0.9, colsample_bytree=0.9, reg_alpha=0.5, reg_lambda=3.0,
        class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
    )


def make_quantile_regressor(alpha):
    return LGBMRegressor(
        objective="quantile", alpha=alpha, max_depth=8, learning_rate=0.04,
        n_estimators=600, subsample=0.9, colsample_bytree=0.9,
        reg_alpha=0.5, reg_lambda=3.0, random_state=42, n_jobs=-1, verbose=-1,
    )


def rearrange_quantiles(pred_matrix):
    return np.sort(pred_matrix, axis=1)


def pinball_loss(y_true, y_pred, quantile):
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def precision_recall_f1(y_true, y_prob, threshold):
    """Cap-hit classification metrics at a given probability threshold.
    Used to tune how aggressively we call a block a predicted cap-hit --
    the goal is a threshold that neither over-calls (low precision, false
    alarms every night) nor under-calls (low recall, misses real cap events)."""
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


CAP_HIT_THRESHOLD_GRID = np.arange(0.05, 0.96, 0.05)


X_cols = rtm_features


# --------------------------------------------------------------------------
# ### Walk-forward validation (5-fold expanding window)

# ==========================================================================
# [cell 28]
# ==========================================================================
# ============================================================================
# SECTION 20 (v3) — Walk-forward validation (expanding window, multi-fold)
# ============================================================================
N_FOLDS      = 5
FOLD_TEST_SZ = int(len(train_full) * 0.08)
MIN_TRAIN_SZ = int(len(train_full) * 0.40)

fold_rows = []
threshold_sweep_rows = []   # [v3.3] cap-hit precision/recall/F1 across a threshold grid
n = len(train_full)
for fold in range(N_FOLDS):
    test_end   = n - (N_FOLDS - 1 - fold) * FOLD_TEST_SZ
    test_start = test_end - FOLD_TEST_SZ
    train_end  = test_start
    if train_end < MIN_TRAIN_SZ or test_start < 0 or test_end > n:
        continue

    tr = train_full.iloc[:train_end]
    te = train_full.iloc[test_start:test_end]
    if len(tr) == 0 or len(te) == 0:
        continue

    w_tr = sample_weights_full[:train_end]

    q_preds = {}
    for q in RTM_QUANTILES:
        m = make_quantile_regressor(q)
        m.fit(tr[X_cols], tr["deviation_target"], sample_weight=w_tr)
        q_preds[q] = m.predict(te[X_cols])
    q_matrix = rearrange_quantiles(np.column_stack([q_preds[q] for q in RTM_QUANTILES]))

    clf_spike = make_classifier()
    clf_spike.fit(tr[X_cols], tr["is_spike"])
    p_spike = clf_spike.predict_proba(te[X_cols])[:, 1]

    clf_crash = make_classifier()
    clf_crash.fit(tr[X_cols], tr["is_crash"])
    p_crash = clf_crash.predict_proba(te[X_cols])[:, 1]

    dev_low, dev_mid = q_matrix[:, 0], q_matrix[:, 1:4].mean(axis=1)
    dev_high = q_matrix[:, 4]
    p_mid = np.clip(1 - p_spike - p_crash, 0, 1)
    norm = p_spike + p_crash + p_mid + 1e-9
    pred_dev = (p_crash * dev_low + p_mid * dev_mid + p_spike * dev_high) / norm
    pred_price = np.clip(te["seasonal_profile"].values + pred_dev, PRICE_FLOOR, PRICE_CAP)

    p_cap_hit_true = te["is_spike"].values
    default_prec, default_rec, default_f1 = precision_recall_f1(p_cap_hit_true, p_spike, 0.5)

    y_true = te["RTM_Price"].values
    row = {
        "Fold": fold + 1,
        "Train_Size": len(tr),
        "Test_Size": len(te),
        "MAE": mean_absolute_error(y_true, pred_price),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, pred_price))),
        "CapHit_Precision@0.5": default_prec,
        "CapHit_Recall@0.5": default_rec,
        "CapHit_F1@0.5": default_f1,
        "Crash_Recall": float((p_crash[te["is_crash"].values == 1] > 0.5).mean()) if te["is_crash"].sum() else np.nan,
    }
    for q in RTM_QUANTILES:
        row[f"Pinball_Q{int(q*100)}"] = pinball_loss(
            te["deviation_target"].values, q_preds[q], q
        )
    fold_rows.append(row)

    for t in CAP_HIT_THRESHOLD_GRID:
        prec, rec, f1 = precision_recall_f1(p_cap_hit_true, p_spike, t)
        threshold_sweep_rows.append({
            "Fold": fold + 1, "Threshold": round(float(t), 2),
            "Precision": prec, "Recall": rec, "F1": f1,
        })

cv_results = pd.DataFrame(fold_rows)
print(cv_results.round(2))
cv_results.to_csv(ARTIFACT_DIR / f"RTM_v3_WalkForwardCV_{PEAK_MONTH}.csv", index=False)
print("\nMean MAE across folds:", round(cv_results["MAE"].mean(), 2),
      "| Mean RMSE:", round(cv_results["RMSE"].mean(), 2))

# [v3.3] Pick the cap-hit probability threshold that maximizes mean F1 across
# folds -- balancing "predicted every night is a cap-hit" (low precision,
# overdoing it) against "never flags a real cap event" (low recall,
# underdoing it), per the brief: some over/under-calling is fine, but the
# threshold shouldn't be tuned to either extreme.
threshold_sweep = pd.DataFrame(threshold_sweep_rows)
threshold_summary = (
    threshold_sweep.groupby("Threshold")[["Precision", "Recall", "F1"]]
    .mean().reset_index().sort_values("Threshold")
)
print("\nCap-hit threshold sweep (mean across folds):")
print(threshold_summary.round(3).to_string(index=False))

BEST_CAP_HIT_THRESHOLD = float(threshold_summary.loc[threshold_summary["F1"].idxmax(), "Threshold"])
best_row = threshold_summary.loc[threshold_summary["F1"].idxmax()]
print(f"\nSelected cap-hit decision threshold: {BEST_CAP_HIT_THRESHOLD:.2f} "
      f"(mean precision={best_row['Precision']:.2f}, recall={best_row['Recall']:.2f}, "
      f"F1={best_row['F1']:.2f})")
threshold_summary.to_csv(ARTIFACT_DIR / f"RTM_v3_CapHitThresholdSweep_{PEAK_MONTH}.csv", index=False)


# --------------------------------------------------------------------------
# ### Production training on full history + conformal calibration of the P05/P95 band

# ==========================================================================
# [cell 30]
# ==========================================================================
# ============================================================================
# SECTION 21 (v3) — Production models (full history) + conformal calibration
# ============================================================================
# Hold out the most recent slice purely for conformal calibration of the
# P05/P95 interval (split-conformal quantile regression, Romano et al. 2019).
CALIB_FRAC  = 0.10
calib_start = int(len(train_full) * (1 - CALIB_FRAC))
fit_df, calib_df = train_full.iloc[:calib_start], train_full.iloc[calib_start:]
fit_w = sample_weights_full[:calib_start]

rtm_quantile_models = {}
for q in RTM_QUANTILES:
    m = make_quantile_regressor(q)
    m.fit(fit_df[X_cols], fit_df["deviation_target"], sample_weight=fit_w)
    rtm_quantile_models[q] = m

rtm_spike_clf = make_classifier()
rtm_spike_clf.fit(fit_df[X_cols], fit_df["is_spike"])

rtm_crash_clf = make_classifier()
rtm_crash_clf.fit(fit_df[X_cols], fit_df["is_crash"])


def predict_deviation_ensemble(feat_df):
    """Returns (pred_price, q_lo, q_hi, p_spike, p_crash) for a feature frame
    that also has a 'seasonal_profile' column already attached."""
    q_preds = {q: rtm_quantile_models[q].predict(feat_df[X_cols]) for q in RTM_QUANTILES}
    q_matrix = rearrange_quantiles(np.column_stack([q_preds[q] for q in RTM_QUANTILES]))
    p_spike = rtm_spike_clf.predict_proba(feat_df[X_cols])[:, 1]
    p_crash = rtm_crash_clf.predict_proba(feat_df[X_cols])[:, 1]

    dev_low, dev_mid, dev_high = q_matrix[:, 0], q_matrix[:, 1:4].mean(axis=1), q_matrix[:, 4]
    p_mid = np.clip(1 - p_spike - p_crash, 0, 1)
    norm  = p_spike + p_crash + p_mid + 1e-9
    pred_dev = (p_crash * dev_low + p_mid * dev_mid + p_spike * dev_high) / norm

    profile = feat_df["seasonal_profile"].values
    pred_price = np.clip(profile + pred_dev, PRICE_FLOOR, PRICE_CAP)
    q_lo = profile + dev_low
    q_hi = profile + dev_high
    return pred_price, q_lo, q_hi, p_spike, p_crash


# ── Split-conformal calibration for the P05/P95 interval ────────────────────
_, calib_lo, calib_hi, _, _ = predict_deviation_ensemble(calib_df)
calib_y = calib_df["RTM_Price"].values
conformity_scores = np.maximum(calib_lo - calib_y, calib_y - calib_hi)
n_calib = len(conformity_scores)
conformal_level = min(1.0, np.ceil((n_calib + 1) * (1 - RTM_CONFORMAL_ALPHA)) / n_calib)
CONFORMAL_MARGIN = max(0.0, float(np.quantile(conformity_scores, conformal_level)))
print(f"Conformal margin (~{int((1-RTM_CONFORMAL_ALPHA)*100)}% target coverage): "
      f"±{CONFORMAL_MARGIN:.0f} Rs/MWh on top of the raw P05/P95 band")

empirical_coverage = float(np.mean(
    (calib_y >= calib_lo - CONFORMAL_MARGIN) & (calib_y <= calib_hi + CONFORMAL_MARGIN)
))
print(f"Calibration-set empirical coverage after conformal adjustment: {empirical_coverage*100:.1f}%")

# [v3.3] Cross-check the CV-selected cap-hit threshold against this genuinely
# held-out, most-recent calibration slice, and average the two F1 curves for
# a more robust final threshold than either holdout alone.
_, _, _, p_spike_calib, _ = predict_deviation_ensemble(calib_df)
calib_is_spike = calib_df["is_spike"].values
calib_sweep_rows = []
for t in CAP_HIT_THRESHOLD_GRID:
    prec, rec, f1 = precision_recall_f1(calib_is_spike, p_spike_calib, t)
    calib_sweep_rows.append({"Threshold": round(float(t), 2), "Precision": prec, "Recall": rec, "F1": f1})
calib_threshold_summary = pd.DataFrame(calib_sweep_rows)

combined_threshold_summary = threshold_summary.merge(
    calib_threshold_summary, on="Threshold", suffixes=("_cv", "_calib")
)
combined_threshold_summary["F1_mean"] = (
    combined_threshold_summary["F1_cv"] + combined_threshold_summary["F1_calib"]
) / 2
FINAL_CAP_HIT_THRESHOLD = float(
    combined_threshold_summary.loc[combined_threshold_summary["F1_mean"].idxmax(), "Threshold"]
)
best_combined = combined_threshold_summary.loc[combined_threshold_summary["F1_mean"].idxmax()]
print(f"\nFinal cap-hit decision threshold: {FINAL_CAP_HIT_THRESHOLD:.2f} "
      f"(CV F1={best_combined['F1_cv']:.2f}, calib-set F1={best_combined['F1_calib']:.2f}, "
      f"calib precision={best_combined['Precision_calib']:.2f}, "
      f"calib recall={best_combined['Recall_calib']:.2f})")
combined_threshold_summary.to_csv(
    ARTIFACT_DIR / f"RTM_v3_CapHitThresholdSweep_Combined_{PEAK_MONTH}.csv", index=False
)

# ── Re-fit on 100% of history for the actual forecast (calibration margin
#    computed above is reused as-is; refitting doesn't change coverage much
#    since it only adds the most recent 10% of rows) ─────────────────────────
rtm_quantile_models = {}
for q in RTM_QUANTILES:
    m = make_quantile_regressor(q)
    m.fit(train_full[X_cols], train_full["deviation_target"], sample_weight=sample_weights_full)
    rtm_quantile_models[q] = m

rtm_spike_clf = make_classifier()
rtm_spike_clf.fit(train_full[X_cols], train_full["is_spike"])
rtm_crash_clf = make_classifier()
rtm_crash_clf.fit(train_full[X_cols], train_full["is_crash"])

print("Production models (5 quantiles + spike/crash classifiers) trained on full history.")


# --------------------------------------------------------------------------
# ### Future market frame + recursive forecast with climatology-reset beyond day 3

# ==========================================================================
# [cell 32]
# ==========================================================================
# ============================================================================
# SECTION 22 (v3) — Future market frame + recursive forecast (climatology-reset)
# ============================================================================
# [v3.2] Market-factor profiles (bid volumes, MCV, scheduled volume, MCP) now
# use the block-level MEDIAN of the trailing 30 days, not the 90th percentile.
# Using P90 fed every single day of the forecast month an artificially
# aggressive "high market pressure" signal -- since historically that pattern
# correlates with spikes, it kept the spike classifier firing (and the
# quantile ensemble leaning toward its upper quantile) almost every night/
# evening block of the month, not just on days that actually spike. That was
# the main source of the forecast running persistently above actuals; the
# median profile gives the model a typical day's market conditions instead.
def build_future_market_frame():
    month_start, month_end = month_bounds(PEAK_MONTH)
    times = pd.date_range(month_start, month_end, freq="15min")
    out   = pd.DataFrame({"Datetime": times})
    out["Date"]  = out["Datetime"].dt.normalize()
    out["Block"] = out["Datetime"].dt.hour * 4 + out["Datetime"].dt.minute // 15 + 1

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
        dam_factor_profile = (
            known_dam_factors.query("Datetime >= @recent_dam_cutoff")
            .groupby("Block")[MARKET_FACTOR_COLS].median()
        )
        for col in MARKET_FACTOR_COLS:
            out[f"dam_{col}"] = out["Block"].map(dam_factor_profile[col])
            out[f"dam_{col}"] = out[f"dam_{col}"].fillna(known_dam_factors[col].median())
            out[f"dam_{col}"] = out[f"dam_{col}"].ffill().bfill()
    else:
        for col in MARKET_FACTOR_COLS:
            out[f"dam_{col}"] = np.nan

    known_rtm_factors = rtm[["Datetime", "Block"] + RTM_MARKET_FACTOR_COLS].copy()
    known_rtm_factors = known_rtm_factors.dropna(subset=RTM_MARKET_FACTOR_COLS, how="all")
    if not known_rtm_factors.empty:
        recent_rtm_cutoff = known_rtm_factors["Datetime"].max() - pd.Timedelta(days=30)
        rtm_factor_profile = (
            known_rtm_factors.query("Datetime >= @recent_rtm_cutoff")
            .groupby("Block")[RTM_MARKET_FACTOR_COLS].median()
        )
        for col in RTM_MARKET_FACTOR_COLS:
            out[col] = out["Block"].map(rtm_factor_profile[col])
            out[col] = out[col].fillna(known_rtm_factors[col].median())
            out[col] = out[col].ffill().bfill()
    else:
        for col in RTM_MARKET_FACTOR_COLS:
            out[col] = np.nan

    month_net = net_forecast[["Datetime", "Forecast_Net_Load"]].rename(
        columns={"Forecast_Net_Load": "Net_Load"}
    )
    hist_net_local = df[["Net_Load"]].reset_index()
    out = out.merge(hist_net_local, on="Datetime", how="left", suffixes=("", "_hist"))
    out = out.merge(month_net, on="Datetime", how="left", suffixes=("", "_forecast"))
    out["Net_Load"] = out["Net_Load"].fillna(out.get("Net_Load_forecast"))
    out = out.drop(columns=[c for c in ["Net_Load_forecast"] if c in out.columns])
    out["Net_Load"] = out["Net_Load"].interpolate().ffill().bfill()
    return out


future_market = build_future_market_frame()

# Precomputed climatology fallbacks used only once we cross the AR-freeze
# horizon (see RTM_AR_FREEZE_DAYS). This is the key structural fix vs v2:
# instead of letting self-generated forecasts feed back into their own lag
# features indefinitely (compounding bias/variance as the horizon grows),
# every self-referential RTM feature gets "reset" to climatology beyond
# RTM_AR_FREEZE_DAYS, so only genuinely exogenous drivers (forecast Net_Load,
# forecast DAM, calendar) steer the long-horizon forecast.
BLOCK_STD_FALLBACK   = train_full["RTM_Price"].std()
BLOCK_STD_PROFILE    = train_full.groupby("block")["RTM_Price"].std().to_dict()
BLOCK_SPREAD_PROFILE = train_full.groupby("block")["DAM_RTM_Spread"].median().to_dict()

history = market[
    ["Datetime", "Date", "Block", "DAM_Price", "Net_Load", "RTM_Price"]
    + RTM_MARKET_FACTOR_COLS + DAM_MARKET_FACTOR_COLS
].copy()

month_start, _ = month_bounds(PEAK_MONTH)
rtm_forecasts, rtm_lo, rtm_hi, rtm_p_spike, rtm_p_crash = [], [], [], [], []

for _, base_row in future_market.iterrows():
    current_row = {
        "Datetime": base_row["Datetime"], "Date": base_row["Date"],
        "Block": base_row["Block"], "DAM_Price": base_row["DAM_Price"],
        "Net_Load": base_row["Net_Load"], "RTM_Price": np.nan,
    }
    for col in RTM_MARKET_FACTOR_COLS:
        current_row[col] = base_row[col]
    for col in DAM_MARKET_FACTOR_COLS:
        current_row[col] = base_row[col]

    current = pd.DataFrame([current_row])
    temp = pd.concat([history.tail(HISTORY_TAIL), current], ignore_index=True)
    temp_feat, _ = add_rtm_features_v3(temp, include_market_factors=True)
    row_feat = temp_feat.tail(1).copy()

    blk = int(base_row["Block"])
    dtp = _day_type(pd.Series([base_row["Datetime"]])).item()
    ssn = _season(pd.Series([base_row["Datetime"]])).item()
    row_feat["seasonal_profile"] = lookup_seasonal_profile(blk, dtp, ssn, q=50)

    day_offset = (base_row["Datetime"] - month_start).days
    if day_offset > RTM_AR_FREEZE_DAYS:
        frozen_val = row_feat["seasonal_profile"].iloc[0]
        frozen_std = BLOCK_STD_PROFILE.get(blk, BLOCK_STD_FALLBACK)
        frozen_spread = BLOCK_SPREAD_PROFILE.get(blk, 0.0)
        for lag in [1, 2, 4, 96, 672]:
            row_feat[f"rtm_lag_{lag}"] = frozen_val
            row_feat[f"spread_lag_{lag}"] = frozen_spread
        row_feat["night_rtm_lag1"] = frozen_val * row_feat["is_night_demand"].values[0]
        for w in [96, 672]:
            row_feat[f"rtm_roll_mean_{w}"] = frozen_val
            row_feat[f"rtm_roll_std_{w}"]  = frozen_std

    pred_price, q_lo, q_hi, p_sp, p_cr = predict_deviation_ensemble(row_feat)
    pred_price = float(np.clip(pred_price[0], PRICE_FLOOR, PRICE_CAP))
    q_lo_adj = float(np.clip(q_lo[0] - CONFORMAL_MARGIN, PRICE_FLOOR, PRICE_CAP))
    q_hi_adj = float(np.clip(q_hi[0] + CONFORMAL_MARGIN, PRICE_FLOOR, PRICE_CAP))

    current["RTM_Price"] = pred_price
    history = pd.concat([history, current], ignore_index=True)

    rtm_forecasts.append(pred_price)
    rtm_lo.append(q_lo_adj)
    rtm_hi.append(q_hi_adj)
    rtm_p_spike.append(float(p_sp[0]))
    rtm_p_crash.append(float(p_cr[0]))

future_market["Forecast_RTM_Price"]     = rtm_forecasts
future_market["Forecast_RTM_P05"]       = rtm_lo
future_market["Forecast_RTM_P95"]       = rtm_hi
future_market["Forecast_RTM_Spike_Prob"] = rtm_p_spike
future_market["Forecast_RTM_Crash_Prob"] = rtm_p_crash

# [v3.3] The primary deliverable: a clean boolean flag for "this block is
# predicted to hit the regulatory cap", using the F1-optimal threshold tuned
# in Sections 13-14 (walk-forward CV + held-out calibration slice) rather
# than an arbitrary 0.5 cutoff -- tuned to neither over-call every night nor
# miss real cap events.
future_market["Predicted_Cap_Hit"] = future_market["Forecast_RTM_Spike_Prob"] >= FINAL_CAP_HIT_THRESHOLD

future_market.to_csv(ARTIFACT_DIR / f"RTM_Forecast_v3_{PEAK_MONTH}.csv", index=False)
print(future_market[[
    "Datetime", "Forecast_RTM_Price", "Forecast_RTM_P05", "Forecast_RTM_P95",
    "Forecast_RTM_Spike_Prob", "Predicted_Cap_Hit",
]].head())
print(f"\nForecast RTM (v3) — mean: {np.mean(rtm_forecasts):.0f}  "
      f"P90: {np.percentile(rtm_forecasts, 90):.0f}  max: {np.max(rtm_forecasts):.0f}")

n_cap_blocks = int(future_market["Predicted_Cap_Hit"].sum())
n_cap_days   = future_market.loc[future_market["Predicted_Cap_Hit"], "Date"].nunique()
print(f"Predicted cap-hit blocks: {n_cap_blocks} / {len(future_market)} "
      f"({n_cap_blocks / len(future_market) * 100:.1f}%) across {n_cap_days} distinct day(s) "
      f"(decision threshold = {FINAL_CAP_HIT_THRESHOLD:.2f})")


# --------------------------------------------------------------------------
# ## SECTION 23 – Peak-hour scoring & selection

# ==========================================================================
# [cell 34]
# ==========================================================================
PEAK_END_BLOCK           = 96
PEAK_NET_LOAD_PERCENTILE = 0.90

# [v3.3] Peak hours should only start once the evening net-load ramp has
# passed its half point, per request ("normally in day time it is low but
# high in evening ... peak hours should maybe start from when the net load
# increasing part has passed its half point"). We find the afternoon trough,
# the evening peak that follows it, and the block where the average curve
# first crosses the halfway point in value between them -- then gate the
# whole candidate-window search (below) so no window can start earlier than
# that block. This uses the target month's own forecasted Net_Load curve
# (same source peak_base will use), computed early here because
# _build_candidate_windows() below caches its window list using
# PEAK_START_BLOCK immediately.
def time_to_block(hh_mm):
    hh, mm = (int(x) for x in hh_mm.split(":"))
    return int(hh * 4 + mm / 15) + 1

RAMP_TROUGH_SEARCH_START = "12:00"   # afternoon-trough search window start
RAMP_TROUGH_SEARCH_END   = "18:00"   # afternoon-trough search window end
RAMP_MIDPOINT_FRACTION   = 0.50      # how far into the rise "half point" means

# [v3.4] Hydro plants are not at full output the instant a peak hour starts --
# they need a ramp-up period first. If the declared window began at the raw
# net-load half point (e.g. 18:15-21:15) the machines would still be ramping
# through the opening blocks and would only hit maximum output part-way in.
# Pushing the earliest allowed start later by the ramp-up time (45 min ->
# 19:00-22:00) means hydro is already at max for the whole declared window.
HYDRO_RAMP_UP_MINUTES = 45
HYDRO_RAMP_UP_BLOCKS  = int(round(HYDRO_RAMP_UP_MINUTES / 15))

_ramp_month_start, _ramp_month_end = month_bounds(PEAK_MONTH)
_nl_for_ramp = future_market[
    (future_market["Datetime"] >= _ramp_month_start)
    & (future_market["Datetime"] <= _ramp_month_end)
][["Block", "Net_Load"]]
_avg_daily_curve = _nl_for_ramp.groupby("Block")["Net_Load"].mean().reindex(range(1, 97))

_trough_search_lo = time_to_block(RAMP_TROUGH_SEARCH_START)
_trough_search_hi = time_to_block(RAMP_TROUGH_SEARCH_END)
_trough_block = int(
    _avg_daily_curve.loc[_trough_search_lo:_trough_search_hi].idxmin()
)
_trough_value = float(_avg_daily_curve.loc[_trough_block])

_evening_peak_block = int(_avg_daily_curve.loc[_trough_block:PEAK_END_BLOCK].idxmax())
_evening_peak_value = float(_avg_daily_curve.loc[_evening_peak_block])

_ramp_half_value = _trough_value + RAMP_MIDPOINT_FRACTION * (_evening_peak_value - _trough_value)
_rise_segment = _avg_daily_curve.loc[_trough_block:_evening_peak_block]
_crossing = _rise_segment[_rise_segment >= _ramp_half_value]

_ramp_half_block = int(_crossing.index.min()) if not _crossing.empty else _trough_block
PEAK_START_BLOCK = min(_ramp_half_block + HYDRO_RAMP_UP_BLOCKS, PEAK_END_BLOCK)

print(
    f"Net-load ramp: afternoon trough at block {_trough_block} "
    f"({block_to_time(_trough_block)}, {_trough_value:,.0f} MW) -> evening peak "
    f"at block {_evening_peak_block} ({block_to_time(_evening_peak_block)}, "
    f"{_evening_peak_value:,.0f} MW). Half-point crossed at block "
    f"{_ramp_half_block} ({block_to_time(_ramp_half_block)}); shifted "
    f"{HYDRO_RAMP_UP_MINUTES} min ({HYDRO_RAMP_UP_BLOCKS} blocks) later for hydro "
    f"ramp-up -- peak-hour candidate windows are now restricted to start at or "
    f"after block {PEAK_START_BLOCK} ({block_to_time(PEAK_START_BLOCK)})."
)

# [v3.1] Net_Load x RTM_Price is now the dominant driver of the block-level
# Peak_Score, per request ("peak hours based on RTM prices * net load for the
# most part"). Standalone RTM/net-load and DAM market pressure remain as
# smaller supporting signals rather than being removed outright.
PEAK_WEIGHT_NL_RTM       = 0.70
PEAK_WEIGHT_NET_LOAD     = 0.08
PEAK_WEIGHT_RTM          = 0.15
PEAK_WEIGHT_DAM_MARKET   = 0.07

assert abs(PEAK_WEIGHT_NL_RTM + PEAK_WEIGHT_NET_LOAD + PEAK_WEIGHT_RTM + PEAK_WEIGHT_DAM_MARKET - 1.0) < 1e-9

# [v3.1] New concept: day-level recency weighting. Forecast quality
# deteriorates the further a day sits from END_DATE (the last day of real
# data) -- more recursive steps, more compounding uncertainty. This halves a
# day's influence on the monthly block score every PEAK_RECENCY_HALFLIFE_DAYS
# of distance from END_DATE, so early-month days (closest to real data) count
# more than late-month days when aggregating Peak_Score across the month.
# This feeds a new "recency_weighted" method below -- the existing
# mean/median/p75/p90/daily_top_frequency/rtm_pressure/robust_blend methods
# are left mechanically unchanged, exactly as requested.
PEAK_RECENCY_HALFLIFE_DAYS = 15


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
        + 0.06 * robust_z(seasonal["dam_Purchase_Bid_MW"])
        + 0.06 * robust_z(seasonal["dam_Sell_Bid_MW"])
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

    # [v3.1] Recency weight: 1.0 on the day right after END_DATE, halving
    # every PEAK_RECENCY_HALFLIFE_DAYS as we move deeper into the forecast
    # month (where the recursive RTM/net-load forecasts are least reliable).
    days_from_end_date = (
        pd.to_datetime(scored["Datetime"]).dt.normalize() - pd.Timestamp(END_DATE)
    ).dt.days
    scored["Recency_Weight"] = 0.5 ** (days_from_end_date / PEAK_RECENCY_HALFLIFE_DAYS)

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
        0.45 * _normalize_block_score(grouped["RTM_Peak_Score"].mean().reindex(range(1, 97)))
        + 0.33 * _normalize_block_score(grouped["DAM_Market_Pressure_Score"].mean().reindex(range(1, 97)))
        + 0.22 * _normalize_block_score(grouped["Purchase_Volume_Score"].mean().reindex(range(1, 97)))
    )
    seasonal_score = grouped["Seasonal_Block_Score"].mean().reindex(range(1, 97))

    # [v3.1] Recency-weighted mean: same idea as "mean" above, but each day's
    # Peak_Score is weighted by how close it is to END_DATE before averaging
    # within a block, instead of every day counting equally.
    recency_weighted_score = grouped.apply(
        lambda g: np.average(g["Peak_Score"], weights=g["Recency_Weight"])
    ).reindex(range(1, 97))

    methods = {
        "mean":               _normalize_block_score(mean_score),
        "median":             _normalize_block_score(median_score),
        "p75":                _normalize_block_score(p75_score),
        "p90":                _normalize_block_score(p90_score),
        "daily_top_frequency": _normalize_block_score(top_frequency),
        "rtm_pressure":       _normalize_block_score(rtm_pressure),
        "recency_weighted":   _normalize_block_score(recency_weighted_score),
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
month_start, month_end = _ramp_month_start, _ramp_month_end
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
peak_base["dam_Purchase_Bid_Ramp"] = peak_base["dam_Purchase_Bid_MW"].diff()
peak_base["dam_Sell_Bid_Ramp"]     = peak_base["dam_Sell_Bid_MW"].diff()
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
# [VOLUMES] DAM_MCP_Score is gone with MCP_Rs_MWh. Nothing is lost: the
# All-India DAM clearing price it was built from is near-identical to
# DAM_Price, which is already a first-class feature and score input. The
# remaining three weights are renormalised from 0.35/0.25/0.25 to sum to 1.
peak_base["DAM_Market_Pressure_Score"] = (
    0.40 * robust_z(peak_base["DAM_Purchase_Score"])
    + 0.30 * robust_z(peak_base["DAM_Sell_Score"])
    + 0.30 * robust_z(peak_base["dam_MCV_MW"])
)
peak_base["Purchase_Volume_Score"] = (
    0.75 * robust_z(peak_base["Purchase_Bid_MW"])
    + 0.25 * robust_z(peak_base["Purchase_Bid_Ramp"].clip(lower=0))
)
peak_base["Sell_Bid_Score"] = (
    0.65 * robust_z(peak_base["Sell_Bid_MW"])
    + 0.35 * robust_z(peak_base["Sell_Bid_Ramp"].clip(lower=0))
)
# [VOLUMES] Final_Scheduled_Volume_MW does not exist in the IEX payload, so
# the cleared-volume score is now MCV and its ramp; MCP_Pressure_Score is gone
# with the clearing price. Tightness renormalises 0.30/0.25/0.20 to sum to 1.
peak_base["Cleared_Volume_Score"] = (
    0.75 * robust_z(peak_base["MCV_MW"])
    + 0.25 * robust_z(peak_base["MCV_Ramp"].clip(lower=0))
)
peak_base["Market_Tightness_Score"] = (
    0.40 * robust_z(peak_base["Bid_Spread_MW"])
    + 0.33 * robust_z(peak_base["Bid_Ratio"])
    + 0.27 * robust_z(peak_base["Cleared_Share_of_Sell"])
)
peak_base["Peak_Score"] = (
    PEAK_WEIGHT_NL_RTM     * robust_z(peak_base["NL_RTM_Interaction"])
    + PEAK_WEIGHT_NET_LOAD * peak_base["Net_Load_Peak_Score"]
    + PEAK_WEIGHT_RTM      * peak_base["RTM_Peak_Score"]
    + PEAK_WEIGHT_DAM_MARKET * robust_z(peak_base["DAM_Market_Pressure_Score"])
)

# [v3.1] Default selection now uses the recency-weighted aggregation (days
# closer to END_DATE count more). Switch this string back to "robust_blend"
# (or any other key in the comparison table) to revert to the old default --
# all methods are still computed and shown side by side below either way.
PEAK_SELECTION_METHOD = "recency_weighted"
best, score_by_block, peak_method_comparison = select_monthly_peak_hours(
    peak_base, selection_method=PEAK_SELECTION_METHOD
)

print(
    f"Peak-hour scoring: NL×RTM interaction ({PEAK_WEIGHT_NL_RTM:.0%}, dominant), "
    f"RTM price ({PEAK_WEIGHT_RTM:.0%}), net-load top-"
    f"{int((1 - PEAK_NET_LOAD_PERCENTILE) * 100)}% excess + top-10% ramp "
    f"({PEAK_WEIGHT_NET_LOAD:.0%}), DAM market ({PEAK_WEIGHT_DAM_MARKET:.0%}). "
    f"No time-of-day prior."
)
print(f"  Day-recency half-life: {PEAK_RECENCY_HALFLIFE_DAYS} days from END_DATE "
      f"({END_DATE}) -- early-month forecasts count more than late-month ones.")
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
        f"NL×RTM interaction ({PEAK_WEIGHT_NL_RTM:.0%}, dominant), "
        f"RTM price ({PEAK_WEIGHT_RTM:.0%}), "
        f"net-load top-{int((1 - PEAK_NET_LOAD_PERCENTILE) * 100)}% excess + top-10% ramp "
        f"({PEAK_WEIGHT_NET_LOAD:.0%}), DAM bid/volume/MCP ({PEAK_WEIGHT_DAM_MARKET:.0%}); "
        f"windows start at/after block {PEAK_START_BLOCK} ({block_to_time(PEAK_START_BLOCK)}, post-ramp-halfpoint); "
        "recency-weighted aggregation across the month "
        f"(half-life {PEAK_RECENCY_HALFLIFE_DAYS}d from END_DATE -- early-month days "
        "weighted more heavily than late-month days); "
        "RTM v3: seasonal-climatology decomposition + 5-quantile deviation ensemble, "
        "learned spike/crash regime probabilities, conformal-calibrated P05/P95 band, "
        "climatology-reset recursive forecast beyond day 3"
    ),
}])

monthly_peak.to_excel(ARTIFACT_DIR / f"Monthly_Peak_Hours_{PEAK_MONTH}.xlsx", index=False)
monthly_peak.to_csv(ARTIFACT_DIR / f"Monthly_Peak_Hours_{PEAK_MONTH}.csv", index=False)
peak_method_comparison.to_excel(ARTIFACT_DIR / f"Peak_Method_Comparison_{PEAK_MONTH}.xlsx", index=False)
peak_method_comparison.to_csv(ARTIFACT_DIR / f"Peak_Method_Comparison_{PEAK_MONTH}.csv", index=False)
print(monthly_peak)


# --------------------------------------------------------------------------
# ## SECTION 24 – Daily audit table

# ==========================================================================
# [cell 36]
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
        "Avg_Net_Load_Peak_Score":     round(selected["Net_Load_Peak_Score"].mean(), 3),
        "Avg_RTM_Peak_Score":          round(selected["RTM_Peak_Score"].mean(), 3),
        "Avg_NL_RTM_Interaction":      round(selected["NL_RTM_Interaction"].mean(), 0),
        "Avg_DAM_Market_Pressure_Score": round(selected["DAM_Market_Pressure_Score"].mean(), 3),
        "Avg_Purchase_Volume_Score":   round(selected["Purchase_Volume_Score"].mean(), 3),
        "Avg_Sell_Bid_Score":          round(selected["Sell_Bid_Score"].mean(), 3),
        "Avg_Cleared_Volume_Score":    round(selected["Cleared_Volume_Score"].mean(), 3),
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
# ## SECTION 25 – Monthly overview chart

# ==========================================================================
# [cell 38]
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
axes[1].set_title("Forecast RTM Price — v3 (Climatology + Quantile-Ensemble + Regime Probabilities)")
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
    ]].mean().reindex(range(1, 97))
)
block_hours = (block_profile.index - 1) / 4

axes[2].fill_between(block_hours, block_profile["Purchase_Bid_MW"].fillna(0).to_numpy(),
                     color="#4f46a3", alpha=0.55, label="Purchase Bid (MW)")
axes[2].fill_between(block_hours, block_profile["Sell_Bid_MW"].fillna(0).to_numpy(),
                     color="gold", alpha=0.55, label="Sell Bid (MW)")
axes[2].plot(block_hours, block_profile["MCV_MW"], color="tomato", lw=1.4, label="MCV (MW)")
axes[2].set_title("Average RTM Market Snapshot by 15-Minute Block (traded volumes)")
axes[2].set_ylabel("MW")
# axes[2].set_xlim(0, 24)
# axes[2].set_xticks(range(0, 25, 1))
# Hourly ticks with HH:MM labels
hour_ticks = np.arange(0, 25, 1)

axes[2].set_xlim(0, 24)
axes[2].set_xticks(hour_ticks)
axes[2].set_xticklabels([f"{int(h):02d}:00" for h in hour_ticks])

axes[2].set_xlabel("Time")
axes[2].grid(alpha=0.3)

axes[3].plot(score_by_block.index, score_by_block.values, color="seagreen", marker="o", ms=3)
for b in best["Blocks"]:
    axes[2].axvspan((b - 1) / 4, b / 4, color="gold", alpha=0.18)
    axes[3].axvspan(b - 0.5, b + 0.5, color="gold", alpha=0.35)

market_lines, market_labels = axes[2].get_legend_handles_labels()
axes[2].legend(
    market_lines, market_labels,
    loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9,
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
# ## SECTION 26 – Single-day diagnostic chart

# ==========================================================================
# [cell 40]
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
    # tick_every = 8
    # tick_pos   = list(range(0, len(blocks), tick_every))
    # tick_labs  = [times_lbl[i] for i in tick_pos]
    # Hourly ticks (every 4 blocks = 1 hour)
    tick_every = 4
    tick_pos   = list(range(0, len(blocks), tick_every))
    tick_labs  = [times_lbl[i] for i in tick_pos]
    for ax in axes:
        ax.tick_params(axis="x", labelbottom=True)

        plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right", fontsize=8)
        plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right", fontsize=8)
        plt.setp(axes[2].get_xticklabels(), rotation=45, ha="right", fontsize=8)
    # Add the last tick (23:45) if not already included
    if tick_pos[-1] != len(blocks) - 1:
        tick_pos.append(len(blocks) - 1)
        tick_labs.append(times_lbl[-1])
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
    axes[1].set_title("Forecast RTM Price — v3")

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


# --------------------------------------------------------------------------
# ## SECTION 27 — Publication figures (Paper_Figures/)
#
# Incorporates `generate_paper_figures.py`. Paths and the target month are taken from the notebook's own variables (`ARTIFACT_DIR`, `PROJECT_DIR`, `PEAK_MONTH`), so this runs unchanged on macOS or Windows and writes PNGs to `<project>/Paper_Figures/`. The shaded peak window is read from the pipeline's actual declared window, not a fixed block range.

# ==========================================================================
# [cell 42]
# ==========================================================================
# ============================================================================
# SECTION 27 — Publication figures  (portable: derives every path + the target
# month from the notebook's own variables, so it runs unchanged on the MacBook
# and the Windows PC and writes to <project>/Paper_Figures/).
#
# Incorporated from generate_paper_figures.py. Changes vs. the standalone
# script: (1) hard-coded C:\Users\chintan\... paths replaced by ARTIFACT_DIR /
# PROJECT_DIR; (2) the "2026-09" month and "2026-09-01" plot-day now come from
# PEAK_MONTH; (3) the shaded peak window is read from the pipeline's actual
# declared window (Peak_Hours) instead of a fixed block range; (4) each figure
# is guarded so one missing input can't abort the rest of the run.
# ============================================================================
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Paths & month, all derived from the notebook (no machine-specific paths) ──
FIG_DATA = ARTIFACT_DIR                      # was DATA = C:\...\monthly_peak_pipeline_outputs
FIG_ROOT = PROJECT_DIR                       # was ROOT = C:\...\NRLDC_Project
FIG_OUT  = PROJECT_DIR / "Paper_Figures"     # was OUT  = C:\...\Paper_Figures
FIG_OUT.mkdir(parents=True, exist_ok=True)

FIG_MONTH = PEAK_MONTH                        # e.g. "2026-09"
FIG_PLOT_DAY = f"{PEAK_MONTH}-01"             # representative single day for fig3/fig8

# Publication rcParams, applied only inside a context so the rest of the
# notebook's plotting defaults are left untouched.
FIG_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.7, "grid.linewidth": 0.4, "lines.linewidth": 1.1,
    "figure.dpi": 300, "savefig.dpi": 300, "axes.grid": True, "grid.alpha": 0.35,
}

COL_NL, COL_RTM, COL_DAM = "#1f5fa6", "#d9740b", "#5b5b5b"
COL_WIN, COL_ACC = "#e07a5f", "#3b7d4f"
SINGLE_W, DOUBLE_W = 3.45, 7.16


def _savefig(fig, name):
    path = FIG_OUT / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)                            # close so it doesn't also render inline
    print("saved", path)


def _time_to_block(hhmm):
    """'18:15' -> 15-min block index (1..96), matching block_to_time()."""
    h, m = map(int, hhmm.strip().split(":"))
    return (h * 60 + m) // 15 + 1


def selected_peak_blocks(default=range(74, 86)):
    """Blocks inside the declared peak window.

    Priority: the live `best` result from the scoring section -> the
    Monthly_Peak_Hours CSV's Peak_Hours string -> the historical default
    (18:15-21:15). Parsing the 'HH:MM-HH:MM[, HH:MM-HH:MM]' string keeps this
    correct even when the pipeline declares a different or split window.
    """
    peak_hours_str = None
    try:
        peak_hours_str = best["Peak_Hours"]          # set in the scoring cell
    except Exception:
        try:
            _mp = pd.read_csv(FIG_DATA / f"Monthly_Peak_Hours_{FIG_MONTH}.csv")
            peak_hours_str = str(_mp["Peak_Hours"].iloc[0])
        except Exception:
            peak_hours_str = None

    if not peak_hours_str or peak_hours_str.lower() == "nan":
        return set(default)

    blocks = set()
    for rng in peak_hours_str.split(","):
        rng = rng.strip()
        if "-" not in rng:
            continue
        a, b = rng.split("-")
        blocks.update(range(_time_to_block(a), _time_to_block(b)))
    return blocks or set(default)


# ---------------------------------------------------------------- Fig 1
def fig1_framework():
    fig, ax = plt.subplots(figsize=(SINGLE_W, 7.6))
    ax.set_xlim(0, 6.2); ax.set_ylim(2.95, 15.25); ax.axis("off")

    def box(x, y, w, h, text, fc="#eef3f8", ec="#1f5fa6", fs=6.7, weight="normal"):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.07",
                           linewidth=0.9, edgecolor=ec, facecolor=fc)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, weight=weight, linespacing=1.3)
        return (x, y, w, h)

    def arrow(b1, b2, side1="bottom", side2="top", text=None, style="-|>",
              color="#333333", tside="right"):
        x1, y1, w1, h1 = b1; x2, y2, w2, h2 = b2
        pts = {"bottom": (x1 + w1 / 2, y1), "top": (x1 + w1 / 2, y1 + h1),
               "left": (x1, y1 + h1 / 2), "right": (x1 + w1, y1 + h1 / 2)}
        pts2 = {"bottom": (x2 + w2 / 2, y2), "top": (x2 + w2 / 2, y2 + h2),
                "left": (x2, y2 + h2 / 2), "right": (x2 + w2, y2 + h2 / 2)}
        a = FancyArrowPatch(pts[side1], pts2[side2], arrowstyle=style, mutation_scale=7,
                            linewidth=0.9, color=color, shrinkA=1, shrinkB=1)
        ax.add_patch(a)
        if text:
            mx, my = (pts[side1][0] + pts2[side2][0]) / 2, (pts[side1][1] + pts2[side2][1]) / 2
            dx = 0.08 if tside == "right" else -0.08
            ax.text(mx + dx, my, text, fontsize=5.8, color=color,
                    ha="left" if tside == "right" else "right", va="center")

    b_net = box(0.15, 14.15, 5.9, 0.85, "Net NR Demand (historical, 15-min)", fc="#eaf1fb")
    b_wx  = box(0.15, 12.95, 5.9, 0.95, "Weather Composite\n(T, RH — 40-station, importance-weighted)", fc="#eaf7ee")
    b_mkt = box(0.15, 11.65, 5.9, 1.05, "Market Signals\n(DAM/RTM price; exchange buy/sell/cleared volumes)", fc="#fbeee3")
    b_nlm = box(0.15, 9.75, 5.9, 1.15, "Net-Load Model\nGradient-boosted regression trees (LightGBM)",
                fc="#dbe8fb", ec="#1f5fa6", weight="bold")
    b_rtm = box(0.15, 7.95, 5.9, 1.45, "RTM Price Model\nClimatology + 5-quantile ensemble, spike/crash\nregimes, conformal band",
                fc="#fbe2cd", ec="#d9740b", weight="bold")
    b_score = box(0.15, 6.15, 5.9, 1.15, "Composite Block Score  S(t)\nrobust z-score fusion;\nNL × RTM interaction dominant (Eq. 4)",
                  fc="#eaf3ea", ec="#3b7d4f", weight="bold")
    b_gate = box(0.15, 4.55, 2.85, 1.05, "Evening-Ramp Gate\n(post afternoon-\ntrough start)", fc="#f5f5f5", ec="#555555", fs=6.3)
    b_agg  = box(3.2, 4.55, 2.85, 1.05, "Monthly\nAggregation\n(recency-weighted)", fc="#f5f5f5", ec="#555555", fs=6.3)
    b_out  = box(0.15, 3.15, 5.9, 0.95, "Declared Peak Window\n(fixed duration; continuous or split)",
                 fc="#fdecec", ec="#c0392b", weight="bold")

    for a, b, s1, s2 in [(b_net, b_nlm, "bottom", "top"), (b_wx, b_nlm, "bottom", "top"),
                         (b_wx, b_rtm, "bottom", "top"), (b_mkt, b_rtm, "bottom", "top"),
                         (b_nlm, b_rtm, "bottom", "top"), (b_nlm, b_score, "bottom", "top"),
                         (b_rtm, b_score, "bottom", "top"), (b_score, b_gate, "bottom", "top"),
                         (b_score, b_agg, "bottom", "top"), (b_gate, b_out, "bottom", "top"),
                         (b_agg, b_out, "bottom", "top")]:
        arrow(a, b, s1, s2)
    ax.text(3.1, 9.63, "forecast net load feeds RTM model", fontsize=5.6,
            color="#1f5fa6", ha="center", va="center", style="italic")
    ax.text(3.1, 6.02, "DAM/RTM volumes also enter the DAM bid/volume term", fontsize=5.4,
            color="#8a5a33", ha="center", va="center", style="italic")
    _savefig(fig, "fig1_framework.png")


# ---------------------------------------------------------------- Fig 2
def fig2_weather_weights():
    df = pd.read_csv(FIG_ROOT / "City_Weather_Feature_Importance.csv")
    df = df.sort_values("Total_Importance", ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(SINGLE_W, 4.0))
    ax.barh(df["City"], df["Total_Importance"], color=COL_NL, height=0.62)
    ax.set_xlabel("Aggregated feature importance (weighting basis)")
    ax.set_ylabel("Weather station")
    ax.set_title("Top 20 of 40 NR load-centre weather stations\nby demand feature-importance weight", fontsize=8)
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    _savefig(fig, "fig2_weather_weights.png")


# ---------------------------------------------------------------- Fig 3
def fig3_rtm_band():
    df = pd.read_csv(FIG_DATA / f"RTM_Forecast_v3_{FIG_MONTH}.csv", parse_dates=["Datetime"])
    day = df[pd.to_datetime(df["Date"]).dt.date == pd.Timestamp(FIG_PLOT_DAY).date()].copy()
    t = day["Datetime"]

    fig, ax1 = plt.subplots(figsize=(SINGLE_W, 2.7))
    ax1.fill_between(t, day["Forecast_RTM_P05"], day["Forecast_RTM_P95"],
                     color=COL_RTM, alpha=0.22, label="P05\u2013P95 conformal band", linewidth=0)
    ax1.plot(t, day["Forecast_RTM_Price"], color=COL_RTM, label="Forecast RTM price (median)")
    ax1.plot(t, day["DAM_Price"], color=COL_DAM, linestyle="--", linewidth=0.9, label="DAM price")
    ax1.set_ylabel("Price (Rs/MWh)"); ax1.set_xlabel(f"Time ({FIG_PLOT_DAY})")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=4))

    ax2 = ax1.twinx()
    ax2.plot(t, day["Forecast_RTM_Spike_Prob"], color="#8e2f2f", linewidth=0.9, alpha=0.85,
             label="Spike-regime probability")
    ax2.set_ylabel("Spike probability", color="#8e2f2f"); ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", colors="#8e2f2f"); ax2.grid(False)

    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper left", fontsize=6, framealpha=0.85)
    ax1.set_title("RTM price forecast with conformal band\nand spike-regime probability", fontsize=8)
    _savefig(fig, "fig3_rtm_band.png")


# ---------------------------------------------------------------- Fig 4
def fig4_cv_metrics():
    df = pd.read_csv(FIG_DATA / f"RTM_v3_WalkForwardCV_{FIG_MONTH}.csv")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(SINGLE_W, 5.1))
    fig.subplots_adjust(hspace=0.55)

    x, w = df["Fold"], 0.35
    ax1.bar(x - w / 2, df["MAE"], width=w, color=COL_NL, label="MAE")
    ax1.bar(x + w / 2, df["RMSE"], width=w, color=COL_RTM, label="RMSE")
    ax1.set_xlabel("Walk-forward fold"); ax1.set_ylabel("Error (Rs/MWh)")
    ax1.set_xticks(x); ax1.set_title("(a) Fold-wise MAE / RMSE", fontsize=8); ax1.legend()

    q_cols = ["Pinball_Q5", "Pinball_Q25", "Pinball_Q50", "Pinball_Q75", "Pinball_Q85"]
    q_labels = ["Q05", "Q25", "Q50", "Q75", "Q85"]
    present = [(c, l) for c, l in zip(q_cols, q_labels) if c in df.columns]
    means = [df[c].mean() for c, _ in present]
    stds  = [df[c].std() for c, _ in present]
    ax2.bar([l for _, l in present], means, yerr=stds, color=COL_ACC, capsize=3)
    ax2.set_xlabel("Quantile"); ax2.set_ylabel("Mean pinball loss (Rs/MWh)")
    ax2.set_title("(b) Pinball loss by quantile\n(mean \u00b1 s.d. across folds)", fontsize=8)
    _savefig(fig, "fig4_cv_metrics.png")


# ---------------------------------------------------------------- Fig 5
def fig5_threshold_sweep():
    df = pd.read_csv(FIG_DATA / f"RTM_v3_CapHitThresholdSweep_Combined_{FIG_MONTH}.csv")
    best_idx = df["F1_mean"].idxmax()
    best_t = df.loc[best_idx, "Threshold"]

    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.7))
    ax.plot(df["Threshold"], df["Precision_cv"], color=COL_NL, marker="o", ms=2.5, label="Precision")
    ax.plot(df["Threshold"], df["Recall_cv"], color=COL_RTM, marker="s", ms=2.5, label="Recall")
    ax.plot(df["Threshold"], df["F1_mean"], color=COL_ACC, marker="^", ms=2.5, label="F1 (mean)")
    ax.axvline(best_t, color="#333333", linestyle=":", linewidth=1.0)
    ax.annotate(f"selected\nthr.={best_t:.2f}", xy=(best_t, df.loc[best_idx, "F1_mean"]),
                xytext=(best_t + 0.07, 0.55), fontsize=6.3,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_xlabel("Cap-hit decision threshold"); ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.set_title("Cap-hit classifier: precision, recall\nand F1 vs. decision threshold", fontsize=8)
    ax.legend(loc="lower left")
    _savefig(fig, "fig5_threshold_sweep.png")


# ---------------------------------------------------------------- Fig 6
def fig6_method_comparison():
    df = pd.read_csv(FIG_DATA / f"Peak_Method_Comparison_{FIG_MONTH}.csv").sort_values("Score")
    colors = [COL_ACC if m == "recency_weighted" else "#9fb4c7" for m in df["Method"]]
    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.9))
    bars = ax.barh(df["Method"].str.replace("_", " "), df["Score"], color=colors)
    for b, hrs in zip(bars, df["Peak_Hours"]):
        ax.text(b.get_width() + 0.4, b.get_y() + b.get_height() / 2, hrs,
                va="center", fontsize=6.2, color="#333333")
    ax.set_xlabel("Aggregated window score")
    ax.set_title(f"Declared window by aggregation method\n({FIG_MONTH}; adopted method in green)", fontsize=8)
    _savefig(fig, "fig6_method_comparison.png")


# ---------------------------------------------------------------- Fig 7
def fig7_monthly_overview():
    df = pd.read_csv(FIG_DATA / f"RTM_Forecast_v3_{FIG_MONTH}.csv", parse_dates=["Datetime"])
    sel_blocks = selected_peak_blocks()
    df["in_window"] = df["Block"].isin(sel_blocks)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(SINGLE_W, 4.3), sharex=True)
    ax1.plot(df["Datetime"], df["Net_Load"], color=COL_NL, linewidth=0.9)
    ax2.plot(df["Datetime"], df["Forecast_RTM_Price"], color=COL_RTM, linewidth=0.8)
    for _, g in df[df["in_window"]].groupby(df["Datetime"].dt.date):
        t0, t1 = g["Datetime"].min(), g["Datetime"].max()
        ax1.axvspan(t0, t1, color=COL_WIN, alpha=0.25, linewidth=0)
        ax2.axvspan(t0, t1, color=COL_WIN, alpha=0.25, linewidth=0)

    win_label = ", ".join(sorted({f"{block_to_time(min(sel_blocks))}\u2013{block_to_time(max(sel_blocks) + 1)}"})) \
        if sel_blocks else "n/a"
    ax1.set_ylabel("Net NR load (MW)")
    ax1.set_title(f"Forecast net load and RTM price \u2014 target month {FIG_MONTH}\n"
                  f"(shaded: declared peak window, {win_label} daily)", fontsize=8)
    ax2.set_ylabel("RTM price (Rs/MWh)"); ax2.set_xlabel("Date")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    fig.autofmt_xdate(rotation=0, ha="center")
    _savefig(fig, "fig7_monthly_overview.png")


# ---------------------------------------------------------------- Fig 8
def fig8_day_diagnostic():
    df = pd.read_csv(FIG_DATA / f"RTM_Forecast_v3_{FIG_MONTH}.csv", parse_dates=["Datetime"])
    day = df[pd.to_datetime(df["Date"]).dt.date == pd.Timestamp(FIG_PLOT_DAY).date()].copy()
    day["interaction"] = day["Net_Load"] * day["Forecast_RTM_Price"]
    sel_blocks = selected_peak_blocks()
    win = day[day["Block"].isin(sel_blocks)]
    t0, t1 = win["Datetime"].min(), win["Datetime"].max()
    p90 = day["Net_Load"].quantile(0.90)

    fig, axes = plt.subplots(3, 1, figsize=(SINGLE_W, 5.4), sharex=True)
    ax1, ax2, ax3 = axes
    ax1.plot(day["Datetime"], day["Net_Load"], color=COL_NL)
    ax1.axhline(p90, color=COL_NL, linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.set_ylabel("Net load (MW)")
    ax1.set_title(f"Net load, RTM price and their interaction \u2014 {FIG_PLOT_DAY}\n"
                  "(shaded: selected window)", fontsize=8)
    ax2.plot(day["Datetime"], day["Forecast_RTM_Price"], color=COL_RTM)
    ax2.set_ylabel("RTM price\n(Rs/MWh)")
    ax3.fill_between(day["Datetime"], day["interaction"], color="#8e6fb0", alpha=0.55, linewidth=0)
    ax3.set_ylabel("NL \u00d7 RTM\n(MW\u00b7Rs/MWh)"); ax3.set_xlabel("Time of day")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax3.set_xlim(day["Datetime"].min(), day["Datetime"].max())
    for ax in axes:
        if pd.notna(t0) and pd.notna(t1):
            ax.axvspan(t0, t1, color=COL_WIN, alpha=0.3, linewidth=0)
    _savefig(fig, "fig8_day_diagnostic.png")


# ── Run all figures (one guarded try each so a single failure is non-fatal) ──
_ALL_FIGS = [
    ("fig1_framework", fig1_framework), ("fig2_weather_weights", fig2_weather_weights),
    ("fig3_rtm_band", fig3_rtm_band), ("fig4_cv_metrics", fig4_cv_metrics),
    ("fig5_threshold_sweep", fig5_threshold_sweep), ("fig6_method_comparison", fig6_method_comparison),
    ("fig7_monthly_overview", fig7_monthly_overview), ("fig8_day_diagnostic", fig8_day_diagnostic),
]
print(f"Generating {len(_ALL_FIGS)} publication figures into {FIG_OUT}")
with plt.rc_context(FIG_RC):
    for _name, _fn in _ALL_FIGS:
        try:
            _fn()
        except Exception as _e:
            print(f"  ! {_name} skipped: {type(_e).__name__}: {_e}")
print("Done — figures in", FIG_OUT)


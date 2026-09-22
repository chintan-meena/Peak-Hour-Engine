"""Reference dump of Peak_Hours_Complete_Pipeline_v5_RTM_matched.ipynb -- code cells only, outputs stripped.

Frozen copy for reading while porting; not meant to be executed as-is.
"""

# --------------------------------------------------------------------------
# # Peak Hours Complete Pipeline — v2
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
END_DATE = "2026-05-31"  # last day of month before PEAK_MONTH
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
# ## SECTION 2 – Net-load: raw build, resample, clean

# ==========================================================================
# [cell 6]
# ==========================================================================
RAW_EXPECTED_COLS = {"HRS", "NR Load", "NR Solar", "NR Wind"}


def build_net_load_from_raw(input_folder=DEMAND_FOLDER):
    csv_files = sorted(Path(input_folder).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No raw demand CSV files found in {input_folder}")

    master_df = []
    for file in csv_files:
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

            try:
                file_date = pd.to_datetime(file.stem, format="%d-%m-%Y")
            except Exception:
                print(f"  Skipping {file.name} (date parse failed)")
                continue

            if not (START_TS <= file_date < END_TS):
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
    print(f"\n✔ Parsed {len(csv_files)} files — {len(master_df):,} rows total")
    return master_df


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


weather_cache = ARTIFACT_DIR / "Weather_Delhi_Hourly.csv"
if weather_cache.exists():
    weather = pd.read_csv(weather_cache, parse_dates=["Datetime"])
else:
    weather = download_weather("Delhi", 28.6139, 77.2090, START_DATE, END_DATE)
    weather.to_csv(weather_cache, index=False)

weather_15 = (
    weather.set_index("Datetime")
    .resample("15min").ffill()
    .loc[START_TS:END_TS]
    .reset_index())

net_load_df = pd.merge(net_load_raw, weather_15, on="Datetime", how="left")
weather_cols = ["Delhi_Temp", "Delhi_Humidity"]
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
print(df[["Net_Load", "Delhi_Temp", "Delhi_Humidity"]].describe())


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

    out["temp_humidity"]  = out["Delhi_Temp"] * out["Delhi_Humidity"]
    out["cooling_degree"] = (out["Delhi_Temp"] - 22).clip(lower=0)
    out["heating_degree"] = (18 - out["Delhi_Temp"]).clip(lower=0)
    out["heat_index"]     = out["Delhi_Temp"] + 0.1 * out["Delhi_Humidity"]

    lags = [1, 2, 4, 8, 96, 192, 672, 1344, 364 * 96]
    for lag in lags:
        out[f"lag_{lag}"] = out["Net_Load"].shift(lag)
        out[f"lag_{lag}_missing"] = out[f"lag_{lag}"].isna().astype("int8")
        out[f"lag_{lag}"] = out[f"lag_{lag}"].fillna(0)

    weather_lags = [96, 672]
    for lag in weather_lags:
        out[f"temp_lag_{lag}"]     = out["Delhi_Temp"].shift(lag)
        out[f"humidity_lag_{lag}"] = out["Delhi_Humidity"].shift(lag)

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
    hist = df[["Delhi_Temp", "Delhi_Humidity"]].copy()
    hist["block"]   = hist.index.hour * 4 + hist.index.minute // 15
    profile = hist.groupby("block")[["Delhi_Temp", "Delhi_Humidity"]].median()
    out = pd.DataFrame(index=target_index)
    out["block"]         = out.index.hour * 4 + out.index.minute // 15
    out["Delhi_Temp"]    = out["block"].map(profile["Delhi_Temp"])
    out["Delhi_Humidity"] = out["block"].map(profile["Delhi_Humidity"])
    return out.drop(columns="block").ffill().bfill()


_forecast_day_index = pd.date_range(forecast_start, weather_forecast_cutoff, freq="D")
_day_weights = {
    day: float(np.exp(-DECAY_LAMBDA * i))
    for i, day in enumerate(_forecast_day_index)
}

future_weather_cache = (
    ARTIFACT_DIR / f"Future_Weather_{forecast_start.date()}_to_{forecast_end.date()}.csv"
)
if future_weather_cache.exists():
    future_weather_raw = pd.read_csv(future_weather_cache, parse_dates=["Datetime"])
else:
    try:
        url = (
            "https://historical-forecast-api.open-meteo.com/v1/forecast"
            "?latitude=28.6139&longitude=77.2090"
            "&hourly=temperature_2m,relative_humidity_2m"
            f"&start_date={forecast_start.date()}&end_date={weather_forecast_cutoff.date()}"
            "&timezone=Asia/Kolkata"
        )
        j = requests.get(url, timeout=120).json()
        future_weather_raw = pd.DataFrame({
            "Datetime":      pd.to_datetime(j["hourly"]["time"]),
            "Delhi_Temp":    j["hourly"]["temperature_2m"],
            "Delhi_Humidity": j["hourly"]["relative_humidity_2m"],
        })
    except Exception as exc:
        print("Future weather API unavailable; using historical profile:", exc)
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
future_weather_15.loc[valid_mask, "Delhi_Temp"] = (
    weight_aligned[valid_mask] * actual_aligned.loc[valid_mask, "Delhi_Temp"]
    + (1 - weight_aligned[valid_mask]) * hist_profile_15.loc[valid_mask, "Delhi_Temp"]
)
future_weather_15.loc[valid_mask, "Delhi_Humidity"] = (
    weight_aligned[valid_mask] * actual_aligned.loc[valid_mask, "Delhi_Humidity"]
    + (1 - weight_aligned[valid_mask]) * hist_profile_15.loc[valid_mask, "Delhi_Humidity"]
)
future_weather_15 = future_weather_15.ffill().bfill()
print(f"Blended weather ready: {future_weather_15.index.min()} → {future_weather_15.index.max()}")

future_df = df.copy()
preds = []
times = pd.date_range(forecast_start, forecast_end, freq="15min")
for next_time in times:
    row = pd.DataFrame(index=[next_time])
    row["Net_Load"]      = np.nan
    row["Delhi_Temp"]    = future_weather_15.loc[next_time, "Delhi_Temp"]
    row["Delhi_Humidity"] = future_weather_15.loc[next_time, "Delhi_Humidity"]
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
# ## SECTION 8 – Market data: DAM / RTM load or fetch

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

    out["Market"]   = market_name
    out[price_col]  = out[["N1", "N2", "N3"]].mean(axis=1)
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


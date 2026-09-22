"""Reference dump of Peak_Hours_Complete_Pipeline_v1.ipynb -- code cells only, outputs stripped.

Frozen copy for reading while porting; not meant to be executed as-is.
"""

# --------------------------------------------------------------------------
# # Peak Hours Complete Pipeline — v4
#
# Monthly peak-hour declaration for the Northern Region, published
# `DECLARATION_LEAD_DAYS` before the month it applies to.
#
# **What v4 changes over v3** — all three are evaluation-integrity fixes, not
# new modelling:
#
# 1. **Horizon-matched validation.** v3's walk-forward CV handed test rows real
#    `rtm_lag_1/2/4` and real DAM, so it measured a 15-minute-ahead problem
#    while the pipeline ships a 10-to-40-day-ahead declaration. Folds now carry
#    the same lead gap production has and are scored under the declaration
#    information set. The old number is still printed, labelled, for contrast.
# 2. **Disjoint fit / calibrate / test.** v3 estimated the conformal margin on
#    a slice and then reported coverage on that same slice — circular. Coverage
#    is now measured on a third slice that neither fitting nor margin
#    estimation touched.
# 3. **Net load gets the same treatment.** The recursive-compounding fix v3
#    claimed was applied to RTM only; the net-load model still fed its own
#    output into `lag_1` for 3,744 steps. It now uses an exogenous-only
#    feature set and is backtested by declaration replay.
#
# Plus: city weather weights anchored on measured state demand rather than
# LightGBM split importance, and a value-capture scorecard for scoring declared
# windows against the ex-post optimum.

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

from iex import get_trade_data, get_market_volumes, get_region_volumes

# ── Configuration ─────────────────────────────────────────────────────────────
START_DATE = "2022-07-03"
PEAK_MONTH = "2026-10"       # month to declare peak hours for, YYYY-MM

# [LEAD] The declaration has to be published ahead of the month it applies to,
# so the pipeline's information set stops DECLARATION_LEAD_DAYS before the
# first day of PEAK_MONTH. This is a *regulatory* constraint, not a data gap:
# the regulation requires at least a week's notice, and 10 days is the lead
# actually used in practice here.
#
# Everything downstream now derives from this single number instead of a
# hard-coded END_DATE, so the "how far ahead are we really forecasting?"
# question has exactly one answer in the code. The forecast horizon runs from
# END_DATE+1 to the end of PEAK_MONTH, i.e. DECLARATION_LEAD_DAYS to
# DECLARATION_LEAD_DAYS + 30 days out.
DECLARATION_LEAD_DAYS = 10
REGULATORY_MIN_LEAD_DAYS = 7

_peak_month_start = pd.Period(PEAK_MONTH, freq="M").start_time
END_TS_DATE = _peak_month_start - pd.Timedelta(days=DECLARATION_LEAD_DAYS)
END_DATE = END_TS_DATE.strftime("%Y-%m-%d")

assert DECLARATION_LEAD_DAYS >= REGULATORY_MIN_LEAD_DAYS, (
    f"DECLARATION_LEAD_DAYS={DECLARATION_LEAD_DAYS} is shorter than the "
    f"{REGULATORY_MIN_LEAD_DAYS}-day minimum notice the regulation requires."
)
print(f"Declaration lead: {DECLARATION_LEAD_DAYS} days "
      f"(regulatory minimum {REGULATORY_MIN_LEAD_DAYS}); "
      f"information set ends {END_DATE}, peak month starts "
      f"{_peak_month_start.date()}")

FORECAST_DAYS         = 35
TEST_DAYS             = 92
NL_MIN                = 10_000
NL_MAX                = 100_000
WEATHER_FORECAST_DAYS = 16
REFRESH_MARKET_DATA   = False
HISTORY_TAIL          = 1_344

# ── [BLOCKER-1] Horizon-matched evaluation & modelling ───────────────────────
# The v3 walk-forward CV scored test rows that still carried *actual*
# rtm_lag_1 / lag_2 / lag_4 -- i.e. it measured a 15-minute-ahead task while
# the pipeline actually ships a 10-to-40-day-ahead declaration. The reported
# MAE ~474 Rs/MWh and cap-hit F1 ~0.92 were therefore not achievable at the
# horizon the product runs at (with a 13% cap-hit base rate and strong
# block-to-block autocorrelation, most of that F1 is persistence).
#
# Two things change:
#   1. CV folds now insert the same DECLARATION_LEAD_DAYS gap between train
#      and test that deployment has, and score the test fold under the
#      deployment information set.
#   2. RTM_HORIZON_MODE controls *how* the self-referential features are
#      handled. "frozen" reproduces v3 (train with real lags, overwrite them
#      with climatology at predict time) -- which is honest about the horizon
#      but leaves a train/serve skew, because the model was fitted believing
#      rtm_lag_1 was informative. "exogenous" instead drops those features
#      from the feature set entirely, so training and serving see the same
#      thing. Both are evaluated side by side; "exogenous" is the default
#      because it is the one that is internally consistent.
RTM_HORIZON_MODE      = "exogenous"   # "exogenous" | "frozen"
EVAL_BOTH_HORIZON_MODES = True        # score both in CV for the comparison table
EVAL_REPORT_ONE_STEP  = True          # also report the old 1-block-ahead number,
                                      # clearly labelled, so the gap is visible

# ── [BLOCKER-3] Disjoint fit / calibrate / test split ────────────────────────
# v3 computed the conformal margin on calib_df and then reported empirical
# coverage on that same calib_df -- which returns the target level by
# construction and proves nothing. There is now a third slice, never touched
# by fitting or by margin estimation, that coverage is measured on. CV folds
# are confined to the fit region so they no longer overlap the calibration
# slice either (in v3, fold 5's test set sat entirely inside it).
CONFORMAL_TEST_FRAC = 0.08   # final slice: honest coverage + honest cap-hit metrics
CONFORMAL_CALIB_FRAC = 0.08  # slice before it: conformal margin + threshold choice

# ── Net-load source ──────────────────────────────────────────────────────────
# "auto"  : SCADA parquet cache (scada-cache/), falling back to NR_DEMAND_TEMP
# "scada" : SCADA cache only - raise if it cannot cover the window
# "raw"   : the original NR_DEMAND_TEMP folder scan
NET_LOAD_SOURCE     = "auto"
UPDATE_SCADA_CACHE  = True   # pull new days off the share first; set False to
                             # use the cache exactly as it stands (much faster
                             # when you are only re-running the model)

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

# Which NR state each weather station sits in. Used to spread the measured
# state demand across that state's stations (see the CITY_WEIGHTS cell). The
# state names must match scada_cache's state_demand columns exactly.
CITY_STATE = {
    "Ludhiana": "Punjab", "Amritsar": "Punjab", "Patiala": "Punjab",
    "Jalandhar": "Punjab", "Bathinda": "Punjab",
    "Hisar": "Haryana", "Gurugram": "Haryana", "Faridabad": "Haryana",
    "Ambala": "Haryana", "Rohtak": "Haryana", "Panipat": "Haryana",
    "Jaipur": "Rajasthan", "Jodhpur": "Rajasthan", "Kota": "Rajasthan",
    "Udaipur": "Rajasthan", "Bikaner": "Rajasthan", "Ajmer": "Rajasthan",
    "Sriganganagar": "Rajasthan",
    "Palam": "Delhi", "Safdarjung": "Delhi", "LodiBhawan": "Delhi",
    "Lucknow": "UP", "Agra": "UP", "Kanpur": "UP", "Varanasi": "UP",
    "Prayagraj": "UP", "Noida": "UP", "Gorakhpur": "UP", "Meerut": "UP",
    "Bareilly": "UP",
    "Dehradun": "Uttarakhand", "Haridwar": "Uttarakhand", "Roorkee": "Uttarakhand",
    "Shimla": "Himachal", "Dharamsala": "Himachal", "Mandi": "Himachal",
    "Jammu": "JK", "Srinagar": "JK", "Leh": "JK",
    "Chandigarh": "Chandigarh",
}
assert set(CITY_STATE) == set(CITY_COORDS), "CITY_STATE and CITY_COORDS disagree"

ARTIFACT_DIR = PROJECT_DIR / "monthly_peak_pipeline_outputs"
from powertools_common import cache_dir as _cache_dir
MARKET_CACHE_DIR = _cache_dir("peak_hours")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

START_TS  = pd.Timestamp(START_DATE)
END_TS    = pd.Timestamp(END_DATE) + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
TEST_SIZE = TEST_DAYS * 96

print("Project:", PROJECT_DIR)
print("Net-load window:", START_DATE, "to", END_DATE)
print("Peak month:", PEAK_MONTH)
print("Artifacts:", ARTIFACT_DIR)


# --------------------------------------------------------------------------
# ### City weights, anchored on measured state demand

# ==========================================================================
# [cell 4]
# ==========================================================================
# ============================================================================
# CITY WEIGHTS — anchored on measured state demand, not split importance
# ============================================================================
# [WEIGHTS-FIX] The composite Weighted_Temp / Weighted_Humidity used to be
# weighted by City_Weather_Feature_Importance.csv -- LightGBM split importance
# from the v5 model. On 40 mutually-correlated temperature series that metric
# does not measure "which city drives NR demand"; gain flows to whichever
# series is *least redundant*, which is the mountain stations with a different
# seasonal profile. The result was anti-physical:
#
#     9 hill stations (HP + Uttarakhand + J&K/Ladakh) -> 36% of the weight
#     Himachal's 3 stations  -> 15.7%
#     Delhi's 3 stations     ->  2.8%   (LodiBhawan got 0.0004)
#     Dharamsala and Leh outranked Lucknow, Kanpur, Jaipur and every Delhi station
#
# i.e. the temperature signal fed to a model of Indo-Gangetic-plain demand was
# skewed toward Himalayan climate. Weights are now built from the actual MW
# each state draws, which is the quantity that physically determines how much
# a city's weather matters to NR net load.
#
# State demand -> city weight:
#   1. mean drawal (MW) per state, from scada_cache's `state_demand` source
#   2. normalise across the nine states -> state weight
#   3. split each state's weight equally across its weather stations
#      (no within-state load split is measured, so equal is the honest choice)
STATE_DEMAND_MIN_DAYS = 90     # below this the weights are not trustworthy
STATE_WEIGHT_CSV = PROJECT_DIR / "State_Demand_Weights.csv"   # optional override

CITY_WEIGHT_PROVENANCE = None
_state_mw = None

# ── 1. Preferred source: measured state drawal from the SCADA cache ──────────
try:
    _scada_dir = PROJECT_DIR / "scada-cache"
    if str(_scada_dir) not in sys.path:
        sys.path.insert(0, str(_scada_dir))
    import scada_cache as _sc

    if UPDATE_SCADA_CACHE:
        try:
            _info = _sc.update_source("state_demand", verbose=False)
            if not _info.get("reachable"):
                print("state_demand: share unreachable - using the cache as it stands")
        except Exception as _exc:
            print("state_demand: update failed:", _exc)

    _states = _sc.load("state_demand", start=START_TS, end=END_TS)
    _cov_days = _states.index.normalize().nunique() if len(_states) else 0

    if _cov_days >= STATE_DEMAND_MIN_DAYS:
        _state_mw = _states.mean().rename("Mean_MW")
        CITY_WEIGHT_PROVENANCE = (
            f"scada_cache state_demand, {_cov_days} days "
            f"({_states.index.min().date()} to {_states.index.max().date()})"
        )
    else:
        print(
            "\n" + "!" * 78 +
            f"\n!! state_demand cache covers only {_cov_days} distinct day(s) "
            f"(need >= {STATE_DEMAND_MIN_DAYS}).\n"
            "!! The nine-state drawal series is what the city weights should be\n"
            "!! anchored on. Refresh it ON THE LAN with:\n"
            "!!     cd scada-cache && python update_cache.py --source state_demand\n"
            "!! Until then the fallback below is used and the weights are NOT\n"
            "!! publication-grade.\n" + "!" * 78 + "\n"
        )
        if _cov_days > 0:
            _state_mw = _states.mean().rename("Mean_MW")
            CITY_WEIGHT_PROVENANCE = (
                f"scada_cache state_demand, ONLY {_cov_days} day(s) "
                f"-- PROVISIONAL, refresh the cache before publishing"
            )
except Exception as _exc:
    print("scada_cache state_demand unavailable:", _exc)

# ── 2. Optional manual override: a two-column CSV (State, Mean_MW) ──────────
# Use this to pin the weights to a published figure (e.g. CEA state peak
# demand) rather than to whatever the cache happens to hold. It wins over the
# cache when present, because it is the number you can cite.
if STATE_WEIGHT_CSV.exists():
    _sw = pd.read_csv(STATE_WEIGHT_CSV)
    _state_mw = _sw.set_index("State")["Mean_MW"]
    CITY_WEIGHT_PROVENANCE = f"{STATE_WEIGHT_CSV.name} (manual/published figures)"
    print(f"State weights taken from {STATE_WEIGHT_CSV.name} (overrides the cache)")

# ── 3. Build city weights ───────────────────────────────────────────────────
if _state_mw is not None:
    _state_mw = _state_mw.reindex(sorted(set(CITY_STATE.values())))
    _missing_states = _state_mw[_state_mw.isna()].index.tolist()
    if _missing_states:
        raise KeyError(
            f"No demand series for {_missing_states}. Either add them to "
            "scada_cache.tags.STATE_DEMAND_TAGS or supply State_Demand_Weights.csv."
        )
    _state_w = _state_mw / _state_mw.sum()
    _cities_per_state = pd.Series(CITY_STATE).groupby(pd.Series(CITY_STATE)).size()
    CITY_WEIGHTS = {
        city: float(_state_w[state] / _cities_per_state[state])
        for city, state in CITY_STATE.items()
    }
else:
    # ── 4. Last resort: the old split-importance CSV, used only so the
    #      notebook still runs end-to-end. Do not publish results built on it.
    CITY_WEIGHTS_CSV = PROJECT_DIR / "City_Weather_Feature_Importance.csv"
    _city_wt_df = pd.read_csv(CITY_WEIGHTS_CSV)
    _city_wt_df = _city_wt_df[_city_wt_df["City"].isin(CITY_COORDS)]
    CITY_WEIGHTS = dict(
        zip(_city_wt_df["City"],
            _city_wt_df["Total_Importance"] / _city_wt_df["Total_Importance"].sum())
    )
    CITY_WEIGHT_PROVENANCE = "City_Weather_Feature_Importance.csv (LEGACY, anti-physical)"
    print("\n*** FALLING BACK to split-importance weights. Results are not "
          "publication-grade until state demand is available. ***\n")

_wsum = sum(CITY_WEIGHTS.values())
CITY_WEIGHTS = {c: w / _wsum for c, w in CITY_WEIGHTS.items()}

print(f"City weights: {len(CITY_WEIGHTS)} cities, provenance = {CITY_WEIGHT_PROVENANCE}")

# ── 5. Side-by-side against the legacy weights (a paper table) ──────────────
_legacy_path = PROJECT_DIR / "City_Weather_Feature_Importance.csv"
if _legacy_path.exists():
    _leg = pd.read_csv(_legacy_path)
    _leg = _leg[_leg["City"].isin(CITY_COORDS)]
    _leg["Legacy_Weight"] = _leg["Total_Importance"] / _leg["Total_Importance"].sum()
    _cmp = pd.DataFrame({
        "City": list(CITY_WEIGHTS),
        "State": [CITY_STATE[c] for c in CITY_WEIGHTS],
        "New_Weight": list(CITY_WEIGHTS.values()),
    }).merge(_leg[["City", "Legacy_Weight"]], on="City", how="left")
    _by_state = (
        _cmp.groupby("State")[["New_Weight", "Legacy_Weight"]].sum()
        .sort_values("New_Weight", ascending=False)
    )
    print("\nWeight by state — demand-anchored vs legacy split-importance:")
    print((_by_state * 100).round(1).to_string())
    _hill = ["Himachal", "Uttarakhand", "JK"]
    print(f"\nHill states (HP+UK+JK): new "
          f"{_by_state.loc[_hill, 'New_Weight'].sum()*100:.1f}% vs legacy "
          f"{_by_state.loc[_hill, 'Legacy_Weight'].sum()*100:.1f}%")
    _cmp.to_csv(ARTIFACT_DIR / "City_Weight_Comparison.csv", index=False)


# --------------------------------------------------------------------------
# ## SECTION 1 – Utility helpers

# ==========================================================================
# [cell 6]
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
# [cell 8]
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
# [cell 10]
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
# [cell 12]
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
# [cell 14]
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
# [cell 16]
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
# ## SECTION 6b — Net load at the declaration horizon (backtest + production model)

# ==========================================================================
# [cell 18]
# ==========================================================================
# ============================================================================
# SECTION 6b — [BLOCKER-1] Net load at the declaration horizon
# ============================================================================
# The net-load model has exactly the same problem the RTM model had, and v3
# never fixed it. The headline MAE 840 MW / R2 0.989 came from a single
# 92-day split in which every test row carried actual lag_1, lag_2, lag_4 --
# a 15-minute-ahead task. The pipeline then fed that model's own output back
# into those same lags for 3,744 consecutive steps to cover the target month.
# The v3 markdown claims the recursive-compounding bug was fixed; it was fixed
# for RTM only.
#
# Self-referential features are dropped here rather than frozen, for the same
# reason as RTM_HORIZON_MODE="exogenous": freezing leaves the model relying on
# a feature that is informative in training and meaningless in production.
# What survives is genuinely knowable DECLARATION_LEAD_DAYS ahead:
#   * calendar and cyclic terms
#   * the weather forecast / climatology blend
#   * lag_34944 (same block ~one year back) and its annual rolling stats,
#     evaluated as of the forecast origin
#
# Everything shorter -- lag_1 .. lag_1344 and the 1-day/7-day/14-day rolling
# windows -- would require target-month net load, which does not exist yet.

NL_SELF_REF_LAGS  = [1, 2, 4, 8, 96, 192, 672, 1344]
NL_LONG_WINDOW    = 364 * 96          # the annual window; knowable at the origin
NL_SELF_REF_ROLLS = [96, 672, 1344]

NL_SELF_REF_COLS = (
    [f"lag_{l}" for l in NL_SELF_REF_LAGS]
    + [f"lag_{l}_missing" for l in NL_SELF_REF_LAGS]
    + [f"rolling_mean_{w}" for w in NL_SELF_REF_ROLLS]
    + [f"rolling_std_{w}" for w in NL_SELF_REF_ROLLS]
    + [f"rolling_mean_{w}_missing" for w in NL_SELF_REF_ROLLS]
    + [f"rolling_std_{w}_missing" for w in NL_SELF_REF_ROLLS]
)

NL_ORIGIN_ANCHORED = [f"rolling_mean_{NL_LONG_WINDOW}", f"rolling_std_{NL_LONG_WINDOW}"]

nl_horizon_features = [c for c in net_features if c not in set(NL_SELF_REF_COLS)]
print(f"Net-load features — operational: {len(net_features)}, "
      f"declaration horizon: {len(nl_horizon_features)}")


def anchor_long_windows(feat_frame, origin_values):
    """Hold the annual rolling statistics at their origin value across the
    whole horizon. Over a 364-day window they barely move, and the origin
    value is the last one actually computable when the declaration is made."""
    out = feat_frame.copy()
    for col, val in origin_values.items():
        if col in out.columns:
            out[col] = val
    return out


def NetLoadHorizonModel():
    return NetLoadModel()


# ── Declaration-replay backtest over the last NL_BACKTEST_MONTHS months ─────
# For each origin: fit on everything up to (month_start - lead), predict the
# whole month with no target-month net load available, compare to actuals.
NL_BACKTEST_MONTHS = 6

_nl_backtest_rows, _nl_daily_rows = [], []
_months = pd.period_range(
    end=pd.Period(END_DATE, freq="M") - 1, periods=NL_BACKTEST_MONTHS, freq="M"
)

for _pm in _months:
    _m_start, _m_end = _pm.start_time, _pm.end_time.floor("15min")
    _origin = _m_start - pd.Timedelta(days=DECLARATION_LEAD_DAYS)

    _tr = model_df.loc[:_origin]
    _te = model_df.loc[_m_start:_m_end]
    if len(_te) == 0 or len(_tr) < 96 * 400:
        continue

    _origin_vals = {c: float(_tr[c].iloc[-1]) for c in NL_ORIGIN_ANCHORED if c in _tr.columns}
    _te_h = anchor_long_windows(_te, _origin_vals)

    _m = NetLoadHorizonModel()
    _m.fit(_tr[nl_horizon_features], _tr["Net_Load"])
    _pred_h = _m.predict(_te_h[nl_horizon_features])

    # The v3-style comparison: same origin, but the test rows keep their real
    # short lags. Not achievable in production; shown to size the gap.
    _m_op = NetLoadModel()
    _m_op.fit(_tr[net_features], _tr["Net_Load"])
    _pred_op = _m_op.predict(_te[net_features])

    _y = _te["Net_Load"].to_numpy()
    for _tag, _p in (("declaration horizon", _pred_h), ("1-block-ahead (not shipped)", _pred_op)):
        _nl_backtest_rows.append({
            "Month": str(_pm), "Evaluation": _tag,
            "MAE": mean_absolute_error(_y, _p),
            "RMSE": float(np.sqrt(mean_squared_error(_y, _p))),
            "MAPE_%": float(np.mean(np.abs((_y - _p) / _y)) * 100),
            "R2": r2_score(_y, _p),
            "Peak_Block_Err_MW": float(abs(_y.argmax() - _p.argmax())),
        })

    _d = pd.DataFrame({
        "Datetime": _te.index, "y": _y, "pred": _pred_h,
    })
    _d["Horizon_Day"] = (_d["Datetime"].dt.normalize()
                         - _origin.normalize()).dt.days
    _nl_daily_rows.append(_d)

nl_backtest = pd.DataFrame(_nl_backtest_rows)
if not nl_backtest.empty:
    print("\n" + "=" * 72)
    print(f"Net-load declaration-replay backtest ({len(_months)} months, "
          f"{DECLARATION_LEAD_DAYS}-day lead)")
    print("=" * 72)
    print(nl_backtest.groupby("Evaluation")[["MAE", "RMSE", "MAPE_%", "R2"]]
          .mean().round(3).to_string())
    nl_backtest.to_csv(ARTIFACT_DIR / f"NetLoad_Backtest_{PEAK_MONTH}.csv", index=False)

    _dd = pd.concat(_nl_daily_rows, ignore_index=True)
    _dd["abs_err"] = (_dd["y"] - _dd["pred"]).abs()
    _by_h = _dd.groupby("Horizon_Day")["abs_err"].mean()
    print("\nMAE by days ahead (declaration horizon):")
    print(_by_h.reindex(range(int(_by_h.index.min()), int(_by_h.index.max()) + 1))
          .round(0).to_string())
    _by_h.to_csv(ARTIFACT_DIR / f"NetLoad_MAE_By_Horizon_{PEAK_MONTH}.csv")

# ── Production net-load model for the declaration ──────────────────────────
NL_ORIGIN_VALUES = {
    c: float(model_df[c].iloc[-1]) for c in NL_ORIGIN_ANCHORED if c in model_df.columns
}
net_load_horizon_model = NetLoadHorizonModel()
net_load_horizon_model.fit(X[nl_horizon_features], y)
print(f"\nDeclaration-horizon net-load model trained on {len(X):,} rows.")


# --------------------------------------------------------------------------
# ## SECTION 7 – Net-load forecast for PEAK_MONTH

# ==========================================================================
# [cell 20]
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

# ── [BLOCKER-1] Direct forecast, no recursion ──────────────────────────────
# v3 looped block by block for 3,744 steps, feeding each prediction straight
# back into lag_1/lag_2/lag_4 of the next one. Over a month that is pure
# compounding: by the end of the horizon the "lags" are entirely the model's
# own drift. The declaration-horizon model does not use those features at
# all, so the whole month is predicted in one pass from weather, calendar and
# the same-block-last-year lag.
times = pd.date_range(forecast_start, forecast_end, freq="15min")

_future_rows = pd.DataFrame(index=times)
_future_rows["Net_Load"] = np.nan
_future_rows["Weighted_Temp"] = future_weather_15["Weighted_Temp"].reindex(times).to_numpy()
_future_rows["Weighted_Humidity"] = future_weather_15["Weighted_Humidity"].reindex(times).to_numpy()

_nl_cols = ["Net_Load", "Weighted_Temp", "Weighted_Humidity"]
_nl_combined = pd.concat([df[_nl_cols], _future_rows[_nl_cols]])
_nl_feat_all, _ = add_net_load_features(_nl_combined)
_nl_future = anchor_long_windows(_nl_feat_all.loc[times], NL_ORIGIN_VALUES)

preds = net_load_horizon_model.predict(_nl_future[nl_horizon_features])

net_forecast = pd.DataFrame({"Datetime": times, "Forecast_Net_Load": preds})
net_forecast.to_csv(ARTIFACT_DIR / "Net_Load_Forecast.csv", index=False)
print(net_forecast["Datetime"].min(), "to", net_forecast["Datetime"].max(), len(net_forecast))
print(f"Horizon: {DECLARATION_LEAD_DAYS} to "
      f"{(times.max().normalize() - pd.Timestamp(END_DATE)).days} days ahead, "
      "predicted directly (no recursive feedback).")


# --------------------------------------------------------------------------
# ## SECTION 15 – Market data: DAM / RTM load or fetch

# ==========================================================================
# [cell 22]
# ==========================================================================
dam_path = MARKET_CACHE_DIR / "DAM_Prices_2022_2026.csv"
rtm_path = MARKET_CACHE_DIR / "RTM_Prices_2022_2026.csv"
# [VOLUMES] The three traded-volume measures the IEX payload actually
# carries. Two columns that used to be listed here are gone for good:
#   Final_Scheduled_Volume_MW -- no scheduled-volume field exists anywhere in
#     the API payload, so this was never anything but NaN.
#   MCP_Rs_MWh -- the All-India clearing price, which is byte-identical to the
#     NR RTM price in 97.3% of blocks (r = 0.992). It IS the target; carrying
#     it as a same-block feature would be straight target leakage.
#
# [SCOPE] These three are ALL-INDIA numbers, and nothing in the column names
# says so. The IEX payload carries bid quantities only at national level --
# there is no NR purchase/sell bid anywhere in it -- so a feature built from
# Purchase_Bid_MW is a national demand signal sitting next to N1/N2/N3 prices
# that are regional. That mismatch is fine as long as it is deliberate; it was
# not, so it is now named.
ALL_INDIA_BID_COLUMNS = [
    "Purchase_Bid_MW",
    "Sell_Bid_MW",
    "MCV_MW",
]

# [NR-VOLUMES] What the payload does carry per area is what CLEARED there.
# Across every cached DAM and RTM block from Jul 2022 to Aug 2026 the 13 area
# Buy_Volume figures sum exactly to the All-India Cleared_Volume, as do the 13
# Sell_Volume figures -- they are the settled buy and sell legs of the
# clearing broken out by area, not bid stacks. get_trade_data() kept only the
# area *price* and dropped these on the floor; they are now carried through.
# The Cleared_ prefix is what keeps them from being read as bids.
NR_VOLUME_COLUMNS = [
    "NR_Cleared_Buy_MW",     # N1+N2+N3 cleared purchase
    "NR_Cleared_Sell_MW",    # N1+N2+N3 cleared sale
    "NR_Net_Buy_MW",         # buy - sell: NR's net exchange position
    "NR_Price_Spread",       # max-min area price, > 0 only under NR congestion
]

# The per-area detail behind the NR totals. Carried in the CSV but kept out of
# the model factor list below: N1/N2/N3 only diverge under congestion, which
# NR_Price_Spread already flags, so the six columns are mostly a redundant
# decomposition of the two NR totals.
NR_AREA_VOLUME_COLUMNS = [
    f"{area}_Cleared_{side}_MW"
    for area in ("N1", "N2", "N3")
    for side in ("Buy", "Sell")
]

MARKET_SNAPSHOT_COLUMNS = set(ALL_INDIA_BID_COLUMNS) | set(NR_VOLUME_COLUMNS)


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
        volumes[["Date", "Block"] + ALL_INDIA_BID_COLUMNS]
        .rename(columns={"Date": "_join_date"}),
        on=["_join_date", "Block"], how="left",
    )

    # [NR-VOLUMES] Same raw payload, area nodes instead of the All-India one,
    # so this costs no extra network round-trip either. NR_Avg_Area_Price is
    # deliberately NOT merged: it is mean(N1, N2, N3), i.e. byte-identical to
    # the {market}_Price column this frame already carries (verified to 1e-12
    # over 17k blocks). Carrying it would hand the RTM model a second copy of
    # its own target under a different name.
    nr_volumes = get_region_volumes(
        _iex_client_date(start), _iex_client_date(end), market_name,
        areas=["N1", "N2", "N3"], prefix="NR",
    )
    nr_volumes["Block"] = pd.to_numeric(nr_volumes["Block"], errors="coerce").astype("Int64")
    _nr_cols = [c for c in NR_VOLUME_COLUMNS + NR_AREA_VOLUME_COLUMNS
                if c in nr_volumes.columns]
    combined = combined.merge(
        nr_volumes[["Date", "Block"] + _nr_cols].rename(columns={"Date": "_join_date"}),
        on=["_join_date", "Block"], how="left",
    )

    combined = combined.drop(columns="_join_date")

    _vol_missing = combined["MCV_MW"].isna().mean()
    _nr_missing  = combined["NR_Cleared_Buy_MW"].isna().mean() if _nr_cols else 1.0
    print(f"  volumes merged: {1 - _vol_missing:.1%} of rows have All-India bid data, "
          f"{1 - _nr_missing:.1%} have NR cleared volumes ({len(_nr_cols)} columns)")

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
# ## SECTION 16 — RTM: "Hurdle-Quantile Ensemble with Climatology Decomposition"
#
# RTM prices behave like a **three-regime mixture**: a normal band that tracks
# DAM, a low-tail "crash" regime (near-zero prices during oversupply / mild
# weather / high renewable output), and a high-tail "spike" regime (prices at
# the regulatory cap during scarcity). The model is:
#
# 1. **Target decomposition**: `RTM_Price = seasonal_climatology(block, day_type,
#    season) + deviation`, climatology computed leak-free (expanding,
#    `shift(1)` per group).
# 2. **Distributional deviation model**: 5 LightGBM quantile regressors at
#    `RTM_QUANTILES` (0.05 / 0.25 / 0.50 / 0.75 / **0.85**), monotonic-rearranged.
#    Note the top quantile is 0.85, not 0.95 — v3's comments called the band
#    "P05/P95" throughout, which was wrong. Output columns are now
#    `Forecast_RTM_Lo` / `Forecast_RTM_Hi` with the nominal levels stated.
# 3. **Learned regime probabilities**: `P(spike)` / `P(crash)` classifiers give a
#    continuous, learned mixing weight in place of fixed Block-range weights.
# 4. **Conformal calibration**: split-conformal adjustment of the band, with
#    coverage measured on a slice held out from both fitting and margin
#    estimation.
# 5. **Declaration information set**: every self-referential RTM feature is
#    unknowable at publication time. `RTM_HORIZON_MODE="exogenous"` drops them
#    from the feature set so training and serving match; `"frozen"` keeps v3's
#    overwrite-at-predict-time behaviour. Both are scored side by side.
# 6. **Declaration-replay validation**: 5-fold walk-forward with a
#    `DECLARATION_LEAD_DAYS` gap and month-long test windows, plus two
#    baselines on the same folds.
#
# *Domain note: CERC introduced DAM market coupling from January 2026, and
# 2024–2025 data shows increasingly frequent price crashes (not just spikes)
# from renewable oversupply — so this design clips historical prices to the
# current regulatory cap (`RTM_REGULATORY_CAP = 10000`) rather than trusting
# whatever cap applied when older rows were recorded, and sample weights are
# recency-decayed. The magnitude term in those weights tilts the training
# distribution, so the fitted quantiles are quantiles of a reweighted
# distribution — worth stating explicitly alongside the pinball losses.*

# ==========================================================================
# [cell 24]
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
# [NR-VOLUMES] The first three are All-India, the last four are NR (see the
# scope note on ALL_INDIA_BID_COLUMNS / NR_VOLUME_COLUMNS above). Until now
# only the national ones existed here, so every market feature in the model
# described the whole country while the target -- an NR peak-hour declaration
# -- is regional. The NR block is the region's own position in the same
# auction: what it cleared on each side, its net exchange position, and
# whether N1/N2/N3 split on price.
#
# Everything downstream is generic over this list: the dam_ mirror, the ramps,
# the 5 lags, the 2 rolling means, the leak filter that excludes same-block
# RTM-side volumes, and the climatology fill in build_future_market_frame()
# all pick the new columns up automatically. The NR volumes are cleared by the
# same auction that sets RTM_Price, so the leak filter correctly admits only
# their lags and rolling means on the RTM side, while the DAM-side NR columns
# stay usable same-block (DAM clears the day before delivery).
MARKET_FACTOR_COLS = [
    "Purchase_Bid_MW",       # ALL-INDIA
    "Sell_Bid_MW",           # ALL-INDIA
    "MCV_MW",                # ALL-INDIA
    "NR_Cleared_Buy_MW",     # NR
    "NR_Cleared_Sell_MW",    # NR
    "NR_Net_Buy_MW",         # NR
    "NR_Price_Spread",       # NR
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


def add_market_pressure_features(frame, purchase_col, sell_col, mcv_col, out_prefix="",
                                 nr_buy_col=None, nr_sell_col=None):
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

    # [NR-VOLUMES] Two ratios the All-India columns structurally cannot
    # express. NR_Share_of_Cleared is how much of the national clearing landed
    # in NR; NR_Buy_Sell_Ratio is how one-sided the region's own position was.
    # A block where NR takes 60% of everything cleared nationally and is a net
    # buyer is a tight block *for NR* regardless of what the national bid
    # spread looks like -- which is exactly the distinction a regional
    # peak-hour declaration turns on.
    if nr_buy_col and nr_sell_col:
        for col in [nr_buy_col, nr_sell_col]:
            if col not in out.columns:
                out[col] = np.nan
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out[f"{pfx}NR_Share_of_Cleared"] = out[nr_buy_col] / (out[mcv_col].abs() + eps)
        out[f"{pfx}NR_Buy_Sell_Ratio"]   = out[nr_buy_col] / (out[nr_sell_col].abs() + eps)
    return out


def add_all_market_pressure_features(frame):
    out = add_market_pressure_features(
        frame, "Purchase_Bid_MW", "Sell_Bid_MW", "MCV_MW", out_prefix="",
        nr_buy_col="NR_Cleared_Buy_MW", nr_sell_col="NR_Cleared_Sell_MW",
    )
    out = add_market_pressure_features(
        out, "dam_Purchase_Bid_MW", "dam_Sell_Bid_MW", "dam_MCV_MW", out_prefix="dam",
        nr_buy_col="dam_NR_Cleared_Buy_MW", nr_sell_col="dam_NR_Cleared_Sell_MW",
    )
    return out


RTM_MARKET_DERIVED_COLS = [
    "Bid_Spread_MW", "Bid_Ratio", "Cleared_Share_of_Purchase",
    "Cleared_Share_of_Sell",
    "NR_Share_of_Cleared", "NR_Buy_Sell_Ratio",
]
DAM_MARKET_DERIVED_COLS = [f"dam_{col}" for col in RTM_MARKET_DERIVED_COLS]
ALL_MARKET_RAW_COLS     = RTM_MARKET_FACTOR_COLS + DAM_MARKET_FACTOR_COLS
ALL_MARKET_DERIVED_COLS = RTM_MARKET_DERIVED_COLS + DAM_MARKET_DERIVED_COLS

print(f"Loaded {len(IN_HOLIDAYS)} India holiday dates spanning {list(_hol_years)}")
print("Market frame:", market.shape, market["Datetime"].min(), "→", market["Datetime"].max())


# --------------------------------------------------------------------------
# ### Feature engineering — calendar, DAM↔RTM spread, volatility, real-time tightness

# ==========================================================================
# [cell 26]
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
# [cell 28]
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
# [cell 30]
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


# [v4] Extended downward and made finer. On the first v4 run the F1 optimum
# landed on the old grid's bottom edge (0.05), which means the grid was
# clipping the search: at the declaration horizon the classifier is weak
# enough that F1 still wants a lower threshold than 0.05. A boundary
# optimum is not a chosen threshold, it is a truncated one.
CAP_HIT_THRESHOLD_GRID = np.round(np.arange(0.01, 0.96, 0.02), 3)


X_cols = rtm_features


# --------------------------------------------------------------------------
# ### The declaration information set — what is knowable when the window is published

# ==========================================================================
# [cell 32]
# ==========================================================================
# ============================================================================
# SECTION 19b — [BLOCKER-1] The declaration information set
# ============================================================================
# Everything in this cell exists to answer one question honestly: what does
# the model actually know when the declaration is published?
#
# It is published DECLARATION_LEAD_DAYS before the month starts, so for every
# block of the target month:
#   * no RTM price from the target month is known  -> every rtm_lag_* /
#     rtm_roll_* is unknowable
#   * no DAM clearing from the target month is known -> DAM_Price and all
#     dam_* lags/rolls are unknowable
#   * no traded volume from the target month is known
#   * calendar, weather-driven Net_Load forecast, and history *up to the
#     origin* are known
#
# v3's walk-forward CV ignored all of that: test rows carried real lagged
# prices, so it measured a 15-minute-ahead problem. The helpers below rebuild
# a feature frame under the real information set, and are used by both the CV
# and the production forecast so the two can never drift apart again.

SELF_REF_LAGS  = [1, 2, 4, 96, 672]
SELF_REF_ROLLS = [96, 672]

RTM_SELF_REF_COLS = (
    [f"rtm_lag_{l}" for l in SELF_REF_LAGS]
    + [f"spread_lag_{l}" for l in SELF_REF_LAGS]
    + [f"rtm_roll_mean_{w}" for w in SELF_REF_ROLLS]
    + [f"rtm_roll_std_{w}" for w in SELF_REF_ROLLS]
    + ["night_rtm_lag1"]
)


def horizon_feature_set(mode):
    """Feature list for a given horizon mode.

    "frozen"    — keep the self-referential RTM features and overwrite them
                  with climatology at predict time (v3's approach). Honest
                  about the horizon, but leaves a train/serve skew: the model
                  was fitted believing rtm_lag_1 carried real information.
    "exogenous" — drop them from the feature set entirely, so training and
                  serving see exactly the same inputs. Internally consistent,
                  and the default.
    """
    if mode == "exogenous":
        drop = set(RTM_SELF_REF_COLS)
        return [c for c in rtm_features if c not in drop]
    if mode == "frozen":
        return list(rtm_features)
    raise ValueError(f"Unknown horizon mode {mode!r}")


def build_declaration_climatology(fit_frame, profile_days=30):
    """Everything the freeze/blind helpers need, computed from `fit_frame`
    ONLY — so a CV fold never reaches past its own training data."""
    cutoff = fit_frame["Datetime"].max() - pd.Timedelta(days=profile_days)
    recent = fit_frame[fit_frame["Datetime"] >= cutoff]
    if recent.empty:
        recent = fit_frame

    blind_cols = ["DAM_Price"] + ALL_MARKET_RAW_COLS + ALL_MARKET_DERIVED_COLS
    blind_cols = [c for c in blind_cols if c in fit_frame.columns]

    return {
        "std_profile":    fit_frame.groupby("block")["RTM_Price"].std().to_dict(),
        "std_fallback":   float(fit_frame["RTM_Price"].std()),
        "spread_profile": fit_frame.groupby("block")["DAM_RTM_Spread"].median().to_dict(),
        "block_profile":  recent.groupby("block")[blind_cols].median(),
        "global_median":  fit_frame[blind_cols].median(),
        "blind_cols":     blind_cols,
    }


def _profile_series(clim, col, blocks, shift_blocks=0):
    """Value of `col`'s block profile at (block - shift_blocks), wrapping
    around midnight. shift_blocks=0 gives the block's own profile value."""
    prof = clim["block_profile"][col]
    idx = ((np.asarray(blocks) - 1 - shift_blocks) % 96) + 1
    vals = pd.Series(idx).map(prof).to_numpy(dtype=float)
    return np.where(np.isnan(vals), float(clim["global_median"][col]), vals)


def freeze_self_referential(feat, clim):
    """Overwrite every RTM feature that would need target-month prices with
    the leak-free climatology. Mirrors what v3 did inside the recursive loop,
    but vectorised and reusable so CV and production share one code path."""
    out = feat.copy()
    frozen_val    = out["seasonal_profile"].astype(float).to_numpy()
    blocks        = out["block"].astype(int).to_numpy()
    frozen_std    = np.array([clim["std_profile"].get(b, clim["std_fallback"]) for b in blocks])
    frozen_spread = np.array([clim["spread_profile"].get(b, 0.0) for b in blocks])

    for lag in SELF_REF_LAGS:
        out[f"rtm_lag_{lag}"]    = frozen_val
        out[f"spread_lag_{lag}"] = frozen_spread
    out["night_rtm_lag1"] = frozen_val * out["is_night_demand"].to_numpy()
    for w in SELF_REF_ROLLS:
        out[f"rtm_roll_mean_{w}"] = frozen_val
        out[f"rtm_roll_std_{w}"]  = frozen_std
    return out


def blind_market_features(feat, clim):
    """Replace DAM price and every traded-volume feature with its block
    profile. At declaration time none of the target month's DAM clearing or
    volumes exist yet — build_future_market_frame() already fills them from a
    30-day block median, so evaluation has to do the same or it scores the
    model on information production will never have."""
    out = feat.copy()
    blocks = out["block"].astype(int).to_numpy()

    for col in clim["blind_cols"]:
        base = _profile_series(clim, col, blocks)
        if col in out.columns:
            out[col] = base

        ramp_col = f"{col}_ramp" if col != "DAM_Price" else "dam_ramp"
        if ramp_col in out.columns:
            prev = _profile_series(clim, col, blocks, shift_blocks=1)
            out[ramp_col] = base - prev

        for lag in SELF_REF_LAGS:
            lag_col = f"dam_lag_{lag}" if col == "DAM_Price" else f"{col}_lag_{lag}"
            if lag_col in out.columns:
                out[lag_col] = _profile_series(clim, col, blocks, shift_blocks=lag)

        # A rolling mean over >= 1 day of a repeating daily profile is just
        # the profile's own daily mean.
        daily_mean = float(clim["block_profile"][col].mean())
        if not np.isfinite(daily_mean):
            daily_mean = float(clim["global_median"][col])
        for w in SELF_REF_ROLLS:
            roll_col = f"dam_roll_mean_{w}" if col == "DAM_Price" else f"{col}_roll_mean_{w}"
            if roll_col in out.columns:
                out[roll_col] = daily_mean
    return out


def apply_declaration_information_set(feat, clim, mode=None):
    """The full transform: blind the market features, then freeze the
    self-referential RTM ones. Use this on any frame that represents blocks
    at or beyond the declaration horizon."""
    out = blind_market_features(feat, clim)
    out = freeze_self_referential(out, clim)
    return out


print("Declaration information-set helpers ready.")
print(f"  self-referential RTM features: {len(RTM_SELF_REF_COLS)}")
print(f"  feature count — frozen mode   : {len(horizon_feature_set('frozen'))}")
print(f"  feature count — exogenous mode: {len(horizon_feature_set('exogenous'))}")


# --------------------------------------------------------------------------
# ### Declaration-replay walk-forward validation
#
# Folds carry the production lead gap and are scored under the declaration
# information set. The v3 one-block-ahead number is reported alongside, clearly
# labelled, because the gap between the two is itself a result.

# ==========================================================================
# [cell 34]
# ==========================================================================
# ============================================================================
# SECTION 20 (v4) — [BLOCKER-1] Declaration-replay walk-forward validation
# ============================================================================
# What changed vs v3, and why:
#
# 1. A DECLARATION_LEAD_DAYS gap now sits between each fold's train end and
#    its test start, exactly as it does in production. v3 tested on the rows
#    immediately following training, so fold-1 test rows were 15 minutes
#    past the last training row rather than 10-40 days past it.
#
# 2. Test folds are scored under the deployment information set
#    (apply_declaration_information_set): no target-month RTM prices, no
#    target-month DAM, no target-month volumes. v3 handed the test fold real
#    rtm_lag_1/2/4 and real DAM, which is why its cap-hit F1 (~0.92) looked
#    so strong -- with a 13% base rate and heavy autocorrelation, most of
#    that was persistence, and none of it was available at declaration time.
#
# 3. Each fold's test window is a calendar month (FOLD_TEST_DAYS), because a
#    month is what actually gets declared.
#
# 4. Folds are confined to the region BEFORE the calibration and test slices,
#    so they no longer overlap them. In v3, fold 5's test set sat entirely
#    inside the conformal calibration slice, and the two were then averaged
#    as if independent.
#
# 5. Two baselines are scored on the same folds, so the ML numbers finally
#    have something to be better than.
#
# The old 1-block-ahead number is still computed and printed, clearly
# labelled, because the gap between the two IS a result worth reporting.

N_FOLDS        = 5
FOLD_TEST_DAYS = 30
FOLD_TEST_SZ   = FOLD_TEST_DAYS * 96
LEAD_BLOCKS    = DECLARATION_LEAD_DAYS * 96
CV_FAST        = False    # True -> fewer trees, for quick iteration only

n_all     = len(train_full)
cut_test  = int(n_all * (1 - CONFORMAL_TEST_FRAC))
cut_calib = int(n_all * (1 - CONFORMAL_TEST_FRAC - CONFORMAL_CALIB_FRAC))

print(f"Row budget: {n_all:,} total")
print(f"  CV region     : [0, {cut_calib:,})            "
      f"{train_full['Datetime'].iloc[0].date()} .. {train_full['Datetime'].iloc[cut_calib-1].date()}")
print(f"  calibration   : [{cut_calib:,}, {cut_test:,})   "
      f"{train_full['Datetime'].iloc[cut_calib].date()} .. {train_full['Datetime'].iloc[cut_test-1].date()}")
print(f"  honest test   : [{cut_test:,}, {n_all:,})   "
      f"{train_full['Datetime'].iloc[cut_test].date()} .. {train_full['Datetime'].iloc[-1].date()}")
print(f"  fold gap      : {LEAD_BLOCKS:,} blocks ({DECLARATION_LEAD_DAYS} days)\n")

HORIZON_MODES = ["exogenous", "frozen"] if EVAL_BOTH_HORIZON_MODES else [RTM_HORIZON_MODE]


def _fit_models(tr, cols, weights):
    q_models = {}
    for q in RTM_QUANTILES:
        m = make_quantile_regressor(q)
        if CV_FAST:
            m.set_params(n_estimators=150)
        m.fit(tr[cols], tr["deviation_target"], sample_weight=weights)
        q_models[q] = m
    spike = make_classifier()
    crash = make_classifier()
    if CV_FAST:
        spike.set_params(n_estimators=120)
        crash.set_params(n_estimators=120)
    spike.fit(tr[cols], tr["is_spike"])
    crash.fit(tr[cols], tr["is_crash"])
    return q_models, spike, crash


def _predict(q_models, spike, crash, feat, cols):
    q_preds  = {q: q_models[q].predict(feat[cols]) for q in RTM_QUANTILES}
    q_matrix = rearrange_quantiles(np.column_stack([q_preds[q] for q in RTM_QUANTILES]))
    p_spike  = spike.predict_proba(feat[cols])[:, 1]
    p_crash  = crash.predict_proba(feat[cols])[:, 1]

    dev_low  = q_matrix[:, 0]
    dev_mid  = q_matrix[:, 1:4].mean(axis=1)
    dev_high = q_matrix[:, 4]
    p_mid    = np.clip(1 - p_spike - p_crash, 0, 1)
    norm     = p_spike + p_crash + p_mid + 1e-9
    pred_dev = (p_crash * dev_low + p_mid * dev_mid + p_spike * dev_high) / norm

    profile = feat["seasonal_profile"].to_numpy()
    price   = np.clip(profile + pred_dev, PRICE_FLOOR, PRICE_CAP)
    return price, profile + dev_low, profile + dev_high, p_spike, p_crash, q_preds


def _score(tag, fold, tr, te, y_true, price, p_spike, p_crash, q_preds):
    prec, rec, f1 = precision_recall_f1(te["is_spike"].to_numpy(), p_spike, 0.5)
    row = {
        "Fold": fold, "Evaluation": tag,
        "Train_Size": len(tr), "Test_Size": len(te),
        "Test_Start": te["Datetime"].iloc[0].date(),
        "MAE":  mean_absolute_error(y_true, price),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, price))),
        "CapHit_Precision@0.5": prec,
        "CapHit_Recall@0.5": rec,
        "CapHit_F1@0.5": f1,
        "Crash_Recall": (float((p_crash[te["is_crash"].to_numpy() == 1] > 0.5).mean())
                         if te["is_crash"].sum() else np.nan),
    }
    if q_preds is not None:
        for q in RTM_QUANTILES:
            row[f"Pinball_Q{int(q*100)}"] = pinball_loss(
                te["deviation_target"].to_numpy(), q_preds[q], q
            )
    return row


fold_rows, threshold_sweep_rows = [], []

for fold in range(1, N_FOLDS + 1):
    test_end   = cut_calib - (N_FOLDS - fold) * FOLD_TEST_SZ
    test_start = test_end - FOLD_TEST_SZ
    train_end  = test_start - LEAD_BLOCKS
    if train_end <= 0 or test_start <= 0:
        print(f"Fold {fold}: not enough history, skipped")
        continue

    tr = train_full.iloc[:train_end]
    te = train_full.iloc[test_start:test_end]
    w_tr = sample_weights_full[:train_end]
    clim = build_declaration_climatology(tr)
    te_decl = apply_declaration_information_set(te, clim)
    y_true = te["RTM_Price"].to_numpy()

    print(f"Fold {fold}: train -> {tr['Datetime'].iloc[-1].date()}, "
          f"declare for {te['Datetime'].iloc[0].date()} .. {te['Datetime'].iloc[-1].date()} "
          f"({len(tr):,} train rows)")

    # ── baselines, scored on the same fold ──────────────────────────────────
    fold_rows.append(_score(
        "baseline: climatology", fold, tr, te, y_true,
        np.clip(te["seasonal_profile"].to_numpy(), PRICE_FLOOR, PRICE_CAP),
        np.zeros(len(te)), np.zeros(len(te)), None,
    ))
    block_caprate = tr.groupby("block")["is_spike"].mean()
    p_block = te["block"].map(block_caprate).fillna(tr["is_spike"].mean()).to_numpy()
    fold_rows.append(_score(
        "baseline: DAM price + block cap-hit rate", fold, tr, te, y_true,
        np.clip(te_decl["DAM_Price"].to_numpy(), PRICE_FLOOR, PRICE_CAP),
        p_block, np.zeros(len(te)), None,
    ))

    # ── the model, per horizon mode ─────────────────────────────────────────
    for mode in HORIZON_MODES:
        cols = horizon_feature_set(mode)
        q_models, spike, crash = _fit_models(tr, cols, w_tr)

        price, _, _, p_spike, p_crash, q_preds = _predict(
            q_models, spike, crash, te_decl, cols
        )
        fold_rows.append(_score(
            f"declaration horizon ({mode})", fold, tr, te,
            y_true, price, p_spike, p_crash, q_preds,
        ))
        if mode == RTM_HORIZON_MODE:
            for t in CAP_HIT_THRESHOLD_GRID:
                p, r, f = precision_recall_f1(te["is_spike"].to_numpy(), p_spike, t)
                threshold_sweep_rows.append({
                    "Fold": fold, "Threshold": round(float(t), 2),
                    "Precision": p, "Recall": r, "F1": f,
                })

        # The v3 number: same fitted models, but the test frame left with its
        # real lagged prices and real DAM. Kept only for the comparison.
        if EVAL_REPORT_ONE_STEP and mode == "frozen":
            price1, _, _, ps1, pc1, qp1 = _predict(q_models, spike, crash, te, cols)
            fold_rows.append(_score(
                "1-block-ahead (v3, NOT the shipped horizon)", fold, tr, te,
                y_true, price1, ps1, pc1, qp1,
            ))

cv_results = pd.DataFrame(fold_rows)
cv_results.to_csv(ARTIFACT_DIR / f"RTM_v4_WalkForwardCV_{PEAK_MONTH}.csv", index=False)

summary_cols = ["MAE", "RMSE", "CapHit_Precision@0.5", "CapHit_Recall@0.5",
                "CapHit_F1@0.5", "Crash_Recall"]
cv_summary = (
    cv_results.groupby("Evaluation")[summary_cols].mean()
    .sort_values("MAE").round(3)
)
print("\n" + "=" * 78)
print("Mean across folds — what the model knows determines what it scores")
print("=" * 78)
print(cv_summary.to_string())
cv_summary.to_csv(ARTIFACT_DIR / f"RTM_v4_CV_Summary_{PEAK_MONTH}.csv")

_decl_key = f"declaration horizon ({RTM_HORIZON_MODE})"
_one_key  = "1-block-ahead (v3, NOT the shipped horizon)"
if _one_key in cv_summary.index and _decl_key in cv_summary.index:
    _d, _o = cv_summary.loc[_decl_key], cv_summary.loc[_one_key]
    print(f"\nHorizon cost — moving from the 1-block-ahead information set to the")
    print(f"real {DECLARATION_LEAD_DAYS}-to-{DECLARATION_LEAD_DAYS + FOLD_TEST_DAYS}-day declaration horizon:")
    print(f"  MAE        {_o['MAE']:>8.1f}  ->  {_d['MAE']:>8.1f} Rs/MWh "
          f"({_d['MAE'] / max(_o['MAE'], 1e-9):.1f}x)")
    print(f"  Cap-hit F1 {_o['CapHit_F1@0.5']:>8.3f}  ->  {_d['CapHit_F1@0.5']:>8.3f}")
    print("  The second row is the number the paper must quote.")

# ── Cap-hit threshold, chosen on the CV region only ─────────────────────────
# v3 averaged this curve with one computed on the calibration slice and called
# it more robust; the two overlapped, so it was averaging a number with
# itself. The threshold is now chosen here and only here. The calibration and
# test slices below are used to *check* it, never to pick it.
threshold_sweep = pd.DataFrame(threshold_sweep_rows)
threshold_summary = (
    threshold_sweep.groupby("Threshold")[["Precision", "Recall", "F1"]]
    .mean().reset_index().sort_values("Threshold")
)
print("\nCap-hit threshold sweep at the declaration horizon (mean across folds):")
print(threshold_summary.round(3).to_string(index=False))

CAP_HIT_THRESHOLD = float(threshold_summary.loc[threshold_summary["F1"].idxmax(), "Threshold"])
_best = threshold_summary.loc[threshold_summary["F1"].idxmax()]
print(f"\nSelected cap-hit decision threshold: {CAP_HIT_THRESHOLD:.2f} "
      f"(precision={_best['Precision']:.2f}, recall={_best['Recall']:.2f}, F1={_best['F1']:.2f})")
threshold_summary.to_csv(ARTIFACT_DIR / f"RTM_v4_CapHitThresholdSweep_{PEAK_MONTH}.csv", index=False)


# --------------------------------------------------------------------------
# ### Disjoint fit / calibrate / test — conformal margin and honest coverage

# ==========================================================================
# [cell 36]
# ==========================================================================
# ============================================================================
# SECTION 21 (v4) — [BLOCKER-3] Disjoint fit / calibrate / test, honest coverage
# ============================================================================
# v3 computed CONFORMAL_MARGIN as a quantile of the conformity scores on
# calib_df and then reported "empirical coverage" on that same calib_df. That
# number is the target level by construction -- it cannot come out any other
# way, so the printed 90.0% demonstrated nothing. Split conformal only
# guarantees coverage on data that had no hand in choosing the margin.
#
# There are now three disjoint slices, each separated by the declaration lead
# so every evaluation sits at the real horizon:
#
#   fit   [0, cut_calib)                    -> fits the quantile models + classifiers
#   calib [cut_calib+lead, cut_test)        -> estimates CONFORMAL_MARGIN
#   test  [cut_test+lead, n)                -> measures coverage. Never touched
#                                              by fitting or by margin estimation.
#
# Naming is also corrected here. RTM_QUANTILES tops out at 0.85, not 0.95, so
# the raw band is P05-P85; v3 called it "P05/P95" throughout. The output
# columns are now Forecast_RTM_Lo / Forecast_RTM_Hi, with the nominal levels
# stated rather than implied.

RTM_BAND_LO_Q = min(RTM_QUANTILES)
RTM_BAND_HI_Q = max(RTM_QUANTILES)

fit_df   = train_full.iloc[:cut_calib]
calib_df = train_full.iloc[cut_calib + LEAD_BLOCKS:cut_test]
test_df  = train_full.iloc[cut_test + LEAD_BLOCKS:]
fit_w    = sample_weights_full[:cut_calib]

print(f"fit   : {len(fit_df):>7,} rows  {fit_df['Datetime'].iloc[0].date()} .. {fit_df['Datetime'].iloc[-1].date()}")
print(f"calib : {len(calib_df):>7,} rows  {calib_df['Datetime'].iloc[0].date()} .. {calib_df['Datetime'].iloc[-1].date()}")
print(f"test  : {len(test_df):>7,} rows  {test_df['Datetime'].iloc[0].date()} .. {test_df['Datetime'].iloc[-1].date()}")
print(f"(each preceded by a {DECLARATION_LEAD_DAYS}-day gap, as in production)\n")

X_cols = horizon_feature_set(RTM_HORIZON_MODE)
print(f"Horizon mode: {RTM_HORIZON_MODE} — {len(X_cols)} features\n")

_fit_clim = build_declaration_climatology(fit_df)
rtm_quantile_models, rtm_spike_clf, rtm_crash_clf = _fit_models(fit_df, X_cols, fit_w)


def predict_deviation_ensemble(feat_df, cols=None):
    """(pred_price, lo, hi, p_spike, p_crash) for a frame that already carries
    a 'seasonal_profile' column. `cols` defaults to the production feature
    set; pass it explicitly when scoring a differently-fitted bundle."""
    cols = X_cols if cols is None else cols
    price, lo, hi, p_spike, p_crash, _ = _predict(
        rtm_quantile_models, rtm_spike_clf, rtm_crash_clf, feat_df, cols
    )
    return price, lo, hi, p_spike, p_crash


# ── Margin estimated on calib, under the declaration information set ────────
calib_decl = apply_declaration_information_set(calib_df, _fit_clim)
_, calib_lo, calib_hi, _, _ = predict_deviation_ensemble(calib_decl)
calib_y = calib_df["RTM_Price"].to_numpy()

conformity_scores = np.maximum(calib_lo - calib_y, calib_y - calib_hi)
n_calib = len(conformity_scores)
conformal_level  = min(1.0, np.ceil((n_calib + 1) * (1 - RTM_CONFORMAL_ALPHA)) / n_calib)
CONFORMAL_MARGIN = max(0.0, float(np.quantile(conformity_scores, conformal_level)))
print(f"Conformal margin from the calibration slice: ±{CONFORMAL_MARGIN:.0f} Rs/MWh "
      f"on the raw P{int(RTM_BAND_LO_Q*100)}-P{int(RTM_BAND_HI_Q*100)} band "
      f"(target {int((1 - RTM_CONFORMAL_ALPHA) * 100)}% coverage)")

_calib_cov = float(np.mean((calib_y >= calib_lo - CONFORMAL_MARGIN)
                           & (calib_y <= calib_hi + CONFORMAL_MARGIN)))
print(f"  in-sample coverage on the calibration slice: {_calib_cov*100:.1f}% "
      f"<- circular by construction, reported only to show it is NOT evidence")

# ── The honest number: coverage on the untouched test slice ─────────────────
test_decl = apply_declaration_information_set(test_df, _fit_clim)
test_price, test_lo, test_hi, test_p_spike, test_p_crash = predict_deviation_ensemble(test_decl)
test_y = test_df["RTM_Price"].to_numpy()

test_lo_adj = np.clip(test_lo - CONFORMAL_MARGIN, PRICE_FLOOR, PRICE_CAP)
test_hi_adj = np.clip(test_hi + CONFORMAL_MARGIN, PRICE_FLOOR, PRICE_CAP)
CONFORMAL_TEST_COVERAGE = float(np.mean((test_y >= test_lo_adj) & (test_y <= test_hi_adj)))
_mean_width = float(np.mean(test_hi_adj - test_lo_adj))

print(f"\n>>> Held-out coverage: {CONFORMAL_TEST_COVERAGE*100:.1f}% "
      f"(target {int((1 - RTM_CONFORMAL_ALPHA)*100)}%), mean interval width "
      f"{_mean_width:,.0f} Rs/MWh")
print(">>> This is the coverage figure the paper should quote.")
if abs(CONFORMAL_TEST_COVERAGE - (1 - RTM_CONFORMAL_ALPHA)) > 0.05:
    print("    NOTE: >5pp off target. Split conformal assumes exchangeability, "
          "which\n          a price series spanning the Jan-2026 market-coupling "
          "change does not\n          satisfy. Report this as approximate, not "
          "as a guarantee.")

# ── Per-block coverage: one scalar margin cannot fit every block ────────────
_cov_by_block = pd.DataFrame({
    "block": test_df["block"].to_numpy(),
    "covered": (test_y >= test_lo_adj) & (test_y <= test_hi_adj),
}).groupby("block")["covered"].mean()
print(f"\nPer-block coverage spread: min {_cov_by_block.min()*100:.0f}% / "
      f"median {_cov_by_block.median()*100:.0f}% / max {_cov_by_block.max()*100:.0f}%"
      "  (a Mondrian/block-conditional margin would tighten this)")

# ── Honest cap-hit metrics at the CV-selected threshold ─────────────────────
_p, _r, _f = precision_recall_f1(test_df["is_spike"].to_numpy(), test_p_spike, CAP_HIT_THRESHOLD)
print(f"\nCap-hit at threshold {CAP_HIT_THRESHOLD:.2f} on the untouched test slice:")
print(f"  precision {_p:.3f} | recall {_r:.3f} | F1 {_f:.3f} "
      f"| base rate {test_df['is_spike'].mean()*100:.1f}%")
print(f"  test-slice MAE {mean_absolute_error(test_y, test_price):,.1f} Rs/MWh")

pd.DataFrame([{
    "Horizon_Mode": RTM_HORIZON_MODE,
    "Declaration_Lead_Days": DECLARATION_LEAD_DAYS,
    "Conformal_Margin": CONFORMAL_MARGIN,
    "Calib_Coverage_Circular": _calib_cov,
    "Test_Coverage_Honest": CONFORMAL_TEST_COVERAGE,
    "Test_Mean_Interval_Width": _mean_width,
    "Cap_Hit_Threshold": CAP_HIT_THRESHOLD,
    "Test_CapHit_Precision": _p,
    "Test_CapHit_Recall": _r,
    "Test_CapHit_F1": _f,
    "Test_MAE": mean_absolute_error(test_y, test_price),
    "City_Weight_Provenance": CITY_WEIGHT_PROVENANCE,
}]).to_csv(ARTIFACT_DIR / f"RTM_v4_HonestMetrics_{PEAK_MONTH}.csv", index=False)

# ── Production refit on all history; margin carried over ────────────────────
# The margin is NOT recomputed here: doing so would require a held-out slice
# that the production fit has not seen, and there isn't one left. Carrying it
# over is standard practice, but it is an assumption, not a guarantee -- the
# refit adds the most recent ~16% of rows, so the true coverage of the shipped
# interval may differ from the test-slice figure above.
rtm_quantile_models, rtm_spike_clf, rtm_crash_clf = _fit_models(
    train_full, X_cols, sample_weights_full
)
PRODUCTION_CLIM = build_declaration_climatology(train_full)
print(f"\nProduction models refitted on all {len(train_full):,} rows "
      f"({len(RTM_QUANTILES)} quantiles + spike/crash). Margin carried over.")


# --------------------------------------------------------------------------
# ### Target-month forecast under the declaration information set
#
# No recursion: every block is past the AR-freeze horizon measured from the
# forecast origin, so the whole month is predicted in one pass.

# ==========================================================================
# [cell 38]
# ==========================================================================
# ============================================================================
# SECTION 22 (v4) — Target-month forecast under the declaration information set
# ============================================================================
# The gap between END_DATE and the first day of PEAK_MONTH is the regulatory
# declaration lead, not a data hole. v3 handled it badly: `history` ended at
# END_DATE, `future_market` began at month_start, and the two were concatenated
# as if adjacent -- so for the first forecast rows rtm_lag_1 pointed at the
# last block of END_DATE, rtm_lag_96 ("yesterday") pointed ~11 days back, and
# rtm_lag_672 ("last week") ~17 days back, all while being labelled as 1-block,
# 1-day and 1-week lags. Worse, `day_offset` was measured from month_start, so
# RTM_AR_FREEZE_DAYS=3 actually froze at day 13 of the real horizon.
#
# The freeze is now anchored to the forecast origin (END_DATE) instead. Since
# DECLARATION_LEAD_DAYS >= RTM_AR_FREEZE_DAYS, *every* block of the target
# month is past the freeze horizon -- so no self-referential feature is ever
# recursed, nothing is mislabelled, and the block-by-block loop disappears
# entirely (one vectorised pass instead of 2,880 iterations).

_horizon_lo = DECLARATION_LEAD_DAYS
_horizon_hi = DECLARATION_LEAD_DAYS + pd.Period(PEAK_MONTH, freq="M").days_in_month - 1
assert DECLARATION_LEAD_DAYS >= RTM_AR_FREEZE_DAYS, (
    "The declaration lead is shorter than the AR-freeze horizon; the early "
    "blocks would need genuine recursion, which this cell no longer does."
)
print(f"Forecast origin: {END_DATE} (last observed block)")
print(f"Target month   : {PEAK_MONTH}")
print(f"True horizon   : {_horizon_lo} to {_horizon_hi} days ahead — every block "
      f"is past the {RTM_AR_FREEZE_DAYS}-day AR-freeze, so all self-referential\n"
      f"                 RTM features come from climatology, none from recursion.")

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
_fm = future_market.copy()
_fm["RTM_Price"] = np.nan

_hist_cols = (["Datetime", "Date", "Block", "DAM_Price", "Net_Load", "RTM_Price"]
              + RTM_MARKET_FACTOR_COLS + DAM_MARKET_FACTOR_COLS)
_combined = pd.concat(
    [market[_hist_cols].tail(HISTORY_TAIL), _fm[[c for c in _hist_cols if c in _fm.columns]]],
    ignore_index=True,
)
_feat_all, _ = add_rtm_features_v3(_combined, include_market_factors=True)
future_feat = _feat_all.tail(len(_fm)).copy().reset_index(drop=True)

# Leak-free climatology anchor for every forecast block.
future_feat["seasonal_profile"] = [
    lookup_seasonal_profile(int(b), d, s, q=50)
    for b, d, s in zip(future_feat["block"], future_feat["day_type"], future_feat["season"])
]

# The one transform that defines the horizon. Same function the CV scored with,
# so the two can no longer drift apart.
future_decl = apply_declaration_information_set(future_feat, PRODUCTION_CLIM)

pred_price, band_lo, band_hi, p_spike, p_crash = predict_deviation_ensemble(future_decl)

future_market["Forecast_RTM_Price"]      = np.clip(pred_price, PRICE_FLOOR, PRICE_CAP)
future_market["Forecast_RTM_Lo"]         = np.clip(band_lo - CONFORMAL_MARGIN, PRICE_FLOOR, PRICE_CAP)
future_market["Forecast_RTM_Hi"]         = np.clip(band_hi + CONFORMAL_MARGIN, PRICE_FLOOR, PRICE_CAP)
future_market["Forecast_RTM_Spike_Prob"] = p_spike
future_market["Forecast_RTM_Crash_Prob"] = p_crash
future_market["Predicted_Cap_Hit"]       = p_spike >= CAP_HIT_THRESHOLD

future_market.to_csv(ARTIFACT_DIR / f"RTM_Forecast_v4_{PEAK_MONTH}.csv", index=False)
print("\n" + future_market[[
    "Datetime", "Forecast_RTM_Price", "Forecast_RTM_Lo", "Forecast_RTM_Hi",
    "Forecast_RTM_Spike_Prob", "Predicted_Cap_Hit",
]].head().to_string(index=False))

print(f"\nForecast RTM — mean: {future_market['Forecast_RTM_Price'].mean():.0f}  "
      f"P90: {future_market['Forecast_RTM_Price'].quantile(0.9):.0f}  "
      f"max: {future_market['Forecast_RTM_Price'].max():.0f}")

n_cap_blocks = int(future_market["Predicted_Cap_Hit"].sum())
n_cap_days   = future_market.loc[future_market["Predicted_Cap_Hit"], "Date"].nunique()
_hist_rate   = train_full["is_spike"].mean()
print(f"Predicted cap-hit blocks: {n_cap_blocks} / {len(future_market)} "
      f"({n_cap_blocks / len(future_market) * 100:.1f}%) across {n_cap_days} day(s) "
      f"at threshold {CAP_HIT_THRESHOLD:.2f}")
print(f"Historical cap-hit base rate: {_hist_rate*100:.1f}%. A large gap between "
      f"the two is\nworth explaining in the write-up — it is either a genuine "
      f"seasonal call or a\nsymptom of the climatology-frozen features flattening "
      f"the spike signal.")


# --------------------------------------------------------------------------
# ## SECTION 23 – Peak-hour scoring & selection

# ==========================================================================
# [cell 40]
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
    f"Candidate windows gated to start at/after block {PEAK_START_BLOCK} "
    f"({block_to_time(PEAK_START_BLOCK)}) — data-derived, but a real "
    f"time-of-day constraint, not the absence of one."
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
        "RTM v4: seasonal-climatology decomposition + 5-quantile deviation ensemble, "
        "learned spike/crash regime probabilities, conformal band (P05-P85 + margin) "
        "validated on a held-out slice, "
        f"forecast made {DECLARATION_LEAD_DAYS} days before the month starts under "
        "the declaration information set (no target-month price, DAM or volume)"
    ),
}])

monthly_peak.to_excel(ARTIFACT_DIR / f"Monthly_Peak_Hours_{PEAK_MONTH}.xlsx", index=False)
monthly_peak.to_csv(ARTIFACT_DIR / f"Monthly_Peak_Hours_{PEAK_MONTH}.csv", index=False)
peak_method_comparison.to_excel(ARTIFACT_DIR / f"Peak_Method_Comparison_{PEAK_MONTH}.xlsx", index=False)
peak_method_comparison.to_csv(ARTIFACT_DIR / f"Peak_Method_Comparison_{PEAK_MONTH}.csv", index=False)
print(monthly_peak)


# --------------------------------------------------------------------------
# ## SECTION 23b — Declaration scorecard: value capture vs. the ex-post optimum

# ==========================================================================
# [cell 42]
# ==========================================================================
# ============================================================================
# SECTION 23b — [BLOCKER-2] Scoring a declaration against what actually happened
# ============================================================================
# The pipeline's deliverable is a window, so the evaluation has to be about
# windows, not about price MAE. This harness scores any declared window for
# any PAST month against the ex-post optimum computed from actuals.
#
# Metric: VALUE CAPTURE. The objective the selector maximises is Net_Load x
# RTM_Price, so for a past month take the actual NL x RTM per block, sum it
# over the declared blocks, and express that as a fraction of what the best
# admissible window would have captured. Regret = 1 - capture. This is
# comparable across months and needs no model to compute.
#
# Drop a Previous_Declarations.csv next to the notebook to score the real
# historical declarations. Recognised columns:
#     Month        YYYY-MM                                     (required)
#     Peak_Hours   "18:30-21:30" or "18:30-20:30, 21:45-22:45"  (required)
# Anything else is carried through untouched.

PREV_DECL_CSV = MARKET_CACHE_DIR / "Previous_Declarations.csv"


def blocks_from_peak_hours(text):
    """'18:30-20:30, 21:45-22:45' -> the tuple of 15-min blocks it covers."""
    blocks = []
    for part in str(text).split(","):
        part = part.strip()
        if not part or "-" not in part:
            continue
        lo, hi = (p.strip() for p in part.split("-", 1))
        b_lo = time_to_block(lo)
        b_hi = time_to_block("24:00" if hi in ("00:00", "24:00") else hi)
        blocks.extend(range(b_lo, b_hi))
    return tuple(sorted(set(blocks)))


def actual_month_value(month_yyyy_mm):
    """Actual per-block NL x RTM for a past month, averaged over its days."""
    m_start, m_end = month_bounds(month_yyyy_mm)
    act = market[(market["Datetime"] >= m_start) & (market["Datetime"] <= m_end)].copy()
    if act.empty:
        return None
    act["NL_RTM"] = act["Net_Load"] * act["RTM_Price"]
    return act.groupby("Block")["NL_RTM"].mean().reindex(range(1, 97)).fillna(0.0)


def score_window(value_by_block, blocks):
    return float(value_by_block.loc[list(blocks)].sum())


_LENGTH_CANDIDATE_CACHE = {}


def candidates_of_length(total_blocks, start_block=1, end_block=96, min_split=4):
    """Continuous and two-segment windows of exactly `total_blocks`.

    Mirrors benchmark_declarations.build_candidates so the notebook and the
    standalone scorer search the same space.
    """
    key = (total_blocks, start_block, end_block, min_split)
    if key in _LENGTH_CANDIDATE_CACHE:
        return _LENGTH_CANDIDATE_CACHE[key]
    out = []
    for s0 in range(start_block, end_block + 1):
        if s0 + total_blocks - 1 > end_block:
            break
        out.append(tuple(range(s0, s0 + total_blocks)))
    for len1 in range(min_split, total_blocks - min_split + 1):
        len2 = total_blocks - len1
        for s1 in range(start_block, end_block + 1):
            e1 = s1 + len1 - 1
            if e1 > end_block:
                break
            for s2 in range(e1 + 2, end_block + 1):
                if s2 + len2 - 1 > end_block:
                    break
                out.append(tuple(range(s1, s1 + len1)) + tuple(range(s2, s2 + len2)))
    _LENGTH_CANDIDATE_CACHE[key] = out
    return out


def best_window(value_by_block, length=None):
    """Ex-post optimal window, searched at `length` blocks.

    LENGTH MATTERS. The pipeline emits 12-block (3h) windows, but every NRPC
    thermal declaration in Previous_Declarations.csv is 16 blocks (4h). Scoring
    a 16-block declaration against the best 12-block window is not a value
    capture at all - a longer window collects strictly more, so 49 of 50 months
    reported >100% capture (mean 122.9%, max 160.7%) and negative regret.
    Passing `length` makes the optimum the best window of the SAME size, which
    is the only comparison that can be read as a percentage.

    With length=None the search falls back to the selector's own candidate set,
    which is correct only when the thing being scored is the selector's output.
    """
    if length is None:
        pool = [blocks for _pattern, _windows, blocks in _CANDIDATE_WINDOWS]
    else:
        pool = candidates_of_length(length)
    best_b, best_v = None, -np.inf
    for blocks in pool:
        v = score_window(value_by_block, blocks)
        if v > best_v:
            best_b, best_v = blocks, v
    return best_b, best_v


FIXED_BASELINE_HOURS = "18:30-21:30"   # the conventional evening window


def evaluate_declaration(month_yyyy_mm, declared_hours, label="declared"):
    vals = actual_month_value(month_yyyy_mm)
    if vals is None:
        return None
    declared_blocks = blocks_from_peak_hours(declared_hours)
    if not declared_blocks:
        return None
    # Everything below is scored at the DECLARED window's own length, so the
    # baselines, the optimum and the declaration are all the same size.
    n_blocks = len(declared_blocks)
    opt_blocks, opt_val = best_window(vals, length=n_blocks)
    fixed_blocks = best_window(
        vals.reindex(range(1, 97)).where(
            vals.index.isin(blocks_from_peak_hours(FIXED_BASELINE_HOURS)), 0.0
        ), length=n_blocks)[0] if False else blocks_from_peak_hours(FIXED_BASELINE_HOURS)
    rows = []
    candidates = {
        label: declared_blocks,
        f"baseline: fixed {FIXED_BASELINE_HOURS}": fixed_blocks,
        "ex-post optimum": opt_blocks,
    }
    prev_year = (pd.Period(month_yyyy_mm, freq="M") - 12)
    prev_vals = actual_month_value(str(prev_year))
    if prev_vals is not None:
        candidates["baseline: last year's optimum"] = best_window(
            prev_vals, length=n_blocks)[0]

    for name, blocks in candidates.items():
        if not blocks:
            continue
        v = score_window(vals, blocks)
        rows.append({
            "Month": month_yyyy_mm,
            "Method": name,
            "Peak_Hours": ", ".join(ranges_from_windows(
                [(min(blocks), len(blocks))])) if len(blocks) == (max(blocks) - min(blocks) + 1)
                else f"{len(blocks)} blocks",
            "Blocks": len(blocks),
            "Value_Capture_%": 100.0 * v / opt_val if opt_val else np.nan,
            "Regret_%": 100.0 * (1 - v / opt_val) if opt_val else np.nan,
        })
    return pd.DataFrame(rows)


if PREV_DECL_CSV.exists():
    _prev = pd.read_csv(PREV_DECL_CSV)
    _need = {"Month", "Peak_Hours"}
    if not _need.issubset(_prev.columns):
        raise KeyError(f"{PREV_DECL_CSV.name} must contain {_need}; got {list(_prev.columns)}")
    _scored = [evaluate_declaration(str(r["Month"]), r["Peak_Hours"], "declared (actual NRLDC)")
               for _, r in _prev.iterrows()]
    _scored = [s for s in _scored if s is not None]
    if _scored:
        declaration_scorecard = pd.concat(_scored, ignore_index=True)
        print("Historical declarations scored against the ex-post optimum:")
        print(declaration_scorecard.pivot_table(
            index="Month", columns="Method", values="Value_Capture_%"
        ).round(1).to_string())
        print("\nMean value capture by method:")
        print(declaration_scorecard.groupby("Method")["Value_Capture_%"]
              .agg(["mean", "std", "min"]).round(1).to_string())
        declaration_scorecard.to_csv(
            ARTIFACT_DIR / "Declaration_Scorecard.csv", index=False
        )
    else:
        print(f"{PREV_DECL_CSV.name} found, but no month in it overlaps the market data.")
else:
    print(f"No {PREV_DECL_CSV.name} yet — drop one in with columns "
          "Month,Peak_Hours to score past declarations.")
    print("Meanwhile, here is the baseline comparison on the last 12 complete months:")
    _rows = []
    for _p in pd.period_range(end=pd.Period(END_DATE, freq="M") - 1, periods=12, freq="M"):
        _s = evaluate_declaration(str(_p), FIXED_BASELINE_HOURS, "fixed window as if declared")
        if _s is not None:
            _rows.append(_s)
    if _rows:
        _bl = pd.concat(_rows, ignore_index=True)
        print(_bl.groupby("Method")["Value_Capture_%"].agg(["mean", "std", "min"]).round(1).to_string())
        _bl.to_csv(ARTIFACT_DIR / "Baseline_Value_Capture.csv", index=False)

# ── Optional corroboration: grid frequency as an independent stress signal ──
# Frequency is a genuinely useful *secondary* check because it is completely
# exogenous to the selector (it feeds no feature and no score), and because it
# is system truth rather than a market outcome: when the NR is short, frequency
# falls. If the declared window really lands on the tightest part of the day,
# it should coincide with the day's lowest frequency blocks.
#
# Two caveats to state in the paper if this is used:
#   * India's grid is synchronous, so this is essentially all-India frequency.
#     It measures national scarcity, not NR-specific scarcity, and will not
#     discriminate a regionally tight hour from a nationally tight one.
#   * Averaging within a 15-min block washes out excursions. Use the block
#     MINIMUM and the fraction of seconds below the IEGC band, not the mean.
#
# The series is not in the cache yet: scada_cache.tags.NR_FRIENDLY_HEADERS has
# no frequency entry, so the parser drops that column. Candidate header names
# have been added there; re-parse ON THE LAN to populate it:
#     cd scada-cache && rm -rf cache/nr && python update_cache.py --source nr
IEGC_BAND_LO, IEGC_BAND_HI = 49.90, 50.05


def load_frequency_15min(start, end):
    """Block-level frequency stress from the 1-minute NR series, or None."""
    try:
        import scada_cache as _sc
        raw = _sc.load("nr", start=start, end=end)
    except Exception as exc:
        print("Frequency unavailable:", exc)
        return None
    freq_col = next((c for c in raw.columns if "freq" in c.lower()), None)
    if freq_col is None:
        print("No frequency column in the nr cache — add it to "
              "scada_cache/tags.py and re-parse on the LAN "
              f"(have: {list(raw.columns)})")
        return None
    f = raw[freq_col].astype(float)
    return pd.DataFrame({
        "Freq_Min":        f.resample("15min").min(),
        "Freq_Mean":       f.resample("15min").mean(),
        "Frac_Below_Band": f.lt(IEGC_BAND_LO).resample("15min").mean(),
    })


def frequency_corroboration(month_yyyy_mm, blocks):
    m_start, m_end = month_bounds(month_yyyy_mm)
    fq = load_frequency_15min(m_start, m_end)
    if fq is None:
        return None
    fq["Block"] = fq.index.hour * 4 + fq.index.minute // 15 + 1
    prof = fq.groupby("Block")[["Freq_Min", "Frac_Below_Band"]].mean()
    inside = prof.loc[prof.index.isin(blocks)]
    outside = prof.loc[~prof.index.isin(blocks)]
    out = pd.DataFrame({
        "Declared_Window": inside.mean(), "Rest_Of_Day": outside.mean(),
    }).T
    out["Rank_Of_Window_By_Freq_Min"] = np.nan
    print(f"\nFrequency corroboration for {month_yyyy_mm} "
          f"(lower Freq_Min / higher Frac_Below_Band = tighter system):")
    print(out.round(4).to_string())
    return out


# --------------------------------------------------------------------------
# ## SECTION 24 – Daily audit table

# ==========================================================================
# [cell 44]
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
        # [NR-VOLUMES] The three above are All-India; these are NR's own
        # position in the same auction. Avg_NR_Net_Buy_MW < 0 means the
        # region was a net seller into the exchange across the declared
        # window -- worth seeing next to a peak-hour declaration, since it
        # says whether NR was actually short during the hours it declared.
        "Avg_NR_Cleared_Buy_MW":       round(selected["NR_Cleared_Buy_MW"].mean(), 2),
        "Avg_NR_Cleared_Sell_MW":      round(selected["NR_Cleared_Sell_MW"].mean(), 2),
        "Avg_NR_Net_Buy_MW":           round(selected["NR_Net_Buy_MW"].mean(), 2),
        "Avg_NR_Share_of_Cleared":     round(selected["NR_Share_of_Cleared"].mean(), 4),
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
# [cell 46]
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
        "NR_Cleared_Buy_MW", "NR_Net_Buy_MW",
    ]].mean().reindex(range(1, 97))
)
block_hours = (block_profile.index - 1) / 4

axes[2].fill_between(block_hours, block_profile["Purchase_Bid_MW"].fillna(0).to_numpy(),
                     color="#4f46a3", alpha=0.55, label="Purchase Bid (MW)")
axes[2].fill_between(block_hours, block_profile["Sell_Bid_MW"].fillna(0).to_numpy(),
                     color="gold", alpha=0.55, label="Sell Bid (MW)")
axes[2].plot(block_hours, block_profile["MCV_MW"], color="tomato", lw=1.4, label="MCV (MW)")
# [NR-VOLUMES] The filled areas above are All-India bid stacks; these two
# lines are NR. NR net buy crossing zero is the block where the region flips
# from net seller to net buyer on the exchange.
axes[2].plot(block_hours, block_profile["NR_Cleared_Buy_MW"], color="#1b7f5a", lw=1.4,
             label="NR cleared buy (MW)")
axes[2].plot(block_hours, block_profile["NR_Net_Buy_MW"], color="#1b7f5a", lw=1.2,
             ls="--", label="NR net buy (MW)")
axes[2].axhline(0, color="#666666", lw=0.8, ls=":")
axes[2].set_title("Average RTM Market Snapshot by 15-Minute Block — "
                  "All-India bids (filled) vs NR cleared (lines)")
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
# ## SECTION 26 – Single-day diagnostic chart

# ==========================================================================
# [cell 48]
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
# [cell 50]
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
    df = pd.read_csv(FIG_DATA / "City_Weight_Comparison.csv")
    df = df.sort_values("New_Weight", ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(SINGLE_W, 4.0))
    ax.barh(df["City"], df["New_Weight"], color=COL_NL, height=0.62)
    ax.set_xlabel("Demand-anchored weight (share of NR state drawal)")
    ax.set_ylabel("Weather station")
    ax.set_title("Top 20 of 40 NR load-centre weather stations\nby measured state-demand weight", fontsize=8)
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    _savefig(fig, "fig2_weather_weights.png")


# ---------------------------------------------------------------- Fig 3
def fig3_rtm_band():
    df = pd.read_csv(FIG_DATA / f"RTM_Forecast_v4_{FIG_MONTH}.csv", parse_dates=["Datetime"])
    day = df[pd.to_datetime(df["Date"]).dt.date == pd.Timestamp(FIG_PLOT_DAY).date()].copy()
    t = day["Datetime"]

    fig, ax1 = plt.subplots(figsize=(SINGLE_W, 2.7))
    ax1.fill_between(t, day["Forecast_RTM_Lo"], day["Forecast_RTM_Hi"],
                     color=COL_RTM, alpha=0.22, label="Conformal band (P05\u2013P85 + margin)", linewidth=0)
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
    df = pd.read_csv(FIG_DATA / f"RTM_v4_WalkForwardCV_{FIG_MONTH}.csv")
    # Only the rows scored under the real declaration information set.
    df = df[df["Evaluation"].str.startswith("declaration horizon")]
    df = df.groupby("Fold", as_index=False).first()
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
    df = pd.read_csv(FIG_DATA / f"RTM_v4_CapHitThresholdSweep_{FIG_MONTH}.csv")
    best_idx = df["F1"].idxmax()
    best_t = df.loc[best_idx, "Threshold"]

    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.7))
    ax.plot(df["Threshold"], df["Precision"], color=COL_NL, marker="o", ms=2.5, label="Precision")
    ax.plot(df["Threshold"], df["Recall"], color=COL_RTM, marker="s", ms=2.5, label="Recall")
    ax.plot(df["Threshold"], df["F1"], color=COL_ACC, marker="^", ms=2.5, label="F1")
    ax.axvline(best_t, color="#333333", linestyle=":", linewidth=1.0)
    ax.annotate(f"selected\nthr.={best_t:.2f}", xy=(best_t, df.loc[best_idx, "F1"]),
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
    df = pd.read_csv(FIG_DATA / f"RTM_Forecast_v4_{FIG_MONTH}.csv", parse_dates=["Datetime"])
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
    df = pd.read_csv(FIG_DATA / f"RTM_Forecast_v4_{FIG_MONTH}.csv", parse_dates=["Datetime"])
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


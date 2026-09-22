"""What history a declaration is allowed to look at.

`same_month_2yr` is `hydro_peak_model.information_set()`, generalized to any
month column name: the SAME calendar month in the two prior years, and
nothing else -- deliberately not the two most recent months. Peak timing
tracks sunset (seasonal), not recent trend, so blending in adjacent months
drags the declared window 1.5-2 hours early. Measured out-of-sample (19
months, frequency stress captured):

    M-12, M-1, M-2   57.91%   (loses to NRPC's 61.32%)
    M-12             67.43%
    M-12, M-24       67.57%   <- this is what's implemented below

`forecast_horizon` -- the thermal notebook's approach (train net-load/RTM
forecast models, predict the target month under the real declaration lead
time) -- is NOT implemented here. It needs the full ML stack (Segment 6),
not a simple history filter, and is left as a documented placeholder so the
interface (`information_set(df, month) -> pd.DataFrame`) is settled now even
though only one implementation exists yet.
"""
from __future__ import annotations

import pandas as pd


def same_month_2yr(df: pd.DataFrame, month: str, month_col: str = "Month") -> pd.DataFrame:
    """Rows from the same calendar month, 12 and 24 months before `month`.

    Never includes `month` itself -- callers relying on this as a leak guard
    (e.g. `seasonal_baseline`'s fallback cascade) can assert on that.
    """
    per = pd.Period(month, freq="M")
    lookback_months = {str(per - 12), str(per - 24)}
    assert month not in lookback_months, "same_month_2yr must never include the target month"
    return df[df[month_col].isin(lookback_months)]


def forecast_horizon(df: pd.DataFrame, month: str, month_col: str = "Month") -> pd.DataFrame:
    """Placeholder -- Segment 6 ML advisory stack. Not yet implemented."""
    raise NotImplementedError(
        "forecast_horizon requires the ML forecasting stack (Segment 6); "
        "use same_month_2yr for the seasonal baseline / hydro model."
    )

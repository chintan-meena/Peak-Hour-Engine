import pandas as pd
import pytest

from peak_hours.engine.information_set import forecast_horizon, same_month_2yr


def _months_df(months):
    return pd.DataFrame({"Month": months, "Value": range(len(months))})


def test_same_month_2yr_selects_only_m12_and_m24():
    df = _months_df(["2024-09", "2025-09", "2026-09", "2026-08", "2023-09"])
    out = same_month_2yr(df, "2026-09")
    assert set(out["Month"]) == {"2025-09", "2024-09"}


def test_same_month_2yr_excludes_target_month_even_if_present():
    df = _months_df(["2026-09", "2025-09", "2024-09"])
    out = same_month_2yr(df, "2026-09")
    assert "2026-09" not in set(out["Month"])


def test_same_month_2yr_excludes_adjacent_months():
    df = _months_df(["2025-08", "2025-09", "2025-10", "2024-09"])
    out = same_month_2yr(df, "2026-09")
    assert set(out["Month"]) == {"2025-09", "2024-09"}


def test_same_month_2yr_empty_when_no_history():
    df = _months_df(["2026-06", "2026-07"])
    out = same_month_2yr(df, "2026-09")
    assert out.empty


def test_forecast_horizon_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        forecast_horizon(_months_df(["2026-09"]), "2026-10")

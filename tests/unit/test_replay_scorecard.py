import pandas as pd
import pytest

from peak_hours.benchmarking import replay_scorecard as rsc


def _fake_actuals():
    """One synthetic month (2026-09) + its prior year (2025-09), full days.

    Block 73-84 (18:00-21:00) carries the highest NL_RTM and RTM_Price in
    both years, so a perfect selector should always land there.
    """
    rows = []
    for month, mult in (("2025-09", 1.0), ("2026-09", 1.0)):
        start = pd.Period(month, freq="M").start_time
        for day_offset in range(3):  # 3 synthetic days is enough to average
            day = start + pd.Timedelta(days=day_offset)
            for block in range(1, 97):
                dt = day + pd.Timedelta(minutes=15 * (block - 1))
                peak = 73 <= block <= 84
                rtm = (500.0 + 400.0 * peak) * mult
                nl = 40_000.0 + 20_000.0 * peak
                rows.append({
                    "Datetime": dt, "Block": block, "Month": month,
                    "Net_Load": nl, "RTM_Price": rtm, "NL_RTM": nl * rtm,
                })
    return pd.DataFrame(rows)


def test_score_month_perfect_candidate_captures_100_percent():
    act = _fake_actuals()
    r = rsc.score_month(
        "2026-09",
        {"perfect": "18:00-21:00"},  # exactly blocks 73-84
        with_baselines=False,
        act=act,
    )
    assert len(r) == 1
    row = r.iloc[0]
    assert row["Value_Capture_%"] == pytest.approx(100.0)
    assert row["Blocks"] == 12
    assert row["Days_Scored"] == 3
    assert row["Days_In_Month"] == 30


def test_score_month_reports_partial_month():
    act = _fake_actuals()
    r = rsc.score_month("2026-09", {"perfect": "18:00-21:00"}, act=act)
    assert (r["Days_Scored"] < r["Days_In_Month"]).all()


def test_score_month_baselines_present_once_per_length():
    act = _fake_actuals()
    r = rsc.score_month(
        "2026-09",
        {"a": "18:00-21:00", "b": "17:45-20:45"},  # both 12 blocks
        act=act,
    )
    baseline_labels = r["Candidate"].str.startswith("baseline:")
    # two baselines (last-year, RTM-only), once for the one distinct length
    assert baseline_labels.sum() == 2


def test_score_month_empty_candidate_raises():
    act = _fake_actuals()
    with pytest.raises(RuntimeError):
        rsc.score_month("2026-09", {"nothing": ""}, act=act, with_baselines=False)


def test_score_month_unknown_month_raises():
    act = _fake_actuals()
    with pytest.raises(RuntimeError):
        rsc.score_month("2030-01", {"perfect": "18:00-21:00"}, act=act)

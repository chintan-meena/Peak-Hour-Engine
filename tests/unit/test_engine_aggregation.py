import numpy as np
import pandas as pd
import pytest

from peak_hours.engine.aggregation import (
    daily_top_frequency,
    mean,
    median,
    normalize,
    p75,
    p90,
    recency_weighted,
)


def _two_day_frame():
    # Block 1: day1=10, day2=30 -> mean 20, median 20
    # Block 2: day1=100, day2=100
    # every other block: no data (NaN after reindex)
    return pd.DataFrame({
        "Date": ["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02"],
        "Block": [1, 1, 2, 2],
        "Score": [10.0, 30.0, 100.0, 100.0],
    })


def test_mean_hand_computed():
    out = mean(_two_day_frame())
    assert out.loc[1] == 20.0
    assert out.loc[2] == 100.0
    assert pd.isna(out.loc[3])          # no data for block 3
    assert len(out) == 96


def test_median_hand_computed():
    out = median(_two_day_frame())
    assert out.loc[1] == 20.0


def test_p75_and_p90_hand_computed():
    df = pd.DataFrame({
        "Date": ["d1", "d2", "d3", "d4"] * 1,
        "Block": [1, 1, 1, 1],
        "Score": [10.0, 20.0, 30.0, 40.0],
    })
    # numpy/pandas default linear interpolation: p75 of [10,20,30,40] = 32.5
    assert p75(df).loc[1] == pytest.approx(32.5)
    assert p90(df).loc[1] == pytest.approx(37.0)


def test_daily_top_frequency_counts_how_often_block_ranks_in_top_n():
    # 2 days, 3 blocks scored each day; top_n=1 -> only the single highest
    # block per day counts.
    df = pd.DataFrame({
        "Date": ["d1", "d1", "d1", "d2", "d2", "d2"],
        "Block": [1, 2, 3, 1, 2, 3],
        "Score": [5.0, 1.0, 1.0, 1.0, 5.0, 1.0],
    })
    out = daily_top_frequency(df, total_blocks=1)
    assert out.loc[1] == pytest.approx(0.5)   # top on day1 only, 1/2 days
    assert out.loc[2] == pytest.approx(0.5)   # top on day2 only
    assert out.loc[3] == pytest.approx(0.0)   # never top


def test_recency_weighted_upweights_days_closer_to_reference():
    # matches real usage: reference_date is the last real-data day (e.g.
    # END_DATE), and scored days are in the forecast month AFTER it, so
    # weight decays as days move further from the reference into the future.
    df = pd.DataFrame({
        "Date": ["2026-08-23", "2026-09-20"],   # 1 day out, ~29 days out
        "Block": [1, 1],
        "Score": [100.0, 0.0],
    })
    out = recency_weighted(df, reference_date="2026-08-22", half_life_days=1)
    # the near day (100, weight 0.5^1) should dominate the far day (0,
    # weight 0.5^29, effectively zero) -> weighted average close to 100,
    # not the plain mean (50)
    assert out.loc[1] > 90.0


def test_normalize_fills_missing_blocks_with_median_then_zscores():
    s = pd.Series({1: 10.0, 2: 20.0, 3: 30.0})   # blocks 4-96 missing
    out = normalize(s)
    assert len(out) == 96
    assert np.all(np.isfinite(out))
    # the filled (median) blocks should all score ~0 after robust-z
    assert abs(out[3]) < 1e-9   # index 3 (0-based) == block 4, filled with median(20)

"""Smoke tests against the real price CSVs in the shared OneDrive cache.

These are lighter-weight than true unit tests (they touch real data rather
than synthetic fixtures) but the data is small enough to load quickly, and
this is exactly the kind of "does the copied-in module still read the same
data the same way" check Segment 1 needs.
"""
import pandas as pd

from peak_hours.io import market_cache


def test_load_rtm_has_expected_columns():
    df = market_cache.load("RTM")
    assert list(df.columns) == ["Datetime", "Block", "Price"]
    assert len(df) > 0
    assert df["Block"].between(1, 96).all()


def test_load_full_rtm_has_price_column():
    df = market_cache.load_full("RTM")
    assert "RTM_Price" in df.columns
    assert "Datetime" in df.columns


def test_load_date_filtering():
    full = market_cache.load("RTM")
    start, end = full["Datetime"].iloc[100], full["Datetime"].iloc[200]
    sub = market_cache.load("RTM", start=start, end=end)
    assert sub["Datetime"].min() >= start
    assert sub["Datetime"].max() <= end
    assert len(sub) <= len(full)


def test_csv_path_points_at_market_cache_dir():
    from peak_hours.paths import MARKET_CACHE_DIR
    p = market_cache.csv_path("RTM")
    assert p.parent == MARKET_CACHE_DIR
    assert p.exists()

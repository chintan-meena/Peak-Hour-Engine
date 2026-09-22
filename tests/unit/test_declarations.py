"""Smoke tests for the declarations parser against the real xlsx.

Ported logic from build_previous_declarations.py -- see test_market_cache.py
for why this touches real (committed, small) data rather than a synthetic
fixture at this stage.
"""
from peak_hours.io import declarations as decl


def test_parse_ranges_both_time_spellings():
    assert decl.parse_ranges("18:15 to 20:15 and 21:30 to 23:30") == [
        ("18:15", "20:15"), ("21:30", "23:30")]
    assert decl.parse_ranges("1815 to 2115 and 2200 to 2300") == [
        ("18:15", "21:15"), ("22:00", "23:00")]


def test_parse_ranges_rejects_invalid_minute():
    # minute must be one of 0/15/30/45
    assert decl.parse_ranges("18:07 to 20:07") == []


def test_n_blocks():
    assert decl.n_blocks([("18:00", "19:00")]) == 4
    assert decl.n_blocks([("18:00", "19:00"), ("20:00", "20:30")]) == 6


def test_build_thermal_produces_expected_block_counts():
    df = decl.build(peak="thermal")
    assert len(df) > 0
    assert set(df["Blocks"].unique()) <= {12, 16}  # thermal is 16-block by practice; some may be short
    assert df["Blocks"].mode().iloc[0] == 16


def test_build_hydro_has_more_morning_segments_than_thermal():
    thermal = decl.build(peak="thermal")
    hydro = decl.build(peak="hydro")
    # hydro's own docstring: most winter hydro declarations carry a morning segment
    assert hydro["Has_Morning"].sum() >= thermal["Has_Morning"].sum()

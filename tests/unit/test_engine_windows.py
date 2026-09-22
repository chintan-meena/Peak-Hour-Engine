import numpy as np
import pytest

from peak_hours.engine.windows import build_candidates


def test_continuous_only_when_min_split_too_large():
    # total=4, min_split=4 -> no valid split (would need two segments of >=4
    # summing to 4, impossible), so only continuous windows exist.
    cand = build_candidates(total_blocks=4, start_block=1, end_block=96, min_split=4)
    assert cand.shape[1] == 4
    assert cand.shape[0] == 96 - 4 + 1
    # every row must be a contiguous run
    for row in cand:
        assert list(row) == list(range(row[0], row[0] + 4))


def test_splits_appear_when_min_split_allows():
    cand = build_candidates(total_blocks=8, start_block=1, end_block=20, min_split=4)
    continuous_count = 20 - 8 + 1
    assert cand.shape[0] > continuous_count
    assert cand.shape[1] == 8


def test_zero_based_output_within_bounds():
    cand = build_candidates(total_blocks=6, start_block=1, end_block=30, min_split=3)
    assert cand.min() >= 0
    assert cand.max() <= 29     # 0-based, end_block=30 -> max index 29


def test_start_block_gate_is_respected():
    cand = build_candidates(total_blocks=4, start_block=50, end_block=96, min_split=4)
    # 0-based, so start_block=50 (1-based) -> minimum index 49
    assert cand.min() == 49


def test_morning_band_adds_split_candidates_restricted_to_band():
    no_morning = build_candidates(total_blocks=12, start_block=1, end_block=96,
                                   min_split=4, morning_band=None)
    with_morning = build_candidates(total_blocks=12, start_block=1, end_block=96,
                                     min_split=4, morning_band=(1, 28))
    assert with_morning.shape[0] > no_morning.shape[0]
    # morning-band candidates are appended after continuous + evening-split
    # ones; every one of them must start within the morning band (0-based:
    # band (1,28) -> indices 0..27)
    morning_only = with_morning[no_morning.shape[0]:]
    assert (morning_only[:, 0] <= 27).all()


def test_raises_when_no_candidates_fit():
    with pytest.raises(ValueError):
        build_candidates(total_blocks=50, start_block=90, end_block=96)


def test_candidate_rows_sum_to_total_blocks_length():
    cand = build_candidates(total_blocks=16, start_block=1, end_block=96, min_split=4)
    assert cand.shape[1] == 16

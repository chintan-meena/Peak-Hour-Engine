import numpy as np
import pytest

from peak_hours.benchmarking import ex_post_optimal as epo


def test_block_time_roundtrip():
    assert epo.block_to_time(1) == "00:00"
    assert epo.block_to_time(97) == "24:00"
    assert epo.time_to_block("00:00") == 1
    assert epo.time_to_block("18:30") == epo.time_to_block("18:30")
    assert epo.time_to_block(epo.block_to_time(75)) == 75


def test_blocks_from_peak_hours_single_range():
    # 18:30-20:30 -> blocks covering [18:30, 20:30), 15-min each
    blocks = epo.blocks_from_peak_hours("18:30-20:30")
    assert blocks == tuple(range(epo.time_to_block("18:30"), epo.time_to_block("20:30")))
    assert len(blocks) == 8  # 2 hours = 8 blocks


def test_blocks_from_peak_hours_split_and_midnight():
    blocks = epo.blocks_from_peak_hours("22:00-24:00, 00:00-01:00")
    # 22:00-24:00 is 8 blocks ending at block 96; 00:00-01:00 is 4 blocks from block 1
    assert 96 in blocks
    assert 1 in blocks
    assert len(blocks) == 12


def test_ranges_from_blocks_merges_contiguous():
    blocks = list(range(epo.time_to_block("18:30"), epo.time_to_block("20:30")))
    text = epo.ranges_from_blocks(blocks)
    assert text == "18:30-20:30"


def test_ranges_from_blocks_splits_gap():
    a = epo.blocks_from_peak_hours("18:30-20:30")
    b = epo.blocks_from_peak_hours("21:30-22:30")
    text = epo.ranges_from_blocks(list(a) + list(b))
    assert text == "18:30-20:30, 21:30-22:30"


def test_ranges_from_blocks_empty():
    assert epo.ranges_from_blocks([]) == ""


def test_build_candidates_continuous_count():
    # total_blocks=4, full day (1..96), no split possible if min_split > total//2
    cand = epo.build_candidates(total_blocks=4, start_block=1, end_block=96, min_split=4)
    # only continuous windows possible since min_split*2 > total_blocks
    assert cand.shape[1] == 4
    assert cand.shape[0] == 96 - 4 + 1  # 93 continuous windows, no valid split


def test_build_candidates_includes_splits():
    cand = epo.build_candidates(total_blocks=8, start_block=1, end_block=20, min_split=4)
    # each row must sum lengths to 8 and be 0-based, within [0, 19]
    assert cand.shape[1] == 8
    assert cand.min() >= 0
    assert cand.max() <= 19
    # continuous-only count for this range
    continuous_count = 20 - 8 + 1
    assert cand.shape[0] > continuous_count  # splits exist beyond continuous


def test_best_of_picks_argmax():
    cand = np.array([[0, 1], [2, 3], [4, 5]])
    vals = np.array([1.0, 1.0, 10.0, 10.0, 0.0, 0.0])
    blocks, value = epo.best_of(cand, vals)
    assert blocks == (3, 4)  # 1-based for candidate [2,3]
    assert value == pytest.approx(20.0)

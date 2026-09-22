import numpy as np
import pytest

from peak_hours.engine.select import best_window


def test_best_window_picks_argmax():
    candidates = np.array([[0, 1], [2, 3], [4, 5]])
    scores = np.array([1.0, 1.0, 10.0, 10.0, 0.0, 0.0])
    blocks, value = best_window(candidates, scores)
    assert blocks == (3, 4)          # 1-based rendering of candidate [2,3]
    assert value == pytest.approx(20.0)


def test_best_window_ties_pick_first_occurrence():
    candidates = np.array([[0, 1], [2, 3]])
    scores = np.array([5.0, 5.0, 5.0, 5.0])   # both candidates score 10
    blocks, value = best_window(candidates, scores)
    assert blocks == (1, 2)          # first of the tied candidates
    assert value == pytest.approx(10.0)

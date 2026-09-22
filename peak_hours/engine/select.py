"""Pick the best-scoring candidate window.

Generalizes `ex_post_optimal.best_of()` / `hydro_peak_model.select()`'s
inline argmax / the notebook's `select_window_from_block_score()` -- all
three do exactly this: sum a per-block score over each candidate's blocks,
take the argmax.

Kept self-contained (not importing `benchmarking.ex_post_optimal.best_of`)
for now even though it's the same three lines -- `engine` is meant to become
the canonical source these consume once Segments 3-4 wire the benchmarking
modules onto it, not the other way around, so this module shouldn't take a
dependency on `benchmarking` in the meantime.
"""
from __future__ import annotations

import numpy as np


def best_window(candidates: np.ndarray, block_scores: np.ndarray) -> tuple[tuple[int, ...], float]:
    """(blocks, 1-based) and total score of the highest-scoring candidate.

    `candidates` is the (n_candidates, window_length) 0-based array from
    `engine.windows.build_candidates`. `block_scores` is a length-96 array
    indexed by 0-based block.
    """
    totals = block_scores[candidates].sum(axis=1)
    i = int(totals.argmax())
    return tuple(int(b) + 1 for b in candidates[i]), float(totals[i])

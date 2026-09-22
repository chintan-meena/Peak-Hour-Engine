"""Candidate peak-hour windows: continuous, evening-split, and morning+evening-split.

Retires three near-duplicate implementations that had drifted apart:
`ex_post_optimal.build_candidates` (continuous + evening-split only, no
morning band), `hydro_peak_model.build_candidates` (adds the morning band,
used because 33/36 winter hydro declarations carry a morning segment), and
the notebook SECTION 23's inline `_build_candidate_windows` (continuous +
split only -- the reason today's thermal declarations can't express a
morning segment even though 19/72 real historical ones have one, per
`benchmark_declarations.py`'s own finding). This is the superset of all
three: pass `morning_band=None` to get the thermal-today behaviour, or a
band to get hydro's.
"""
from __future__ import annotations

import numpy as np


def build_candidates(
    total_blocks: int,
    start_block: int = 1,
    end_block: int = 96,
    min_split: int = 4,
    morning_band: tuple[int, int] | None = None,
) -> np.ndarray:
    """All candidate windows of exactly `total_blocks`, as 0-based block indices.

    Returns an (n_candidates, total_blocks) array so a whole month can be
    scored with one `vals[cand].sum(axis=1)` rather than per-candidate loops.

    Three shapes, in order:
      1. Continuous: a single run of `total_blocks` starting anywhere in
         [start_block, end_block].
      2. Evening split: two runs (each >= min_split blocks, summing to
         total_blocks), both anywhere in [start_block, end_block], separated
         by at least one block.
      3. Morning+evening split (only if `morning_band` given): the FIRST run
         restricted to `morning_band = (lo, hi)`, the second free in
         [start_block, end_block] -- the shape winter hydro declarations use
         (see hydro_peak_model.py's docstring for the physical reasoning).
    """
    out = []
    for s in range(start_block, end_block + 1):
        if s + total_blocks - 1 > end_block:
            break
        out.append(list(range(s, s + total_blocks)))

    for len1 in range(min_split, total_blocks - min_split + 1):
        len2 = total_blocks - len1
        for s1 in range(start_block, end_block + 1):
            e1 = s1 + len1 - 1
            if e1 > end_block:
                break
            for s2 in range(e1 + 2, end_block + 1):
                if s2 + len2 - 1 > end_block:
                    break
                out.append(list(range(s1, s1 + len1)) + list(range(s2, s2 + len2)))

    if morning_band:
        m_lo, m_hi = morning_band
        for len1 in range(min_split, total_blocks - min_split + 1):
            len2 = total_blocks - len1
            for s1 in range(m_lo, m_hi - len1 + 2):
                for s2 in range(start_block, end_block - len2 + 2):
                    out.append(list(range(s1, s1 + len1)) + list(range(s2, s2 + len2)))

    if not out:
        raise ValueError(
            f"no candidate windows of {total_blocks} blocks fit in "
            f"[{start_block}, {end_block}] with min_split={min_split}"
        )
    return np.asarray(out, dtype=np.int32) - 1        # to 0-based

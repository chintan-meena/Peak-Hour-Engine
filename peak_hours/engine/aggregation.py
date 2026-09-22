"""Aggregating a per-day, per-block score into one monthly per-block score.

Generalizes the 6 method-independent aggregators from the notebook SECTION
23's `build_block_score_methods()` -- `mean`, `median`, `p75`, `p90`,
`daily_top_frequency`, `recency_weighted`. Each takes a long-format frame
with one row per (day, block) and a score column, and returns a raw
(not yet normalised) per-block Series indexed 1-96.

Two methods from the original notebook are deliberately NOT ported here:
`rtm_pressure` and `robust_blend` are thermal-specific compositions over
*multiple* named sub-scores (RTM_Peak_Score, DAM_Market_Pressure_Score,
Purchase_Volume_Score, Seasonal_Block_Score) that only exist once Segment 4's
thermal feature pipeline is built and audited -- porting their hand-set
composition weights here, unaudited, would defeat the point of that audit.
They can be added as thin functions over this module's primitives once
Segment 4 lands, if they survive it.

None of these 6 are claimed to be *good* choices -- the notebook's own
default (`recency_weighted`) was picked with no ablation, and the 7 methods
are documented to disagree by up to 3 hours. They're kept selectable, for
now, only as inputs to the advisory signal (Segment 5/6); Segment 5's
seasonal-baseline primary method does not use any of them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from peak_hours.engine.scoring import robust_z

ALL_BLOCKS = range(1, 97)


def normalize(score: pd.Series) -> np.ndarray:
    """Median-fill any missing block, then robust-z. Ported from the
    notebook's `_normalize_block_score`."""
    s = pd.Series(score, index=ALL_BLOCKS, dtype=float).replace([np.inf, -np.inf], np.nan)
    s = s.fillna(s.median()).fillna(0)
    return robust_z(s.to_numpy())


def mean(scored: pd.DataFrame, score_col: str = "Score", block_col: str = "Block") -> pd.Series:
    return scored.groupby(block_col)[score_col].mean().reindex(ALL_BLOCKS)


def median(scored: pd.DataFrame, score_col: str = "Score", block_col: str = "Block") -> pd.Series:
    return scored.groupby(block_col)[score_col].median().reindex(ALL_BLOCKS)


def p75(scored: pd.DataFrame, score_col: str = "Score", block_col: str = "Block") -> pd.Series:
    return scored.groupby(block_col)[score_col].quantile(0.75).reindex(ALL_BLOCKS)


def p90(scored: pd.DataFrame, score_col: str = "Score", block_col: str = "Block") -> pd.Series:
    return scored.groupby(block_col)[score_col].quantile(0.90).reindex(ALL_BLOCKS)


def daily_top_frequency(
    scored: pd.DataFrame,
    total_blocks: int,
    score_col: str = "Score",
    block_col: str = "Block",
    date_col: str = "Date",
) -> pd.Series:
    """For each day, how often does a block rank in that day's top `total_blocks`?

    `total_blocks` must be the actual declaration window length (16 for
    thermal, 12 for hydro) -- the notebook hardcoded this to 12 in the one
    place it's called from, which silently matched hydro's window length,
    not thermal's real 16-block practice (see benchmark_declarations.py's
    `EXPECTED_BLOCKS`). Passing it explicitly here is the fix.
    """
    s = scored.copy()
    s["_rank"] = s.groupby(date_col)[score_col].rank(method="first", ascending=False)
    n_days = max(s[date_col].nunique(), 1)
    return (
        s[s["_rank"] <= total_blocks]
        .groupby(block_col).size()
        .reindex(ALL_BLOCKS, fill_value=0)
        / n_days
    )


def recency_weighted(
    scored: pd.DataFrame,
    reference_date,
    half_life_days: float,
    score_col: str = "Score",
    block_col: str = "Block",
    date_col: str = "Date",
) -> pd.Series:
    """Weighted mean per block; a day's weight halves every `half_life_days`
    of distance from `reference_date` (e.g. the declaration's own END_DATE --
    days closer to real data count more, since forecast quality degrades
    with distance from it)."""
    s = scored.copy()
    days_from_ref = (pd.to_datetime(s[date_col]) - pd.Timestamp(reference_date)).dt.days
    s["_weight"] = 0.5 ** (days_from_ref / half_life_days)
    return (
        s.groupby(block_col)
        .apply(lambda g: np.average(g[score_col], weights=g["_weight"]), include_groups=False)
        .reindex(ALL_BLOCKS)
    )


# Name -> (callable, extra_kwargs_required). Callers needing every method
# side by side (e.g. an advisory-signal comparison) can iterate this rather
# than hand-listing the 4 parameterless ones and special-casing the other 2.
METHODS = {
    "mean": mean,
    "median": median,
    "p75": p75,
    "p90": p90,
    "daily_top_frequency": daily_top_frequency,
    "recency_weighted": recency_weighted,
}

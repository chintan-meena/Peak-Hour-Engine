"""Block scoring: turning a per-block profile into a comparable score.

`robust_z` is the normalisation both `hydro_peak_model.py` and the notebook
already use (median/MAD z-score, clipped to +-5 so one spike can't hijack a
window) -- ported here verbatim since it was previously duplicated in both
places rather than shared.

Only `rtm_only` is provided as a ready-made scorer here: it's
`hydro_peak_model.py`'s own scorer (RTM price alone, no weights, no
product), and its docstring documents *why* -- an out-of-sample ablation
showed price alone beats a Net_Load x RTM interaction on frequency-stress
capture (64.79% vs 62.82%, p=0.002), because RTM already prices in load.

The thermal side's `Peak_Score` (a hand-weighted blend of ~8 sub-scores,
0.70/0.08/0.15/0.07 across NL x RTM interaction, net-load excess+ramp, RTM
price+ramp, and DAM market pressure) is deliberately NOT ported here. That
whole formula is what Segment 4's audit exists to test -- porting it
unchanged into "core engine" code would smuggle every one of its unproven
weights past the ablation it's supposed to survive first. A scorer is just
any `profile -> length-96 np.ndarray` callable; Segment 4 adds thermal-specific
scorers here (or in a sibling module) only as each survives its ablation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def robust_z(a: np.ndarray) -> np.ndarray:
    """Median/MAD z-score, clipped to +-5. Falls back to std if MAD is 0/NaN."""
    med = np.nanmedian(a)
    mad = np.nanmedian(np.abs(a - med))
    denom = 1.4826 * mad if mad and np.isfinite(mad) else np.nanstd(a)
    if not denom or not np.isfinite(denom):
        return np.zeros_like(a, dtype=float)
    return np.clip((a - med) / denom, -5, 5)


def rtm_only(profile: pd.DataFrame, price_col: str = "RTM") -> np.ndarray:
    """Score = RTM price alone, robust-z'd. See module docstring for why."""
    return robust_z(profile[price_col].to_numpy())

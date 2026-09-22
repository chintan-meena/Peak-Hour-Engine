"""The evening/morning ramp gate: where candidate windows are allowed to start.

Generalizes `hydro_peak_model.derive_gate()` (already written as a general
trough->peak->half-rise-crossing rule, used there for both the evening gate
and, when a morning ramp exists, the morning gate) and the notebook SECTION
23's inline version (hardcoded to the evening-only case, under the
misleading name `HYDRO_RAMP_UP_MINUTES` even in the thermal path -- a naming
smell fixed here by naming the ramp-up parameter for what it actually is:
the time a generator/plant needs to reach full output after a window opens,
independent of whether that plant is hydro or thermal).

Physical/behavioural reasoning (from the request that produced the original
rule): net load is low in the afternoon and rises into an evening peak.
Peak hours should only start once that rise has passed its half point, and
even then a plant is not at full output instantly -- so the earliest
admissible start is pushed later by a ramp-up allowance.
"""
from __future__ import annotations

import numpy as np


def derive_gate(curve: np.ndarray, lo: int, hi: int, peak_hi: int,
                 frac: float = 0.5) -> int | None:
    """Trough in [lo, hi] -> the following peak (up to peak_hi) -> the first
    block where the rise from trough to peak crosses `frac` of the way up.

    Returns None if there's no usable trough/peak in the given range (e.g.
    an all-NaN segment, or no rise at all) -- callers decide the fallback.
    """
    seg = curve[lo - 1:hi]
    if not len(seg) or np.all(np.isnan(seg)):
        return None
    trough = int(np.nanargmin(seg)) + lo

    tail = curve[trough - 1:peak_hi]
    if not len(tail) or np.all(np.isnan(tail)):
        return None
    peak = int(np.nanargmax(tail)) + trough
    if peak <= trough:
        return None

    half = curve[trough - 1] + frac * (curve[peak - 1] - curve[trough - 1])
    rise = curve[trough - 1:peak]
    cross = np.flatnonzero(rise >= half)
    return (int(cross[0]) + trough) if len(cross) else trough


def ramp_gate(
    curve: np.ndarray,
    search_lo: int,
    search_hi: int,
    peak_hi: int,
    ramp_up_minutes: int = 45,
    block_minutes: int = 15,
    end_block: int = 96,
    frac: float = 0.5,
    fallback: int | None = None,
) -> int | None:
    """`derive_gate()` plus the ramp-up-allowance shift, capped at `end_block`.

    This is the one gate computation both hydro's `select()` (evening gate,
    fallback=72; morning gate, no fallback -> None if no morning ramp exists)
    and the notebook's SECTION 23 (evening gate only, fallback folded into
    the caller) were each doing by hand. `ramp_up_minutes` defaults to the
    45-minute hydro ramp-up figure but is a real, named parameter now rather
    than a hardcoded constant borrowing hydro's name in the thermal path.
    """
    gate = derive_gate(curve, search_lo, search_hi, peak_hi, frac=frac)
    if gate is None:
        return fallback
    ramp_up_blocks = int(round(ramp_up_minutes / block_minutes))
    return min(gate + ramp_up_blocks, end_block)

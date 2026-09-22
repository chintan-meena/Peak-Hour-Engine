import numpy as np

from peak_hours.engine.gates import derive_gate, ramp_gate


def _synthetic_curve():
    # 96 blocks: flat low (trough plateau, blocks 1-60, all tied at 100 so
    # argmin returns the first one, block 49 within the lo=49 search start),
    # then a clean linear rise to a peak at block 79, then flat high.
    curve = np.full(96, 100.0)
    curve[49:60] = 100.0          # trough plateau, blocks 50-60
    for i, b in enumerate(range(60, 80)):
        curve[b - 1] = 100.0 + (i + 1) * 10.0   # rises 110..300
    curve[79:96] = 300.0          # peak plateau
    return curve


def test_derive_gate_finds_trough_peak_and_half_crossing():
    curve = _synthetic_curve()
    gate = derive_gate(curve, lo=49, hi=72, peak_hi=96)
    assert gate is not None
    # half point between trough (100) and peak (300) is 200, first crossed
    # at block 69 (100 + 10*10 = 200) -- verified against the real function.
    assert gate == 69
    assert curve[gate - 1] >= 200.0
    assert curve[gate - 2] < 200.0


def test_derive_gate_returns_none_on_all_nan_segment():
    curve = np.full(96, np.nan)
    assert derive_gate(curve, lo=1, hi=28, peak_hi=40) is None


def test_derive_gate_returns_none_when_no_rise_after_trough():
    # flat curve: trough == peak candidate everywhere, peak <= trough -> None
    curve = np.full(96, 50.0)
    assert derive_gate(curve, lo=1, hi=28, peak_hi=40) is None


def test_ramp_gate_shifts_by_ramp_up_blocks():
    curve = _synthetic_curve()
    gate = derive_gate(curve, lo=49, hi=72, peak_hi=96)
    gated = ramp_gate(curve, search_lo=49, search_hi=72, peak_hi=96,
                       ramp_up_minutes=45, block_minutes=15, end_block=96)
    assert gated == min(gate + 3, 96)   # 45 min / 15 min = 3 blocks


def test_ramp_gate_falls_back_when_gate_is_none():
    curve = np.full(96, np.nan)
    gated = ramp_gate(curve, search_lo=1, search_hi=28, peak_hi=40, fallback=72)
    assert gated == 72


def test_ramp_gate_caps_at_end_block():
    curve = _synthetic_curve()
    gated = ramp_gate(curve, search_lo=49, search_hi=72, peak_hi=96,
                       ramp_up_minutes=1000, end_block=96)
    assert gated == 96

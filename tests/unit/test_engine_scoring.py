import numpy as np
import pandas as pd

from peak_hours.engine.scoring import robust_z, rtm_only


def test_robust_z_centers_on_median():
    a = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    z = robust_z(a)
    # the median (30) should map to ~0
    assert abs(z[2]) < 1e-9


def test_robust_z_clips_to_plus_minus_5():
    a = np.array([0.0, 1.0, 1.0, 1.0, 1000000.0])
    z = robust_z(a)
    assert z.max() <= 5.0
    assert z.min() >= -5.0


def test_robust_z_zero_mad_falls_back_to_std():
    # all but one value identical -> MAD is 0, must not divide by zero / NaN
    a = np.array([5.0, 5.0, 5.0, 5.0, 9.0])
    z = robust_z(a)
    assert np.all(np.isfinite(z))


def test_robust_z_constant_array_returns_zeros():
    a = np.array([7.0, 7.0, 7.0, 7.0])
    z = robust_z(a)
    assert np.all(z == 0.0)


def test_rtm_only_scores_from_price_column():
    profile = pd.DataFrame({"RTM": [100.0, 200.0, 300.0, 400.0, 500.0]})
    z = rtm_only(profile)
    assert len(z) == 5
    # higher price -> higher (or equal) score, monotonic since robust_z is
    # a monotonic transform of the input
    assert (np.diff(z) >= 0).all()

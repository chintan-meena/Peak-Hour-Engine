"""The vendored path fallback must answer exactly what the real library does.

`peak_hours._powertools_fallback` exists so a clone of this repo runs without
the power-libraries checkout. The danger of any such copy is drift: the shared
library gets a fix, the copy doesn't, and the two quietly resolve to different
directories. That has already happened once in this project -- scada-cache's
own `powertools_home()` skipped OneDrive detection and read a local-only cache
the MacBook could never see, and nothing failed, it just returned the wrong
data.

So these tests compare the two implementations directly, under a range of
environments, whenever the real library is importable. On a machine that
doesn't have it (the situation the fallback is *for*) they skip -- there is
nothing to compare against there.
"""

from __future__ import annotations

import importlib

import pytest

from peak_hours import _powertools_fallback as fallback

real = pytest.importorskip(
    "powertools_common.paths",
    reason="power-libraries not on this machine -- nothing to compare the fallback to",
)


def test_exports_every_function_peak_hours_uses():
    """paths.py and scada_cache.config import these three by name."""
    for name in ("cache_dir", "reports_dir", "powertools_home"):
        assert hasattr(fallback, name), f"fallback is missing {name}()"


@pytest.mark.parametrize(
    "env",
    [
        pytest.param({}, id="ambient"),
        pytest.param({"POWERTOOLS_HOME": "/tmp/ph-test-home"}, id="explicit-home"),
        pytest.param(
            {"POWERTOOLS_HOME": None, "POWERTOOLS_ONEDRIVE": None,
             "OneDriveCommercial": None, "OneDrive": None},
            id="nothing-configured",
        ),
    ],
)
def test_resolves_identically_to_the_shared_library(env, monkeypatch, tmp_path):
    """Same inputs, same answer -- for every root the engine depends on."""
    for var, value in env.items():
        if value is None:
            monkeypatch.delenv(var, raising=False)
        else:
            monkeypatch.setenv(var, str(tmp_path / "home"))

    assert fallback.onedrive_root() == real.onedrive_root()
    assert fallback.powertools_home() == real.powertools_home()
    assert fallback.local_root() == real.local_root()

    # create=False so comparing paths never makes directories as a side effect.
    assert fallback.cache_dir("peak_hours", create=False) == real.cache_dir(
        "peak_hours", create=False
    )
    assert fallback.reports_dir(
        "Peak_Hour_Engine", "monthly_peak_pipeline_outputs", create=False
    ) == real.reports_dir(
        "Peak_Hour_Engine", "monthly_peak_pipeline_outputs", create=False
    )
    assert fallback.scratch_dir("Peak_Hour_Engine", create=False) == real.scratch_dir(
        "Peak_Hour_Engine", create=False
    )


def test_paths_module_prefers_the_shared_library():
    """When power-libraries is present it must win -- the fallback is a backup,
    not a fork. Checked by identity: the names peak_hours.paths bound are the
    real library's objects, not the copy's."""
    paths = importlib.import_module("peak_hours.paths")
    import powertools_common

    assert paths.cache_dir is powertools_common.cache_dir
    assert paths.reports_dir is powertools_common.reports_dir

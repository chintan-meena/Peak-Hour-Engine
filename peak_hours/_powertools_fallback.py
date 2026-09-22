"""Standalone copy of the path resolution this engine needs.

The real implementation is `powertools_common.paths` in the `power-libraries`
checkout, shared by every power-tools program on both machines. This module is
a fallback used only when that checkout isn't importable -- a fresh clone of
Peak_Hour_Engine on its own, most likely the MacBook -- so the engine is a
genuinely standalone project rather than one that dies at import time on any
machine that doesn't also have power-libraries on PYTHONPATH.

`powertools_common` still wins whenever it's present: it is the single source
of truth, and `peak_hours.paths` prefers it (see the import there). This copy
exists to keep the engine running without it, not to replace it.

Keeping a second copy of path logic is exactly how the scada-cache OneDrive bug
happened -- `scada_cache/config.py` had its own `powertools_home()` that had
quietly stopped matching the shared one, so it read a local-only cache the
MacBook could never see. `tests/unit/test_powertools_fallback.py` therefore
asserts this module resolves every path identically to the real library
whenever both are importable, so drift fails a test instead of silently
splitting the data directory in two.

Only the four functions the engine actually calls are copied. `sibling_repo()`
is deliberately left out: it locates a checkout *relative to the library's own
file*, which would mean something different from inside this package, and
nothing here uses it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = [
    "onedrive_root",
    "powertools_home",
    "local_root",
    "reports_dir",
    "scratch_dir",
    "cache_dir",
]

CONFIG_DIR = Path.home() / ".power_tools"

_IS_MACOS = sys.platform == "darwin"


def onedrive_root():
    """The business OneDrive folder on this machine, or ``None``.

    Windows populates ``%OneDriveCommercial%``/``%OneDrive%`` itself; macOS has
    no such variable but puts the folder under ``~/Library/CloudStorage``. Only
    a directory that actually exists is accepted -- a stale, non-syncing
    ``~/OneDrive`` once collected a cache nobody could see from the other
    machine.
    """
    for var in ("POWERTOOLS_ONEDRIVE", "OneDriveCommercial", "OneDrive"):
        value = os.environ.get(var)
        if value:
            candidate = Path(value)
            if candidate.is_dir():
                return candidate

    if _IS_MACOS:
        cloud = Path.home() / "Library" / "CloudStorage"
        if cloud.is_dir():
            # A work tenant looks like OneDrive-grid-india.in; a personal
            # account is plain "OneDrive-Personal". Prefer the work one.
            business = sorted(
                p for p in cloud.glob("OneDrive-*")
                if p.is_dir() and p.name != "OneDrive-Personal"
            )
            if business:
                return business[0]

            personal = cloud / "OneDrive-Personal"
            if personal.is_dir():
                return personal

    return None


def powertools_home():
    """Root of the shared data area: ``$POWERTOOLS_HOME``, else
    ``<OneDrive>/PowerTools``, else ``~/PowerTools``."""
    custom = os.environ.get("POWERTOOLS_HOME")
    if custom:
        return Path(custom)

    onedrive = onedrive_root()
    if onedrive is not None:
        return onedrive / "PowerTools"

    return Path.home() / "PowerTools"


def local_root():
    """``~/.power_tools`` -- this machine only, never synced."""
    return CONFIG_DIR


def _resolve(base, parts, create):
    path = base.joinpath(*parts) if parts else base
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def reports_dir(program, *sub, create=True):
    """``<PowerTools>/reports/<program>/`` -- finished work, synced."""
    return _resolve(powertools_home() / "reports" / program, sub, create)


def scratch_dir(program, *sub, create=True):
    """``~/.power_tools/scratch/<program>/`` -- bulk intermediates, local."""
    return _resolve(local_root() / "scratch" / program, sub, create)


def cache_dir(name, *sub, create=True):
    """``<PowerTools>/cache/<name>/`` -- parsed data worth keeping, synced."""
    return _resolve(powertools_home() / "cache" / name, sub, create)

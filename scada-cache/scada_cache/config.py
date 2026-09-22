"""Load configuration and resolve the right network path for *this* machine.

The whole point of this file is to make the project switch cleanly between a
MacBook and a Windows PC. The same ``config.json`` is committed to git; each
machine just reads the path that matches its own operating system:

    Windows : \\\\192.168.50.247\\scadashare\\Stack 5minute
    macOS   : /Volumes/scadashare/Stack 5minute   (after connecting the share)
    Linux   : /mnt/scadashare/Stack 5minute       (however you mounted it)

Per-machine or one-off overrides never need a code change:

* Set an env var ``SCADA_<SOURCE>_DIR`` (e.g. ``SCADA_STATE_DEMAND_DIR``) to
  point a single source somewhere else for one run.
* Or drop a ``config.local.json`` next to ``config.json`` (git-ignored) whose
  keys shallow-merge over the committed file.
"""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path

# Repo root = one level above this package directory.
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

_OS_KEY = {"windows": "windows", "darwin": "darwin", "linux": "linux"}.get(
    platform.system().lower(), "linux"
)


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(config_path=None) -> dict:
    """Read ``config.json`` (+ optional ``config.local.json``) and return it."""
    config_path = Path(config_path) if config_path else REPO_ROOT / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config found at {config_path}. Copy config.example.json to "
            f"config.json and set your paths."
        )
    cfg = json.loads(config_path.read_text())

    local = config_path.with_name("config.local.json")
    if local.exists():
        cfg = _deep_merge(cfg, json.loads(local.read_text()))
    return cfg


def powertools_home() -> Path:
    """Root of the shared PowerTools data area, as the `iex` library defines it.

    ``$POWERTOOLS_HOME`` when set, else ``~/PowerTools``. On both machines that
    location is a OneDrive folder (directly on Windows, via a symlink on the
    MacBook), so whatever is written there syncs between them.
    """
    custom = os.environ.get("POWERTOOLS_HOME")
    if custom:
        return Path(custom)
    return Path.home() / "PowerTools"


def default_cache_dir() -> Path:
    """``$POWERTOOLS_HOME/cache/scada`` - the sibling of ``cache/wbes``."""
    return powertools_home() / "cache" / "scada"


def cache_dir(cfg: dict) -> Path:
    """Absolute path to the parsed-parquet cache directory.

    The cache is NOT tracked by git - it is bulk derived data that would bloat
    every clone, and the same rule already applies to the wbes and iex caches
    in power-libraries. It lives in the shared PowerTools area instead, so the
    other machine still loads warm without the repo carrying 100 MB.

    Resolution order:

    1. ``SCADA_CACHE_DIR`` env var          - one-off override for a single run
    2. an absolute ``cache_dir`` in config  - per-machine via config.local.json
    3. a relative ``cache_dir`` in config   - legacy repo-local layout, honoured
                                              only if that folder actually exists
    4. ``$POWERTOOLS_HOME/cache/scada``     - the default
    """
    env_override = os.environ.get("SCADA_CACHE_DIR")
    if env_override:
        return Path(env_override)

    raw = cfg.get("cache_dir") or ""
    if raw:
        path = Path(raw)
        if path.is_absolute():
            return path
        legacy = REPO_ROOT / path
        if legacy.exists():
            return legacy

    return default_cache_dir()


def source_names(cfg: dict) -> list[str]:
    return list(cfg.get("sources", {}).keys())


def source_spec(cfg: dict, name: str) -> dict:
    try:
        return cfg["sources"][name]
    except KeyError as exc:
        raise KeyError(
            f"Source {name!r} is not in config. Known: {source_names(cfg)}"
        ) from exc


def resolve_source_dir(cfg: dict, name: str):
    """Return the raw-folder Path for a source on this machine, or None.

    Resolution order: ``SCADA_<NAME>_DIR`` env var, then the OS-specific entry
    under ``paths``. Returns ``None`` when neither is configured so callers can
    treat an unreachable/unconfigured share as "nothing new to ingest" rather
    than crashing.
    """
    env_key = f"SCADA_{name.upper()}_DIR"
    if os.environ.get(env_key):
        return Path(os.environ[env_key])

    paths = source_spec(cfg, name).get("paths", {})
    raw = paths.get(_OS_KEY)
    return Path(raw) if raw else None


def current_os_key() -> str:
    return _OS_KEY

"""Path resolution for Peak_Hour_Engine's data.

Peak_Hour_Engine is now a fully standalone repository (it started as a
sibling folder to ML_Peak_Hour_Declaration during the redesign; scada-cache's
code has since been copied in here too, so nothing depends on that sibling
checkout existing on disk any more).

The data itself -- market price CSVs, RLDC's declaration record, and the
generated monthly outputs -- lives in the shared OneDrive PowerTools area via
`powertools_common`, so both this machine and the MacBook see the same files
without either duplicating them in git or depending on a sibling checkout.
The SCADA parquet cache is the one exception that still resolves itself via
`scada_cache/config.py` (also OneDrive, `$POWERTOOLS_HOME/cache/scada`).
"""
from __future__ import annotations

from pathlib import Path

from powertools_common import cache_dir, reports_dir

PACKAGE_DIR = Path(__file__).resolve().parent
ENGINE_DIR = PACKAGE_DIR.parent

# scada-cache's code lives in this repo now (copied in for self-containment);
# the parquet bytes it reads resolve to OneDrive on their own.
SCADA_DIR = ENGINE_DIR / "scada-cache"

# RTM/DAM price CSVs + Previous_Declarations* -- rebuildable but expensive to
# re-pull, so cached rather than treated as scratch.
MARKET_CACHE_DIR = cache_dir("peak_hours")

# The monthly pipeline's generated output (declarations, benchmarks, weather,
# run history) -- finished work, synced via OneDrive so it shows up on the
# other machine too.
PIPELINE_ARTIFACTS = reports_dir("Peak_Hour_Engine", "monthly_peak_pipeline_outputs")

# Peak_Hour_Engine's own generated outputs (run history, archived results).
# Local-only and separate from PIPELINE_ARTIFACTS -- run provenance doesn't
# need to sync across machines the way finished pipeline output does.
ENGINE_ARTIFACTS = ENGINE_DIR / "artifacts"

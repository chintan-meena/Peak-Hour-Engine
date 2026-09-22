"""scada_cache — fast, incremental, cross-platform cache for NRLDC Stack exports.

Typical use::

    import scada_cache as sc

    sc.update_all()                       # ingest any new raw CSVs (no-op off-LAN)
    states = sc.load("state_demand")      # instant read from parquet
    net    = sc.load_net_load(freq="15min")
"""

from .cache import (
    cache_status,
    load,
    load_net_load,
    update_all,
    update_source,
)
from .config import load_config
from .parser import parse_stack_csv

__all__ = [
    "update_all",
    "update_source",
    "load",
    "load_net_load",
    "cache_status",
    "load_config",
    "parse_stack_csv",
]

__version__ = "0.1.0"

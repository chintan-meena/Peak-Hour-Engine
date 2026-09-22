#!/usr/bin/env python3
"""Refresh the SCADA cache from the network shares, then print a status table.

Run this whenever you want to pull in newly-appeared raw files:

    python update_cache.py                 # update every source
    python update_cache.py --source nr     # just one source
    python update_cache.py --status        # don't ingest, just show what's cached

Safe to run on a machine that can't see the share — it simply reports the
folder as unreachable and leaves the existing (e.g. committed) cache in place.
"""

from __future__ import annotations

import argparse

import scada_cache as sc
from scada_cache import config as cfg_mod


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="Only update this source (default: all).")
    parser.add_argument(
        "--status", action="store_true", help="Show cache status without ingesting."
    )
    args = parser.parse_args()

    cfg = sc.load_config()

    if args.status:
        print(sc.cache_status(cfg).to_string(index=False))
        return

    if args.source:
        sc.update_source(args.source, cfg)
    else:
        sc.update_all(cfg)

    print("\nCache status")
    print(sc.cache_status(cfg).to_string(index=False))


if __name__ == "__main__":
    main()

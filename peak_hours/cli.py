"""Single CLI entry point for peak_hours.

Placeholder for Segment 1 -- only wires up what Segment 1 actually built
(the ex-post-optimal benchmark). declare/backtest/register subcommands are
added in later segments as the engine/forecasting pieces land.
"""
from __future__ import annotations

import sys

from peak_hours.benchmarking import ex_post_optimal
from peak_hours.provenance import registry


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"benchmark", "register"}:
        print("usage: peak-hours {benchmark,register} ...\n"
              "  (more subcommands land as later migration segments complete)")
        raise SystemExit(1)
    sub, sys.argv[1:] = sys.argv[1], sys.argv[2:]
    {"benchmark": ex_post_optimal.main, "register": registry.main}[sub]()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score several candidate declarations for one month side by side.

Built for the model-replay experiment `HANDOFF_v4.md` itself flags as
missing: *"the pipeline's own declarations are still not scored against
that 99.8% baseline. This is the experiment the paper turns on."*

Where `ex_post_optimal.run()` scores exactly one declaration per month (read
from `Previous_Declarations.csv`, one row per month), this module scores any
number of *named* candidates for the *same* month at once -- e.g. a model's
frozen forecast-horizon recommendation, the window actually filed, and the
zero-parameter baselines already validated elsewhere in this codebase
(`hydro_peak_model.py`, `HANDOFF_v4.md`) -- and it works on a month that is
still in progress, scoring against actuals for the elapsed days only. It
reuses `ex_post_optimal.build_candidates` / `best_of` / `load_actuals`
rather than redefining them, so both modules search and score the same way.

    from peak_hours.benchmarking.replay_scorecard import score_month
    score_month("2026-09", {
        "model (frozen Aug-22)": "19:00-21:00, 21:45-22:45",
        "actual filed":          "18:15-20:15, 21:30-23:30",
    })

    python -m peak_hours.benchmarking.replay_scorecard 2026-09 \\
        --candidate "model (frozen Aug-22)=19:00-21:00, 21:45-22:45" \\
        --candidate "actual filed=18:15-20:15, 21:30-23:30"

A month is never silently treated as complete: every row carries
`Days_Scored` / `Days_In_Month`, so a 17-day-in read is never mistaken for a
final one.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from peak_hours.benchmarking.ex_post_optimal import (
    best_of,
    blocks_from_peak_hours,
    build_candidates,
    load_actuals,
    ranges_from_blocks,
)


def _block_means(g: pd.DataFrame, col: str) -> np.ndarray:
    return (g.groupby("Block")[col].mean()
            .reindex(range(1, 97)).fillna(0.0).to_numpy())


def _month_actuals(act: pd.DataFrame, month: str):
    g = act[act["Month"] == month]
    days_scored = int(g["Datetime"].dt.normalize().nunique())
    days_in_month = pd.Period(month, freq="M").days_in_month
    nl_rtm = _block_means(g, "NL_RTM")
    rtm = _block_means(g, "RTM_Price")
    return nl_rtm, rtm, days_scored, days_in_month


def score_month(month: str, candidates: dict[str, str], *, gate_start: int = 72,
                 with_baselines: bool = True, act: pd.DataFrame | None = None) -> pd.DataFrame:
    """Score `candidates` (label -> 'HH:MM-HH:MM, ...') for `month` against actuals.

    Each candidate is scored against the ex-post optimum of its OWN length --
    a longer window collects strictly more, so mixing lengths is not a
    percentage (the same reasoning `ex_post_optimal.best_window`/`run` already
    document). `with_baselines=True` adds, once per distinct length present:
    last year's same-month optimum, and an RTM-price-only ranking (the
    zero-parameter selector `hydro_peak_model.py` validated as outperforming
    a hand-weighted score out-of-sample).
    """
    if act is None:
        act = load_actuals()
    nl_rtm_vals, rtm_vals, days_scored, days_in_month = _month_actuals(act, month)
    if days_scored == 0:
        raise RuntimeError(f"no cached actuals for {month} yet")

    cand_cache: dict[tuple, np.ndarray] = {}

    def cands(length: int, start_block: int = 1) -> np.ndarray:
        key = (length, start_block)
        if key not in cand_cache:
            cand_cache[key] = build_candidates(length, start_block=start_block)
        return cand_cache[key]

    def window_value(blocks) -> float:
        return float(nl_rtm_vals[[b - 1 for b in blocks]].sum())

    prev_month = str(pd.Period(month, freq="M") - 12)
    prev_g = act[act["Month"] == prev_month]
    prev_nl_rtm = _block_means(prev_g, "NL_RTM") if len(prev_g) else None

    rows: list[dict] = []
    lengths_done: set[int] = set()

    for label, hours in candidates.items():
        if not hours:
            continue
        blocks = blocks_from_peak_hours(hours)
        if not blocks:
            continue
        length = len(blocks)
        free = cands(length, 1)
        gated = cands(length, gate_start)
        opt_blocks, opt_v = best_of(free, nl_rtm_vals)
        _, gated_v = best_of(gated, nl_rtm_vals)

        def row(candidate_label: str, candidate_hours: str, candidate_blocks) -> dict:
            v = window_value(candidate_blocks)
            return {
                "Month": month,
                "Days_Scored": days_scored,
                "Days_In_Month": days_in_month,
                "Candidate": candidate_label,
                "Peak_Hours": candidate_hours,
                "Blocks": length,
                "Value_Capture_%": 100.0 * v / opt_v if opt_v else float("nan"),
                "Gated_Optimum_%": 100.0 * gated_v / opt_v if opt_v else float("nan"),
                "Ex_Post_Optimum": ranges_from_blocks(opt_blocks),
            }

        rows.append(row(label, hours, blocks))

        if with_baselines and length not in lengths_done:
            lengths_done.add(length)
            if prev_nl_rtm is not None:
                ly_blocks, _ = best_of(free, prev_nl_rtm)
                rows.append(row(
                    f"baseline: last year's optimum ({length}-block)",
                    ranges_from_blocks(ly_blocks), ly_blocks,
                ))
            rtm_blocks, _ = best_of(free, rtm_vals)
            rows.append(row(
                f"baseline: RTM-price-only ranking ({length}-block)",
                ranges_from_blocks(rtm_blocks), rtm_blocks,
            ))

    if not rows:
        raise RuntimeError("no candidate produced any scoreable blocks")

    return pd.DataFrame(rows)


def _parse_candidate_arg(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        sys.exit(f"error: --candidate must be 'label=HH:MM-HH:MM, ...', got: {raw!r}")
    label, hours = raw.split("=", 1)
    return label.strip(), hours.strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("month", help="YYYY-MM")
    p.add_argument("--candidate", action="append", default=[],
                   metavar="LABEL=HH:MM-HH:MM,...",
                   help="repeatable; a named window to score")
    p.add_argument("--gate-start", type=int, default=72,
                   help="pipeline's earliest admissible start block")
    p.add_argument("--no-baselines", action="store_true",
                   help="skip last-year-optimum / RTM-only-ranking baselines")
    a = p.parse_args()

    if not a.candidate:
        sys.exit("error: pass at least one --candidate 'label=HH:MM-HH:MM, ...'")
    candidates = dict(_parse_candidate_arg(c) for c in a.candidate)

    try:
        r = score_month(a.month, candidates, gate_start=a.gate_start,
                        with_baselines=not a.no_baselines)
    except RuntimeError as exc:
        sys.exit(f"error: {exc}")

    days = r.iloc[0]
    print(f"{a.month}: {int(days.Days_Scored)}/{int(days.Days_In_Month)} days of "
          f"actuals available -- {'PARTIAL MONTH, ' if days.Days_Scored < days.Days_In_Month else ''}"
          f"figures below are not a final month-end score until it is.\n")

    show = ["Candidate", "Peak_Hours", "Blocks", "Value_Capture_%", "Gated_Optimum_%"]
    print(r[show].to_string(index=False,
          formatters={c: "{:.1f}".format for c in show if c.endswith("%")}))


if __name__ == "__main__":
    main()

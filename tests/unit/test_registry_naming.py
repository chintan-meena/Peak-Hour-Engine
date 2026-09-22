"""Declared-capture column naming: NRLDC is correct, NRPC is the legacy name.

Peak hours are declared by NRLDC, the load despatch centre. The legacy hydro
model labelled the "what was actually declared, scored" column `NRPC_%` after
the regional power *committee*, which does not issue the declaration. Files
already written to OneDrive carry that name, so the reader has to accept both
while anything new uses the right one.

These tests pin that behaviour, including the precedence: if a file somehow
carries both columns, the correctly-named one wins.
"""

from __future__ import annotations

import pandas as pd

from peak_hours.provenance.registry import _declared_capture_column


def test_prefers_the_correct_nrldc_name():
    d = pd.DataFrame({"Month": ["2026-01"], "Model_%": [98.0], "NRLDC_%": [90.0]})
    assert _declared_capture_column(d) == "NRLDC_%"


def test_falls_back_to_the_legacy_nrpc_name():
    """Every Hydro_Model_Backtest.csv written before this correction."""
    d = pd.DataFrame({"Month": ["2026-01"], "Model_%": [98.0], "NRPC_%": [90.0]})
    assert _declared_capture_column(d) == "NRPC_%"


def test_nrldc_wins_when_a_file_carries_both():
    """A file straddling the rename must not be scored off the stale column."""
    d = pd.DataFrame({"NRPC_%": [11.0], "NRLDC_%": [90.0]})
    assert _declared_capture_column(d) == "NRLDC_%"


def test_returns_none_when_neither_is_present():
    """Missing is not an error -- harvest() skips the metric rather than
    raising, so a partial pipeline run still records everything else."""
    d = pd.DataFrame({"Month": ["2026-01"], "Model_%": [98.0]})
    assert _declared_capture_column(d) is None

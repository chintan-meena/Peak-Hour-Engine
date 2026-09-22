"""Parse a single raw SCADA Stack export into a tidy, wide DataFrame.

The raw files are messy exports with ~12 rows of calendar/metadata junk at the
top, a friendly-header row, a machine-tag row, then the actual time series.
This module isolates all of that ugliness so the rest of the codebase only ever
sees a clean ``Datetime``-indexed frame.

Layout that the detector expects (indices are auto-detected, not hard-coded, so
the parser survives small template shifts):

    row N   : col0 == "HRS"        -> friendly header row
    row N+1 : machine tags         -> e.g. ...!PSE_DRW!P.MvMoment
    row N+2 : first data row        -> col0 == "HH:MM:SS", then values

Two quirks handled here:

* The date is taken from the *filename* (``DD-MM-YYYY.csv``), never from the
  in-sheet metadata, because the filename is the authoritative, unambiguous
  key and is what the existing v9 pipeline already trusts.

* The 5-minute state file stacks the day twice (00:00->23:55 appears a second
  time with different values, i.e. a second day/scenario in the same sheet).
  We keep the *first* occurrence of each timestamp and drop the rest, so one
  file maps to exactly one calendar day. The 1-minute file has no such
  duplication and is unaffected.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .tags import PROFILES

_TIME_RE = re.compile(r"^\d{1,2}:\d{2}:\d{2}$")


def parse_stack_csv(path, profile: str) -> pd.DataFrame:
    """Parse one raw Stack CSV.

    Parameters
    ----------
    path
        Path to a ``DD-MM-YYYY.csv`` Stack export.
    profile
        ``"state"`` (nine state demands, matched on machine tag) or ``"nr"``
        (NR-level series, matched on friendly header). See ``tags.PROFILES``.

    Returns
    -------
    DataFrame indexed by ``Datetime`` with one float column per mapped series,
    sorted and de-duplicated. Empty (but correctly-typed) if the file has no
    matching columns.
    """
    path = Path(path)
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; expected one of {list(PROFILES)}")
    spec = PROFILES[profile]

    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    col0 = raw[0].astype(str).str.strip()

    # Locate the header ("HRS") row; fall back to the historical index 13.
    hrs_rows = col0.index[col0 == "HRS"]
    header_idx = int(hrs_rows[0]) if len(hrs_rows) else 13
    tag_idx = header_idx + 1

    friendly = raw.iloc[header_idx].astype(str).str.strip().str.replace("\n", " ", regex=False)
    tags = raw.iloc[tag_idx].astype(str).str.strip()

    # First real data row = first HH:MM:SS in col0 below the header.
    is_time = col0.str.match(_TIME_RE)
    below = is_time[is_time & (is_time.index > tag_idx)]
    if below.empty:
        return _empty_frame(spec["columns"])
    data_start = int(below.index[0])

    data = raw.iloc[data_start:].copy()
    data = data[data[0].astype(str).str.strip().str.match(_TIME_RE)]
    if data.empty:
        return _empty_frame(spec["columns"])

    file_date = _date_from_filename(path)
    dt = pd.to_datetime(
        file_date.strftime("%Y-%m-%d") + " " + data[0].astype(str).str.strip(),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )

    out = pd.DataFrame({"Datetime": dt.values})
    report: dict[str, int] = {}
    for col_idx in range(data.shape[1]):
        name = _match_column(spec, tags.iat[col_idx], friendly.iat[col_idx])
        if name is not None:
            out[name] = _to_measurement(
                data.iloc[:, col_idx], stamps=dt.values, file_date=file_date,
                report=report, column=name,
            )

    out = (
        out.dropna(subset=["Datetime"])
        .drop_duplicates(subset="Datetime", keep="first")  # collapse stacked blocks
        .set_index("Datetime")
        .sort_index()
    )
    # Guarantee a stable column order matching the mapping definition.
    # Several raw spellings may map to the SAME clean name (e.g. "Frequency",
    # "Freq" and "FREQ" all -> NR_Frequency), so de-duplicate before reindexing:
    # passing a repeated label to reindex(columns=...) would emit that column
    # once per alias, and the duplicate labels then break the concat in
    # cache._write_year with "Reindexing only valid with uniquely valued Index".
    wanted = list(dict.fromkeys(spec["columns"].values()))
    out = out.reindex(columns=[c for c in wanted if c in out.columns])
    # Recovery is never silent again: the caller logs whatever fired here.
    out.attrs["recovery"] = report
    return out


_EXCEL_EPOCH = pd.Timestamp("1899-12-30")

# A recovered serial that decodes to a date this close to the file's own date,
# AND to that row's own clock time, is a stray timestamp rather than a reading.
_STRAY_DATE_DAYS = 7
_STRAY_CLOCK_SECONDS = 60
# Nothing outside this band is a credible MW reading. NR_Load peaks near
# 90,000 MW; a state draws under 12,000. The bound only ever applies to
# recovered cells, never to values that parsed as numbers in the first place.
_RECOVERY_MIN, _RECOVERY_MAX = -100_000.0, 100_000.0


def _to_measurement(col: pd.Series, stamps=None, file_date=None,
                    report: dict | None = None, column: str = "") -> pd.Series:
    """Numeric MW values, recovering any that the export rendered as dates.

    Some columns in the 5-minute export arrive Excel-formatted: a reading of
    7184.42 is written as ``01-09-1919 10:04:48`` (its Excel serial rendered as
    a date). A plain ``to_numeric(errors="coerce")`` turns every one of those
    into NaN, which silently deleted 83% of the UP series - all of it after
    08:00, so both the morning and evening peaks were missing.

    Anything that parses as a number is taken as-is; anything that instead
    parses as a date is converted back to its Excel serial.

    Two guards on that conversion, because it is otherwise unfalsifiable. An
    Excel serial for any date in 2020-2028 lands in 44,000-47,000, and NR_Load
    and NR_Thermal genuinely run in that range - so a cell that is really a
    stray *timestamp* decodes to a completely plausible MW number and would be
    injected silently:

    1. **Filename cross-check.** ``file_date`` is authoritative (the in-sheet
       dates are exactly what is untrustworthy here). If a decoded value lands
       within a week of the file's own date *and* on that row's own clock time,
       it is the row's timestamp leaking into a data column, not a measurement.
       Dropping it to NaN is the safe side: a NaN is visible and interpolable,
       an injected 45,306 MW is neither. A genuine Excel-formatted reading
       decodes to 1919 (7184.42) and is nowhere near either test.
    2. **Magnitude bound.** Recovered values outside +/-100,000 MW are refused.

    Counts of what fired land in ``report`` so the caller can log them; the
    original bug survived precisely because this path said nothing.
    """
    raw = np.asarray(col.to_numpy(), dtype=object)
    values = np.asarray(pd.to_numeric(raw, errors="coerce"), dtype="float64")
    bad = pd.isna(values)
    if not bad.any():
        return pd.Series(values)

    where = np.flatnonzero(bad)
    # Positional throughout - `raw[bad]` is a plain array, so the Series it
    # builds has a clean RangeIndex and cannot align against `col`'s labels.
    # format="mixed" parses each cell on its own terms. Without it pandas infers
    # ONE format from the first non-null cell and coerces every differently
    # rendered cell to NaT - so a single odd first value silently discards the
    # whole column's worth of recoverable readings.
    decoded = pd.to_datetime(
        pd.Series(raw[bad]), errors="coerce", format="mixed", dayfirst=True
    )
    serial = ((decoded - _EXCEL_EPOCH).dt.total_seconds() / 86400.0).to_numpy()

    keep = (serial >= _RECOVERY_MIN) & (serial <= _RECOVERY_MAX)   # NaN -> False
    n_range = int((~keep & np.isfinite(serial)).sum())

    n_stray = 0
    if stamps is not None and file_date is not None:
        dec = pd.DatetimeIndex(decoded)
        rows = pd.DatetimeIndex(np.asarray(stamps)[bad])
        near_date = (
            np.abs((dec.normalize() - pd.Timestamp(file_date)).to_numpy())
            <= np.timedelta64(_STRAY_DATE_DAYS, "D")
        )
        tod_dec = (dec - dec.normalize()).total_seconds().to_numpy()
        tod_row = (rows - rows.normalize()).total_seconds().to_numpy()
        same_clock = np.abs(tod_dec - tod_row) <= _STRAY_CLOCK_SECONDS
        stray = near_date & same_clock                              # NaT -> False
        n_stray = int((stray & keep).sum())
        keep &= ~stray

    values[where[keep]] = serial[keep]

    if report is not None:
        n_rec = int(keep.sum())
        if n_rec:
            report[f"{column}:recovered"] = report.get(f"{column}:recovered", 0) + n_rec
        if n_stray:
            report[f"{column}:stray_timestamp"] = report.get(f"{column}:stray_timestamp", 0) + n_stray
        if n_range:
            report[f"{column}:out_of_range"] = report.get(f"{column}:out_of_range", 0) + n_range
    return pd.Series(values)


def _match_column(spec, tag: str, header: str):
    """Return the clean series name for a raw column, or None if it isn't one."""
    if spec["match"] == "tag":
        for sub, name in spec["columns"].items():
            if sub in tag:
                return name
        return None
    # match == "header"
    return spec["columns"].get(header.strip())


def _date_from_filename(path: Path) -> pd.Timestamp:
    """Parse the ``DD-MM-YYYY`` filename stem; raise a clear error if it can't."""
    try:
        return pd.to_datetime(path.stem, format="%d-%m-%Y")
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise ValueError(
            f"Filename {path.name!r} is not DD-MM-YYYY.csv; cannot derive its date."
        ) from exc


def _empty_frame(columns) -> pd.DataFrame:
    idx = pd.DatetimeIndex([], name="Datetime")
    return pd.DataFrame({name: pd.Series(dtype="float64") for name in columns.values()}, index=idx)

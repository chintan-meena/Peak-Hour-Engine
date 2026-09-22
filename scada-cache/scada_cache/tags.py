"""Column definitions for the two SCADA Stack export families.

Two very different keys are used depending on the source, because the raw
files label their columns differently:

* The 5-minute "Stack 5minute" export carries the nine state demand series
  ONLY as machine tags on the tag row (row 14). Their human-readable header
  cells (row 13) are blank, so we must match on a substring of the tag, e.g.
  ``PSE_DRW`` inside ``!COMPANIES!PSTCL!REPOT_PS!LINE!PSE_DRW!P.MvMoment``.

* The 1-minute "Stack1Minute" export carries the NR-level series (including
  NR Load = the NR demand the pipeline needs) as friendly headers on row 13,
  e.g. ``NR Load``, ``NR Solar``, ``NR Wind``.

Keeping both maps here means adding or renaming a series is a one-line edit
and never touches the parser or cache logic.
"""

# Match on a unique substring of the machine tag (row 14). Value = clean name.
STATE_DEMAND_TAGS = {
    "PSE_DRW": "Punjab",        # !COMPANIES!PSTCL!REPOT_PS!LINE!PSE_DRW!P.MvMoment
    "HV_DRWL": "Haryana",       # !COMPANIES!HVPNL!HVPNL_HS!LINE!HV_DRWL!P.MvMoment
    "RS_DWL": "Rajasthan",      # !COMPANIES!PGCIL!RPTRS_PG!SENT!RS_DWL!P.MvMoment
    "DTL_DRL": "Delhi",         # !COMPANIES!PGCIL!NRLDC_PG!LINE!DTL_DRL!P.MvMoment
    "UPCL_DRL": "UP",           # !COMPANIES!PGCIL!NRLDC_PG!LINE!UPCL_DRL!P.MvMoment
    "UTR_DRL": "Uttarakhand",   # !COMPANIES!PGCIL!NRLDC_PG!LINE!UTR_DRL!P.MvMoment
    "DRAWL_HP": "Himachal",     # !COMPANIES!HPSEBL!HPDUM_HP!PL!DRAWL_HP!P.MvMoment
    "JKS_DRL": "JK",            # !COMPANIES!PGCIL!NRLDC_PG!LINE!JKS_DRL!P.MvMoment
    "CHND_DRL": "Chandigarh",   # !COMPANIES!PGCIL!NRLDC_PG!LINE!CHND_DRL!P.MvMoment
}

# Match on the exact friendly header text (row 13, whitespace-stripped).
NR_FRIENDLY_HEADERS = {
    # Grid frequency. Not previously mapped, so the parser dropped the column
    # and the cache has no frequency series. The exact friendly header in the
    # Stack1Minute export is not known from here (the share is off-LAN), so
    # several plausible spellings are listed -- only the one that matches will
    # ever fire, and the rest are inert. Re-parse on the LAN to populate it:
    #     rm -rf cache/nr && python update_cache.py --source nr
    # Frequency is an independent, exogenous check on a peak-hour declaration:
    # it feeds no model feature, and a genuinely tight window should sit on the
    # day's lowest-frequency blocks. Note it is all-India frequency (one
    # synchronous grid), so it reflects national rather than NR-local scarcity.
    "Frequency": "NR_Frequency",
    "NR Frequency": "NR_Frequency",
    "Freq": "NR_Frequency",
    "FREQ": "NR_Frequency",
    "Grid Frequency": "NR_Frequency",
    "NR Hydro": "NR_Hydro",
    "NR Thermal": "NR_Thermal",
    "NR Solar": "NR_Solar",
    "NR Wind": "NR_Wind",
    "NR Load": "NR_Load",       # this is the NR demand used for net-load
}

# Which key each parse "profile" uses. The parser dispatches on this.
PROFILES = {
    "state": {"match": "tag", "columns": STATE_DEMAND_TAGS},
    "nr": {"match": "header", "columns": NR_FRIENDLY_HEADERS},
}

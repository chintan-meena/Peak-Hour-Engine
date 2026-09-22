#!/usr/bin/env python3
"""One entry point for a full, recorded pipeline run.

    python run_pipeline.py                  # everything, then record the run
    python run_pipeline.py --skip notebook  # models + scoring only (no `iex` needed)
    python run_pipeline.py --list           # show the steps and exit

Why an orchestrator rather than "just run the notebook"
-------------------------------------------------------
The notebook alone leaves no trace of what produced its numbers. On 2026-08-27
its outputs were written at 20:15 while the file itself had last been saved at
19:16, so the saved cells did not correspond to the results sitting next to
them. Running through here fixes that in three ways:

* the notebook is executed into an executed COPY, so code and outputs are
  captured together and the working file is never half-saved;
* the declaration scorecard comes from benchmark_declarations.py, which scores
  every window at its own length (the notebook's own scorecard reported >100%
  capture for six years because it compared 16-block declarations against a
  12-block optimum);
* every run ends in run_registry.py, which appends the headline scores plus a
  content fingerprint of each input to Run_History.csv and archives the small
  result CSVs under runs/<run_id>/.

Progress reporting
------------------
The notebook step takes ~40 minutes. Where `nbclient` is importable it is
executed cell by cell in-process, so the status line names the cell actually
running and the run leaves a per-cell timing table behind
(Notebook_Cell_Timings.csv) - which is how you find out which cell owns the 40
minutes. Otherwise it falls back to `jupyter nbconvert`, which cannot report
per-cell progress, and only the elapsed clock ticks.

Steps, in order
---------------
  notebook   execute the v1 pipeline -> declaration + RTM/net-load metrics
             (needs the `iex` library; skip it off-LAN and the rest still runs)
  hydro      hydro_peak_model.py  -> 37-month replay backtest
  benchmark  benchmark_declarations.py -> length-matched value capture, both
             the 16-block thermal and 12-block hydro declaration series
  register   run_registry.py -> append to Run_History.csv, archive outputs
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from powertools_common import cache_dir, reports_dir

PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS = reports_dir("Peak_Hour_Engine", "monthly_peak_pipeline_outputs")
MARKET_CACHE_DIR = cache_dir("peak_hours")
DECL_THERMAL = MARKET_CACHE_DIR / "Previous_Declarations.csv"
DECL_HYDRO = MARKET_CACHE_DIR / "Previous_Declarations_Hydro.csv"
NOTEBOOK = PROJECT_DIR / "Peak_Hours_Complete_Pipeline_v1.ipynb"
EXECUTED = PROJECT_DIR / "Peak_Hours_Complete_Pipeline_v1.EXECUTED.ipynb"
CELL_TIMINGS = ARTIFACTS / "Notebook_Cell_Timings.csv"

STEPS = ["notebook", "hydro", "benchmark", "register"]
CELL_TIMEOUT = 7200


def hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


class Console:
    """A scrolling log with one live status line pinned underneath it.

    Subprocess output and the ticking clock share one stream, so the status
    line has to be erased before any log line is written and redrawn after.
    Without that the carriage returns interleave with the child's stdout and
    the result is unreadable. When stdout is not a terminal (piped to a file,
    or CI) there is nothing to erase, so the status is emitted as an ordinary
    line at a slow interval instead.
    """

    def __init__(self, interval: float = 1.0, quiet_interval: float = 30.0):
        self._lock = threading.Lock()
        self._status = ""
        self._t0 = time.monotonic()
        self._step_t0 = self._t0
        self._drawn = False
        self._stop = threading.Event()
        self._tty = sys.stdout.isatty()
        self._interval = interval if self._tty else quiet_interval
        self._thread: threading.Thread | None = None

    # -- internals -----------------------------------------------------------
    def _erase(self) -> None:
        if self._tty and self._drawn:
            sys.stdout.write("\r\033[2K")
            self._drawn = False

    def _draw(self) -> None:
        if not self._status:
            return
        line = (f"  [{hms(time.monotonic() - self._step_t0)} step | "
                f"{hms(time.monotonic() - self._t0)} total]  {self._status}")
        if self._tty:
            sys.stdout.write("\r\033[2K" + line[:shutil.get_terminal_size((100, 24)).columns - 1])
            self._drawn = True
        else:
            sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def _tick(self) -> None:
        while not self._stop.wait(self._interval):
            with self._lock:
                self._draw()

    # -- public --------------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._tick, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        with self._lock:
            self._erase()
            sys.stdout.flush()

    def log(self, msg: str = "") -> None:
        with self._lock:
            self._erase()
            sys.stdout.write(msg + "\n")
            sys.stdout.flush()
            # Redraw only on a terminal. Piped to a file there is nothing to
            # redraw onto, and calling _draw() here would emit a duplicate
            # status line after every single line of child output.
            if self._tty:
                self._draw()

    def status(self, msg: str) -> None:
        with self._lock:
            self._status = msg
            if self._tty:
                self._draw()

    def begin_step(self, n: int, total: int, name: str, detail: str) -> None:
        self._step_t0 = time.monotonic()
        self.log("")
        self.log("=" * 78)
        self.log(f"{n}/{total}  {name} — {detail}")
        self.log(f"         started {datetime.now().strftime('%H:%M:%S')}"
                 f"   (+{hms(time.monotonic() - self._t0)} into the run)")
        self.log("=" * 78)

    def end_step(self, name: str, ok: bool) -> float:
        dt = time.monotonic() - self._step_t0
        self.status("")
        self.log(f"  -> {name}: {'ok' if ok else 'FAILED'} in {hms(dt)} "
                 f"(finished {datetime.now().strftime('%H:%M:%S')})")
        return dt

    @property
    def total_elapsed(self) -> float:
        return time.monotonic() - self._t0


CON = Console()


# ── running child processes ─────────────────────────────────────────────────
def run(cmd: list[str], status_prefix: str = "") -> bool:
    """Run a child process, streaming its output above the live status line."""
    CON.log("  $ " + " ".join(str(c) for c in cmd))
    CON.status(status_prefix or "running…")
    try:
        proc = subprocess.Popen(cmd, cwd=PROJECT_DIR, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
    except FileNotFoundError as exc:
        CON.log(f"  ! cannot start: {exc}")
        return False
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            CON.log("  | " + line)
    proc.wait()
    return proc.returncode == 0


# ── the notebook step ───────────────────────────────────────────────────────
def _cell_label(cell, idx: int) -> str:
    src = "".join(cell.get("source", "")) if not isinstance(cell.get("source"), str) else cell["source"]
    m = re.search(r"SECTION\s*\d+[a-z]?", src)
    if m:
        return m.group(0)
    for line in src.splitlines():
        t = line.strip()
        if t and not t.startswith("#"):
            return t[:46]
    return f"cell {idx}"


def _execute_in_process() -> bool:
    """Per-cell execution via nbclient, so progress is actually visible."""
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError:
        return NotImplemented  # caller falls back

    import csv
    import warnings

    # nbformat raises MissingIDFieldWarning once per cell during read(), before
    # anything can be normalised, so the read itself has to be quiet. normalize()
    # RETURNS the repaired notebook - it is not purely in-place - so the result
    # has to be captured or the ids are silently discarded again on write.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nb = nbformat.read(str(NOTEBOOK), as_version=4)
        try:
            result = nbformat.validator.normalize(nb)
            if isinstance(result, tuple) and len(result) == 2:
                nb = result[1]
            elif result is not None:
                nb = result
        except Exception:
            pass

    client = NotebookClient(
        nb, timeout=CELL_TIMEOUT, kernel_name="python3", allow_errors=False,
        resources={"metadata": {"path": str(PROJECT_DIR)}},
    )
    code = [(i, c) for i, c in enumerate(nb.cells) if c.cell_type == "code"]
    total = len(code)
    timings, ok = [], True
    CON.log(f"  executing {total} code cells in-process (nbclient)")

    try:
        with client.setup_kernel():
            for n, (i, cell) in enumerate(code, 1):
                label = _cell_label(cell, i)
                CON.status(f"cell {n}/{total}  {label}")
                t0 = time.monotonic()
                try:
                    client.execute_cell(cell, i)
                    dt = time.monotonic() - t0
                    timings.append((n, i, label, round(dt, 2), "ok"))
                    if dt >= 20:            # only the slow ones are worth a line
                        CON.log(f"  | cell {n}/{total} {label} — {hms(dt)}")
                except Exception as exc:
                    dt = time.monotonic() - t0
                    timings.append((n, i, label, round(dt, 2), "FAILED"))
                    CON.log(f"  ! cell {n}/{total} ({label}) failed after {hms(dt)}: "
                            f"{type(exc).__name__}: {str(exc)[:300]}")
                    ok = False
                    break
    finally:
        # Always keep what did run - a 35-minute partial result is still worth
        # having when cell 24 is the one that broke.
        nbformat.write(nb, str(EXECUTED))
        CON.log(f"  wrote {EXECUTED.name}")
        if timings:
            CELL_TIMINGS.parent.mkdir(parents=True, exist_ok=True)
            with CELL_TIMINGS.open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["Order", "Cell_Index", "Label", "Seconds", "Status"])
                w.writerows(timings)
            slow = sorted(timings, key=lambda r: -r[3])[:3]
            CON.log("  slowest cells: " + ", ".join(f"{r[2]} {hms(r[3])}" for r in slow))
    return ok


def _log_declaration_summary() -> None:
    """Surface the target-month declaration in this console.

    nbclient captures each cell's print() output into the executed notebook,
    not this process's stdout, so the declaration SECTION 22/23 print — the
    whole point of the run — never appeared in the transcript. The notebook
    also writes it to Monthly_Peak_Hours_<month>.csv; read that back instead
    of re-parsing captured cell output.
    """
    import csv as _csv
    files = sorted(ARTIFACTS.glob("Monthly_Peak_Hours_*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        CON.log("  (no Monthly_Peak_Hours_*.csv found — cannot summarise the declaration)")
        return
    latest = files[-1]
    with latest.open(newline="") as fh:
        rows = list(_csv.DictReader(fh))
    if not rows:
        return
    CON.log("")
    for r in rows:
        CON.log(f"  >> Declared peak hours for {r.get('Month', '?')}: {r.get('Peak_Hours', '?')} "
                 f"({r.get('Pattern', '?')}, method={r.get('Selection_Method', '?')})")
    CON.log(f"     from {latest.name}")


def step_notebook() -> bool:
    result = _execute_in_process()
    if result is NotImplemented:
        CON.log("  nbclient not importable here — falling back to `jupyter nbconvert`.")
        CON.log("  (no per-cell progress on this path; only the clock ticks.)")
        CON.log(f"  for per-cell progress run this under a python that has nbclient, e.g.")
        CON.log(f"      /opt/homebrew/bin/python3 {Path(__file__).name}")
        result = run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
                      "--execute", f"--ExecutePreprocessor.timeout={CELL_TIMEOUT}",
                      "--output", EXECUTED.name, str(NOTEBOOK)],
                     status_prefix="nbconvert — no per-cell progress on this path")
    if result:
        _log_declaration_summary()
    return result


# ── the other steps ─────────────────────────────────────────────────────────
def step_hydro() -> bool:
    return run([sys.executable, "hydro_peak_model.py"], "hydro_peak_model.py — replay backtest")


def step_benchmark() -> bool:
    # Regenerate the derived CSVs from Previous_Declarations.xlsx first, so a
    # declaration added to the workbook since the last run is never scored
    # against a stale cache (the exact staleness bug this orchestrator exists
    # to avoid — see the module docstring).
    CON.status("benchmark: rebuilding Previous_Declarations.csv (thermal)")
    ok = run([sys.executable, "build_previous_declarations.py",
              "--peak", "thermal", "--out", str(DECL_THERMAL)],
             "build_previous_declarations.py — thermal")
    CON.status("benchmark: rebuilding Previous_Declarations_Hydro.csv (hydro)")
    ok &= run([sys.executable, "build_previous_declarations.py",
               "--peak", "hydro", "--out", str(DECL_HYDRO)],
              "build_previous_declarations.py — hydro")

    CON.status("benchmark: thermal (16 blocks)")
    ok &= run([sys.executable, "benchmark_declarations.py",
               "--decl", str(DECL_THERMAL),
               "--out", str(ARTIFACTS / "Declaration_Benchmark.csv")], "benchmark — thermal declarations (16 blocks)")
    CON.status("benchmark: hydro (12 blocks)")
    ok &= run([sys.executable, "benchmark_declarations.py",
               "--decl", str(DECL_HYDRO),
               "--out", str(ARTIFACTS / "Declaration_Benchmark_Hydro.csv")], "benchmark — hydro declarations (12 blocks)")
    return ok


def step_register(label: str, note: str) -> bool:
    return run([sys.executable, "run_registry.py", "--label", label, "--note", note],
               "run_registry.py — recording run")


DETAIL = {
    "notebook": "full pipeline (~40 min, needs `iex`)",
    "hydro": "37-month replay backtest",
    "benchmark": "length-matched value capture, thermal + hydro",
    "register": "append to Run_History.csv, archive outputs",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip", nargs="*", default=[], choices=STEPS,
                   help="steps to skip (e.g. --skip notebook when off-LAN)")
    p.add_argument("--only", nargs="*", default=None, choices=STEPS,
                   help="run only these steps")
    p.add_argument("--label", default=None, help="label for this run in the history")
    p.add_argument("--note", default="", help="free-text note stored with the run")
    p.add_argument("--list", action="store_true", help="print the steps and exit")
    a = p.parse_args()

    if a.list:
        print(__doc__)
        return

    wanted = [s for s in STEPS if (a.only is None or s in a.only) and s not in a.skip]
    label = a.label or ("+".join(w for w in wanted if w != "register") or "no-op")

    CON.start()
    started = datetime.now()
    CON.log(f"Run started {started.strftime('%Y-%m-%d %H:%M:%S')}")
    CON.log(f"Steps: {' -> '.join(wanted)}")

    failed, took = [], {}
    try:
        for n, s in enumerate(wanted, 1):
            CON.begin_step(n, len(wanted), s, DETAIL[s])
            ok = {"notebook": step_notebook,
                  "hydro": step_hydro,
                  "benchmark": step_benchmark,
                  "register": lambda: step_register(label, a.note)}[s]()
            took[s] = CON.end_step(s, ok)
            if not ok:
                failed.append(s)
                if s in ("notebook", "hydro", "benchmark"):
                    CON.log(f"  ! {s} failed; later steps may score stale outputs.")
    except KeyboardInterrupt:
        CON.log("\n  interrupted by user")
        failed.append("interrupted")
    finally:
        CON.stop()

    print("\n" + "=" * 78)
    print(f"Finished {datetime.now().strftime('%H:%M:%S')} — total {hms(CON.total_elapsed)}")
    for s in wanted:
        if s in took:
            print(f"    {s:12s} {hms(took[s]):>9s}"
                  + ("   FAILED" if s in failed else ""))
    if failed:
        print("FAILED steps: " + ", ".join(failed))
        sys.exit(1)
    print("\nCompare against the previous run with:")
    print("    python run_registry.py --compare")


if __name__ == "__main__":
    main()

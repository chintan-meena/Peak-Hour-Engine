#!/usr/bin/env python3
"""Tkinter front end for Peak_Hours_Complete_Pipeline_v1.ipynb.

Pick the history window and the month to declare peak hours for, hit Run, and
watch the notebook execute cell by cell. The declared window is shown when it
finishes.

How it drives the notebook
--------------------------
The notebook keeps its three knobs as plain literals in one config cell::

    START_DATE = "2022-07-03"
    PEAK_MONTH = "2026-09"
    END_DATE   = "2026-08-22"

Those same lines also feed START_TS / END_TS in that cell, so the values are
rewritten *in place* in an in-memory copy of the notebook before execution —
injecting an override cell afterwards would leave the derived timestamps
stale. Your .ipynb on disk is untouched unless you tick "write parameters
back".

Run it with:

    python peak_hours_gui.py

Dependencies: tkcalendar, nbformat, nbclient (plus whatever the notebook
itself needs — use "Check environment" to confirm those before a long run).
"""
from __future__ import annotations

import calendar
import csv
import importlib.util
import json
import os
import queue
import re
import subprocess
import sys
import threading
import traceback
from datetime import date, timedelta
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

try:
    from tkcalendar import DateEntry
except ImportError:
    DateEntry = None

PROJECT_DIR = Path(__file__).resolve().parent
NOTEBOOK = PROJECT_DIR / "Peak_Hours_Complete_Pipeline_v1.ipynb"
ARTIFACT_DIR = PROJECT_DIR / "monthly_peak_pipeline_outputs"

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

# START_DATE = "2022-07-03"   ->  indent, name, ' = ', value, trailing comment
PARAM_RE = re.compile(
    r'^(?P<indent>\s*)(?P<name>START_DATE|END_DATE|PEAK_MONTH)'
    r'(?P<eq>\s*=\s*)(?P<q>["\'])(?P<val>[^"\']*)(?P=q)(?P<rest>.*)$'
)


# ════════════════════════════════════════════════════════════════════════════
# Notebook parameter handling
# ════════════════════════════════════════════════════════════════════════════
def read_notebook_params(path: Path) -> dict[str, str]:
    """Current START_DATE / END_DATE / PEAK_MONTH literals, for prefilling."""
    found: dict[str, str] = {}
    try:
        nb = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return found
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for line in "".join(cell.get("source", [])).splitlines():
            m = PARAM_RE.match(line)
            if m and m.group("name") not in found:
                found[m.group("name")] = m.group("val")
    return found


def patch_notebook_params(nb, params: dict[str, str]) -> int:
    """Rewrite the parameter literals in place. Returns how many lines changed."""
    changed = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        lines = cell.source.splitlines(keepends=True)
        for i, line in enumerate(lines):
            stripped = line.rstrip("\n")
            m = PARAM_RE.match(stripped)
            if not m or m.group("name") not in params:
                continue
            newline = "\n" if line.endswith("\n") else ""
            lines[i] = (f'{m.group("indent")}{m.group("name")}{m.group("eq")}'
                        f'"{params[m.group("name")]}"{m.group("rest")}{newline}')
            changed += 1
        cell.source = "".join(lines)
    return changed


def last_day_before_month(peak_month: str) -> date:
    """Last calendar day of the month preceding 'YYYY-MM'."""
    year, month = (int(x) for x in peak_month.split("-"))
    return date(year, month, 1) - timedelta(days=1)


# ════════════════════════════════════════════════════════════════════════════
# Preflight
# ════════════════════════════════════════════════════════════════════════════
def _module_ok(name: str) -> bool:
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _scada_span() -> tuple[date | None, date | None, str]:
    """(first_day, last_day, human_text) for the cached 'nr' source."""
    scada = PROJECT_DIR / "scada-cache"
    if not scada.exists():
        return None, None, "scada-cache not present"
    if str(scada) not in sys.path:
        sys.path.insert(0, str(scada))
    try:
        import scada_cache as sc
        status = sc.cache_status()
        row = status[status["source"] == "nr"]
        if row.empty or not row.iloc[0]["rows"]:
            return None, None, "cache empty"
        lo, hi = row.iloc[0]["start"], row.iloc[0]["end"]
        return lo.date(), hi.date(), f"{lo.date()} -> {hi.date()}"
    except Exception as e:
        return None, None, f"unreadable ({type(e).__name__})"


def check_environment(start: str | None = None,
                      end: str | None = None) -> list[tuple[str, bool, str]]:
    """(label, ok, detail) for each prerequisite the notebook needs.

    When start/end are supplied, the demand-data check verifies the cache
    actually spans that window instead of merely existing.
    """
    out: list[tuple[str, bool, str]] = []

    out.append(("Notebook", NOTEBOOK.exists(), str(NOTEBOOK)))

    csv_path = PROJECT_DIR / "City_Weather_Feature_Importance.csv"
    out.append(("City weights CSV", csv_path.exists(),
                csv_path.name if csv_path.exists() else "missing — v1 fails at startup"))

    for mod, note in [("iex", "provides get_trade_data (market data)"),
                      ("lightgbm", ""), ("holidays", ""), ("sklearn", ""),
                      ("matplotlib", ""), ("pandas", ""), ("requests", "")]:
        ok = _module_ok(mod)
        out.append((f"module: {mod}", ok, note if ok else "NOT importable"))

    demand = PROJECT_DIR / "NR_DEMAND_TEMP"
    n_csv = len(list(demand.glob("*.csv"))) if demand.exists() else 0
    lo, hi, span_text = _scada_span()

    detail = [f"scada-cache nr: {span_text}"]
    if n_csv:
        detail.append(f"NR_DEMAND_TEMP: {n_csv} csv")

    covers = lo is not None
    if covers and start and end:
        try:
            want_lo = date(*(int(x) for x in start.split("-")))
            want_hi = date(*(int(x) for x in end.split("-")))
            covers = lo <= want_lo and hi >= want_hi
            if not covers:
                detail.append(f"does NOT cover {want_lo} -> {want_hi}")
        except Exception:
            pass
    if not covers and not n_csv:
        detail.append("run: cd scada-cache && python update_cache.py --source nr")

    out.append(("Demand data", covers or n_csv > 0, "; ".join(detail)))

    for mod in ("nbformat", "nbclient"):
        out.append((f"module: {mod}", _module_ok(mod), ""))

    return out


# ════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════
class PipelineRunner(threading.Thread):
    """Executes the notebook in a worker thread, reporting through a Queue."""

    def __init__(self, params: dict[str, str], out_q: "queue.Queue",
                 stop_event: threading.Event, save_executed: bool,
                 write_back: bool):
        super().__init__(daemon=True)
        self.params = params
        self.q = out_q
        self.stop_event = stop_event
        self.save_executed = save_executed
        self.write_back = write_back
        self.client = None

    def emit(self, kind: str, payload=None) -> None:
        self.q.put((kind, payload))

    def run(self) -> None:
        try:
            self._run()
        except Exception:
            self.emit("log", traceback.format_exc())
            self.emit("failed", "Unexpected error — see the log.")

    def _run(self) -> None:
        import nbformat
        from nbclient import NotebookClient
        from nbclient.exceptions import CellExecutionError

        self.emit("status", "Loading notebook …")
        nb = nbformat.read(str(NOTEBOOK), as_version=4)

        n = patch_notebook_params(nb, self.params)
        self.emit("log", f"Patched {n} parameter line(s): "
                         + ", ".join(f"{k}={v}" for k, v in self.params.items()))
        if n < len(self.params):
            self.emit("log", "! Not every parameter was found in the notebook — "
                             "check the config cell still uses plain literals.")

        if self.write_back:
            nbformat.write(nb, str(NOTEBOOK))
            self.emit("log", f"Wrote parameters back into {NOTEBOOK.name}")

        code_idx = [i for i, c in enumerate(nb.cells) if c.cell_type == "code"]
        total = len(code_idx)
        self.emit("log", f"{total} code cells to execute.\n")

        self.client = NotebookClient(
            nb, timeout=None, kernel_name="python3", allow_errors=False,
            resources={"metadata": {"path": str(PROJECT_DIR)}},
        )

        try:
            with self.client.setup_kernel():
                for pos, idx in enumerate(code_idx, start=1):
                    if self.stop_event.is_set():
                        self.emit("log", "\nStopped before cell "
                                         f"{pos}/{total}.")
                        self.emit("failed", "Stopped by user.")
                        return
                    self.emit("progress", (pos, total))
                    self.emit("status", f"Running cell {pos} of {total} …")
                    try:
                        self.client.execute_cell(nb.cells[idx], idx)
                    except CellExecutionError as e:
                        self._dump_outputs(nb.cells[idx], pos)
                        self.emit("log", f"\n--- cell {pos} failed ---\n{e}")
                        self.emit("failed", f"Cell {pos} of {total} failed.")
                        return
                    self._dump_outputs(nb.cells[idx], pos)
        finally:
            if self.save_executed:
                try:
                    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
                    dest = (ARTIFACT_DIR /
                            f"Executed_v1_{self.params['PEAK_MONTH']}.ipynb")
                    nbformat.write(nb, str(dest))
                    self.emit("log", f"\nExecuted notebook saved -> {dest.name}")
                except Exception as e:
                    self.emit("log", f"\nCould not save executed notebook: {e}")

        self.emit("progress", (total, total))
        self.emit("done", self.params["PEAK_MONTH"])

    def _dump_outputs(self, cell, pos: int) -> None:
        for out in cell.get("outputs", []):
            kind = out.get("output_type")
            if kind == "stream":
                text = out.get("text", "")
                if text.strip():
                    self.emit("log", text.rstrip())
            elif kind == "error":
                self.emit("log", f"{out.get('ename')}: {out.get('evalue')}")
            elif kind in ("execute_result", "display_data"):
                text = out.get("data", {}).get("text/plain", "")
                if isinstance(text, list):
                    text = "".join(text)
                if text.strip():
                    self.emit("log", text.rstrip())

    def interrupt(self) -> None:
        try:
            if self.client is not None and getattr(self.client, "km", None):
                self.client.km.interrupt_kernel()
        except Exception:
            pass


def read_result(peak_month: str) -> dict | None:
    """Peak_Hours row from Monthly_Peak_Hours_<PEAK_MONTH>.csv, if written."""
    path = ARTIFACT_DIR / f"Monthly_Peak_Hours_{peak_month}.csv"
    if not path.exists():
        return None
    try:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        return rows[0] if rows else None
    except Exception:
        return None


def open_folder(path: Path) -> None:
    try:
        if sys.platform == "win32":
            os.startfile(str(path))            # noqa: S606
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as e:
        messagebox.showerror("Could not open folder", str(e))


# ════════════════════════════════════════════════════════════════════════════
# GUI
# ════════════════════════════════════════════════════════════════════════════
class PeakHoursGUI(ttk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master, padding=12)
        self.master = master
        self.grid(row=0, column=0, sticky="nsew")
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        self.q: queue.Queue = queue.Queue()
        self.runner: PipelineRunner | None = None
        self.stop_event = threading.Event()

        current = read_notebook_params(NOTEBOOK)
        self._build_params(current)
        self._build_actions()
        self._build_log()
        self._build_result()

        self.after(120, self._drain_queue)

    # ---------------------------------------------------------------- params
    def _build_params(self, current: dict[str, str]) -> None:
        box = ttk.LabelFrame(self, text="Parameters", padding=10)
        box.grid(row=0, column=0, sticky="ew")
        box.columnconfigure(1, weight=1)

        # Peak month -------------------------------------------------------
        ttk.Label(box, text="Peak month to declare:").grid(
            row=0, column=0, sticky="w", pady=4)
        pm = ttk.Frame(box)
        pm.grid(row=0, column=1, sticky="w", pady=4)

        pm_cur = current.get("PEAK_MONTH", "")
        try:
            cy, cm = (int(x) for x in pm_cur.split("-"))
        except Exception:
            today = date.today()
            cy, cm = today.year, today.month

        self.month_var = tk.StringVar(value=MONTHS[cm - 1])
        self.year_var = tk.StringVar(value=str(cy))
        ttk.Combobox(pm, textvariable=self.month_var, values=MONTHS,
                     state="readonly", width=12).pack(side="left")
        ttk.Spinbox(pm, textvariable=self.year_var, from_=2020, to=2040,
                    width=6, command=self._on_month_change).pack(side="left", padx=(6, 8))
        self.month_var.trace_add("write", lambda *_: self._on_month_change())
        self.year_var.trace_add("write", lambda *_: self._on_month_change())

        self.pm_label = ttk.Label(pm, text="", foreground="#555")
        self.pm_label.pack(side="left")

        # History start ----------------------------------------------------
        ttk.Label(box, text="History start date:").grid(
            row=1, column=0, sticky="w", pady=4)
        self.start_entry = self._date_widget(
            box, current.get("START_DATE", "2022-07-03"))
        self.start_entry.grid(row=1, column=1, sticky="w", pady=4)

        # Data end ---------------------------------------------------------
        ttk.Label(box, text="Data end date:").grid(
            row=2, column=0, sticky="w", pady=4)
        end_row = ttk.Frame(box)
        end_row.grid(row=2, column=1, sticky="w", pady=4)
        self.end_entry = self._date_widget(
            end_row, current.get("END_DATE", "2026-08-22"))
        self.end_entry.pack(side="left")

        self.auto_end = tk.BooleanVar(value=False)
        ttk.Checkbutton(end_row, text="last day before peak month",
                        variable=self.auto_end,
                        command=self._on_month_change).pack(side="left", padx=8)

        hint = ("The notebook trains on history from the start date up to the "
                "data end date, then declares peak hours for the peak month.")
        ttk.Label(box, text=hint, foreground="#555", wraplength=560,
                  justify="left").grid(row=3, column=0, columnspan=2,
                                       sticky="w", pady=(8, 0))
        self._on_month_change()

    def _date_widget(self, parent, initial: str):
        """tkcalendar picker when available, else a validated text entry."""
        try:
            y, m, d = (int(x) for x in initial.split("-"))
            init = date(y, m, d)
        except Exception:
            init = date.today()
        if DateEntry is not None:
            return DateEntry(parent, date_pattern="yyyy-mm-dd", width=12,
                             year=init.year, month=init.month, day=init.day)
        var = tk.StringVar(value=init.isoformat())
        entry = ttk.Entry(parent, textvariable=var, width=14)
        entry._var = var                      # noqa: SLF001 - read back below
        return entry

    def _read_date(self, widget) -> str:
        if DateEntry is not None and isinstance(widget, DateEntry):
            return widget.get_date().isoformat()
        return widget._var.get().strip()      # noqa: SLF001

    def peak_month(self) -> str:
        month = MONTHS.index(self.month_var.get()) + 1
        return f"{int(self.year_var.get()):04d}-{month:02d}"

    def _on_month_change(self) -> None:
        try:
            pm = self.peak_month()
        except Exception:
            return
        self.pm_label.config(text=f"= {pm}")
        if self.auto_end.get():
            end = last_day_before_month(pm)
            if DateEntry is not None and isinstance(self.end_entry, DateEntry):
                self.end_entry.set_date(end)
            else:
                self.end_entry._var.set(end.isoformat())   # noqa: SLF001
            self.end_entry.configure(state="disabled")
        else:
            self.end_entry.configure(state="normal")

    # --------------------------------------------------------------- actions
    def _build_actions(self) -> None:
        bar = ttk.Frame(self)
        bar.grid(row=1, column=0, sticky="ew", pady=(10, 6))

        self.check_btn = ttk.Button(bar, text="Check environment",
                                    command=self.on_check)
        self.check_btn.pack(side="left")
        self.run_btn = ttk.Button(bar, text="Run pipeline", command=self.on_run)
        self.run_btn.pack(side="left", padx=6)
        self.stop_btn = ttk.Button(bar, text="Stop", command=self.on_stop,
                                   state="disabled")
        self.stop_btn.pack(side="left")
        ttk.Button(bar, text="Open outputs",
                   command=lambda: open_folder(ARTIFACT_DIR)).pack(side="left", padx=6)

        self.save_exec = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="save executed notebook",
                        variable=self.save_exec).pack(side="left", padx=(14, 0))
        self.write_back = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="write params into .ipynb",
                        variable=self.write_back).pack(side="left", padx=6)

        prog = ttk.Frame(self)
        prog.grid(row=2, column=0, sticky="ew")
        prog.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(prog, mode="determinate")
        self.progress.grid(row=0, column=0, sticky="ew")
        self.status = ttk.Label(prog, text="Idle", foreground="#555")
        self.status.grid(row=1, column=0, sticky="w", pady=(3, 0))

    def _build_log(self) -> None:
        box = ttk.LabelFrame(self, text="Log", padding=6)
        box.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        box.columnconfigure(0, weight=1)
        box.rowconfigure(0, weight=1)
        self.log = scrolledtext.ScrolledText(box, height=18, wrap="word",
                                             state="disabled",
                                             font=("Consolas", 9))
        self.log.grid(row=0, column=0, sticky="nsew")

    def _build_result(self) -> None:
        box = ttk.LabelFrame(self, text="Declared peak hours", padding=10)
        box.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        self.result_var = tk.StringVar(value="—")
        ttk.Label(box, textvariable=self.result_var,
                  font=("Segoe UI", 12, "bold")).pack(anchor="w")

    # ----------------------------------------------------------------- utils
    def append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    # --------------------------------------------------------------- handlers
    def on_check(self) -> None:
        self.clear_log()
        self.append_log("Environment check\n" + "-" * 60)
        try:
            start = self._read_date(self.start_entry)
            end = self._read_date(self.end_entry)
            self.append_log(f"checking against {start} -> {end}\n")
        except Exception:
            start = end = None
        rows = check_environment(start, end)
        for label, ok, detail in rows:
            mark = "OK  " if ok else "FAIL"
            self.append_log(f"[{mark}] {label:22s} {detail}")
        missing = [r[0] for r in rows if not r[1]]
        if missing:
            self.append_log("\nMissing: " + ", ".join(missing))
            self.append_log("The notebook will fail until these are resolved.")
            self.status.config(text=f"{len(missing)} prerequisite(s) missing")
        else:
            self.append_log("\nAll prerequisites present.")
            self.status.config(text="Environment OK")

    def _validate(self) -> dict[str, str] | None:
        try:
            start = self._read_date(self.start_entry)
            end = self._read_date(self.end_entry)
            pm = self.peak_month()
            sy, sm, sd = (int(x) for x in start.split("-"))
            ey, em, ed = (int(x) for x in end.split("-"))
            s, e = date(sy, sm, sd), date(ey, em, ed)
        except Exception:
            messagebox.showerror("Invalid input",
                                 "Dates must be valid and formatted YYYY-MM-DD.")
            return None
        if s >= e:
            messagebox.showerror("Invalid range",
                                 "History start must be before the data end date.")
            return None
        peak_start = date(int(pm[:4]), int(pm[5:]), 1)
        if e >= peak_start:
            if not messagebox.askyesno(
                "Data end overlaps peak month",
                f"The data end date ({e}) is inside the peak month ({pm}).\n\n"
                "The pipeline normally trains on history strictly before the "
                "month it declares. Continue anyway?"):
                return None
        return {"START_DATE": s.isoformat(), "END_DATE": e.isoformat(),
                "PEAK_MONTH": pm}

    def on_run(self) -> None:
        if self.runner and self.runner.is_alive():
            return
        if not NOTEBOOK.exists():
            messagebox.showerror("Notebook missing", str(NOTEBOOK))
            return
        params = self._validate()
        if params is None:
            return

        self.clear_log()
        self.result_var.set("—")
        self.progress.config(value=0, maximum=100)
        self.stop_event.clear()
        self.run_btn.config(state="disabled")
        self.check_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        self.append_log(f"Peak month : {params['PEAK_MONTH']}")
        self.append_log(f"History    : {params['START_DATE']} -> {params['END_DATE']}")
        self.append_log("-" * 60)

        self.runner = PipelineRunner(params, self.q, self.stop_event,
                                     self.save_exec.get(), self.write_back.get())
        self.runner.start()

    def on_stop(self) -> None:
        if not (self.runner and self.runner.is_alive()):
            return
        self.stop_event.set()
        self.runner.interrupt()
        self.status.config(text="Stopping after the current cell …")
        self.stop_btn.config(state="disabled")

    def _finish(self, message: str) -> None:
        self.run_btn.config(state="normal")
        self.check_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status.config(text=message)

    def _drain_queue(self) -> None:
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self.append_log(str(payload))
                elif kind == "status":
                    self.status.config(text=str(payload))
                elif kind == "progress":
                    pos, total = payload
                    self.progress.config(maximum=total, value=pos)
                elif kind == "done":
                    self._on_done(payload)
                elif kind == "failed":
                    self.append_log("\n" + str(payload))
                    self._finish(str(payload))
        except queue.Empty:
            pass
        self.after(120, self._drain_queue)

    def _on_done(self, peak_month: str) -> None:
        row = read_result(peak_month)
        if row and row.get("Peak_Hours"):
            self.result_var.set(f"{peak_month}:  {row['Peak_Hours']}")
            self.append_log(f"\nDeclared peak hours for {peak_month}: "
                            f"{row['Peak_Hours']}")
            extra = {k: v for k, v in row.items() if k != "Peak_Hours"}
            if extra:
                self.append_log("  " + "  ".join(f"{k}={v}" for k, v in extra.items()))
        else:
            self.result_var.set("finished — no result CSV found")
            self.append_log(f"\nFinished, but {ARTIFACT_DIR.name}/"
                            f"Monthly_Peak_Hours_{peak_month}.csv was not found.")
        self._finish("Done")


def main() -> None:
    root = tk.Tk()
    root.title("NR Peak Hour Declaration — Pipeline v1")
    root.geometry("760x720")
    root.minsize(680, 600)
    try:
        ttk.Style().theme_use("vista" if sys.platform == "win32" else "clam")
    except tk.TclError:
        pass
    PeakHoursGUI(root)
    if DateEntry is None:
        messagebox.showwarning(
            "tkcalendar not installed",
            "Falling back to plain text date entry (YYYY-MM-DD).\n\n"
            "For a calendar picker:  pip install tkcalendar")
    root.mainloop()


if __name__ == "__main__":
    main()

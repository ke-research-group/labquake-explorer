"""Run-level view of inter-event (per-cycle) metrics."""
import tkinter as tk
from tkinter import ttk

import numpy as np

from labquake_explorer.analysis.interevent import (
    creep_per_cycle, event_times_from_run, interevent_metrics, RESULT_VERSION,
)
from labquake_explorer.ui.actions import register_view
from labquake_explorer.ui.context import RUN
from labquake_explorer.ui.views.base import RunView, event_list
from labquake_explorer.data.channels import aligned_fields, get_field


def aligned_arrays(run: dict) -> list:
    """Paths of the run's numeric series aligned with run['time']: top-level
    arrays and rows of channel arrays (``'slip/slip_3'``)."""
    time = run.get("time")
    if time is None:
        return []
    n = len(time)
    out = []
    for path in aligned_fields(run, n, recurse=False):
        if path == "time":
            continue
        value = get_field(run, path)
        if value is not None and np.asarray(value).dtype.kind in "iuf":
            out.append(path)
    return sorted(out)


@register_view("Inter-event Metrics", kinds=[RUN], order=20)
class InterEventView(RunView):
    """Recurrence, load-point advance and fault slip per stick-slip cycle.

    Each event is sampled ``delay`` seconds after its ``event_time`` by a
    mean over ``width`` seconds; the per-cycle value is the difference to the
    previous event.  Coseismic slip comes from each event's saved
    ``event_analysis['displacement']`` when that record was analysed on the
    SAME run signal as the fault-slip combobox (``x_field``, schema v2):
    creep = slip per cycle - coseismic slip.  Records analysed on another X
    (e.g. ``LP_displacement`` for machine stiffness, or ``time``) and legacy
    v1 records (no ``x_field``, unsigned displacement) leave the coseismic
    slip NaN and are counted in the status line.
    Saved under ``runs/[r]['interevent']``.
    """

    window_title = "Inter-event Metrics"
    result_key = "interevent"

    COLUMNS = (
        ("event", "Event", 50), ("event_time", "Time (s)", 90), ("recurrence", "Recurrence (s)", 100),
        ("lp_per_cycle", "LP advance", 90), ("slip_per_cycle", "Fault slip", 90),
        ("coseismic_slip", "Coseismic slip", 100), ("creep", "Creep", 80),
    )

    def __init__(self, app, run_idx):
        self.result = None
        super().__init__(app, run_idx)

    # ------------------------------------------------------------------ ui
    def build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        controls = ttk.LabelFrame(self, text="Sampling")
        controls.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ttk.Label(controls, text="Load-point signal:").grid(row=0, column=0, padx=5, pady=3, sticky="e")
        self.lp_combo = ttk.Combobox(controls, state="readonly", width=18)
        self.lp_combo.grid(row=0, column=1, padx=5, pady=3, sticky="w")
        ttk.Label(controls, text="Fault slip signal:").grid(row=0, column=2, padx=5, pady=3, sticky="e")
        self.slip_combo = ttk.Combobox(controls, state="readonly", width=18)
        self.slip_combo.grid(row=0, column=3, padx=5, pady=3, sticky="w")
        ttk.Label(controls, text="Delay (s):").grid(row=1, column=0, padx=5, pady=3, sticky="e")
        self.delay_var = tk.StringVar(value="0.05")
        ttk.Entry(controls, textvariable=self.delay_var, width=10).grid(row=1, column=1, padx=5, pady=3, sticky="w")
        ttk.Label(controls, text="Average width (s):").grid(row=1, column=2, padx=5, pady=3, sticky="e")
        self.width_var = tk.StringVar(value="0.05")
        ttk.Entry(controls, textvariable=self.width_var, width=10).grid(row=1, column=3, padx=5, pady=3, sticky="w")
        ttk.Button(controls, text="Compute", command=self.compute).grid(row=0, column=4, rowspan=2, padx=8, pady=3)
        ttk.Button(controls, text="Save", command=self.save).grid(row=0, column=5, rowspan=2, padx=8, pady=3)
        self.status_var = tk.StringVar(value="")
        ttk.Label(controls, textvariable=self.status_var).grid(row=2, column=0, columnspan=6, padx=5, pady=3, sticky="w")

        table_frame = ttk.Frame(self)
        table_frame.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.table = ttk.Treeview(table_frame, columns=[c[0] for c in self.COLUMNS], show="headings", height=6)
        for key, heading, width in self.COLUMNS:
            self.table.heading(key, text=heading)
            self.table.column(key, width=width, anchor="e")
        self.table.pack(side="left", fill="x", expand=True)
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")

        self.make_figure(figsize=(9, 5), row=2, column=0, padx=5, pady=5, sticky="nsew")
        self.ax_recurrence, self.ax_slip = self.figure.subplots(2, 1, sharex=True)

    def on_run_loaded(self):
        candidates = aligned_arrays(self.run)
        self.lp_combo.config(values=candidates)
        self.slip_combo.config(values=candidates)
        saved = self.load_results()
        lp = saved.get("lp_field") if saved else None
        slip = saved.get("slip_field") if saved else None
        self.lp_combo.set(lp if lp in candidates else ("LP_displacement" if "LP_displacement" in candidates else (candidates[0] if candidates else "")))
        self.slip_combo.set(slip if slip in candidates else ("displacement" if "displacement" in candidates else (candidates[0] if candidates else "")))
        if saved:
            self.delay_var.set(f"{saved.get('delay_s', 0.05):g}")
            self.width_var.set(f"{saved.get('width_s', 0.05):g}")
        self.compute()

    # ------------------------------------------------------------ analysis
    def parameters(self):
        try:
            delay = float(self.delay_var.get())
            width = float(self.width_var.get())
        except ValueError:
            raise ValueError("delay and width must be numbers")
        if not np.isfinite(delay) or not np.isfinite(width) or width < 0:
            raise ValueError("delay must be finite and width >= 0")
        return delay, width

    def coseismic_slips(self, n: int, slip_field: str) -> tuple[np.ndarray, dict]:
        """Per-event coseismic slip from saved ``event_analysis`` records.

        Only a record analysed on ``slip_field`` (``x_field == slip_field``,
        schema v2) is used: ``displacement`` is X(rupture end) - X(rupture
        start) of WHATEVER X the analyser was run on, so a record analysed on
        ``LP_displacement`` or ``time`` is not a fault slip.  Legacy v1
        records (no ``x_field``, unsigned displacement) are skipped too.
        Returns ``(slips, skipped)`` with ``skipped`` mapping a reason text
        to the number of events it applies to.
        """
        out = np.full(n, np.nan)
        skipped: dict[str, int] = {}
        for j, event in enumerate(event_list(self.run)[:n]):
            analysis = event.get("event_analysis") if isinstance(event, dict) else None
            if not isinstance(analysis, dict) or "displacement" not in analysis:
                continue
            x_field = analysis.get("x_field")
            if x_field is None:
                reason = "legacy v1 record without x_field (unsigned displacement)"
            elif str(x_field) != slip_field:
                reason = f"analysed on {x_field}"
            else:
                reason = None
            if reason is not None:
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            try:
                out[j] = float(analysis["displacement"])
            except (TypeError, ValueError):
                pass
        return out, skipped

    def compute(self):
        event_times = event_times_from_run(self.run)
        if event_times is None or event_times.size == 0:
            self.status_var.set("No events with event_time in this run")
            self.result = None
            self.refresh_table()
            return
        lp_field, slip_field = self.lp_combo.get(), self.slip_combo.get()
        if not lp_field or not slip_field:
            self.status_var.set("Select the load-point and fault slip signals")
            return
        try:
            delay, width = self.parameters()
            lp, slip = get_field(self.run, lp_field), get_field(self.run, slip_field)
            if lp is None or slip is None:
                raise KeyError(f"signal not found: {lp_field if lp is None else slip_field}")
            metrics = interevent_metrics(self.run["time"], event_times, {"lp": lp, "slip": slip},
                                         delay=delay, width=width)
        except (ValueError, KeyError) as e:
            self.status_var.set(str(e))
            return
        n = event_times.size
        coseismic, skipped = self.coseismic_slips(n, slip_field)
        creep = creep_per_cycle(metrics["slip_per_cycle"], coseismic)
        self.result = {
            "version": RESULT_VERSION,
            "delay_s": delay,
            "width_s": width,
            "lp_field": lp_field,
            "slip_field": slip_field,
            "event_times": metrics["event_times"],
            "recurrence": metrics["recurrence"],
            "lp_after": metrics["lp_after"],
            "slip_after": metrics["slip_after"],
            "lp_per_cycle": metrics["lp_per_cycle"],
            "slip_per_cycle": metrics["slip_per_cycle"],
            "coseismic_slip": coseismic.tolist(),
            "coseismic_field": slip_field,
            "coseismic_skipped": dict(skipped),
            "creep": creep.tolist(),
            "note": "per-cycle values are differences between samples taken delay s after consecutive events; "
                    "creep = slip_per_cycle - coseismic_slip (from event_analysis records analysed on "
                    "x_field == slip_field; other records are skipped)",
        }
        n_valid = int(np.sum(np.isfinite(metrics["lp_per_cycle"])))
        text = (f"{n} events, {n_valid} cycles; coseismic slip from event_analysis ({slip_field}) for "
                f"{int(np.sum(np.isfinite(coseismic)))} events")
        for reason, count in skipped.items():
            text += f"; coseismic slip skipped for {count} event(s): {reason}"
        self.status_var.set(text)
        self.refresh_table()
        self.plot()

    def refresh_table(self):
        self.table.delete(*self.table.get_children())
        if not self.result:
            return
        r = self.result
        for j, t in enumerate(r["event_times"]):
            row = [j, self.fmt(t), self.fmt(r["recurrence"][j]), self.fmt(r["lp_per_cycle"][j]),
                   self.fmt(r["slip_per_cycle"][j]), self.fmt(r["coseismic_slip"][j]), self.fmt(r["creep"][j])]
            self.table.insert("", "end", values=row)

    @staticmethod
    def fmt(value) -> str:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "n/a"
        return "n/a" if not np.isfinite(value) else f"{value:.5g}"

    def plot(self):
        for ax in (self.ax_recurrence, self.ax_slip):
            ax.clear()
            ax.grid(True, linestyle="--", alpha=0.3)
        if not self.result:
            self.canvas.draw_idle()
            return
        r = self.result
        idx = np.arange(len(r["event_times"]))
        self.ax_recurrence.plot(idx, r["recurrence"], "o-", color="#33CCC4")
        self.ax_recurrence.set_ylabel("recurrence (s)")
        self.ax_slip.plot(idx, r["lp_per_cycle"], "s-", color="#CC3366", label=f"{r['lp_field']} per cycle")
        self.ax_slip.plot(idx, r["slip_per_cycle"], "o-", color="#33CC66", label=f"{r['slip_field']} per cycle")
        if np.any(np.isfinite(r["creep"])):
            self.ax_slip.plot(idx, r["creep"], "^-", color="#9966CC", label="creep")
        self.ax_slip.set_ylabel("per cycle")
        self.ax_slip.set_xlabel("event index")
        self.ax_slip.legend(loc="upper left", fontsize="small")
        self.ax_recurrence.set_title(f"{self.experiment_name()} run{self.run_idx:02d}")
        self.canvas.draw_idle()

    def save(self):
        if self.result is None:
            self.compute()
        if self.result is not None:
            self.save_results(dict(self.result))
            self.status_var.set(self.status_var.get() + " - saved")

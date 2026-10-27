"""Coseismic slip of every sensor along the fault, each with its own event time.

Opened from an event that carries a slip channel array (``event['slip']``: one
row per eddy-current sensor, in micrometres, with sensor ``positions``).  Each
sensor ``i`` gets its own event time ``t_i``, found at its largest smoothed
slip rate or set by hand, and its coseismic slip is the difference at ``t_i``
of two lines fitted before and after it (see
:mod:`labquake_explorer.analysis.coseismic_slip`).

The trace panel shows the selected sensor: its slip, the two fit windows, the
lines extrapolated to ``t_i`` and, on the right axis, the smoothed slip rate
used for the automatic pick.  ``t_i`` and the four window edges can be dragged
(or a window by its body); dragging ``t_i`` makes that sensor manual.  The same
values can be typed in the entries.  With "same windows for all sensors" a
window edit applies to every sensor, relative to each one's ``t_i``.
Everything recomputes as soon as a value changes.

Save stores ``event['slip_analysis']``: the record of
:meth:`SlipAnalysis.as_dict` plus ``field``, ``unit`` and ``same_windows``.
"""
from __future__ import annotations

import math
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Optional

import numpy as np

from labquake_explorer.analysis import coseismic_slip as cs
from labquake_explorer.data.channels import channel_names, is_channel_array
from labquake_explorer.data.picks import extraction_record
from labquake_explorer.ui.actions import register_view
from labquake_explorer.ui.context import EVENT
from labquake_explorer.ui.views.base import EventView

LIVE_DELAY_MS = 400                    # recompute this long after the last keystroke
HANDLE_PICK_PX = 6                     # a press this close (pixels) to t_i or a window edge grabs it
ZOOM_HALF_S = 0.2                      # "Zoom to t_i" shows t_i +- this
METHOD_LABELS = {cs.METHOD_MAX_RATE: "max slip rate", cs.METHOD_MANUAL: "manual"}
BEFORE_COLOR, AFTER_COLOR, TI_COLOR = "#1f77b4", "#ff7f0e", "#d62728"
TABLE_COLUMNS = (("sensor", 70), ("x (mm)", 60), ("t_i from", 90), ("t_i - event (ms)", 100),
                 ("slip (um)", 75), ("sigma (um)", 75), ("flags", 260))
HANDLE_NAMES = ("t_i", "before_start", "before_end", "after_start", "after_end")


def _text(value) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(value):
        return ""
    return f"{value:.6g}"


def _parse(text: str, name: str, scale: float = 1.0) -> float:
    try:
        value = float(str(text).strip())
    except ValueError:
        raise ValueError(f"{name}: enter a number") from None
    if not math.isfinite(value):
        raise ValueError(f"{name}: enter a finite number")
    return value * scale


def slip_fields(event) -> list:
    """Channel arrays of an event aligned with its time axis and in micrometres
    (``slip`` first): the fields this view can analyse."""
    if not isinstance(event, dict) or "time" not in event:
        return []
    n = len(np.asarray(event["time"]).ravel())
    out = []
    for key, value in event.items():
        if is_channel_array(value) and value["data"].shape[1] == n and str(value.get("unit", "")) == "um":
            out.append(str(key))
    return sorted(out, key=lambda k: (k != "slip", k))


@register_view("Slip Distribution", kinds=[EVENT], order=15)
class SlipDistributionView(EventView):
    window_title = "Slip Distribution"
    result_key = "slip_analysis"

    def __init__(self, app, run_idx, event_idx):
        self.field: Optional[str] = None
        self.channels: list = []
        self.x_mm = np.array([])
        self.unit = "um"
        self.data: Optional[np.ndarray] = None
        self.time: Optional[np.ndarray] = None
        self.t_event = 0.0
        self.start_s = -1.0
        self.end_s = 1.0
        self.methods: list = []
        self.t_shift: list = []
        self.fit_before: list = []
        self.fit_after: list = []
        self.sensor = 0
        self.result: Optional[cs.SlipAnalysis] = None
        self.saved: Optional[dict] = None
        self.handles: dict = {}
        self.spans: dict = {}
        self._drag = None
        self._compute_after_id = None
        self._pending = None
        self._syncing = False
        self._shown: dict = {}         # entry variable name -> the text the view last put there
        super().__init__(app, run_idx, event_idx)

    # ------------------------------------------------------------------ ui
    def build_ui(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        left = ttk.Frame(self)
        left.grid(row=0, column=0, rowspan=3, padx=5, pady=5, sticky="nsw")

        frame = ttk.LabelFrame(left, text="Event")
        frame.pack(fill="x", pady=(0, 4))
        self.build_event_selector(frame, row=0, column=0)
        ttk.Label(frame, text="Field:").grid(row=1, column=0, padx=4, pady=2, sticky="e")
        self.field_combobox = ttk.Combobox(frame, width=10, state="readonly")
        self.field_combobox.grid(row=1, column=1, padx=4, pady=2, sticky="w")
        self.field_combobox.bind("<<ComboboxSelected>>", self.on_field_selected)

        frame = ttk.LabelFrame(left, text="Sensor")
        frame.pack(fill="x", pady=4)
        row = ttk.Frame(frame)
        row.grid(row=0, column=0, columnspan=3, padx=2, pady=2, sticky="w")
        self.sensor_combobox = ttk.Combobox(row, width=18, state="readonly")
        self.sensor_combobox.pack(side="left", padx=2)
        self.sensor_combobox.bind("<<ComboboxSelected>>", self.on_sensor_selected)
        self.prev_button = ttk.Button(row, text="◀", width=2, command=lambda: self.step_sensor(-1))
        self.prev_button.pack(side="left")
        self.next_button = ttk.Button(row, text="▶", width=2, command=lambda: self.step_sensor(+1))
        self.next_button.pack(side="left")
        ttk.Label(frame, text="t_i from:").grid(row=1, column=0, padx=4, pady=2, sticky="e")
        self.method_combobox = ttk.Combobox(frame, width=13, state="readonly", values=list(METHOD_LABELS.values()))
        self.method_combobox.grid(row=1, column=1, columnspan=2, padx=4, pady=2, sticky="w")
        self.method_combobox.bind("<<ComboboxSelected>>", self.on_method_selected)
        self.t_shift_var = tk.StringVar(master=self)
        self.entries = {}
        self._entry(frame, "t_i - event time (ms):", self.t_shift_var, 2, self.on_t_shift_edited)
        self.before_start_var = tk.StringVar(master=self)
        self.before_end_var = tk.StringVar(master=self)
        self.after_start_var = tk.StringVar(master=self)
        self.after_end_var = tk.StringVar(master=self)
        ttk.Label(frame, text="Fit before (s):").grid(row=3, column=0, padx=4, pady=2, sticky="e")
        self._entry_only(frame, self.before_start_var, 3, 1, "before start", self.on_window_edited)
        self._entry_only(frame, self.before_end_var, 3, 2, "before end", self.on_window_edited)
        ttk.Label(frame, text="Fit after (s):").grid(row=4, column=0, padx=4, pady=2, sticky="e")
        self._entry_only(frame, self.after_start_var, 4, 1, "after start", self.on_window_edited)
        self._entry_only(frame, self.after_end_var, 4, 2, "after end", self.on_window_edited)
        ttk.Label(frame, text="windows are relative to t_i", foreground="#606060").grid(
            row=5, column=0, columnspan=3, padx=4, sticky="w")
        self.same_windows_var = tk.BooleanVar(master=self, value=True)
        ttk.Checkbutton(frame, text="same windows for all sensors", variable=self.same_windows_var,
                        command=self.on_same_windows_toggled).grid(row=6, column=0, columnspan=3, padx=4, pady=2, sticky="w")
        buttons = ttk.Frame(frame)
        buttons.grid(row=7, column=0, columnspan=3, padx=2, pady=2, sticky="w")
        ttk.Button(buttons, text="Zoom to t_i", command=self.zoom_to_event_time).pack(side="left", padx=2)
        ttk.Button(buttons, text="Whole window", command=self.show_whole_window).pack(side="left", padx=2)

        frame = ttk.LabelFrame(left, text="Max-rate picks")
        frame.pack(fill="x", pady=4)
        self.reference_var = tk.StringVar(master=self,
                                          value=_text(cs.DEFAULT_REFERENCE_HALFWIDTH_S * 1e3))
        self.smoothing_var = tk.StringVar(master=self, value=_text(cs.DEFAULT_RATE_SMOOTHING_S * 1e3))
        self.search_var = tk.StringVar(master=self, value=_text(cs.DEFAULT_SEARCH_HALFWIDTH_S * 1e3))
        self.snr_var = tk.StringVar(master=self, value=_text(cs.DEFAULT_RATE_SNR))
        for var in (self.reference_var, self.smoothing_var, self.search_var, self.snr_var):
            self._shown[str(var)] = var.get()
        self._entry(frame, "Reference +- (ms):", self.reference_var, 0, self.on_settings_edited)
        self._entry(frame, "Rate smoothing (ms):", self.smoothing_var, 1, self.on_settings_edited)
        self._entry(frame, "Search +- (ms):", self.search_var, 2, self.on_settings_edited)
        self._entry(frame, "Flag below (x rate noise):", self.snr_var, 3, self.on_settings_edited)
        ttk.Label(frame, text="reference: peak of the mean slip rate of\nall sensors near the event time; each\n"
                              "sensor is searched around it", foreground="#606060", justify="left").grid(
            row=4, column=0, columnspan=2, padx=4, sticky="w")
        ttk.Button(frame, text="Re-pick all sensors", command=self.repick_all).grid(
            row=5, column=0, columnspan=2, padx=4, pady=2, sticky="w")

        frame = ttk.Frame(left)
        frame.pack(fill="x", pady=4)
        self.save_button = ttk.Button(frame, text="Save", command=self.save)
        self.save_button.grid(row=0, column=0, padx=4, pady=2, sticky="w")
        ttk.Label(frame, text="recomputes as the values change", foreground="#606060").grid(
            row=0, column=1, padx=6, sticky="w")
        self.status_var = tk.StringVar(master=self, value="")
        ttk.Label(left, textvariable=self.status_var, wraplength=280, foreground="#a03030", justify="left").pack(
            fill="x", pady=2)

        self.make_figure(figsize=(9, 5), figure_kwargs={"layout": "constrained"},
                         row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.ax_trace = self.figure.add_subplot(111)
        self.ax_rate = self.ax_trace.twinx()
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)

        self.table = ttk.Treeview(self, columns=[c for c, _ in TABLE_COLUMNS], show="headings", height=8,
                                  selectmode="browse")
        for name, width in TABLE_COLUMNS:
            self.table.heading(name, text=name)
            self.table.column(name, width=width, anchor="e" if name not in ("sensor", "t_i from", "flags") else "w")
        self.table.grid(row=2, column=1, padx=5, pady=(0, 5), sticky="ew")
        self.table.bind("<ButtonPress-1>", lambda e: self.flush_pending(), add="+")    # before the class binding
        self.table.bind("<<TreeviewSelect>>", self.on_table_select)

    def _entry(self, master, label, var, row, callback, width=9):
        ttk.Label(master, text=label).grid(row=row, column=0, padx=4, pady=2, sticky="e")
        return self._entry_only(master, var, row, 1, label, callback, width)

    def _entry_only(self, master, var, row, column, name, callback, width=9):
        entry = ttk.Entry(master, textvariable=var, width=width)
        entry.grid(row=row, column=column, padx=4, pady=2, sticky="w")
        entry.bind("<KeyRelease>", lambda e: self.schedule(callback), add="+")
        entry.bind("<Return>", lambda e: self.schedule(callback, now=True), add="+")
        entry.bind("<FocusOut>", lambda e: self.schedule(callback, now=True), add="+")
        self.entries[name] = entry
        return entry

    def _show(self, var: tk.StringVar, text: str) -> None:
        """Put ``text`` in an entry and remember it, so that re-reading the same
        (rounded) text never counts as an edit."""
        self._shown[str(var)] = text
        var.set(text)

    def _edited(self, *variables) -> bool:
        """True when the user changed the text of any of these entries."""
        return any(var.get() != self._shown.get(str(var)) for var in variables)

    def on_close(self) -> None:
        if self._compute_after_id is not None:
            try:
                self.after_cancel(self._compute_after_id)
            except tk.TclError:
                pass
            self._compute_after_id = None
            self._pending = None
        super().on_close()

    # ------------------------------------------------------------- loading
    def set_event(self, event_idx: int) -> None:
        """Switch events, refusing (with a warning) one without a slip array."""
        self.flush_pending()
        event_idx = int(event_idx)
        try:
            candidate = self.data_manager.get_data(f"{self.run_path}/events/[{event_idx}]")
        except (KeyError, IndexError, TypeError):
            candidate = None
        if not slip_fields(candidate):
            self.refresh_event_selector()
            messagebox.showwarning("No slip data", f"Event {event_idx} has no slip channel array; "
                                                   f"staying on event {self.event_idx}.", parent=self)
            return
        super().set_event(event_idx)

    def on_event_loaded(self) -> None:
        fields = slip_fields(self.event)
        if not fields:
            raise ValueError(f"{self.event_path} has no slip channel array (one row per sensor, in um) "
                             "to compute the coseismic slip of")
        saved = self.load_results()
        saved_field = str(saved.get("field")) if isinstance(saved, dict) and saved.get("field") is not None else None
        if saved_field in fields:
            field = saved_field
        elif self.field in fields:
            field = self.field
        else:
            field = fields[0]
        self.field_combobox.config(values=fields)
        self.field_combobox.set(field)
        self.load_field(field, saved)

    def on_field_selected(self, event=None) -> None:
        field = self.field_combobox.get()
        if field and field != self.field:
            self.flush_pending()
            self.load_field(field, self.load_results())

    def extraction_window(self) -> tuple[float, float]:
        """The event's extraction window ``(start_s, end_s)`` around the event
        time: from the run's ``event_extraction``, else the event's time axis."""
        try:
            record = extraction_record(self.get_run())
            start, end = float(record["start_s"]), float(record["end_s"])
            if start < 0.0 < end:
                return start, end
        except (KeyError, TypeError, ValueError):
            pass
        dt = cs.sampling_interval(self.time)
        return min(float(self.time[0]) - self.t_event, -dt), max(float(self.time[-1]) - self.t_event + dt, dt)

    def load_field(self, field: str, saved=None) -> None:
        array = self.event[field]
        self.field = field
        self.channels = channel_names(array)
        self.data = np.asarray(array["data"], dtype=float)
        self.unit = str(array.get("unit", "um"))
        positions = array.get("positions")
        try:
            x = np.asarray(positions["x"], dtype=float).ravel()
        except (KeyError, TypeError, ValueError):
            x = np.full(len(self.channels), np.nan)
        self.x_mm = x if x.size == len(self.channels) else np.full(len(self.channels), np.nan)
        self.time = np.asarray(self.event["time"], dtype=float).ravel()
        self.t_event = float(self.event["event_time"])
        self.start_s, self.end_s = self.extraction_window()
        n = len(self.channels)
        record = saved if isinstance(saved, dict) and str(saved.get("field", field)) == field else None
        inputs = cs.restore_inputs(record, self.channels) if record is not None else None
        if inputs is not None:
            self.methods = list(inputs["methods"])
            self.t_shift = list(inputs["t_shift_s"])
            self.fit_before = list(inputs["fit_before_s"])
            self.fit_after = list(inputs["fit_after_s"])
            self._show(self.reference_var, _text(inputs["reference_halfwidth_s"] * 1e3))
            self._show(self.smoothing_var, _text(inputs["rate_smoothing_s"] * 1e3))
            self._show(self.search_var, _text(inputs["search_halfwidth_s"] * 1e3))
            self._show(self.snr_var, _text(inputs["rate_snr"]))
            self.same_windows_var.set(bool(np.asarray(record.get("same_windows", True)).ravel()[0]))
            self.saved = record
        else:
            before, after = cs.default_fit_windows(self.start_s, self.end_s)
            self.methods = [cs.METHOD_MAX_RATE] * n
            self.t_shift = [float("nan")] * n
            self.fit_before = [before] * n
            self.fit_after = [after] * n
            self.saved = None
        self.sensor_combobox.config(values=[self.sensor_label(i) for i in range(n)])
        self.sensor = min(max(self.sensor, 0), n - 1)
        self.result = None                           # never keep another event's or field's result
        if not self.recompute(reset_view=True):
            self.clear_display()

    def sensor_label(self, i: int) -> str:
        x = self.x_mm[i]
        return f"{i + 1}: {self.channels[i]}" + (f" ({x:g} mm)" if math.isfinite(x) else "")

    def clear_display(self) -> None:
        """No result: empty table and axes; the status keeps the reason."""
        self.table.delete(*self.table.get_children())
        self.ax_trace.clear()
        self.ax_rate.clear()
        self.handles = {}
        self.spans = {}
        self.canvas.draw_idle()

    # ----------------------------------------------------------- settings
    def read_settings(self) -> dict:
        reference = _parse(self.reference_var.get(), "reference half-width (ms)", 1e-3)
        smoothing = _parse(self.smoothing_var.get(), "rate smoothing (ms)", 1e-3)
        search = _parse(self.search_var.get(), "search half-width (ms)", 1e-3)
        snr = _parse(self.snr_var.get(), "flag threshold")
        if reference <= 0:
            raise ValueError("the reference half-width must be > 0")
        if smoothing < 0:
            raise ValueError("rate smoothing must be >= 0")
        if search <= 0:
            raise ValueError("the search half-width must be > 0")
        if snr < 0:
            raise ValueError("the flag threshold must be >= 0")
        return {"rate_smoothing_s": smoothing, "search_halfwidth_s": search, "rate_snr": snr,
                "reference_halfwidth_s": reference}

    def schedule(self, callback, now: bool = False) -> None:
        """Run ``callback`` at once, or once the keys have been quiet for ``LIVE_DELAY_MS``."""
        if self._compute_after_id is not None:
            try:
                self.after_cancel(self._compute_after_id)
            except tk.TclError:
                pass
            self._compute_after_id = None
        pending, self._pending = self._pending, None
        if pending is not None and pending is not callback:
            pending()                                   # a different edit was waiting: apply it first
        if now:
            callback()
        else:
            self._pending = callback
            self._compute_after_id = self.after(LIVE_DELAY_MS, self._run_scheduled)

    def _run_scheduled(self) -> None:
        self._compute_after_id = None
        callback, self._pending = self._pending, None
        if callback is not None:
            callback()

    def flush_pending(self) -> None:
        """Apply a typed edit that is still waiting for the keys to settle."""
        if self._compute_after_id is not None:
            try:
                self.after_cancel(self._compute_after_id)
            except tk.TclError:
                pass
            self._compute_after_id = None
        callback, self._pending = self._pending, None
        if callback is not None:
            callback()

    def recompute(self, reset_view: bool = False) -> bool:
        """Analyse every sensor with the current inputs and redraw; False (and the
        reason in the status line) when an input is unusable."""
        try:
            settings = self.read_settings()
            result = cs.analyze_slip(self.time, self.data, self.t_event, self.start_s, self.end_s,
                                     channels=self.channels, x_mm=self.x_mm, methods=self.methods,
                                     t_shift_s=self.t_shift, fit_before_s=self.fit_before,
                                     fit_after_s=self.fit_after, **settings)
        except ValueError as e:
            self.status_var.set(str(e))
            return False
        for var in (self.reference_var, self.smoothing_var, self.search_var, self.snr_var):
            self._shown[str(var)] = var.get()
        self.result = result
        self.methods = [s.method for s in result.sensors]
        self.t_shift = [s.t_shift_s for s in result.sensors]
        self.show_sensor()
        self.fill_table()
        self.plot(reset_view=reset_view)
        self.update_status()
        return True

    # --------------------------------------------------------------- sensor
    def current(self) -> Optional[cs.SensorSlip]:
        if self.result is None or not (0 <= self.sensor < len(self.result.sensors)):
            return None
        return self.result.sensors[self.sensor]

    def show_sensor(self) -> None:
        """Put the selected sensor's values into the controls."""
        s = self.current()
        if s is None:
            return
        self._syncing = True
        try:
            self.sensor_combobox.set(self.sensor_label(self.sensor))
            self.method_combobox.set(METHOD_LABELS[s.method])
            self._show(self.t_shift_var, _text(round(s.t_shift_s * 1e3, 6)))
            self._show(self.before_start_var, _text(s.fit_before_s[0]))
            self._show(self.before_end_var, _text(s.fit_before_s[1]))
            self._show(self.after_start_var, _text(s.fit_after_s[0]))
            self._show(self.after_end_var, _text(s.fit_after_s[1]))
            iid = str(self.sensor)
            if self.table.exists(iid) and tuple(self.table.selection()) != (iid,):
                self.table.selection_set(iid)
                self.table.see(iid)
        finally:
            self._syncing = False

    def set_sensor(self, i: int) -> None:
        if self.result is None:
            return
        self.flush_pending()
        i = min(max(int(i), 0), len(self.channels) - 1)
        if i == self.sensor and self.sensor_combobox.get() == self.sensor_label(i):
            return
        self.sensor = i
        self.show_sensor()
        self.plot(reset_view=False, new_sensor=True)
        self.update_status()

    def step_sensor(self, delta: int) -> None:
        self.set_sensor(self.sensor + delta)

    def on_sensor_selected(self, event=None) -> None:
        values = list(self.sensor_combobox["values"])
        text = self.sensor_combobox.get()
        if text in values:
            self.set_sensor(values.index(text))

    def on_table_select(self, event=None) -> None:
        if self._syncing:
            return
        selection = self.table.selection()
        if selection:
            self.set_sensor(int(selection[0]))

    def fill_table(self) -> None:
        self._syncing = True
        try:
            self.table.delete(*self.table.get_children())
            for i, s in enumerate(self.result.sensors):
                self.table.insert("", "end", iid=str(i), values=(
                    s.channel, _text(s.x_mm), METHOD_LABELS[s.method], f"{s.t_shift_s * 1e3:.2f}",
                    f"{s.coseismic_slip:.3f}" if math.isfinite(s.coseismic_slip) else "n/a",
                    f"{s.noise:.3f}" if math.isfinite(s.noise) else "n/a", "; ".join(s.flags)))
            if self.table.exists(str(self.sensor)):
                self.table.selection_set(str(self.sensor))
        finally:
            self._syncing = False

    # --------------------------------------------------------------- edits
    def on_method_selected(self, event=None) -> None:
        self.flush_pending()
        if self.result is None:
            return
        label = self.method_combobox.get()
        method = next((k for k, v in METHOD_LABELS.items() if v == label), None)
        if method is None or method == self.methods[self.sensor]:
            return
        self.methods[self.sensor] = method       # manual keeps the current t_i as its start
        self.recompute()

    def on_t_shift_edited(self) -> None:
        if self._syncing or self.current() is None or not self._edited(self.t_shift_var):
            return
        try:
            shift = _parse(self.t_shift_var.get(), "t_i - event time (ms)", 1e-3)
        except ValueError as e:
            self.status_var.set(str(e))
            return
        self.set_event_time(self.sensor, shift)

    def set_event_time(self, i: int, shift_s: float) -> None:
        """Make sensor ``i`` manual at ``event_time + shift_s`` and recompute."""
        self.methods[i] = cs.METHOD_MANUAL
        self.t_shift[i] = float(shift_s)
        self.recompute()

    def read_windows(self) -> tuple[tuple, tuple]:
        """The windows typed in the entries (ValueError when unusable)."""
        before = cs.check_window((_parse(self.before_start_var.get(), "fit before start (s)"),
                                  _parse(self.before_end_var.get(), "fit before end (s)")), "fit before")
        after = cs.check_window((_parse(self.after_start_var.get(), "fit after start (s)"),
                                 _parse(self.after_end_var.get(), "fit after end (s)")), "fit after")
        self.check_sides(before, after)
        return before, after

    def on_window_edited(self) -> None:
        window_vars = (self.before_start_var, self.before_end_var, self.after_start_var, self.after_end_var)
        if self._syncing or self.result is None or not self._edited(*window_vars):
            return
        try:
            before, after = self.read_windows()
        except ValueError as e:
            self.status_var.set(str(e))
            return
        self.set_windows(before, after)

    @staticmethod
    def check_sides(before, after) -> None:
        if before[1] > 0.0:
            raise ValueError("the before window must end at or before t_i (end <= 0)")
        if after[0] < 0.0:
            raise ValueError("the after window must start at or after t_i (start >= 0)")

    def set_windows(self, before, after) -> None:
        """Windows of the selected sensor, or of every sensor with "same windows"."""
        targets = range(len(self.channels)) if self.same_windows_var.get() else [self.sensor]
        for i in targets:
            self.fit_before[i] = tuple(float(v) for v in before)
            self.fit_after[i] = tuple(float(v) for v in after)
        self.recompute()

    def on_same_windows_toggled(self) -> None:
        self.flush_pending()
        if self.same_windows_var.get() and self.result is not None:
            s = self.current()
            self.set_windows(s.fit_before_s, s.fit_after_s)
        else:
            self.update_status()

    def on_settings_edited(self) -> None:
        # with no result (a bad value stopped the last recompute) any value is worth a try
        if self.result is None or self._edited(self.reference_var, self.smoothing_var, self.search_var, self.snr_var):
            self.recompute()

    def repick_all(self) -> None:
        """Every sensor back to the automatic max-rate pick."""
        self.flush_pending()
        self.methods = [cs.METHOD_MAX_RATE] * len(self.channels)
        self.recompute()

    # ------------------------------------------------------------- plotting
    def plot(self, reset_view: bool = False, new_sensor: bool = False) -> None:
        ax, ax_rate = self.ax_trace, self.ax_rate
        keep = not reset_view and ax.has_data()
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.clear()
        ax_rate.clear()
        # the rate axis sits behind the trace, so the slip, the handles and the legend draw on top
        # and the mouse events land on the trace axes; clear() resets these, so set them every time
        ax_rate.set_zorder(ax.get_zorder() - 1)
        ax.patch.set_visible(False)
        ax_rate.yaxis.set_label_position("right")
        ax_rate.yaxis.tick_right()
        self.handles = {}
        self.spans = {}
        s = self.current()
        if s is None:
            self.canvas.draw_idle()
            return
        t_rel = self.time - self.t_event
        y = self.data[self.sensor]
        ti = s.t_shift_s
        (b0, b1), (a0, a1) = s.fit_before_s, s.fit_after_s
        ax.plot(t_rel, y, color="#4d4d4d", lw=0.8, label=s.channel, zorder=2)
        self.spans["before"] = ax.axvspan(ti + b0, ti + b1, color=BEFORE_COLOR, alpha=0.12, zorder=0)
        self.spans["after"] = ax.axvspan(ti + a0, ti + a1, color=AFTER_COLOR, alpha=0.12, zorder=0)
        if s.before.valid:
            tb = np.linspace(ti + b0, ti, 50)
            ax.plot(tb, s.before.at(tb - ti), "--", color=BEFORE_COLOR, lw=1.4, label="fit before", zorder=3)
        if s.after.valid:
            ta = np.linspace(ti, ti + a1, 50)
            ax.plot(ta, s.after.at(ta - ti), "--", color=AFTER_COLOR, lw=1.4, label="fit after", zorder=3)
        self.handles["t_i"] = ax.axvline(ti, color=TI_COLOR, lw=1.8, label="t_i", zorder=5)
        for name, x, color in (("before_start", ti + b0, BEFORE_COLOR), ("before_end", ti + b1, BEFORE_COLOR),
                               ("after_start", ti + a0, AFTER_COLOR), ("after_end", ti + a1, AFTER_COLOR)):
            self.handles[name] = ax.axvline(x, color=color, lw=1.2, label="_" + name, zorder=5)
        if s.before.valid and s.after.valid:
            ax.plot([ti, ti], [s.before.value, s.after.value], "-", color="k", lw=2.2, zorder=6,
                    marker="_", markersize=14, markeredgewidth=2.2,
                    label=f"slip {s.coseismic_slip:.3g} {self.unit}")
        rate = self.result.rate[self.sensor]
        ax_rate.plot(t_rel, rate, color="#9a9a9a", lw=0.6, label="slip rate (smoothed)", zorder=1)
        if s.method == cs.METHOD_MAX_RATE:
            t_ref = self.result.t_ref - self.t_event
            hw = self.result.search_halfwidth_s
            ax_rate.axvspan(t_ref - hw, t_ref + hw, color="#cccccc", alpha=0.25, zorder=0)
        ax_rate.set_ylabel(f"slip rate ({self.unit}/s)")
        ax.set_xlabel(f"time - event time {self.t_event:.4f} s (s)")
        ax.set_ylabel(f"{self.field} ({self.unit})")
        flags = f"  [{'; '.join(s.flags)}]" if s.flags else ""
        ax.set_title(f"{self.figure_title()}  {self.sensor_label(self.sensor)}  t_i from {METHOD_LABELS[s.method]}"
                     f"{flags}", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_rate.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
        if keep:
            ax.set_xlim(xlim)
            if new_sensor:
                self._fit_ylim()
            else:
                ax.set_ylim(ylim)
        else:
            ax.set_xlim(float(t_rel[0]), float(t_rel[-1]))
        self._fit_rate_ylim()
        self.canvas.draw_idle()

    def _visible(self) -> np.ndarray:
        lo, hi = self.ax_trace.get_xlim()
        t_rel = self.time - self.t_event
        return (t_rel >= lo) & (t_rel <= hi)

    def _fit_ylim(self) -> None:
        if self.result is None:
            return
        y = self.data[self.sensor][self._visible()]
        y = y[np.isfinite(y)]
        if y.size and y.max() > y.min():
            pad = 0.05 * (y.max() - y.min())
            self.ax_trace.set_ylim(y.min() - pad, y.max() + pad)

    def _fit_rate_ylim(self) -> None:
        if self.result is None:
            return
        r = self.result.rate[self.sensor][self._visible()]
        r = r[np.isfinite(r)]
        if r.size and r.max() > r.min():
            pad = 0.05 * (r.max() - r.min())
            self.ax_rate.set_ylim(r.min() - pad, r.max() + pad)

    def zoom_to_event_time(self) -> None:
        s = self.current()
        if s is None:
            return
        self.ax_trace.set_xlim(s.t_shift_s - ZOOM_HALF_S, s.t_shift_s + ZOOM_HALF_S)
        self._fit_ylim()
        self._fit_rate_ylim()
        self.canvas.draw_idle()

    def show_whole_window(self) -> None:
        if self.result is None:
            return
        t_rel = self.time - self.t_event
        self.ax_trace.set_xlim(float(t_rel[0]), float(t_rel[-1]))
        self._fit_ylim()
        self._fit_rate_ylim()
        self.canvas.draw_idle()

    # ------------------------------------------------------------- dragging
    def _in_trace(self, event) -> bool:
        return event.inaxes in (self.ax_trace, self.ax_rate) and event.xdata is not None

    def handle_at(self, event) -> Optional[str]:
        """The handle (t_i or a window edge) within ``HANDLE_PICK_PX`` of the
        press, the nearest one; t_i wins a tie."""
        best, best_d = None, HANDLE_PICK_PX + 1e-9
        for name in HANDLE_NAMES:
            line = self.handles.get(name)
            if line is None:
                continue
            x = float(line.get_xdata()[0])
            px = self.ax_trace.transData.transform((x, 0.0))[0]
            d = abs(px - float(event.x))
            if d < best_d:
                best, best_d = name, d
        return best

    def on_press(self, event) -> None:
        """Grab ``t_i`` or a window edge near the press, else a fit window by its
        body.  The hit test is done here in pixels: pick events would go to the
        rate axes, which matplotlib asks first."""
        self.flush_pending()
        if self.toolbar_active() or self._drag is not None or self.current() is None or not self._in_trace(event):
            return
        if event.button != 1:
            return
        x_data = float(self.ax_trace.transData.inverted().transform((event.x, event.y))[0])
        name = self.handle_at(event)
        if name is not None:
            self.begin_drag(name, x_data)
            return
        s = self.current()
        x = x_data - s.t_shift_s
        if s.fit_before_s[0] <= x <= s.fit_before_s[1]:
            self.begin_drag("before", x_data)
        elif s.fit_after_s[0] <= x <= s.fit_after_s[1]:
            self.begin_drag("after", x_data)

    def begin_drag(self, what: str, x: float) -> None:
        s = self.current()
        self._drag = {"what": what, "x0": float(x), "t_i": s.t_shift_s,
                      "before": tuple(s.fit_before_s), "after": tuple(s.fit_after_s), "moved": False}

    def on_motion(self, event) -> None:
        if self._drag is None or not self._in_trace(event):
            return
        self.drag_to(float(self.ax_trace.transData.inverted().transform((event.x, event.y))[0]))

    def drag_to(self, x: float) -> dict:
        """Move the grabbed handle (or window) to ``x`` (s from the event time);
        the entries and the handles follow, the analysis waits for the release."""
        d = self._drag
        dt = cs.sampling_interval(self.time)
        ti = d["t_i"]
        (b0, b1), (a0, a1) = d["before"], d["after"]
        what = d["what"]
        rel = x - ti
        if what == "t_i":
            ti = x
        elif what == "before_start":
            b0 = min(rel, b1 - dt)
        elif what == "before_end":
            b1 = min(max(rel, b0 + dt), 0.0)
        elif what == "after_start":
            a0 = max(min(rel, a1 - dt), 0.0)
        elif what == "after_end":
            a1 = max(rel, a0 + dt)
        elif what == "before":
            shift = min(x - d["x0"], -b1)
            b0, b1 = b0 + shift, b1 + shift
        elif what == "after":
            shift = max(x - d["x0"], -a0)
            a0, a1 = a0 + shift, a1 + shift
        state = {"t_i": ti, "before": (b0, b1), "after": (a0, a1)}
        d["state"] = state
        d["moved"] = True
        self._syncing = True
        try:
            self._show(self.t_shift_var, _text(round(ti * 1e3, 6)))
            self._show(self.before_start_var, _text(round(b0, 6)))
            self._show(self.before_end_var, _text(round(b1, 6)))
            self._show(self.after_start_var, _text(round(a0, 6)))
            self._show(self.after_end_var, _text(round(a1, 6)))
        finally:
            self._syncing = False
        positions = {"t_i": ti, "before_start": ti + b0, "before_end": ti + b1,
                     "after_start": ti + a0, "after_end": ti + a1}
        for name, line in self.handles.items():
            line.set_xdata([positions[name], positions[name]])
        for name, (lo, hi) in (("before", (ti + b0, ti + b1)), ("after", (ti + a0, ti + a1))):
            patch = self.spans.get(name)
            if patch is not None:
                patch.set_x(lo)
                patch.set_width(hi - lo)
        self.canvas.draw_idle()
        return state

    def on_release(self, event) -> None:
        if self._drag is not None:
            self.end_drag()

    def end_drag(self) -> None:
        """The dragged values become the sensor's settings: recompute."""
        d, self._drag = self._drag, None
        if d is None or not d.get("moved"):
            return
        state = d["state"]
        if d["what"] == "t_i":
            self.set_event_time(self.sensor, state["t_i"])
        else:
            self.set_windows(state["before"], state["after"])

    # ----------------------------------------------------------------- save
    def record(self) -> Optional[dict]:
        """The ``slip_analysis`` record of the current state, or None."""
        if self.result is None:
            return None
        rec = self.result.as_dict()
        rec.update({"field": self.field, "unit": self.unit, "same_windows": bool(self.same_windows_var.get())})
        return rec

    def save(self) -> bool:
        """Store the record; refused while an entry holds a value that was not
        applied, or when there is no result for this event."""
        self.flush_pending()
        try:
            self.read_settings()
            _parse(self.t_shift_var.get(), "t_i - event time (ms)")
            self.read_windows()
        except ValueError as e:
            self.status_var.set(f"not saved: {e}")
            return False
        rec = self.record()
        if rec is None or self.result.event_time != self.t_event:
            self.status_var.set("nothing to save - the current values give no result")
            return False
        self.save_results(rec)
        self.saved = rec
        self.update_status()
        return True

    def matches_saved(self) -> bool:
        """True when the inputs equal those of the saved record."""
        rec = self.record()
        if rec is None or not isinstance(self.saved, dict):
            return False
        inputs = cs.restore_inputs(self.saved, self.channels)
        if inputs is None or str(self.saved.get("field", self.field)) != self.field:
            return False
        same = (inputs["methods"] == rec["time_method"]
                and np.allclose(inputs["fit_before_s"], rec["fit_before_s"])
                and np.allclose(inputs["fit_after_s"], rec["fit_after_s"])
                and math.isclose(inputs["rate_smoothing_s"], rec["rate_smoothing_s"])
                and math.isclose(inputs["search_halfwidth_s"], rec["search_halfwidth_s"])
                and math.isclose(inputs["rate_snr"], rec["rate_snr"])
                and math.isclose(inputs["reference_halfwidth_s"], rec["reference_halfwidth_s"])
                and bool(np.asarray(self.saved.get("same_windows", True)).ravel()[0]) == rec["same_windows"])
        manual = [i for i, m in enumerate(rec["time_method"]) if m == cs.METHOD_MANUAL]
        return bool(same and all(math.isclose(inputs["t_shift_s"][i], rec["t_shift_s"][i], abs_tol=1e-12)
                                 for i in manual))

    def update_status(self) -> None:
        if self.result is None:
            return
        flagged = [s for s in self.result.sensors if s.flags]
        manual = sum(1 for s in self.result.sensors if s.method == cs.METHOD_MANUAL)
        parts = [f"{len(self.result.sensors)} sensors, {manual} manual; "
                 f"reference at {1e3 * (self.result.t_ref - self.t_event):+.1f} ms"]
        if flagged:
            parts.append("check: " + ", ".join(f"{s.channel} ({'; '.join(s.flags)})" for s in flagged))
        if self.saved is None:
            parts.append("not saved")
        else:
            parts.append("matches the saved record" if self.matches_saved() else "differs from the saved record (Save to update)")
        self.status_var.set(" | ".join(parts))

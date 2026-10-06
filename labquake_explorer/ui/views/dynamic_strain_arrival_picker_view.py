"""Pick rupture-front arrival times on the strain-gauge array of one event.

The figure stacks the four mechanical traces above the strain waveforms,
each waveform drawn at its location along the fault.  One draggable marker
per enabled channel marks the arrival; a straight line through the markers
of the *fitting* channels gives the rupture speed ``Cf``.

Results are written into the event's existing layout:

* ``event['rupture_speed']`` (only when the fit is well-posed; a degenerate
  fit removes the key so downstream views fall back to their defaults)
* ``event['strain']['enabled_channels']``, ``event['strain']['fitting_channels']``
* ``picked_idx`` and ``rupture_arrival_time`` next to the record's samples
  (``event['waveform'][recorder]``, or ``event['strain']['original']`` for the PSU layout)
"""
from __future__ import annotations

import math
import tkinter as tk
from tkinter import messagebox, ttk
import warnings

import matplotlib.patches as patches
import numpy as np
from scipy import signal

from labquake_explorer.ui.actions import register_view
from labquake_explorer.ui.context import EVENT
from labquake_explorer.ui.views.base import EventView
from labquake_explorer.data.channels import aligned_fields, get_field
from labquake_explorer.data.sources import (pick_waveform_block, waveform_block, waveform_data, waveform_store,
                                            waveform_time)

# Experiments from this number on use the 16-channel layout with three
# extra gauges at the ends of the fault; older ones use paired gauges.
NEW_LAYOUT_EXP_NUMBER = 5958

# Savitzky-Golay polynomial order; the window must be longer than this.
FILTER_POLYORDER = 2
MIN_FILTER_WINDOW = FILTER_POLYORDER + 1
DEFAULT_FILTER_WINDOW = 51

_RankWarning = getattr(np, "exceptions", np).RankWarning


@register_view("Pick Arrivals", kinds=[EVENT], order=20)
class DynamicStrainArrivalPickerView(EventView):
    window_title = "Pick Arrivals"

    def __init__(self, app, run_idx, event_idx):
        self.enabled_channels = None
        self.fitting_channels = None
        self.picked_idx = None
        self.lines = []
        self.axs = None
        self.fitting_markers = []
        self.not_fitting_markers = []
        self.offset = [0, 0]
        self.current_artist = None
        self.currently_dragging = False
        self.rupture_speed = None
        self.fitted_line = None
        self.filtering = False
        self.xlim = None
        super().__init__(app, run_idx, event_idx)

    # ------------------------------------------------------------------ ui
    def build_ui(self):
        # 7 columns, 4 rows: controls / figure / toolbar / filter row
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(5, weight=1)

        self.build_event_selector(self, row=0, column=0)

        self.enabled_channels_mb = tk.Menubutton(self, text="Enabled Channels")
        self.enabled_channels_mb.grid(row=0, column=2, padx=5, pady=5)
        self.enabled_channels_mb.menu = tk.Menu(self.enabled_channels_mb, tearoff=0)
        self.enabled_channels_mb["menu"] = self.enabled_channels_mb.menu
        self.enabled_channels_mb.items = []

        self.fitting_channels_mb = tk.Menubutton(self, text="Fitting Channels")
        self.fitting_channels_mb.grid(row=0, column=3, padx=5, pady=5)
        self.fitting_channels_mb.menu = tk.Menu(self.fitting_channels_mb, tearoff=0)
        self.fitting_channels_mb["menu"] = self.fitting_channels_mb.menu
        self.fitting_channels_mb.items = []

        self.cf_label = ttk.Label(self, text="Cf = n/a")
        self.cf_label.grid(row=0, column=4, padx=5, pady=5, sticky="e")
        self.magic_button = tk.Button(self, text="Magic", command=self.magic)
        self.magic_button.grid(row=0, column=5, padx=5, pady=5, sticky="e")
        self.save_button = tk.Button(self, text="Save", command=self.save)
        self.save_button.grid(row=0, column=6, padx=5, pady=5, sticky="e")

        # figure (row 1) and toolbar (row 2)
        self.make_figure(figsize=(7, 7), row=1, column=0, columnspan=7, padx=5, pady=5, sticky="nsew")
        self.figure.set_layout_engine("constrained")
        self.fig = self.figure  # legacy alias

        # filter row
        ttk.Label(self, text="Filter:", justify="left").grid(row=3, column=0, padx=5, pady=5, sticky="ew")
        self.filter_combobox = ttk.Combobox(self, state="disabled")
        self.filter_combobox.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        self.filter_combobox["values"] = ("scipy.savgol_filter",)
        self.filter_combobox.current(0)
        ttk.Label(self, text="Window length", justify="right").grid(row=3, column=2, padx=5, pady=5, sticky="w")
        self.filter_window_length = tk.StringVar(value=str(DEFAULT_FILTER_WINDOW))
        self.filter_window_length_box = ttk.Spinbox(self, from_=MIN_FILTER_WINDOW, to=201, increment=2,
                                                    textvariable=self.filter_window_length)
        self.filter_window_length_box.grid(row=3, column=3, padx=5, pady=5, sticky="ew")
        self.filter_toggle = tk.Button(self, text="Filter Off", relief="raised", command=self.toggle_filter)
        self.filter_toggle.grid(row=3, column=4, padx=5, pady=5, sticky="ew")

        self.figure.canvas.mpl_connect("pick_event", self.on_pick)
        self.figure.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.figure.canvas.mpl_connect("button_press_event", self.on_press)
        self.figure.canvas.mpl_connect("button_release_event", self.on_release)
        self.figure.canvas.mpl_connect("resize_event", self.on_resize)
        self.figure.canvas.mpl_connect("scroll_event", self.on_resize)
        self.filter_window_length_box.bind("<ButtonRelease>", self.on_filter_window_length_box_changed)

    # ------------------------------------------------------------- loading
    def exp_number(self) -> int:
        """Experiment number parsed from the name, e.g. 'p5958' -> 5958 (0 if unparsable)."""
        try:
            return int(self.experiment_name()[1:5])
        except ValueError:
            return 0

    @staticmethod
    def strain_block(event):
        """The event's full-rate record: ``strain`` when present, else the first
        waveform block (``waveform/elsys``, ``waveform/ni``, ...); None without any."""
        if not isinstance(event, dict):
            return None
        key = pick_waveform_block(event, prefer_keys=("strain",))
        return waveform_block(event, key) if key else None

    @staticmethod
    def fault_slip(event):
        """The event's fault slip trace: ``displacement`` (PSU files), else the
        first slip channel (``slip/slip_1``); None without either."""
        n = len(event.get("time", []))
        for path in ["displacement"] + [f for f in aligned_fields(event, n) if f.split("/")[-1].lower().startswith("slip")]:
            value = get_field(event, path)
            if value is not None and np.asarray(value).shape == (n,):
                return np.asarray(value, dtype=float)
        return None

    @staticmethod
    def block_locations(block, n_channels: int):
        """Gauge positions along the fault from the record's ``positions`` table
        (its ``x``), when it has one finite entry per channel; else None."""
        positions = block.get("positions") if isinstance(block, dict) else None
        xs = positions.get("x") if isinstance(positions, dict) else None
        if xs is None or len(xs) != n_channels:
            return None
        xs = [float(v) for v in xs]
        return xs if all(math.isfinite(v) for v in xs) else None

    def set_event(self, event_idx: int) -> None:
        """Switch events, but refuse (with a warning) an event without strain data.

        The check happens before ``EventView.set_event`` touches ``event_idx``,
        ``event``, the title or the combobox, so a refused switch leaves the
        view exactly as it was.
        """
        event_idx = int(event_idx)
        try:
            candidate = self.data_manager.get_data(f"{self.run_path}/events/[{event_idx}]")
        except (KeyError, IndexError, TypeError):
            candidate = None
        if self.strain_block(candidate) is None:
            self.refresh_event_selector()  # combobox may already show the refused index
            messagebox.showwarning(
                "No strain data",
                f"Event {event_idx} has no strain data to pick arrivals on; "
                f"staying on event {self.event_idx}.",
                parent=self,
            )
            return
        super().set_event(event_idx)

    def on_event_loaded(self):
        strain = self.strain_block(self.event)
        if strain is None:
            if self.axs is None:  # first load: nothing to show, do not leave a dead window
                self.on_close()
            raise ValueError(f"{self.event_path} has no strain data to pick arrivals on")
        store = waveform_store(strain)

        # saved channel state lives in the strain dict, not on the event
        if "enabled_channels" in strain:
            self.enabled_channels = [bool(v) for v in strain["enabled_channels"]]
        else:
            self.enabled_channels = None
        if "fitting_channels" in strain:
            self.fitting_channels = [bool(v) for v in strain["fitting_channels"]]
        else:
            self.fitting_channels = None
        if "picked_idx" in store:
            self.picked_idx = [int(v) for v in store["picked_idx"]]
        else:
            self.picked_idx = None

        self.xlim = None
        self.fitting_markers = []
        self.not_fitting_markers = []
        self.offset = [0, 0]
        self.current_artist = None
        self.currently_dragging = False
        self.rupture_speed = None
        self.fitted_line = None
        self.plot()
        self.init_enabled_channels_mb()
        self.init_fitting_channels_mb()
        self.update_fitted_line()

    # ------------------------------------------------------------ defaults
    @staticmethod
    def default_enabled_channels(n_channels: int, exp_number: int) -> list:
        if exp_number >= NEW_LAYOUT_EXP_NUMBER:
            return [i < 13 for i in range(n_channels)]
        return [i % 2 == 0 for i in range(n_channels)]

    @staticmethod
    def default_fitting_channels(n_channels: int, exp_number: int) -> list:
        if exp_number >= NEW_LAYOUT_EXP_NUMBER:
            base = [False] + [True] * 8 + [False] * 7
        else:
            base = [i in (6, 8, 10, 12) for i in range(16)]
        return (base + [False] * n_channels)[:n_channels]

    @staticmethod
    def default_locations(n_channels: int, exp_number: int) -> list:
        """Gauge positions along the fault in mm."""
        if exp_number >= NEW_LAYOUT_EXP_NUMBER:
            locs = [2 + 12 * i for i in range(16)]
            locs[-3:] = [2, 74, 146]
            return (locs + [locs[-1]] * n_channels)[:n_channels]
        return [10.5 + 12 * (i // 2) for i in range(n_channels)]

    # ---------------------------------------------------------------- plot
    def filter_window(self) -> int:
        """Savgol window length from the spinbox, as typed (even values are
        fine for scipy's savgol_filter); clamped to > polyorder, 51 if unparsable."""
        try:
            n = int(float(self.filter_window_length.get()))
        except (ValueError, tk.TclError):
            n = DEFAULT_FILTER_WINDOW
        return max(n, MIN_FILTER_WINDOW)

    def plot(self):
        exp_number = self.exp_number()
        strain = self.strain_block(self.event)
        linestyle = ".-"

        self.figure.clear()
        self.fitting_markers = []
        self.not_fitting_markers = []
        self.fitted_line = None
        gs = self.figure.add_gridspec(5, hspace=0, height_ratios=[1, 1, 1, 1, 10])
        self.axs = gs.subplots(sharex=True)
        self.axs[0].set_ylabel(r"$\tau$ (MPa)")
        self.axs[1].set_ylabel(r"$\mu$")
        self.axs[2].set_ylabel(r"$\delta_\mathrm{LP}\ \mathrm{({\mu}m)}$")
        self.axs[3].set_ylabel(r"$\delta\ \mathrm{({\mu}m)}$")

        event_time = self.event["event_time"]
        t = np.asarray(self.event["time"]) - event_time
        self.axs[0].plot(t, self.event["shear_stress"], linestyle, color="C0")
        self.axs[1].plot(t, self.event["friction"], linestyle, color="C0")
        lp = np.asarray(self.event["LP_displacement"])
        self.axs[2].plot(t, lp - lp[0], linestyle, color="C0")
        disp = self.fault_slip(self.event)
        if disp is not None:
            self.axs[3].plot(t, disp - disp[0], linestyle, color="C0")

        tt = waveform_time(strain) - event_time
        y = np.array(waveform_data(strain), dtype=float, copy=True)
        n_channels = y.shape[0]

        if self.enabled_channels is None or len(self.enabled_channels) != n_channels:
            self.enabled_channels = self.default_enabled_channels(n_channels, exp_number)
        if self.fitting_channels is None or len(self.fitting_channels) != n_channels:
            self.fitting_channels = self.default_fitting_channels(n_channels, exp_number)
        if "locations" not in strain or len(strain["locations"]) != n_channels:
            strain["locations"] = self.block_locations(strain, n_channels) or self.default_locations(n_channels, exp_number)

        if self.filtering:
            nf = self.filter_window()
            # keep the spinbox showing the window that was actually applied
            if self.filter_window_length.get() != str(nf):
                self.filter_window_length.set(str(nf))
            for i in range(n_channels):
                y[i, :] = signal.savgol_filter(y[i, :], nf, FILTER_POLYORDER)

        self.lines = [None] * n_channels
        line_idx = 0
        for i in range(n_channels):
            if not self.enabled_channels[i]:
                continue
            loc = strain["locations"][i]
            self.axs[4].plot([tt[0], tt[-1]], [loc, loc], "k:", zorder=-101)
            span = y[i, :].max() - y[i, :].min()
            ratio = -12 / span if span > 0 else 0.0
            self.lines[i] = self.axs[4].plot(tt, y[i, :] * ratio + loc, color="C%d" % line_idx, zorder=-100)[0]
            line_idx += 1
        self.axs[4].set_ylabel("location along fault (mm)")
        self.axs[4].set_xlabel("time - %f (s)" % event_time)
        if exp_number >= NEW_LAYOUT_EXP_NUMBER:
            self.axs[4].set_ylim(160, -10)
        else:
            self.axs[4].set_ylim(105, 0)
        if self.xlim is None:
            self.axs[0].set_xlim(tt[0], tt[-1])
        else:
            self.axs[0].set_xlim(self.xlim)
        self.figure.suptitle(self.figure_title())

        if self.picked_idx is None or len(self.picked_idx) != n_channels:
            middle_idx = y.shape[1] // 2
            self.picked_idx = [middle_idx] * n_channels
        self.draw_markers()

    def draw_markers(self):
        width, height = self.get_circle_dims()
        for marker in self.fitting_markers + self.not_fitting_markers:
            try:
                marker.remove()
            except (ValueError, NotImplementedError):
                pass
        self.fitting_markers = []
        self.not_fitting_markers = []
        for i, idx in enumerate(self.picked_idx):
            if not self.enabled_channels[i] or self.lines[i] is None:
                continue
            color = "red" if self.fitting_channels[i] else "black"
            x, y = self.lines[i].get_data()
            marker = patches.Ellipse((x[idx], y[idx]), width=width, height=height, color=color,
                                     fill=False, lw=2, picker=8, label=str(i))
            self.axs[4].add_patch(marker)
            if self.fitting_channels[i]:
                self.fitting_markers.append(marker)
            else:
                self.not_fitting_markers.append(marker)
        self.canvas.draw()

    def get_circle_dims(self):
        xl = self.axs[4].get_xlim()
        yl = self.axs[4].get_ylim()
        ratio = (yl[-1] - yl[0]) / (xl[-1] - xl[0])
        bbox = self.axs[4].get_window_extent()
        ax_size = [bbox.width, bbox.height]
        ratio *= ax_size[0] / ax_size[1]
        width = (xl[-1] - xl[0]) / ax_size[0] * 0.1 * self.figure.dpi
        return width, width * ratio

    def on_resize(self, event=None):
        self.canvas.draw()
        if self.axs is not None:
            width, height = self.get_circle_dims()
            for marker in self.fitting_markers + self.not_fitting_markers:
                marker.set_width(width)
                marker.set_height(height)
            self.canvas.draw()
            self.xlim = self.axs[0].get_xlim()

    # ------------------------------------------------------------- dragging
    def on_pick(self, event):
        if self.toolbar_active():
            return
        if self.current_artist is None and isinstance(event.artist, patches.Ellipse):
            self.current_artist = event.artist
            x0, y0 = self.current_artist.center
            x1, y1 = event.mouseevent.xdata, event.mouseevent.ydata
            if x1 is None or y1 is None:
                self.offset = [0, 0]
            else:
                self.offset = [(x0 - x1), (y0 - y1)]

    def on_press(self, event):
        if self.toolbar_active():
            return
        self.currently_dragging = True

    def on_release(self, event):
        self.current_artist = None
        self.currently_dragging = False
        self.on_resize()

    def on_motion(self, event):
        if self.toolbar_active():
            return
        if not self.currently_dragging or self.current_artist is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        if not isinstance(self.current_artist, patches.Ellipse):
            return
        try:
            channel = int(self.current_artist.get_label())
            dx, dy = self.offset
            cx, cy = event.xdata + dx, event.ydata + dy
            xl = self.axs[4].get_xlim()
            yl = self.axs[4].get_ylim()
            yw = yl[-1] - yl[0]
            xw = xl[-1] - xl[0]
            x, y = self.lines[channel].get_data()
            idx = int(np.argmin(((x - cx) / xw) ** 2 + ((y - cy) / yw) ** 2))
            self.current_artist.set_center((x[idx], y[idx]))
            self.picked_idx[channel] = idx
            self.update_fitted_line()
        except Exception as e:
            print(f"Error in on_motion: {e}")

    # ------------------------------------------------------------------ fit
    @staticmethod
    def fit_arrival_line(locations, times):
        """Least-squares ``time = slope * location + intercept``; None if ill-posed.

        Ill-posed means fewer than two picks, fewer than two distinct locations,
        all picked times identical (no propagation to measure: polyfit would
        return a slope of ~1e-20 rather than 0 whenever the common time is not
        exactly 0), a rank-deficient fit, or non-finite coefficients.
        """
        locations = np.asarray(locations, dtype=float)
        times = np.asarray(times, dtype=float)
        if len(times) < 2 or len(np.unique(locations)) < 2:
            return None
        if not np.all(np.isfinite(times)) or np.ptp(times) == 0:
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", _RankWarning)
                coeffs = np.polyfit(locations, times, 1)
        except Exception:
            return None
        if not np.all(np.isfinite(coeffs)):
            return None
        return float(coeffs[0]), float(coeffs[1])

    def update_fitted_line(self):
        if self.fitted_line is not None:
            try:
                self.fitted_line.remove()
            except ValueError:
                pass
            self.fitted_line = None
        locations = self.strain_block(self.event)["locations"]
        x = [marker.get_center()[0] for marker in self.fitting_markers]
        y = [locations[int(marker.get_label())] for marker in self.fitting_markers]
        self.rupture_speed = math.nan
        fit = self.fit_arrival_line(y, x)
        if fit is not None:
            slope, intercept = fit
            yy = np.asarray(y, dtype=float)
            self.fitted_line = self.axs[4].plot(slope * yy + intercept, yy, "r--")[0]
            if slope != 0:
                self.rupture_speed = -1e-3 / slope  # mm/s -> m/s, sign = propagation direction
        self.canvas.draw()
        self.update_cf_label()
        self.update_idletasks()

    def has_rupture_speed(self) -> bool:
        return self.rupture_speed is not None and bool(np.isfinite(self.rupture_speed))

    def update_cf_label(self):
        cf = self.rupture_speed
        if not self.has_rupture_speed():
            text = "Cf = n/a"
        elif abs(cf) < 1e4:
            text = f"Cf = {cf:.2f} m/s"
        else:
            text = f"Cf = {cf:.2e} m/s"
        self.cf_label.configure(text=text)

    # ---------------------------------------------------------------- save
    def save(self):
        strain = self.strain_block(self.event)
        store = waveform_store(strain)
        picked = [int(i) for i in self.picked_idx]
        strain["enabled_channels"] = [bool(v) for v in self.enabled_channels]
        strain["fitting_channels"] = [bool(v) for v in self.fitting_channels]
        store["picked_idx"] = picked
        store["rupture_arrival_time"] = waveform_time(strain)[picked]
        if self.has_rupture_speed():
            rupture_speed = float(self.rupture_speed)
            self.data_manager.set_data(f"{self.event_path}/rupture_speed", rupture_speed, True)
            self.event["rupture_speed"] = rupture_speed
        else:
            # A degenerate fit has no speed; do not leave a nan (or a stale value
            # from an earlier save) for the CZM fitter to pick up as Cf.
            self.event.pop("rupture_speed", None)
            print(f"No rupture speed saved for runs[{self.run_idx}]/events[{self.event_idx}]: "
                  "the arrival picks do not define a line (Cf = n/a).")
        self.app.refresh_tree()
        print(f"Saved runs[{self.run_idx}]/events[{self.event_idx}] to data.")

    # --------------------------------------------------------------- menus
    @staticmethod
    def _fill_channel_menu(menubutton, flags, command):
        menubutton.menu.delete(0, "end")
        menubutton.items = [tk.IntVar(value=int(bool(flag))) for flag in flags]
        for i, var in enumerate(menubutton.items):
            menubutton.menu.add_checkbutton(label="channel %d" % i, variable=var, command=command)

    def init_enabled_channels_mb(self):
        self._fill_channel_menu(self.enabled_channels_mb, self.enabled_channels, self.enabled_channels_changed)

    def init_fitting_channels_mb(self):
        self._fill_channel_menu(self.fitting_channels_mb, self.fitting_channels, self.fitting_channels_changed)

    def enabled_channels_changed(self):
        self.enabled_channels = [bool(var.get()) for var in self.enabled_channels_mb.items]
        self.plot()
        self.update_fitted_line()

    def fitting_channels_changed(self):
        self.fitting_channels = [bool(var.get()) for var in self.fitting_channels_mb.items]
        self.draw_markers()
        self.update_fitted_line()

    # -------------------------------------------------------------- filter
    def toggle_filter(self):
        self.filtering = not self.filtering
        if self.filtering:
            self.filter_toggle.config(text="Filter On", relief="sunken")
        else:
            self.filter_toggle.config(text="Filter Off", relief="raised")
        self.plot()
        self.update_fitted_line()

    def on_filter_window_length_box_changed(self, event=None):
        self.plot()
        self.update_fitted_line()

    # --------------------------------------------------------------- magic
    def magic(self):
        """Auto-pick: the extreme of each enabled (plotted) trace."""
        for i, line in enumerate(self.lines):
            if line is None:
                continue
            x, y = line.get_data()
            self.picked_idx[i] = int(np.argmin(y))
        self.draw_markers()
        self.update_fitted_line()

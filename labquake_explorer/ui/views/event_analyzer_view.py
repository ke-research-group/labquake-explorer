import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.patches as patches
import numpy as np

from labquake_explorer.analysis.event_metrics import (
    EventPicks, analyze_event, picks_from_result, picks_from_windows, windows_from_result,
)
from labquake_explorer.analysis.fitting import METHODS
from labquake_explorer.ui.actions import register_view
from labquake_explorer.ui.context import EVENT
from labquake_explorer.ui.views.base import EventView

FIT_METHOD_LABELS = {"ols": "OLS", "theilsen": "Theil-Sen"}
MARKER_COLORS = ['#33CCC4', '#33CCC4', '#CC3366', '#CC3366',
                 '#33CC66', '#33CC66', '#9966CC', '#9966CC']


@register_view("Analyze Event", kinds=[EVENT], order=10)
class EventAnalyzerView(EventView):
    """Pick ranges on one event and compute its mechanical metrics.

    Eight draggable markers snapped to samples:
      0,1  loading range    -> loading stiffness dY/dX; also the pre-event trend Y(t)
      2,3  unloading range  -> unloading stiffness dY/dX
      4,5  rupture start/end -> stress drop Y4-Y5 and slip X5-X4
      6,7  post-event range -> post-event trend Y(t)
    The trend-extrapolated drop is pre-trend(t_event) - post-trend(t_event).
    Numbers come from :func:`labquake_explorer.analysis.event_metrics.analyze_event`
    and are saved under ``event['event_analysis']`` (schema version 2).
    """

    window_title = "Event Analyzer"
    result_key = "event_analysis"

    def __init__(self, app, run_idx, event_idx, item_y="shear_stress", item_x="displacement"):
        self.item_y = item_y
        self.item_x = item_x
        self.data_x = np.array([])
        self.data_y = np.array([])
        self.data_t = np.array([])
        self.markers = []
        self.current_artist = None
        self.currently_dragging = False
        self.offset = [0, 0]
        self.picked_idx = []
        self.loading_line = None
        self.rupture_line = None
        self.post_line = None
        self.rupture_span = None
        self.result = None
        super().__init__(app, run_idx, event_idx)

    # ------------------------------------------------------------------ ui
    def build_ui(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=0)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(2, weight=1)

        event_selection_frame = ttk.LabelFrame(self, text="Event Selection")
        event_selection_frame.grid(row=0, column=0, rowspan=2, padx=5, pady=5, sticky="nsw")
        self.build_event_selector(event_selection_frame, row=0, column=0)
        self.save_button = ttk.Button(event_selection_frame, text="Save Event", command=self.save_event, width=15)
        self.save_button.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        self.apply_all_button = ttk.Button(event_selection_frame, text="Apply to All Events",
                                           command=self.apply_to_all_events, width=18)
        self.apply_all_button.grid(row=2, column=0, columnspan=2, padx=5, pady=2, sticky="ew")

        data_frame = ttk.LabelFrame(self, text="Data Fields")
        data_frame.grid(row=0, column=1, rowspan=2, columnspan=2, padx=5, pady=5, sticky="nsew")
        tk.Label(data_frame, text="X Data:", width=8, anchor="e").grid(row=0, column=0, padx=5, pady=3, sticky="e")
        self.data_x_combo = ttk.Combobox(data_frame, state="readonly", width=20)
        self.data_x_combo.grid(row=0, column=1, padx=5, pady=3, sticky="w")
        tk.Label(data_frame, text="Y Data:", width=8, anchor="e").grid(row=1, column=0, padx=5, pady=3, sticky="e")
        self.data_y_combo = ttk.Combobox(data_frame, state="readonly", width=20)
        self.data_y_combo.grid(row=1, column=1, padx=5, pady=3, sticky="w")
        tk.Label(data_frame, text="Fit:", width=8, anchor="e").grid(row=2, column=0, padx=5, pady=3, sticky="e")
        self.fit_combo = ttk.Combobox(data_frame, state="readonly", width=20,
                                      values=[FIT_METHOD_LABELS[m] for m in METHODS])
        self.fit_combo.current(0)
        self.fit_combo.grid(row=2, column=1, padx=5, pady=3, sticky="w")
        self.data_y_combo.bind("<<ComboboxSelected>>", self.data_selected)
        self.data_x_combo.bind("<<ComboboxSelected>>", self.data_selected)
        self.fit_combo.bind("<<ComboboxSelected>>", lambda e: self.update_analysis())

        results_frame = ttk.LabelFrame(self, text="Analysis Results")
        results_frame.grid(row=0, column=3, rowspan=2, columnspan=1, padx=5, pady=5, sticky="nsew")
        self.result_entries = {}
        layout = [
            ("Loading Stiffness:", "loading_stiffness", 0, 0),
            ("Unloading Stiffness:", "unloading_stiffness", 1, 0),
            ("Loading R²:", "loading_r2", 2, 0),
            ("Stress Drop:", "stress_drop", 0, 2),
            ("Displacement:", "displacement", 1, 2),
            ("Stress Drop (trend):", "stress_drop_trend", 2, 2),
            ("Displacement (trend):", "displacement_trend", 3, 2),
        ]
        for text, key, row, col in layout:
            tk.Label(results_frame, text=text, anchor="e", width=19).grid(row=row, column=col, padx=5, pady=2, sticky="e")
            entry = tk.Entry(results_frame, state="readonly", width=12, justify="right")
            entry.grid(row=row, column=col + 1, padx=5, pady=2, sticky="w")
            self.result_entries[key] = entry
        # legacy attribute names used by older code/tests
        self.loading_slope_text = self.result_entries["loading_stiffness"]
        self.rupture_slope_text = self.result_entries["unloading_stiffness"]
        self.stress_drop_text = self.result_entries["stress_drop"]
        self.displacement_text = self.result_entries["displacement"]

        self.make_figure(figsize=(10, 6), row=2, column=0, columnspan=4, padx=5, pady=5, sticky="nsew")
        self.figure.set_facecolor('#f5f5f5')
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#ffffff')

        self.figure.canvas.mpl_connect('pick_event', self.on_pick)
        self.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.figure.canvas.mpl_connect('resize_event', self.on_resize)

    def on_event_loaded(self):
        self.init_comboboxes()
        saved = self.load_results()
        picks = picks_from_result(saved, len(self.data_y)) if saved else None
        if picks is None:
            picks = EventPicks.defaults(len(self.data_y))
        if saved and saved.get("fit_method") in METHODS:
            self.fit_combo.set(FIT_METHOD_LABELS[saved["fit_method"]])
        self.picked_idx = picks.to_list()
        self.plot_picked_points()

    @property
    def fit_method(self) -> str:
        label = self.fit_combo.get()
        for key, value in FIT_METHOD_LABELS.items():
            if value == label:
                return key
        return "ols"

    def _set_default_point_positions(self):
        self.picked_idx = EventPicks.defaults(len(self.data_y)).to_list()

    def init_comboboxes(self):
        """Initialize comboboxes with event fields of the same length as time"""
        if 'time' in self.event:
            time_length = len(self.event['time'])
        else:
            print("Warning: 'time' field not found in event data")
            return
        
        def find_matching_arrays(data, path=""):
            matching_fields = []
            if isinstance(data, dict):
                for key, value in data.items():
                    new_path = f"{path}/{key}" if path else key
                    if isinstance(value, (list, np.ndarray)) and len(value) == time_length:
                        matching_fields.append(new_path)
                    elif isinstance(value, dict):
                        matching_fields.extend(find_matching_arrays(value, new_path))
            return matching_fields
        
        matching_fields = find_matching_arrays(self.event)
        matching_fields.sort()
        self.data_x_combo.config(values=matching_fields)
        self.data_y_combo.config(values=matching_fields)
        
        if not (self.item_x and self.item_x in matching_fields):
            displacement_fields = [f for f in matching_fields if 'displacement' in f.lower()]
            if displacement_fields:
                self.item_x = displacement_fields[0]
            elif 'time' in matching_fields:
                self.item_x = 'time'
            elif matching_fields:
                self.item_x = matching_fields[0]
        self.data_x_combo.set(self.item_x or "")
        
        if not (self.item_y and self.item_y in matching_fields):
            stress_fields = [f for f in matching_fields if 'shear_stress' in f.lower()]
            other_stress = [f for f in matching_fields if 'stress' in f.lower()]
            if stress_fields:
                self.item_y = stress_fields[0]
            elif other_stress:
                self.item_y = other_stress[0]
            elif matching_fields:
                self.item_y = matching_fields[0]
        self.data_y_combo.set(self.item_y or "")
        
        self.plot_data()
    
    def data_selected(self, event=None):
        """Handle data selection from comboboxes"""
        self.item_y = self.data_y_combo.get()
        self.item_x = self.data_x_combo.get()
        self.plot_data()
        n = len(self.data_y)
        if not (n > 0 and len(self.picked_idx) == 8 and max(self.picked_idx) < n):
            self._set_default_point_positions()
        self.plot_picked_points()

    def get_field(self, path):
        """Access a nested event field using 'a/b/c' notation."""
        current = self.event
        for part in path.split('/'):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                print(f"Warning: Path '{path}' not found in data")
                return None
        return current

    def plot_data(self):
        """Plot the selected data"""
        if not self.item_y:
            return
        self.ax.clear()
        data_y = self.get_field(self.item_y)
        if data_y is None:
            return
        self.data_y = np.asarray(data_y, dtype=float)
        if self.item_x:
            data_x = self.get_field(self.item_x)
            if data_x is None:
                return
            self.data_x = np.asarray(data_x, dtype=float)
            self.ax.set_xlabel(self.item_x)
        else:
            self.data_x = np.arange(len(self.data_y), dtype=float)
            self.ax.set_xlabel("Index")
        time = self.event.get('time')
        if time is not None and len(time) == len(self.data_y):
            self.data_t = np.asarray(time, dtype=float)
        else:
            self.data_t = np.arange(len(self.data_y), dtype=float)
        self.ax.plot(self.data_x, self.data_y, zorder=-100, linewidth=1.5)
        self.ax.set_ylabel(self.item_y)
        self.ax.grid(True, linestyle='--', alpha=0.3)
        self.ax.set_title(f"{self.figure_title()}: {self.item_y} vs {self.item_x if self.item_x else 'Index'}")
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.tick_params(direction='out')
        self.canvas.draw()
    
    def plot_picked_points(self):
        """Plot the marker points and connecting lines"""
        if len(self.picked_idx) != 8 or len(self.data_y) == 0:
            return
        width, height = self.get_circle_dims()
        for marker in self.markers:
            if marker in self.ax.patches:
                marker.remove()
        self.markers = []
        for line in (self.loading_line, self.rupture_line, self.post_line):
            if line is not None and line in self.ax.lines:
                line.remove()
        self._remove_span()

        if max(self.picked_idx) >= len(self.data_y):
            self._set_default_point_positions()
                    
        for i, idx in enumerate(self.picked_idx):
            marker = patches.Ellipse((self.data_x[idx], self.data_y[idx]), width=width, height=height, 
                                     color=MARKER_COLORS[i], fill=False, lw=2, picker=8, label=str(i))
            self.ax.add_patch(marker)
            self.markers.append(marker)
        
        self.loading_line, = self.ax.plot(*self._segment(0, 1), '--', color=MARKER_COLORS[0],
                                          linewidth=2, zorder=-50, label='Loading')
        self.rupture_line, = self.ax.plot(*self._segment(2, 3), '--', color=MARKER_COLORS[2],
                                          linewidth=2, zorder=-50, label='Unloading')
        self.post_line, = self.ax.plot(*self._segment(6, 7), '--', color=MARKER_COLORS[6],
                                       linewidth=2, zorder=-50, label='Post-event')
        self.rupture_span = self.ax.axvspan(self.data_x[self.picked_idx[4]], self.data_x[self.picked_idx[5]],
                                            alpha=0.15, color=MARKER_COLORS[4], zorder=-100)
        self.canvas.draw()
        self.update_analysis()

    def _segment(self, i, j):
        return ([self.data_x[self.picked_idx[i]], self.data_x[self.picked_idx[j]]],
                [self.data_y[self.picked_idx[i]], self.data_y[self.picked_idx[j]]])

    def _remove_span(self):
        if self.rupture_span:
            try:
                self.rupture_span.remove()
            except Exception:
                pass
            self.rupture_span = None
    
    def get_circle_dims(self):
        """Calculate appropriate dimensions for marker circles based on plot scaling"""
        self.canvas.draw()
        xl = self.ax.get_xlim()
        yl = self.ax.get_ylim()
        ratio = (yl[-1] - yl[0]) / (xl[-1] - xl[0])
        fig_size = self.figure.get_size_inches()
        ratio *= fig_size[0] / fig_size[1]
        width = (xl[-1] - xl[0]) / fig_size[0] * 0.15
        return width, width * ratio

    # ------------------------------------------------------------ analysis
    def update_analysis(self):
        """Recompute all metrics from the current picks and show them."""
        if len(self.picked_idx) != 8 or len(self.data_y) == 0:
            return
        picks = EventPicks.from_list(self.picked_idx)
        event_time = self.event.get('event_time', self.data_t[0] if len(self.data_t) else 0.0)
        try:
            self.result = analyze_event(self.data_t, self.data_x, self.data_y, float(event_time), picks,
                                        method=self.fit_method, x_field=self.item_x or "",
                                        y_field=self.item_y or "")
        except ValueError as e:
            print(f"Error calculating values: {e}")
            return
        r = self.result
        values = {
            "loading_stiffness": r["loading_stiffness"],
            "unloading_stiffness": r["unloading_stiffness"],
            "loading_r2": r["loading_fit"]["r2"],
            "stress_drop": r["stress_drop"],
            "displacement": r["displacement"],
            "stress_drop_trend": r["stress_drop_trend"],
            "displacement_trend": r["displacement_trend"],
        }
        for key, value in values.items():
            self.set_textbox(self.result_entries[key], self.format_value(value))

    @staticmethod
    def format_value(value) -> str:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "n/a"
        return "n/a" if not np.isfinite(value) else f"{value:.6g}"
    
    def set_textbox(self, textbox, text):
        """Helper method to set text in a readonly textbox"""
        textbox.config(state="normal")
        textbox.delete(0, tk.END)
        textbox.insert(0, text)
        textbox.config(state="readonly")
    
    # ------------------------------------------------------------- dragging
    def on_pick(self, event):
        if self.toolbar_active():
            return
        if self.current_artist is None and isinstance(event.artist, patches.Ellipse):
            self.current_artist = event.artist
            x0, y0 = self.current_artist.center
            self.offset = [(x0 - event.mouseevent.xdata), (y0 - event.mouseevent.ydata)]
    
    def on_press(self, event):
        self.currently_dragging = True
    
    def on_release(self, event):
        self.current_artist = None
        self.currently_dragging = False
        self.on_resize(None)

    def move_point(self, point_idx: int, idx: int) -> None:
        """Move marker ``point_idx`` to sample ``idx`` and refresh lines and results."""
        idx = int(min(max(idx, 0), len(self.data_y) - 1))
        self.picked_idx[point_idx] = idx
        if point_idx < len(self.markers):
            self.markers[point_idx].set_center((self.data_x[idx], self.data_y[idx]))
        if point_idx in (0, 1):
            self.loading_line.set_data(*self._segment(0, 1))
        elif point_idx in (2, 3):
            self.rupture_line.set_data(*self._segment(2, 3))
        elif point_idx in (4, 5):
            self._remove_span()
            self.rupture_span = self.ax.axvspan(self.data_x[self.picked_idx[4]], self.data_x[self.picked_idx[5]],
                                                alpha=0.15, color=MARKER_COLORS[4], zorder=-100)
        elif point_idx in (6, 7):
            self.post_line.set_data(*self._segment(6, 7))
        self.update_analysis()
        self.canvas.draw_idle()
    
    def on_motion(self, event):
        """Handle mouse motion events for dragging markers"""
        if not self.currently_dragging or self.current_artist is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        if not isinstance(self.current_artist, patches.Ellipse):
            return
        try:
            dx, dy = self.offset
            cx, cy = event.xdata + dx, event.ydata + dy
            xl = self.ax.get_xlim()
            yl = self.ax.get_ylim()
            yw = yl[-1] - yl[0]
            xw = xl[-1] - xl[0]
            distances = ((self.data_x - cx) / xw) ** 2 + ((self.data_y - cy) / yw) ** 2
            idx = int(np.argmin(distances))
            self.move_point(int(self.current_artist.get_label()), idx)
        except Exception as e:
            print(f"Error in on_motion: {e}")
    
    def on_resize(self, event):
        """Handle window resize events to adjust marker sizes"""
        if hasattr(self, 'ax') and self.markers:
            width, height = self.get_circle_dims()
            for marker in self.markers:
                marker.set_width(width)
                marker.set_height(height)
            self.canvas.draw()

    # ---------------------------------------------------------------- save
    def apply_to_all_events(self, confirm=True):
        """Apply the current relative-time windows to every event of the run and save.

        Returns the number of events written.  Events whose fields are missing
        are skipped.
        """
        if self.result is None:
            self.update_analysis()
        windows = windows_from_result(self.result or {})
        if windows is None:
            return 0
        n_events = self.n_events()
        if confirm and not messagebox.askokcancel(
                "Apply to all events",
                f"Recompute and overwrite event_analysis for all {n_events} events of run {self.run_idx} "
                f"using the current windows?", icon=messagebox.WARNING):
            return 0
        written = 0
        for j in range(n_events):
            event = self.data_manager.get_data(f"{self.run_path}/events/[{j}]")
            try:
                t = np.asarray(event['time'], dtype=float)
                x = np.asarray(self.get_field_of(event, self.item_x), dtype=float)
                y = np.asarray(self.get_field_of(event, self.item_y), dtype=float)
                event_time = float(event['event_time'])
            except (KeyError, TypeError, ValueError):
                continue
            if not (t.size and t.size == x.size == y.size):
                continue
            picks = picks_from_windows(t - event_time, windows["loading"], windows["unloading"],
                                       windows["rupture"], windows["post"])
            result = analyze_event(t, x, y, event_time, picks, method=self.fit_method,
                                   x_field=self.item_x or "", y_field=self.item_y or "")
            self.data_manager.set_data(f"{self.run_path}/events/[{j}]/event_analysis", result, True)
            event['event_analysis'] = result
            written += 1
        self.app.refresh_tree()
        self.set_event(self.event_idx)
        return written

    @staticmethod
    def get_field_of(event, path):
        current = event
        for part in (path or "").split('/'):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                raise KeyError(path)
        return current

    def save_event(self):
        """Save the analysis results to the data manager"""
        if self.result is None:
            self.update_analysis()
        if self.result is None:
            return
        try:
            self.save_results(dict(self.result))
        except Exception as e:
            messagebox.showerror("Save Failed", f"Error saving results: {e}")

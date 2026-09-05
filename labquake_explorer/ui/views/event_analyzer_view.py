import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.patches as patches
import numpy as np
from scipy import stats

from labquake_explorer.ui.actions import register_view
from labquake_explorer.ui.context import EVENT
from labquake_explorer.ui.views.base import EventView


@register_view("Analyze Event", kinds=[EVENT], order=10)
class EventAnalyzerView(EventView):
    """Pick loading/unloading ranges and rupture start/end on one event.

    Six draggable markers snapped to samples:
      0,1  loading range   -> loading stiffness (linear regression of Y on X)
      2,3  unloading range -> unloading stiffness
      4,5  rupture start/end -> stress drop and displacement
    Results are saved under ``event['event_analysis']``.
    """

    window_title = "Event Analyzer"
    result_key = "event_analysis"

    def __init__(self, app, run_idx, event_idx, item_y="shear_stress", item_x="displacement"):
        self.item_y = item_y
        self.item_x = item_x
        self.data_x = []
        self.data_y = []
        self.markers = []
        self.current_artist = None
        self.currently_dragging = False
        self.offset = [0, 0]
        self.picked_idx = []
        self.loading_line = None
        self.rupture_line = None
        self.rupture_span = None
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

        data_frame = ttk.LabelFrame(self, text="Data Fields")
        data_frame.grid(row=0, column=1, rowspan=2, columnspan=2, padx=5, pady=5, sticky="nsew")
        tk.Label(data_frame, text="X Data:", width=8, anchor="e").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.data_x_combo = ttk.Combobox(data_frame, state="readonly", width=20)
        self.data_x_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        tk.Label(data_frame, text="Y Data:", width=8, anchor="e").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.data_y_combo = ttk.Combobox(data_frame, state="readonly", width=20)
        self.data_y_combo.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.data_y_combo.bind("<<ComboboxSelected>>", self.data_selected)
        self.data_x_combo.bind("<<ComboboxSelected>>", self.data_selected)

        results_frame = ttk.LabelFrame(self, text="Analysis Results")
        results_frame.grid(row=0, column=3, rowspan=2, columnspan=1, padx=5, pady=5, sticky="nsew")
        tk.Label(results_frame, text="Loading Stiffness:", anchor="e", width=18).grid(row=0, column=0, padx=5, pady=2, sticky="e")
        self.loading_slope_text = tk.Entry(results_frame, state="readonly", width=12, justify="right")
        self.loading_slope_text.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        tk.Label(results_frame, text="Unloading Stiffness:", anchor="e", width=18).grid(row=1, column=0, padx=5, pady=2, sticky="e")
        self.rupture_slope_text = tk.Entry(results_frame, state="readonly", width=12, justify="right")
        self.rupture_slope_text.grid(row=1, column=1, padx=5, pady=2, sticky="w")
        tk.Label(results_frame, text="Stress Drop:", anchor="e", width=12).grid(row=0, column=2, padx=5, pady=2, sticky="e")
        self.stress_drop_text = tk.Entry(results_frame, state="readonly", width=12, justify="right")
        self.stress_drop_text.grid(row=0, column=3, padx=5, pady=2, sticky="w")
        tk.Label(results_frame, text="Displacement:", anchor="e", width=12).grid(row=1, column=2, padx=5, pady=2, sticky="e")
        self.displacement_text = tk.Entry(results_frame, state="readonly", width=12, justify="right")
        self.displacement_text.grid(row=1, column=3, padx=5, pady=2, sticky="w")

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
        if saved and all(k in saved for k in ('loading_indices', 'unloading_indices',
                                               'rupture_start_index', 'rupture_end_index')):
            self.picked_idx = [
                saved['loading_indices'][0], saved['loading_indices'][1],
                saved['unloading_indices'][0], saved['unloading_indices'][1],
                saved['rupture_start_index'], saved['rupture_end_index'],
            ]
        else:
            self._set_default_point_positions()
        self.plot_picked_points()

    def _set_default_point_positions(self):
        """Helper method to set default point positions"""
        n = len(self.data_y)
        self.picked_idx = [
            int(n * 0.25),     # Loading slope start
            int(n * 0.35),    # Loading slope end
            int(n * 0.5),     # Rupture slope start
            int(n * 0.6),     # Rupture slope end
            int(n * 0.4),     # Rupture start
            int(n * 0.7)      # Rupture end
        ]

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
        
        if self.item_x and self.item_x in matching_fields:
            self.data_x_combo.set(self.item_x)
        else:
            displacement_fields = [f for f in matching_fields if 'displacement' in f.lower()]
            if displacement_fields:
                self.item_x = displacement_fields[0]
            elif 'time' in matching_fields:
                self.item_x = 'time'
            elif matching_fields:
                self.item_x = matching_fields[0]
            self.data_x_combo.set(self.item_x)
        
        if self.item_y and self.item_y in matching_fields:
            self.data_y_combo.set(self.item_y)
        else:
            stress_fields = [f for f in matching_fields if 'shear_stress' in f.lower()]
            other_stress = [f for f in matching_fields if 'stress' in f.lower()]
            if stress_fields:
                self.item_y = stress_fields[0]
            elif other_stress:
                self.item_y = other_stress[0]
            elif matching_fields:
                self.item_y = matching_fields[0]
            self.data_y_combo.set(self.item_y)
        
        self.plot_data()
    
    def data_selected(self, event=None):
        """Handle data selection from comboboxes"""
        self.item_y = self.data_y_combo.get()
        self.item_x = self.data_x_combo.get()
        self.plot_data()
        if len(self.data_y) > 0 and len(self.picked_idx) == 6:
            max_idx = len(self.data_y) - 1
            if any(idx > max_idx for idx in self.picked_idx):
                self._set_default_point_positions()
        else:
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
        if self.item_y is None:
            return
        self.ax.clear()
        if self.item_x is None:
            self.data_y = self.get_field(self.item_y)
            if self.data_y is None:
                return
            self.data_x = np.arange(len(self.data_y))
            self.ax.set_xlabel("Index")
        else:
            self.data_x = self.get_field(self.item_x)
            self.data_y = self.get_field(self.item_y)
            if self.data_x is None or self.data_y is None:
                return
            self.ax.set_xlabel(self.item_x)
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
        if not self.picked_idx or len(self.picked_idx) != 6 or len(self.data_y) == 0:
            return
        width, height = self.get_circle_dims()
        for marker in self.markers:
            if marker in self.ax.patches:
                marker.remove()
        self.markers = []
        if self.loading_line and self.loading_line in self.ax.lines:
            self.loading_line.remove()
        if self.rupture_line and self.rupture_line in self.ax.lines:
            self.rupture_line.remove()
        self._remove_span()

        max_idx = len(self.data_y) - 1
        if any(idx > max_idx for idx in self.picked_idx):
            self._set_default_point_positions()
                    
        colors = ['#33CCC4', '#33CCC4', '#CC3366', '#CC3366', '#33CC66', '#33CC66']
        for i, idx in enumerate(self.picked_idx):
            if idx >= len(self.data_x) or idx >= len(self.data_y):
                continue
            marker = patches.Ellipse((self.data_x[idx], self.data_y[idx]), width=width, height=height, 
                                     color=colors[i], fill=False, lw=2, picker=8, label=str(i))
            self.ax.add_patch(marker)
            self.markers.append(marker)
        
        self.loading_line, = self.ax.plot(*self._segment(0, 1), '--', color='#33CCC4',
                                          linewidth=2, zorder=-50, label='Loading Stiffness')
        self.rupture_line, = self.ax.plot(*self._segment(2, 3), '--', color='#CC3366',
                                          linewidth=2, zorder=-50, label='Unloading Stiffness')
        self.rupture_span = self.ax.axvspan(self.data_x[self.picked_idx[4]], self.data_x[self.picked_idx[5]],
                                            alpha=0.15, color='#33CC66', zorder=-100)
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

    def update_analysis(self):
        """Update all calculated values using linear regression for slope calculations"""
        if len(self.picked_idx) != 6:
            return
        try:
            loading_indices = range(min(self.picked_idx[0], self.picked_idx[1]), max(self.picked_idx[0], self.picked_idx[1]) + 1)
            rupture_indices = range(min(self.picked_idx[2], self.picked_idx[3]), max(self.picked_idx[2], self.picked_idx[3]) + 1)
            x_loading = [self.data_x[i] for i in loading_indices]
            y_loading = [self.data_y[i] for i in loading_indices]
            x_rupture = [self.data_x[i] for i in rupture_indices]
            y_rupture = [self.data_y[i] for i in rupture_indices]
            slope_loading = stats.linregress(x_loading, y_loading).slope
            slope_rupture = stats.linregress(x_rupture, y_rupture).slope
            x4, y4 = self.data_x[self.picked_idx[4]], self.data_y[self.picked_idx[4]]
            x5, y5 = self.data_x[self.picked_idx[5]], self.data_y[self.picked_idx[5]]
            stress_drop = abs(y4 - y5)
            displacement = abs(x5 - x4)
            self.set_textbox(self.loading_slope_text, f"{slope_loading:.6g}")
            self.set_textbox(self.rupture_slope_text, f"{slope_rupture:.6g}")
            self.set_textbox(self.stress_drop_text, f"{stress_drop:.6g}")
            self.set_textbox(self.displacement_text, f"{displacement:.6g}")
        except (IndexError, ZeroDivisionError, ValueError) as e:
            print(f"Error calculating values: {e}")
    
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
            self.current_artist.set_center((self.data_x[idx], self.data_y[idx]))
            point_idx = int(self.current_artist.get_label())
            self.picked_idx[point_idx] = idx
            if point_idx in (0, 1):
                self.loading_line.set_data(*self._segment(0, 1))
            elif point_idx in (2, 3):
                self.rupture_line.set_data(*self._segment(2, 3))
            elif point_idx in (4, 5):
                self._remove_span()
                self.rupture_span = self.ax.axvspan(self.data_x[self.picked_idx[4]], self.data_x[self.picked_idx[5]],
                                                    alpha=0.15, color='#33CC66', zorder=-100)
            self.update_analysis()
            self.canvas.draw_idle()
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
    def collect_results(self):
        return {
            'loading_indices': [int(self.picked_idx[0]), int(self.picked_idx[1])],
            'unloading_indices': [int(self.picked_idx[2]), int(self.picked_idx[3])],
            'rupture_start_index': int(self.picked_idx[4]),
            'rupture_end_index': int(self.picked_idx[5]),
            'loading_stiffness': float(self.loading_slope_text.get()),
            'unloading_stiffness': float(self.rupture_slope_text.get()),
            'stress_drop': float(self.stress_drop_text.get()),
            'displacement': float(self.displacement_text.get()),
        }

    def save_event(self):
        """Save the analysis results to the data manager"""
        if len(self.picked_idx) != 6:
            return
        try:
            self.save_results(self.collect_results())
        except Exception as e:
            messagebox.showerror("Save Failed", f"Error saving results: {e}")

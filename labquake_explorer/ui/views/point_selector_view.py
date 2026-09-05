"""Pick points on a given curve with draggable markers.

Used by the main window for "Pick Events" (editable, with a save callback)
and "Min/Max" (two fixed markers).  Markers snap to samples; with
``add_remove_enabled`` a left double-click on the curve adds a marker and a
right double-click on a marker removes it.

The view never shares its pick list: the constructor copies ``picked_idx`` and
``save`` hands the callback a fresh sorted copy, so stored data only changes
when the user presses Save (edits made afterwards stay local until the next
Save).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional, Sequence

import matplotlib.patches as patches
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from labquake_explorer.ui.views.base import BaseView


class PointsSelectorView(BaseView):
    window_title = "Points Selector"

    def __init__(self, app, x, y, picked_idx: Sequence[int], add_remove_enabled: bool = False,
                 callback: Optional[Callable] = None, xlabel=None, ylabel=None, title=None):
        self.x_values = np.asarray(x)
        self.y_values = np.asarray(y)
        self.picked_idx: list[int] = [int(i) for i in picked_idx]
        self.add_remove_enabled = add_remove_enabled
        self.callback = callback
        self.markers: list[patches.Ellipse] = []
        self.offset = [0.0, 0.0]
        self.current_artist = None
        self.currently_dragging = False
        super().__init__(app)

        if callback:
            self.save_button = tk.Button(self, text="Save", command=self.save)
            self.save_button.pack(side=tk.TOP, padx=5)

        self.figure = Figure()
        self.fig = self.figure  # legacy alias
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()

        toolbar_frame = ttk.Frame(self)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        self.canvas_widget.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)

        self.ax.plot(self.x_values, self.y_values, ".-", color="C0", zorder=-100)
        if xlabel:
            self.ax.set_xlabel(xlabel)
        if ylabel:
            self.ax.set_ylabel(ylabel)
        if title:
            self.ax.set_title(title)
        self.plot_data_points()

        self.canvas.mpl_connect("pick_event", self.on_pick)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("resize_event", self.on_resize)
        self.canvas.mpl_connect("scroll_event", self.on_resize)

    # ------------------------------------------------------------- markers
    def _make_marker(self, idx: int, label: int, width: float, height: float) -> patches.Ellipse:
        marker = patches.Ellipse((self.x_values[idx], self.y_values[idx]), width=width, height=height,
                                 color="red", fill=False, lw=2, picker=8, label=str(label))
        self.ax.add_patch(marker)
        self.markers.append(marker)
        return marker

    def plot_data_points(self):
        for marker in self.markers:
            if marker in self.ax.patches:
                marker.remove()
        self.markers = []
        width, height = self.get_circle_dims()
        for i, idx in enumerate(self.picked_idx):
            self._make_marker(idx, i, width, height)
        self.canvas.draw()

    def renumber_markers(self):
        """Keep marker label == position in ``picked_idx``."""
        for i, marker in enumerate(self.markers):
            marker.set_label(str(i))

    def add_point(self, idx: int) -> int:
        """Append sample ``idx`` as a new pick; returns its position in ``picked_idx``."""
        idx = int(idx)
        width, height = self.get_circle_dims()
        self._make_marker(idx, len(self.picked_idx), width, height)
        self.picked_idx.append(idx)
        self.canvas.draw()
        return len(self.picked_idx) - 1

    def remove_point(self, i: int) -> None:
        """Remove the pick at position ``i`` and renumber the remaining markers."""
        marker = self.markers.pop(i)
        marker.remove()
        del self.picked_idx[i]
        if self.current_artist is marker:
            self.current_artist = None
        self.renumber_markers()
        self.canvas.draw()

    def nearest_index(self, cx: float, cy: float) -> int:
        xl = self.ax.get_xlim()
        yl = self.ax.get_ylim()
        xw = xl[-1] - xl[0]
        yw = yl[-1] - yl[0]
        return int(np.argmin(((self.x_values - cx) / xw) ** 2 + ((self.y_values - cy) / yw) ** 2))

    # ------------------------------------------------------------- dragging
    def on_pick(self, event):
        if self.toolbar_active():
            return
        if self.current_artist is not None or not isinstance(event.artist, patches.Ellipse):
            return
        self.current_artist = event.artist
        mouse = event.mouseevent
        if mouse.dblclick:
            if self.add_remove_enabled and mouse.button != MouseButton.LEFT:
                self.remove_point(int(self.current_artist.get_label()))
                self.current_artist = None
        else:
            x0, y0 = self.current_artist.center
            self.offset = [x0 - mouse.xdata, y0 - mouse.ydata]

    def on_motion(self, event):
        if not self.currently_dragging or self.current_artist is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        if not isinstance(self.current_artist, patches.Ellipse):
            return
        try:
            dx, dy = self.offset
            idx = self.nearest_index(event.xdata + dx, event.ydata + dy)
            self.current_artist.set_center((self.x_values[idx], self.y_values[idx]))
            self.picked_idx[int(self.current_artist.get_label())] = idx
            self.canvas.draw_idle()
        except Exception as e:
            print(f"Error in on_motion: {e}")

    def on_press(self, event):
        self.currently_dragging = True
        if self.toolbar_active():
            return
        # The figure dispatches pick events before this handler, so a double
        # click on an existing marker has already set current_artist.
        if (event.button == MouseButton.LEFT and event.dblclick and self.add_remove_enabled
                and self.current_artist is None and event.xdata is not None and event.ydata is not None):
            self.add_point(self.nearest_index(event.xdata, event.ydata))

    def on_release(self, event):
        self.current_artist = None
        self.currently_dragging = False
        self.on_resize(None)

    def get_circle_dims(self):
        self.canvas.draw()
        xl = self.ax.get_xlim()
        yl = self.ax.get_ylim()
        ratio = (yl[-1] - yl[0]) / (xl[-1] - xl[0])
        fig_size = self.figure.get_size_inches()
        ratio *= fig_size[0] / fig_size[1]
        width = (xl[-1] - xl[0]) / fig_size[0] * 0.15
        return width, width * ratio

    def on_resize(self, event):
        if getattr(self, "ax", None) is None or not self.markers:
            return
        width, height = self.get_circle_dims()
        for marker in self.markers:
            marker.set_width(width)
            marker.set_height(height)
        self.canvas.draw()

    # ----------------------------------------------------------------- save
    def save(self):
        """Sort the picks (markers follow) and pass a *copy* to the callback.

        The callback typically stores the list verbatim (DataManager.set_data
        does not copy), so handing over ``self.picked_idx`` itself would alias
        the stored data with the live list and let later drags/adds/removes in
        this window edit the stored data without another Save.
        """
        order = sorted(range(len(self.picked_idx)), key=self.picked_idx.__getitem__)
        self.picked_idx.sort()
        if len(self.markers) == len(order):
            self.markers = [self.markers[j] for j in order]
        self.renumber_markers()
        if self.callback:
            self.callback(list(self.picked_idx))

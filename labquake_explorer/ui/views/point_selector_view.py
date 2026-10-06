"""Pick points on a given curve with draggable markers.

Used by the main window for "Min/Max" (two fixed markers) and available to
any caller that wants editable picks with a save callback.  Markers snap to
samples; with ``add_remove_enabled`` a left double-click on the curve adds a
marker and a right double-click on a marker removes it (see
:class:`~labquake_explorer.ui.views.point_picker.PointPicker`).

The view never shares its pick list: the constructor copies ``picked_idx`` and
``save`` hands the callback a fresh sorted copy, so stored data only changes
when the user presses Save (edits made afterwards stay local until the next
Save).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional, Sequence

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from labquake_explorer.ui.views.base import BaseView
from labquake_explorer.ui.views.point_picker import PointPicker


class PointsSelectorView(BaseView):
    window_title = "Points Selector"

    def __init__(self, app, x, y, picked_idx: Sequence[int], add_remove_enabled: bool = False,
                 callback: Optional[Callable] = None, xlabel=None, ylabel=None, title=None):
        self.x_values = np.asarray(x)
        self.y_values = np.asarray(y)
        self.add_remove_enabled = add_remove_enabled
        self.callback = callback
        super().__init__(app)

        if add_remove_enabled:
            help_text = ("Left double-click on the curve: add a point at the nearest sample.   "
                         "Right double-click on a point: remove it.   Drag a point to move it.   "
                         "Save stores the picks.")
        else:
            help_text = "Drag a point to move it along the curve."
        self.help_label = ttk.Label(self, text=help_text, wraplength=900, justify="left")
        self.help_label.pack(side=tk.TOP, anchor="w", padx=8, pady=(6, 2))
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
        self.picker = PointPicker(self.ax, self.canvas, self.x_values, self.y_values, picked_idx,
                                 add_remove=add_remove_enabled, toolbar_active=self.toolbar_active)
        self.picker.draw_markers()

    # ------------------------------------------------------------- picks
    @property
    def picked_idx(self) -> list:
        """The live pick list (sample indices)."""
        return self.picker.picks

    @property
    def markers(self) -> list:
        return self.picker.markers

    def add_point(self, idx: int) -> int:
        return self.picker.add_point(idx)

    def remove_point(self, i: int) -> None:
        self.picker.remove_point(i)

    def nearest_index(self, cx: float, cy: float) -> int:
        return self.picker.nearest_index(cx, cy)

    def plot_data_points(self) -> None:
        self.picker.draw_markers()

    def renumber_markers(self) -> None:
        self.picker.renumber_markers()

    # ----------------------------------------------------------------- save
    def save(self):
        """Sort the picks (markers follow) and pass a *copy* to the callback.

        The callback typically stores the list verbatim (DataManager.set_data
        does not copy), so handing over the live list would alias the stored
        data with it and let later drags/adds/removes in this window edit the
        stored data without another Save.
        """
        self.picker.sort()
        if self.callback:
            self.callback(list(self.picker.picks))

"""A bare figure window: the main window plots into ``view.ax`` on double-click.

``BaseView`` already registers the window with the application, so callers
must not append it to ``app.child_windows`` again; ``on_close`` nevertheless
drops *every* occurrence of the view so a duplicate registration by a caller
cannot leave a destroyed Toplevel behind in the list.
"""
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from labquake_explorer.ui.views.base import BaseView


class SimplePlotView(BaseView):
    """One axes with a navigation toolbar; the caller draws into ``self.ax``."""

    window_title = "Simple Plot"

    def __init__(self, app):
        super().__init__(app)
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()

        toolbar_frame = ttk.Frame(self)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        self.canvas_widget.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)

    def on_close(self) -> None:
        # ``unregister_child`` removes one occurrence; guard against a caller
        # that appended the view a second time (see module docstring).
        windows = getattr(self.app, "child_windows", None)
        if windows is not None:
            while self in windows:
                windows.remove(self)
        super().on_close()

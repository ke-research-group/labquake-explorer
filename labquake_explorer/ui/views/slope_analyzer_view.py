"""Two draggable markers on an array; reports the slope of the chord between them."""
from __future__ import annotations

import numpy as np

from labquake_explorer.ui.views.index_picker_view import ArrayPairView


class SlopeAnalyzerView(ArrayPairView):
    """Slope of Y against X (or index) between two sample-snapped picks.

    The slope is written to the readout box and copied to the clipboard on
    every change.
    """

    window_title = "Slope Analyzer"
    readout_label = "Slope"
    #: sibling arrays tried, in order, as the default X: a slip channel first
    #: (loading stiffness is d(stress)/d(slip)), then the load-point displacement
    default_x_preference = ("displacement", "slip", "LP_displacement", "time")

    def default_x(self, names):
        for wanted in self.default_x_preference:
            if wanted in names and wanted != self.item_y:
                return wanted
        for name in names:
            if name != self.item_y and ("slip" in name or "displacement" in name):
                return name
        return None

    def __init__(self, app, item_y=None, item_x=None):
        self.slope_line = None
        self.slope = float("nan")
        super().__init__(app, item_y=item_y, item_x=item_x)

    @property
    def slope_textbox(self):
        return self.readout_textbox

    def set_slope_textbox(self, text: str):
        self.set_readout(text)

    def _chord(self):
        i0, i1 = self.picked_idx[0], self.picked_idx[1]
        return ([self.data_x[i0], self.data_x[i1]], [self.data_y[i0], self.data_y[i1]])

    def draw_overlays(self):
        if self.slope_line is not None and self.slope_line in self.ax.lines:
            self.slope_line.remove()
        self.slope_line = None
        if len(self.picked_idx) >= 2:
            self.slope_line, = self.ax.plot(*self._chord(), "--", color="gray", zorder=-50)

    def update_overlays(self):
        if self.slope_line is not None and len(self.picked_idx) >= 2:
            self.slope_line.set_data(*self._chord())

    def compute_slope(self) -> float:
        if len(self.picked_idx) < 2 or len(self.data_y) == 0:
            return float("nan")
        (x0, x1), (y0, y1) = self._chord()
        with np.errstate(divide="ignore", invalid="ignore"):
            return float((y1 - y0) / (x1 - x0))

    def update_slope(self):
        self.slope = self.compute_slope()
        self.set_readout(str(self.slope))

    def update_readout(self):
        self.update_slope()

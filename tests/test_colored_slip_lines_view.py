"""Headless-safe tests for explicit colored run-signal previews."""

import copy
import unittest
from unittest import mock

import numpy as np
from matplotlib.figure import Figure

from labquake_explorer.ui.labquake_explorer import LabquakeExplorer
from labquake_explorer.ui.views import ColoredSlipLinesView as ExportedView
from labquake_explorer.ui.views.colored_slip_lines_view import (
    ColoredSlipLinesView,
    find_colored_line_candidates,
    parse_line_width,
)


class FakeVariable:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class FakeWidget(FakeVariable):
    def __init__(self, value=""):
        super().__init__(value)
        self.options = {}
        self.destroyed = False

    def configure(self, **kwargs):
        self.options.update(kwargs)

    def destroy(self):
        self.destroyed = True


class FakeCanvas:
    def draw_idle(self):
        pass


def make_row(signal=None, *, color="C0", visible=True, width="1.5", label=""):
    return {
        "frame": FakeWidget(),
        "signal_name": signal,
        "signal_selector": FakeWidget(signal or ""),
        "color_var": FakeVariable(color),
        "visible_var": FakeVariable(visible),
        "linewidth_var": FakeVariable(width),
        "label_var": FakeVariable(label),
    }


def make_view(run_data=None):
    run_data = run_data or {
        "time": np.array([0.0, 1.0, 2.0]),
        "sensor z": np.array([1.0, 2.0, 3.0]),
        "sensor a": np.array([3.0, 2.0, 1.0]),
    }
    view = ColoredSlipLinesView.__new__(ColoredSlipLinesView)
    view.run_idx = 3
    view.run_data = run_data
    view.signal_candidates = find_colored_line_candidates(run_data)
    view.signal_rows = []
    view._color_index = 0
    view.status_var = FakeVariable()
    view.figure = Figure()
    view.ax = view.figure.add_subplot(111)
    view.canvas = FakeCanvas()
    return view


class CandidateTests(unittest.TestCase):
    def test_candidates_follow_insertion_order_without_name_inference(self):
        run = {
            "time": np.arange(3.0),
            "z unrelated": np.ones(3),
            "eddy": np.ones((3, 1)),
            "a unrelated": [1, 2, 3],
        }
        self.assertEqual(
            find_colored_line_candidates(run),
            ["z unrelated", "a unrelated"],
        )

    def test_candidate_filter_rejects_bool_complex_nonfinite_and_wrong_shape(self):
        run = {
            "time": np.arange(3.0),
            "bool": np.array([True, False, True]),
            "complex": np.ones(3, dtype=complex),
            "nan": np.array([1.0, np.nan, 2.0]),
            "short": np.ones(2),
            "two_d": np.ones((3, 1)),
            5: np.ones(3),
            "good": np.ones(3),
        }
        self.assertEqual(find_colored_line_candidates(run), ["good"])

    def test_invalid_run_time_or_mapping_produces_no_candidates(self):
        for run in (None, [], {}, {"time": [True, False]}, {"time": [[1, 2]]}):
            with self.subTest(run=run):
                self.assertEqual(find_colored_line_candidates(run), [])

    def test_candidate_filter_does_not_modify_or_copy_run_arrays(self):
        time = np.arange(3.0)
        signal = np.arange(3.0)
        run = {"time": time, "signal": signal}
        before = signal.copy()
        self.assertEqual(find_colored_line_candidates(run), ["signal"])
        self.assertIs(run["signal"], signal)
        np.testing.assert_array_equal(signal, before)

    def test_line_width_requires_finite_positive_number(self):
        self.assertEqual(parse_line_width("2.25"), 2.25)
        for value in ("", "x", "0", "-1", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_line_width(value)


class ContextAndStateTests(unittest.TestCase):
    def test_refresh_uses_exact_run_path_and_preserves_run_identity(self):
        run = {"time": np.arange(2.0), "signal": np.ones(2)}
        manager = mock.Mock()
        manager.get_data.return_value = run
        view = make_view()
        view.data_manager = manager
        view.signal_rows = []
        view._refresh_run_context(12)
        manager.get_data.assert_called_once_with("runs/[12]")
        self.assertIs(view.run_data, run)
        self.assertEqual(view.run_idx, 12)

    def test_same_run_refresh_preserves_valid_row_state(self):
        run = {"time": np.arange(2.0), "signal": np.ones(2)}
        row = make_row("signal", color="#123456", visible=False, width="4", label="E1")
        view = make_view(run)
        view.signal_rows = [row]
        view.data_manager = mock.Mock()
        view.data_manager.get_data.return_value = run
        view._refresh_run_context()
        self.assertEqual(row["signal_name"], "signal")
        self.assertEqual(row["color_var"].get(), "#123456")
        self.assertFalse(row["visible_var"].get())
        self.assertEqual(row["label_var"].get(), "E1")

    def test_stale_binding_is_cleared_without_reassignment(self):
        old_run = {"time": np.arange(2.0), "old": np.ones(2)}
        new_run = {"time": np.arange(2.0), "new": np.ones(2)}
        row = make_row("old")
        view = make_view(old_run)
        view.signal_rows = [row]
        view.data_manager = mock.Mock()
        view.data_manager.get_data.return_value = new_run
        view._refresh_run_context(4)
        self.assertIsNone(row["signal_name"])
        self.assertEqual(row["signal_selector"].get(), "")
        self.assertEqual(row["signal_selector"].options["values"], ["new"])

    def test_explicit_and_duplicate_bindings_are_allowed(self):
        view = make_view()
        first = make_row()
        second = make_row()
        view.signal_rows = [first, second]
        for row in view.signal_rows:
            row["signal_selector"].set("sensor z")
            view.on_signal_changed(row)
        self.assertEqual([row["signal_name"] for row in view.signal_rows], ["sensor z", "sensor z"])

    def test_invalid_selection_clears_binding_without_fallback(self):
        view = make_view()
        row = make_row("sensor z")
        row["signal_selector"].set("missing")
        view.signal_rows = [row]
        view.on_signal_changed(row)
        self.assertIsNone(row["signal_name"])

    def test_remove_deletes_only_selected_row(self):
        view = make_view()
        first, second = make_row("sensor z"), make_row("sensor a")
        view.signal_rows = [first, second]
        view.remove_signal_row(first)
        self.assertEqual(view.signal_rows, [second])
        self.assertTrue(first["frame"].destroyed)


class PlotTests(unittest.TestCase):
    def test_plot_uses_raw_time_signal_and_independent_styles(self):
        view = make_view()
        view.signal_rows = [
            make_row("sensor z", color="red", width="2.5", label="Push"),
            make_row("sensor a", color="blue", width="1", label="Pull"),
        ]
        original = copy.deepcopy(view.run_data)
        view._plot_lines()
        self.assertEqual(len(view.ax.lines), 2)
        np.testing.assert_array_equal(view.ax.lines[0].get_xdata(), view.run_data["time"])
        np.testing.assert_array_equal(view.ax.lines[0].get_ydata(), view.run_data["sensor z"])
        self.assertEqual(view.ax.lines[0].get_color(), "red")
        self.assertEqual(view.ax.lines[0].get_linewidth(), 2.5)
        self.assertEqual(view.ax.lines[0].get_label(), "Push")
        np.testing.assert_array_equal(view.run_data["sensor z"], original["sensor z"])

    def test_visibility_hides_only_that_row(self):
        view = make_view()
        view.signal_rows = [
            make_row("sensor z", visible=False),
            make_row("sensor a", visible=True),
        ]
        view._plot_lines()
        self.assertEqual(len(view.ax.lines), 1)
        np.testing.assert_array_equal(view.ax.lines[0].get_ydata(), view.run_data["sensor a"])

    def test_blank_label_uses_exact_signal_key_and_legend_is_present(self):
        view = make_view()
        view.signal_rows = [make_row("sensor z", label="")]
        view._plot_lines()
        self.assertEqual(view.ax.lines[0].get_label(), "sensor z")
        self.assertIsNotNone(view.ax.get_legend())

    def test_no_visible_bound_rows_has_no_legend(self):
        view = make_view()
        view.signal_rows = [make_row(None), make_row("sensor z", visible=False)]
        view._plot_lines()
        self.assertEqual(len(view.ax.lines), 0)
        self.assertIsNone(view.ax.get_legend())

    def test_plot_does_not_call_analysis_or_persistence(self):
        view = make_view()
        view.signal_rows = [make_row("sensor z")]
        with mock.patch.object(view, "data_manager", create=True) as manager:
            view._plot_lines()
        manager.set_data.assert_not_called()
        manager.save_file.assert_not_called()

    def test_invalid_style_is_reported_without_mutating_data(self):
        view = make_view()
        view.signal_rows = [make_row("sensor z", width="bad")]
        before = view.run_data["sensor z"].copy()
        with mock.patch(
            "labquake_explorer.ui.views.colored_slip_lines_view.messagebox.showerror"
        ) as showerror:
            view.recompute_preview()
        self.assertIn("Line width", view.status_var.get())
        showerror.assert_called_once()
        np.testing.assert_array_equal(view.run_data["sensor z"], before)


class WiringAndLifecycleTests(unittest.TestCase):
    def test_view_is_exported_and_controller_method_exists(self):
        self.assertIs(ExportedView, ColoredSlipLinesView)
        self.assertTrue(callable(LabquakeExplorer.analyze_colored_slip_lines))

    @mock.patch("labquake_explorer.ui.labquake_explorer.ColoredSlipLinesView")
    def test_controller_uses_shared_parser_and_opens_run_view(self, view_class):
        explorer = LabquakeExplorer.__new__(LabquakeExplorer)
        explorer.get_full_path = mock.Mock(return_value=(r"runs\[3]\events\[7]", "[7]"))
        explorer._extract_run_event_indices = mock.Mock(return_value=(3, 7))
        explorer.set_window_icon = mock.Mock()
        explorer.child_windows = []
        view = view_class.return_value
        explorer.analyze_colored_slip_lines()
        explorer._extract_run_event_indices.assert_called_once_with(r"runs\[3]\events\[7]")
        view_class.assert_called_once_with(explorer, 3)
        explorer.set_window_icon.assert_called_once_with(view)
        self.assertEqual(explorer.child_windows, [view])

    def test_context_menu_has_colored_lines_entry(self):
        explorer = LabquakeExplorer.__new__(LabquakeExplorer)
        explorer.root = object()

        class FakeMenu:
            def __init__(self, *args, **kwargs):
                self.commands = []

            def add_command(self, **kwargs):
                self.commands.append(kwargs)

        with mock.patch("labquake_explorer.ui.labquake_explorer.tk.Menu", FakeMenu):
            explorer.create_context_menus()
        labels = [item["label"] for item in explorer.event_menu.commands]
        self.assertIn("Analyze Colored Slip Lines", labels)

    def test_close_removes_view_without_persistence(self):
        view = ColoredSlipLinesView.__new__(ColoredSlipLinesView)
        view.parent = mock.Mock()
        view.parent.child_windows = [view]
        view.destroy = mock.Mock()
        view.on_close()
        self.assertNotIn(view, view.parent.child_windows)
        view.destroy.assert_called_once()
        view.parent.data_manager.set_data.assert_not_called()


if __name__ == "__main__":
    unittest.main()

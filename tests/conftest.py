import tkinter as tk

import pytest

from labquake_explorer.ui.labquake_explorer import LabquakeExplorer
from tests.synthetic import make_experiment


@pytest.fixture(scope="session")
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - no display
        pytest.skip(f"Tk unavailable: {exc}")
    root.withdraw()
    yield root
    try:
        root.destroy()
    except tk.TclError:
        pass


@pytest.fixture
def experiment():
    data, truth = make_experiment()
    return data, truth


@pytest.fixture
def app(tk_root, experiment):
    """A LabquakeExplorer with a synthetic experiment loaded into its tree."""
    data, truth = experiment
    application = LabquakeExplorer(tk_root)
    application.data_manager.data = data
    application.refresh_tree()
    application.truth = truth
    yield application
    for window in list(application.child_windows):
        try:
            window.destroy()
        except tk.TclError:
            pass
    application.child_windows.clear()
    try:
        application.data_tree.destroy()
    except tk.TclError:
        pass

import tkinter as tk
from tkinter import messagebox

import pytest

from tests.synthetic import make_experiment, make_experiment_with_strain


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


@pytest.fixture(autouse=True)
def no_dialogs(monkeypatch):
    """Never block on a modal dialog under pytest; errors become failures."""
    calls = []

    def fail(title, message, **kwargs):
        calls.append((title, message))
        raise AssertionError(f"dialog {title!r}: {message}")

    monkeypatch.setattr(messagebox, "showerror", fail)
    monkeypatch.setattr(messagebox, "showwarning", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(messagebox, "showinfo", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(messagebox, "askokcancel", lambda *a, **k: True)
    monkeypatch.setattr(messagebox, "askyesno", lambda *a, **k: True)
    return calls


@pytest.fixture
def experiment():
    data, truth = make_experiment(recurrence=12.0)
    return data, truth


@pytest.fixture
def experiment_with_strain():
    data, truth = make_experiment_with_strain(recurrence=12.0)
    return data, truth


def _make_app(tk_root, data, truth):
    from labquake_explorer.ui.labquake_explorer import LabquakeExplorer
    application = LabquakeExplorer(tk_root)
    application.data_manager.data = data
    application.refresh_tree()
    application.truth = truth
    return application


def _close_app(application):
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


@pytest.fixture
def app(tk_root, experiment):
    """A LabquakeExplorer with a synthetic experiment loaded into its tree."""
    application = _make_app(tk_root, *experiment)
    yield application
    _close_app(application)


@pytest.fixture
def app_with_strain(tk_root, experiment_with_strain):
    """Same as ``app`` but every event carries a synthetic strain block."""
    application = _make_app(tk_root, *experiment_with_strain)
    yield application
    _close_app(application)

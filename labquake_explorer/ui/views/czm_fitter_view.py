"""Fit a cohesive-zone (slip-weakening) rupture model to near-fault strain.

Two shared-x axes show the shear (Exy, selected gauge) and normal (Eyy,
gauge 14; the lower panel is left empty with a note when the strain block
has fewer channels) strain around one event together with the cohesive-zone
prediction.  Three draggable vertical-line pairs mark the zeroing point for
Eyy (x_min), the rupture tip (x_tip) and the zeroing point / fit end for Exy
(x_max).  ``Fit`` adjusts Gc and Xc by L-BFGS-B over the tip..max window.
Results are saved under ``event['czm_parms']`` as a dict.
"""
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from scipy import optimize, signal

from labquake_explorer.data.data_processor import DataProcessor
from labquake_explorer.ui.actions import register_view
from labquake_explorer.ui.context import EVENT
from labquake_explorer.ui.views.base import EventView
from labquake_explorer.data.sources import pick_waveform_block, waveform_block, waveform_data, waveform_time
from labquake_explorer.utils.cohesive_crack import CohesiveCrack


@register_view("Fit Cohesive Zone Model", kinds=[EVENT], order=30)
class CZMFitterView(EventView):
    window_title = "Cohesive Zone Model Fitting"
    result_key = "czm_parms"

    # Material properties
    E = 51e9      # Young's modulus (Pa)
    nu = 0.25     # Poisson's ratio
    C_s = 2760    # Shear wave speed (m/s)
    C_d = 4790    # Longitudinal wave speed (m/s)

    DEFAULT_GAUGE = 6
    EYY_GAUGE = 14

    def __init__(self, app, run_idx, event_idx):
        self.filtering = False
        self.num_gauges = 0
        self.x_lim_min, self.x_lim_max = -0.1, 0.1
        self.line_positions = []
        self.vlines = []
        self.vlines_twin = []
        self.active_line_idx = None
        self.drag_active = False
        super().__init__(app, run_idx, event_idx)

    # ------------------------------------------------------------------ ui
    def build_ui(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.strain_gauge = tk.IntVar(master=self, value=self.DEFAULT_GAUGE)
        self.filter_window = tk.IntVar(master=self, value=51)
        self.Cf = tk.DoubleVar(master=self)
        self.y = tk.DoubleVar(master=self)
        self.Xc = tk.DoubleVar(master=self)
        self.Gc = tk.DoubleVar(master=self)

        self.create_control_frame()
        self.create_matplotlib_figure()
        self.create_parameters_frame()

        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def create_control_frame(self):
        control_frame = ttk.Frame(self)
        control_frame.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        # Event selection (columns 0-1)
        self.build_event_selector(control_frame, row=0, column=0)

        # Strain gauge selection
        ttk.Label(control_frame, text="Strain Gauge:").grid(row=0, column=2, padx=5, sticky="w")
        self.gauge_combobox = ttk.Combobox(control_frame, textvariable=self.strain_gauge,
                                           width=5, state="readonly")
        self.gauge_combobox.grid(row=0, column=3, padx=5, sticky="w")
        self.gauge_combobox.bind("<<ComboboxSelected>>", self.update_plot)

        # Filter controls
        filter_frame = ttk.Frame(control_frame)
        filter_frame.grid(row=0, column=4, padx=10, sticky="w")
        ttk.Label(filter_frame, text="Filter Window:").pack(side=tk.LEFT, padx=2)
        self.filter_spinbox = ttk.Spinbox(
            filter_frame, from_=3, to=201, increment=2, textvariable=self.filter_window,
            width=10, validate='focusout',
            validatecommand=(self.register(self.validate_filter_window), '%P'))
        self.filter_spinbox.pack(side=tk.LEFT)
        self.filter_spinbox.bind("<Return>", self.update_plot)

        self.filter_button = tk.Button(control_frame, text="Filter Off", relief="raised",
                                       command=self.toggle_filter)
        self.filter_button.grid(row=0, column=5, padx=5, sticky="w")

    def create_matplotlib_figure(self):
        self.make_figure(figsize=(10, 6), row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.fig = self.figure
        self.gs = self.figure.add_gridspec(2, hspace=0.3)
        self.axs = self.gs.subplots(sharex=True)
        for ax in self.axs:
            ax.grid(True)

    def create_parameters_frame(self):
        params_frame = ttk.Frame(self)
        params_frame.grid(row=3, column=0, padx=5, pady=5, sticky="ew")

        param_configs = [
            ("Cf", self.Cf, 10),
            ("y", self.y, 1e-3),
            ("Xc", self.Xc, 1),
            ("Gc", self.Gc, 1),
        ]
        self.param_spinboxes = {}
        for label_text, var, increment in param_configs:
            frame = ttk.Frame(params_frame)
            frame.pack(side=tk.LEFT, padx=10)
            ttk.Label(frame, text=label_text).pack(side=tk.LEFT, padx=2)
            spinbox = ttk.Spinbox(frame, textvariable=var, width=10, from_=0, to=1e6, increment=increment)
            spinbox.pack(side=tk.LEFT)
            self.param_spinboxes[label_text] = spinbox

        button_frame = ttk.Frame(params_frame)
        button_frame.pack(side=tk.LEFT, padx=10)
        self.update_button = ttk.Button(button_frame, text="Update", command=self.update_plot)
        self.update_button.pack(side=tk.LEFT, padx=5)
        self.fit_button = ttk.Button(button_frame, text="Fit", command=self.fit_parameters)
        self.fit_button.pack(side=tk.LEFT, padx=5)
        self.save_button = ttk.Button(button_frame, text="Save", command=self.save_parameters)
        self.save_button.pack(side=tk.LEFT, padx=5)

    # -------------------------------------------------------------- loading
    @staticmethod
    def strain_block(event):
        """The event's full-rate record: ``strain`` when present, else the first
        waveform block (``elsys``, ``ni``, ...); None without any."""
        if not isinstance(event, dict):
            return None
        key = pick_waveform_block(event, prefer_keys=("strain",))
        return waveform_block(event, key) if key else None

    @staticmethod
    def strain_raw(event):
        """The event's full-rate samples (2-D, >= 1 channel), else None."""
        strain = CZMFitterView.strain_block(event)
        if strain is None:
            return None
        try:
            raw = waveform_data(strain)
        except (KeyError, TypeError, ValueError):
            return None
        try:
            if np.ndim(raw) != 2 or len(raw) == 0:
                return None
        except TypeError:
            return None
        return raw

    def set_event(self, event_idx: int) -> None:
        """Switch events, but refuse (with a warning) an event without strain data.

        The check happens before ``EventView.set_event`` touches ``event_idx``,
        ``event``, the title or the combobox, so a refused switch leaves the
        view (traces, marker lines, parameters) exactly as it was.
        """
        event_idx = int(event_idx)
        try:
            candidate = self.data_manager.get_data(f"{self.run_path}/events/[{event_idx}]")
        except (KeyError, IndexError, TypeError):
            candidate = None
        if self.strain_raw(candidate) is None:
            self.refresh_event_selector()  # combobox may already show the refused index
            messagebox.showwarning(
                "No strain data",
                f"Event {event_idx} has no strain data to fit a cohesive zone model to; "
                f"staying on event {self.event_idx}.",
                parent=self,
            )
            return
        super().set_event(event_idx)

    def on_event_loaded(self):
        self._clear_vlines()

        raw = self.strain_raw(self.event)
        if raw is None:
            # first load only (set_event refuses such events): EventView.__init__
            # unregisters and destroys the half-built window before re-raising
            raise ValueError(f"{self.event_path} has no strain data (no waveform record or strain block) "
                             "to fit a cohesive zone model to")
        self.num_gauges = len(raw)
        self.gauge_combobox.config(values=[str(i) for i in range(self.num_gauges)])
        default_gauge = min(self.DEFAULT_GAUGE, self.num_gauges - 1)

        params = self.event.get(self.result_key)
        if isinstance(params, (list, tuple, np.ndarray)) and np.ndim(params) == 1 and len(params) == 8:
            # legacy layout: [Cf, y, Xc, Gc, x_min, x_tip, x_lim_min, x_lim_max]
            # (a list in memory / .npz, an ndarray after an HDF5 round trip)
            params = [float(v) for v in params]
            self._set_parameters(*params[:4])
            x0, x1 = params[4], params[5]
            self.line_positions = [x0, x1, 2 * x1 - x0]
            self.x_lim_min, self.x_lim_max = params[6], params[7]
            self.strain_gauge.set(default_gauge)
        elif isinstance(params, dict):
            self._set_parameters(params['Cf'], params['y'], params['Xc'], params['Gc'])
            self.line_positions = [params['x_min'], params['x_tip'], params['x_max']]
            self.x_lim_min, self.x_lim_max = params['x_lim_min'], params['x_lim_max']
            gauge = params.get('strain_gauge')
            if isinstance(gauge, (int, np.integer)) and 0 <= gauge < self.num_gauges:
                self.strain_gauge.set(int(gauge))
            else:
                self.strain_gauge.set(default_gauge)
        else:
            self._set_default_parameters()
            self.line_positions = []
            if not 0 <= self.strain_gauge.get() < self.num_gauges:
                self.strain_gauge.set(default_gauge)
        self.gauge_combobox.set(self.strain_gauge.get())

        self.axs[0].set_xlim(self.x_lim_min, self.x_lim_max)
        self.update_plot()

    def _set_parameters(self, Cf, y, Xc, Gc):
        self.Cf.set(Cf)
        self.y.set(y)
        self.Xc.set(Xc)
        self.Gc.set(Gc)

    def _set_default_parameters(self):
        self.x_lim_min, self.x_lim_max = -0.1, 0.1
        try:
            self.Cf.set(float(np.abs(self.event["rupture_speed"])))
        except (KeyError, TypeError, ValueError):
            self.Cf.set(10)
        self.y.set(8e-3)
        self.Xc.set(1)
        self.Gc.set(1)

    def _clear_vlines(self):
        for line in self.vlines + self.vlines_twin:
            try:
                line.remove()
            except (ValueError, NotImplementedError):
                pass
        self.vlines = []
        self.vlines_twin = []

    def current_line_positions(self):
        """x positions of the three marker lines (from the artists when present)."""
        if self.vlines:
            return [float(line.get_xdata()[0]) for line in self.vlines]
        if self.line_positions:
            return list(self.line_positions)
        # evenly spaced across the view: 5 points, keep the middle 3
        return list(np.linspace(self.x_lim_min, self.x_lim_max, 5)[1:-1])

    # --------------------------------------------------------------- filter
    def toggle_filter(self):
        self.filtering = not self.filtering
        if self.filtering:
            self.filter_button.config(text="Filter On", relief="sunken")
        else:
            self.filter_button.config(text="Filter Off", relief="raised")
        self.update_plot()

    def validate_filter_window(self, value):
        """Validate that the filter window value is an odd integer."""
        if value == "":  # Allow empty field for editing
            return True
        try:
            val = int(value)
            return 3 <= val <= 201 and val % 2 == 1
        except ValueError:
            return False

    def _filter_window_length(self):
        window_length = self.filter_window.get()
        if window_length % 2 == 0:
            window_length += 1
            self.filter_window.set(window_length)
        return window_length

    def _strain(self, gauge_idx):
        """Strain of one gauge, Savitzky-Golay filtered when filtering is on."""
        strain = DataProcessor.voltage_to_strain(waveform_data(self.strain_block(self.event))[gauge_idx])
        if self.filtering:
            strain = signal.savgol_filter(strain, self._filter_window_length(), 2)
        return strain

    def eyy_gauge(self):
        """Index of the Eyy gauge, or None when the strain block has no channel ``EYY_GAUGE``."""
        return self.EYY_GAUGE if 0 <= self.EYY_GAUGE < self.num_gauges else None

    def _time(self):
        return waveform_time(self.strain_block(self.event)) - self.event["event_time"]

    def _model_strains(self, t, x_tip, Xc, Gc):
        """Cohesive-zone (Exy, Eyy) strains along the gauge line at times ``t``."""
        Cf = self.Cf.get()
        x_zeroed = -t * Cf + x_tip * Cf  # position relative to the tip, in meters
        delta_sigma_xx, delta_sigma_xy, delta_sigma_yy = CohesiveCrack.delta_sigmas(
            x_zeroed, self.y.get(), Xc, Cf, self.C_s, self.C_d, self.nu, Gc, self.E)
        delta_e_xx, delta_e_xy, delta_e_yy = DataProcessor.stress_to_strain(
            self.E, self.nu, delta_sigma_xx, delta_sigma_xy, delta_sigma_yy)
        return delta_e_xy, delta_e_yy

    # ----------------------------------------------------------------- plot
    def update_plot(self, event=None):
        line_positions = self.current_line_positions()

        xlim_temp = self.axs[0].get_xlim()
        for ax in self.axs:
            ax.clear()

        t = self._time()
        exy = self._strain(self.strain_gauge.get())
        eyy_gauge = self.eyy_gauge()

        idx_zero_xy = int(np.argmin(np.abs(t - line_positions[2])))
        idx_zero_yy = int(np.argmin(np.abs(t - line_positions[0])))
        self.axs[0].plot(t, exy - exy[idx_zero_xy], 'b-', label='Exy')

        delta_e_xy, delta_e_yy = self._model_strains(t, line_positions[1], self.Xc.get(), self.Gc.get())
        delta_e_xy = delta_e_xy - delta_e_xy[idx_zero_xy]
        delta_e_yy = delta_e_yy - delta_e_yy[idx_zero_yy]
        self.axs[0].plot(t, delta_e_xy, 'g--', label='CZM')
        if eyy_gauge is not None:
            eyy = self._strain(eyy_gauge)
            self.axs[1].plot(t, eyy - eyy[idx_zero_yy], 'r-', label='Eyy')
            self.axs[1].plot(t, delta_e_yy, 'g--', label='CZM')
        else:
            self.axs[1].text(0.5, 0.5, f"no Eyy gauge: channel {self.EYY_GAUGE} missing "
                                       f"({self.num_gauges} strain channels)",
                             ha='center', va='center', transform=self.axs[1].transAxes, color='gray')

        self.axs[1].set_xlabel('Time (s)')
        self.axs[0].set_ylabel('Exy')
        self.axs[1].set_ylabel('Eyy')
        self.figure.suptitle(self.figure_title())

        self.vlines = []
        self.vlines_twin = []
        for x_pos in line_positions:
            self.vlines.append(self.axs[0].axvline(x=x_pos, color='g', linestyle='--', alpha=0.5))
            self.vlines_twin.append(self.axs[1].axvline(x=x_pos, color='g', linestyle='--', alpha=0.5))
        self.line_positions = list(line_positions)

        for ax in self.axs:
            ax.grid(True)
            if ax.get_legend_handles_labels()[0]:
                ax.legend()
        self.axs[0].set_xlim(xlim_temp)
        self.canvas.draw()

    # ------------------------------------------------------------- dragging
    def on_mouse_press(self, event):
        if self.toolbar_active() or event.button != 1 or not event.inaxes or event.xdata is None:
            return
        xlim = self.axs[0].get_xlim()
        tolerance = 0.01 * (xlim[1] - xlim[0])
        for i, vline in enumerate(self.vlines):
            if abs(event.xdata - vline.get_xdata()[0]) < tolerance:
                self.drag_active = True
                self.active_line_idx = i
                break

    def on_mouse_release(self, event):
        self.drag_active = False
        self.active_line_idx = None

    def on_mouse_move(self, event):
        if self.toolbar_active():
            return
        if self.drag_active and event.inaxes and self.active_line_idx is not None:
            self.move_line(self.active_line_idx, event.xdata)

    def move_line(self, idx, new_x):
        """Move marker pair ``idx`` to ``new_x`` and redraw."""
        new_x = float(new_x)
        self.vlines[idx].set_xdata([new_x, new_x])
        self.vlines_twin[idx].set_xdata([new_x, new_x])
        self.update_plot()

    # ------------------------------------------------------------------ fit
    def build_fit_objective(self):
        """Return ``objective([Gc, Xc])`` for the Exy data between the tip and max lines.

        The objective is the sum of squared residuals (in nano-strain) between
        the zeroed measured Exy and the cohesive-zone prediction over the
        tip..max window.  Returns None when the three marker lines are not
        available or no samples lie inside the window.
        """
        if len(self.vlines) < 3:
            print("Need 3 vertical lines to define fitting region")
            return None
        t0, t1, t2 = sorted(self.current_line_positions())

        t = self._time()
        exy = self._strain(self.strain_gauge.get())
        mask = (t >= t1) & (t <= t2)
        if not np.any(mask):
            print("No samples between the tip and max lines")
            return None
        t_fit = t[mask]
        exy_fit = exy[mask]
        idx_zero = int(np.argmin(np.abs(t_fit - t2)))
        exy_fit = exy_fit - exy_fit[idx_zero]

        def objective(params):
            Gc, Xc = params
            delta_e_xy, _ = self._model_strains(t_fit, t1, Xc, Gc)
            delta_e_xy = delta_e_xy - delta_e_xy[idx_zero]
            return float(np.sum(((exy_fit - delta_e_xy) * 1e9) ** 2))

        return objective

    def fit_parameters(self):
        """Fit Gc and Xc to the Exy data between the tip and max lines.

        Returns the scipy ``OptimizeResult`` (with ``initial_fun`` added: the
        objective at the starting guess), or None if no fit could be set up.
        """
        objective = self.build_fit_objective()
        if objective is None:
            return None

        initial_guess = [self.Gc.get(), self.Xc.get()]
        bounds = ((1e-6, None), (1e-6, None))
        result = optimize.minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        result.initial_fun = objective(initial_guess)

        if result.success:
            self.Gc.set(float(result.x[0]))
            self.Xc.set(float(result.x[1]))
            self.update_plot()
            print(f"Fitted parameters: Gc={result.x[0]:.2e}, Xc={result.x[1]:.2f}")
        else:
            print("Fitting failed:", result.message)
        return result

    # ----------------------------------------------------------------- save
    def collect_results(self):
        x0, x1, x2 = self.current_line_positions()
        self.x_lim_min, self.x_lim_max = (float(v) for v in self.axs[0].get_xlim())
        return {
            'Cf': self.Cf.get(),
            'y': self.y.get(),
            'Xc': self.Xc.get(),
            'Gc': self.Gc.get(),
            'x_min': x0,
            'x_tip': x1,
            'x_max': x2,
            'x_lim_min': self.x_lim_min,
            'x_lim_max': self.x_lim_max,
            'strain_gauge': int(self.strain_gauge.get()),
        }

    def save_parameters(self):
        """Save the current parameters to the event data."""
        if len(self.vlines) < 3:
            return
        results = self.collect_results()
        self.save_results(results)
        print(f"Saved parameters for event {self.event_idx}: {results}")

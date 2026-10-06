"""Pure numerical analysis for Labquake Explorer.

Functions here take arrays in and return dataclasses or dicts; they know
nothing about Tk, the data tree, or persistence.  Views call them and decide
what to display and save.
"""
from labquake_explorer.analysis.fitting import LinearFit, linear_fit
from labquake_explorer.analysis.event_metrics import (
    EventPicks, analyze_event, picks_from_result, picks_from_windows, windows_from_picks, RESULT_VERSION,
)
from labquake_explorer.analysis.interevent import interevent_metrics, sample_after, creep_per_cycle
from labquake_explorer.analysis.scaling import PowerLawFit, fit_power_law, bootstrap_exponent, reference_line
from labquake_explorer.analysis.spectrum import (
    WindowResult, Calibration, SpectrumResult, SpectralFit,
    extract_window, taper_window, load_calibration_csv, t_star_from_q,
    compute_spectrum, bin_spectrum, fit_omega_n, brune_spectrum, brune_pulse,
)
from labquake_explorer.analysis.source import (
    PHASE_CONSTANTS, SourceParameters, source_parameters, seismic_moment, moment_magnitude,
    stress_drop_eshelby, source_radius, radiation_pattern, radiation_coefficient,
    free_surface_amplification, geometry,
)

__all__ = [
    "LinearFit", "linear_fit",
    "EventPicks", "analyze_event", "picks_from_result", "picks_from_windows", "windows_from_picks", "RESULT_VERSION",
    "interevent_metrics", "sample_after", "creep_per_cycle",
    "PowerLawFit", "fit_power_law", "bootstrap_exponent", "reference_line",
    "WindowResult", "Calibration", "SpectrumResult", "SpectralFit",
    "extract_window", "taper_window", "load_calibration_csv", "t_star_from_q",
    "compute_spectrum", "bin_spectrum", "fit_omega_n", "brune_spectrum", "brune_pulse",
    "PHASE_CONSTANTS", "SourceParameters", "source_parameters", "seismic_moment", "moment_magnitude",
    "stress_drop_eshelby", "source_radius", "radiation_pattern", "radiation_coefficient",
    "free_surface_amplification", "geometry",
]

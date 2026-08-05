# Manual Smoke Test Checklist

Use real canonical data and treat every integrated window as a read-only preview.
Before testing, confirm the working tree is clean and make a separate backup of
the experimental data. None of these checks should write or modify source data.

## General application checks

- [ ] Record the canonical input file path and its modification time.
- [ ] Start Labquake Explorer and open the real canonical file.
- [ ] Expand at least one run and its events.
- [ ] Confirm the event context menu contains Event Drop, Loading Stiffness,
      Colored Slip Lines, PZT Time Domain, PZT Spectrum, and PZT EGF entries.
- [ ] Open and close each applicable view.
- [ ] Confirm each child window closes normally and the main window remains usable.
- [ ] Confirm no console traceback appears during ordinary operations.
- [ ] Confirm no save prompt or DataManager write indication appears.

## Event Drop

- [ ] Confirm there is one event-local **Signal** selector and no Tau/Mu/LVDT/Slip role selectors.
- [ ] Confirm Preview starts disabled and becomes enabled only after explicit selection.
- [ ] Change the selected signal; confirm the old fit/result clears and the raw plot changes.
- [ ] Confirm signed delta and magnitude are both displayed.
- [ ] Confirm pre/post fitted values and fit lines are displayed.
- [ ] Drag all four fitting endpoints repeatedly; confirm motion continues without disconnecting.
- [ ] Release a dragged endpoint; confirm shading and lines redraw at the new values.
- [ ] Edit fitting parameters manually; confirm the prior result becomes stale without automatic analysis.
- [ ] On the first event, confirm D_Push, D_max, and D_reference are unavailable.
- [ ] On a later event with no run signals bound, confirm only D_Push can be calculated.
- [ ] Select D_max and reference independently from full-run candidates.
- [ ] Bind the same exact signal to D_max and reference; confirm both remain permitted.
- [ ] Confirm D_max responds to its smoothing setting.
- [ ] Change D_max smoothing; confirm D_reference is unaffected.
- [ ] Exercise a delayed target outside the run; confirm a Potter-compatible endpoint result remains valid.
- [ ] Change push speed; confirm only D_Push becomes stale.
- [ ] Change delay; confirm D_max and D_reference become stale.
- [ ] Change D_max smoothing or binding; confirm only D_max becomes stale.
- [ ] Change reference binding; confirm only D_reference becomes stale.
- [ ] Change the event-local signal; confirm existing D results are not cleared.
- [ ] Switch event; confirm the event-local signal/result clears.
- [ ] Confirm still-valid full-run D bindings remain after event switching.
- [ ] Confirm a binding absent from refreshed run candidates is cleared, not remapped.
- [ ] Close the view and confirm no callback error occurs.

## Loading stiffness

- [ ] Confirm Tau and Slip fields are analysis roles whose exact signals are explicitly selected.
- [ ] Bind two exact full-run signals; confirm no automatic physical identity selection occurs.
- [ ] Run OLS preview and confirm signed `k` and intercept display.
- [ ] Run RANSAC preview and record that repeated values may vary with sklearn defaults.
- [ ] Test reasonable high-pass and low-pass cutoffs.
- [ ] Confirm cutoff `0` means off.
- [ ] Confirm cutoff at or above Nyquist is treated as off without an error.
- [ ] Confirm enabled processing applies high-pass before low-pass.
- [ ] Use a window too short for filtering; confirm an invalid result rather than a crash.
- [ ] Drag both pre-window endpoints repeatedly; confirm motion does not disconnect.
- [ ] Release an endpoint; confirm the plot redraws at the new window.
- [ ] Edit a parameter; confirm the previous result clears without automatic analysis.
- [ ] Switch event; confirm bindings/results clear while controls retain their values.
- [ ] Close the view and confirm no callback error occurs.

## Colored slip lines

- [ ] Confirm the view initially contains no signal row.
- [ ] Add a row and bind an exact run-level signal.
- [ ] Add another row bound to the same signal; confirm duplicate binding is allowed.
- [ ] Toggle visibility and confirm only that row changes.
- [ ] Change color, line width, and label; confirm the plot and legend update.
- [ ] Remove one row; confirm other rows remain.
- [ ] Confirm plotted values are raw run time versus raw signal.
- [ ] Confirm there is no offset, normalization, smoothing, filtering, or analysis.
- [ ] Refresh the same run; confirm valid row state remains.
- [ ] Remove a bound signal from a test copy/context; confirm its stale binding clears without remapping.
- [ ] Close the view normally.

## PZT time-domain

- [ ] Open an event with canonical `strain.original.time/raw` data.
- [ ] Confirm 1-D raw data offers one channel and 2-D raw data offers explicit channel indices.
- [ ] Confirm no channel is selected automatically.
- [ ] Select a channel explicitly and run Preview.
- [ ] Confirm canonical `event_time` aligns with the analysis trigger at `t=0`.
- [ ] With a multi-pulse trace, confirm preview remains on the canonical pulse rather than a larger pulse.
- [ ] Exercise pre/post controls and confirm the returned window, taper, and noise displays update.
- [ ] Switch event; confirm channel/result clear while controls retain values.
- [ ] Confirm no save/apply/export control or persistence occurs.

## PZT spectrum

- [ ] Select valid calibration and Q CSV files and record their paths.
- [ ] Cancel each browse dialog; confirm the existing path is unchanged.
- [ ] Enter an invalid path; confirm a clear expected-error dialog appears.
- [ ] Select a canonical strain channel explicitly and compute the spectrum.
- [ ] Confirm all three spectrum plots render without log-mask errors.
- [ ] With a multi-pulse trace, confirm the canonical pulse is used.
- [ ] Change a spectrum parameter; confirm spectrum and fit clear.
- [ ] Confirm Fit is enabled only after successful spectrum computation.
- [ ] Run omega-n fitting; confirm the existing spectrum is not recomputed.
- [ ] Change only fit parameters; confirm only the fit clears.
- [ ] Confirm Omega0, fc, n, c, R², fit interval, and model overlay display.
- [ ] Switch event; confirm paths/controls remain while channel/spectrum/fit clear.
- [ ] Confirm expected file/CSV errors use a dialog and programming errors are not hidden.
- [ ] Confirm no persistence occurs.

## PZT EGF

- [ ] Select and record a valid calibration CSV path.
- [ ] Select a canonical strain channel explicitly.
- [ ] Confirm the canonical trigger is used.
- [ ] With a multi-pulse trace, confirm analysis does not jump to an automatic peak.
- [ ] Confirm time, noise, and spectrum plots render.
- [ ] Confirm Fit is enabled only after successful spectrum computation.
- [ ] Fit the existing spectrum and confirm spectrum analysis is not rerun.
- [ ] Confirm Omega0, fc, n, R², M0, Mw, fit interval, and model overlay display.
- [ ] Note that the current GUI does not expose a separate `t_star` result; record if one is required.
- [ ] Change fit controls; confirm only the fit becomes stale.
- [ ] Change spectrum controls; confirm both spectrum and fit become stale.
- [ ] Switch event; confirm controls/path remain while channel/results clear.
- [ ] Confirm an expected fitting `RuntimeError` is presented and programming errors propagate.
- [ ] Confirm there is no pairing, spectral ratio, STF/deconvolution, or persistence.

## Source analysis APIs

These are **API/test verified — no standalone GUI entry**.

- [ ] Verify BAC source parameters with explicit experiment-derived inputs:

  ```python
  result = calculate_bac_source_parameters(
      omega0=omega0,
      corner_frequency_hz=fc,
      source_receiver_distance_m=distance_m,
      radiation_coefficient=sa,
  )
  ```

- [ ] Verify rectangular or circular source geometry with explicit sensor/source
      coordinates, fault dimensions, and any non-default grid/radiation inputs.
- [ ] Verify power-law scaling with explicit paired arrays using keyword-only
      calls:

  ```python
  coefficient, exponent = fit_power_law(
      x=x_values,
      y=y_values,
  )

  r_squared = compute_power_law_r_squared(
      x=x_values,
      y=y_values,
      coefficient=coefficient,
      exponent=exponent,
  )

  curve_x, curve_y = build_fit_curve(
      x_min=x_min,
      x_max=x_max,
      coefficient=coefficient,
      exponent=exponent,
  )
  ```
- [ ] Confirm these API calls do not infer case, filename, sensor, or paper-record identity.

## No-write verification

- [ ] Record source-file modification times before running previews.
- [ ] Exercise all applicable preview views, then close the application.
- [ ] Confirm canonical input-file modification times are unchanged.
- [ ] Confirm `git status --short` is still empty.
- [ ] Confirm no new NPZ, CSV, or JSON result file appeared.
- [ ] Confirm no DataManager save/write prompt or action was observed.

Unchanged modification times are a practical smoke check, not complete proof
that no write occurred. Use filesystem auditing if formal verification is needed.

## Completion criteria

- [ ] All applicable checks above are complete.
- [ ] Any problem screenshots and console tracebacks are saved.
- [ ] Actual canonical dataset/run/event and channel names are recorded.
- [ ] Calibration and Q file paths are recorded.
- [ ] The canonical input and backup remain unmodified.
- [ ] The full automated suite still passes.
- [ ] `git diff --check` passes.
- [ ] The working tree is clean after the documentation commit.

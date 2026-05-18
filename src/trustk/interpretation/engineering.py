"""Engineering-practice aquifer-test interpretation methods.

These functions intentionally follow the semi-log workflows commonly used in
engineering reports: Cooper-Jacob late-time straight-line pumping analysis and
semi-log slug-test recovery analysis.
"""

from __future__ import annotations

import numpy as np

from trustk.interpretation.conventional import InterpretationFit


def estimate_transmissivity_cooper_jacob(
    *,
    times_s: np.ndarray,
    drawdown_m: np.ndarray,
    radius_m: float,
    pumping_rate_m3_s: float,
    storativity: float | None = None,
    min_points: int = 8,
) -> InterpretationFit:
    """Estimate transmissivity from the late-time Cooper-Jacob straight line."""

    if radius_m <= 0.0:
        raise ValueError("radius_m must be positive")
    if pumping_rate_m3_s <= 0.0:
        raise ValueError("pumping_rate_m3_s must be positive")
    times, drawdown = _positive_finite_pairs(times_s, drawdown_m)
    if len(times) < max(3, min_points):
        raise ValueError("Cooper-Jacob interpretation requires enough positive drawdown points")

    order = np.argsort(times)
    times = times[order]
    drawdown = drawdown[order]
    x = np.log10(times)
    min_points = max(3, int(min_points))
    candidate = None
    for candidate_min_points in range(min_points, 2, -1):
        try:
            candidate = _best_straight_line_window(
                x,
                drawdown,
                min_points=candidate_min_points,
                slope_sign=1.0,
                require_positive_fitted=True,
            )
            break
        except ValueError:
            continue
    if candidate is None:
        if len(x) < 2:
            raise ValueError("no valid Cooper-Jacob straight-line fitting window found")
        slope, intercept = np.polyfit(x[-2:], drawdown[-2:], 1)
        fitted_endpoint = intercept + slope * x[-2:]
        if slope <= 0.0 or np.any(fitted_endpoint <= 0.0):
            raise ValueError("no valid Cooper-Jacob straight-line fitting window found")
        candidate = _Window(start=len(x) - 2, stop=len(x), r_squared=np.nan, rmse=0.0)
    fit_x = x[candidate.start : candidate.stop]
    fit_y = drawdown[candidate.start : candidate.stop]

    slope, intercept = np.polyfit(fit_x, fit_y, 1)
    if slope <= 0.0:
        raise ValueError("Cooper-Jacob straight-line slope must be positive")
    transmissivity = 2.3 * pumping_rate_m3_s / (4.0 * np.pi * slope)
    fitted = intercept + slope * fit_x
    rmse_log = _rmse_log_positive(fit_y, fitted)

    inferred_storativity = None
    if storativity is not None and storativity > 0.0:
        log_t0 = -intercept / slope
        t0 = float(10.0**log_t0)
        if np.isfinite(t0) and t0 > 0.0:
            inferred_storativity = float(2.25 * transmissivity * t0 / radius_m**2)

    return InterpretationFit(
        transmissivity_m2_s=float(transmissivity),
        rmse_log_response=float(rmse_log),
        fit_time_min_s=float(times[candidate.start]),
        fit_time_max_s=float(times[candidate.stop - 1]),
        fit_point_count=int(candidate.stop - candidate.start),
        intercept=float(intercept),
        slope=float(slope),
        r_squared=float(candidate.r_squared),
        storativity=inferred_storativity,
        method="Cooper-Jacob",
    )


def estimate_transmissivity_slug_semilog(
    *,
    times_s: np.ndarray,
    normalized_head: np.ndarray,
    well_storage_m2: float,
    well_radius_m: float,
    outer_radius_m: float,
    min_points: int = 8,
) -> InterpretationFit:
    """Estimate transmissivity from a straight-line segment of log recovery."""

    if well_storage_m2 <= 0.0:
        raise ValueError("well_storage_m2 must be positive")
    if well_radius_m <= 0.0 or outer_radius_m <= well_radius_m:
        raise ValueError("outer_radius_m must be greater than well_radius_m")

    times = np.asarray(times_s, dtype=float)
    head = np.asarray(normalized_head, dtype=float)
    if times.shape != head.shape:
        raise ValueError("times_s and normalized_head must have the same shape")
    mask = np.isfinite(times) & np.isfinite(head) & (times > 0.0) & (head > 0.03) & (head < 0.90)
    if np.sum(mask) < max(3, min_points):
        mask = np.isfinite(times) & np.isfinite(head) & (times > 0.0) & (head > 1.0e-4) & (head < 0.98)
    if np.sum(mask) < max(3, min_points):
        mask = np.isfinite(times) & np.isfinite(head) & (times > 0.0) & (head > 1.0e-8)
    if np.sum(mask) < max(3, min_points):
        raise ValueError("slug semi-log interpretation requires enough usable recovery points")

    fit_times_all = times[mask]
    fit_log_head_all = np.log(head[mask])
    order = np.argsort(fit_times_all)
    fit_times_all = fit_times_all[order]
    fit_log_head_all = fit_log_head_all[order]
    min_points = max(3, int(min_points))
    candidate = _best_straight_line_window(
        fit_times_all,
        fit_log_head_all,
        min_points=min_points,
        slope_sign=-1.0,
        require_positive_fitted=False,
    )
    fit_times = fit_times_all[candidate.start : candidate.stop]
    fit_log_head = fit_log_head_all[candidate.start : candidate.stop]
    slope, intercept = np.polyfit(fit_times, fit_log_head, 1)
    if slope >= 0.0:
        raise ValueError("slug semi-log recovery slope must be negative")

    decay_rate = -float(slope)
    transmissivity = decay_rate * well_storage_m2 * np.log(outer_radius_m / well_radius_m) / (2.0 * np.pi)
    fitted = intercept + slope * fit_times
    rmse = float(np.sqrt(np.mean((fitted - fit_log_head) ** 2)))
    return InterpretationFit(
        transmissivity_m2_s=float(transmissivity),
        rmse_log_response=rmse,
        fit_time_min_s=float(fit_times[0]),
        fit_time_max_s=float(fit_times[-1]),
        fit_point_count=int(len(fit_times)),
        intercept=float(intercept),
        slope=float(slope),
        r_squared=float(candidate.r_squared),
        storativity=None,
        method="semi-log slug",
    )


class _Window:
    def __init__(self, start: int, stop: int, r_squared: float, rmse: float):
        self.start = start
        self.stop = stop
        self.r_squared = r_squared
        self.rmse = rmse


def _best_straight_line_window(
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_points: int,
    slope_sign: float,
    require_positive_fitted: bool,
) -> _Window:
    """Select the most linear contiguous semi-log window with a late-window bias."""

    n = len(x)
    best: tuple[float, _Window] | None = None
    min_fraction = max(min_points, int(np.ceil(0.30 * n)))
    for start in range(0, n - min_points + 1):
        for stop in range(start + min_fraction, n + 1):
            if stop - start < min_points:
                continue
            xs = x[start:stop]
            ys = y[start:stop]
            slope, intercept = np.polyfit(xs, ys, 1)
            if slope_sign * slope <= 0.0:
                continue
            fitted = intercept + slope * xs
            if require_positive_fitted and np.any(fitted <= 0.0):
                continue
            residual = ys - fitted
            ss_res = float(np.sum(residual**2))
            ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
            r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
            rmse = float(np.sqrt(np.mean(residual**2)))
            late_score = start / max(n - 1, 1)
            length_score = (stop - start) / n
            score = r2 + 0.012 * late_score + 0.004 * length_score - 0.002 * rmse
            window = _Window(start=start, stop=stop, r_squared=r2, rmse=rmse)
            if best is None or score > best[0]:
                best = (score, window)
    if best is None:
        raise ValueError("no valid straight-line fitting window found")
    return best[1]


def _positive_finite_pairs(times_s: np.ndarray, response: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times_s, dtype=float)
    values = np.asarray(response, dtype=float)
    if times.shape != values.shape:
        raise ValueError("times and response arrays must have the same shape")
    mask = np.isfinite(times) & np.isfinite(values) & (times > 0.0) & (values > 0.0)
    return times[mask], values[mask]


def _rmse_log_positive(observed: np.ndarray, fitted: np.ndarray) -> float:
    floor = np.finfo(float).tiny
    observed = np.maximum(np.asarray(observed, dtype=float), floor)
    fitted = np.maximum(np.asarray(fitted, dtype=float), floor)
    return float(np.sqrt(np.mean((np.log(observed) - np.log(fitted)) ** 2)))

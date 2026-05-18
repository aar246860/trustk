"""Conventional interpretation formulas used as TRUST-K method outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from trustk.analytical.theis import theis_drawdown


@dataclass(frozen=True)
class InterpretationFit:
    """Compact fit diagnostics for a conventional aquifer-test interpretation."""

    transmissivity_m2_s: float
    rmse_log_response: float
    fit_time_min_s: float
    fit_time_max_s: float
    fit_point_count: int
    intercept: float | None = None
    slope: float | None = None
    r_squared: float | None = None
    storativity: float | None = None
    method: str | None = None


def estimate_transmissivity_theis(
    *,
    times_s: np.ndarray,
    drawdown_m: np.ndarray,
    radius_m: float,
    storativity: float,
    pumping_rate_m3_s: float,
    log_transmissivity_bounds: tuple[float, float] = (-30.0, 5.0),
) -> InterpretationFit:
    """Fit transmissivity by matching a Theis curve with fixed storativity."""

    times, response = _positive_finite_pairs(times_s, drawdown_m)
    if len(times) < 3:
        raise ValueError("Theis interpretation requires at least three positive drawdown points")

    log_obs = np.log(response)

    def objective(log_t: float) -> float:
        transmissivity = float(np.exp(log_t))
        simulated = theis_drawdown(
            radius=radius_m,
            time=times,
            transmissivity=transmissivity,
            storativity=storativity,
            pumping_rate=pumping_rate_m3_s,
        )
        simulated = np.maximum(simulated, np.finfo(float).tiny)
        return float(np.mean((np.log(simulated) - log_obs) ** 2))

    fit = minimize_scalar(objective, bounds=log_transmissivity_bounds, method="bounded")
    if not fit.success:
        raise RuntimeError(f"Theis interpretation did not converge: {fit.message}")
    transmissivity = float(np.exp(fit.x))
    return InterpretationFit(
        transmissivity_m2_s=transmissivity,
        rmse_log_response=float(np.sqrt(objective(float(fit.x)))),
        fit_time_min_s=float(times[0]),
        fit_time_max_s=float(times[-1]),
        fit_point_count=int(len(times)),
    )


def estimate_transmissivity_slug_quasi_steady(
    *,
    times_s: np.ndarray,
    normalized_head: np.ndarray,
    well_storage_m2: float,
    well_radius_m: float,
    outer_radius_m: float,
) -> InterpretationFit:
    """Estimate transmissivity from the quasi-steady exponential slug limit."""

    if well_storage_m2 <= 0.0:
        raise ValueError("well_storage_m2 must be positive")
    if well_radius_m <= 0.0 or outer_radius_m <= well_radius_m:
        raise ValueError("outer_radius_m must be greater than well_radius_m")

    times = np.asarray(times_s, dtype=float)
    head = np.asarray(normalized_head, dtype=float)
    if times.shape != head.shape:
        raise ValueError("times_s and normalized_head must have the same shape")

    mask = np.isfinite(times) & np.isfinite(head) & (times > 0.0) & (head > 1.0e-8) & (head < 0.999999)
    if np.sum(mask) < 3:
        mask = np.isfinite(times) & np.isfinite(head) & (times > 0.0) & (head > 1.0e-8)
    if np.sum(mask) < 3:
        raise ValueError("slug interpretation requires at least three positive recovery points")

    fit_times = times[mask]
    fit_log_head = np.log(head[mask])
    slope, intercept = np.polyfit(fit_times, fit_log_head, 1)
    decay_rate = float(-slope)
    if decay_rate <= 0.0:
        endpoint_rate = (fit_log_head[0] - fit_log_head[-1]) / (fit_times[-1] - fit_times[0])
        decay_rate = max(float(endpoint_rate), 1.0e-30)

    transmissivity = decay_rate * well_storage_m2 * np.log(outer_radius_m / well_radius_m) / (2.0 * np.pi)
    fitted = intercept - decay_rate * fit_times
    rmse = float(np.sqrt(np.mean((fitted - fit_log_head) ** 2)))
    return InterpretationFit(
        transmissivity_m2_s=float(transmissivity),
        rmse_log_response=rmse,
        fit_time_min_s=float(fit_times[0]),
        fit_time_max_s=float(fit_times[-1]),
        fit_point_count=int(len(fit_times)),
    )


def _positive_finite_pairs(times_s: np.ndarray, response: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times_s, dtype=float)
    values = np.asarray(response, dtype=float)
    if times.shape != values.shape:
        raise ValueError("times and response arrays must have the same shape")
    mask = np.isfinite(times) & np.isfinite(values) & (times > 0.0) & (values > 0.0)
    return times[mask], values[mask]

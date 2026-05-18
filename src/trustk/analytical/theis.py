"""Theis analytical solution for confined-aquifer pumping tests."""

from __future__ import annotations

import numpy as np
from scipy.special import exp1


def theis_drawdown(
    radius: np.ndarray | float,
    time: np.ndarray | float,
    transmissivity: float,
    storativity: float,
    pumping_rate: float,
) -> np.ndarray:
    """Compute Theis drawdown for a constant-rate pumping test.

    Parameters use consistent units. Drawdown is positive for pumping.
    """

    if transmissivity <= 0:
        raise ValueError("transmissivity must be positive")
    if storativity <= 0:
        raise ValueError("storativity must be positive")
    if pumping_rate <= 0:
        raise ValueError("pumping_rate must be positive")

    r = np.asarray(radius, dtype=float)
    t = np.asarray(time, dtype=float)
    if np.any(r <= 0):
        raise ValueError("radius values must be positive")
    if np.any(t <= 0):
        raise ValueError("time values must be positive")

    u = (r**2 * storativity) / (4.0 * transmissivity * t)
    return pumping_rate / (4.0 * np.pi * transmissivity) * exp1(u)


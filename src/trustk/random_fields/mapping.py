"""Mapping utilities between Cartesian fields and TRUST-K polar meshes."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from trustk.mesh.polar_mesh import PolarMesh
from trustk.random_fields.gaussian_field import GaussianField2D


def map_cartesian_field_to_polar(field: GaussianField2D, mesh: PolarMesh) -> np.ndarray:
    """Interpolate a Cartesian ``ln K`` field to polar cell centers."""

    interpolator = RegularGridInterpolator(
        (field.y, field.x),
        field.logk,
        bounds_error=False,
        fill_value=np.nan,
    )
    points = np.column_stack((mesh.y_centers.ravel(), mesh.x_centers.ravel()))
    mapped = interpolator(points).reshape(mesh.shape)

    if np.any(~np.isfinite(mapped)):
        x_min, x_max, y_min, y_max = field.extent
        raise ValueError(
            "Cartesian field does not cover the polar mesh domain: "
            f"field extent=({x_min:.3g}, {x_max:.3g}, {y_min:.3g}, {y_max:.3g})"
        )

    return mapped


def radial_symmetry_score(values: np.ndarray) -> float:
    """Return a dimensionless measure of angular variability in a polar field.

    A perfectly radial-only field has score zero. A two-dimensional field mapped
    to the polar mesh should retain substantial angular variability.
    """

    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError("values must be a two-dimensional polar array")
    overall = float(np.std(arr))
    if overall <= 0.0:
        return 0.0
    theta_std_by_radius = np.std(arr, axis=1)
    return float(np.mean(theta_std_by_radius) / overall)

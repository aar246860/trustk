"""Area-support targets for synthetic TRUST-K truth fields."""

from __future__ import annotations

import numpy as np

from trustk.mesh.polar_mesh import PolarMesh


def area_weighted_geometric_mean_k(logk: np.ndarray, area: np.ndarray, mask: np.ndarray) -> float:
    """Return the area-weighted geometric mean conductivity inside a support mask."""

    y = np.asarray(logk, dtype=float)
    a = np.asarray(area, dtype=float)
    m = np.asarray(mask, dtype=bool)
    if y.shape != a.shape or y.shape != m.shape:
        raise ValueError("logk, area, and mask must have the same shape")
    if np.any(~np.isfinite(y)) or np.any(~np.isfinite(a)):
        raise ValueError("logk and area must be finite")
    if np.any(a <= 0.0):
        raise ValueError("area values must be positive")
    if not np.any(m):
        raise ValueError("support mask must contain at least one cell")

    weights = a[m]
    return float(np.exp(np.sum(weights * y[m]) / np.sum(weights)))


def support_targets_from_mapped_logk(
    *,
    mapped_logk: np.ndarray,
    mesh: PolarMesh,
    pumping_radius_m: float,
    slug_radius_m: float,
) -> dict[str, float | int]:
    """Compute method-specific Target A support means for pumping and slug tests."""

    y = np.asarray(mapped_logk, dtype=float)
    if y.shape != mesh.shape:
        raise ValueError(f"mapped_logk shape must be {mesh.shape}")
    if pumping_radius_m <= 0.0 or slug_radius_m <= 0.0:
        raise ValueError("support radii must be positive")

    radius = np.sqrt(mesh.x_centers**2 + mesh.y_centers**2)
    pumping_mask = _radius_mask(radius, pumping_radius_m)
    slug_mask = _radius_mask(radius, slug_radius_m)
    pumping_area = float(np.sum(mesh.area[pumping_mask]))
    slug_area = float(np.sum(mesh.area[slug_mask]))

    return {
        "K_star_pumping_m_s": area_weighted_geometric_mean_k(y, mesh.area, pumping_mask),
        "K_star_slug_m_s": area_weighted_geometric_mean_k(y, mesh.area, slug_mask),
        "pumping_support_radius_m": float(pumping_radius_m),
        "slug_support_radius_m": float(slug_radius_m),
        "pumping_support_area_m2": pumping_area,
        "slug_support_area_m2": slug_area,
        "pumping_cell_count": int(np.sum(pumping_mask)),
        "slug_cell_count": int(np.sum(slug_mask)),
    }


def _radius_mask(radius: np.ndarray, support_radius: float) -> np.ndarray:
    mask = radius <= support_radius
    if np.any(mask):
        return mask
    closest = np.unravel_index(int(np.argmin(radius)), radius.shape)
    mask = np.zeros_like(radius, dtype=bool)
    mask[closest] = True
    return mask

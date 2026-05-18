"""2D horizontal polar mesh for TRUST-K."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PolarMesh:
    """Cell-centered polar mesh with log-spaced radius and uniform angle."""

    r_faces: np.ndarray
    theta_faces: np.ndarray
    r_centers: np.ndarray
    theta_centers: np.ndarray
    area: np.ndarray
    x_centers: np.ndarray
    y_centers: np.ndarray

    @property
    def n_r(self) -> int:
        return len(self.r_centers)

    @property
    def n_theta(self) -> int:
        return len(self.theta_centers)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_r, self.n_theta)


def make_log_polar_mesh(r_w: float, r_max: float, n_r: int, n_theta: int) -> PolarMesh:
    """Create a log-radial, uniform-theta polar mesh.

    Parameters are in consistent length units. The returned arrays use shape
    `(n_r, n_theta)` for cell-centered quantities.
    """

    if r_w <= 0:
        raise ValueError("r_w must be positive")
    if r_max <= r_w:
        raise ValueError("r_max must be greater than r_w")
    if n_r < 1 or n_theta < 3:
        raise ValueError("n_r must be >= 1 and n_theta must be >= 3")

    r_faces = r_w * (r_max / r_w) ** (np.arange(n_r + 1) / n_r)
    theta_faces = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)
    r_centers = np.sqrt(r_faces[:-1] * r_faces[1:])
    theta_centers = 0.5 * (theta_faces[:-1] + theta_faces[1:])

    dr2 = r_faces[1:] ** 2 - r_faces[:-1] ** 2
    dtheta = theta_faces[1] - theta_faces[0]
    area_1d = 0.5 * dr2 * dtheta
    area = np.repeat(area_1d[:, None], n_theta, axis=1)

    rr, tt = np.meshgrid(r_centers, theta_centers, indexing="ij")
    x_centers = rr * np.cos(tt)
    y_centers = rr * np.sin(tt)

    return PolarMesh(
        r_faces=r_faces,
        theta_faces=theta_faces,
        r_centers=r_centers,
        theta_centers=theta_centers,
        area=area,
        x_centers=x_centers,
        y_centers=y_centers,
    )


def total_area(mesh: PolarMesh) -> float:
    """Return total active mesh area."""

    return float(np.sum(mesh.area))


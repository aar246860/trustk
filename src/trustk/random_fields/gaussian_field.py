"""Gaussian random fields for synthetic TRUST-K truth cases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianField2D:
    """Cartesian two-dimensional Gaussian field."""

    x: np.ndarray
    y: np.ndarray
    logk: np.ndarray
    mean_logk: float
    sigma_logk: float
    corr_len_x: float
    corr_len_y: float
    orientation_rad: float = 0.0
    seed: int | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.logk.shape

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (float(self.x[0]), float(self.x[-1]), float(self.y[0]), float(self.y[-1]))


def generate_gaussian_logk_field(
    *,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    mean_logk: float,
    sigma_logk: float,
    corr_len_x: float,
    corr_len_y: float,
    orientation_rad: float = 0.0,
    seed: int | None = None,
) -> GaussianField2D:
    """Generate a stationary Cartesian Gaussian random field for ``ln K``.

    The field is filtered in Fourier space with an anisotropic Gaussian kernel
    and then normalized to exactly match the requested sample mean and standard
    deviation. Coordinates are centered at the pumping well location `(0, 0)`.
    """

    _validate_field_inputs(nx, ny, dx, dy, sigma_logk, corr_len_x, corr_len_y)

    rng = np.random.default_rng(seed)
    white = rng.normal(size=(ny, nx))

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kkx, kky = np.meshgrid(kx, ky, indexing="xy")
    cos_phi = np.cos(orientation_rad)
    sin_phi = np.sin(orientation_rad)
    k_major = kkx * cos_phi + kky * sin_phi
    k_minor = -kkx * sin_phi + kky * cos_phi
    spectral_filter = np.exp(-0.25 * ((k_major * corr_len_x) ** 2 + (k_minor * corr_len_y) ** 2))

    smooth = np.fft.ifft2(np.fft.fft2(white) * spectral_filter).real
    smooth -= float(np.mean(smooth))
    smooth_std = float(np.std(smooth))
    if smooth_std <= 0.0:
        raise ValueError("Filtered random field has zero variance")

    logk = mean_logk + sigma_logk * smooth / smooth_std
    x = (np.arange(nx, dtype=float) - 0.5 * (nx - 1)) * dx
    y = (np.arange(ny, dtype=float) - 0.5 * (ny - 1)) * dy

    return GaussianField2D(
        x=x,
        y=y,
        logk=logk,
        mean_logk=mean_logk,
        sigma_logk=sigma_logk,
        corr_len_x=corr_len_x,
        corr_len_y=corr_len_y,
        orientation_rad=orientation_rad,
        seed=seed,
    )


def _validate_field_inputs(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    sigma_logk: float,
    corr_len_x: float,
    corr_len_y: float,
) -> None:
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must both be at least 2")
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("dx and dy must be positive")
    if sigma_logk <= 0.0:
        raise ValueError("sigma_logk must be positive")
    if corr_len_x <= 0.0 or corr_len_y <= 0.0:
        raise ValueError("corr_len_x and corr_len_y must be positive")

"""Linear-Gaussian spatial assimilation for TRUST-K soft observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SpatialPosterior:
    """Posterior latent-field summary on a fixed grid."""

    coords: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    approach: str


def squared_exponential_covariance(
    coords: np.ndarray,
    *,
    variance: float,
    corr_len: float,
    nugget: float = 1.0e-8,
) -> np.ndarray:
    """Build a squared-exponential covariance matrix on Cartesian coordinates."""

    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be an n by 2 array")
    if variance <= 0.0 or corr_len <= 0.0:
        raise ValueError("variance and corr_len must be positive")
    diff = coords[:, None, :] - coords[None, :, :]
    dist2 = np.sum(diff**2, axis=2)
    cov = variance * np.exp(-0.5 * dist2 / corr_len**2)
    cov += nugget * np.eye(len(coords))
    return cov


def build_support_operator(coords: np.ndarray, observations: pd.DataFrame) -> np.ndarray:
    """Build row-normalized support operators for observation support radii."""

    coords = np.asarray(coords, dtype=float)
    required = {"x", "y", "support_radius"}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"observations missing required columns: {sorted(missing)}")
    rows = []
    for _, obs in observations.iterrows():
        center = np.array([float(obs["x"]), float(obs["y"])])
        radius = float(obs["support_radius"])
        distances = np.sqrt(np.sum((coords - center) ** 2, axis=1))
        mask = distances <= radius
        weights = np.zeros(len(coords), dtype=float)
        if np.any(mask):
            weights[mask] = 1.0 / float(np.sum(mask))
        else:
            weights[int(np.argmin(distances))] = 1.0
        rows.append(weights)
    return np.vstack(rows)


def assimilate_linear_gaussian(
    *,
    coords: np.ndarray,
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    observations: pd.DataFrame,
    correction_column: str | None,
    sigma_column: str | None,
    default_sigma: float = 0.10,
    approach: str = "posterior",
) -> SpatialPosterior:
    """Assimilate method-derived K observations into a latent log-K field."""

    required = {"actual_log_K_hat", "x", "y", "support_radius"}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"observations missing required columns: {sorted(missing)}")
    coords = np.asarray(coords, dtype=float)
    prior_mean = np.asarray(prior_mean, dtype=float)
    prior_cov = np.asarray(prior_cov, dtype=float)
    if prior_cov.shape != (len(coords), len(coords)):
        raise ValueError("prior_cov shape does not match coords")
    if prior_mean.shape != (len(coords),):
        raise ValueError("prior_mean shape does not match coords")

    correction = np.zeros(len(observations), dtype=float)
    if correction_column is not None:
        correction = observations[correction_column].to_numpy(dtype=float)
    sigma = np.full(len(observations), float(default_sigma), dtype=float)
    if sigma_column is not None:
        sigma = observations[sigma_column].to_numpy(dtype=float)
    sigma = np.maximum(sigma, 1.0e-6)
    z = observations["actual_log_K_hat"].to_numpy(dtype=float) - correction
    h = build_support_operator(coords, observations)

    innovation_cov = h @ prior_cov @ h.T + np.diag(sigma**2)
    gain = prior_cov @ h.T @ np.linalg.solve(innovation_cov, np.eye(len(observations)))
    posterior_mean = prior_mean + gain @ (z - h @ prior_mean)
    posterior_cov = prior_cov - gain @ h @ prior_cov
    posterior_var = np.maximum(np.diag(posterior_cov), 0.0)
    return SpatialPosterior(coords=coords, mean=posterior_mean, variance=posterior_var, approach=approach)


def evaluate_spatial_posterior(posterior: SpatialPosterior, truth_logk: np.ndarray) -> dict[str, float]:
    """Evaluate posterior mean error and marginal interval coverage."""

    truth = np.asarray(truth_logk, dtype=float)
    error = posterior.mean - truth
    sigma = np.sqrt(np.maximum(posterior.variance, 1.0e-12))
    return {
        "approach": posterior.approach,
        "rmse_logk": float(np.sqrt(np.mean(error**2))),
        "bias_logk": float(np.mean(error)),
        "mae_logk": float(np.mean(np.abs(error))),
        "coverage_95": float(np.mean(np.abs(error) <= 1.96 * sigma)),
        "mean_sd": float(np.mean(sigma)),
    }

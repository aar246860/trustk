"""Dimensionless case registry for TRUST-K synthetic experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import qmc


@dataclass(frozen=True)
class DimensionlessDesignRanges:
    """Sampling ranges for the pilot dimensionless TRUST-K design."""

    sigma_y2: tuple[float, float] = (0.05, 2.25)
    lambda1_over_ri: tuple[float, float] = (0.05, 2.0)
    lambda2_over_lambda1: tuple[float, float] = (0.10, 1.0)
    phi_lambda: tuple[float, float] = (0.0, np.pi)
    r_skin_over_rw: tuple[float, float] = (1.0, 5.0)
    k_skin_over_k0: tuple[float, float] = (0.01, 10.0)
    rmax_over_ri: tuple[float, float] = (2.0, 8.0)
    r_obs_over_ri_p: tuple[float, float] = (0.05, 1.0)
    td_p_max: tuple[float, float] = (1.0e2, 1.0e5)
    cd_w: tuple[float, float] = (1.0e2, 1.0e5)
    rc_over_rw: tuple[float, float] = (0.3, 1.0)
    td_s_max: tuple[float, float] = (1.0e1, 1.0e4)


LINEAR_COLUMNS = {
    "sigma_Y2",
    "lambda2_over_lambda1",
    "phi_lambda_rad",
    "r_skin_over_rw",
    "r_obs_over_RI_P",
    "r_c_over_r_w",
}

LOG_COLUMNS = {
    "lambda1_over_RI",
    "K_skin_over_K0",
    "Rmax_over_RI",
    "tD_P_max",
    "C_D_w",
    "tD_S_max",
}


def hydraulic_scales(
    *,
    mean_logk: float,
    aquifer_thickness: float,
    storativity: float,
    well_radius: float,
    pumping_rate: float,
    well_storage: float | None = None,
    casing_radius: float | None = None,
) -> dict[str, float]:
    """Compute common diffusion and method-specific normalization scales."""

    if aquifer_thickness <= 0.0:
        raise ValueError("aquifer_thickness must be positive")
    if storativity <= 0.0:
        raise ValueError("storativity must be positive")
    if well_radius <= 0.0:
        raise ValueError("well_radius must be positive")
    if pumping_rate <= 0.0:
        raise ValueError("pumping_rate must be positive")

    k0 = float(np.exp(mean_logk))
    t0 = k0 * aquifer_thickness
    diffusion_time = storativity * well_radius**2 / t0
    pumping_head_scale = pumping_rate / (4.0 * np.pi * t0)
    out = {
        "K0_m_s": k0,
        "T0_m2_s": t0,
        "L0_m": float(well_radius),
        "t0_s": float(diffusion_time),
        "H_P_m": float(pumping_head_scale),
    }
    if well_storage is not None:
        if well_storage <= 0.0:
            raise ValueError("well_storage must be positive")
        out["C_D_w"] = float(well_storage / (storativity * well_radius**2))
        out["C_D_w_pi"] = float(well_storage / (storativity * np.pi * well_radius**2))
    if casing_radius is not None:
        if casing_radius <= 0.0:
            raise ValueError("casing_radius must be positive")
        out["r_c_over_r_w"] = float(casing_radius / well_radius)
    return out


def build_dimensionless_case_table(
    *,
    n_cases: int,
    seed: int,
    ranges: DimensionlessDesignRanges | None = None,
) -> pd.DataFrame:
    """Build a reproducible pilot design table in TRUST-K Pi space."""

    if n_cases < 2:
        raise ValueError("n_cases must be at least 2")
    ranges = ranges or DimensionlessDesignRanges()
    names = [
        "sigma_Y2",
        "lambda1_over_RI",
        "lambda2_over_lambda1",
        "phi_lambda_rad",
        "r_skin_over_rw",
        "K_skin_over_K0",
        "Rmax_over_RI",
        "r_obs_over_RI_P",
        "tD_P_max",
        "C_D_w",
        "r_c_over_r_w",
        "tD_S_max",
    ]
    raw = qmc.LatinHypercube(d=len(names), seed=seed).random(n_cases)
    sampled = pd.DataFrame(index=np.arange(n_cases))
    range_map = {
        "sigma_Y2": ranges.sigma_y2,
        "lambda1_over_RI": ranges.lambda1_over_ri,
        "lambda2_over_lambda1": ranges.lambda2_over_lambda1,
        "phi_lambda_rad": ranges.phi_lambda,
        "r_skin_over_rw": ranges.r_skin_over_rw,
        "K_skin_over_K0": ranges.k_skin_over_k0,
        "Rmax_over_RI": ranges.rmax_over_ri,
        "r_obs_over_RI_P": ranges.r_obs_over_ri_p,
        "tD_P_max": ranges.td_p_max,
        "C_D_w": ranges.cd_w,
        "r_c_over_r_w": ranges.rc_over_rw,
        "tD_S_max": ranges.td_s_max,
    }

    for idx, name in enumerate(names):
        lo, hi = range_map[name]
        _validate_range(name, lo, hi)
        if name in LOG_COLUMNS:
            sampled[name] = np.exp(np.log(lo) + raw[:, idx] * (np.log(hi) - np.log(lo)))
        else:
            sampled[name] = lo + raw[:, idx] * (hi - lo)

    sampled.insert(0, "case_id", [f"case_{i:04d}" for i in range(n_cases)])
    sampled.insert(1, "random_seed", seed + np.arange(n_cases, dtype=int))
    sampled["lambda2_over_RI"] = sampled["lambda1_over_RI"] * sampled["lambda2_over_lambda1"]
    sampled["pumping_interpretation_method"] = "Theis"
    sampled["slug_interpretation_method"] = "Bouwer-Rice"

    ordered = [
        "case_id",
        "random_seed",
        "sigma_Y2",
        "lambda1_over_RI",
        "lambda2_over_lambda1",
        "lambda2_over_RI",
        "phi_lambda_rad",
        "r_skin_over_rw",
        "K_skin_over_K0",
        "Rmax_over_RI",
        "r_obs_over_RI_P",
        "tD_P_max",
        "C_D_w",
        "r_c_over_r_w",
        "tD_S_max",
        "pumping_interpretation_method",
        "slug_interpretation_method",
    ]
    return sampled[ordered]


def summarize_dimensionless_registry(table: pd.DataFrame) -> dict:
    """Return coverage and rule checks for a dimensionless case table."""

    required_columns = {
        "case_id",
        "random_seed",
        "sigma_Y2",
        "lambda1_over_RI",
        "lambda2_over_lambda1",
        "lambda2_over_RI",
        "phi_lambda_rad",
        "r_skin_over_rw",
        "K_skin_over_K0",
        "Rmax_over_RI",
        "r_obs_over_RI_P",
        "tD_P_max",
        "C_D_w",
        "r_c_over_r_w",
        "tD_S_max",
        "pumping_interpretation_method",
        "slug_interpretation_method",
    }
    missing = sorted(required_columns.difference(table.columns))
    no_dimensional_stress_controls = not {"Q", "H0", "pumping_rate", "initial_slug_displacement"}.intersection(
        table.columns
    )
    finite_numeric = bool(np.isfinite(table.select_dtypes(include=[np.number]).to_numpy()).all())
    separated_methods = bool(
        "pumping_interpretation_method" in table.columns and "slug_interpretation_method" in table.columns
    )
    return {
        "n_cases": int(len(table)),
        "checks": {
            "required_columns": {"pass": len(missing) == 0, "missing": missing},
            "no_dimensional_stress_controls": {"pass": bool(no_dimensional_stress_controls)},
            "finite_numeric_values": {"pass": finite_numeric},
            "separated_method_controls": {"pass": separated_methods},
        },
        "ranges": _numeric_ranges(table),
    }


def _numeric_ranges(table: pd.DataFrame) -> dict[str, dict[str, float]]:
    ranges = {}
    for column in table.select_dtypes(include=[np.number]).columns:
        ranges[column] = {
            "min": float(table[column].min()),
            "max": float(table[column].max()),
            "median": float(table[column].median()),
        }
    return ranges


def _validate_range(name: str, lo: float, hi: float) -> None:
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError(f"{name} range must be finite")
    if lo <= 0.0 and name in LOG_COLUMNS:
        raise ValueError(f"{name} log-sampling range must be positive")
    if hi <= lo:
        raise ValueError(f"{name} range upper bound must exceed lower bound")

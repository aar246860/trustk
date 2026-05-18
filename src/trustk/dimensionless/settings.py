"""Convert TRUST-K dimensionless cases into dimensional solver settings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DimensionalBaseConfig:
    """Fixed dimensional scales used to run the synthetic pilot simulations."""

    mean_logk: float = -11.0
    aquifer_thickness_m: float = 20.0
    storativity: float = 2.0e-4
    well_radius_m: float = 0.1
    pumping_rate_m3_s: float = 1.0e-3
    initial_slug_head_m: float = 1.0
    cells_per_minor_correlation: float = 4.0
    cartesian_domain_margin: float = 1.10
    max_cartesian_n: int = 1024
    mesh_n_r: int = 96
    mesh_n_theta: int = 48
    pumping_n_times: int = 32
    slug_n_times: int = 48


REQUIRED_REGISTRY_COLUMNS = {
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


def convert_case_registry_to_solver_settings(
    registry: pd.DataFrame,
    base: DimensionalBaseConfig | None = None,
) -> pd.DataFrame:
    """Convert dimensionless Pi rows into dimensional settings for the solvers."""

    base = base or DimensionalBaseConfig()
    _validate_base_config(base)
    missing = sorted(REQUIRED_REGISTRY_COLUMNS.difference(registry.columns))
    if missing:
        raise ValueError(f"registry is missing required columns: {missing}")

    table = registry.copy()
    k0 = float(np.exp(base.mean_logk))
    t0 = float(k0 * base.aquifer_thickness_m)
    diffusion_time = float(base.storativity * base.well_radius_m**2 / t0)

    ri_p = base.well_radius_m * np.sqrt(table["tD_P_max"].to_numpy(dtype=float))
    ri_s = base.well_radius_m * np.sqrt(table["tD_S_max"].to_numpy(dtype=float))
    ri_common = np.maximum(ri_p, ri_s)

    out = pd.DataFrame(
        {
            "case_id": table["case_id"],
            "random_seed": table["random_seed"].astype(int),
            "K0_m_s": k0,
            "T0_m2_s": t0,
            "aquifer_thickness_m": base.aquifer_thickness_m,
            "storativity": base.storativity,
            "well_radius_m": base.well_radius_m,
            "t0_s": diffusion_time,
            "sigma_Y2": table["sigma_Y2"].to_numpy(dtype=float),
            "sigma_logk": np.sqrt(table["sigma_Y2"].to_numpy(dtype=float)),
            "RI_P_m": ri_p,
            "RI_S_m": ri_s,
            "RI_common_m": ri_common,
            "lambda1_m": table["lambda1_over_RI"].to_numpy(dtype=float) * ri_common,
            "lambda2_m": table["lambda2_over_RI"].to_numpy(dtype=float) * ri_common,
            "phi_lambda_rad": table["phi_lambda_rad"].to_numpy(dtype=float),
            "r_skin_m": table["r_skin_over_rw"].to_numpy(dtype=float) * base.well_radius_m,
            "K_skin_m_s": table["K_skin_over_K0"].to_numpy(dtype=float) * k0,
            "r_max_m": table["Rmax_over_RI"].to_numpy(dtype=float) * ri_common,
            "r_obs_m": table["r_obs_over_RI_P"].to_numpy(dtype=float) * ri_p,
            "pumping_time_max_s": table["tD_P_max"].to_numpy(dtype=float) * diffusion_time,
            "slug_time_max_s": table["tD_S_max"].to_numpy(dtype=float) * diffusion_time,
            "well_storage_m2": table["C_D_w"].to_numpy(dtype=float)
            * base.storativity
            * base.well_radius_m**2,
            "casing_radius_m": table["r_c_over_r_w"].to_numpy(dtype=float) * base.well_radius_m,
            "pumping_rate_m3_s": base.pumping_rate_m3_s,
            "initial_slug_head_m": base.initial_slug_head_m,
            "mesh_n_r": base.mesh_n_r,
            "mesh_n_theta": base.mesh_n_theta,
            "pumping_n_times": base.pumping_n_times,
            "slug_n_times": base.slug_n_times,
            "pumping_interpretation_method": table["pumping_interpretation_method"],
            "slug_interpretation_method": table["slug_interpretation_method"],
        }
    )

    out["cartesian_extent_m"] = base.cartesian_domain_margin * out["r_max_m"]
    out["cartesian_dx_target_m"] = out["lambda2_m"] / base.cells_per_minor_correlation
    out["cartesian_n_required"] = _odd_grid_size(
        np.ceil(2.0 * out["cartesian_extent_m"] / out["cartesian_dx_target_m"]).to_numpy(dtype=int) + 1
    )
    out["cartesian_cells_required"] = out["cartesian_n_required"] ** 2
    out["minor_corr_cells_at_required_dx"] = out["lambda2_m"] / out["cartesian_dx_target_m"]
    out["outer_boundary_to_influence"] = out["r_max_m"] / out["RI_common_m"]
    out["ready_for_pilot_solver"] = out["cartesian_n_required"] <= base.max_cartesian_n
    out["max_cartesian_n_limit"] = base.max_cartesian_n
    return out


def summarize_solver_settings(settings: pd.DataFrame) -> dict:
    """Summarize dimensional settings and readiness checks."""

    numeric = settings.select_dtypes(include=[np.number])
    finite = bool(np.isfinite(numeric.to_numpy()).all())
    positive_columns = [
        "K0_m_s",
        "T0_m2_s",
        "storativity",
        "well_radius_m",
        "t0_s",
        "RI_P_m",
        "RI_S_m",
        "RI_common_m",
        "lambda1_m",
        "lambda2_m",
        "r_max_m",
        "r_obs_m",
        "pumping_time_max_s",
        "slug_time_max_s",
        "well_storage_m2",
        "casing_radius_m",
    ]
    positive = bool((settings[positive_columns] > 0.0).all().all())
    boundary_ok = bool((settings["outer_boundary_to_influence"] >= 2.0).all())
    stress_recorded = bool({"pumping_rate_m3_s", "initial_slug_head_m"}.issubset(settings.columns))
    ready_count = int(settings["ready_for_pilot_solver"].sum()) if "ready_for_pilot_solver" in settings else 0
    return {
        "n_cases": int(len(settings)),
        "ready_case_count": ready_count,
        "checks": {
            "finite_solver_settings": {"pass": finite},
            "positive_times_and_lengths": {"pass": positive},
            "outer_boundary_exceeds_influence_radius": {"pass": boundary_ok},
            "fixed_stress_amplitudes_recorded": {"pass": stress_recorded},
            "at_least_one_ready_case": {"pass": ready_count > 0},
        },
        "ranges": _numeric_ranges(settings),
    }


def _odd_grid_size(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=int)
    arr = np.maximum(arr, 3)
    return arr + (arr % 2 == 0)


def _numeric_ranges(table: pd.DataFrame) -> dict[str, dict[str, float]]:
    ranges = {}
    for column in table.select_dtypes(include=[np.number, bool]).columns:
        values = table[column].astype(float)
        ranges[column] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(values.median()),
        }
    return ranges


def _validate_base_config(base: DimensionalBaseConfig) -> None:
    numeric_positive = {
        "aquifer_thickness_m": base.aquifer_thickness_m,
        "storativity": base.storativity,
        "well_radius_m": base.well_radius_m,
        "pumping_rate_m3_s": base.pumping_rate_m3_s,
        "initial_slug_head_m": base.initial_slug_head_m,
        "cells_per_minor_correlation": base.cells_per_minor_correlation,
        "cartesian_domain_margin": base.cartesian_domain_margin,
    }
    for name, value in numeric_positive.items():
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if base.max_cartesian_n < 3:
        raise ValueError("max_cartesian_n must be at least 3")
    if base.mesh_n_r < 3 or base.mesh_n_theta < 8:
        raise ValueError("mesh_n_r must be >= 3 and mesh_n_theta must be >= 8")
    if base.pumping_n_times < 3 or base.slug_n_times < 3:
        raise ValueError("pumping_n_times and slug_n_times must be at least 3")

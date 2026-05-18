import json

import numpy as np
import pandas as pd

from trustk.analytical.theis import theis_drawdown
from trustk.dimensionless.settings import DimensionalBaseConfig, convert_case_registry_to_solver_settings
from trustk.experiments.run_synthetic_population import run_synthetic_population
from trustk.interpretation.conventional import (
    estimate_transmissivity_slug_quasi_steady,
    estimate_transmissivity_theis,
)
from trustk.mesh.polar_mesh import make_log_polar_mesh
from trustk.targets.support_area import area_weighted_geometric_mean_k, support_targets_from_mapped_logk


def _small_ready_settings() -> pd.DataFrame:
    registry = pd.DataFrame(
        {
            "case_id": ["case_0000", "case_0001"],
            "random_seed": [101, 102],
            "sigma_Y2": [0.25, 0.5],
            "lambda1_over_RI": [0.8, 0.6],
            "lambda2_over_lambda1": [0.6, 0.5],
            "lambda2_over_RI": [0.48, 0.30],
            "phi_lambda_rad": [0.1, 1.0],
            "r_skin_over_rw": [1.0, 1.0],
            "K_skin_over_K0": [1.0, 1.0],
            "Rmax_over_RI": [3.0, 4.0],
            "r_obs_over_RI_P": [0.4, 0.5],
            "tD_P_max": [100.0, 300.0],
            "C_D_w": [100.0, 200.0],
            "r_c_over_r_w": [0.7, 0.8],
            "tD_S_max": [30.0, 80.0],
            "pumping_interpretation_method": ["Theis", "Theis"],
            "slug_interpretation_method": ["Bouwer-Rice", "Bouwer-Rice"],
        }
    )
    base = DimensionalBaseConfig(max_cartesian_n=256, mesh_n_r=24, mesh_n_theta=12, pumping_n_times=8, slug_n_times=8)
    settings = convert_case_registry_to_solver_settings(registry, base)
    settings["ready_for_pilot_solver"] = True
    settings["mesh_n_r"] = 18
    settings["mesh_n_theta"] = 8
    settings["pumping_n_times"] = 6
    settings["slug_n_times"] = 6
    return settings


def test_area_weighted_geometric_mean_matches_constant_field():
    area = np.array([[1.0, 2.0], [3.0, 4.0]])
    logk = np.full_like(area, np.log(2.5e-5))
    mask = np.array([[True, False], [True, True]])

    assert np.isclose(area_weighted_geometric_mean_k(logk, area, mask), 2.5e-5)


def test_support_targets_use_method_specific_supports():
    mesh = make_log_polar_mesh(r_w=0.1, r_max=100.0, n_r=20, n_theta=12)
    logk = np.log(1.0e-6) + 0.9 * np.log(mesh.x_centers**2 + mesh.y_centers**2 + 1.0)

    targets = support_targets_from_mapped_logk(
        mapped_logk=logk,
        mesh=mesh,
        pumping_radius_m=50.0,
        slug_radius_m=0.6,
    )

    assert targets["pumping_cell_count"] > targets["slug_cell_count"]
    assert targets["K_star_pumping_m_s"] > targets["K_star_slug_m_s"]
    assert targets["pumping_support_area_m2"] > targets["slug_support_area_m2"]


def test_conventional_interpreters_recover_homogeneous_reference():
    true_t = 2.0e-4
    storativity = 2.0e-4
    pumping_rate = 1.0e-3
    radius = 8.0
    times = np.geomspace(10.0, 100000.0, 60)
    drawdown = theis_drawdown(radius, times, true_t, storativity, pumping_rate)

    pump_fit = estimate_transmissivity_theis(
        times_s=times,
        drawdown_m=drawdown,
        radius_m=radius,
        storativity=storativity,
        pumping_rate_m3_s=pumping_rate,
    )

    well_storage = 0.015
    r_w = 0.1
    r_max = 60.0
    conductance = 2.0 * np.pi * true_t / np.log(r_max / r_w)
    normalized_head = np.exp(-(conductance / well_storage) * times)
    slug_fit = estimate_transmissivity_slug_quasi_steady(
        times_s=times,
        normalized_head=normalized_head,
        well_storage_m2=well_storage,
        well_radius_m=r_w,
        outer_radius_m=r_max,
    )

    assert np.isclose(pump_fit.transmissivity_m2_s, true_t, rtol=0.05)
    assert np.isclose(slug_fit.transmissivity_m2_s, true_t, rtol=0.05)
    assert pump_fit.rmse_log_response < 0.02
    assert slug_fit.rmse_log_response < 0.02


def test_synthetic_population_script_writes_targets_residuals_report_and_figure(tmp_path):
    settings_path = tmp_path / "solver_settings.csv"
    curves_path = tmp_path / "synthetic_population_curves.csv"
    residuals_path = tmp_path / "synthetic_population_residuals.csv"
    report_path = tmp_path / "synthetic_population.json"
    figure_prefix = tmp_path / "fig_support_residuals"
    _small_ready_settings().to_csv(settings_path, index=False)

    report = run_synthetic_population(
        settings_path=settings_path,
        curves_path=curves_path,
        residuals_path=residuals_path,
        report_path=report_path,
        figure_prefix=figure_prefix,
        max_cases=2,
        cartesian_n_cap=129,
        mesh_n_r=18,
        mesh_n_theta=8,
    )
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    residuals = pd.read_csv(residuals_path)

    required = {
        "K_star_pumping_m_s",
        "K_star_slug_m_s",
        "K_hat_pumping_m_s",
        "K_hat_slug_m_s",
        "log_residual_pumping",
        "log_residual_slug",
    }

    assert report["checks"]["finite_targets_and_residuals"]["pass"]
    assert loaded["simulated_case_count"] == 2
    assert required.issubset(residuals.columns)
    assert np.isfinite(residuals[list(required)].to_numpy(dtype=float)).all()
    assert curves_path.exists()
    assert figure_prefix.with_suffix(".pdf").exists()
    assert figure_prefix.with_suffix(".svg").exists()
    assert figure_prefix.with_suffix(".png").exists()

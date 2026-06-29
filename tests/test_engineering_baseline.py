import json

import numpy as np
import pandas as pd

from trustk.analytical.theis import theis_drawdown
from trustk.diagnostics.fit_quality import summarize_fit_quality
from trustk.dimensionless.settings import DimensionalBaseConfig, convert_case_registry_to_solver_settings
from trustk.experiments.run_synthetic_population import run_synthetic_population
from trustk.interpretation.engineering import (
    estimate_hydraulic_conductivity_bouwer_rice,
    estimate_transmissivity_cooper_jacob,
    estimate_transmissivity_slug_semilog,
)


def _small_ready_settings() -> pd.DataFrame:
    registry = pd.DataFrame(
        {
            "case_id": ["case_0000", "case_0001"],
            "random_seed": [301, 302],
            "sigma_Y2": [0.25, 0.6],
            "lambda1_over_RI": [0.8, 0.5],
            "lambda2_over_lambda1": [0.6, 0.5],
            "lambda2_over_RI": [0.48, 0.25],
            "phi_lambda_rad": [0.1, 1.0],
            "r_skin_over_rw": [1.0, 1.0],
            "K_skin_over_K0": [1.0, 1.0],
            "Rmax_over_RI": [3.0, 4.0],
            "r_obs_over_RI_P": [0.4, 0.5],
            "tD_P_max": [100.0, 300.0],
            "C_D_w": [100.0, 200.0],
            "r_c_over_r_w": [0.7, 0.8],
            "tD_S_max": [30.0, 80.0],
            "pumping_interpretation_method": ["Cooper-Jacob", "Cooper-Jacob"],
            "slug_interpretation_method": ["semi-log slug", "semi-log slug"],
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


def test_cooper_jacob_uses_late_time_straight_line_and_recovers_transmissivity():
    true_t = 2.0e-4
    storativity = 2.0e-4
    pumping_rate = 1.0e-3
    radius = 8.0
    times = np.geomspace(1.0, 1.0e5, 80)
    drawdown = theis_drawdown(radius, times, true_t, storativity, pumping_rate)

    fit = estimate_transmissivity_cooper_jacob(
        times_s=times,
        drawdown_m=drawdown,
        radius_m=radius,
        pumping_rate_m3_s=pumping_rate,
        storativity=storativity,
        min_points=10,
    )

    assert np.isclose(fit.transmissivity_m2_s, true_t, rtol=0.10)
    assert fit.fit_time_min_s > times[0]
    assert fit.fit_point_count >= 10
    assert fit.r_squared is not None and fit.r_squared > 0.995
    assert fit.slope is not None and fit.slope > 0.0
    assert fit.storativity is not None and fit.storativity > 0.0


def test_cooper_jacob_window_rejects_nonpositive_fitted_drawdown():
    times = np.array([0.385763, 0.482053, 0.602378, 0.752737, 0.940627, 1.175417, 1.468811, 1.835440, 2.293583, 2.866082])
    drawdown = np.array(
        [
            7.740632e-09,
            7.218523e-08,
            5.647520e-07,
            3.723868e-06,
            2.081338e-05,
            9.927780e-05,
            4.072734e-04,
            1.449283e-03,
            4.514682e-03,
            1.242979e-02,
        ]
    )

    fit = estimate_transmissivity_cooper_jacob(
        times_s=times,
        drawdown_m=drawdown,
        radius_m=1.0,
        pumping_rate_m3_s=1.0e-3,
        min_points=5,
    )
    used = (times >= fit.fit_time_min_s) & (times <= fit.fit_time_max_s)
    fitted = fit.intercept + fit.slope * np.log10(times[used])

    assert np.all(fitted > 0.0)
    assert fit.rmse_log_response < 2.0


def test_fit_quality_fails_two_point_engineering_fallback():
    fitted = pd.DataFrame(
        {
            "case_id": ["case_x", "case_x", "case_x", "case_x"],
            "method": ["pumping", "pumping", "pumping", "pumping"],
            "time_s": [5.0, 10.0, 20.0, 40.0],
            "response_value": [0.01, 0.1, 0.2, 0.3],
            "fitted_response_value": [0.01, 0.1, 0.2, 0.3],
            "observed_for_log": [0.01, 0.1, 0.2, 0.3],
            "fitted_for_log": [0.01, 0.1, 0.2, 0.3],
            "log_error": [0.0, 0.0, 0.0, 0.0],
            "used_for_fit": [False, True, True, False],
        }
    )
    residuals = pd.DataFrame(
        {
            "case_id": ["case_x"],
            "K_hat_pumping_m_s": [1.0e-5],
            "K_star_pumping_m_s": [1.0e-5],
            "K_hat_slug_m_s": [1.0e-5],
            "K_star_slug_m_s": [1.0e-5],
            "log_residual_pumping": [0.0],
            "log_residual_slug": [0.0],
        }
    )

    qc = summarize_fit_quality(fitted, residuals)

    assert qc.loc[0, "qc_class"] == "fail"
    assert np.isnan(qc.loc[0, "rmse_log_response"])


def test_slug_semilog_window_ignores_early_and_late_nonideal_points():
    true_t = 1.6e-4
    well_storage = 0.015
    well_radius = 0.1
    outer_radius = 60.0
    times = np.geomspace(0.5, 250.0, 80)
    conductance = 2.0 * np.pi * true_t / np.log(outer_radius / well_radius)
    normalized_head = np.exp(-(conductance / well_storage) * times)
    normalized_head[:6] = normalized_head[:6] ** 0.85
    normalized_head[-8:] = np.maximum(normalized_head[-8:], 0.12)

    fit = estimate_transmissivity_slug_semilog(
        times_s=times,
        normalized_head=normalized_head,
        well_storage_m2=well_storage,
        well_radius_m=well_radius,
        outer_radius_m=outer_radius,
        min_points=10,
    )

    assert np.isclose(fit.transmissivity_m2_s, true_t, rtol=0.12)
    assert fit.fit_time_min_s > times[0]
    assert fit.fit_time_max_s < times[-1]
    assert fit.r_squared is not None and fit.r_squared > 0.995
    assert fit.slope is not None and fit.slope < 0.0


def test_bouwer_rice_interpreter_recovers_hydraulic_conductivity():
    true_k = 2.5e-5
    casing_radius = 0.04
    well_radius = 0.1
    effective_radius = 40.0
    screen_length = 12.0
    aquifer_thickness = 20.0
    times = np.geomspace(0.5, 500.0, 90)
    decay_rate = 2.0 * screen_length * true_k / (casing_radius**2 * np.log(effective_radius / well_radius))
    normalized_head = np.exp(-decay_rate * times)
    normalized_head[:5] = normalized_head[:5] ** 0.9
    normalized_head[-6:] = np.maximum(normalized_head[-6:], 0.04)

    fit = estimate_hydraulic_conductivity_bouwer_rice(
        times_s=times,
        normalized_head=normalized_head,
        casing_radius_m=casing_radius,
        well_radius_m=well_radius,
        screen_length_m=screen_length,
        aquifer_thickness_m=aquifer_thickness,
        effective_radius_m=effective_radius,
        min_points=10,
    )

    assert np.isclose(fit.transmissivity_m2_s / aquifer_thickness, true_k, rtol=0.15)
    assert fit.fit_time_min_s > times[0]
    assert fit.r_squared is not None and fit.r_squared > 0.995
    assert fit.slope is not None and fit.slope < 0.0


def test_engineering_baseline_script_writes_engineering_residuals_qc_and_prior(tmp_path):
    from trustk.experiments.run_engineering_baseline import run_engineering_baseline

    settings_path = tmp_path / "settings.csv"
    curves_path = tmp_path / "curves.csv"
    base_residuals_path = tmp_path / "base_residuals.csv"
    base_report_path = tmp_path / "base_report.json"
    base_figure_prefix = tmp_path / "fig_base"
    _small_ready_settings().to_csv(settings_path, index=False)
    run_synthetic_population(
        settings_path=settings_path,
        curves_path=curves_path,
        residuals_path=base_residuals_path,
        report_path=base_report_path,
        figure_prefix=base_figure_prefix,
        max_cases=2,
        cartesian_n_cap=129,
        mesh_n_r=18,
        mesh_n_theta=8,
    )

    residuals_path = tmp_path / "engineering_residuals.csv"
    fitted_path = tmp_path / "engineering_fitted.csv"
    qc_path = tmp_path / "engineering_qc.csv"
    prior_path = tmp_path / "engineering_prior.csv"
    report_path = tmp_path / "engineering_report.json"
    qc_figure_prefix = tmp_path / "fig_qc"
    prior_figure_prefix = tmp_path / "fig_prior"
    report = run_engineering_baseline(
        curves_path=curves_path,
        support_residuals_path=base_residuals_path,
        settings_path=settings_path,
        residuals_path=residuals_path,
        fitted_curves_path=fitted_path,
        qc_path=qc_path,
        prior_path=prior_path,
        report_path=report_path,
        qc_figure_prefix=qc_figure_prefix,
        prior_figure_prefix=prior_figure_prefix,
        min_prior_cases=1,
    )
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    residuals = pd.read_csv(residuals_path)
    qc = pd.read_csv(qc_path)

    assert report["checks"]["finite_engineering_residuals"]["pass"]
    assert loaded["simulated_case_count"] == 2
    assert {"K_hat_pumping_m_s", "K_hat_slug_m_s", "log_residual_pumping", "log_residual_slug"}.issubset(
        residuals.columns
    )
    assert {"pumping", "slug"} == set(qc["method"])
    assert np.isfinite(residuals[["K_hat_pumping_m_s", "K_hat_slug_m_s"]].to_numpy(dtype=float)).all()
    assert qc_figure_prefix.with_suffix(".pdf").exists()
    assert prior_figure_prefix.with_suffix(".svg").exists()


def test_bouwer_rice_sensitivity_script_writes_prior_comparison(tmp_path):
    from trustk.experiments.run_bouwer_rice_sensitivity import run_bouwer_rice_sensitivity
    from trustk.experiments.run_engineering_baseline import run_engineering_baseline

    settings_path = tmp_path / "settings.csv"
    curves_path = tmp_path / "curves.csv"
    base_residuals_path = tmp_path / "base_residuals.csv"
    _small_ready_settings().to_csv(settings_path, index=False)
    run_synthetic_population(
        settings_path=settings_path,
        curves_path=curves_path,
        residuals_path=base_residuals_path,
        report_path=tmp_path / "base_report.json",
        figure_prefix=tmp_path / "fig_base",
        max_cases=2,
        cartesian_n_cap=129,
        mesh_n_r=18,
        mesh_n_theta=8,
    )
    engineering_prior_path = tmp_path / "engineering_prior.csv"
    run_engineering_baseline(
        curves_path=curves_path,
        support_residuals_path=base_residuals_path,
        settings_path=settings_path,
        residuals_path=tmp_path / "engineering_residuals.csv",
        fitted_curves_path=tmp_path / "engineering_fitted.csv",
        qc_path=tmp_path / "engineering_qc.csv",
        prior_path=engineering_prior_path,
        report_path=tmp_path / "engineering_report.json",
        qc_figure_prefix=tmp_path / "fig_qc",
        prior_figure_prefix=tmp_path / "fig_prior",
        min_prior_cases=1,
    )

    report = run_bouwer_rice_sensitivity(
        curves_path=curves_path,
        support_residuals_path=base_residuals_path,
        settings_path=settings_path,
        engineering_prior_path=engineering_prior_path,
        residuals_path=tmp_path / "br_residuals.csv",
        fitted_curves_path=tmp_path / "br_fitted.csv",
        qc_path=tmp_path / "br_qc.csv",
        prior_path=tmp_path / "br_prior.csv",
        comparison_path=tmp_path / "br_comparison.csv",
        report_path=tmp_path / "br_report.json",
        figure_prefix=tmp_path / "fig_br",
        min_prior_cases=1,
    )
    comparison = pd.read_csv(tmp_path / "br_comparison.csv")

    assert report["checks"]["finite_residuals"]["pass"]
    assert {"semi-log slug", "Bouwer-Rice slug"} == set(comparison["slug_interpretation"])
    assert (tmp_path / "fig_br.pdf").exists()

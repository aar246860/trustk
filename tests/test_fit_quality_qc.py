import json

import numpy as np
import pandas as pd

from trustk.analytical.theis import theis_drawdown
from trustk.diagnostics.fit_quality import build_fitted_curve_table, summarize_fit_quality
from trustk.experiments.run_fit_quality_qc import run_fit_quality_qc
from trustk.priors.transformation_uncertainty import estimate_transformation_uncertainty


def _toy_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    times = np.geomspace(10.0, 10000.0, 8)
    cases = []
    curves = []
    residuals = []
    for case_id, response_multiplier, log_residual in [
        ("good", 1.0, 0.1),
        ("bad", np.linspace(0.2, 4.0, len(times)), 1.2),
    ]:
        transmissivity = 2.0e-4
        thickness = 20.0
        storativity = 2.0e-4
        q = 1.0e-3
        radius = 8.0
        well_storage = 0.015
        r_w = 0.1
        r_max = 60.0
        k_hat = transmissivity / thickness
        drawdown = theis_drawdown(radius, times, transmissivity, storativity, q)
        observed_drawdown = drawdown * response_multiplier
        s_d = 4.0 * np.pi * transmissivity * observed_drawdown / q
        conductance = 2.0 * np.pi * transmissivity / np.log(r_max / r_w)
        slug_head = np.exp(-(conductance / well_storage) * times)
        observed_slug = np.clip(slug_head * response_multiplier, 1.0e-6, 0.999)

        cases.append(
            {
                "case_id": case_id,
                "T0_m2_s": transmissivity,
                "aquifer_thickness_m": thickness,
                "storativity": storativity,
                "pumping_rate_m3_s": q,
                "well_storage_m2": well_storage,
                "well_radius_m": r_w,
                "r_max_m": r_max,
                "sigma_Y2": 0.5,
            }
        )
        residuals.append(
            {
                "case_id": case_id,
                "K_hat_pumping_m_s": k_hat,
                "K_hat_slug_m_s": k_hat,
                "K_star_pumping_m_s": k_hat / np.exp(log_residual),
                "K_star_slug_m_s": k_hat / np.exp(log_residual),
                "log_residual_pumping": log_residual,
                "log_residual_slug": log_residual,
                "sigma_Y2": 0.5,
            }
        )
        for t, value, dd in zip(times, s_d, observed_drawdown):
            curves.append(
                {
                    "case_id": case_id,
                    "method": "pumping",
                    "time_s": t,
                    "time_D": t / 10.0,
                    "response_value": value,
                    "drawdown_m": dd,
                    "observation_radius_m": radius,
                }
            )
        for t, value in zip(times, observed_slug):
            curves.append(
                {
                    "case_id": case_id,
                    "method": "slug",
                    "time_s": t,
                    "time_D": t / 10.0,
                    "response_value": value,
                    "drawdown_m": np.nan,
                    "observation_radius_m": r_w,
                }
            )
    return pd.DataFrame(curves), pd.DataFrame(residuals), pd.DataFrame(cases)


def test_fit_quality_reconstructs_fitted_curves_and_classifies_cases():
    curves, residuals, settings = _toy_tables()

    fitted = build_fitted_curve_table(curves, residuals, settings)
    qc = summarize_fit_quality(
        fitted,
        residuals,
        pass_thresholds={"pumping": 0.05, "slug": 0.05},
        warning_thresholds={"pumping": 0.25, "slug": 0.25},
    )

    good = qc[(qc["case_id"] == "good") & (qc["method"] == "pumping")].iloc[0]
    bad = qc[(qc["case_id"] == "bad") & (qc["method"] == "pumping")].iloc[0]

    assert np.isfinite(fitted["fitted_response_value"]).all()
    assert good["qc_class"] == "pass"
    assert good["rmse_log_response"] < 0.05
    assert bad["qc_class"] == "fail"
    assert bad["rmse_log_response"] > 0.25


def test_transformation_uncertainty_uses_only_requested_qc_class():
    qc = pd.DataFrame(
        {
            "case_id": ["a", "b", "c", "d"],
            "method": ["pumping", "pumping", "slug", "slug"],
            "qc_class": ["pass", "fail", "pass", "pass"],
            "log_residual": [0.1, 4.0, -0.2, 0.4],
            "rmse_log_response": [0.01, 2.0, 0.02, 0.03],
        }
    )

    prior = estimate_transformation_uncertainty(qc, accepted_classes=("pass",), min_cases=1)

    pumping = prior[prior["method"] == "pumping"].iloc[0]
    slug = prior[prior["method"] == "slug"].iloc[0]

    assert pumping["n_cases"] == 1
    assert pumping["mean_log_residual"] == 0.1
    assert slug["n_cases"] == 2
    assert np.isclose(slug["mean_log_residual"], 0.1)
    assert np.isclose(slug["bias_factor_c"], np.exp(0.1))


def test_fit_quality_qc_runner_writes_outputs(tmp_path):
    curves, residuals, settings = _toy_tables()
    curves_path = tmp_path / "curves.csv"
    residuals_path = tmp_path / "residuals.csv"
    settings_path = tmp_path / "settings.csv"
    fitted_curves_path = tmp_path / "fitted_curves.csv"
    qc_path = tmp_path / "qc.csv"
    prior_path = tmp_path / "prior.csv"
    report_path = tmp_path / "report.json"
    qc_figure_prefix = tmp_path / "fig_qc"
    prior_figure_prefix = tmp_path / "fig_prior"
    curves.to_csv(curves_path, index=False)
    residuals.to_csv(residuals_path, index=False)
    settings.to_csv(settings_path, index=False)

    report = run_fit_quality_qc(
        curves_path=curves_path,
        residuals_path=residuals_path,
        settings_path=settings_path,
        fitted_curves_path=fitted_curves_path,
        qc_path=qc_path,
        prior_path=prior_path,
        report_path=report_path,
        qc_figure_prefix=qc_figure_prefix,
        prior_figure_prefix=prior_figure_prefix,
        min_prior_cases=1,
    )

    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    qc = pd.read_csv(qc_path)
    prior = pd.read_csv(prior_path)

    assert report["checks"]["finite_fit_quality"]["pass"]
    assert loaded["fit_quality_rows"] == 4
    assert {"pass", "fail"}.issubset(set(qc["qc_class"]))
    assert set(prior["method"]) == {"pumping", "slug"}
    assert fitted_curves_path.exists()
    assert qc_figure_prefix.with_suffix(".pdf").exists()
    assert prior_figure_prefix.with_suffix(".pdf").exists()

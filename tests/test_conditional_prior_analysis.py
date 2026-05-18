import json

import numpy as np
import pandas as pd

from trustk.priors.conditional import (
    build_prior_dataset,
    evaluate_baselines,
    evaluate_holdout,
    fit_conditional_prior,
    split_train_validation,
)


def _synthetic_registry(n_cases: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    return pd.DataFrame(
        {
            "case_id": [f"case_{i:04d}" for i in range(n_cases)],
            "sigma_Y2": rng.uniform(0.1, 2.0, n_cases),
            "lambda1_over_RI": np.exp(rng.uniform(np.log(0.1), np.log(1.5), n_cases)),
            "lambda2_over_lambda1": rng.uniform(0.15, 1.0, n_cases),
            "phi_lambda_rad": rng.uniform(0.0, np.pi, n_cases),
            "r_skin_over_rw": rng.uniform(1.0, 5.0, n_cases),
            "K_skin_over_K0": np.exp(rng.uniform(np.log(0.1), np.log(5.0), n_cases)),
            "Rmax_over_RI": np.exp(rng.uniform(np.log(2.0), np.log(6.0), n_cases)),
            "r_obs_over_RI_P": rng.uniform(0.1, 1.0, n_cases),
            "tD_P_max": np.exp(rng.uniform(np.log(1.0e2), np.log(1.0e4), n_cases)),
            "C_D_w": np.exp(rng.uniform(np.log(1.0e2), np.log(1.0e4), n_cases)),
            "r_c_over_r_w": rng.uniform(0.3, 1.0, n_cases),
            "tD_S_max": np.exp(rng.uniform(np.log(1.0e1), np.log(1.0e3), n_cases)),
        }
    )


def _synthetic_residuals(registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(456)
    x = registry["sigma_Y2"].to_numpy()
    lam = np.log(registry["lambda1_over_RI"].to_numpy())
    c_d = np.log(registry["C_D_w"].to_numpy())
    pump_r = 0.15 + 0.45 * x - 0.20 * lam + rng.normal(0.0, 0.05, len(registry))
    slug_r = 0.55 + 0.25 * x + 0.06 * c_d + rng.normal(0.0, 0.06, len(registry))
    k_star_p = np.full(len(registry), 1.0e-5)
    k_star_s = np.full(len(registry), 2.0e-5)
    residuals = pd.DataFrame(
        {
            "case_id": registry["case_id"],
            "K_star_pumping_m_s": k_star_p,
            "K_star_slug_m_s": k_star_s,
            "K_hat_pumping_m_s": k_star_p * np.exp(pump_r),
            "K_hat_slug_m_s": k_star_s * np.exp(slug_r),
            "log_residual_pumping": pump_r,
            "log_residual_slug": slug_r,
        }
    )
    qc = pd.DataFrame(
        [
            {"case_id": row.case_id, "method": "pumping", "qc_class": "pass", "log_residual": pump_r[i]}
            for i, row in registry.iterrows()
        ]
        + [
            {"case_id": row.case_id, "method": "slug", "qc_class": "pass", "log_residual": slug_r[i]}
            for i, row in registry.iterrows()
        ]
    )
    return residuals, qc


def test_conditional_prior_beats_method_constant_on_structured_residuals():
    registry = _synthetic_registry()
    residuals, qc = _synthetic_residuals(registry)
    data = build_prior_dataset(residuals, registry, qc)
    train, validation = split_train_validation(data, validation_fraction=0.30, seed=42)

    model = fit_conditional_prior(train)
    holdout = evaluate_holdout(model, validation)
    baselines = evaluate_baselines(model, train, validation)

    assert set(holdout["method"]) == {"pumping", "slug"}
    assert (holdout["coverage_95"] > 0.80).all()
    assert baselines.loc[baselines["approach"].eq("conditional"), "rmse_log"].mean() < baselines.loc[
        baselines["approach"].eq("method_constant"), "rmse_log"
    ].mean()
    assert baselines.loc[baselines["approach"].eq("conditional"), "coverage_95"].between(0.70, 1.0).all()


def test_conditional_prior_script_writes_three_analysis_figures(tmp_path):
    from trustk.experiments.run_conditional_prior_analysis import run_conditional_prior_analysis

    registry = _synthetic_registry(36)
    residuals, qc = _synthetic_residuals(registry)
    residuals_path = tmp_path / "residuals.csv"
    registry_path = tmp_path / "registry.csv"
    qc_path = tmp_path / "qc.csv"
    model_report_path = tmp_path / "model_report.json"
    prediction_path = tmp_path / "predictions.csv"
    holdout_path = tmp_path / "holdout.csv"
    baseline_path = tmp_path / "baselines.csv"
    surfaces_prefix = tmp_path / "fig06"
    holdout_prefix = tmp_path / "fig07"
    baseline_prefix = tmp_path / "fig08"
    residuals.to_csv(residuals_path, index=False)
    registry.to_csv(registry_path, index=False)
    qc.to_csv(qc_path, index=False)

    report = run_conditional_prior_analysis(
        residuals_path=residuals_path,
        registry_path=registry_path,
        qc_path=qc_path,
        predictions_path=prediction_path,
        holdout_metrics_path=holdout_path,
        baseline_metrics_path=baseline_path,
        report_path=model_report_path,
        surfaces_figure_prefix=surfaces_prefix,
        holdout_figure_prefix=holdout_prefix,
        baseline_figure_prefix=baseline_prefix,
        validation_fraction=0.25,
        seed=9,
    )
    loaded = json.loads(model_report_path.read_text(encoding="utf-8"))

    assert report["checks"]["conditional_beats_method_constant_rmse"]["pass"]
    assert loaded["n_rows"] == 72
    assert prediction_path.exists()
    assert holdout_path.exists()
    assert baseline_path.exists()
    for prefix in [surfaces_prefix, holdout_prefix, baseline_prefix]:
        assert prefix.with_suffix(".pdf").exists()
        assert prefix.with_suffix(".svg").exists()
        assert prefix.with_suffix(".png").exists()

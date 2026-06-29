from __future__ import annotations

import argparse
import json
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import qmc

from trustk.dimensionless.registry import build_dimensionless_case_table
from trustk.interpretation.engineering import (
    estimate_hydraulic_conductivity_bouwer_rice,
    estimate_transmissivity_cooper_jacob,
)
from trustk.mesh.polar_mesh import make_log_polar_mesh
from trustk.physics.fv_solver import simulate_constant_rate_pumping
from trustk.physics.slug_solver import simulate_slug_recovery
from trustk.plotting.style import export_figure, journal_width, set_trustk_style
from trustk.priors.conditional import (
    evaluate_baselines,
    evaluate_holdout,
    fit_conditional_prior,
    predict_conditional_prior,
    split_train_validation,
)
from trustk.priors.transformation_uncertainty import estimate_transformation_uncertainty
from trustk.random_fields.joint_fields import JointFieldConfig, generate_joint_log_fields
from trustk.random_fields.mapping import map_cartesian_field_to_polar
from trustk.targets.support_area import support_targets_from_mapped_logk_logss


JOINT_FEATURE_COLUMNS = (
    "sigma_Y2",
    "log_lambda1_over_RI",
    "log_lambda2_over_lambda1",
    "sin_phi_lambda",
    "cos_phi_lambda",
    "log_Rmax_over_RI",
    "r_obs_over_RI_P",
    "log_tD_P_max",
    "log_C_D_w",
    "log_tD_S_max",
    "sigma_lnSs2",
    "storage_correlation",
    "log_lambda_Ss_over_lambda_K",
    "log_aquifer_thickness_m",
    "log_specific_storage_gmean_m_inv",
    "log_Ss_star_m_inv",
    "log_diffusivity_star_m2_s",
    "lnK_lnSs_corr",
)
FORMAL_MIN_CASE_COUNT = 4096
JOINT_QC_PASS_THRESHOLDS = {"pumping": 0.75, "slug_bouwer_rice": 0.75}
JOINT_QC_WARNING_THRESHOLDS = {"pumping": 1.50, "slug_bouwer_rice": 1.50}


def build_joint_storage_design(
    *,
    n_cases: int,
    seed: int,
    mesh_n_r: int = 32,
    mesh_n_theta: int = 16,
) -> pd.DataFrame:
    if n_cases < 2:
        raise ValueError("n_cases must be at least 2")
    table = build_dimensionless_case_table(n_cases=n_cases, seed=seed).copy()
    raw = qmc.LatinHypercube(d=5, seed=seed + 19_019).random(n_cases)
    correlations = np.array([-0.5, 0.0, 0.5])
    ratios = np.array([0.5, 1.0, 2.0])
    table["aquifer_thickness_m"] = _log_sample(raw[:, 0], 5.0, 200.0)
    table["specific_storage_gmean_m_inv"] = _log_sample(raw[:, 1], 1.0e-6, 1.0e-4)
    table["sigma_lnSs2"] = 0.05 + raw[:, 2] * (1.0 - 0.05)
    table["storage_correlation"] = correlations[np.minimum((raw[:, 3] * 3).astype(int), 2)]
    table["lambda_Ss_over_lambda_K"] = ratios[np.minimum((raw[:, 4] * 3).astype(int), 2)]
    return _complete_joint_design(table, mesh_n_r=mesh_n_r, mesh_n_theta=mesh_n_theta)


def run_joint_storage_reanalysis(
    *,
    design_path: str | Path = "data/processed/formal_joint_storage_design.csv",
    curves_path: str | Path = "data/processed/formal_joint_storage_curves.csv",
    summary_path: str | Path = "data/processed/formal_joint_storage_residuals.csv",
    qc_path: str | Path = "data/processed/formal_joint_storage_fit_quality_qc.csv",
    prior_path: str | Path = "data/processed/formal_joint_storage_transformation_prior.csv",
    prior_dataset_path: str | Path = "data/processed/formal_joint_storage_conditional_prior_dataset.csv",
    conditional_predictions_path: str | Path = "data/processed/formal_joint_storage_conditional_predictions.csv",
    holdout_path: str | Path = "data/processed/formal_joint_storage_conditional_holdout.csv",
    baseline_path: str | Path = "data/processed/formal_joint_storage_conditional_baselines.csv",
    report_path: str | Path = "outputs/reports/formal_joint_storage_prior.json",
    figure_prefix: str | Path = "outputs/figures/fig_joint_storage_formal_prior",
    n_cases: int = 4096,
    seed: int = 20260629,
    cartesian_n_cap: int | None = 161,
    mesh_n_r: int = 24,
    mesh_n_theta: int = 12,
    make_figure: bool = True,
    min_prior_cases: int = 3600,
) -> dict:
    design = build_joint_storage_design(
        n_cases=n_cases,
        seed=seed,
        mesh_n_r=mesh_n_r,
        mesh_n_theta=mesh_n_theta,
    )
    curves = []
    summaries = []
    for _, row in design.iterrows():
        case_curves, summary = simulate_joint_storage_case(row, cartesian_n_cap=cartesian_n_cap)
        curves.append(case_curves)
        summaries.append(summary)

    curves_table = pd.concat(curves, ignore_index=True)
    summary_table = pd.DataFrame(summaries)
    qc_table = _build_joint_qc_table(summary_table)
    prior_table = estimate_transformation_uncertainty(qc_table, min_cases=min_prior_cases)
    prior_dataset = _build_joint_prior_dataset(summary_table, design, qc_table)
    predictions, holdout, baselines = _build_conditional_validation(prior_dataset, seed=seed)
    report = _summarize_joint_storage(
        curves_table,
        summary_table,
        qc_table,
        prior_table,
        predictions,
        holdout,
        min_prior_cases=min_prior_cases,
    )
    report["purpose"] = "formal joint K-Ss-b synthetic transformation-uncertainty prior"
    report["n_cases_requested"] = int(n_cases)
    report["seed"] = int(seed)
    report["cartesian_n_cap"] = None if cartesian_n_cap is None else int(cartesian_n_cap)
    report["mesh"] = {"n_r": int(mesh_n_r), "n_theta": int(mesh_n_theta)}

    _write_table(design_path, design)
    _write_table(curves_path, curves_table)
    _write_table(summary_path, summary_table)
    _write_table(qc_path, qc_table)
    _write_table(prior_path, prior_table)
    _write_table(prior_dataset_path, prior_dataset)
    _write_table(conditional_predictions_path, predictions)
    _write_table(holdout_path, holdout)
    _write_table(baseline_path, baselines)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if make_figure:
        _plot_joint_storage(summary_table, Path(figure_prefix))
        Path(figure_prefix).with_suffix(".csv").write_text(summary_table.to_csv(index=False), encoding="utf-8")
        Path(figure_prefix).with_suffix(".json").write_text(
            json.dumps(
                {
                    "figure": Path(figure_prefix).name,
                    "palette": "official cmcrameri cmc.batlow and cmc.vik",
                    "purpose": report["purpose"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return report


def simulate_joint_storage_case(
    row: pd.Series,
    *,
    cartesian_n_cap: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    row = row.copy()
    n_cart = int(row["cartesian_n_required"])
    if cartesian_n_cap is not None:
        n_cart = min(n_cart, int(cartesian_n_cap))
    n_cart = _odd_grid_size(n_cart)
    extent = float(row["cartesian_extent_m"])
    dx = 2.0 * extent / (n_cart - 1)
    config = JointFieldConfig(
        nx=n_cart,
        ny=n_cart,
        dx=dx,
        dy=dx,
        mean_logk=float(np.log(row["K0_m_s"])),
        sigma_logk=float(row["sigma_logk"]),
        corr_len_k_x=float(row["lambda1_m"]),
        corr_len_k_y=float(row["lambda2_m"]),
        orientation_rad=float(row["phi_lambda_rad"]),
        mean_logss=float(np.log(row["specific_storage_gmean_m_inv"])),
        sigma_logss=float(row["sigma_lnSs"]),
        corr_len_ss_x=float(row["lambda1_ss_m"]),
        corr_len_ss_y=float(row["lambda2_ss_m"]),
        log_correlation=float(row["storage_correlation"]),
        min_logss=float(np.log(row["specific_storage_min_m_inv"])),
        max_logss=float(np.log(row["specific_storage_max_m_inv"])),
        seed=int(row["random_seed"]),
    )
    field = generate_joint_log_fields(config)
    mesh = make_log_polar_mesh(
        r_w=float(row["well_radius_m"]),
        r_max=float(row["r_max_m"]),
        n_r=int(row["mesh_n_r"]),
        n_theta=int(row["mesh_n_theta"]),
    )
    mapped_logk = map_cartesian_field_to_polar(field.k_field, mesh)
    mapped_logss = map_cartesian_field_to_polar(field.ss_field, mesh)
    thickness = float(row["aquifer_thickness_m"])
    transmissivity = np.exp(mapped_logk) * thickness
    storativity = np.exp(mapped_logss) * thickness
    support_targets = support_targets_from_mapped_logk_logss(
        mapped_logk=mapped_logk,
        mapped_logss=mapped_logss,
        mesh=mesh,
        pumping_radius_m=min(float(row["RI_P_m"]), float(row["r_max_m"])),
        slug_radius_m=min(max(float(row["r_skin_m"]), 5.0 * float(row["well_radius_m"])), float(row["r_max_m"])),
    )
    pumping_curves, pumping_summary = _run_joint_pumping(row, mesh, transmissivity, storativity, support_targets)
    slug_curves, slug_summary = _run_joint_slug(row, mesh, transmissivity, storativity, support_targets)
    summary = {
        "case_id": row["case_id"],
        "random_seed": int(row["random_seed"]),
        "used_cartesian_n": int(n_cart),
        "required_cartesian_n": int(row["cartesian_n_required"]),
        "aquifer_thickness_m": thickness,
        "specific_storage_gmean_m_inv": float(row["specific_storage_gmean_m_inv"]),
        "specific_storage_clipped_fraction": float(field.specific_storage_clipped_fraction),
        "sigma_Y2": float(row["sigma_Y2"]),
        "sigma_lnSs2": float(row["sigma_lnSs2"]),
        "storage_correlation": float(row["storage_correlation"]),
        "lambda_Ss_over_lambda_K": float(row["lambda_Ss_over_lambda_K"]),
        "t0_s": float(row["t0_s"]),
        **support_targets,
        **pumping_summary,
        **slug_summary,
        "mapped_logk_mean": float(np.mean(mapped_logk)),
        "mapped_logk_std": float(np.std(mapped_logk)),
        "mapped_logss_mean": float(np.mean(mapped_logss)),
        "mapped_logss_std": float(np.std(mapped_logss)),
    }
    curves = pd.concat([pumping_curves, slug_curves], ignore_index=True)
    curves["sigma_Y2"] = float(row["sigma_Y2"])
    curves["sigma_lnSs2"] = float(row["sigma_lnSs2"])
    curves["storage_correlation"] = float(row["storage_correlation"])
    return curves, summary


def _run_joint_pumping(
    row: pd.Series,
    mesh,
    transmissivity: np.ndarray,
    storativity: np.ndarray,
    support: dict,
) -> tuple[pd.DataFrame, dict]:
    times = _geom_times(float(row["pumping_time_max_s"]), int(row["pumping_n_times"]))
    result = simulate_constant_rate_pumping(
        mesh,
        transmissivity=transmissivity,
        storativity=storativity,
        pumping_rate=float(row["pumping_rate_m3_s"]),
        times=times,
    )
    obs_idx = int(np.argmin(np.abs(mesh.r_centers - float(row["r_obs_m"]))))
    obs_radius = float(mesh.r_centers[obs_idx])
    drawdown = result.drawdown[:, obs_idx, :].mean(axis=1)
    fit = estimate_transmissivity_cooper_jacob(
        times_s=times,
        drawdown_m=drawdown,
        radius_m=obs_radius,
        pumping_rate_m3_s=float(row["pumping_rate_m3_s"]),
        storativity=float(row["interpretation_storativity"]),
        min_points=max(3, min(8, int(row["pumping_n_times"]) // 3)),
    )
    k_hat = fit.transmissivity_m2_s / float(row["aquifer_thickness_m"])
    curve = pd.DataFrame(
        {
            "case_id": row["case_id"],
            "method": "pumping",
            "time_s": times,
            "time_D": times / float(row["t0_s"]),
            "response_value": 4.0 * np.pi * float(row["T0_m2_s"]) * drawdown / float(row["pumping_rate_m3_s"]),
            "drawdown_m": drawdown,
            "observation_radius_m": obs_radius,
        }
    )
    summary = _fit_summary("pumping", fit, k_hat, support["K_star_pumping_m_s"], result.mass_balance_error[-1])
    summary["pumping_fit_method"] = fit.method
    summary["pumping_observation_radius_m"] = obs_radius
    summary["pumping_final_drawdown_m"] = float(drawdown[-1])
    return curve, summary


def _run_joint_slug(row: pd.Series, mesh, transmissivity: np.ndarray, storativity: np.ndarray, support: dict) -> tuple[pd.DataFrame, dict]:
    times = _geom_times(float(row["slug_time_max_s"]), int(row["slug_n_times"]))
    result = simulate_slug_recovery(
        mesh,
        transmissivity=transmissivity,
        storativity=storativity,
        well_storage=float(row["well_storage_m2"]),
        initial_well_head=float(row["initial_slug_head_m"]),
        times=times,
    )
    normalized = result.well_head / float(row["initial_slug_head_m"])
    fit = estimate_hydraulic_conductivity_bouwer_rice(
        times_s=times,
        normalized_head=normalized,
        casing_radius_m=float(row["casing_radius_m"]),
        well_radius_m=float(row["well_radius_m"]),
        screen_length_m=float(row["screen_length_m"]),
        aquifer_thickness_m=float(row["aquifer_thickness_m"]),
        effective_radius_m=float(row["r_max_m"]),
        min_points=max(3, min(8, int(row["slug_n_times"]) // 3)),
    )
    k_hat = fit.transmissivity_m2_s / float(row["aquifer_thickness_m"])
    curve = pd.DataFrame(
        {
            "case_id": row["case_id"],
            "method": "slug_bouwer_rice",
            "time_s": times,
            "time_D": times / float(row["t0_s"]),
            "response_value": normalized,
            "drawdown_m": np.nan,
            "observation_radius_m": float(row["well_radius_m"]),
        }
    )
    summary = _fit_summary("slug", fit, k_hat, support["K_star_slug_m_s"], result.mass_balance_error[-1])
    summary["slug_fit_method"] = fit.method
    summary["slug_final_normalized_head"] = float(normalized[-1])
    return curve, summary


def _complete_joint_design(table: pd.DataFrame, *, mesh_n_r: int, mesh_n_theta: int) -> pd.DataFrame:
    well_radius = 0.1
    k0 = float(np.exp(-11.0))
    table["K0_m_s"] = k0
    table["T0_m2_s"] = table["K0_m_s"] * table["aquifer_thickness_m"]
    table["storativity_gmean"] = table["specific_storage_gmean_m_inv"] * table["aquifer_thickness_m"]
    table["interpretation_storativity"] = table["storativity_gmean"]
    table["well_radius_m"] = well_radius
    table["t0_s"] = table["specific_storage_gmean_m_inv"] * well_radius**2 / k0
    table["sigma_logk"] = np.sqrt(table["sigma_Y2"])
    table["sigma_lnSs"] = np.sqrt(table["sigma_lnSs2"])
    table["RI_P_m"] = well_radius * np.sqrt(table["tD_P_max"])
    table["RI_S_m"] = well_radius * np.sqrt(table["tD_S_max"])
    table["RI_common_m"] = np.maximum(table["RI_P_m"], table["RI_S_m"])
    table["lambda1_m"] = table["lambda1_over_RI"] * table["RI_common_m"]
    table["lambda2_m"] = table["lambda2_over_RI"] * table["RI_common_m"]
    table["lambda1_ss_m"] = table["lambda1_m"] * table["lambda_Ss_over_lambda_K"]
    table["lambda2_ss_m"] = table["lambda2_m"] * table["lambda_Ss_over_lambda_K"]
    table["r_skin_m"] = table["r_skin_over_rw"] * well_radius
    table["r_max_m"] = table["Rmax_over_RI"] * table["RI_common_m"]
    table["r_obs_m"] = table["r_obs_over_RI_P"] * table["RI_P_m"]
    table["pumping_time_max_s"] = table["tD_P_max"] * table["t0_s"]
    table["slug_time_max_s"] = table["tD_S_max"] * table["t0_s"]
    table["well_storage_m2"] = table["C_D_w"] * table["storativity_gmean"] * well_radius**2
    table["casing_radius_m"] = table["r_c_over_r_w"] * well_radius
    table["screen_length_m"] = table["aquifer_thickness_m"]
    table["pumping_rate_m3_s"] = 1.0e-3
    table["initial_slug_head_m"] = 1.0
    table["specific_storage_min_m_inv"] = 1.0e-7
    table["specific_storage_max_m_inv"] = 1.0e-3
    table["mesh_n_r"] = int(mesh_n_r)
    table["mesh_n_theta"] = int(mesh_n_theta)
    table["pumping_n_times"] = 24
    table["slug_n_times"] = 32
    table["cartesian_extent_m"] = 1.1 * table["r_max_m"]
    table["cartesian_dx_target_m"] = table["lambda2_m"] / 3.0
    table["cartesian_n_required"] = _odd_grid_size(
        np.ceil(2.0 * table["cartesian_extent_m"] / table["cartesian_dx_target_m"]).to_numpy(dtype=int) + 1
    )
    return table


def _summarize_joint_storage(
    curves: pd.DataFrame,
    summary: pd.DataFrame,
    qc: pd.DataFrame,
    prior: pd.DataFrame,
    predictions: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    min_prior_cases: int,
) -> dict:
    finite_curves = bool(np.isfinite(curves["response_value"].to_numpy(dtype=float)).all())
    residual_columns = ["log_residual_pumping", "log_residual_slug"]
    finite_residuals = bool(np.isfinite(summary[residual_columns].to_numpy(dtype=float)).all())
    storage_columns = {"Ss_star_pumping_m_inv", "Ss_star_slug_m_inv", "diffusivity_star_pumping_m2_s"}
    method_stats = {
        "pumping": _residual_stats(summary["log_residual_pumping"]),
        "slug_bouwer_rice": _residual_stats(summary["log_residual_slug"]),
    }
    formal_prior = _formal_prior_summary(prior, min_prior_cases=min_prior_cases)
    validation = _validation_summary(predictions, holdout)
    return {
        "simulated_case_count": int(len(summary)),
        "curve_row_count": int(len(curves)),
        "residual_summary": method_stats,
        "formal_prior": formal_prior,
        "conditional_validation": validation,
        "convergence_diagnostics": _four_block_stability(summary),
        "formal_acceptance": _formal_acceptance(summary, qc, predictions),
        "checks": {
            "finite_responses": {"pass": finite_curves},
            "finite_log_residuals": {"pass": finite_residuals},
            "joint_storage_columns_present": {"pass": storage_columns.issubset(summary.columns)},
            "formal_prior_methods_present": {
                "pass": {"pumping", "slug_bouwer_rice"}.issubset(set(prior["method"])) if "method" in prior.columns else False
            },
            "storage_bounds_respected": {
                "pass": bool(summary["specific_storage_clipped_fraction"].between(0.0, 0.25).all())
            },
        },
        "ranges": _numeric_ranges(
            summary[
                [
                    "aquifer_thickness_m",
                    "specific_storage_gmean_m_inv",
                    "Ss_star_pumping_m_inv",
                    "Ss_star_slug_m_inv",
                    "diffusivity_star_pumping_m2_s",
                    "diffusivity_star_slug_m2_s",
                    "storage_correlation",
                    "lambda_Ss_over_lambda_K",
                    "log_residual_pumping",
                    "log_residual_slug",
                ]
            ]
        ),
    }


def _build_joint_qc_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        rows.append(_joint_qc_row(row, method="pumping", residual_prefix="pumping"))
        rows.append(_joint_qc_row(row, method="slug_bouwer_rice", residual_prefix="slug"))
    return pd.DataFrame(rows)


def _joint_qc_row(row: pd.Series, *, method: str, residual_prefix: str) -> dict:
    rmse = float(row[f"{residual_prefix}_fit_rmse_log_response"])
    log_residual = float(row[f"log_residual_{residual_prefix}"])
    return {
        "case_id": row["case_id"],
        "method": method,
        "interpretation_method": "Cooper-Jacob" if method == "pumping" else "Bouwer-Rice",
        "qc_class": _classify_joint_qc(method, rmse),
        "rmse_log_response": rmse,
        "fit_point_count": int(row[f"{residual_prefix}_fit_point_count"]),
        "fit_time_min_s": float(row[f"{residual_prefix}_fit_time_min_s"]),
        "fit_time_max_s": float(row[f"{residual_prefix}_fit_time_max_s"]),
        "log_residual": log_residual,
        "abs_log_residual": abs(log_residual),
        "K_hat_m_s": float(row[f"K_hat_{residual_prefix}_m_s"]),
        "K_star_m_s": float(row[f"K_star_{residual_prefix}_m_s"]),
        "sigma_Y2": float(row["sigma_Y2"]),
        "sigma_lnSs2": float(row["sigma_lnSs2"]),
        "storage_correlation": float(row["storage_correlation"]),
        "lambda_Ss_over_lambda_K": float(row["lambda_Ss_over_lambda_K"]),
    }


def _classify_joint_qc(method: str, rmse: float) -> str:
    if not np.isfinite(rmse):
        return "fail"
    if rmse <= JOINT_QC_PASS_THRESHOLDS[method]:
        return "pass"
    if rmse <= JOINT_QC_WARNING_THRESHOLDS[method]:
        return "warning"
    return "fail"


def _build_joint_prior_dataset(summary: pd.DataFrame, design: pd.DataFrame, qc: pd.DataFrame) -> pd.DataFrame:
    features = _joint_feature_table(design)
    data = summary.merge(features, on="case_id", how="left")
    rows = []
    for _, row in data.iterrows():
        rows.append(_joint_prior_row(row, method="pumping", residual_prefix="pumping"))
        rows.append(_joint_prior_row(row, method="slug_bouwer_rice", residual_prefix="slug"))
    long = pd.DataFrame(rows)
    qc_small = qc[["case_id", "method", "qc_class"]].drop_duplicates()
    long = long.merge(qc_small, on=["case_id", "method"], how="left")
    long = long[long["qc_class"].eq("pass")].copy()
    numeric = long.select_dtypes(include=[np.number]).columns
    return long[np.isfinite(long[numeric]).all(axis=1)].reset_index(drop=True)


def _joint_feature_table(design: pd.DataFrame) -> pd.DataFrame:
    out = design.copy()
    out["log_lambda1_over_RI"] = _safe_log(out["lambda1_over_RI"])
    out["log_lambda2_over_lambda1"] = _safe_log(out["lambda2_over_lambda1"])
    out["sin_phi_lambda"] = np.sin(out["phi_lambda_rad"].to_numpy(dtype=float))
    out["cos_phi_lambda"] = np.cos(out["phi_lambda_rad"].to_numpy(dtype=float))
    out["log_Rmax_over_RI"] = _safe_log(out["Rmax_over_RI"])
    out["log_tD_P_max"] = _safe_log(out["tD_P_max"])
    out["log_C_D_w"] = _safe_log(out["C_D_w"])
    out["log_tD_S_max"] = _safe_log(out["tD_S_max"])
    out["log_lambda_Ss_over_lambda_K"] = _safe_log(out["lambda_Ss_over_lambda_K"])
    out["log_aquifer_thickness_m"] = _safe_log(out["aquifer_thickness_m"])
    out["log_specific_storage_gmean_m_inv"] = _safe_log(out["specific_storage_gmean_m_inv"])
    return out[
        [
            "case_id",
            "log_lambda1_over_RI",
            "log_lambda2_over_lambda1",
            "sin_phi_lambda",
            "cos_phi_lambda",
            "log_Rmax_over_RI",
            "r_obs_over_RI_P",
            "log_tD_P_max",
            "log_C_D_w",
            "log_tD_S_max",
            "log_lambda_Ss_over_lambda_K",
            "log_aquifer_thickness_m",
            "log_specific_storage_gmean_m_inv",
        ]
    ]


def _joint_prior_row(row: pd.Series, *, method: str, residual_prefix: str) -> dict:
    support_prefix = "pumping" if residual_prefix == "pumping" else "slug"
    out = {
        "case_id": row["case_id"],
        "method": method,
        "K_hat_m_s": float(row[f"K_hat_{residual_prefix}_m_s"]),
        "K_star_m_s": float(row[f"K_star_{residual_prefix}_m_s"]),
        "log_residual": float(row[f"log_residual_{residual_prefix}"]),
        "sigma_Y2": float(row["sigma_Y2"]),
        "sigma_lnSs2": float(row["sigma_lnSs2"]),
        "storage_correlation": float(row["storage_correlation"]),
        "log_Ss_star_m_inv": float(np.log(row[f"Ss_star_{support_prefix}_m_inv"])),
        "log_diffusivity_star_m2_s": float(np.log(row[f"diffusivity_star_{support_prefix}_m2_s"])),
        "lnK_lnSs_corr": float(row[f"lnK_lnSs_corr_{support_prefix}"]),
    }
    for column in JOINT_FEATURE_COLUMNS:
        if column not in out:
            out[column] = float(row[column])
    return out


def _build_conditional_validation(prior_dataset: pd.DataFrame, *, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if prior_dataset.empty or prior_dataset["case_id"].nunique() < 4:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    train, validation = split_train_validation(prior_dataset, validation_fraction=0.25, seed=seed + 73)
    if train.empty or validation.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    model = fit_conditional_prior(train, feature_columns=JOINT_FEATURE_COLUMNS, ridge_alpha=2.0)
    predictions = predict_conditional_prior(model, validation)
    holdout = evaluate_holdout(model, validation)
    baselines = evaluate_baselines(model, train, validation)
    return predictions, holdout, baselines


def _formal_prior_summary(prior: pd.DataFrame, *, min_prior_cases: int) -> dict:
    methods = {}
    labels = {"pumping": "Cooper-Jacob", "slug_bouwer_rice": "Bouwer-Rice"}
    for method, label in labels.items():
        method_prior = prior[prior["method"].eq(method)] if "method" in prior.columns else pd.DataFrame()
        if method_prior.empty:
            methods[method] = {
                "interpretation_method": label,
                "n_cases": 0,
                "is_sufficient": False,
            }
            continue
        row = method_prior.iloc[0]
        methods[method] = {
            "interpretation_method": label,
            "n_cases": int(row["n_cases"]),
            "minimum_cases": int(min_prior_cases),
            "is_sufficient": bool(row["is_sufficient"]),
            "mean_log_residual": float(row["mean_log_residual"]),
            "sd_log_residual": float(row["sd_log_residual"]),
            "bias_factor_c": float(row["bias_factor_c"]),
            "target_correction_factor": float(row["target_correction_factor"]),
            "scatter_factor": float(row["scatter_factor"]),
            "q05_factor": float(row["q05_factor"]),
            "q95_factor": float(row["q95_factor"]),
        }
    return {
        "feature_columns": list(JOINT_FEATURE_COLUMNS),
        "methods": methods,
    }


def _validation_summary(predictions: pd.DataFrame, holdout: pd.DataFrame) -> dict:
    rows = {}
    for method, group in predictions.groupby("method"):
        z = group["standardized_residual_error"].to_numpy(dtype=float)
        rows[method] = {
            "n": int(len(group)),
            "standardized_mean": float(np.mean(z)),
            "standardized_sd": float(np.std(z, ddof=1)) if len(z) > 1 else float("nan"),
        }
    if not holdout.empty:
        for _, row in holdout.iterrows():
            method = row["method"]
            rows.setdefault(method, {})
            rows[method].update(
                {
                    "holdout_bias": float(row["bias"]),
                    "holdout_rmse": float(row["rmse"]),
                    "coverage_80": float(row["coverage_80"]),
                    "coverage_95": float(row["coverage_95"]),
                    "mean_sigma": float(row["mean_sigma"]),
                }
            )
    return rows


def _formal_acceptance(summary: pd.DataFrame, qc: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    mean_se = _mean_se_checks(summary)
    block = _four_block_stability(summary)
    qc_pass = _qc_pass_fraction(qc)
    validation = _validation_acceptance(predictions)
    return {
        "minimum_case_count": {
            "pass": bool(len(summary) >= FORMAL_MIN_CASE_COUNT),
            "actual": int(len(summary)),
            "minimum": int(FORMAL_MIN_CASE_COUNT),
        },
        "qc_pass_fraction": qc_pass,
        "mean_log_residual_se": mean_se,
        "four_block_stability": block,
        "conditional_validation": validation,
        "all_thresholds_pass": bool(
            len(summary) >= FORMAL_MIN_CASE_COUNT
            and all(item["pass"] for item in qc_pass.values())
            and all(item["pass"] for item in mean_se.values())
            and bool(block["pass"])
            and all(item["pass"] for item in validation.values())
        ),
    }


def _mean_se_checks(summary: pd.DataFrame) -> dict:
    specs = {
        "pumping": ("log_residual_pumping", 0.05),
        "slug_bouwer_rice": ("log_residual_slug", 0.10),
    }
    out = {}
    for method, (column, threshold) in specs.items():
        values = summary[column].to_numpy(dtype=float)
        se = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")
        out[method] = {"pass": bool(np.isfinite(se) and se < threshold), "se": se, "threshold": threshold}
    return out


def _qc_pass_fraction(qc: pd.DataFrame) -> dict:
    out = {}
    for method, group in qc.groupby("method"):
        fraction = float(group["qc_class"].eq("pass").mean())
        out[method] = {"pass": bool(fraction >= 0.90), "fraction": fraction, "threshold": 0.90}
    return out


def _validation_acceptance(predictions: pd.DataFrame) -> dict:
    out = {}
    for method, group in predictions.groupby("method"):
        err = group["log_residual"].to_numpy(dtype=float) - group["predicted_log_residual_mean"].to_numpy(dtype=float)
        sigma = group["predicted_log_residual_sigma"].to_numpy(dtype=float)
        z = group["standardized_residual_error"].to_numpy(dtype=float)
        coverage = float(np.mean(np.abs(err) <= 1.96 * sigma))
        z_mean = float(np.mean(z))
        z_sd = float(np.std(z, ddof=1)) if len(z) > 1 else float("nan")
        passed = 0.90 <= coverage <= 0.98 and abs(z_mean) <= 0.15 and 0.80 <= z_sd <= 1.25
        out[method] = {
            "pass": bool(passed),
            "coverage_95": coverage,
            "coverage_95_range": [0.90, 0.98],
            "standardized_mean": z_mean,
            "standardized_sd": z_sd,
            "standardized_mean_abs_max": 0.15,
            "standardized_sd_range": [0.80, 1.25],
        }
    return out


def _fit_summary(prefix: str, fit, k_hat: float, k_star: float, mass_error: float) -> dict:
    return {
        f"K_hat_{prefix}_m_s": float(k_hat),
        f"log_residual_{prefix}": float(np.log(k_hat) - np.log(k_star)),
        f"{prefix}_fit_rmse_log_response": float(fit.rmse_log_response),
        f"{prefix}_fit_point_count": int(fit.fit_point_count),
        f"{prefix}_fit_time_min_s": float(fit.fit_time_min_s),
        f"{prefix}_fit_time_max_s": float(fit.fit_time_max_s),
        f"{prefix}_mass_balance_error_final": float(mass_error),
    }


def _four_block_stability(summary: pd.DataFrame) -> dict:
    if len(summary) < 8:
        return {"pass": True, "reason": "fewer than eight cases; skipped"}
    ordered = summary.sort_values("random_seed")
    blocks = [ordered.iloc[idx] for idx in np.array_split(np.arange(len(ordered)), 4)]
    thresholds = {"log_residual_pumping": 0.15, "log_residual_slug": 0.30}
    diagnostics = {}
    passes = []
    for column, threshold in thresholds.items():
        full_mean = float(ordered[column].mean())
        block_means = [float(block[column].mean()) for block in blocks if len(block) > 0]
        max_delta = float(max(abs(value - full_mean) for value in block_means))
        method = "pumping" if column == "log_residual_pumping" else "slug_bouwer_rice"
        passed = bool(max_delta < threshold)
        diagnostics[method] = {
            "pass": passed,
            "full_mean": full_mean,
            "block_means": block_means,
            "max_abs_mean_delta": max_delta,
            "threshold": threshold,
        }
        passes.append(passed)
    return {"pass": bool(all(passes)), "methods": diagnostics}


def _plot_joint_storage(summary: pd.DataFrame, figure_prefix: Path) -> None:
    set_trustk_style()
    fig, axes = plt.subplots(2, 2, figsize=(journal_width(170), 5.7))
    flat = axes.ravel()
    _scatter_truth_hat(flat[0], summary, "pumping")
    _scatter_truth_hat(flat[1], summary, "slug")
    flat[2].scatter(
        summary["specific_storage_gmean_m_inv"],
        summary["log_residual_pumping"],
        s=16,
        c=summary["aquifer_thickness_m"],
        cmap=cmc.batlow,
        edgecolor="none",
        alpha=0.72,
    )
    flat[2].scatter(
        summary["specific_storage_gmean_m_inv"],
        summary["log_residual_slug"],
        s=16,
        color=cmc.vik(0.78),
        edgecolor="none",
        alpha=0.48,
    )
    flat[2].set_xscale("log")
    flat[2].axhline(0.0, color="0.3", lw=0.8, ls="--")
    flat[2].set_xlabel(r"Geometric-mean $S_s$ (m$^{-1}$)")
    flat[2].set_ylabel("Log residual")
    flat[2].text(0.03, 0.96, "(c)", transform=flat[2].transAxes, ha="left", va="top", fontsize=9, fontweight="bold")
    parts = flat[3].violinplot(
        [summary["log_residual_pumping"], summary["log_residual_slug"]],
        positions=[1.0, 2.0],
        widths=0.46,
        showmeans=True,
        showextrema=False,
    )
    for body, color in zip(parts["bodies"], [cmc.batlow(0.62), cmc.vik(0.76)]):
        body.set_facecolor(color)
        body.set_edgecolor("0.25")
        body.set_alpha(0.72)
    parts["cmeans"].set_color("0.15")
    flat[3].axhline(0.0, color="0.3", lw=0.8, ls="--")
    flat[3].set_xticks([1.0, 2.0])
    flat[3].set_xticklabels(["CJ", "BR slug"])
    flat[3].set_ylabel("Log residual")
    flat[3].text(0.03, 0.96, "(d)", transform=flat[3].transAxes, ha="left", va="top", fontsize=9, fontweight="bold")
    for ax in flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.11, top=0.95, wspace=0.36, hspace=0.42)
    figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    export_figure(fig, figure_prefix)
    plt.close(fig)


def _scatter_truth_hat(ax, summary: pd.DataFrame, method: str) -> None:
    k_star = summary[f"K_star_{method}_m_s"]
    k_hat = summary[f"K_hat_{method}_m_s"]
    ax.scatter(k_star, k_hat, s=18, c=summary["sigma_lnSs2"], cmap=cmc.batlow, edgecolor="none", alpha=0.72)
    values = np.r_[k_star.to_numpy(dtype=float), k_hat.to_numpy(dtype=float)]
    low = 10.0 ** np.floor(np.log10(np.min(values[values > 0.0])))
    high = 10.0 ** np.ceil(np.log10(np.max(values[values > 0.0])))
    ax.plot([low, high], [low, high], color="0.3", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel(r"$K_m^*$ (m s$^{-1}$)")
    ax.set_ylabel(r"$\hat K_m$ (m s$^{-1}$)")
    label = "(a)" if method == "pumping" else "(b)"
    title = "Cooper-Jacob pumping" if method == "pumping" else "Bouwer-Rice slug"
    ax.set_title(title, fontsize=9)
    ax.text(0.03, 0.96, label, transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")


def _residual_stats(values: pd.Series) -> dict:
    arr = values.to_numpy(dtype=float)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    se = float(sd / np.sqrt(len(arr))) if len(arr) > 1 else float("nan")
    return {
        "n_cases": int(len(arr)),
        "mean_log_residual": mean,
        "sd_log_residual": sd,
        "bias_factor_c": float(np.exp(mean)),
        "target_correction_factor": float(np.exp(-mean)),
        "scatter_factor": float(np.exp(sd)),
        "bias_factor_95_ci": [float(np.exp(mean - 1.96 * se)), float(np.exp(mean + 1.96 * se))],
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


def _write_table(path: str | Path, table: pd.DataFrame) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)


def _geom_times(max_time: float, n_times: int) -> np.ndarray:
    if max_time <= 0.0:
        raise ValueError("max_time must be positive")
    if n_times < 3:
        raise ValueError("n_times must be at least 3")
    return np.geomspace(max_time / 1.0e3, max_time, n_times)


def _odd_grid_size(value) -> np.ndarray | int:
    arr = np.asarray(value, dtype=int)
    arr = np.maximum(arr, 17)
    out = arr + (arr % 2 == 0)
    if out.ndim == 0:
        return int(out)
    return out


def _log_sample(unit: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.exp(np.log(lo) + unit * (np.log(hi) - np.log(lo)))


def _safe_log(values: pd.Series) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.log(np.maximum(arr, np.finfo(float).tiny))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", default="data/processed/formal_joint_storage_design.csv")
    parser.add_argument("--curves", default="data/processed/formal_joint_storage_curves.csv")
    parser.add_argument("--summary", default="data/processed/formal_joint_storage_residuals.csv")
    parser.add_argument("--qc", default="data/processed/formal_joint_storage_fit_quality_qc.csv")
    parser.add_argument("--prior", default="data/processed/formal_joint_storage_transformation_prior.csv")
    parser.add_argument("--prior-dataset", default="data/processed/formal_joint_storage_conditional_prior_dataset.csv")
    parser.add_argument("--conditional-predictions", default="data/processed/formal_joint_storage_conditional_predictions.csv")
    parser.add_argument("--holdout", default="data/processed/formal_joint_storage_conditional_holdout.csv")
    parser.add_argument("--baseline", default="data/processed/formal_joint_storage_conditional_baselines.csv")
    parser.add_argument("--report", default="outputs/reports/formal_joint_storage_prior.json")
    parser.add_argument("--figure-prefix", default="outputs/figures/fig_joint_storage_formal_prior")
    parser.add_argument("--n-cases", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--cartesian-n-cap", type=int, default=161)
    parser.add_argument("--mesh-n-r", type=int, default=24)
    parser.add_argument("--mesh-n-theta", type=int, default=12)
    parser.add_argument("--min-prior-cases", type=int, default=3600)
    args = parser.parse_args(argv)
    report = run_joint_storage_reanalysis(
        design_path=args.design,
        curves_path=args.curves,
        summary_path=args.summary,
        qc_path=args.qc,
        prior_path=args.prior,
        prior_dataset_path=args.prior_dataset,
        conditional_predictions_path=args.conditional_predictions,
        holdout_path=args.holdout,
        baseline_path=args.baseline,
        report_path=args.report,
        figure_prefix=args.figure_prefix,
        n_cases=args.n_cases,
        seed=args.seed,
        cartesian_n_cap=args.cartesian_n_cap,
        mesh_n_r=args.mesh_n_r,
        mesh_n_theta=args.mesh_n_theta,
        min_prior_cases=args.min_prior_cases,
    )
    all_pass = all(item["pass"] for item in report["checks"].values())
    print(f"simulated_case_count={report['simulated_case_count']}")
    print(f"joint_storage_reanalysis_pass={all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

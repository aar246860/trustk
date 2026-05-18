"""Fit-quality diagnostics for conventional TRUST-K interpretations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trustk.analytical.theis import theis_drawdown


DEFAULT_PASS_THRESHOLDS = {"pumping": 0.75, "slug": 0.25}
DEFAULT_WARNING_THRESHOLDS = {"pumping": 1.50, "slug": 0.75}


def build_fitted_curve_table(
    curves: pd.DataFrame,
    residuals: pd.DataFrame,
    settings: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct conventional fitted curves for QC against simulated responses."""

    required_curves = {"case_id", "method", "time_s", "response_value"}
    required_residuals = {"case_id", "K_hat_pumping_m_s", "K_hat_slug_m_s"}
    required_settings = {
        "case_id",
        "T0_m2_s",
        "aquifer_thickness_m",
        "storativity",
        "pumping_rate_m3_s",
        "well_storage_m2",
        "well_radius_m",
        "r_max_m",
    }
    _require_columns(curves, required_curves, "curves")
    _require_columns(residuals, required_residuals, "residuals")
    _require_columns(settings, required_settings, "settings")

    residual_cols = list(
        {
            "case_id",
            "K_hat_pumping_m_s",
            "K_hat_slug_m_s",
            "log_residual_pumping",
            "log_residual_slug",
            "sigma_Y2",
        }.intersection(residuals.columns)
    )
    setting_cols = list(required_settings.union({"sigma_Y2"}).intersection(settings.columns))
    data = curves.merge(residuals[residual_cols], on="case_id", how="left", suffixes=("", "_residual"))
    data = data.merge(settings[setting_cols], on="case_id", how="left", suffixes=("", "_setting"))
    if "sigma_Y2" not in data.columns and "sigma_Y2_residual" in data.columns:
        data["sigma_Y2"] = data["sigma_Y2_residual"]

    data["fitted_response_value"] = np.nan
    data["observed_for_log"] = np.nan
    data["fitted_for_log"] = np.nan

    pumping = data["method"].eq("pumping")
    if pumping.any():
        t_hat = data.loc[pumping, "K_hat_pumping_m_s"].to_numpy(dtype=float) * data.loc[
            pumping, "aquifer_thickness_m"
        ].to_numpy(dtype=float)
        fitted_drawdown = np.asarray(
            [
                theis_drawdown(
                    radius=float(radius),
                    time=float(time),
                    transmissivity=float(transmissivity),
                    storativity=float(storativity),
                    pumping_rate=float(rate),
                )
                for radius, time, transmissivity, storativity, rate in zip(
                    data.loc[pumping, "observation_radius_m"].to_numpy(dtype=float),
                    data.loc[pumping, "time_s"].to_numpy(dtype=float),
                    t_hat,
                    data.loc[pumping, "storativity"].to_numpy(dtype=float),
                    data.loc[pumping, "pumping_rate_m3_s"].to_numpy(dtype=float),
                )
            ],
            dtype=float,
        )
        fitted_response = (
            4.0
            * np.pi
            * data.loc[pumping, "T0_m2_s"].to_numpy(dtype=float)
            * fitted_drawdown
            / data.loc[pumping, "pumping_rate_m3_s"].to_numpy(dtype=float)
        )
        observed = data.loc[pumping, "drawdown_m"].to_numpy(dtype=float)
        fallback = data.loc[pumping, "response_value"].to_numpy(dtype=float)
        observed = np.where(np.isfinite(observed) & (observed > 0.0), observed, fallback)
        data.loc[pumping, "fitted_response_value"] = fitted_response
        data.loc[pumping, "observed_for_log"] = observed
        data.loc[pumping, "fitted_for_log"] = fitted_drawdown

    slug = data["method"].eq("slug")
    if slug.any():
        t_hat = data.loc[slug, "K_hat_slug_m_s"].to_numpy(dtype=float) * data.loc[
            slug, "aquifer_thickness_m"
        ].to_numpy(dtype=float)
        conductance = 2.0 * np.pi * t_hat / np.log(
            data.loc[slug, "r_max_m"].to_numpy(dtype=float) / data.loc[slug, "well_radius_m"].to_numpy(dtype=float)
        )
        fitted_response = np.exp(
            -conductance
            * data.loc[slug, "time_s"].to_numpy(dtype=float)
            / data.loc[slug, "well_storage_m2"].to_numpy(dtype=float)
        )
        data.loc[slug, "fitted_response_value"] = fitted_response
        data.loc[slug, "observed_for_log"] = data.loc[slug, "response_value"].to_numpy(dtype=float)
        data.loc[slug, "fitted_for_log"] = fitted_response

    floor = np.finfo(float).tiny
    data["log_error"] = np.log(np.maximum(data["observed_for_log"].to_numpy(dtype=float), floor)) - np.log(
        np.maximum(data["fitted_for_log"].to_numpy(dtype=float), floor)
    )
    data["abs_log_error"] = np.abs(data["log_error"])
    return data


def summarize_fit_quality(
    fitted_curves: pd.DataFrame,
    residuals: pd.DataFrame,
    *,
    pass_thresholds: dict[str, float] | None = None,
    warning_thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Summarize curve-fit quality and attach method-specific residuals."""

    pass_thresholds = DEFAULT_PASS_THRESHOLDS if pass_thresholds is None else pass_thresholds
    warning_thresholds = DEFAULT_WARNING_THRESHOLDS if warning_thresholds is None else warning_thresholds
    _require_columns(fitted_curves, {"case_id", "method", "log_error"}, "fitted_curves")

    rows = []
    for (case_id, method), group in fitted_curves.groupby(["case_id", "method"], sort=False):
        used = _qc_signal_window(group, method)
        errors = used["log_error"].to_numpy(dtype=float)
        finite = errors[np.isfinite(errors)]
        if len(used) < 3 or len(finite) < 3:
            rmse = np.nan
            mean_error = np.nan
            max_abs = np.nan
            early_bias = np.nan
            late_bias = np.nan
        else:
            rmse = float(np.sqrt(np.mean(finite**2)))
            mean_error = float(np.mean(finite))
            max_abs = float(np.max(np.abs(finite)))
            ordered = used.sort_values("time_s")
            n_tail = max(1, int(np.ceil(len(ordered) * 0.25)))
            early_bias = float(ordered["log_error"].head(n_tail).mean())
            late_bias = float(ordered["log_error"].tail(n_tail).mean())
        rows.append(
            {
                "case_id": case_id,
                "method": method,
                "rmse_log_response": rmse,
                "mean_log_error": mean_error,
                "max_abs_log_error": max_abs,
                "early_mean_log_error": early_bias,
                "late_mean_log_error": late_bias,
                "fit_point_count": int(len(used)),
                "qc_class": classify_fit_quality(
                    method,
                    rmse,
                    pass_thresholds=pass_thresholds,
                    warning_thresholds=warning_thresholds,
                ),
            }
        )
    qc = pd.DataFrame(rows)
    residual_long = _residuals_to_long(residuals)
    return qc.merge(residual_long, on=["case_id", "method"], how="left")


def _qc_signal_window(group: pd.DataFrame, method: str) -> pd.DataFrame:
    if "used_for_fit" in group.columns:
        fit_window = group[group["used_for_fit"].astype(bool)]
        if len(fit_window) > 0:
            return fit_window
    if method != "slug":
        return group
    usable = group[
        (group["response_value"].to_numpy(dtype=float) > 1.0e-4)
        & (group["fitted_response_value"].to_numpy(dtype=float) > 1.0e-4)
    ]
    if len(usable) >= 3:
        return usable
    return group


def classify_fit_quality(
    method: str,
    rmse_log_response: float,
    *,
    pass_thresholds: dict[str, float] | None = None,
    warning_thresholds: dict[str, float] | None = None,
) -> str:
    """Classify fit quality as pass, warning, or fail from log-response RMSE."""

    pass_thresholds = DEFAULT_PASS_THRESHOLDS if pass_thresholds is None else pass_thresholds
    warning_thresholds = DEFAULT_WARNING_THRESHOLDS if warning_thresholds is None else warning_thresholds
    if not np.isfinite(rmse_log_response):
        return "fail"
    if rmse_log_response <= pass_thresholds[method]:
        return "pass"
    if rmse_log_response <= warning_thresholds[method]:
        return "warning"
    return "fail"


def select_representative_cases(qc: pd.DataFrame, method: str) -> dict[str, str]:
    """Pick good, median, and bad case IDs for response-level QC plotting."""

    method_qc = qc[qc["method"].eq(method)].sort_values("rmse_log_response").reset_index(drop=True)
    if method_qc.empty:
        raise ValueError(f"no QC rows for method {method!r}")
    pass_rows = method_qc[method_qc["qc_class"].eq("pass")]
    good = pass_rows.iloc[0] if not pass_rows.empty else method_qc.iloc[0]
    median = method_qc.iloc[len(method_qc) // 2]
    bad = method_qc.iloc[-1]
    return {"good": str(good["case_id"]), "median": str(median["case_id"]), "bad": str(bad["case_id"])}


def _residuals_to_long(residuals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in residuals.iterrows():
        common = {"case_id": row["case_id"], "sigma_Y2": row.get("sigma_Y2", np.nan)}
        rows.append(
            {
                **common,
                "method": "pumping",
                "log_residual": row["log_residual_pumping"],
                "abs_log_residual": abs(row["log_residual_pumping"]),
                "K_hat_m_s": row["K_hat_pumping_m_s"],
                "K_star_m_s": row["K_star_pumping_m_s"],
            }
        )
        rows.append(
            {
                **common,
                "method": "slug",
                "log_residual": row["log_residual_slug"],
                "abs_log_residual": abs(row["log_residual_slug"]),
                "K_hat_m_s": row["K_hat_slug_m_s"],
                "K_star_m_s": row["K_star_slug_m_s"],
            }
        )
    return pd.DataFrame(rows)


def _require_columns(table: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")

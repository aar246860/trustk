"""Conditional TRUST-K residual-prior models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
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
]


@dataclass(frozen=True)
class MethodPriorModel:
    method: str
    feature_columns: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    train_standardized_features: np.ndarray
    train_errors: np.ndarray
    global_sigma: float


@dataclass(frozen=True)
class ConditionalPriorModel:
    methods: dict[str, MethodPriorModel]
    feature_columns: tuple[str, ...] = tuple(FEATURE_COLUMNS)


def build_prior_dataset(
    residuals: pd.DataFrame,
    registry: pd.DataFrame,
    qc: pd.DataFrame,
    *,
    accepted_qc_classes: tuple[str, ...] = ("pass",),
) -> pd.DataFrame:
    """Create a long-format residual and Pi-control table."""

    required_registry = {
        "case_id",
        "sigma_Y2",
        "lambda1_over_RI",
        "lambda2_over_lambda1",
        "phi_lambda_rad",
        "Rmax_over_RI",
        "r_obs_over_RI_P",
        "tD_P_max",
        "C_D_w",
        "tD_S_max",
    }
    required_residuals = {
        "case_id",
        "K_star_pumping_m_s",
        "K_star_slug_m_s",
        "K_hat_pumping_m_s",
        "K_hat_slug_m_s",
        "log_residual_pumping",
        "log_residual_slug",
    }
    required_qc = {"case_id", "method", "qc_class"}
    _require_columns(registry, required_registry, "registry")
    _require_columns(residuals, required_residuals, "residuals")
    _require_columns(qc, required_qc, "qc")

    pi = _feature_table(registry)
    long_rows = []
    for _, row in residuals.iterrows():
        long_rows.append(
            {
                "case_id": row["case_id"],
                "method": "pumping",
                "K_hat_m_s": row["K_hat_pumping_m_s"],
                "K_star_m_s": row["K_star_pumping_m_s"],
                "log_residual": row["log_residual_pumping"],
            }
        )
        long_rows.append(
            {
                "case_id": row["case_id"],
                "method": "slug",
                "K_hat_m_s": row["K_hat_slug_m_s"],
                "K_star_m_s": row["K_star_slug_m_s"],
                "log_residual": row["log_residual_slug"],
            }
        )
    long = pd.DataFrame(long_rows)
    qc_small = qc[["case_id", "method", "qc_class"]].drop_duplicates()
    data = long.merge(qc_small, on=["case_id", "method"], how="left")
    data = data.merge(pi, on="case_id", how="left")
    data = data[data["qc_class"].isin(accepted_qc_classes)].copy()
    numeric = data.select_dtypes(include=[np.number]).columns
    data = data[np.isfinite(data[numeric]).all(axis=1)].reset_index(drop=True)
    return data


def split_train_validation(
    data: pd.DataFrame,
    *,
    validation_fraction: float = 0.25,
    seed: int = 20260517,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by case ID so pumping and slug rows from one case stay together."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    case_ids = np.array(sorted(data["case_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(case_ids)
    n_validation = max(1, int(round(len(case_ids) * validation_fraction)))
    validation_ids = set(case_ids[:n_validation])
    validation = data[data["case_id"].isin(validation_ids)].copy().reset_index(drop=True)
    train = data[~data["case_id"].isin(validation_ids)].copy().reset_index(drop=True)
    return train, validation


def fit_conditional_prior(
    train: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = tuple(FEATURE_COLUMNS),
    ridge_alpha: float = 1.0,
    min_sigma: float = 0.05,
) -> ConditionalPriorModel:
    """Fit method-specific conditional residual mean and scatter models."""

    models = {}
    for method, group in train.groupby("method"):
        x = group[list(feature_columns)].to_numpy(dtype=float)
        y = group["log_residual"].to_numpy(dtype=float)
        mean = np.mean(x, axis=0)
        scale = np.std(x, axis=0, ddof=1)
        scale = np.where(scale <= 0.0, 1.0, scale)
        xz = (x - mean) / scale
        design = _design_matrix(xz)
        penalty = ridge_alpha * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        train_mean = design @ coefficients
        errors = y - train_mean
        global_sigma = max(float(np.std(errors, ddof=1)), min_sigma)
        models[method] = MethodPriorModel(
            method=str(method),
            feature_columns=feature_columns,
            feature_mean=mean,
            feature_scale=scale,
            coefficients=coefficients,
            train_standardized_features=xz,
            train_errors=errors,
            global_sigma=global_sigma,
        )
    return ConditionalPriorModel(methods=models, feature_columns=feature_columns)


def predict_conditional_prior(model: ConditionalPriorModel, data: pd.DataFrame) -> pd.DataFrame:
    """Predict conditional residual mean, scatter, and corrected log target."""

    frames = []
    for method, group in data.groupby("method", sort=False):
        if method not in model.methods:
            continue
        method_model = model.methods[method]
        x = group[list(method_model.feature_columns)].to_numpy(dtype=float)
        xz = (x - method_model.feature_mean) / method_model.feature_scale
        mean = _design_matrix(xz) @ method_model.coefficients
        sigma = _local_sigma(method_model, xz)
        out = group.copy()
        out["predicted_log_residual_mean"] = mean
        out["predicted_log_residual_sigma"] = sigma
        out["predicted_log_K_star"] = np.log(out["K_hat_m_s"].to_numpy(dtype=float)) - mean
        out["actual_log_K_star"] = np.log(out["K_star_m_s"].to_numpy(dtype=float))
        out["standardized_residual_error"] = (out["log_residual"].to_numpy(dtype=float) - mean) / sigma
        frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def evaluate_holdout(model: ConditionalPriorModel, validation: pd.DataFrame) -> pd.DataFrame:
    """Evaluate conditional residual prediction on held-out cases."""

    predictions = predict_conditional_prior(model, validation)
    rows = []
    for method, group in predictions.groupby("method"):
        err = group["log_residual"].to_numpy(dtype=float) - group["predicted_log_residual_mean"].to_numpy(dtype=float)
        sigma = group["predicted_log_residual_sigma"].to_numpy(dtype=float)
        rows.append(
            {
                "method": method,
                "n": int(len(group)),
                "bias": float(np.mean(err)),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "mae": float(np.mean(np.abs(err))),
                "coverage_80": float(np.mean(np.abs(err) <= 1.2816 * sigma)),
                "coverage_95": float(np.mean(np.abs(err) <= 1.96 * sigma)),
                "mean_sigma": float(np.mean(sigma)),
                "nlpd": float(np.mean(0.5 * np.log(2.0 * np.pi * sigma**2) + 0.5 * (err / sigma) ** 2)),
            }
        )
    return pd.DataFrame(rows)


def evaluate_baselines(
    model: ConditionalPriorModel,
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    """Compare hard, method-constant, and conditional correction baselines."""

    constant = _method_constants(train)
    conditional = predict_conditional_prior(model, validation)
    rows = []
    for method, group in validation.groupby("method"):
        y = np.log(group["K_star_m_s"].to_numpy(dtype=float))
        log_khat = np.log(group["K_hat_m_s"].to_numpy(dtype=float))
        rows.append(_baseline_metrics(method, "hard", y, log_khat, None))

        const_mean, const_sigma = constant[method]
        const_prediction = log_khat - const_mean
        rows.append(_baseline_metrics(method, "method_constant", y, const_prediction, np.full(len(group), const_sigma)))

        cond = conditional[conditional["method"].eq(method)]
        rows.append(
            _baseline_metrics(
                method,
                "conditional",
                y,
                cond["predicted_log_K_star"].to_numpy(dtype=float),
                cond["predicted_log_residual_sigma"].to_numpy(dtype=float),
            )
        )
    return pd.DataFrame(rows)


def make_response_surface(
    model: ConditionalPriorModel,
    reference_data: pd.DataFrame,
    *,
    method: str,
    x_feature: str = "sigma_Y2",
    y_feature: str = "log_lambda1_over_RI",
    n_grid: int = 60,
) -> pd.DataFrame:
    """Generate a two-dimensional response surface for plotting."""

    method_data = reference_data[reference_data["method"].eq(method)]
    if method_data.empty:
        raise ValueError(f"no rows for method {method!r}")
    reference = method_data[list(model.feature_columns)].median(numeric_only=True).to_dict()
    x_values = np.linspace(float(method_data[x_feature].min()), float(method_data[x_feature].max()), n_grid)
    y_values = np.linspace(float(method_data[y_feature].min()), float(method_data[y_feature].max()), n_grid)
    rows = []
    for x in x_values:
        for y in y_values:
            row = {column: reference[column] for column in model.feature_columns}
            row[x_feature] = x
            row[y_feature] = y
            row.update({"case_id": "surface", "method": method, "K_hat_m_s": 1.0, "K_star_m_s": 1.0, "log_residual": 0.0})
            rows.append(row)
    surface = pd.DataFrame(rows)
    predicted = predict_conditional_prior(model, surface)
    return predicted[[x_feature, y_feature, "method", "predicted_log_residual_mean", "predicted_log_residual_sigma"]]


def _feature_table(registry: pd.DataFrame) -> pd.DataFrame:
    out = registry.copy()
    positive_log_columns = {
        "lambda1_over_RI": "log_lambda1_over_RI",
        "lambda2_over_lambda1": "log_lambda2_over_lambda1",
        "Rmax_over_RI": "log_Rmax_over_RI",
        "tD_P_max": "log_tD_P_max",
        "C_D_w": "log_C_D_w",
        "tD_S_max": "log_tD_S_max",
    }
    for source, target in positive_log_columns.items():
        out[target] = np.log(np.maximum(out[source].to_numpy(dtype=float), np.finfo(float).tiny))
    out["sin_phi_lambda"] = np.sin(out["phi_lambda_rad"].to_numpy(dtype=float))
    out["cos_phi_lambda"] = np.cos(out["phi_lambda_rad"].to_numpy(dtype=float))
    return out[["case_id", *FEATURE_COLUMNS]]


def _design_matrix(xz: np.ndarray) -> np.ndarray:
    xz = np.asarray(xz, dtype=float)
    columns = [np.ones(len(xz))]
    columns.extend([xz[:, i] for i in range(xz.shape[1])])
    columns.extend([xz[:, i] ** 2 for i in range(xz.shape[1])])
    return np.column_stack(columns)


def _local_sigma(model: MethodPriorModel, xz: np.ndarray, *, k: int = 80, min_sigma: float = 0.05) -> np.ndarray:
    train_x = model.train_standardized_features
    errors = model.train_errors
    sigmas = []
    k = min(k, len(errors))
    for row in xz:
        distances = np.sum((train_x - row) ** 2, axis=1)
        idx = np.argpartition(distances, k - 1)[:k]
        local = errors[idx]
        sigma = float(np.sqrt(np.mean(local**2)))
        sigmas.append(max(sigma, min_sigma, model.global_sigma))
    return np.asarray(sigmas, dtype=float)


def _method_constants(train: pd.DataFrame) -> dict[str, tuple[float, float]]:
    constants = {}
    for method, group in train.groupby("method"):
        values = group["log_residual"].to_numpy(dtype=float)
        constants[method] = (float(np.mean(values)), max(float(np.std(values, ddof=1)), 0.05))
    return constants


def _baseline_metrics(method: str, approach: str, truth: np.ndarray, prediction: np.ndarray, sigma: np.ndarray | None) -> dict:
    err = prediction - truth
    row = {
        "method": method,
        "approach": approach,
        "n": int(len(truth)),
        "bias_log": float(np.mean(err)),
        "rmse_log": float(np.sqrt(np.mean(err**2))),
        "mae_log": float(np.mean(np.abs(err))),
    }
    if sigma is not None:
        row["coverage_95"] = float(np.mean(np.abs(err) <= 1.96 * sigma))
        row["mean_interval_width_95"] = float(np.mean(2.0 * 1.96 * sigma))
    else:
        row["coverage_95"] = np.nan
        row["mean_interval_width_95"] = np.nan
    return row


def _require_columns(table: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")

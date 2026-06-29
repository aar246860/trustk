"""Field-style predictive validation products for the Lovelock case."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from scipy.optimize import least_squares

from trustk.api import MANUSCRIPT_METHOD_PRIORS
from trustk.data.extract_slug import SLUG_ANALYSIS_WORKBOOKS
from trustk.validation.lovelock import PUMP_START, PUMP_STOP, cyclic_theis_drawdown


@dataclass(frozen=True)
class FieldStyleValidation:
    """Tables needed by the field-style Figure 9 validation."""

    posterior_wells: pd.DataFrame
    pumping_predictions: pd.DataFrame
    slug_predictions: pd.DataFrame
    metrics: pd.DataFrame


def build_field_style_validation(
    processed_dir: str | Path = "data/processed",
    *,
    aquifer_thickness_m: float = 152.4,
    zip_path: str | Path = "field data/OFR2019_1133_DataRelease.zip",
) -> FieldStyleValidation:
    """Build predictive-check tables for a real field case with no known K truth."""

    processed = Path(processed_dir)
    summary = pd.read_csv(processed / "lovelock_overlap_summary.csv")
    comparison = pd.read_csv(processed / "lovelock_bongo_model_comparison.csv", parse_dates=["datetime"])
    slug_curves = _load_slug_segments_or_fallback(zip_path, processed / "slug_curves.csv")
    pumping_well = pd.read_csv(processed / "lovelock_bongo_pumping_well.csv").iloc[0]
    prior = _load_formal_joint_prior(processed)

    pumping_predictions, pumping_metrics = _build_pumping_leave_one_out(
        summary=summary,
        comparison=comparison,
        pumping_rate_m3_s=float(pumping_well["q_m3_s"]),
        aquifer_thickness_m=aquifer_thickness_m,
    )
    slug_predictions, slug_metrics = _build_slug_tail_validation(slug_curves)
    posterior_wells = _build_apparent_posterior(
        summary,
        prior,
        aquifer_thickness_m,
        pump_x_m=float(pumping_well["x_m"]),
        pump_y_m=float(pumping_well["y_m"]),
    )

    metrics = pd.concat([pumping_metrics, slug_metrics], ignore_index=True)
    metrics = metrics.sort_values(["validation_target", "method", "well"]).reset_index(drop=True)
    return FieldStyleValidation(
        posterior_wells=posterior_wells,
        pumping_predictions=pumping_predictions,
        slug_predictions=slug_predictions,
        metrics=metrics,
    )


def _load_formal_joint_prior(processed: Path) -> pd.DataFrame:
    prior_path = processed / "formal_joint_storage_transformation_prior.csv"
    if prior_path.exists():
        return pd.read_csv(prior_path)
    rows = [
        {
            "method": prior.method,
            "mean_log_residual": prior.mean_log_residual,
            "sd_log_residual": prior.sd_log_residual,
            "n_cases": prior.n_cases,
        }
        for prior in MANUSCRIPT_METHOD_PRIORS.values()
    ]
    return pd.DataFrame(rows)


def _build_pumping_leave_one_out(
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    pumping_rate_m3_s: float,
    aquifer_thickness_m: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    duration_s = (PUMP_STOP - PUMP_START).total_seconds()
    meta_cols = [
        "well",
        "distance_to_pump_m",
        "k_slug_m_s",
        "theis_transmissivity_m2_s",
        "theis_storativity",
        "null_rmse_m",
        "usgs_model_rmse_m",
        "theis_rmse_m",
    ]
    data = comparison.merge(summary[meta_cols], on="well", how="inner")
    data = data[
        (data["elapsed_since_pump_start_s"] > 6 * 3600)
        & (data["measured_drawdown_m"] >= 0)
    ].copy()

    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    for held_out in summary["well"].tolist():
        train_wells = [well for well in summary["well"].tolist() if well != held_out]
        bias, storativity, train_rmse = _fit_hard_slug_transfer(
            data=data,
            train_wells=train_wells,
            pumping_rate_m3_s=pumping_rate_m3_s,
            aquifer_thickness_m=aquifer_thickness_m,
            duration_s=duration_s,
        )
        held = data[data["well"].eq(held_out)].copy()
        central = _predict_hard_slug_rows(
            held,
            bias_factor=bias,
            storativity=storativity,
            pumping_rate_m3_s=pumping_rate_m3_s,
            aquifer_thickness_m=aquifer_thickness_m,
            duration_s=duration_s,
        )
        held["hard_slug_loo_drawdown_m"] = central
        held["hard_slug_loo_p05_m"] = np.maximum(0.0, central - 1.96 * train_rmse)
        held["hard_slug_loo_p95_m"] = central + 1.96 * train_rmse
        held["hard_slug_loo_bias_factor"] = bias
        held["hard_slug_loo_storativity"] = storativity
        held["hard_slug_loo_training_rmse_m"] = train_rmse
        held["time_days"] = held["elapsed_since_pump_start_s"] / 86400.0
        prediction_rows.append(
            held[
                [
                    "well",
                    "datetime",
                    "phase",
                    "time_days",
                    "distance_to_pump_m",
                    "elapsed_since_pump_start_s",
                    "measured_drawdown_m",
                    "simulated_drawdown_m",
                    "hard_slug_loo_drawdown_m",
                    "hard_slug_loo_p05_m",
                    "hard_slug_loo_p95_m",
                    "hard_slug_loo_bias_factor",
                    "hard_slug_loo_storativity",
                    "hard_slug_loo_training_rmse_m",
                ]
            ]
        )

        y = held["measured_drawdown_m"].to_numpy(float)
        null_rmse = _rmse(y, np.zeros_like(y))
        metric_rows.append(
            _metric_row(
                held_out,
                "pumping_response",
                "Hard slug LOO",
                _rmse(y, central),
                null_rmse,
                "leave-one-well-out transfer from slug K",
            )
        )

    for row in summary.itertuples(index=False):
        metric_rows.append(
            _metric_row(
                row.well,
                "pumping_response",
                "USGS calibrated",
                float(row.usgs_model_rmse_m),
                float(row.null_rmse_m),
                "distributed calibrated MODFLOW comparison",
            )
        )
        metric_rows.append(
            _metric_row(
                row.well,
                "pumping_response",
                "Independent Theis",
                float(row.theis_rmse_m),
                float(row.null_rmse_m),
                "single-well in-sample curve-fit reference",
            )
        )

    predictions = pd.concat(prediction_rows, ignore_index=True).sort_values(["well", "datetime"])
    metrics = pd.DataFrame(metric_rows)
    return predictions.reset_index(drop=True), metrics


def _fit_hard_slug_transfer(
    data: pd.DataFrame,
    train_wells: list[str],
    pumping_rate_m3_s: float,
    aquifer_thickness_m: float,
    duration_s: float,
) -> tuple[float, float, float]:
    train = data[data["well"].isin(train_wells)].copy()
    y = train["measured_drawdown_m"].to_numpy(float)
    scale = max(float(np.nanpercentile(y, 90)), 0.01)

    def residual(log_params: np.ndarray) -> np.ndarray:
        bias_factor, storativity = np.exp(log_params)
        pred = _predict_hard_slug_rows(
            train,
            bias_factor=float(bias_factor),
            storativity=float(storativity),
            pumping_rate_m3_s=pumping_rate_m3_s,
            aquifer_thickness_m=aquifer_thickness_m,
            duration_s=duration_s,
        )
        return (pred - y) / scale

    result = least_squares(
        residual,
        x0=np.log([1.0, 1e-3]),
        bounds=(np.log([1e-5, 1e-8]), np.log([1e5, 0.5])),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=300,
    )
    bias, storativity = np.exp(result.x)
    train_pred = _predict_hard_slug_rows(
        train,
        bias_factor=float(bias),
        storativity=float(storativity),
        pumping_rate_m3_s=pumping_rate_m3_s,
        aquifer_thickness_m=aquifer_thickness_m,
        duration_s=duration_s,
    )
    return float(bias), float(storativity), _rmse(y, train_pred)


def _predict_hard_slug_rows(
    rows: pd.DataFrame,
    bias_factor: float,
    storativity: float,
    pumping_rate_m3_s: float,
    aquifer_thickness_m: float,
    duration_s: float,
) -> np.ndarray:
    transmissivity = rows["k_slug_m_s"].to_numpy(float) * aquifer_thickness_m * bias_factor
    radius = rows["distance_to_pump_m"].to_numpy(float)
    elapsed = rows["elapsed_since_pump_start_s"].to_numpy(float)
    return np.array(
        [
            cyclic_theis_drawdown(r, np.array([t]), T, storativity, pumping_rate_m3_s, duration_s)[0]
            for r, t, T in zip(radius, elapsed, transmissivity)
        ],
        dtype=float,
    )


def _load_slug_segments_or_fallback(zip_path: str | Path, fallback_slug_curves: Path) -> pd.DataFrame:
    zip_file = Path(zip_path)
    if zip_file.exists():
        return _extract_slug_segments(zip_file)
    curves = pd.read_csv(fallback_slug_curves, parse_dates=["datetime"])
    curves["segment"] = 1
    signal = curves.groupby("well")["drawdown_m"].transform(lambda s: np.nanpercentile(np.abs(s), 98))
    curves["recovery_amplitude_norm"] = np.clip(np.abs(curves["drawdown_m"]) / signal, 0.0, 1.25)
    curves["plot_elapsed_s"] = np.maximum(curves["elapsed_s"], 1.0)
    curves["time_min"] = curves["elapsed_s"] / 60.0
    return curves


def _extract_slug_segments(zip_path: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as archive:
        for well, member in SLUG_ANALYSIS_WORKBOOKS.items():
            workbook = openpyxl.load_workbook(
                io.BytesIO(archive.read(member)),
                read_only=True,
                data_only=True,
            )
            rows.append(_extract_slug_segments_from_workbook(workbook, well))
    return pd.concat(rows, ignore_index=True).sort_values(["well", "segment", "elapsed_s"])


def _extract_slug_segments_from_workbook(workbook: openpyxl.Workbook, well: str) -> pd.DataFrame:
    output = workbook["OUTPUT"]
    table = []
    for values in output.iter_rows(min_row=2, values_only=True):
        if len(values) < 11:
            continue
        event, timestamp, feet = values[8], values[9], values[10]
        if event is None or not hasattr(timestamp, "year") or not isinstance(feet, (int, float)):
            continue
        table.append({"event": str(event).strip().lower(), "datetime": pd.Timestamp(timestamp), "feet": float(feet)})

    static_level = None
    segment = 0
    after_injection = False
    buffered: list[dict] = []
    segment_rows: list[dict] = []
    for item in table:
        if item["event"] == "use" and static_level is None:
            static_level = item["feet"]
        if item["event"] == "inject":
            if buffered:
                segment_rows.extend(buffered)
            buffered = []
            segment += 1
            after_injection = True
            continue
        if item["event"] == "use" and after_injection and static_level is not None:
            buffered.append(
                {
                    "well": well,
                    "segment": segment,
                    "datetime": item["datetime"],
                    "displacement_ft": item["feet"] - static_level,
                    "drawdown_m": (item["feet"] - static_level) * 0.3048,
                }
            )
    if buffered:
        segment_rows.extend(buffered)
    if not segment_rows:
        return pd.DataFrame()

    data = pd.DataFrame(segment_rows)
    parts = []
    for segment_id, group in data.groupby("segment"):
        segment_data = group.sort_values("datetime").copy()
        segment_data["elapsed_s"] = (
            segment_data["datetime"] - segment_data["datetime"].iloc[0]
        ).dt.total_seconds()
        amplitude = float(np.nanmax(np.abs(segment_data["displacement_ft"])))
        if amplitude <= 0:
            continue
        segment_data["recovery_amplitude_norm"] = np.clip(
            np.abs(segment_data["displacement_ft"]) / amplitude,
            0.0,
            1.25,
        )
        positive_step = segment_data.loc[segment_data["elapsed_s"] > 0, "elapsed_s"].min()
        replacement = float(positive_step) * 0.5 if np.isfinite(positive_step) else 1.0
        segment_data["plot_elapsed_s"] = np.where(segment_data["elapsed_s"] <= 0, replacement, segment_data["elapsed_s"])
        segment_data["time_min"] = segment_data["elapsed_s"] / 60.0
        parts.append(segment_data)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _build_slug_tail_validation(slug_curves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[pd.DataFrame] = []
    holdout_by_well: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for (well, segment), group in slug_curves.groupby(["well", "segment"], sort=True):
        curve = group.sort_values("elapsed_s").copy()
        valid = curve[
            (curve["elapsed_s"] > 0)
            & (curve["recovery_amplitude_norm"] > 0.04)
            & (curve["recovery_amplitude_norm"] < 1.05)
        ].copy()
        if len(valid) < 12:
            prediction_rows.append(_unfit_slug_segment(curve))
            continue
        split_s = float(valid["elapsed_s"].quantile(0.60))
        fit = valid[valid["elapsed_s"] <= split_s].copy()
        holdout = valid[valid["elapsed_s"] > split_s].copy()
        if len(fit) < 6 or len(holdout) < 4:
            prediction_rows.append(_unfit_slug_segment(curve))
            continue
        x = fit["elapsed_s"].to_numpy(float)
        y = np.log(np.clip(fit["recovery_amplitude_norm"].to_numpy(float), 1e-4, None))
        slope, intercept = np.polyfit(x, y, deg=1)
        prediction = np.exp(intercept + slope * curve["elapsed_s"].to_numpy(float))
        curve["semi_log_tail_prediction"] = np.clip(prediction, 0.0, 1.25)
        curve["holdout_split_s"] = split_s
        curve["is_tail_holdout"] = curve["elapsed_s"] > split_s
        prediction_rows.append(
            curve[
                [
                    "well",
                    "segment",
                    "datetime",
                    "elapsed_s",
                    "plot_elapsed_s",
                    "time_min",
                    "drawdown_m",
                    "recovery_amplitude_norm",
                    "semi_log_tail_prediction",
                    "holdout_split_s",
                    "is_tail_holdout",
                ]
            ]
        )

        observed = holdout["recovery_amplitude_norm"].to_numpy(float)
        predicted = np.exp(intercept + slope * holdout["elapsed_s"].to_numpy(float))
        predicted = np.clip(predicted, 0.0, 1.25)
        holdout_by_well.setdefault(well, []).append((observed, predicted))

    metric_rows = []
    for well, values in sorted(holdout_by_well.items()):
        observed = np.concatenate([item[0] for item in values])
        predicted = np.concatenate([item[1] for item in values])
        null_rmse = _rmse(observed, np.zeros_like(observed))
        metric_rows.append(
            _metric_row(
                well,
                "slug_recovery",
                "Semi-log slug tail",
                _rmse(observed, predicted),
                null_rmse,
                "segmented early-response fit evaluated on tail response",
            )
        )

    predictions = pd.concat(prediction_rows, ignore_index=True).sort_values(["well", "segment", "elapsed_s"])
    metrics = pd.DataFrame(metric_rows)
    return predictions.reset_index(drop=True), metrics


def _unfit_slug_segment(curve: pd.DataFrame) -> pd.DataFrame:
    segment = curve.copy()
    segment["semi_log_tail_prediction"] = np.nan
    segment["holdout_split_s"] = np.nan
    segment["is_tail_holdout"] = False
    return segment[
        [
            "well",
            "segment",
            "datetime",
            "elapsed_s",
            "plot_elapsed_s",
            "time_min",
            "drawdown_m",
            "recovery_amplitude_norm",
            "semi_log_tail_prediction",
            "holdout_split_s",
            "is_tail_holdout",
        ]
    ]


def _build_apparent_posterior(
    summary: pd.DataFrame,
    prior: pd.DataFrame,
    aquifer_thickness_m: float,
    pump_x_m: float,
    pump_y_m: float,
) -> pd.DataFrame:
    prior_by_method = prior.set_index("method")
    rows = []
    for row in summary.itertuples(index=False):
        observations = [
            (
                np.log(float(row.k_slug_m_s)) - float(prior_by_method.loc["slug_bouwer_rice", "mean_log_residual"]),
                float(prior_by_method.loc["slug_bouwer_rice", "sd_log_residual"]),
            ),
            (
                np.log(float(row.theis_transmissivity_m2_s) / aquifer_thickness_m)
                - float(prior_by_method.loc["pumping", "mean_log_residual"]),
                float(prior_by_method.loc["pumping", "sd_log_residual"]),
            ),
        ]
        precision = sum(1.0 / sigma**2 for _, sigma in observations)
        mean_log_k = sum(value / sigma**2 for value, sigma in observations) / precision
        sd_log_k = np.sqrt(1.0 / precision)
        rows.append(
            {
                "well": row.well,
                "x_m": float(row.x_m),
                "y_m": float(row.y_m),
                "model_layer": int(row.model_layer),
                "distance_to_pump_m": float(row.distance_to_pump_m),
                "peak_measured_drawdown_m": float(row.peak_measured_drawdown_m),
                "slug_log10K": float(np.log10(row.k_slug_m_s)),
                "pumping_log10K": float(np.log10(row.theis_transmissivity_m2_s / aquifer_thickness_m)),
                "apparent_logK_mean": float(mean_log_k),
                "apparent_logK_sd": float(sd_log_k),
                "apparent_log10K_mean": float(mean_log_k / np.log(10.0)),
            }
        )
    posterior = pd.DataFrame(rows).sort_values(["distance_to_pump_m", "well"]).reset_index(drop=True)
    posterior["x_rel_km"] = (posterior["x_m"] - pump_x_m) / 1000.0
    posterior["y_rel_km"] = (posterior["y_m"] - pump_y_m) / 1000.0
    return posterior


def _metric_row(
    well: str,
    validation_target: str,
    method: str,
    rmse: float,
    null_rmse: float,
    validation_scheme: str,
) -> dict:
    return {
        "well": well,
        "validation_target": validation_target,
        "method": method,
        "rmse": float(rmse),
        "null_rmse": float(null_rmse),
        "normalized_rmse": float(rmse / null_rmse) if null_rmse > 0 else np.nan,
        "validation_scheme": validation_scheme,
    }


def _rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.nanmean((observed - predicted) ** 2)))

"""Field-validation helpers for the USGS Lovelock Bongo pumping test."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from trustk.analytical.theis import theis_drawdown
from trustk.data.extract_pumping import (
    BONGO_COORDS,
    FT_TO_M,
    GPM_TO_M3_S,
    OVERLAP_WELLS,
    PUMPING_TEST2_WORKBOOK,
    normalize_well_name,
)

BONGO_DIS = (
    "OFR2019_1133_DataRelease/Multi-WellPumpingTests/analyses/MF_LH-Bongo/"
    "INPUT_LH-Bongo.dis.txt"
)
BONGO_WEL = (
    "OFR2019_1133_DataRelease/Multi-WellPumpingTests/analyses/MF_LH-Bongo/"
    "INPUT_LH-Bongo.wel.txt"
)
BONGO_MEASURED = (
    "OFR2019_1133_DataRelease/Multi-WellPumpingTests/analyses/MF_LH-Bongo/"
    "Pcomp_LH-Bongo.WLmeasured.txt"
)
BONGO_SIMULATED = (
    "OFR2019_1133_DataRelease/Multi-WellPumpingTests/analyses/MF_LH-Bongo/"
    "Pcomp_LH-Bongo.WLsimulated.txt"
)

FT3_DAY_TO_M3_S = 0.028316846592 / 86400.0
# The USGS report gives the MWP2 aquifer-test interval as
# 2017-11-28 10:53 to 2017-12-04 11:28.
PUMP_START = pd.Timestamp("2017-11-28 10:53:00")
PUMP_STOP = pd.Timestamp("2017-12-04 11:28:00")


@dataclass(frozen=True)
class LovelockValidation:
    pumping_well: pd.DataFrame
    model_comparison: pd.DataFrame
    overlap_summary: pd.DataFrame
    theis_fits: pd.DataFrame
    hard_slug_predictions: pd.DataFrame
    hard_slug_summary: pd.DataFrame


def _archive_text(archive: zipfile.ZipFile, member: str) -> str:
    return archive.read(member).decode("latin-1", errors="replace")


def _parse_drawdown_text(text: str, value_name: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        well = normalize_well_name(parts[0])
        dt = pd.to_datetime(f"{parts[1]} {parts[2]}", errors="coerce")
        if pd.isna(dt):
            continue
        rows.append(
            {
                "well": well,
                "datetime": dt,
                f"{value_name}_drawdown_ft": float(parts[3]),
                f"{value_name}_drawdown_m": float(parts[3]) * FT_TO_M,
            }
        )
    return pd.DataFrame(rows).sort_values(["well", "datetime"]).reset_index(drop=True)


def _phase_for_time(t: pd.Timestamp) -> str:
    if t < PUMP_START:
        return "pretest"
    if t <= PUMP_STOP:
        return "pumping"
    return "recovery"


def extract_bongo_model_comparison(zip_path: str | Path) -> pd.DataFrame:
    """Extract measured and USGS MODFLOW-simulated Bongo drawdowns."""

    with zipfile.ZipFile(zip_path) as archive:
        measured = _parse_drawdown_text(_archive_text(archive, BONGO_MEASURED), "measured")
        simulated = _parse_drawdown_text(_archive_text(archive, BONGO_SIMULATED), "simulated")

    data = measured.merge(simulated, on=["well", "datetime"], how="inner")
    data["residual_m"] = data["measured_drawdown_m"] - data["simulated_drawdown_m"]
    data["phase"] = data["datetime"].map(_phase_for_time)
    data["elapsed_since_pump_start_s"] = (data["datetime"] - PUMP_START).dt.total_seconds()
    data["elapsed_since_recovery_start_s"] = (data["datetime"] - PUMP_STOP).dt.total_seconds()
    data["is_slug_overlap_well"] = data["well"].isin(OVERLAP_WELLS)
    return data.sort_values(["well", "datetime"]).reset_index(drop=True)


def _parse_dis_arrays(text: str) -> tuple[float, float, list[float], list[float]]:
    upper_left = re.search(r"Upper left corner: \(([^,]+), ([^)]+)\)", text)
    if not upper_left:
        raise ValueError("Could not find upper-left grid coordinate in DIS file")
    x0_ft = float(upper_left.group(1))
    y0_ft = float(upper_left.group(2))
    lines = text.splitlines()

    def read_array(marker: str, count: int) -> list[float]:
        start = next(i for i, line in enumerate(lines) if marker in line) + 1
        values: list[float] = []
        for line in lines[start:]:
            if "INTERNAL" in line or "CONSTANT" in line:
                if values:
                    break
                continue
            values.extend(float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:E[-+]?\d+)?", line))
            if len(values) >= count:
                return values[:count]
        raise ValueError(f"Could not parse {count} values for {marker}")

    delr_ft = read_array("# DELR", 117)
    delc_ft = read_array("# DELC", 157)
    return x0_ft, y0_ft, delr_ft, delc_ft


def extract_bongo_pumping_well(zip_path: str | Path) -> pd.DataFrame:
    """Locate the Bongo pumping well from MODFLOW WEL and DIS files."""

    with zipfile.ZipFile(zip_path) as archive:
        dis_text = _archive_text(archive, BONGO_DIS)
        wel_text = _archive_text(archive, BONGO_WEL)
    x0_ft, y0_ft, delr_ft, delc_ft = _parse_dis_arrays(dis_text)

    matches = []
    for line in wel_text.splitlines():
        parts = line.split()
        if len(parts) == 4:
            try:
                layer = int(float(parts[0]))
                row = int(float(parts[1]))
                col = int(float(parts[2]))
                q_ft3_day = float(parts[3])
            except ValueError:
                continue
            if abs(q_ft3_day) > 0:
                matches.append((layer, row, col, q_ft3_day))
    if not matches:
        raise ValueError("No active pumping well found in WEL file")

    layer, row, col, q_ft3_day = matches[0]
    x_ft = x0_ft + sum(delr_ft[: col - 1]) + 0.5 * delr_ft[col - 1]
    y_ft = y0_ft - sum(delc_ft[: row - 1]) - 0.5 * delc_ft[row - 1]
    return pd.DataFrame(
        [
            {
                "well": "BONGO",
                "model_layer": layer,
                "model_row": row,
                "model_col": col,
                "x_m": x_ft * FT_TO_M,
                "y_m": y_ft * FT_TO_M,
                "q_ft3_day": q_ft3_day,
                "q_m3_s": abs(q_ft3_day) * FT3_DAY_TO_M3_S,
                "source_file": BONGO_WEL,
            }
        ]
    )


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required processed file not found: {path}")
    return pd.read_csv(path)


def build_overlap_summary(
    comparison: pd.DataFrame,
    pumping_well: pd.DataFrame,
    processed_dir: str | Path = "data/processed",
) -> pd.DataFrame:
    """Join slug-test K, observation coordinates, distance, and drawdown metrics."""

    processed = Path(processed_dir)
    slug = _load_csv(processed / "slug_tests.csv")
    coords = _load_csv(processed / "pumping_test2_well_coordinates.csv")
    pump = pumping_well.iloc[0]

    overlap = (
        coords[coords["well"].isin(OVERLAP_WELLS)]
        .merge(slug, on="well", how="inner", suffixes=("", "_slug"))
        .copy()
    )
    overlap["distance_to_pump_m"] = np.hypot(overlap["x_m"] - pump["x_m"], overlap["y_m"] - pump["y_m"])

    metrics = []
    for well, group in comparison[comparison["well"].isin(OVERLAP_WELLS)].groupby("well"):
        pumping = group[group["phase"] == "pumping"]
        recovery = group[group["phase"] == "recovery"]
        metrics.append(
            {
                "well": well,
                "n_response_points": len(group),
                "peak_measured_drawdown_m": group["measured_drawdown_m"].max(),
                "end_pumping_measured_drawdown_m": pumping["measured_drawdown_m"].iloc[-1],
                "end_recovery_measured_drawdown_m": recovery["measured_drawdown_m"].iloc[-1],
                "usgs_model_rmse_m": float(np.sqrt(np.mean(group["residual_m"] ** 2))),
                "usgs_model_bias_m": float(group["residual_m"].mean()),
            }
        )

    summary = overlap.merge(pd.DataFrame(metrics), on="well", how="left")
    summary["xy_key"] = summary["x_m"].round(2).astype(str) + "_" + summary["y_m"].round(2).astype(str)
    summary["co_located_wells"] = summary.groupby("xy_key")["well"].transform(lambda s: ",".join(sorted(s)))
    return summary.sort_values(["distance_to_pump_m", "well"]).reset_index(drop=True)


def cyclic_theis_drawdown(
    radius_m: np.ndarray | float,
    elapsed_s: np.ndarray,
    transmissivity_m2_s: float,
    storativity: float,
    pumping_rate_m3_s: float,
    duration_s: float,
) -> np.ndarray:
    """Theis drawdown for pumping followed by recovery using superposition."""

    elapsed = np.asarray(elapsed_s, dtype=float)
    positive = np.maximum(elapsed, 1.0)
    drawdown = theis_drawdown(
        radius_m,
        positive,
        transmissivity_m2_s,
        storativity,
        pumping_rate_m3_s,
    )
    after_stop = elapsed > duration_s
    if np.any(after_stop):
        drawdown[after_stop] -= theis_drawdown(
            radius_m,
            elapsed[after_stop] - duration_s,
            transmissivity_m2_s,
            storativity,
            pumping_rate_m3_s,
        )
    drawdown[elapsed <= 0] = 0.0
    return drawdown


def fit_theis_overlap(
    comparison: pd.DataFrame,
    overlap_summary: pd.DataFrame,
    pumping_rate_m3_s: float,
) -> pd.DataFrame:
    """Fit a homogeneous Theis response independently at each overlap well."""

    duration_s = (PUMP_STOP - PUMP_START).total_seconds()
    rows = []
    for _, meta in overlap_summary.iterrows():
        well = meta["well"]
        group = comparison[
            (comparison["well"] == well)
            & (comparison["elapsed_since_pump_start_s"] > 6 * 3600)
            & (comparison["measured_drawdown_m"] >= 0)
        ].copy()
        if len(group) < 8:
            continue
        t = group["elapsed_since_pump_start_s"].to_numpy(float)
        y = group["measured_drawdown_m"].to_numpy(float)
        radius = float(meta["distance_to_pump_m"])

        def residual(log_params: np.ndarray) -> np.ndarray:
            transmissivity = float(np.exp(log_params[0]))
            storativity = float(np.exp(log_params[1]))
            pred = cyclic_theis_drawdown(radius, t, transmissivity, storativity, pumping_rate_m3_s, duration_s)
            scale = max(float(np.nanpercentile(y, 90)), 0.01)
            return (pred - y) / scale

        result = least_squares(
            residual,
            x0=np.log([1e-2, 1e-3]),
            bounds=(np.log([1e-7, 1e-8]), np.log([10.0, 0.5])),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=300,
        )
        pred = cyclic_theis_drawdown(
            radius,
            t,
            float(np.exp(result.x[0])),
            float(np.exp(result.x[1])),
            pumping_rate_m3_s,
            duration_s,
        )
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        rows.append(
            {
                "well": well,
                "theis_transmissivity_m2_s": float(np.exp(result.x[0])),
                "theis_storativity": float(np.exp(result.x[1])),
                "theis_rmse_m": rmse,
                "theis_status": int(result.status),
                "theis_n_points": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values("well").reset_index(drop=True)


def fit_hard_slug_baseline(
    comparison: pd.DataFrame,
    overlap_summary: pd.DataFrame,
    pumping_rate_m3_s: float,
    aquifer_thickness_m: float = 152.4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a global bias and storage while treating slug K as hard local data.

    The baseline is intentionally simple: each observation well receives
    transmissivity from its Bouwer-Rice slug-test K multiplied by one assumed
    aquifer thickness and one global multiplicative correction. A single
    storativity is shared by all wells. This represents the common practice of
    transferring interpreted K values directly into a predictive model.
    """

    duration_s = (PUMP_STOP - PUMP_START).total_seconds()
    meta = overlap_summary[
        [
            "well",
            "distance_to_pump_m",
            "k_slug_m_s",
        ]
    ].copy()
    data = comparison.merge(meta, on="well", how="inner")
    data = data[
        (data["elapsed_since_pump_start_s"] > 6 * 3600)
        & (data["measured_drawdown_m"] >= 0)
    ].copy()
    if data.empty:
        raise ValueError("No positive overlap drawdowns available for hard-slug baseline")

    t = data["elapsed_since_pump_start_s"].to_numpy(float)
    y = data["measured_drawdown_m"].to_numpy(float)
    radius = data["distance_to_pump_m"].to_numpy(float)
    base_t = data["k_slug_m_s"].to_numpy(float) * aquifer_thickness_m
    scale = max(float(np.nanpercentile(y, 90)), 0.01)

    def residual(log_params: np.ndarray) -> np.ndarray:
        bias = float(np.exp(log_params[0]))
        storativity = float(np.exp(log_params[1]))
        pred = np.empty_like(y)
        for i in range(len(y)):
            pred[i] = cyclic_theis_drawdown(
                radius[i],
                np.array([t[i]]),
                float(base_t[i] * bias),
                storativity,
                pumping_rate_m3_s,
                duration_s,
            )[0]
        return (pred - y) / scale

    result = least_squares(
        residual,
        x0=np.log([1.0, 1e-3]),
        bounds=(np.log([1e-5, 1e-8]), np.log([1e5, 0.5])),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=300,
    )

    bias = float(np.exp(result.x[0]))
    storativity = float(np.exp(result.x[1]))
    predictions = data[
        [
            "well",
            "datetime",
            "phase",
            "elapsed_since_pump_start_s",
            "measured_drawdown_m",
            "simulated_drawdown_m",
            "distance_to_pump_m",
            "k_slug_m_s",
        ]
    ].copy()
    predictions["hard_slug_bias_factor"] = bias
    predictions["hard_slug_storativity"] = storativity
    predictions["hard_slug_transmissivity_m2_s"] = predictions["k_slug_m_s"] * aquifer_thickness_m * bias
    predictions["hard_slug_drawdown_m"] = [
        cyclic_theis_drawdown(
            float(row.distance_to_pump_m),
            np.array([float(row.elapsed_since_pump_start_s)]),
            float(row.hard_slug_transmissivity_m2_s),
            storativity,
            pumping_rate_m3_s,
            duration_s,
        )[0]
        for row in predictions.itertuples(index=False)
    ]
    predictions["hard_slug_residual_m"] = (
        predictions["measured_drawdown_m"] - predictions["hard_slug_drawdown_m"]
    )

    summary_rows = []
    for well, group in predictions.groupby("well"):
        null_rmse = float(np.sqrt(np.mean(group["measured_drawdown_m"] ** 2)))
        hard_slug_rmse = float(np.sqrt(np.mean(group["hard_slug_residual_m"] ** 2)))
        summary_rows.append(
            {
                "well": well,
                "null_rmse_m": null_rmse,
                "hard_slug_rmse_m": hard_slug_rmse,
                "hard_slug_skill_vs_null": 1.0 - hard_slug_rmse / null_rmse if null_rmse > 0 else np.nan,
                "hard_slug_bias_m": float(group["hard_slug_residual_m"].mean()),
                "hard_slug_n_points": int(len(group)),
                "hard_slug_bias_factor": bias,
                "hard_slug_storativity": storativity,
                "hard_slug_aquifer_thickness_m": aquifer_thickness_m,
                "hard_slug_status": int(result.status),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("well").reset_index(drop=True)
    return predictions.sort_values(["well", "datetime"]).reset_index(drop=True), summary


def extract_lovelock_validation(
    zip_path: str | Path,
    processed_dir: str | Path = "data/processed",
) -> LovelockValidation:
    pumping_well = extract_bongo_pumping_well(zip_path)
    comparison = extract_bongo_model_comparison(zip_path)
    overlap = build_overlap_summary(comparison, pumping_well, processed_dir)
    theis = fit_theis_overlap(comparison, overlap, float(pumping_well.iloc[0]["q_m3_s"]))
    overlap = overlap.merge(theis, on="well", how="left")
    hard_slug_predictions, hard_slug_summary = fit_hard_slug_baseline(
        comparison,
        overlap,
        float(pumping_well.iloc[0]["q_m3_s"]),
    )
    overlap = overlap.merge(hard_slug_summary, on="well", how="left")
    return LovelockValidation(
        pumping_well=pumping_well,
        model_comparison=comparison,
        overlap_summary=overlap,
        theis_fits=theis,
        hard_slug_predictions=hard_slug_predictions,
        hard_slug_summary=hard_slug_summary,
    )


def write_lovelock_outputs(validation: LovelockValidation, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    validation.pumping_well.to_csv(out / "lovelock_bongo_pumping_well.csv", index=False)
    validation.model_comparison.to_csv(out / "lovelock_bongo_model_comparison.csv", index=False)
    validation.overlap_summary.to_csv(out / "lovelock_overlap_summary.csv", index=False)
    validation.theis_fits.to_csv(out / "lovelock_theis_fits.csv", index=False)
    validation.hard_slug_predictions.to_csv(out / "lovelock_hard_slug_predictions.csv", index=False)
    validation.hard_slug_summary.to_csv(out / "lovelock_hard_slug_summary.csv", index=False)


def write_lovelock_report(validation: LovelockValidation, report_path: str | Path) -> None:
    report = {
        "source_workbook": PUMPING_TEST2_WORKBOOK,
        "measured_drawdown_file": BONGO_MEASURED,
        "simulated_drawdown_file": BONGO_SIMULATED,
        "overlap_wells": sorted(OVERLAP_WELLS),
        "pumping_well": validation.pumping_well.iloc[0].to_dict(),
        "n_model_comparison_rows": int(len(validation.model_comparison)),
        "n_overlap_wells": int(len(validation.overlap_summary)),
        "median_usgs_model_rmse_m": float(validation.overlap_summary["usgs_model_rmse_m"].median()),
        "median_theis_rmse_m": float(validation.overlap_summary["theis_rmse_m"].median()),
        "median_hard_slug_rmse_m": float(validation.overlap_summary["hard_slug_rmse_m"].median()),
        "hard_slug_global_bias_factor": float(validation.hard_slug_summary["hard_slug_bias_factor"].iloc[0]),
        "hard_slug_global_storativity": float(validation.hard_slug_summary["hard_slug_storativity"].iloc[0]),
    }
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, help="Path to OFR2019_1133_DataRelease.zip")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--report", default="outputs/reports/lovelock_field_validation.json")
    args = parser.parse_args(argv)

    validation = extract_lovelock_validation(args.zip, args.processed_dir)
    write_lovelock_outputs(validation, args.processed_dir)
    write_lovelock_report(validation, args.report)
    print(f"comparison_rows={len(validation.model_comparison)}")
    print(f"overlap_wells={len(validation.overlap_summary)}")
    print(f"median_usgs_model_rmse_m={validation.overlap_summary['usgs_model_rmse_m'].median():.4g}")
    print(f"median_theis_rmse_m={validation.overlap_summary['theis_rmse_m'].median():.4g}")
    print(f"median_hard_slug_rmse_m={validation.overlap_summary['hard_slug_rmse_m'].median():.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

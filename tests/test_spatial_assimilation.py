import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trustk.assimilation.spatial import (
    assimilate_linear_gaussian,
    build_support_operator,
    evaluate_spatial_posterior,
    squared_exponential_covariance,
)

REQUIRED_PROCESSED_FILES = [
    Path("data/processed/formal_joint_storage_conditional_prior_dataset.csv"),
]
pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in REQUIRED_PROCESSED_FILES),
    reason="Processed spatial-assimilation data are not present",
)


def _grid(n: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-20.0, 20.0, n)
    y = np.linspace(-20.0, 20.0, n)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    truth = -11.0 + 0.35 * np.sin(xx / 8.0) + 0.25 * np.cos(yy / 10.0)
    return coords, truth.ravel(), x


def test_support_operator_averages_cells_inside_radius_and_normalizes_rows():
    coords, _, _ = _grid(5)
    observations = pd.DataFrame(
        {
            "x": [0.0, 100.0],
            "y": [0.0, 100.0],
            "support_radius": [15.0, 2.0],
        }
    )

    h = build_support_operator(coords, observations)

    assert h.shape == (2, len(coords))
    assert np.allclose(h.sum(axis=1), 1.0)
    assert np.count_nonzero(h[0]) > 1
    assert np.count_nonzero(h[1]) == 1


def test_conditional_soft_observations_improve_spatial_field_reconstruction():
    coords, truth, _ = _grid(11)
    prior_mean = np.full(len(truth), -11.0)
    prior_cov = squared_exponential_covariance(coords, variance=0.45**2, corr_len=18.0, nugget=1.0e-6)
    observations = pd.DataFrame(
        {
            "x": [-15, -5, 5, 15, -15, -5, 5, 15],
            "y": [-15, -5, 5, 15, 15, 5, -5, -15],
            "support_radius": [4, 4, 4, 4, 9, 9, 9, 9],
            "method": ["slug", "slug", "slug", "slug", "pumping", "pumping", "pumping", "pumping"],
            "actual_log_K_hat": np.nan,
            "constant_correction": [0.45, 0.45, 0.45, 0.45, 0.18, 0.18, 0.18, 0.18],
            "constant_sigma": [0.45, 0.45, 0.45, 0.45, 0.30, 0.30, 0.30, 0.30],
            "conditional_correction": [0.70, 0.65, 0.60, 0.55, 0.26, 0.22, 0.18, 0.14],
            "conditional_sigma": [0.20, 0.20, 0.20, 0.20, 0.16, 0.16, 0.16, 0.16],
        }
    )
    h = build_support_operator(coords, observations)
    true_support = h @ truth
    observations["actual_log_K_hat"] = true_support + observations["conditional_correction"].to_numpy(dtype=float)

    hard = assimilate_linear_gaussian(
        coords=coords,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        observations=observations,
        correction_column=None,
        sigma_column=None,
        default_sigma=0.08,
    )
    method_constant = assimilate_linear_gaussian(
        coords=coords,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        observations=observations,
        correction_column="constant_correction",
        sigma_column="constant_sigma",
    )
    conditional = assimilate_linear_gaussian(
        coords=coords,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        observations=observations,
        correction_column="conditional_correction",
        sigma_column="conditional_sigma",
    )

    hard_metrics = evaluate_spatial_posterior(hard, truth)
    constant_metrics = evaluate_spatial_posterior(method_constant, truth)
    conditional_metrics = evaluate_spatial_posterior(conditional, truth)

    assert conditional_metrics["rmse_logk"] < constant_metrics["rmse_logk"]
    assert conditional_metrics["rmse_logk"] < hard_metrics["rmse_logk"]
    assert conditional_metrics["coverage_95"] >= constant_metrics["coverage_95"]


def test_spatial_assimilation_script_writes_metrics_and_figure(tmp_path):
    from trustk.experiments.run_spatial_assimilation import run_spatial_assimilation

    output_prefix = tmp_path / "fig08_spatial_assimilation"
    metrics_path = tmp_path / "spatial_metrics.csv"
    fields_path = tmp_path / "spatial_fields.csv"
    observations_path = tmp_path / "spatial_observations.csv"
    report_path = tmp_path / "spatial_report.json"

    report = run_spatial_assimilation(
        prior_dataset_path="data/processed/formal_joint_storage_conditional_prior_dataset.csv",
        metrics_path=metrics_path,
        fields_path=fields_path,
        observations_path=observations_path,
        report_path=report_path,
        figure_prefix=output_prefix,
        grid_n=17,
        n_observations=16,
        seed=22,
    )
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(metrics_path)

    assert loaded["checks"]["trustk_beats_method_constant_rmse"]["pass"]
    assert report["checks"]["trustk_beats_hard_rmse"]["pass"]
    assert "TRUST-K conditional" in set(metrics["approach"])
    assert output_prefix.with_suffix(".pdf").exists()
    assert output_prefix.with_suffix(".svg").exists()
    assert output_prefix.with_suffix(".png").exists()
    assert fields_path.exists()
    assert observations_path.exists()

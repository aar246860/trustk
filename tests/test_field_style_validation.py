from pathlib import Path

import pandas as pd
import pytest

from trustk.plotting.field_figures import figure_field_leave_one_out
from trustk.validation.field_style import build_field_style_validation

PROCESSED_DIR = Path("data/processed")
pytestmark = pytest.mark.skipif(not PROCESSED_DIR.exists(), reason="Processed field validation data are not present")


def test_field_style_validation_uses_predictive_metrics_not_true_k():
    result = build_field_style_validation("data/processed")

    assert set(result.posterior_wells["well"]) == {"BD", "BS", "MD", "MS", "PINN", "WD", "WS"}
    assert {"pumping_response", "slug_recovery"}.issubset(set(result.metrics["validation_target"]))
    assert {"USGS calibrated", "Independent Theis", "Hard slug LOO"}.issubset(
        set(result.metrics["method"])
    )
    assert (result.metrics["normalized_rmse"] >= 0).all()
    assert "segment" in result.slug_predictions.columns
    assert result.slug_predictions[result.slug_predictions["well"].eq("MD")]["segment"].nunique() == 3
    assert result.slug_predictions[result.slug_predictions["well"].eq("BD")]["segment"].nunique() == 2

    forbidden = " ".join(result.metrics.columns.tolist() + result.posterior_wells.columns.tolist()).lower()
    assert "true" not in forbidden
    assert "latent" not in forbidden


def test_field_leave_one_out_figure_exports_required_files(tmp_path: Path):
    figure_field_leave_one_out("data/processed", tmp_path)

    prefix = tmp_path / "fig09_field_leave_one_out"
    for suffix in [".pdf", ".svg", ".png", ".csv", ".json"]:
        assert prefix.with_suffix(suffix).exists()

    metrics = pd.read_csv(prefix.with_suffix(".csv"))
    assert {"well", "method", "validation_target", "normalized_rmse"}.issubset(metrics.columns)
    metadata = prefix.with_suffix(".json").read_text(encoding="utf-8")
    assert "does not claim true field K" in metadata

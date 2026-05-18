import json

import numpy as np
import pandas as pd

from trustk.dimensionless.settings import DimensionalBaseConfig, convert_case_registry_to_solver_settings
from trustk.experiments.run_synthetic_pilot import run_synthetic_pilot, simulate_paired_case


def _small_ready_settings() -> pd.DataFrame:
    registry = pd.DataFrame(
        {
            "case_id": ["case_0000", "case_0001"],
            "random_seed": [101, 102],
            "sigma_Y2": [0.25, 0.5],
            "lambda1_over_RI": [0.8, 0.6],
            "lambda2_over_lambda1": [0.6, 0.5],
            "lambda2_over_RI": [0.48, 0.30],
            "phi_lambda_rad": [0.1, 1.0],
            "r_skin_over_rw": [1.0, 1.0],
            "K_skin_over_K0": [1.0, 1.0],
            "Rmax_over_RI": [3.0, 4.0],
            "r_obs_over_RI_P": [0.4, 0.5],
            "tD_P_max": [100.0, 300.0],
            "C_D_w": [100.0, 200.0],
            "r_c_over_r_w": [0.7, 0.8],
            "tD_S_max": [30.0, 80.0],
            "pumping_interpretation_method": ["Theis", "Theis"],
            "slug_interpretation_method": ["Bouwer-Rice", "Bouwer-Rice"],
        }
    )
    base = DimensionalBaseConfig(max_cartesian_n=256, mesh_n_r=24, mesh_n_theta=12, pumping_n_times=6, slug_n_times=6)
    settings = convert_case_registry_to_solver_settings(registry, base)
    settings["ready_for_pilot_solver"] = True
    settings["mesh_n_r"] = 18
    settings["mesh_n_theta"] = 8
    settings["pumping_n_times"] = 5
    settings["slug_n_times"] = 5
    return settings


def test_paired_case_simulation_produces_pumping_and_slug_responses():
    settings = _small_ready_settings()
    curves, summary = simulate_paired_case(settings.iloc[0], cartesian_n_cap=129)

    pumping = curves[curves["method"] == "pumping"]
    slug = curves[curves["method"] == "slug"]

    assert set(curves["method"]) == {"pumping", "slug"}
    assert len(pumping) == 5
    assert len(slug) == 5
    assert np.all(np.isfinite(curves["response_value"]))
    assert pumping["response_value"].iloc[-1] > pumping["response_value"].iloc[0]
    assert slug["response_value"].iloc[-1] < slug["response_value"].iloc[0]
    assert summary["pumping_final_drawdown_m"] > 0.0
    assert 0.0 <= summary["slug_final_normalized_head"] <= 1.0


def test_synthetic_pilot_script_writes_curves_summary_report_and_figure(tmp_path):
    settings_path = tmp_path / "solver_settings.csv"
    curves_path = tmp_path / "synthetic_pilot_curves.csv"
    summary_path = tmp_path / "synthetic_pilot_summary.csv"
    report_path = tmp_path / "synthetic_pilot.json"
    figure_prefix = tmp_path / "fig_synthetic_pilot"
    _small_ready_settings().to_csv(settings_path, index=False)

    report = run_synthetic_pilot(
        settings_path=settings_path,
        curves_path=curves_path,
        summary_path=summary_path,
        report_path=report_path,
        figure_prefix=figure_prefix,
        max_cases=2,
        cartesian_n_cap=129,
    )
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    curves = pd.read_csv(curves_path)
    summary = pd.read_csv(summary_path)

    assert report["checks"]["paired_methods_present"]["pass"]
    assert loaded["simulated_case_count"] == 2
    assert set(curves["method"]) == {"pumping", "slug"}
    assert len(summary) == 2
    assert figure_prefix.with_suffix(".pdf").exists()
    assert figure_prefix.with_suffix(".svg").exists()
    assert figure_prefix.with_suffix(".png").exists()

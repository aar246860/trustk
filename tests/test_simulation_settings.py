import json

import numpy as np
import pandas as pd

from trustk.dimensionless.settings import (
    DimensionalBaseConfig,
    convert_case_registry_to_solver_settings,
    summarize_solver_settings,
)


def _example_registry() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": ["case_a", "case_b"],
            "random_seed": [11, 12],
            "sigma_Y2": [1.0, 0.25],
            "lambda1_over_RI": [0.5, 0.2],
            "lambda2_over_lambda1": [0.4, 0.5],
            "lambda2_over_RI": [0.2, 0.1],
            "phi_lambda_rad": [0.0, 1.0],
            "r_skin_over_rw": [2.0, 3.0],
            "K_skin_over_K0": [1.0, 0.5],
            "Rmax_over_RI": [4.0, 5.0],
            "r_obs_over_RI_P": [0.5, 0.25],
            "tD_P_max": [1.0e4, 1.0e2],
            "C_D_w": [1.0e3, 2.0e3],
            "r_c_over_r_w": [0.8, 0.6],
            "tD_S_max": [1.0e2, 1.0e4],
            "pumping_interpretation_method": ["Theis", "Theis"],
            "slug_interpretation_method": ["Bouwer-Rice", "Bouwer-Rice"],
        }
    )


def test_registry_rows_convert_to_dimensional_solver_settings():
    base = DimensionalBaseConfig(
        mean_logk=-11.0,
        aquifer_thickness_m=20.0,
        storativity=2.0e-4,
        well_radius_m=0.1,
        pumping_rate_m3_s=1.0e-3,
        initial_slug_head_m=1.0,
    )
    settings = convert_case_registry_to_solver_settings(_example_registry(), base)
    row = settings.iloc[0]
    k0 = np.exp(-11.0)
    t0 = 2.0e-4 * 0.1**2 / (k0 * 20.0)

    assert np.isclose(row["K0_m_s"], k0)
    assert np.isclose(row["T0_m2_s"], k0 * 20.0)
    assert np.isclose(row["t0_s"], t0)
    assert np.isclose(row["RI_P_m"], 10.0)
    assert np.isclose(row["RI_S_m"], 1.0)
    assert np.isclose(row["RI_common_m"], 10.0)
    assert np.isclose(row["r_max_m"], 40.0)
    assert np.isclose(row["r_obs_m"], 5.0)
    assert np.isclose(row["lambda1_m"], 5.0)
    assert np.isclose(row["lambda2_m"], 2.0)
    assert np.isclose(row["well_storage_m2"], 1.0e3 * 2.0e-4 * 0.1**2)
    assert np.isclose(row["casing_radius_m"], 0.08)
    assert np.isclose(row["pumping_time_max_s"], 1.0e4 * t0)
    assert np.isclose(row["slug_time_max_s"], 1.0e2 * t0)
    assert row["pumping_rate_m3_s"] == base.pumping_rate_m3_s
    assert row["initial_slug_head_m"] == base.initial_slug_head_m


def test_solver_settings_summary_checks_domain_and_resolution():
    base = DimensionalBaseConfig(
        mean_logk=-11.0,
        aquifer_thickness_m=20.0,
        storativity=2.0e-4,
        well_radius_m=0.1,
        pumping_rate_m3_s=1.0e-3,
        initial_slug_head_m=1.0,
        max_cartesian_n=2048,
    )
    settings = convert_case_registry_to_solver_settings(_example_registry(), base)
    summary = summarize_solver_settings(settings)

    assert summary["checks"]["finite_solver_settings"]["pass"]
    assert summary["checks"]["outer_boundary_exceeds_influence_radius"]["pass"]
    assert summary["checks"]["positive_times_and_lengths"]["pass"]
    assert summary["checks"]["fixed_stress_amplitudes_recorded"]["pass"]
    assert summary["ready_case_count"] == len(settings)


def test_solver_settings_script_writes_table_report_and_figure(tmp_path):
    from trustk.experiments.run_solver_settings import run_solver_settings

    registry_path = tmp_path / "registry.csv"
    settings_path = tmp_path / "solver_settings.csv"
    report_path = tmp_path / "solver_settings.json"
    figure_prefix = tmp_path / "fig_solver_settings"
    _example_registry().to_csv(registry_path, index=False)

    report = run_solver_settings(
        registry_path=registry_path,
        settings_path=settings_path,
        report_path=report_path,
        figure_prefix=figure_prefix,
        max_cartesian_n=2048,
    )
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    settings = pd.read_csv(settings_path)

    assert report["checks"]["positive_times_and_lengths"]["pass"]
    assert loaded["ready_case_count"] == len(settings)
    assert figure_prefix.with_suffix(".pdf").exists()
    assert figure_prefix.with_suffix(".svg").exists()
    assert figure_prefix.with_suffix(".png").exists()

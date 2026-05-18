import json

import numpy as np
import pandas as pd

from trustk.dimensionless.registry import (
    DimensionlessDesignRanges,
    build_dimensionless_case_table,
    hydraulic_scales,
)


def test_hydraulic_scales_follow_common_diffusion_scaling():
    scales = hydraulic_scales(
        mean_logk=-11.0,
        aquifer_thickness=20.0,
        storativity=2.0e-4,
        well_radius=0.1,
        pumping_rate=1.0e-3,
        well_storage=0.02,
        casing_radius=0.08,
    )

    expected_k0 = np.exp(-11.0)
    expected_t0 = 2.0e-4 * 0.1**2 / (expected_k0 * 20.0)

    assert np.isclose(scales["K0_m_s"], expected_k0)
    assert np.isclose(scales["T0_m2_s"], expected_k0 * 20.0)
    assert np.isclose(scales["L0_m"], 0.1)
    assert np.isclose(scales["t0_s"], expected_t0)
    assert np.isclose(scales["H_P_m"], 1.0e-3 / (4.0 * np.pi * expected_k0 * 20.0))
    assert np.isclose(scales["C_D_w"], 0.02 / (2.0e-4 * 0.1**2))
    assert np.isclose(scales["r_c_over_r_w"], 0.8)


def test_dimensionless_case_table_is_reproducible_and_separates_methods():
    ranges = DimensionlessDesignRanges(
        sigma_y2=(0.05, 1.5),
        lambda1_over_ri=(0.05, 2.0),
        lambda2_over_lambda1=(0.15, 1.0),
        cd_w=(1.0e2, 1.0e5),
    )
    table_a = build_dimensionless_case_table(n_cases=32, seed=20260517, ranges=ranges)
    table_b = build_dimensionless_case_table(n_cases=32, seed=20260517, ranges=ranges)

    assert list(table_a.columns) == list(table_b.columns)
    assert table_a.equals(table_b)
    assert len(table_a) == 32
    assert "Q" not in table_a.columns
    assert "H0" not in table_a.columns
    assert set(table_a["pumping_interpretation_method"]) == {"Theis"}
    assert set(table_a["slug_interpretation_method"]) == {"Bouwer-Rice"}

    required = {
        "case_id",
        "random_seed",
        "sigma_Y2",
        "lambda1_over_RI",
        "lambda2_over_lambda1",
        "lambda2_over_RI",
        "phi_lambda_rad",
        "r_skin_over_rw",
        "K_skin_over_K0",
        "Rmax_over_RI",
        "r_obs_over_RI_P",
        "tD_P_max",
        "C_D_w",
        "r_c_over_r_w",
        "tD_S_max",
    }
    assert required.issubset(table_a.columns)
    assert np.allclose(table_a["lambda2_over_RI"], table_a["lambda1_over_RI"] * table_a["lambda2_over_lambda1"])
    assert table_a["phi_lambda_rad"].between(0.0, np.pi).all()
    assert table_a["sigma_Y2"].between(0.05, 1.5).all()
    assert table_a["C_D_w"].between(1.0e2, 1.0e5).all()


def test_dimensionless_registry_script_writes_report_table_and_figure(tmp_path):
    from trustk.experiments.run_dimensionless_registry import run_dimensionless_registry

    table_path = tmp_path / "dimensionless_case_registry.csv"
    report_path = tmp_path / "dimensionless_registry.json"
    figure_prefix = tmp_path / "fig_dimensionless_registry"

    report = run_dimensionless_registry(
        table_path=table_path,
        report_path=report_path,
        figure_prefix=figure_prefix,
        n_cases=48,
        seed=123,
    )
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    table = pd.read_csv(table_path)

    assert report["checks"]["required_columns"]["pass"]
    assert loaded["checks"]["no_dimensional_stress_controls"]["pass"]
    assert len(table) == 48
    assert figure_prefix.with_suffix(".pdf").exists()
    assert figure_prefix.with_suffix(".svg").exists()
    assert figure_prefix.with_suffix(".png").exists()

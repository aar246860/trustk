import numpy as np
import pandas as pd

from trustk.experiments.run_joint_storage_reanalysis import (
    build_joint_storage_design,
    run_joint_storage_reanalysis,
)
from trustk.mesh.polar_mesh import make_log_polar_mesh
from trustk.random_fields.joint_fields import JointFieldConfig, generate_joint_log_fields
from trustk.random_fields.mapping import map_cartesian_field_to_polar
from trustk.targets.support_area import support_targets_from_mapped_logk_logss


def test_joint_log_fields_match_requested_correlation_and_storage_bounds():
    config = JointFieldConfig(
        nx=65,
        ny=65,
        dx=1.0,
        dy=1.0,
        mean_logk=-11.0,
        sigma_logk=0.8,
        corr_len_k_x=8.0,
        corr_len_k_y=4.0,
        orientation_rad=0.35,
        mean_logss=np.log(1.0e-5),
        sigma_logss=0.45,
        corr_len_ss_x=4.0,
        corr_len_ss_y=2.0,
        log_correlation=-0.5,
        min_logss=np.log(1.0e-7),
        max_logss=np.log(1.0e-3),
        seed=44,
    )
    field = generate_joint_log_fields(config)
    realized_correlation = np.corrcoef(field.logk.ravel(), field.logss.ravel())[0, 1]

    assert field.logk.shape == (65, 65)
    assert field.logss.shape == (65, 65)
    assert np.isclose(np.mean(field.logk), -11.0)
    assert np.isclose(np.std(field.logk), 0.8)
    assert -0.65 < realized_correlation < -0.35
    assert np.min(field.logss) >= np.log(1.0e-7)
    assert np.max(field.logss) <= np.log(1.0e-3)


def test_support_targets_include_specific_storage_and_diffusivity():
    config = JointFieldConfig(
        nx=41,
        ny=41,
        dx=1.0,
        dy=1.0,
        mean_logk=-11.0,
        sigma_logk=0.5,
        corr_len_k_x=8.0,
        corr_len_k_y=6.0,
        orientation_rad=0.0,
        mean_logss=np.log(1.0e-5),
        sigma_logss=0.35,
        corr_len_ss_x=8.0,
        corr_len_ss_y=6.0,
        log_correlation=0.0,
        min_logss=np.log(1.0e-7),
        max_logss=np.log(1.0e-3),
        seed=12,
    )
    mesh = make_log_polar_mesh(r_w=0.1, r_max=12.0, n_r=16, n_theta=10)
    field = generate_joint_log_fields(config)
    targets = support_targets_from_mapped_logk_logss(
        mapped_logk=map_cartesian_field_to_polar(field.k_field, mesh),
        mapped_logss=map_cartesian_field_to_polar(field.ss_field, mesh),
        mesh=mesh,
        pumping_radius_m=8.0,
        slug_radius_m=1.0,
    )

    assert targets["K_star_pumping_m_s"] > 0.0
    assert targets["Ss_star_pumping_m_inv"] > 0.0
    assert targets["diffusivity_star_pumping_m2_s"] > 0.0
    assert -1.0 <= targets["lnK_lnSs_corr_pumping"] <= 1.0


def test_joint_storage_reanalysis_writes_core_outputs(tmp_path):
    report = run_joint_storage_reanalysis(
        design_path=tmp_path / "joint_design.csv",
        curves_path=tmp_path / "joint_curves.csv",
        summary_path=tmp_path / "joint_summary.csv",
        qc_path=tmp_path / "joint_qc.csv",
        prior_path=tmp_path / "joint_prior.csv",
        prior_dataset_path=tmp_path / "joint_prior_dataset.csv",
        conditional_predictions_path=tmp_path / "joint_predictions.csv",
        holdout_path=tmp_path / "joint_holdout.csv",
        baseline_path=tmp_path / "joint_baselines.csv",
        report_path=tmp_path / "joint_report.json",
        figure_prefix=tmp_path / "fig_joint_storage",
        n_cases=8,
        seed=2026,
        cartesian_n_cap=65,
        mesh_n_r=14,
        mesh_n_theta=8,
        make_figure=False,
        min_prior_cases=2,
    )

    assert report["simulated_case_count"] == 8
    assert report["checks"]["finite_responses"]["pass"]
    assert report["checks"]["joint_storage_columns_present"]["pass"]
    assert report["formal_prior"]["methods"]["pumping"]["interpretation_method"] == "Cooper-Jacob"
    assert report["formal_prior"]["methods"]["slug_bouwer_rice"]["interpretation_method"] == "Bouwer-Rice"
    assert report["formal_acceptance"]["minimum_case_count"]["minimum"] == 4096
    assert (tmp_path / "joint_summary.csv").exists()
    assert (tmp_path / "joint_qc.csv").exists()
    assert (tmp_path / "joint_prior.csv").exists()
    assert (tmp_path / "joint_prior_dataset.csv").exists()
    assert (tmp_path / "joint_predictions.csv").exists()
    assert (tmp_path / "joint_holdout.csv").exists()
    assert (tmp_path / "joint_baselines.csv").exists()
    assert (tmp_path / "joint_report.json").exists()

    qc = pd.read_csv(tmp_path / "joint_qc.csv")
    prior_data = pd.read_csv(tmp_path / "joint_prior_dataset.csv")
    assert set(qc["method"]) == {"pumping", "slug_bouwer_rice"}
    assert {"sigma_lnSs2", "storage_correlation", "log_Ss_star_m_inv"}.issubset(prior_data.columns)
    assert set(prior_data["method"]).issubset({"pumping", "slug_bouwer_rice"})


def test_joint_design_covers_b_and_storage_controls():
    design = build_joint_storage_design(n_cases=64, seed=8, mesh_n_r=16, mesh_n_theta=8)

    assert {"aquifer_thickness_m", "specific_storage_gmean_m_inv", "storage_correlation"}.issubset(
        design.columns
    )
    assert design["aquifer_thickness_m"].between(5.0, 200.0).all()
    assert design["aquifer_thickness_m"].max() > 150.0
    assert design["specific_storage_gmean_m_inv"].between(1.0e-6, 1.0e-4).all()
    assert set(np.round(design["storage_correlation"], 1)).issubset({-0.5, 0.0, 0.5})

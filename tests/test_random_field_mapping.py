import json

import numpy as np
import pytest

from trustk.mesh.polar_mesh import make_log_polar_mesh
from trustk.random_fields.gaussian_field import GaussianField2D, generate_gaussian_logk_field
from trustk.random_fields.mapping import map_cartesian_field_to_polar, radial_symmetry_score


def test_gaussian_logk_field_is_reproducible_and_normalized():
    field_a = generate_gaussian_logk_field(
        nx=64,
        ny=48,
        dx=5.0,
        dy=5.0,
        mean_logk=-11.0,
        sigma_logk=0.8,
        corr_len_x=40.0,
        corr_len_y=25.0,
        seed=20260517,
    )
    field_b = generate_gaussian_logk_field(
        nx=64,
        ny=48,
        dx=5.0,
        dy=5.0,
        mean_logk=-11.0,
        sigma_logk=0.8,
        corr_len_x=40.0,
        corr_len_y=25.0,
        seed=20260517,
    )

    assert field_a.logk.shape == (48, 64)
    assert np.allclose(field_a.logk, field_b.logk)
    assert abs(float(np.mean(field_a.logk)) + 11.0) < 1.0e-12
    assert abs(float(np.std(field_a.logk)) - 0.8) < 1.0e-12


def test_gaussian_logk_field_rejects_invalid_geometry():
    with pytest.raises(ValueError):
        generate_gaussian_logk_field(
            nx=1,
            ny=48,
            dx=5.0,
            dy=5.0,
            mean_logk=-11.0,
            sigma_logk=0.8,
            corr_len_x=40.0,
            corr_len_y=25.0,
            seed=1,
        )
    with pytest.raises(ValueError):
        generate_gaussian_logk_field(
            nx=64,
            ny=48,
            dx=5.0,
            dy=5.0,
            mean_logk=-11.0,
            sigma_logk=0.8,
            corr_len_x=-40.0,
            corr_len_y=25.0,
            seed=1,
        )


def test_gaussian_logk_field_uses_orientation_for_anisotropy():
    field_x = generate_gaussian_logk_field(
        nx=64,
        ny=64,
        dx=4.0,
        dy=4.0,
        mean_logk=-11.0,
        sigma_logk=0.8,
        corr_len_x=60.0,
        corr_len_y=12.0,
        orientation_rad=0.0,
        seed=99,
    )
    field_y = generate_gaussian_logk_field(
        nx=64,
        ny=64,
        dx=4.0,
        dy=4.0,
        mean_logk=-11.0,
        sigma_logk=0.8,
        corr_len_x=60.0,
        corr_len_y=12.0,
        orientation_rad=np.pi / 2.0,
        seed=99,
    )

    assert not np.allclose(field_x.logk, field_y.logk)
    assert abs(float(np.mean(field_y.logk)) + 11.0) < 1.0e-12
    assert abs(float(np.std(field_y.logk)) - 0.8) < 1.0e-12


def test_cartesian_field_maps_to_polar_mesh_without_radial_symmetry():
    field = generate_gaussian_logk_field(
        nx=120,
        ny=120,
        dx=4.0,
        dy=4.0,
        mean_logk=-11.0,
        sigma_logk=0.7,
        corr_len_x=65.0,
        corr_len_y=30.0,
        seed=42,
    )
    mesh = make_log_polar_mesh(r_w=0.2, r_max=180.0, n_r=70, n_theta=72)

    mapped = map_cartesian_field_to_polar(field, mesh)

    assert mapped.shape == mesh.shape
    assert np.all(np.isfinite(mapped))
    assert float(np.mean(np.std(mapped[-20:, :], axis=1))) > 0.05
    assert radial_symmetry_score(mapped) > 0.20


def test_cartesian_mapping_rejects_uncovered_polar_domain():
    field = generate_gaussian_logk_field(
        nx=30,
        ny=30,
        dx=2.0,
        dy=2.0,
        mean_logk=-11.0,
        sigma_logk=0.7,
        corr_len_x=20.0,
        corr_len_y=20.0,
        seed=7,
    )
    mesh = make_log_polar_mesh(r_w=0.2, r_max=80.0, n_r=40, n_theta=48)

    with pytest.raises(ValueError, match="does not cover"):
        map_cartesian_field_to_polar(field, mesh)


def test_random_field_verification_script_writes_report_and_figure(tmp_path):
    from trustk.experiments.run_random_field_verification import run_random_field_verification

    report_path = tmp_path / "random_field_verification.json"
    figure_prefix = tmp_path / "fig_random_field_mapping"

    report = run_random_field_verification(
        report_path=report_path,
        figure_prefix=figure_prefix,
        nx=96,
        ny=96,
        dx=5.0,
        dy=5.0,
        r_max=170.0,
        n_r=50,
        n_theta=48,
    )
    loaded = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["checks"]["theta_varying_heterogeneity"]["pass"]
    assert loaded["checks"]["cartesian_moments"]["pass"]
    assert figure_prefix.with_suffix(".pdf").exists()
    assert figure_prefix.with_suffix(".svg").exists()
    assert figure_prefix.with_suffix(".png").exists()

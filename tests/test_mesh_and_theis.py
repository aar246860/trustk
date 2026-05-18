import numpy as np
import pytest

from trustk.analytical.theis import theis_drawdown
from trustk.mesh.polar_mesh import make_log_polar_mesh, total_area


def test_log_polar_mesh_area_conservation():
    mesh = make_log_polar_mesh(r_w=0.1, r_max=100.0, n_r=120, n_theta=96)

    expected = np.pi * (100.0**2 - 0.1**2)

    assert mesh.shape == (120, 96)
    assert np.isclose(total_area(mesh), expected, rtol=1e-12)
    assert np.all(np.diff(mesh.r_faces) > 0)
    assert mesh.x_centers.shape == mesh.area.shape


def test_log_polar_mesh_rejects_invalid_geometry():
    with pytest.raises(ValueError):
        make_log_polar_mesh(r_w=0.0, r_max=100.0, n_r=10, n_theta=8)
    with pytest.raises(ValueError):
        make_log_polar_mesh(r_w=10.0, r_max=1.0, n_r=10, n_theta=8)


def test_theis_drawdown_monotonic_behavior():
    times = np.array([10.0, 100.0, 1000.0])
    late_drawdown = theis_drawdown(10.0, times, transmissivity=1e-3, storativity=1e-4, pumping_rate=1e-3)

    radii = np.array([5.0, 10.0, 20.0])
    radial_drawdown = theis_drawdown(radii, 1000.0, transmissivity=1e-3, storativity=1e-4, pumping_rate=1e-3)

    assert np.all(np.diff(late_drawdown) > 0)
    assert np.all(np.diff(radial_drawdown) < 0)
    assert np.all(late_drawdown > 0)


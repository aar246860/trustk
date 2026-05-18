import numpy as np

from trustk.analytical.theis import theis_drawdown
from trustk.mesh.polar_mesh import make_log_polar_mesh
from trustk.physics.fv_solver import (
    PumpingSimulationResult,
    radial_average,
    simulate_constant_rate_pumping,
)


def test_constant_rate_solver_preserves_angular_symmetry_for_uniform_field():
    mesh = make_log_polar_mesh(r_w=0.1, r_max=400.0, n_r=60, n_theta=32)
    result = simulate_constant_rate_pumping(
        mesh,
        transmissivity=np.full(mesh.shape, 2e-3),
        storativity=1e-4,
        pumping_rate=1e-3,
        times=np.array([100.0, 300.0, 1000.0]),
    )

    assert isinstance(result, PumpingSimulationResult)
    assert result.drawdown.shape == (3, *mesh.shape)
    assert np.max(result.drawdown[-1].std(axis=1)) < 1e-10
    assert np.all(result.drawdown[-1] >= 0)


def test_constant_rate_solver_is_close_to_theis_before_outer_boundary_effects():
    mesh = make_log_polar_mesh(r_w=0.05, r_max=2500.0, n_r=180, n_theta=16)
    transmissivity = 4e-3
    storativity = 2e-4
    pumping_rate = 1.2e-3
    times = np.geomspace(30.0, 2.0e5, 28)

    result = simulate_constant_rate_pumping(
        mesh,
        transmissivity=np.full(mesh.shape, transmissivity),
        storativity=storativity,
        pumping_rate=pumping_rate,
        times=times,
    )
    simulated = radial_average(result)[-1]
    analytical = theis_drawdown(
        mesh.r_centers,
        times[-1],
        transmissivity=transmissivity,
        storativity=storativity,
        pumping_rate=pumping_rate,
    )
    mask = (mesh.r_centers > 1.0) & (mesh.r_centers < 400.0)
    relative_error = np.linalg.norm(simulated[mask] - analytical[mask]) / np.linalg.norm(analytical[mask])

    assert relative_error < 0.18


def test_solver_rejects_nonmonotonic_times_and_bad_transmissivity_shape():
    mesh = make_log_polar_mesh(r_w=0.1, r_max=100.0, n_r=10, n_theta=8)

    try:
        simulate_constant_rate_pumping(
            mesh,
            transmissivity=np.ones(mesh.shape),
            storativity=1e-4,
            pumping_rate=1e-3,
            times=np.array([10.0, 5.0]),
        )
    except ValueError as exc:
        assert "strictly increasing" in str(exc)
    else:
        raise AssertionError("nonmonotonic times should fail")

    try:
        simulate_constant_rate_pumping(
            mesh,
            transmissivity=np.ones((mesh.n_r,)),
            storativity=1e-4,
            pumping_rate=1e-3,
            times=np.array([10.0]),
        )
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("bad transmissivity shape should fail")

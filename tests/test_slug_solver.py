import numpy as np

from trustk.mesh.polar_mesh import make_log_polar_mesh
from trustk.physics.slug_solver import (
    SlugSimulationResult,
    quasi_steady_slug_head,
    simulate_slug_recovery,
)


def test_slug_solver_produces_monotonic_well_recovery():
    mesh = make_log_polar_mesh(r_w=0.05, r_max=200.0, n_r=80, n_theta=16)
    times = np.geomspace(1.0, 2.0e4, 40)
    result = simulate_slug_recovery(
        mesh,
        transmissivity=np.full(mesh.shape, 2e-3),
        storativity=1e-4,
        well_storage=0.01,
        initial_well_head=1.0,
        times=times,
    )

    assert isinstance(result, SlugSimulationResult)
    assert result.aquifer_head.shape == (len(times), *mesh.shape)
    assert np.min(result.well_head) > -1e-8
    active_signal = result.well_head[:-1] > 1e-8
    assert np.all(np.diff(result.well_head)[active_signal] < 0)


def test_slug_solver_matches_quasi_steady_exponential_limit():
    mesh = make_log_polar_mesh(r_w=0.05, r_max=100.0, n_r=180, n_theta=24)
    transmissivity = 2e-3
    well_storage = 0.02
    h0 = 1.0
    times = np.linspace(1.0, 1200.0, 160)

    result = simulate_slug_recovery(
        mesh,
        transmissivity=np.full(mesh.shape, transmissivity),
        storativity=1e-9,
        well_storage=well_storage,
        initial_well_head=h0,
        times=times,
    )
    reference = quasi_steady_slug_head(
        times,
        initial_well_head=h0,
        transmissivity=transmissivity,
        well_storage=well_storage,
        r_w=0.05,
        r_max=100.0,
    )
    relative_error = np.linalg.norm(result.well_head - reference) / np.linalg.norm(reference)

    assert relative_error < 0.08


def test_slug_solver_rejects_invalid_inputs():
    mesh = make_log_polar_mesh(r_w=0.05, r_max=100.0, n_r=20, n_theta=8)
    try:
        simulate_slug_recovery(
            mesh,
            transmissivity=np.ones(mesh.shape),
            storativity=1e-4,
            well_storage=0.0,
            initial_well_head=1.0,
            times=np.array([1.0]),
        )
    except ValueError as exc:
        assert "well_storage" in str(exc)
    else:
        raise AssertionError("zero well storage should fail")

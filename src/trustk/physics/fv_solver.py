"""Finite-volume solvers for 2D horizontal polar groundwater flow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

from trustk.mesh.polar_mesh import PolarMesh
from trustk.physics.storage import storativity_array


@dataclass(frozen=True)
class PumpingSimulationResult:
    """Drawdown snapshots from a constant-rate pumping simulation."""

    times: np.ndarray
    drawdown: np.ndarray
    mass_balance_error: np.ndarray


def _validate_inputs(
    mesh: PolarMesh,
    transmissivity: np.ndarray,
    storativity: float | np.ndarray,
    pumping_rate: float,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or len(t) == 0:
        raise ValueError("times must be a one-dimensional non-empty array")
    if np.any(t <= 0) or np.any(np.diff(t) <= 0):
        raise ValueError("times must be positive and strictly increasing")
    if pumping_rate <= 0:
        raise ValueError("pumping_rate must be positive")
    storage_values = storativity_array(storativity, mesh)
    trans = np.asarray(transmissivity, dtype=float)
    if trans.shape != mesh.shape:
        raise ValueError(f"transmissivity shape must be {mesh.shape}")
    if np.any(trans <= 0):
        raise ValueError("transmissivity values must be positive")
    return trans, storage_values, t


def _harmonic(a: float, b: float) -> float:
    return 2.0 * a * b / (a + b)


def _cell_index(i: int, j: int, n_theta: int) -> int:
    return i * n_theta + j


def _assemble_conductance_matrix(mesh: PolarMesh, transmissivity: np.ndarray) -> csr_matrix:
    n_r, n_theta = mesh.shape
    n = n_r * n_theta
    matrix = lil_matrix((n, n), dtype=float)
    dtheta = mesh.theta_faces[1] - mesh.theta_faces[0]

    for i in range(n_r):
        r_inner = mesh.r_faces[i]
        r_outer = mesh.r_faces[i + 1]
        radial_width = r_outer - r_inner
        r_center = mesh.r_centers[i]
        for j in range(n_theta):
            idx = _cell_index(i, j, n_theta)
            diagonal = 0.0

            if i > 0:
                neighbor = _cell_index(i - 1, j, n_theta)
                t_face = _harmonic(transmissivity[i, j], transmissivity[i - 1, j])
                distance = mesh.r_centers[i] - mesh.r_centers[i - 1]
                conductance = t_face * r_inner * dtheta / distance
                matrix[idx, neighbor] -= conductance
                diagonal += conductance

            if i < n_r - 1:
                neighbor = _cell_index(i + 1, j, n_theta)
                t_face = _harmonic(transmissivity[i, j], transmissivity[i + 1, j])
                distance = mesh.r_centers[i + 1] - mesh.r_centers[i]
                conductance = t_face * r_outer * dtheta / distance
                matrix[idx, neighbor] -= conductance
                diagonal += conductance
            else:
                distance = r_outer - r_center
                conductance = transmissivity[i, j] * r_outer * dtheta / distance
                diagonal += conductance

            for jj in ((j - 1) % n_theta, (j + 1) % n_theta):
                neighbor = _cell_index(i, jj, n_theta)
                t_face = _harmonic(transmissivity[i, j], transmissivity[i, jj])
                conductance = t_face * radial_width / (r_center * dtheta)
                matrix[idx, neighbor] -= conductance
                diagonal += conductance

            matrix[idx, idx] += diagonal

    return matrix.tocsr()


def simulate_constant_rate_pumping(
    mesh: PolarMesh,
    transmissivity: np.ndarray,
    storativity: float | np.ndarray,
    pumping_rate: float,
    times: np.ndarray,
) -> PumpingSimulationResult:
    """Simulate positive drawdown from a constant-rate well at the inner radius.

    The outer radial boundary is fixed at zero drawdown. The pumping rate is
    distributed uniformly over the inner radial boundary cells.
    """

    trans, storage_values, t = _validate_inputs(mesh, transmissivity, storativity, pumping_rate, times)
    conductance = _assemble_conductance_matrix(mesh, trans)
    storage = storage_values.ravel() * mesh.area.ravel()
    source = np.zeros(mesh.n_r * mesh.n_theta, dtype=float)
    source[: mesh.n_theta] = pumping_rate / mesh.n_theta

    state = np.zeros(mesh.n_r * mesh.n_theta, dtype=float)
    snapshots = []
    mass_error = []
    previous_time = 0.0
    cumulative_pumped = 0.0

    for current_time in t:
        dt = float(current_time - previous_time)
        lhs = conductance.copy().tolil()
        lhs.setdiag(lhs.diagonal() + storage / dt)
        rhs = storage / dt * state + source
        state = spsolve(lhs.tocsr(), rhs)
        cumulative_pumped += pumping_rate * dt
        aquifer_storage = float(np.sum(storage * state))
        # Boundary leakage is expected with the finite outer boundary. This
        # residual is a diagnostic, not an assertion of closed-domain storage.
        mass_error.append((aquifer_storage - cumulative_pumped) / cumulative_pumped)
        snapshots.append(state.reshape(mesh.shape).copy())
        previous_time = float(current_time)

    return PumpingSimulationResult(
        times=t,
        drawdown=np.asarray(snapshots),
        mass_balance_error=np.asarray(mass_error),
    )


def radial_average(result: PumpingSimulationResult) -> np.ndarray:
    """Average drawdown over theta for each radial ring."""

    return result.drawdown.mean(axis=2)

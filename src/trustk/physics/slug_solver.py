"""Wellbore-storage slug recovery solver for a polar finite-volume aquifer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from trustk.mesh.polar_mesh import PolarMesh
from trustk.physics.fv_solver import _assemble_conductance_matrix


@dataclass(frozen=True)
class SlugSimulationResult:
    """Well and aquifer head-displacement snapshots for a slug recovery test."""

    times: np.ndarray
    well_head: np.ndarray
    aquifer_head: np.ndarray
    mass_balance_error: np.ndarray


def _validate_slug_inputs(
    mesh: PolarMesh,
    transmissivity: np.ndarray,
    storativity: float,
    well_storage: float,
    initial_well_head: float,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    trans = np.asarray(transmissivity, dtype=float)
    if trans.shape != mesh.shape:
        raise ValueError(f"transmissivity shape must be {mesh.shape}")
    if np.any(trans <= 0):
        raise ValueError("transmissivity values must be positive")
    if storativity <= 0:
        raise ValueError("storativity must be positive")
    if well_storage <= 0:
        raise ValueError("well_storage must be positive")
    if initial_well_head <= 0:
        raise ValueError("initial_well_head must be positive")
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or len(t) == 0:
        raise ValueError("times must be a one-dimensional non-empty array")
    if np.any(t <= 0) or np.any(np.diff(t) <= 0):
        raise ValueError("times must be positive and strictly increasing")
    return trans, t


def _inner_well_conductance(mesh: PolarMesh, transmissivity: np.ndarray) -> np.ndarray:
    dtheta = mesh.theta_faces[1] - mesh.theta_faces[0]
    distance = mesh.r_centers[0] - mesh.r_faces[0]
    return transmissivity[0, :] * mesh.r_faces[0] * dtheta / distance


def simulate_slug_recovery(
    mesh: PolarMesh,
    transmissivity: np.ndarray,
    storativity: float,
    well_storage: float,
    initial_well_head: float,
    times: np.ndarray,
) -> SlugSimulationResult:
    """Simulate slug recovery with a coupled well node and aquifer grid.

    Positive `well_head` denotes initial well-water displacement above the
    formation reference head. The outer radial boundary is fixed at zero.
    """

    trans, t = _validate_slug_inputs(
        mesh,
        transmissivity,
        storativity,
        well_storage,
        initial_well_head,
        times,
    )
    aquifer_matrix = _assemble_conductance_matrix(mesh, trans)
    storage = storativity * mesh.area.ravel()
    well_g = _inner_well_conductance(mesh, trans)
    n_cells = mesh.n_r * mesh.n_theta
    well_idx = n_cells

    system = lil_matrix((n_cells + 1, n_cells + 1), dtype=float)
    system[:n_cells, :n_cells] = aquifer_matrix
    for j, conductance in enumerate(well_g):
        cell = j
        system[cell, cell] += conductance
        system[cell, well_idx] -= conductance
        system[well_idx, cell] -= conductance
        system[well_idx, well_idx] += conductance
    system = system.tocsr()

    mass = np.r_[storage, well_storage]
    state = np.zeros(n_cells + 1, dtype=float)
    state[well_idx] = float(initial_well_head)
    previous_time = 0.0
    initial_mass = well_storage * initial_well_head
    well_history = []
    aquifer_history = []
    mass_error = []

    theta_weight = 0.65
    for current_time in t:
        dt = float(current_time - previous_time)
        lhs = system.copy().tolil()
        lhs *= theta_weight
        lhs.setdiag(lhs.diagonal() + mass / dt)
        rhs = (mass / dt) * state - (1.0 - theta_weight) * system.dot(state)

        state = spsolve(lhs.tocsr(), rhs)
        aquifer_state = state[:n_cells]
        well_head = float(state[well_idx])
        current_mass = float(well_storage * well_head + np.sum(storage * aquifer_state))
        mass_error.append((current_mass - initial_mass) / initial_mass)
        well_history.append(well_head)
        aquifer_history.append(aquifer_state.reshape(mesh.shape).copy())
        previous_time = float(current_time)

    return SlugSimulationResult(
        times=t,
        well_head=np.asarray(well_history),
        aquifer_head=np.asarray(aquifer_history),
        mass_balance_error=np.asarray(mass_error),
    )


def quasi_steady_slug_head(
    times: np.ndarray,
    *,
    initial_well_head: float,
    transmissivity: float,
    well_storage: float,
    r_w: float,
    r_max: float,
) -> np.ndarray:
    """Quasi-steady exponential slug recovery in a confined radial aquifer."""

    if transmissivity <= 0:
        raise ValueError("transmissivity must be positive")
    if well_storage <= 0:
        raise ValueError("well_storage must be positive")
    if r_w <= 0 or r_max <= r_w:
        raise ValueError("r_max must be greater than r_w")
    t = np.asarray(times, dtype=float)
    conductance = 2.0 * np.pi * transmissivity / np.log(r_max / r_w)
    return initial_well_head * np.exp(-conductance * t / well_storage)

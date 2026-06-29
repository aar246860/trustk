from __future__ import annotations

import numpy as np

from trustk.mesh.polar_mesh import PolarMesh


def storativity_array(storativity: float | np.ndarray, mesh: PolarMesh) -> np.ndarray:
    values = np.asarray(storativity, dtype=float)
    if values.ndim == 0:
        scalar = float(values)
        if not np.isfinite(scalar) or scalar <= 0.0:
            raise ValueError("storativity must be positive and finite")
        return np.full(mesh.shape, scalar, dtype=float)
    try:
        values = np.broadcast_to(values, mesh.shape)
    except ValueError:
        raise ValueError(f"storativity shape must be {mesh.shape}")
    if values.shape != mesh.shape:
        raise ValueError(f"storativity shape must be {mesh.shape}")
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("storativity values must be positive and finite")
    return values.astype(float, copy=False)

"""Run numerical verification checks for the TRUST-K pumping solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trustk.analytical.theis import theis_drawdown
from trustk.mesh.polar_mesh import make_log_polar_mesh
from trustk.physics.fv_solver import radial_average, simulate_constant_rate_pumping


TRANSMISSIVITY = 4e-3
STORATIVITY = 2e-4
PUMPING_RATE = 1.2e-3
R_W = 0.05
FINAL_TIME = 1.0e5
COMPARISON_MIN_R = 1.0
COMPARISON_MAX_R = 350.0


def _profile_error(
    *,
    n_r: int,
    n_theta: int,
    r_max: float,
    n_steps: int,
) -> dict:
    mesh = make_log_polar_mesh(r_w=R_W, r_max=r_max, n_r=n_r, n_theta=n_theta)
    times = np.geomspace(30.0, FINAL_TIME, n_steps)
    result = simulate_constant_rate_pumping(
        mesh,
        transmissivity=np.full(mesh.shape, TRANSMISSIVITY),
        storativity=STORATIVITY,
        pumping_rate=PUMPING_RATE,
        times=times,
    )
    simulated = radial_average(result)[-1]
    analytical = theis_drawdown(
        mesh.r_centers,
        times[-1],
        transmissivity=TRANSMISSIVITY,
        storativity=STORATIVITY,
        pumping_rate=PUMPING_RATE,
    )
    mask = (mesh.r_centers >= COMPARISON_MIN_R) & (mesh.r_centers <= COMPARISON_MAX_R)
    relative_l2_error = float(np.linalg.norm(simulated[mask] - analytical[mask]) / np.linalg.norm(analytical[mask]))
    return {
        "n_r": n_r,
        "n_theta": n_theta,
        "r_max_m": r_max,
        "n_steps": n_steps,
        "relative_l2_error": relative_l2_error,
        "max_absolute_error_m": float(np.max(np.abs(simulated[mask] - analytical[mask]))),
        "final_mass_balance_error": float(result.mass_balance_error[-1]),
    }


def run_numerical_verification(
    report_path: str | Path = "outputs/reports/numerical_verification.json",
) -> dict:
    """Run grid, time-step, and outer-boundary verification checks."""

    grid = [
        _profile_error(n_r=n_r, n_theta=8, r_max=2500.0, n_steps=25)
        for n_r in (80, 120, 180)
    ]
    time = [
        _profile_error(n_r=140, n_theta=8, r_max=2500.0, n_steps=n_steps)
        for n_steps in (10, 18, 30)
    ]
    boundary = [
        _profile_error(n_r=140, n_theta=8, r_max=r_max, n_steps=25)
        for r_max in (900.0, 2500.0, 5000.0)
    ]
    baseline = _profile_error(n_r=180, n_theta=16, r_max=2500.0, n_steps=25)

    grid_errors = [row["relative_l2_error"] for row in grid]
    time_errors = [row["relative_l2_error"] for row in time]
    boundary_errors = [row["relative_l2_error"] for row in boundary]
    checks = {
        "baseline_theis_error": {
            "pass": baseline["relative_l2_error"] < 0.06,
            "criterion": "relative_l2_error < 0.06 inside 1-350 m benchmark window",
        },
        "grid_refinement": {
            "pass": max(grid_errors) < 0.06 and (max(grid_errors) - min(grid_errors)) < 0.005,
            "criterion": "all grid errors remain below 0.06 and vary by < 0.005",
        },
        "time_step_refinement": {
            "pass": time_errors[-1] < time_errors[0] and time_errors[-1] < 0.06,
            "criterion": "finest time discretization improves over coarsest and remains below 0.06",
        },
        "outer_boundary_sensitivity": {
            "pass": boundary_errors[1] < 0.06 and abs(boundary_errors[2] - boundary_errors[1]) < 0.02,
            "criterion": "baseline boundary error < 0.06 and larger boundary changes error by < 0.02",
        },
    }
    report = {
        "benchmark": "TRUST-K finite-volume pumping solver numerical verification",
        "baseline": baseline,
        "grid_refinement": grid,
        "time_step_refinement": time,
        "outer_boundary_sensitivity": boundary,
        "checks": checks,
    }

    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    table_dir = path.parent.parent / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(grid).to_csv(table_dir / "numerical_grid_refinement.csv", index=False)
    pd.DataFrame(time).to_csv(table_dir / "numerical_time_step_refinement.csv", index=False)
    pd.DataFrame(boundary).to_csv(table_dir / "numerical_boundary_sensitivity.csv", index=False)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="outputs/reports/numerical_verification.json")
    args = parser.parse_args(argv)
    report = run_numerical_verification(args.report)
    for name, check in report["checks"].items():
        status = "pass" if check["pass"] else "fail"
        print(f"{name}={status}")
    print(f"baseline_relative_l2_error={report['baseline']['relative_l2_error']:.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the homogeneous confined-aquifer Theis benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trustk.analytical.theis import theis_drawdown
from trustk.mesh.polar_mesh import make_log_polar_mesh
from trustk.physics.fv_solver import radial_average, simulate_constant_rate_pumping
from trustk.plotting.style import export_figure, journal_width, set_trustk_style


def _make_benchmark_plot(
    radius: np.ndarray,
    simulated: np.ndarray,
    analytical: np.ndarray,
    relative_error: np.ndarray,
    figure_prefix: Path,
) -> None:
    set_trustk_style()
    fig, axes = plt.subplots(1, 2, figsize=(journal_width(170), 3.1), gridspec_kw={"width_ratios": [1.1, 1.0]})
    ax = axes[0]
    ax.plot(radius, analytical, color=cmc.vik(0.22), lw=1.2, label="Theis solution")
    ax.plot(radius, simulated, color=cmc.vik(0.78), lw=1.0, ls="--", label="finite-volume")
    ax.set_xscale("log")
    ax.set_xlim(0.06, 500.0)
    ax.set_xlabel("Radius from pumping well (m)")
    ax.set_ylabel("Drawdown (m)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.02, 0.96, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")

    ax = axes[1]
    ax.plot(radius, relative_error, color=cmc.vik(0.55), lw=1.0)
    ax.axhline(0, color="0.25", lw=0.7)
    ax.set_xscale("log")
    ax.set_xlim(0.06, 500.0)
    ax.set_ylim(-0.08, 0.04)
    ax.set_xlabel("Radius from pumping well (m)")
    ax.set_ylabel("Relative error")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.02, 0.96, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")
    fig.subplots_adjust(wspace=0.35)
    export_figure(fig, figure_prefix)
    plt.close(fig)


def run_theis_benchmark(
    report_path: str | Path = "outputs/reports/theis_benchmark.json",
    figure_prefix: str | Path = "outputs/figures/fig03_benchmark_theis",
    *,
    n_r: int = 180,
    n_theta: int = 16,
) -> dict:
    """Run a uniform-field finite-volume simulation and compare to Theis."""

    report_path = Path(report_path)
    figure_prefix = Path(figure_prefix)
    mesh = make_log_polar_mesh(r_w=0.05, r_max=2500.0, n_r=n_r, n_theta=n_theta)
    transmissivity = 4e-3
    storativity = 2e-4
    pumping_rate = 1.2e-3
    times = np.geomspace(30.0, 1.0e5, 25)

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
    relative_error_profile = (simulated - analytical) / np.maximum(np.abs(analytical), 1e-12)
    mask = (mesh.r_centers > 1.0) & (mesh.r_centers < 350.0)
    relative_l2_error = float(
        np.linalg.norm(simulated[mask] - analytical[mask]) / np.linalg.norm(analytical[mask])
    )
    max_absolute_error = float(np.max(np.abs(simulated[mask] - analytical[mask])))

    figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    _make_benchmark_plot(mesh.r_centers, simulated, analytical, relative_error_profile, figure_prefix)
    pd.DataFrame(
        {
            "radius_m": mesh.r_centers,
            "fv_drawdown_m": simulated,
            "theis_drawdown_m": analytical,
            "relative_error": relative_error_profile,
        }
    ).to_csv(figure_prefix.with_suffix(".csv"), index=False)
    figure_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "figure": figure_prefix.name,
                "palette": "official cmcrameri cmc.vik",
                "benchmark": "homogeneous confined-aquifer Theis solution",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report = {
        "benchmark": "Theis homogeneous confined pumping",
        "n_r": n_r,
        "n_theta": n_theta,
        "r_w_m": 0.05,
        "r_max_m": 2500.0,
        "transmissivity_m2_s": transmissivity,
        "storativity": storativity,
        "pumping_rate_m3_s": pumping_rate,
        "comparison_time_s": float(times[-1]),
        "comparison_radius_min_m": 1.0,
        "comparison_radius_max_m": 350.0,
        "relative_l2_error": relative_l2_error,
        "max_absolute_error_m": max_absolute_error,
        "outer_boundary_mass_error_final": float(result.mass_balance_error[-1]),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="outputs/reports/theis_benchmark.json")
    parser.add_argument("--figure-prefix", default="outputs/figures/fig03_benchmark_theis")
    parser.add_argument("--n-r", type=int, default=180)
    parser.add_argument("--n-theta", type=int, default=16)
    args = parser.parse_args(argv)
    report = run_theis_benchmark(args.report, args.figure_prefix, n_r=args.n_r, n_theta=args.n_theta)
    print(f"relative_l2_error={report['relative_l2_error']:.4g}")
    print(f"max_absolute_error_m={report['max_absolute_error_m']:.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

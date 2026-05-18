"""Run the homogeneous quasi-steady slug-recovery benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trustk.mesh.polar_mesh import make_log_polar_mesh
from trustk.physics.slug_solver import quasi_steady_slug_head, simulate_slug_recovery
from trustk.plotting.style import export_figure, journal_width, set_trustk_style


def _plot_slug_benchmark(times: np.ndarray, numerical: np.ndarray, reference: np.ndarray, figure_prefix: Path) -> None:
    set_trustk_style()
    active = reference > 1e-3
    plot_times = times[active]
    plot_numerical = np.maximum(numerical[active], 1e-12)
    plot_reference = reference[active]
    fig, axes = plt.subplots(1, 2, figsize=(journal_width(170), 3.1), gridspec_kw={"width_ratios": [1.05, 1.0]})

    ax = axes[0]
    ax.plot(plot_times, plot_reference, color=cmc.vik(0.22), lw=1.2, label="quasi-steady reference")
    ax.plot(plot_times, plot_numerical, color=cmc.vik(0.78), lw=1.0, ls="--", label="finite-volume")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Normalized well head, $H_w/H_0$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.02, 0.96, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")

    ax = axes[1]
    relative_error = (plot_numerical - plot_reference) / np.maximum(plot_reference, 1e-12)
    ax.plot(plot_times, relative_error, color=cmc.vik(0.55), lw=1.0)
    ax.axhline(0, color="0.25", lw=0.7)
    ax.set_ylim(-0.05, 0.22)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Relative error")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.02, 0.96, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")
    fig.subplots_adjust(wspace=0.35)
    export_figure(fig, figure_prefix)
    plt.close(fig)


def run_slug_benchmark(
    report_path: str | Path = "outputs/reports/slug_benchmark.json",
    figure_prefix: str | Path = "outputs/figures/fig04_benchmark_slug",
    *,
    n_r: int = 120,
    n_theta: int = 12,
) -> dict:
    """Run the quasi-steady low-storativity slug benchmark."""

    report_path = Path(report_path)
    figure_prefix = Path(figure_prefix)
    r_w = 0.05
    r_max = 100.0
    transmissivity = 2e-3
    storativity = 1e-9
    well_storage = 0.02
    initial_well_head = 1.0
    times = np.linspace(1.0, 1200.0, 160)

    mesh = make_log_polar_mesh(r_w=r_w, r_max=r_max, n_r=n_r, n_theta=n_theta)
    result = simulate_slug_recovery(
        mesh,
        transmissivity=np.full(mesh.shape, transmissivity),
        storativity=storativity,
        well_storage=well_storage,
        initial_well_head=initial_well_head,
        times=times,
    )
    reference = quasi_steady_slug_head(
        times,
        initial_well_head=initial_well_head,
        transmissivity=transmissivity,
        well_storage=well_storage,
        r_w=r_w,
        r_max=r_max,
    )
    relative_l2_error = float(np.linalg.norm(result.well_head - reference) / np.linalg.norm(reference))
    active = result.well_head[:-1] > 1e-8
    monotonic_recovery_pass = bool(np.min(result.well_head) > -1e-8 and np.all(np.diff(result.well_head)[active] < 0))

    figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    _plot_slug_benchmark(times, result.well_head / initial_well_head, reference / initial_well_head, figure_prefix)
    pd.DataFrame(
        {
            "time_s": times,
            "fv_normalized_head": result.well_head / initial_well_head,
            "reference_normalized_head": reference / initial_well_head,
            "relative_error": (result.well_head - reference) / np.maximum(reference, 1e-12),
        }
    ).to_csv(figure_prefix.with_suffix(".csv"), index=False)
    figure_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "figure": figure_prefix.name,
                "palette": "official cmcrameri cmc.vik",
                "benchmark": "low-storativity quasi-steady slug recovery",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report = {
        "benchmark": "homogeneous quasi-steady slug recovery",
        "n_r": n_r,
        "n_theta": n_theta,
        "r_w_m": r_w,
        "r_max_m": r_max,
        "transmissivity_m2_s": transmissivity,
        "storativity": storativity,
        "well_storage_m2": well_storage,
        "initial_well_head_m": initial_well_head,
        "relative_l2_error": relative_l2_error,
        "monotonic_recovery_pass": monotonic_recovery_pass,
        "final_mass_balance_error": float(result.mass_balance_error[-1]),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="outputs/reports/slug_benchmark.json")
    parser.add_argument("--figure-prefix", default="outputs/figures/fig04_benchmark_slug")
    parser.add_argument("--n-r", type=int, default=120)
    parser.add_argument("--n-theta", type=int, default=12)
    args = parser.parse_args(argv)
    report = run_slug_benchmark(args.report, args.figure_prefix, n_r=args.n_r, n_theta=args.n_theta)
    print(f"relative_l2_error={report['relative_l2_error']:.4g}")
    print(f"monotonic_recovery_pass={report['monotonic_recovery_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Verify Cartesian random-field generation and polar-mesh mapping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trustk.mesh.polar_mesh import make_log_polar_mesh
from trustk.plotting.style import export_figure, journal_width, set_trustk_style
from trustk.random_fields.gaussian_field import GaussianField2D, generate_gaussian_logk_field
from trustk.random_fields.mapping import map_cartesian_field_to_polar, radial_symmetry_score


def _make_mapping_plot(
    field: GaussianField2D,
    mapped_logk: np.ndarray,
    mesh,
    figure_prefix: Path,
    radial_theta_std: np.ndarray,
) -> None:
    set_trustk_style()
    fig = plt.figure(figsize=(journal_width(170), 4.7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.05], width_ratios=[1.0, 1.0, 0.82])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]

    vmin = float(min(np.min(field.logk), np.min(mapped_logk)))
    vmax = float(max(np.max(field.logk), np.max(mapped_logk)))

    ax = axes[0]
    image = ax.pcolormesh(field.x, field.y, field.logk, shading="auto", cmap=cmc.lipari, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")

    ax = axes[1]
    rf, tf = np.meshgrid(mesh.r_faces, mesh.theta_faces, indexing="ij")
    x_faces = rf * np.cos(tf)
    y_faces = rf * np.sin(tf)
    ax.pcolormesh(x_faces, y_faces, mapped_logk, shading="flat", cmap=cmc.lipari, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.02, 0.97, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")

    ax = axes[2]
    ax.plot(mesh.r_centers, radial_theta_std, color=cmc.vik(0.70), lw=1.2)
    ax.axhline(float(np.mean(radial_theta_std)), color="0.35", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("Radius (m)")
    ax.set_ylabel(r"Angular SD of $\ln K$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.04, 0.97, "(c)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")

    cax = fig.add_subplot(gs[1, 0:2])
    cbar = fig.colorbar(image, cax=cax, orientation="horizontal")
    cbar.set_label(r"$\ln K$")

    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.16, top=0.96, wspace=0.34, hspace=0.36)
    export_figure(fig, figure_prefix)
    plt.close(fig)


def run_random_field_verification(
    report_path: str | Path = "outputs/reports/random_field_verification.json",
    figure_prefix: str | Path = "outputs/figures/fig05_random_field_mapping",
    *,
    nx: int = 160,
    ny: int = 160,
    dx: float = 5.0,
    dy: float = 5.0,
    mean_logk: float = -11.0,
    sigma_logk: float = 0.8,
    corr_len_x: float = 80.0,
    corr_len_y: float = 35.0,
    seed: int = 20260517,
    r_w: float = 0.2,
    r_max: float = 350.0,
    n_r: int = 96,
    n_theta: int = 96,
) -> dict:
    """Generate one field, map it to the polar mesh, and write diagnostics."""

    report_path = Path(report_path)
    figure_prefix = Path(figure_prefix)
    field = generate_gaussian_logk_field(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        mean_logk=mean_logk,
        sigma_logk=sigma_logk,
        corr_len_x=corr_len_x,
        corr_len_y=corr_len_y,
        seed=seed,
    )
    mesh = make_log_polar_mesh(r_w=r_w, r_max=r_max, n_r=n_r, n_theta=n_theta)
    mapped_logk = map_cartesian_field_to_polar(field, mesh)

    radial_theta_std = np.std(mapped_logk, axis=1)
    outer_mean_theta_std = float(np.mean(radial_theta_std[-max(5, n_r // 5) :]))
    symmetry_score = radial_symmetry_score(mapped_logk)

    cart_mean = float(np.mean(field.logk))
    cart_std = float(np.std(field.logk))
    polar_mean = float(np.mean(mapped_logk))
    polar_std = float(np.std(mapped_logk))
    checks = {
        "cartesian_moments": {
            "pass": bool(abs(cart_mean - mean_logk) < 1e-10 and abs(cart_std - sigma_logk) < 1e-10),
            "mean_tolerance": 1e-10,
            "std_tolerance": 1e-10,
        },
        "polar_domain_covered": {
            "pass": bool(np.all(np.isfinite(mapped_logk))),
        },
        "theta_varying_heterogeneity": {
            "pass": bool(symmetry_score > 0.20 and outer_mean_theta_std > 0.05),
            "minimum_radial_symmetry_score": 0.20,
            "minimum_outer_mean_theta_std": 0.05,
        },
    }

    report = {
        "purpose": "verify that synthetic lnK fields are generated in Cartesian coordinates and retain angular heterogeneity after mapping to the polar solver mesh",
        "field": {
            "nx": nx,
            "ny": ny,
            "dx_m": dx,
            "dy_m": dy,
            "mean_logk_target": mean_logk,
            "sigma_logk_target": sigma_logk,
            "corr_len_x_m": corr_len_x,
            "corr_len_y_m": corr_len_y,
            "seed": seed,
            "extent": field.extent,
            "cartesian_mean_logk": cart_mean,
            "cartesian_std_logk": cart_std,
        },
        "polar_mesh": {
            "r_w_m": r_w,
            "r_max_m": r_max,
            "n_r": n_r,
            "n_theta": n_theta,
            "polar_mean_logk": polar_mean,
            "polar_std_logk": polar_std,
            "radial_symmetry_score": symmetry_score,
            "outer_mean_theta_std_logk": outer_mean_theta_std,
        },
        "checks": checks,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    _make_mapping_plot(field, mapped_logk, mesh, figure_prefix, radial_theta_std)
    pd.DataFrame(
        {
            "radius_m": mesh.r_centers,
            "theta_std_logk": radial_theta_std,
        }
    ).to_csv(figure_prefix.with_suffix(".csv"), index=False)
    figure_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "figure": figure_prefix.name,
                "palette": "official cmcrameri cmc.lipari and cmc.vik",
                "purpose": "Cartesian random-field and polar-mapping verification",
                "radial_symmetry_score": symmetry_score,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="outputs/reports/random_field_verification.json")
    parser.add_argument("--figure-prefix", default="outputs/figures/fig05_random_field_mapping")
    parser.add_argument("--nx", type=int, default=160)
    parser.add_argument("--ny", type=int, default=160)
    parser.add_argument("--dx", type=float, default=5.0)
    parser.add_argument("--dy", type=float, default=5.0)
    parser.add_argument("--r-max", type=float, default=350.0)
    parser.add_argument("--n-r", type=int, default=96)
    parser.add_argument("--n-theta", type=int, default=96)
    args = parser.parse_args(argv)
    report = run_random_field_verification(
        report_path=args.report,
        figure_prefix=args.figure_prefix,
        nx=args.nx,
        ny=args.ny,
        dx=args.dx,
        dy=args.dy,
        r_max=args.r_max,
        n_r=args.n_r,
        n_theta=args.n_theta,
    )
    score = report["polar_mesh"]["radial_symmetry_score"]
    passed = report["checks"]["theta_varying_heterogeneity"]["pass"]
    print(f"radial_symmetry_score={score:.3f}")
    print(f"theta_varying_heterogeneity_pass={passed}")
    return 0 if all(item["pass"] for item in report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

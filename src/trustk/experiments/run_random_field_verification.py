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
from trustk.random_fields.joint_fields import JointFieldConfig, JointLogField2D, generate_joint_log_fields
from trustk.random_fields.mapping import map_cartesian_field_to_polar, radial_symmetry_score


def _make_mapping_plot(
    field: JointLogField2D,
    mapped_logk: np.ndarray,
    mapped_logss: np.ndarray,
    mesh,
    figure_prefix: Path,
    radial_theta_std_logk: np.ndarray,
    radial_theta_std_logss: np.ndarray,
) -> None:
    set_trustk_style()
    fig = plt.figure(figsize=(journal_width(170), 6.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.06], width_ratios=[1.0, 1.0, 0.92])
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
    ]

    logk_min = float(min(np.min(field.logk), np.min(mapped_logk)))
    logk_max = float(max(np.max(field.logk), np.max(mapped_logk)))
    logss_min = float(min(np.min(field.logss), np.min(mapped_logss)))
    logss_max = float(max(np.max(field.logss), np.max(mapped_logss)))
    mapped_logdiff = mapped_logk - mapped_logss

    ax = axes[0]
    image_k = ax.pcolormesh(field.x, field.y, field.logk, shading="auto", cmap=cmc.lipari, vmin=logk_min, vmax=logk_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")
    ax.set_title(r"Cartesian $\ln K$", fontsize=9)

    ax = axes[1]
    image_ss = ax.pcolormesh(field.x, field.y, field.logss, shading="auto", cmap=cmc.batlow, vmin=logss_min, vmax=logss_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.02, 0.97, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")
    ax.set_title(r"Cartesian $\ln S_s$", fontsize=9)

    ax = axes[2]
    ax.plot(mesh.r_centers, radial_theta_std_logk, color=cmc.lipari(0.72), lw=1.2, label=r"$\ln K$")
    ax.plot(mesh.r_centers, radial_theta_std_logss, color=cmc.batlow(0.62), lw=1.2, label=r"$\ln S_s$")
    ax.set_xscale("log")
    ax.set_xlabel("Radius (m)")
    ax.set_ylabel("Angular SD")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc="best")
    ax.text(0.04, 0.97, "(c)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")

    ax = axes[3]
    rf, tf = np.meshgrid(mesh.r_faces, mesh.theta_faces, indexing="ij")
    x_faces = rf * np.cos(tf)
    y_faces = rf * np.sin(tf)
    ax.pcolormesh(x_faces, y_faces, mapped_logk, shading="flat", cmap=cmc.lipari, vmin=logk_min, vmax=logk_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.02, 0.97, "(d)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")
    ax.set_title(r"Mapped $\ln K$", fontsize=9)

    ax = axes[4]
    ax.pcolormesh(x_faces, y_faces, mapped_logss, shading="flat", cmap=cmc.batlow, vmin=logss_min, vmax=logss_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.02, 0.97, "(e)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")
    ax.set_title(r"Mapped $\ln S_s$", fontsize=9)

    ax = axes[5]
    image_diff = ax.pcolormesh(x_faces, y_faces, mapped_logdiff, shading="flat", cmap=cmc.vik)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.02, 0.97, "(f)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")
    ax.set_title(r"Mapped $\ln(K/S_s)$", fontsize=9)

    cax_k = fig.add_subplot(gs[2, 0])
    cbar = fig.colorbar(image_k, cax=cax_k, orientation="horizontal")
    cbar.set_label(r"$\ln K$")
    cax_ss = fig.add_subplot(gs[2, 1])
    cbar = fig.colorbar(image_ss, cax=cax_ss, orientation="horizontal")
    cbar.set_label(r"$\ln S_s$")
    cax_diff = fig.add_subplot(gs[2, 2])
    cbar = fig.colorbar(image_diff, cax=cax_diff, orientation="horizontal")
    cbar.set_label(r"$\ln(K/S_s)$")

    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.12, top=0.95, wspace=0.32, hspace=0.46)
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
    mean_logss: float = -11.512925464970229,
    sigma_logss: float = 0.6,
    corr_len_ss_x: float = 55.0,
    corr_len_ss_y: float = 28.0,
    log_correlation: float = 0.5,
    min_logss: float = -16.11809565095832,
    max_logss: float = -6.907755278982137,
    seed: int = 20260517,
    r_w: float = 0.2,
    r_max: float = 350.0,
    n_r: int = 96,
    n_theta: int = 96,
) -> dict:
    report_path = Path(report_path)
    figure_prefix = Path(figure_prefix)
    field = generate_joint_log_fields(
        JointFieldConfig(
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            mean_logk=mean_logk,
            sigma_logk=sigma_logk,
            corr_len_k_x=corr_len_x,
            corr_len_k_y=corr_len_y,
            orientation_rad=0.0,
            mean_logss=mean_logss,
            sigma_logss=sigma_logss,
            corr_len_ss_x=corr_len_ss_x,
            corr_len_ss_y=corr_len_ss_y,
            log_correlation=log_correlation,
            min_logss=min_logss,
            max_logss=max_logss,
            seed=seed,
        )
    )
    mesh = make_log_polar_mesh(r_w=r_w, r_max=r_max, n_r=n_r, n_theta=n_theta)
    mapped_logk = map_cartesian_field_to_polar(field.k_field, mesh)
    mapped_logss = map_cartesian_field_to_polar(field.ss_field, mesh)

    radial_theta_std_logk = np.std(mapped_logk, axis=1)
    radial_theta_std_logss = np.std(mapped_logss, axis=1)
    outer_mean_theta_std_logk = float(np.mean(radial_theta_std_logk[-max(5, n_r // 5) :]))
    outer_mean_theta_std_logss = float(np.mean(radial_theta_std_logss[-max(5, n_r // 5) :]))
    symmetry_score_logk = radial_symmetry_score(mapped_logk)
    symmetry_score_logss = radial_symmetry_score(mapped_logss)

    cart_mean = float(np.mean(field.logk))
    cart_std = float(np.std(field.logk))
    ss_mean = float(np.mean(field.logss))
    ss_std = float(np.std(field.logss))
    polar_mean = float(np.mean(mapped_logk))
    polar_std = float(np.std(mapped_logk))
    polar_mean_logss = float(np.mean(mapped_logss))
    polar_std_logss = float(np.std(mapped_logss))
    checks = {
        "cartesian_moments": {
            "pass": bool(abs(cart_mean - mean_logk) < 1e-10 and abs(cart_std - sigma_logk) < 1e-10),
            "mean_tolerance": 1e-10,
            "std_tolerance": 1e-10,
        },
        "specific_storage_moments": {
            "pass": bool(abs(ss_mean - mean_logss) < 0.08 and abs(ss_std - sigma_logss) < 0.08),
            "mean_tolerance": 0.08,
            "std_tolerance": 0.08,
        },
        "polar_domain_covered": {
            "pass": bool(np.all(np.isfinite(mapped_logk)) and np.all(np.isfinite(mapped_logss))),
        },
        "theta_varying_heterogeneity": {
            "pass": bool(
                symmetry_score_logk > 0.20
                and symmetry_score_logss > 0.20
                and outer_mean_theta_std_logk > 0.05
                and outer_mean_theta_std_logss > 0.05
            ),
            "minimum_radial_symmetry_score": 0.20,
            "minimum_outer_mean_theta_std": 0.05,
        },
        "specific_storage_clipping": {
            "pass": bool(field.specific_storage_clipped_fraction <= 0.05),
            "maximum_clipped_fraction": 0.05,
        },
    }

    report = {
        "purpose": "verify that synthetic lnK and lnSs fields are generated in Cartesian coordinates and retain angular heterogeneity after mapping to the polar solver mesh",
        "field": {
            "nx": nx,
            "ny": ny,
            "dx_m": dx,
            "dy_m": dy,
            "mean_logk_target": mean_logk,
            "sigma_logk_target": sigma_logk,
            "corr_len_x_m": corr_len_x,
            "corr_len_y_m": corr_len_y,
            "mean_logss_target": mean_logss,
            "sigma_logss_target": sigma_logss,
            "corr_len_ss_x_m": corr_len_ss_x,
            "corr_len_ss_y_m": corr_len_ss_y,
            "log_correlation": log_correlation,
            "seed": seed,
            "extent": field.k_field.extent,
            "cartesian_mean_logk": cart_mean,
            "cartesian_std_logk": cart_std,
            "cartesian_mean_logss": ss_mean,
            "cartesian_std_logss": ss_std,
            "specific_storage_clipped_fraction": field.specific_storage_clipped_fraction,
        },
        "polar_mesh": {
            "r_w_m": r_w,
            "r_max_m": r_max,
            "n_r": n_r,
            "n_theta": n_theta,
            "polar_mean_logk": polar_mean,
            "polar_std_logk": polar_std,
            "polar_mean_logss": polar_mean_logss,
            "polar_std_logss": polar_std_logss,
            "radial_symmetry_score_logk": symmetry_score_logk,
            "radial_symmetry_score_logss": symmetry_score_logss,
            "outer_mean_theta_std_logk": outer_mean_theta_std_logk,
            "outer_mean_theta_std_logss": outer_mean_theta_std_logss,
        },
        "checks": checks,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    _make_mapping_plot(field, mapped_logk, mapped_logss, mesh, figure_prefix, radial_theta_std_logk, radial_theta_std_logss)
    pd.DataFrame(
        {
            "radius_m": mesh.r_centers,
            "theta_std_logk": radial_theta_std_logk,
            "theta_std_logss": radial_theta_std_logss,
        }
    ).to_csv(figure_prefix.with_suffix(".csv"), index=False)
    figure_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "figure": figure_prefix.name,
                "palette": "official cmcrameri cmc.lipari, cmc.batlow, and cmc.vik",
                "purpose": "Cartesian joint random-field and polar-mapping verification",
                "radial_symmetry_score_logk": symmetry_score_logk,
                "radial_symmetry_score_logss": symmetry_score_logss,
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
    score = report["polar_mesh"]["radial_symmetry_score_logk"]
    score_ss = report["polar_mesh"]["radial_symmetry_score_logss"]
    passed = report["checks"]["theta_varying_heterogeneity"]["pass"]
    print(f"radial_symmetry_score={score:.3f}")
    print(f"radial_symmetry_score_logss={score_ss:.3f}")
    print(f"theta_varying_heterogeneity_pass={passed}")
    return 0 if all(item["pass"] for item in report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

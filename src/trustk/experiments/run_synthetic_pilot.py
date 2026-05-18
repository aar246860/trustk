"""Run paired synthetic pumping and slug pilot simulations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trustk.mesh.polar_mesh import make_log_polar_mesh
from trustk.physics.fv_solver import simulate_constant_rate_pumping
from trustk.physics.slug_solver import simulate_slug_recovery
from trustk.plotting.style import export_figure, journal_width, set_trustk_style
from trustk.interpretation.conventional import (
    estimate_transmissivity_slug_quasi_steady,
    estimate_transmissivity_theis,
)
from trustk.random_fields.gaussian_field import generate_gaussian_logk_field
from trustk.random_fields.mapping import map_cartesian_field_to_polar
from trustk.targets.support_area import support_targets_from_mapped_logk


def simulate_paired_case(row: pd.Series, *, cartesian_n_cap: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Run pumping and slug responses for one dimensional solver-settings row."""

    row = row.copy()
    n_cart = int(row["cartesian_n_required"])
    if cartesian_n_cap is not None:
        n_cart = min(n_cart, int(cartesian_n_cap))
    n_cart = _odd_grid_size(n_cart)
    extent = float(row["cartesian_extent_m"])
    dx = 2.0 * extent / (n_cart - 1)

    field = generate_gaussian_logk_field(
        nx=n_cart,
        ny=n_cart,
        dx=dx,
        dy=dx,
        mean_logk=float(np.log(row["K0_m_s"])),
        sigma_logk=float(row["sigma_logk"]),
        corr_len_x=float(row["lambda1_m"]),
        corr_len_y=float(row["lambda2_m"]),
        orientation_rad=float(row["phi_lambda_rad"]),
        seed=int(row["random_seed"]),
    )
    mesh = make_log_polar_mesh(
        r_w=float(row["well_radius_m"]),
        r_max=float(row["r_max_m"]),
        n_r=int(row["mesh_n_r"]),
        n_theta=int(row["mesh_n_theta"]),
    )
    mapped_logk = map_cartesian_field_to_polar(field, mesh)
    transmissivity = np.exp(mapped_logk) * float(row["aquifer_thickness_m"])
    pumping_support_radius = min(float(row["RI_P_m"]), float(row["r_max_m"]))
    slug_support_radius = min(
        max(float(row["r_skin_m"]), 5.0 * float(row["well_radius_m"])),
        float(row["r_max_m"]),
    )
    support_targets = support_targets_from_mapped_logk(
        mapped_logk=mapped_logk,
        mesh=mesh,
        pumping_radius_m=pumping_support_radius,
        slug_radius_m=slug_support_radius,
    )

    pumping_times = _geom_times(float(row["pumping_time_max_s"]), int(row["pumping_n_times"]))
    slug_times = _geom_times(float(row["slug_time_max_s"]), int(row["slug_n_times"]))

    pumping = simulate_constant_rate_pumping(
        mesh,
        transmissivity=transmissivity,
        storativity=float(row["storativity"]),
        pumping_rate=float(row["pumping_rate_m3_s"]),
        times=pumping_times,
    )
    obs_idx = int(np.argmin(np.abs(mesh.r_centers - float(row["r_obs_m"]))))
    obs_radius = float(mesh.r_centers[obs_idx])
    drawdown = pumping.drawdown[:, obs_idx, :].mean(axis=1)
    pumping_s_d = 4.0 * np.pi * float(row["T0_m2_s"]) * drawdown / float(row["pumping_rate_m3_s"])
    pumping_fit = estimate_transmissivity_theis(
        times_s=pumping_times,
        drawdown_m=drawdown,
        radius_m=obs_radius,
        storativity=float(row["storativity"]),
        pumping_rate_m3_s=float(row["pumping_rate_m3_s"]),
    )
    k_hat_pumping = pumping_fit.transmissivity_m2_s / float(row["aquifer_thickness_m"])
    pumping_curve = pd.DataFrame(
        {
            "case_id": row["case_id"],
            "method": "pumping",
            "time_s": pumping_times,
            "time_D": pumping_times / float(row["t0_s"]),
            "response_name": "s_D",
            "response_value": pumping_s_d,
            "drawdown_m": drawdown,
            "observation_radius_m": obs_radius,
        }
    )

    slug = simulate_slug_recovery(
        mesh,
        transmissivity=transmissivity,
        storativity=float(row["storativity"]),
        well_storage=float(row["well_storage_m2"]),
        initial_well_head=float(row["initial_slug_head_m"]),
        times=slug_times,
    )
    normalized_head = slug.well_head / float(row["initial_slug_head_m"])
    slug_fit = estimate_transmissivity_slug_quasi_steady(
        times_s=slug_times,
        normalized_head=normalized_head,
        well_storage_m2=float(row["well_storage_m2"]),
        well_radius_m=float(row["well_radius_m"]),
        outer_radius_m=float(row["r_max_m"]),
    )
    k_hat_slug = slug_fit.transmissivity_m2_s / float(row["aquifer_thickness_m"])
    slug_curve = pd.DataFrame(
        {
            "case_id": row["case_id"],
            "method": "slug",
            "time_s": slug_times,
            "time_D": slug_times / float(row["t0_s"]),
            "response_name": "H_w/H_0",
            "response_value": normalized_head,
            "drawdown_m": np.nan,
            "observation_radius_m": float(row["well_radius_m"]),
        }
    )

    summary = {
        "case_id": row["case_id"],
        "random_seed": int(row["random_seed"]),
        "sigma_Y2": float(row["sigma_Y2"]),
        "lambda1_m": float(row["lambda1_m"]),
        "lambda2_m": float(row["lambda2_m"]),
        "r_max_m": float(row["r_max_m"]),
        "used_cartesian_n": n_cart,
        "required_cartesian_n": int(row["cartesian_n_required"]),
        "used_mesh_n_r": int(row["mesh_n_r"]),
        "used_mesh_n_theta": int(row["mesh_n_theta"]),
        "pumping_observation_radius_m": obs_radius,
        **support_targets,
        "K_hat_pumping_m_s": float(k_hat_pumping),
        "K_hat_slug_m_s": float(k_hat_slug),
        "T_hat_pumping_m2_s": float(pumping_fit.transmissivity_m2_s),
        "T_hat_slug_m2_s": float(slug_fit.transmissivity_m2_s),
        "log_residual_pumping": float(np.log(k_hat_pumping) - np.log(support_targets["K_star_pumping_m_s"])),
        "log_residual_slug": float(np.log(k_hat_slug) - np.log(support_targets["K_star_slug_m_s"])),
        "pumping_fit_rmse_log_response": float(pumping_fit.rmse_log_response),
        "slug_fit_rmse_log_response": float(slug_fit.rmse_log_response),
        "pumping_fit_point_count": int(pumping_fit.fit_point_count),
        "slug_fit_point_count": int(slug_fit.fit_point_count),
        "pumping_fit_time_min_s": float(pumping_fit.fit_time_min_s),
        "pumping_fit_time_max_s": float(pumping_fit.fit_time_max_s),
        "slug_fit_time_min_s": float(slug_fit.fit_time_min_s),
        "slug_fit_time_max_s": float(slug_fit.fit_time_max_s),
        "pumping_final_drawdown_m": float(drawdown[-1]),
        "pumping_final_s_D": float(pumping_s_d[-1]),
        "slug_initial_normalized_head": float(normalized_head[0]),
        "slug_final_normalized_head": float(normalized_head[-1]),
        "pumping_mass_balance_error_final": float(pumping.mass_balance_error[-1]),
        "slug_mass_balance_error_final": float(slug.mass_balance_error[-1]),
        "mapped_logk_mean": float(np.mean(mapped_logk)),
        "mapped_logk_std": float(np.std(mapped_logk)),
    }
    curves = pd.concat([pumping_curve, slug_curve], ignore_index=True)
    curves["sigma_Y2"] = float(row["sigma_Y2"])
    return curves, summary


def run_synthetic_pilot(
    settings_path: str | Path = "data/processed/solver_simulation_settings.csv",
    curves_path: str | Path = "data/processed/synthetic_pilot_curves.csv",
    summary_path: str | Path = "data/processed/synthetic_pilot_summary.csv",
    report_path: str | Path = "outputs/reports/synthetic_pilot.json",
    figure_prefix: str | Path = "outputs/figures/fig08_synthetic_pilot_responses",
    *,
    max_cases: int = 12,
    cartesian_n_cap: int | None = 1024,
    mesh_n_r: int | None = 48,
    mesh_n_theta: int | None = 24,
) -> dict:
    """Run a representative paired synthetic pilot batch."""

    settings_path = Path(settings_path)
    curves_path = Path(curves_path)
    summary_path = Path(summary_path)
    report_path = Path(report_path)
    figure_prefix = Path(figure_prefix)

    settings = pd.read_csv(settings_path)
    selected = _select_ready_cases(settings, max_cases=max_cases)
    if mesh_n_r is not None:
        selected["mesh_n_r"] = int(mesh_n_r)
    if mesh_n_theta is not None:
        selected["mesh_n_theta"] = int(mesh_n_theta)

    curve_frames = []
    summaries = []
    for _, row in selected.iterrows():
        curves, summary = simulate_paired_case(row, cartesian_n_cap=cartesian_n_cap)
        curve_frames.append(curves)
        summaries.append(summary)

    all_curves = pd.concat(curve_frames, ignore_index=True)
    summary_table = pd.DataFrame(summaries)
    report = _summarize_pilot(all_curves, summary_table)
    report["purpose"] = "paired synthetic pilot simulations for pumping and slug responses on shared heterogeneous fields"
    report["settings_path"] = str(settings_path)
    report["max_cases_requested"] = int(max_cases)
    report["cartesian_n_cap"] = None if cartesian_n_cap is None else int(cartesian_n_cap)
    report["mesh_override"] = {"n_r": mesh_n_r, "n_theta": mesh_n_theta}

    curves_path.parent.mkdir(parents=True, exist_ok=True)
    all_curves.to_csv(curves_path, index=False)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_table.to_csv(summary_path, index=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    _plot_synthetic_pilot(all_curves, summary_table, figure_prefix)
    figure_prefix.with_suffix(".csv").write_text(all_curves.to_csv(index=False), encoding="utf-8")
    figure_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "figure": figure_prefix.name,
                "palette": "official cmcrameri cmc.batlow and cmc.vik",
                "purpose": "paired pumping and slug pilot responses",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report


def _select_ready_cases(settings: pd.DataFrame, *, max_cases: int) -> pd.DataFrame:
    ready = settings[settings["ready_for_pilot_solver"].astype(bool)].copy()
    if ready.empty:
        raise ValueError("settings table contains no ready cases")
    ready = ready.sort_values("cartesian_n_required").reset_index(drop=True)
    if len(ready) <= max_cases:
        return ready.copy()
    positions = np.unique(np.round(np.linspace(0, len(ready) - 1, max_cases)).astype(int))
    selected = ready.iloc[positions].copy()
    while len(selected) < max_cases:
        remaining = ready.drop(selected.index, errors="ignore")
        selected = pd.concat([selected, remaining.head(max_cases - len(selected))])
    return selected.reset_index(drop=True)


def _summarize_pilot(curves: pd.DataFrame, summary: pd.DataFrame) -> dict:
    methods = set(curves["method"])
    finite = bool(np.isfinite(curves["response_value"].to_numpy(dtype=float)).all())
    pumping = curves[curves["method"] == "pumping"]
    slug = curves[curves["method"] == "slug"]
    pumping_positive = bool((pumping["response_value"] >= 0.0).all())
    slug_decreases = bool((summary["slug_final_normalized_head"] < summary["slug_initial_normalized_head"]).all())
    return {
        "simulated_case_count": int(len(summary)),
        "curve_row_count": int(len(curves)),
        "checks": {
            "paired_methods_present": {"pass": methods == {"pumping", "slug"}},
            "finite_responses": {"pass": finite},
            "pumping_drawdown_nonnegative": {"pass": pumping_positive},
            "slug_recovery_decreases": {"pass": slug_decreases},
        },
        "ranges": {
            "pumping_final_s_D": _range(summary["pumping_final_s_D"]),
            "pumping_final_drawdown_m": _range(summary["pumping_final_drawdown_m"]),
            "slug_final_normalized_head": _range(summary["slug_final_normalized_head"]),
            "used_cartesian_n": _range(summary["used_cartesian_n"]),
        },
    }


def _plot_synthetic_pilot(curves: pd.DataFrame, summary: pd.DataFrame, figure_prefix: Path) -> None:
    set_trustk_style()
    fig, axes = plt.subplots(1, 3, figsize=(journal_width(170), 3.25))
    norm = plt.Normalize(summary["sigma_Y2"].min(), summary["sigma_Y2"].max())

    ax = axes[0]
    for _, case_summary in summary.iterrows():
        case = case_summary["case_id"]
        data = curves[(curves["case_id"] == case) & (curves["method"] == "pumping")]
        color = cmc.batlow(norm(case_summary["sigma_Y2"]))
        ax.plot(data["time_D"], data["response_value"], color=color, lw=0.9, alpha=0.82)
    ax.set_xscale("log")
    ax.set_xlabel(r"$t_D$")
    ax.set_ylabel(r"Pumping $s_D$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.03, 0.96, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")

    ax = axes[1]
    for _, case_summary in summary.iterrows():
        case = case_summary["case_id"]
        data = curves[(curves["case_id"] == case) & (curves["method"] == "slug")]
        color = cmc.batlow(norm(case_summary["sigma_Y2"]))
        ax.plot(data["time_D"], data["response_value"], color=color, lw=0.9, alpha=0.82)
    ax.set_xscale("log")
    ax.set_ylim(bottom=max(-0.05, curves[curves["method"] == "slug"]["response_value"].min() - 0.05), top=1.05)
    ax.set_xlabel(r"$t_D$")
    ax.set_ylabel(r"Slug $H_w/H_0$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.03, 0.96, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")

    ax = axes[2]
    sc = ax.scatter(
        summary["pumping_final_s_D"],
        summary["slug_final_normalized_head"],
        c=summary["sigma_Y2"],
        s=34,
        cmap=cmc.batlow,
        edgecolor="0.25",
        linewidth=0.3,
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"Final pumping $s_D$")
    ax.set_ylabel(r"Final slug $H_w/H_0$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.03, 0.96, "(c)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")

    cbar = fig.colorbar(sc, ax=axes, location="bottom", pad=0.22, shrink=0.72, aspect=35)
    cbar.set_label(r"$\sigma_Y^2$")
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.34, top=0.94, wspace=0.46)
    export_figure(fig, figure_prefix)
    plt.close(fig)


def _geom_times(max_time: float, n_times: int) -> np.ndarray:
    if max_time <= 0.0:
        raise ValueError("max_time must be positive")
    if n_times < 2:
        raise ValueError("n_times must be at least 2")
    return np.geomspace(max_time / 1.0e3, max_time, n_times)


def _odd_grid_size(value: int) -> int:
    value = max(int(value), 17)
    return value + (value % 2 == 0)


def _range(values: pd.Series) -> dict[str, float]:
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(values.median()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", default="data/processed/solver_simulation_settings.csv")
    parser.add_argument("--curves", default="data/processed/synthetic_pilot_curves.csv")
    parser.add_argument("--summary", default="data/processed/synthetic_pilot_summary.csv")
    parser.add_argument("--report", default="outputs/reports/synthetic_pilot.json")
    parser.add_argument("--figure-prefix", default="outputs/figures/fig08_synthetic_pilot_responses")
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--cartesian-n-cap", type=int, default=1024)
    parser.add_argument("--mesh-n-r", type=int, default=48)
    parser.add_argument("--mesh-n-theta", type=int, default=24)
    args = parser.parse_args(argv)
    report = run_synthetic_pilot(
        settings_path=args.settings,
        curves_path=args.curves,
        summary_path=args.summary,
        report_path=args.report,
        figure_prefix=args.figure_prefix,
        max_cases=args.max_cases,
        cartesian_n_cap=args.cartesian_n_cap,
        mesh_n_r=args.mesh_n_r,
        mesh_n_theta=args.mesh_n_theta,
    )
    all_pass = all(item["pass"] for item in report["checks"].values())
    print(f"simulated_case_count={report['simulated_case_count']}")
    print(f"synthetic_pilot_pass={all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

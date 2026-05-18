"""Run TRUST-K synthetic population support targets and residuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trustk.experiments.run_synthetic_pilot import _range, _select_ready_cases, simulate_paired_case
from trustk.plotting.style import export_figure, journal_width, set_trustk_style


def run_synthetic_population(
    settings_path: str | Path = "data/processed/solver_simulation_settings.csv",
    curves_path: str | Path = "data/processed/synthetic_population_curves.csv",
    residuals_path: str | Path = "data/processed/synthetic_population_residuals.csv",
    report_path: str | Path = "outputs/reports/synthetic_population.json",
    figure_prefix: str | Path = "outputs/figures/fig09_support_residuals",
    *,
    max_cases: int = 9999,
    cartesian_n_cap: int | None = 1024,
    mesh_n_r: int | None = 48,
    mesh_n_theta: int | None = 24,
) -> dict:
    """Run all ready synthetic cases and compute support targets and residuals."""

    settings_path = Path(settings_path)
    curves_path = Path(curves_path)
    residuals_path = Path(residuals_path)
    report_path = Path(report_path)
    figure_prefix = Path(figure_prefix)

    settings = pd.read_csv(settings_path)
    ready_count = int(settings["ready_for_pilot_solver"].astype(bool).sum())
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
    residuals = pd.DataFrame(summaries)
    report = _summarize_population(residuals)
    report["purpose"] = "baseline synthetic population support targets and method-specific log residuals"
    report["settings_path"] = str(settings_path)
    report["ready_case_count_in_settings"] = ready_count
    report["max_cases_requested"] = int(max_cases)
    report["cartesian_n_cap"] = None if cartesian_n_cap is None else int(cartesian_n_cap)
    report["mesh_override"] = {"n_r": mesh_n_r, "n_theta": mesh_n_theta}
    report["support_target_definition"] = (
        "Target A: area-weighted geometric mean K inside method-specific support radius"
    )

    curves_path.parent.mkdir(parents=True, exist_ok=True)
    all_curves.to_csv(curves_path, index=False)
    residuals_path.parent.mkdir(parents=True, exist_ok=True)
    residuals.to_csv(residuals_path, index=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    _plot_support_residuals(residuals, figure_prefix)
    figure_prefix.with_suffix(".csv").write_text(residuals.to_csv(index=False), encoding="utf-8")
    figure_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "figure": figure_prefix.name,
                "palette": "official cmcrameri cmc.batlow and cmc.vik",
                "purpose": "support target comparison and method-specific log residuals",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report


def _summarize_population(residuals: pd.DataFrame) -> dict:
    finite_columns = [
        "K_star_pumping_m_s",
        "K_star_slug_m_s",
        "K_hat_pumping_m_s",
        "K_hat_slug_m_s",
        "log_residual_pumping",
        "log_residual_slug",
    ]
    finite = bool(np.isfinite(residuals[finite_columns].to_numpy(dtype=float)).all())
    positive_targets = bool((residuals[["K_star_pumping_m_s", "K_star_slug_m_s"]] > 0.0).all().all())
    positive_interpreted = bool((residuals[["K_hat_pumping_m_s", "K_hat_slug_m_s"]] > 0.0).all().all())
    return {
        "simulated_case_count": int(len(residuals)),
        "checks": {
            "finite_targets_and_residuals": {"pass": finite},
            "positive_targets": {"pass": positive_targets},
            "positive_interpreted_conductivity": {"pass": positive_interpreted},
        },
        "ranges": {
            "K_star_pumping_m_s": _range(residuals["K_star_pumping_m_s"]),
            "K_star_slug_m_s": _range(residuals["K_star_slug_m_s"]),
            "K_hat_pumping_m_s": _range(residuals["K_hat_pumping_m_s"]),
            "K_hat_slug_m_s": _range(residuals["K_hat_slug_m_s"]),
            "log_residual_pumping": _range(residuals["log_residual_pumping"]),
            "log_residual_slug": _range(residuals["log_residual_slug"]),
        },
        "residual_statistics": {
            "pumping": _residual_stats(residuals["log_residual_pumping"]),
            "slug": _residual_stats(residuals["log_residual_slug"]),
        },
    }


def _residual_stats(values: pd.Series) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "median": float(values.median()),
        "q05": float(values.quantile(0.05)),
        "q95": float(values.quantile(0.95)),
    }


def _plot_support_residuals(residuals: pd.DataFrame, figure_prefix: Path) -> None:
    set_trustk_style()
    fig, axes = plt.subplots(2, 2, figsize=(journal_width(170), 5.55))
    sigma = residuals["sigma_Y2"].to_numpy(dtype=float)
    norm = plt.Normalize(float(np.min(sigma)), float(np.max(sigma)))
    many_cases = len(residuals) > 500
    point_size = 4.8 + (10.0 if many_cases else 30.0) * _scaled(sigma)
    point_alpha = 0.30 if many_cases else 0.75
    all_k = residuals[
        [
            "K_star_slug_m_s",
            "K_star_pumping_m_s",
            "K_hat_pumping_m_s",
            "K_hat_slug_m_s",
        ]
    ].to_numpy(dtype=float)
    k_low, k_high = _log_axis_limits(all_k)

    ax = axes[0, 0]
    sc = ax.scatter(
        residuals["K_star_slug_m_s"],
        residuals["K_star_pumping_m_s"],
        c=sigma,
        s=point_size,
        cmap=cmc.batlow,
        edgecolor="none",
        linewidth=0.0,
        alpha=point_alpha,
        rasterized=True,
    )
    _one_to_one(ax, k_low, k_high)
    _apply_log_axes(ax, k_low, k_high)
    ax.set_xlabel(r"Slug target $K_S^*$ (m s$^{-1}$)")
    ax.set_ylabel(r"Pumping target $K_P^*$ (m s$^{-1}$)")
    _panel_tag(ax, "(a)")
    _n_tag(ax, len(residuals))

    ax = axes[0, 1]
    ax.scatter(
        residuals["K_star_pumping_m_s"],
        residuals["K_hat_pumping_m_s"],
        c=sigma,
        s=point_size,
        cmap=cmc.batlow,
        edgecolor="none",
        linewidth=0.0,
        alpha=point_alpha,
        rasterized=True,
    )
    _one_to_one(ax, k_low, k_high)
    _apply_log_axes(ax, k_low, k_high)
    ax.set_xlabel(r"$K_P^*$ (m s$^{-1}$)")
    ax.set_ylabel(r"$\hat K_P$ (m s$^{-1}$)")
    _panel_tag(ax, "(b)")
    _n_tag(ax, len(residuals))

    ax = axes[1, 0]
    ax.scatter(
        residuals["K_star_slug_m_s"],
        residuals["K_hat_slug_m_s"],
        c=sigma,
        s=point_size,
        cmap=cmc.batlow,
        edgecolor="none",
        linewidth=0.0,
        alpha=point_alpha,
        rasterized=True,
    )
    _one_to_one(ax, k_low, k_high)
    _apply_log_axes(ax, k_low, k_high)
    ax.set_xlabel(r"$K_S^*$ (m s$^{-1}$)")
    ax.set_ylabel(r"$\hat K_S$ (m s$^{-1}$)")
    _panel_tag(ax, "(c)")
    _n_tag(ax, len(residuals))

    ax = axes[1, 1]
    positions = np.array([0.85, 1.15])
    colors = [cmc.vik(0.28), cmc.vik(0.72)]
    data = [residuals["log_residual_pumping"].to_numpy(dtype=float), residuals["log_residual_slug"].to_numpy(dtype=float)]
    parts = ax.violinplot(data, positions=positions, widths=0.22, showmeans=True, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.58)
    parts["cmeans"].set_color("0.15")
    rng = np.random.default_rng(20260517)
    for x, values, color in zip(positions, data, colors):
        jitter = rng.uniform(-0.050, 0.050, len(values))
        ax.scatter(
            np.full(len(values), x) + jitter,
            values,
            s=4.0 if many_cases else 10,
            color=color,
            edgecolor="none",
            linewidth=0.0,
            alpha=0.28 if many_cases else 0.70,
            rasterized=True,
        )
    ax.axhline(0.0, color="0.25", lw=0.8, ls="--")
    ax.set_xlim(0.55, 1.45)
    ax.set_xticks(positions)
    ax.set_xticklabels(["Pumping", "Slug"])
    ax.set_ylabel(r"Log residual $r_m$")
    _panel_tag(ax, "(d)")
    _n_tag(ax, len(residuals), text="n=1728 per method")

    for panel in axes.ravel():
        panel.spines[["top", "right"]].set_visible(False)
        panel.spines[["left", "bottom"]].set_linewidth(0.55)

    fig.subplots_adjust(left=0.085, right=0.850, bottom=0.105, top=0.965, wspace=0.34, hspace=0.37)
    cax = fig.add_axes([0.890, 0.235, 0.014, 0.555])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label(r"$\sigma_Y^2$")
    cbar.outline.set_visible(False)
    export_figure(fig, figure_prefix)
    plt.close(fig)
    Path(figure_prefix).with_suffix(".json").write_text(
        json.dumps(
            {
                "figure": Path(figure_prefix).name,
                "palette": "official cmcrameri cmc.batlow for scatter and cmc.vik for residual distributions",
                "case_count": int(len(residuals)),
                "overlap_check": "All scatter panels use identical log-axis limits and n annotations; colorbar outline removed; top/right spines removed.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _one_to_one(ax, low: float, high: float) -> None:
    ax.plot([low, high], [low, high], color="0.2", lw=0.8, ls="--", zorder=0)


def _log_axis_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    low = 10.0 ** np.floor(np.log10(float(np.min(finite))))
    high = 10.0 ** np.ceil(np.log10(float(np.max(finite))))
    return low, high


def _apply_log_axes(ax, low: float, high: float) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)


def _panel_tag(ax, label: str) -> None:
    ax.text(
        0.030,
        0.955,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.0},
    )


def _n_tag(ax, n: int, text: str | None = None) -> None:
    ax.text(
        0.965,
        0.045,
        text or f"n={n}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.3,
        color="0.28",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
    )


def _scaled(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    span = float(np.max(values) - np.min(values))
    if span <= 0.0:
        return np.zeros_like(values)
    return (values - np.min(values)) / span


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", default="data/processed/solver_simulation_settings.csv")
    parser.add_argument("--curves", default="data/processed/synthetic_population_curves.csv")
    parser.add_argument("--residuals", default="data/processed/synthetic_population_residuals.csv")
    parser.add_argument("--report", default="outputs/reports/synthetic_population.json")
    parser.add_argument("--figure-prefix", default="outputs/figures/fig09_support_residuals")
    parser.add_argument("--max-cases", type=int, default=9999)
    parser.add_argument("--cartesian-n-cap", type=int, default=1024)
    parser.add_argument("--mesh-n-r", type=int, default=48)
    parser.add_argument("--mesh-n-theta", type=int, default=24)
    args = parser.parse_args(argv)
    report = run_synthetic_population(
        settings_path=args.settings,
        curves_path=args.curves,
        residuals_path=args.residuals,
        report_path=args.report,
        figure_prefix=args.figure_prefix,
        max_cases=args.max_cases,
        cartesian_n_cap=args.cartesian_n_cap,
        mesh_n_r=args.mesh_n_r,
        mesh_n_theta=args.mesh_n_theta,
    )
    all_pass = all(item["pass"] for item in report["checks"].values())
    print(f"simulated_case_count={report['simulated_case_count']}")
    print(f"synthetic_population_pass={all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

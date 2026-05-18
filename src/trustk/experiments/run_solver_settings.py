"""Convert TRUST-K Pi-space registry rows into dimensional solver settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from trustk.dimensionless.settings import (
    DimensionalBaseConfig,
    convert_case_registry_to_solver_settings,
    summarize_solver_settings,
)
from trustk.plotting.style import export_figure, journal_width, set_trustk_style


def _plot_solver_settings(settings: pd.DataFrame, figure_prefix: Path) -> None:
    set_trustk_style()
    fig, axes = plt.subplots(1, 3, figsize=(journal_width(170), 3.25))
    ready = settings["ready_for_pilot_solver"].astype(bool).to_numpy()
    colors = [cmc.batlow(0.72) if item else cmc.batlow(0.22) for item in ready]

    ax = axes[0]
    ax.scatter(
        settings["lambda2_m"],
        settings["r_max_m"],
        s=28,
        c=colors,
        edgecolor="0.25",
        linewidth=0.25,
        alpha=0.9,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda_2$ (m)")
    ax.set_ylabel(r"$R_{max}$ (m)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=cmc.batlow(0.72), markeredgecolor="0.25", label="ready"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=cmc.batlow(0.22), markeredgecolor="0.25", label="high grid"),
        ],
        frameon=False,
        loc="lower right",
        handletextpad=0.4,
        borderpad=0.2,
    )
    ax.text(0.03, 0.96, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")

    ax = axes[1]
    sc = ax.scatter(
        settings["pumping_time_max_s"],
        settings["slug_time_max_s"],
        c=np.log10(settings["cartesian_n_required"]),
        s=28,
        cmap=cmc.lipari,
        edgecolor="0.25",
        linewidth=0.25,
        alpha=0.9,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Pumping duration (s)")
    ax.set_ylabel("Slug duration (s)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.03, 0.96, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")

    ax = axes[2]
    bins = np.arange(0, 7)
    ax.hist(
        np.log10(settings["cartesian_n_required"]),
        bins=12,
        color=cmc.batlow(0.62),
        edgecolor="0.25",
        linewidth=0.35,
    )
    limit = float(np.log10(settings["max_cartesian_n_limit"].iloc[0]))
    ax.axvline(limit, color=cmc.vik(0.18), lw=1.1, ls="--")
    ax.text(limit + 0.015, 0.95, "limit", transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=7)
    ax.set_xlabel(r"$\log_{10}(N_{cart})$")
    ax.set_ylabel("Case count")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.03, 0.96, "(c)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")

    cbar = fig.colorbar(sc, ax=axes[1], location="bottom", pad=0.25, shrink=0.8, aspect=25)
    cbar.set_label(r"$\log_{10}(N_{cart})$")
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.34, top=0.94, wspace=0.46)
    export_figure(fig, figure_prefix)
    plt.close(fig)


def run_solver_settings(
    registry_path: str | Path = "data/processed/dimensionless_case_registry.csv",
    settings_path: str | Path = "data/processed/solver_simulation_settings.csv",
    report_path: str | Path = "outputs/reports/solver_settings.json",
    figure_prefix: str | Path = "outputs/figures/fig07_solver_settings",
    *,
    max_cartesian_n: int = 1024,
) -> dict:
    """Convert the Pi registry into dimensional settings and write diagnostics."""

    registry_path = Path(registry_path)
    settings_path = Path(settings_path)
    report_path = Path(report_path)
    figure_prefix = Path(figure_prefix)

    registry = pd.read_csv(registry_path)
    base = DimensionalBaseConfig(max_cartesian_n=max_cartesian_n)
    settings = convert_case_registry_to_solver_settings(registry, base)
    report = summarize_solver_settings(settings)
    report["purpose"] = (
        "convert dimensionless TRUST-K registry rows into dimensional solver inputs "
        "and screen Cartesian random-field resolution requirements"
    )
    report["base_config"] = {
        "mean_logk": base.mean_logk,
        "aquifer_thickness_m": base.aquifer_thickness_m,
        "storativity": base.storativity,
        "well_radius_m": base.well_radius_m,
        "pumping_rate_m3_s": base.pumping_rate_m3_s,
        "initial_slug_head_m": base.initial_slug_head_m,
        "cells_per_minor_correlation": base.cells_per_minor_correlation,
        "cartesian_domain_margin": base.cartesian_domain_margin,
        "max_cartesian_n": base.max_cartesian_n,
    }

    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings.to_csv(settings_path, index=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    _plot_solver_settings(settings, figure_prefix)
    figure_prefix.with_suffix(".csv").write_text(settings.to_csv(index=False), encoding="utf-8")
    figure_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "figure": figure_prefix.name,
                "palette": "official cmcrameri cmc.batlow, cmc.lipari, and cmc.vik",
                "purpose": "dimensional solver readiness from Pi-space registry",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="data/processed/dimensionless_case_registry.csv")
    parser.add_argument("--settings", default="data/processed/solver_simulation_settings.csv")
    parser.add_argument("--report", default="outputs/reports/solver_settings.json")
    parser.add_argument("--figure-prefix", default="outputs/figures/fig07_solver_settings")
    parser.add_argument("--max-cartesian-n", type=int, default=1024)
    args = parser.parse_args(argv)
    report = run_solver_settings(
        registry_path=args.registry,
        settings_path=args.settings,
        report_path=args.report,
        figure_prefix=args.figure_prefix,
        max_cartesian_n=args.max_cartesian_n,
    )
    all_pass = all(item["pass"] for item in report["checks"].values())
    print(f"ready_case_count={report['ready_case_count']} of {report['n_cases']}")
    print(f"solver_settings_pass={all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Create and verify the TRUST-K pilot dimensionless case registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trustk.dimensionless.registry import build_dimensionless_case_table, summarize_dimensionless_registry
from trustk.plotting.style import export_figure, journal_width, set_trustk_style


def _plot_dimensionless_registry(table: pd.DataFrame, figure_prefix: Path) -> None:
    set_trustk_style()
    fig, axes = plt.subplots(1, 3, figsize=(journal_width(170), 3.25))
    size = 18.0 + 34.0 * _normalize(table["sigma_Y2"].to_numpy())

    ax = axes[0]
    sc = ax.scatter(
        table["lambda1_over_RI"],
        table["lambda2_over_lambda1"],
        c=table["sigma_Y2"],
        s=size,
        cmap=cmc.batlow,
        edgecolor="0.25",
        linewidth=0.25,
        alpha=0.9,
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda_1/R_I$")
    ax.set_ylabel(r"$\lambda_2/\lambda_1$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.03, 0.96, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")

    ax = axes[1]
    ax.scatter(
        table["tD_P_max"],
        table["r_obs_over_RI_P"],
        c=table["sigma_Y2"],
        s=26,
        cmap=cmc.batlow,
        edgecolor="0.25",
        linewidth=0.25,
        alpha=0.9,
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"$t_{D,P}^{max}$")
    ax.set_ylabel(r"$r_{obs}/R_{I,P}$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.03, 0.96, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")

    ax = axes[2]
    ax.scatter(
        table["C_D_w"],
        table["tD_S_max"],
        c=table["sigma_Y2"],
        s=26,
        cmap=cmc.batlow,
        edgecolor="0.25",
        linewidth=0.25,
        alpha=0.9,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$C_{D,w}$")
    ax.set_ylabel(r"$t_{D,S}^{max}$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.03, 0.96, "(c)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")

    cbar = fig.colorbar(sc, ax=axes, location="bottom", pad=0.18, shrink=0.72, aspect=35)
    cbar.set_label(r"$\sigma_Y^2$")
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.34, top=0.94, wspace=0.46)
    export_figure(fig, figure_prefix)
    plt.close(fig)


def run_dimensionless_registry(
    table_path: str | Path = "data/processed/dimensionless_case_registry.csv",
    report_path: str | Path = "outputs/reports/dimensionless_registry.json",
    figure_prefix: str | Path = "outputs/figures/fig06_dimensionless_registry",
    *,
    n_cases: int = 96,
    seed: int = 20260517,
) -> dict:
    """Write the pilot Pi-space registry, diagnostics, and coverage figure."""

    table_path = Path(table_path)
    report_path = Path(report_path)
    figure_prefix = Path(figure_prefix)
    table = build_dimensionless_case_table(n_cases=n_cases, seed=seed)
    report = summarize_dimensionless_registry(table)
    report["purpose"] = (
        "verify that TRUST-K synthetic cases are indexed by common, pumping-specific, "
        "and slug-specific dimensionless controls before paired simulations are run"
    )
    report["sampling"] = {
        "method": "Latin Hypercube Sampling",
        "seed": seed,
        "n_cases": n_cases,
    }

    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    _plot_dimensionless_registry(table, figure_prefix)
    figure_prefix.with_suffix(".csv").write_text(table.to_csv(index=False), encoding="utf-8")
    figure_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "figure": figure_prefix.name,
                "palette": "official cmcrameri cmc.batlow",
                "purpose": "Pi-space coverage for the pilot TRUST-K synthetic design",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    span = float(np.max(values) - np.min(values))
    if span <= 0.0:
        return np.zeros_like(values)
    return (values - float(np.min(values))) / span


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="data/processed/dimensionless_case_registry.csv")
    parser.add_argument("--report", default="outputs/reports/dimensionless_registry.json")
    parser.add_argument("--figure-prefix", default="outputs/figures/fig06_dimensionless_registry")
    parser.add_argument("--n-cases", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260517)
    args = parser.parse_args(argv)
    report = run_dimensionless_registry(
        table_path=args.table,
        report_path=args.report,
        figure_prefix=args.figure_prefix,
        n_cases=args.n_cases,
        seed=args.seed,
    )
    all_pass = all(item["pass"] for item in report["checks"].values())
    print(f"n_cases={report['n_cases']}")
    print(f"dimensionless_registry_pass={all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

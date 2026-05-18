"""Run fit-quality QC and baseline transformation-uncertainty summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trustk.diagnostics.fit_quality import (
    DEFAULT_PASS_THRESHOLDS,
    DEFAULT_WARNING_THRESHOLDS,
    build_fitted_curve_table,
    summarize_fit_quality,
)
from trustk.plotting.style import export_figure, journal_width, set_trustk_style
from trustk.priors.transformation_uncertainty import estimate_transformation_uncertainty


def run_fit_quality_qc(
    curves_path: str | Path = "data/processed/synthetic_population_curves.csv",
    residuals_path: str | Path = "data/processed/synthetic_population_residuals.csv",
    settings_path: str | Path = "data/processed/solver_simulation_settings.csv",
    fitted_curves_path: str | Path = "data/processed/synthetic_fit_quality_fitted_curves.csv",
    qc_path: str | Path = "data/processed/synthetic_fit_quality_qc.csv",
    prior_path: str | Path = "data/processed/transformation_uncertainty_prior.csv",
    report_path: str | Path = "outputs/reports/fit_quality_qc.json",
    qc_figure_prefix: str | Path = "outputs/figures/fig10_fit_quality_qc",
    prior_figure_prefix: str | Path = "outputs/figures/fig11_transformation_uncertainty_prior",
    *,
    min_prior_cases: int = 5,
) -> dict:
    """Build QC diagnostics, figures, and a first method-level uncertainty prior."""

    curves_path = Path(curves_path)
    residuals_path = Path(residuals_path)
    settings_path = Path(settings_path)
    fitted_curves_path = Path(fitted_curves_path)
    qc_path = Path(qc_path)
    prior_path = Path(prior_path)
    report_path = Path(report_path)
    qc_figure_prefix = Path(qc_figure_prefix)
    prior_figure_prefix = Path(prior_figure_prefix)

    curves = pd.read_csv(curves_path)
    residuals = pd.read_csv(residuals_path)
    settings = pd.read_csv(settings_path)
    fitted = build_fitted_curve_table(curves, residuals, settings)
    qc = summarize_fit_quality(fitted, residuals)
    prior = estimate_transformation_uncertainty(qc, min_cases=min_prior_cases)
    report = _summarize_qc(qc, prior)
    report["purpose"] = "fit-quality screen before estimating method-specific transformation uncertainty"
    report["pass_thresholds"] = DEFAULT_PASS_THRESHOLDS
    report["warning_thresholds"] = DEFAULT_WARNING_THRESHOLDS
    report["accepted_prior_qc_classes"] = ["pass"]

    fitted_curves_path.parent.mkdir(parents=True, exist_ok=True)
    fitted.to_csv(fitted_curves_path, index=False)
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    qc.to_csv(qc_path, index=False)
    prior_path.parent.mkdir(parents=True, exist_ok=True)
    prior.to_csv(prior_path, index=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    qc_figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    _plot_fit_quality_qc(fitted, qc, qc_figure_prefix)
    qc_figure_prefix.with_suffix(".csv").write_text(qc.to_csv(index=False), encoding="utf-8")
    qc_figure_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "figure": qc_figure_prefix.name,
                "palette": "official cmcrameri cmc.batlow and cmc.vik",
                "purpose": "curve-level fit quality screen before residual-prior estimation",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    prior_figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    _plot_transformation_uncertainty(qc, prior, prior_figure_prefix)
    prior_figure_prefix.with_suffix(".csv").write_text(prior.to_csv(index=False), encoding="utf-8")
    prior_figure_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "figure": prior_figure_prefix.name,
                "palette": "official cmcrameri cmc.batlow and cmc.vik",
                "purpose": "QC-screened method-level transformation uncertainty",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report


def _summarize_qc(qc: pd.DataFrame, prior: pd.DataFrame) -> dict:
    finite = bool(np.isfinite(qc[["rmse_log_response", "log_residual"]].to_numpy(dtype=float)).all())
    counts = (
        qc.groupby(["method", "qc_class"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["method", "qc_class"])
        .to_dict(orient="records")
    )
    method_rows = {}
    for method, group in qc.groupby("method"):
        method_rows[method] = {
            "n": int(len(group)),
            "pass": int(group["qc_class"].eq("pass").sum()),
            "warning": int(group["qc_class"].eq("warning").sum()),
            "fail": int(group["qc_class"].eq("fail").sum()),
            "rmse_median": float(group["rmse_log_response"].median()),
            "rmse_q95": float(group["rmse_log_response"].quantile(0.95)),
        }
    prior_rows = prior.set_index("method").to_dict(orient="index") if not prior.empty else {}
    return {
        "fit_quality_rows": int(len(qc)),
        "checks": {
            "finite_fit_quality": {"pass": finite},
            "prior_methods_present": {"pass": set(prior["method"]) == {"pumping", "slug"}},
        },
        "qc_counts": counts,
        "method_summary": method_rows,
        "transformation_uncertainty_prior": prior_rows,
    }


def _plot_fit_quality_qc(fitted: pd.DataFrame, qc: pd.DataFrame, figure_prefix: Path) -> None:
    set_trustk_style()
    fig, axes = plt.subplots(1, 2, figsize=(journal_width(170), 3.25))
    many_rows = len(qc) > 500
    rng = np.random.default_rng(20260517)

    ax = axes[0]
    positions = {"pumping": 0.85, "slug": 1.15}
    method_colors = {"pumping": cmc.vik(0.25), "slug": cmc.vik(0.75)}
    data = [qc[qc["method"].eq("pumping")]["rmse_log_response"], qc[qc["method"].eq("slug")]["rmse_log_response"]]
    parts = ax.violinplot(data, positions=[positions["pumping"], positions["slug"]], widths=0.22, showmeans=True, showextrema=False)
    for body, color in zip(parts["bodies"], [method_colors["pumping"], method_colors["slug"]]):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.55)
    parts["cmeans"].set_color("0.15")
    for method in ["pumping", "slug"]:
        values = qc[qc["method"].eq(method)]["rmse_log_response"].to_numpy(dtype=float)
        jitter = rng.uniform(-0.050, 0.050, len(values))
        ax.scatter(
            np.full(len(values), positions[method]) + jitter,
            values,
            s=5.5 if many_rows else 10,
            color=method_colors[method],
            edgecolor="none",
            linewidth=0.0,
            alpha=0.32 if many_rows else 0.72,
        )
        ax.hlines(DEFAULT_PASS_THRESHOLDS[method], positions[method] - 0.13, positions[method] + 0.13, colors="0.2", linestyles="-", lw=0.8)
        ax.hlines(DEFAULT_WARNING_THRESHOLDS[method], positions[method] - 0.13, positions[method] + 0.13, colors="0.2", linestyles="--", lw=0.8)
    ax.set_xticks([positions["pumping"], positions["slug"]])
    ax.set_xticklabels(["Pumping", "Slug"])
    ax.set_yscale("log")
    ax.set_ylabel(r"Fit RMSE in log response")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.11, 1.07, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5}, clip_on=False)

    ax = axes[1]
    markers = {"pass": "o", "warning": "s", "fail": "^"}
    for method in ["pumping", "slug"]:
        for qc_class, marker in markers.items():
            subset = qc[qc["method"].eq(method) & qc["qc_class"].eq(qc_class)]
            if subset.empty:
                continue
            ax.scatter(
                subset["rmse_log_response"],
                subset["abs_log_residual"],
                s=10 if many_rows else 24,
                color=method_colors[method],
                marker=marker,
                edgecolor="none",
                linewidth=0.0,
                alpha=0.36 if many_rows else 0.78,
                label=f"{method[0].upper()} {qc_class}",
            )
    ax.set_xscale("log")
    ax.set_xlabel(r"Fit RMSE in log response")
    ax.set_ylabel(r"Residual magnitude $|r_m|$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.11, 1.07, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5}, clip_on=False)
    ax.legend(loc="upper right", frameon=False, ncol=1, handletextpad=0.25, borderaxespad=0.2, fontsize=6)

    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.20, top=0.90, wspace=0.34)
    export_figure(fig, figure_prefix)
    plt.close(fig)


def _plot_transformation_uncertainty(qc: pd.DataFrame, prior: pd.DataFrame, figure_prefix: Path) -> None:
    set_trustk_style()
    fig, axes = plt.subplots(1, 2, figsize=(journal_width(170), 3.2))
    accepted = qc[qc["qc_class"].eq("pass")].copy()
    positions = {"pumping": 0.85, "slug": 1.15}
    colors = {"pumping": cmc.vik(0.25), "slug": cmc.vik(0.75)}

    ax = axes[0]
    for method in ["pumping", "slug"]:
        method_values = accepted[accepted["method"].eq(method)]["log_residual"].to_numpy(dtype=float)
        if len(method_values) >= 2:
            parts = ax.violinplot(
                [method_values],
                positions=[positions[method]],
                widths=0.22,
                showmeans=True,
                showextrema=False,
            )
            parts["bodies"][0].set_facecolor(colors[method])
            parts["bodies"][0].set_edgecolor("0.25")
            parts["bodies"][0].set_alpha(0.72)
            parts["cmeans"].set_color("0.15")
        if len(method_values) > 0:
            jitter = np.linspace(-0.045, 0.045, len(method_values))
            ax.scatter(
                np.full(len(method_values), positions[method]) + jitter,
                method_values,
                s=12,
                color=colors[method],
                edgecolor="0.25",
                linewidth=0.2,
                alpha=0.75,
            )
    ax.axhline(0.0, color="0.25", lw=0.8, ls="--")
    ax.set_xticks([positions["pumping"], positions["slug"]])
    ax.set_xticklabels(["Pumping", "Slug"])
    ax.set_ylabel(r"QC-pass log residual $r_m$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.03, 0.96, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")

    ax = axes[1]
    for method in ["pumping", "slug"]:
        method_prior = prior[prior["method"].eq(method)]
        if method_prior.empty:
            continue
        row = method_prior.iloc[0]
        x = positions[method]
        y = row["bias_factor_c"]
        yerr = np.array([[y - row["q05_factor"]], [row["q95_factor"] - y]])
        ax.errorbar(x, y, yerr=yerr, fmt="o", color=colors[method], mec="0.25", ms=5, lw=1.0, capsize=3)
        ax.text(x, y * 1.08, f"n={int(row['n_cases'])}", ha="center", va="bottom", fontsize=7)
    ax.axhline(1.0, color="0.25", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xticks([positions["pumping"], positions["slug"]])
    ax.set_xticklabels(["Pumping", "Slug"])
    ax.set_ylabel(r"Bias factor $c_m=\exp(E[r_m])$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.03, 0.96, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")

    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.22, top=0.94, wspace=0.34)
    export_figure(fig, figure_prefix)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", default="data/processed/synthetic_population_curves.csv")
    parser.add_argument("--residuals", default="data/processed/synthetic_population_residuals.csv")
    parser.add_argument("--settings", default="data/processed/solver_simulation_settings.csv")
    parser.add_argument("--fitted-curves", default="data/processed/synthetic_fit_quality_fitted_curves.csv")
    parser.add_argument("--qc", default="data/processed/synthetic_fit_quality_qc.csv")
    parser.add_argument("--prior", default="data/processed/transformation_uncertainty_prior.csv")
    parser.add_argument("--report", default="outputs/reports/fit_quality_qc.json")
    parser.add_argument("--qc-figure-prefix", default="outputs/figures/fig10_fit_quality_qc")
    parser.add_argument("--prior-figure-prefix", default="outputs/figures/fig11_transformation_uncertainty_prior")
    args = parser.parse_args(argv)
    report = run_fit_quality_qc(
        curves_path=args.curves,
        residuals_path=args.residuals,
        settings_path=args.settings,
        fitted_curves_path=args.fitted_curves,
        qc_path=args.qc,
        prior_path=args.prior,
        report_path=args.report,
        qc_figure_prefix=args.qc_figure_prefix,
        prior_figure_prefix=args.prior_figure_prefix,
    )
    all_pass = all(item["pass"] for item in report["checks"].values())
    print(f"fit_quality_rows={report['fit_quality_rows']}")
    print(f"fit_quality_qc_pass={all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

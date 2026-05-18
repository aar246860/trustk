"""Run TRUST-K conditional prior, hold-out, and baseline-comparison analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trustk.plotting.style import export_figure, journal_width, set_trustk_style
from trustk.priors.conditional import (
    build_prior_dataset,
    evaluate_baselines,
    evaluate_holdout,
    fit_conditional_prior,
    make_response_surface,
    predict_conditional_prior,
    split_train_validation,
)


def run_conditional_prior_analysis(
    residuals_path: str | Path = "data/processed/formal_population_1728_engineering_residuals.csv",
    registry_path: str | Path = "data/processed/formal_dimensionless_case_registry_2000.csv",
    qc_path: str | Path = "data/processed/formal_population_1728_engineering_qc.csv",
    predictions_path: str | Path = "data/processed/conditional_prior_predictions.csv",
    holdout_metrics_path: str | Path = "data/processed/conditional_prior_holdout_metrics.csv",
    baseline_metrics_path: str | Path = "data/processed/conditional_prior_baseline_metrics.csv",
    report_path: str | Path = "outputs/reports/conditional_prior_analysis.json",
    surfaces_figure_prefix: str | Path = "outputs/figures/fig06_conditional_prior_surfaces",
    holdout_figure_prefix: str | Path = "outputs/figures/fig07_holdout_validation",
    baseline_figure_prefix: str | Path = "outputs/figures/fig08_assimilation_baseline_comparison",
    *,
    validation_fraction: float = 0.25,
    seed: int = 20260517,
) -> dict:
    """Run the three-stage TRUST-K analysis after engineering residuals are available."""

    residuals = pd.read_csv(residuals_path)
    registry = pd.read_csv(registry_path)
    qc = pd.read_csv(qc_path)
    data = build_prior_dataset(residuals, registry, qc)
    train, validation = split_train_validation(data, validation_fraction=validation_fraction, seed=seed)
    model = fit_conditional_prior(train)
    predictions = predict_conditional_prior(model, validation)
    holdout = evaluate_holdout(model, validation)
    baselines = evaluate_baselines(model, train, validation)
    report = _summarize(data, train, validation, holdout, baselines)
    report["purpose"] = "conditional TRUST-K prior surfaces, hold-out calibration, and support-target baseline comparison"
    report["inputs"] = {
        "residuals": str(residuals_path),
        "registry": str(registry_path),
        "qc": str(qc_path),
    }
    report["validation_fraction"] = validation_fraction
    report["seed"] = seed

    _write_table(predictions_path, predictions)
    _write_table(holdout_metrics_path, holdout)
    _write_table(baseline_metrics_path, baselines)
    _write_json(report_path, report)

    surfaces = pd.concat(
        [
            make_response_surface(model, data, method="pumping"),
            make_response_surface(model, data, method="slug"),
        ],
        ignore_index=True,
    )
    _plot_surfaces(surfaces, Path(surfaces_figure_prefix))
    _write_table(Path(surfaces_figure_prefix).with_suffix(".csv"), surfaces)
    _write_json(
        Path(surfaces_figure_prefix).with_suffix(".json"),
        {"figure": Path(surfaces_figure_prefix).name, "palette": "official cmcrameri cmc.vik and cmc.lipari"},
    )

    _plot_holdout(predictions, holdout, Path(holdout_figure_prefix))
    _write_table(Path(holdout_figure_prefix).with_suffix(".csv"), predictions)
    _write_json(
        Path(holdout_figure_prefix).with_suffix(".json"),
        {"figure": Path(holdout_figure_prefix).name, "palette": "official cmcrameri cmc.batlow and cmc.vik"},
    )

    _plot_baselines(baselines, Path(baseline_figure_prefix))
    _write_table(Path(baseline_figure_prefix).with_suffix(".csv"), baselines)
    _write_json(
        Path(baseline_figure_prefix).with_suffix(".json"),
        {"figure": Path(baseline_figure_prefix).name, "palette": "official cmcrameri cmc.batlow and cmc.vik"},
    )
    return report


def _summarize(
    data: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    holdout: pd.DataFrame,
    baselines: pd.DataFrame,
) -> dict:
    conditional = baselines[baselines["approach"].eq("conditional")]
    constant = baselines[baselines["approach"].eq("method_constant")]
    conditional_rmse = float(conditional["rmse_log"].mean())
    constant_rmse = float(constant["rmse_log"].mean())
    min_coverage = float(conditional["coverage_95"].min())
    return {
        "n_rows": int(len(data)),
        "n_train_rows": int(len(train)),
        "n_validation_rows": int(len(validation)),
        "method_counts": data.groupby("method").size().astype(int).to_dict(),
        "holdout": holdout.to_dict(orient="records"),
        "baseline_metrics": baselines.to_dict(orient="records"),
        "checks": {
            "conditional_beats_method_constant_rmse": {"pass": conditional_rmse < constant_rmse},
            "conditional_95_coverage_at_least_80": {"pass": min_coverage >= 0.80},
            "both_methods_present": {"pass": set(data["method"]) == {"pumping", "slug"}},
        },
    }


def _plot_surfaces(surface: pd.DataFrame, figure_prefix: Path) -> None:
    set_trustk_style()
    fig, axes = plt.subplots(2, 2, figsize=(journal_width(170), 5.8), sharex=True, sharey=True)
    panels = [
        ("pumping", "predicted_log_residual_mean", axes[0, 0], cmc.vik, r"Pumping $\ln c_P(\Pi)$"),
        ("slug", "predicted_log_residual_mean", axes[0, 1], cmc.vik, r"Slug $\ln c_S(\Pi)$"),
        ("pumping", "predicted_log_residual_sigma", axes[1, 0], cmc.lipari, r"Pumping $\sigma_P(\Pi)$"),
        ("slug", "predicted_log_residual_sigma", axes[1, 1], cmc.lipari, r"Slug $\sigma_S(\Pi)$"),
    ]
    labels = ["(a)", "(b)", "(c)", "(d)"]
    for label, (method, value, ax, cmap, title) in zip(labels, panels):
        data = surface[surface["method"].eq(method)]
        pivot = data.pivot_table(index="log_lambda1_over_RI", columns="sigma_Y2", values=value)
        x = pivot.columns.to_numpy(dtype=float)
        y = pivot.index.to_numpy(dtype=float)
        z = pivot.to_numpy(dtype=float)
        mesh = ax.pcolormesh(x, y, z, shading="auto", cmap=cmap)
        ax.contour(x, y, z, colors="0.2", linewidths=0.35, alpha=0.55)
        ax.set_title(title)
        ax.text(0.03, 0.96, label, transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.tick_params(labelsize=7)
    axes[1, 0].set_xlabel(r"$\sigma_Y^2$")
    axes[1, 1].set_xlabel(r"$\sigma_Y^2$")
    axes[0, 0].set_ylabel(r"$\ln(\lambda_1/R_I)$")
    axes[1, 0].set_ylabel(r"$\ln(\lambda_1/R_I)$")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.93, wspace=0.30, hspace=0.36)
    export_figure(fig, figure_prefix)
    plt.close(fig)


def _plot_holdout(predictions: pd.DataFrame, holdout: pd.DataFrame, figure_prefix: Path) -> None:
    set_trustk_style()
    fig, axes = plt.subplots(1, 3, figsize=(journal_width(170), 3.35))
    colors = {"pumping": cmc.vik(0.25), "slug": cmc.vik(0.75)}
    linestyles = {"pumping": "-", "slug": "--"}

    ax = axes[0]
    for method, group in predictions.groupby("method"):
        ax.scatter(
            group["log_residual"],
            group["predicted_log_residual_mean"],
            s=16,
            color=colors[method],
            edgecolor="0.25",
            linewidth=0.2,
            alpha=0.75,
            label=method.capitalize(),
        )
    low = float(np.nanmin(predictions[["log_residual", "predicted_log_residual_mean"]].to_numpy()))
    high = float(np.nanmax(predictions[["log_residual", "predicted_log_residual_mean"]].to_numpy()))
    ax.plot([low, high], [low, high], color="0.25", lw=0.8, ls="--")
    ax.set_xlabel(r"Observed residual $r_m$")
    ax.set_ylabel(r"Predicted mean $\hat r_m$")
    ax.legend(frameon=False, loc="lower right")
    ax.text(-0.13, 1.08, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5}, clip_on=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.axvspan(-1.96, 1.96, color="0.92", zorder=0, label="nominal 95%")
    for method, group in predictions.groupby("method"):
        standardized = group["standardized_residual_error"].to_numpy(dtype=float)
        sorted_values = np.sort(standardized[np.isfinite(standardized)])
        probabilities = (np.arange(1, len(sorted_values) + 1) - 0.5) / len(sorted_values)
        ax.plot(sorted_values, probabilities, color=colors[method], lw=1.15, ls=linestyles[method], label=method.capitalize())
    ax.axvline(-1.96, color="0.35", lw=0.8, ls="--")
    ax.axvline(1.96, color="0.35", lw=0.8, ls="--")
    ax.set_xlabel("Standardized hold-out error")
    ax.set_ylabel("Empirical probability")
    ax.legend(frameon=False, loc="lower right", handlelength=1.8)
    ax.text(-0.13, 1.08, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5}, clip_on=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    x = np.arange(len(holdout))
    coverage_delta = holdout["coverage_95"].to_numpy(dtype=float) - 0.95
    ax.bar(x, coverage_delta, color=[colors[m] for m in holdout["method"]], edgecolor="0.25", linewidth=0.35)
    ax.axhline(0.0, color="0.25", lw=0.8, ls="--")
    pad = max(0.01, float(np.nanmax(np.abs(coverage_delta))) * 0.35)
    ax.set_ylim(float(np.nanmin(coverage_delta)) - pad, float(np.nanmax(coverage_delta)) + pad)
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in holdout["method"]], rotation=0)
    ax.set_ylabel("Coverage minus 0.95")
    for xi, coverage, delta in zip(x, holdout["coverage_95"], coverage_delta):
        va = "bottom" if delta >= 0 else "top"
        y = delta + (0.002 if delta >= 0 else -0.002)
        ax.text(xi, y, f"{coverage:.3f}", ha="center", va=va, fontsize=7)
    ax.text(-0.13, 1.08, "(c)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5}, clip_on=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.22, top=0.94, wspace=0.42)
    export_figure(fig, figure_prefix)
    plt.close(fig)


def _plot_baselines(baselines: pd.DataFrame, figure_prefix: Path) -> None:
    set_trustk_style()
    fig, axes = plt.subplots(1, 3, figsize=(journal_width(170), 3.35))
    order = ["hard", "method_constant", "conditional"]
    labels = ["Hard", "Method c", "TRUST-K"]
    colors = [cmc.batlow(0.25), cmc.batlow(0.55), cmc.batlow(0.82)]

    for ax, method, panel in zip(axes[:2], ["pumping", "slug"], ["(a)", "(b)"]):
        data = baselines[baselines["method"].eq(method)].set_index("approach").loc[order]
        ax.bar(np.arange(len(order)), data["rmse_log"], color=colors, edgecolor="0.25", linewidth=0.35)
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel(r"RMSE in $\ln K^*$")
        ax.set_title(method.capitalize())
        ax.text(0.03, 0.96, panel, transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    conditional = baselines[baselines["approach"].eq("conditional")]
    method_constant = baselines[baselines["approach"].eq("method_constant")]
    x = np.arange(len(conditional))
    width = 0.34
    ax.bar(x - width / 2, method_constant["coverage_95"], width, color=cmc.vik(0.35), edgecolor="0.25", linewidth=0.35, label="Method c")
    ax.bar(x + width / 2, conditional["coverage_95"], width, color=cmc.vik(0.75), edgecolor="0.25", linewidth=0.35, label="TRUST-K")
    ax.axhline(0.95, color="0.25", lw=0.8, ls="--")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in conditional["method"]])
    ax.set_ylabel("95% coverage")
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.03, 0.96, "(c)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.25, top=0.88, wspace=0.45)
    export_figure(fig, figure_prefix)
    plt.close(fig)


def _write_table(path: str | Path, table: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def _write_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residuals", default="data/processed/formal_population_1728_engineering_residuals.csv")
    parser.add_argument("--registry", default="data/processed/formal_dimensionless_case_registry_2000.csv")
    parser.add_argument("--qc", default="data/processed/formal_population_1728_engineering_qc.csv")
    parser.add_argument("--predictions", default="data/processed/conditional_prior_predictions.csv")
    parser.add_argument("--holdout-metrics", default="data/processed/conditional_prior_holdout_metrics.csv")
    parser.add_argument("--baseline-metrics", default="data/processed/conditional_prior_baseline_metrics.csv")
    parser.add_argument("--report", default="outputs/reports/conditional_prior_analysis.json")
    parser.add_argument("--surfaces-figure-prefix", default="outputs/figures/fig06_conditional_prior_surfaces")
    parser.add_argument("--holdout-figure-prefix", default="outputs/figures/fig07_holdout_validation")
    parser.add_argument("--baseline-figure-prefix", default="outputs/figures/fig08_assimilation_baseline_comparison")
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260517)
    args = parser.parse_args(argv)
    report = run_conditional_prior_analysis(
        residuals_path=args.residuals,
        registry_path=args.registry,
        qc_path=args.qc,
        predictions_path=args.predictions,
        holdout_metrics_path=args.holdout_metrics,
        baseline_metrics_path=args.baseline_metrics,
        report_path=args.report,
        surfaces_figure_prefix=args.surfaces_figure_prefix,
        holdout_figure_prefix=args.holdout_figure_prefix,
        baseline_figure_prefix=args.baseline_figure_prefix,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    all_pass = all(item["pass"] for item in report["checks"].values())
    print(f"n_rows={report['n_rows']}")
    print(f"conditional_prior_analysis_pass={all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

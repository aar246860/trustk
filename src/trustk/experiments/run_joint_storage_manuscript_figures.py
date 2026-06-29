from __future__ import annotations

import argparse
import json
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trustk.experiments.run_joint_storage_reanalysis import _plot_joint_storage
from trustk.plotting.style import export_figure, journal_width, set_trustk_style


def run_joint_storage_manuscript_figures(
    summary_path: str | Path = "data/processed/formal_joint_storage_residuals.csv",
    qc_path: str | Path = "data/processed/formal_joint_storage_fit_quality_qc.csv",
    predictions_path: str | Path = "data/processed/formal_joint_storage_conditional_predictions.csv",
    *,
    support_prefix: str | Path = "outputs/figures/fig03_support_residuals",
    qc_prefix: str | Path = "outputs/figures/fig04_fit_quality_qc",
    conditional_prefix: str | Path = "outputs/figures/fig05_conditional_prior_surfaces",
    validation_prefix: str | Path = "outputs/figures/fig06_withheld_validation",
) -> dict:
    summary = pd.read_csv(summary_path)
    qc = pd.read_csv(qc_path)
    predictions = pd.read_csv(predictions_path)

    _plot_joint_storage(summary, Path(support_prefix))
    Path(support_prefix).with_suffix(".csv").write_text(summary.to_csv(index=False), encoding="utf-8")
    _write_metadata(Path(support_prefix), "formal joint K-Ss-b support-scale and residual summary")

    _plot_formal_qc(qc, Path(qc_prefix))
    Path(qc_prefix).with_suffix(".csv").write_text(qc.to_csv(index=False), encoding="utf-8")
    _write_metadata(Path(qc_prefix), "formal joint K-Ss-b response-fit classification")

    _plot_conditional_projection(predictions, Path(conditional_prefix))
    _write_metadata(Path(conditional_prefix), "storage-aware conditional transformation-prior projections")

    _plot_holdout_validation(predictions, Path(validation_prefix))
    _write_metadata(Path(validation_prefix), "formal joint K-Ss-b withheld residual validation")

    return {
        "summary_rows": int(len(summary)),
        "qc_rows": int(len(qc)),
        "prediction_rows": int(len(predictions)),
        "figures": [
            str(Path(support_prefix).with_suffix(".pdf")),
            str(Path(qc_prefix).with_suffix(".pdf")),
            str(Path(conditional_prefix).with_suffix(".pdf")),
            str(Path(validation_prefix).with_suffix(".pdf")),
        ],
    }


def _plot_formal_qc(qc: pd.DataFrame, figure_prefix: Path) -> None:
    set_trustk_style()
    fig, axes = plt.subplots(1, 2, figsize=(journal_width(170), 3.35))
    methods = ["pumping", "slug_bouwer_rice"]
    colors = {"pumping": cmc.batlow(0.58), "slug_bouwer_rice": cmc.vik(0.76)}
    labels = {"pumping": "Cooper-Jacob", "slug_bouwer_rice": "Bouwer-Rice slug"}

    grouped = [qc.loc[qc["method"].eq(method), "rmse_log_response"].to_numpy(dtype=float) for method in methods]
    parts = axes[0].violinplot(grouped, positions=np.arange(len(methods)) + 1, widths=0.55, showmeans=True, showextrema=False)
    for body, method in zip(parts["bodies"], methods):
        body.set_facecolor(colors[method])
        body.set_edgecolor("0.25")
        body.set_alpha(0.72)
    parts["cmeans"].set_color("0.15")
    axes[0].axhline(0.75, color="0.25", lw=0.8, ls="--")
    axes[0].set_xticks(np.arange(len(methods)) + 1)
    axes[0].set_xticklabels([labels[method] for method in methods], rotation=12, ha="right")
    axes[0].set_ylabel("Log-response fit RMSE")
    axes[0].text(0.03, 0.95, "(a)", transform=axes[0].transAxes, ha="left", va="top", fontsize=9, fontweight="bold")

    for method in methods:
        subset = qc[qc["method"].eq(method)]
        axes[1].scatter(
            subset["rmse_log_response"],
            subset["abs_log_residual"],
            s=10,
            color=colors[method],
            alpha=0.52,
            edgecolor="none",
            label=labels[method],
        )
    axes[1].axvline(0.75, color="0.25", lw=0.8, ls="--")
    axes[1].set_xlabel("Log-response fit RMSE")
    axes[1].set_ylabel("Absolute log residual")
    axes[1].legend(frameon=False, fontsize=7, loc="upper left")
    axes[1].text(0.03, 0.95, "(b)", transform=axes[1].transAxes, ha="left", va="top", fontsize=9, fontweight="bold")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.2, top=0.94, wspace=0.34)
    export_figure(fig, figure_prefix)
    plt.close(fig)


def _plot_conditional_projection(predictions: pd.DataFrame, figure_prefix: Path) -> None:
    set_trustk_style()
    fig, axes = plt.subplots(2, 2, figsize=(journal_width(170), 5.3), sharex=True, sharey=True)
    methods = [("pumping", "Cooper-Jacob pumping"), ("slug_bouwer_rice", "Bouwer-Rice slug")]
    panels = [
        ("predicted_log_residual_mean", r"Predicted mean residual"),
        ("predicted_log_residual_sigma", r"Predicted residual SD"),
    ]
    for row_idx, (value_col, color_label) in enumerate(panels):
        for col_idx, (method, method_label) in enumerate(methods):
            ax = axes[row_idx, col_idx]
            subset = predictions[predictions["method"].eq(method)]
            scatter = ax.scatter(
                subset["log_Ss_star_m_inv"],
                subset["sigma_Y2"],
                c=subset[value_col],
                s=10,
                cmap=cmc.vik if value_col.endswith("mean") else cmc.batlow,
                edgecolor="none",
                alpha=0.72,
            )
            cbar = fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.025)
            cbar.ax.set_title(color_label, fontsize=7, pad=3)
            ax.set_title(method_label if row_idx == 0 else "", fontsize=9)
            ax.text(
                0.03,
                0.95,
                f"({chr(97 + row_idx * 2 + col_idx)})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                fontweight="bold",
            )
            ax.spines[["top", "right"]].set_visible(False)
            if row_idx == 1:
                ax.set_xlabel(r"$\ln S_s^*$")
            if col_idx == 0:
                ax.set_ylabel(r"$\sigma_Y^2$")
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.12, top=0.92, wspace=0.28, hspace=0.28)
    export_figure(fig, figure_prefix)
    plt.close(fig)


def _plot_holdout_validation(predictions: pd.DataFrame, figure_prefix: Path) -> None:
    set_trustk_style()
    fig, axes = plt.subplots(1, 3, figsize=(journal_width(170), 3.35))
    colors = {"pumping": cmc.batlow(0.58), "slug_bouwer_rice": cmc.vik(0.76)}
    labels = {"pumping": "Cooper-Jacob", "slug_bouwer_rice": "Bouwer-Rice slug"}

    for method, group in predictions.groupby("method"):
        axes[0].scatter(
            group["predicted_log_residual_mean"],
            group["log_residual"],
            s=10,
            color=colors.get(method, "0.45"),
            alpha=0.52,
            edgecolor="none",
            label=labels.get(method, method),
        )
    limits = np.r_[
        predictions["predicted_log_residual_mean"].to_numpy(dtype=float),
        predictions["log_residual"].to_numpy(dtype=float),
    ]
    lo = float(np.nanpercentile(limits, 1))
    hi = float(np.nanpercentile(limits, 99))
    axes[0].plot([lo, hi], [lo, hi], color="0.25", lw=0.8, ls="--")
    axes[0].set_xlim(lo, hi)
    axes[0].set_ylim(lo, hi)
    axes[0].set_xlabel("Predicted residual mean")
    axes[0].set_ylabel("Observed residual")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    axes[0].text(0.03, 0.95, "(a)", transform=axes[0].transAxes, ha="left", va="top", fontsize=9, fontweight="bold")

    bins = np.linspace(-4.0, 4.0, 37)
    for method, group in predictions.groupby("method"):
        axes[1].hist(
            group["standardized_residual_error"].to_numpy(dtype=float),
            bins=bins,
            histtype="step",
            lw=1.2,
            color=colors.get(method, "0.45"),
            label=labels.get(method, method),
        )
    axes[1].axvspan(-1.96, 1.96, color="0.85", alpha=0.45, zorder=0)
    axes[1].set_xlabel("Standardized residual error")
    axes[1].set_ylabel("Count")
    axes[1].text(0.03, 0.95, "(b)", transform=axes[1].transAxes, ha="left", va="top", fontsize=9, fontweight="bold")

    coverage_rows = []
    for method, group in predictions.groupby("method"):
        errors = group["log_residual"].to_numpy(dtype=float) - group["predicted_log_residual_mean"].to_numpy(dtype=float)
        sigma = group["predicted_log_residual_sigma"].to_numpy(dtype=float)
        coverage_rows.append((method, float(np.mean(np.abs(errors) <= 1.96 * sigma))))
    x = np.arange(len(coverage_rows))
    axes[2].bar(x, [row[1] for row in coverage_rows], color=[colors.get(row[0], "0.45") for row in coverage_rows])
    axes[2].axhline(0.95, color="0.25", lw=0.8, ls="--")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([labels.get(row[0], row[0]) for row in coverage_rows], rotation=12, ha="right")
    axes[2].set_ylabel("95% coverage")
    axes[2].text(0.03, 0.95, "(c)", transform=axes[2].transAxes, ha="left", va="top", fontsize=9, fontweight="bold")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.2, top=0.94, wspace=0.42)
    export_figure(fig, figure_prefix)
    plt.close(fig)


def _write_metadata(prefix: Path, purpose: str) -> None:
    prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "figure": prefix.name,
                "purpose": purpose,
                "palette": "official cmcrameri cmc.batlow, cmc.vik, and cmc.lipari",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="data/processed/formal_joint_storage_residuals.csv")
    parser.add_argument("--qc", default="data/processed/formal_joint_storage_fit_quality_qc.csv")
    parser.add_argument("--predictions", default="data/processed/formal_joint_storage_conditional_predictions.csv")
    args = parser.parse_args(argv)
    report = run_joint_storage_manuscript_figures(
        summary_path=args.summary,
        qc_path=args.qc,
        predictions_path=args.predictions,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

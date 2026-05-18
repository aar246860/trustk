"""Upgrade Figure 8 to spatial latent-field assimilation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from trustk.assimilation.spatial import (
    assimilate_linear_gaussian,
    build_support_operator,
    evaluate_spatial_posterior,
    squared_exponential_covariance,
)
from trustk.plotting.style import export_figure, journal_width, set_trustk_style
from trustk.priors.conditional import (
    build_prior_dataset,
    fit_conditional_prior,
    predict_conditional_prior,
)
from trustk.random_fields.gaussian_field import generate_gaussian_logk_field


def run_spatial_assimilation(
    conditional_predictions_path: str | Path = "data/processed/conditional_prior_predictions.csv",
    residuals_path: str | Path = "data/processed/formal_population_1728_engineering_residuals.csv",
    registry_path: str | Path = "data/processed/formal_dimensionless_case_registry_2000.csv",
    qc_path: str | Path = "data/processed/formal_population_1728_engineering_qc.csv",
    metrics_path: str | Path = "data/processed/spatial_assimilation_metrics.csv",
    fields_path: str | Path = "data/processed/spatial_assimilation_fields.csv",
    observations_path: str | Path = "data/processed/spatial_assimilation_observations.csv",
    report_path: str | Path = "outputs/reports/spatial_assimilation.json",
    figure_prefix: str | Path = "outputs/figures/fig08_spatial_assimilation",
    *,
    grid_n: int = 29,
    n_observations: int = 30,
    seed: int = 20260518,
) -> dict:
    """Generate a controlled spatial field and compare assimilation approaches."""

    residuals = pd.read_csv(residuals_path)
    registry = pd.read_csv(registry_path)
    qc = pd.read_csv(qc_path)
    prior_data = build_prior_dataset(residuals, registry, qc)
    model = fit_conditional_prior(prior_data)
    observations = _make_spatial_observations(
        prior_data=prior_data,
        model=model,
        grid_n=grid_n,
        n_observations=n_observations,
        seed=seed,
    )
    coords = observations.attrs["coords"]
    truth = observations.attrs["truth"]
    prior_mean = np.full(len(truth), float(np.mean(truth)))
    prior_cov = squared_exponential_covariance(coords, variance=0.65**2, corr_len=30.0, nugget=1.0e-6)

    approaches = [
        (
            "Hard interpretation",
            None,
            None,
            0.08,
        ),
        (
            "Method-constant soft data",
            "constant_correction",
            "constant_sigma",
            0.10,
        ),
        (
            "TRUST-K conditional",
            "conditional_correction",
            "conditional_sigma",
            0.10,
        ),
    ]
    posteriors = []
    metrics = []
    for name, correction_col, sigma_col, default_sigma in approaches:
        posterior = assimilate_linear_gaussian(
            coords=coords,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            observations=observations,
            correction_column=correction_col,
            sigma_column=sigma_col,
            default_sigma=default_sigma,
            approach=name,
        )
        posteriors.append(posterior)
        metrics.append(evaluate_spatial_posterior(posterior, truth))
    metrics_table = pd.DataFrame(metrics)
    fields = _field_table(coords, truth, posteriors)
    report = _summarize(metrics_table, observations)

    _write_table(metrics_path, metrics_table)
    _write_table(fields_path, fields)
    obs_to_save = observations.drop(columns=["actual_log_K_star"], errors="ignore").copy()
    _write_table(observations_path, obs_to_save)
    _write_json(report_path, report)
    _plot_spatial_assimilation(fields, observations, metrics_table, Path(figure_prefix), grid_n=grid_n)
    _write_table(Path(figure_prefix).with_suffix(".csv"), metrics_table)
    _write_json(
        Path(figure_prefix).with_suffix(".json"),
        {"figure": Path(figure_prefix).name, "palette": "official cmcrameri cmc.batlow, cmc.vik, cmc.lipari"},
    )
    return report


def _make_spatial_observations(
    *,
    prior_data: pd.DataFrame,
    model,
    grid_n: int,
    n_observations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    extent = 120.0
    dx = 2.0 * extent / (grid_n - 1)
    field = generate_gaussian_logk_field(
        nx=grid_n,
        ny=grid_n,
        dx=dx,
        dy=dx,
        mean_logk=-11.0,
        sigma_logk=0.65,
        corr_len_x=45.0,
        corr_len_y=22.0,
        orientation_rad=0.65,
        seed=seed,
    )
    xx, yy = np.meshgrid(field.x, field.y, indexing="xy")
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    truth = field.logk.ravel()

    half = n_observations // 2
    methods = np.array(["slug"] * half + ["pumping"] * (n_observations - half))
    rng.shuffle(methods)
    sample_rows = []
    for method in methods:
        candidates = prior_data[prior_data["method"].eq(method)]
        sample_rows.append(candidates.sample(1, random_state=int(rng.integers(0, 2**31 - 1))).iloc[0])
    sampled = pd.DataFrame(sample_rows).reset_index(drop=True)
    sampled["x"] = rng.uniform(-0.72 * extent, 0.72 * extent, len(sampled))
    sampled["y"] = rng.uniform(-0.72 * extent, 0.72 * extent, len(sampled))
    sampled["support_radius"] = np.where(sampled["method"].eq("slug"), 12.0, 28.0)
    h = build_support_operator(coords, sampled)
    true_support = h @ truth
    conditional_pred = predict_conditional_prior(model, sampled)
    constant = _method_constants(prior_data)
    sampled["conditional_correction"] = conditional_pred["predicted_log_residual_mean"].to_numpy(dtype=float)
    sampled["conditional_sigma"] = conditional_pred["predicted_log_residual_sigma"].to_numpy(dtype=float)
    sampled["constant_correction"] = [constant[method][0] for method in sampled["method"]]
    sampled["constant_sigma"] = [constant[method][1] for method in sampled["method"]]
    noise = rng.normal(0.0, sampled["conditional_sigma"].to_numpy(dtype=float) * 0.45)
    sampled["actual_log_K_star"] = true_support
    sampled["actual_log_K_hat"] = sampled["actual_log_K_star"] + sampled["conditional_correction"] + noise
    sampled.attrs["coords"] = coords
    sampled.attrs["truth"] = truth
    return sampled


def _method_constants(prior_data: pd.DataFrame) -> dict[str, tuple[float, float]]:
    constants = {}
    for method, group in prior_data.groupby("method"):
        values = group["log_residual"].to_numpy(dtype=float)
        constants[method] = (float(np.mean(values)), max(float(np.std(values, ddof=1)), 0.05))
    return constants


def _field_table(coords: np.ndarray, truth: np.ndarray, posteriors: list) -> pd.DataFrame:
    table = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "truth_logk": truth})
    for posterior in posteriors:
        key = _slugify(posterior.approach)
        table[f"{key}_mean"] = posterior.mean
        table[f"{key}_sd"] = np.sqrt(np.maximum(posterior.variance, 0.0))
        table[f"{key}_error"] = posterior.mean - truth
    return table


def _summarize(metrics: pd.DataFrame, observations: pd.DataFrame) -> dict:
    indexed = metrics.set_index("approach")
    trustk_rmse = float(indexed.loc["TRUST-K conditional", "rmse_logk"])
    constant_rmse = float(indexed.loc["Method-constant soft data", "rmse_logk"])
    hard_rmse = float(indexed.loc["Hard interpretation", "rmse_logk"])
    return {
        "n_observations": int(len(observations)),
        "method_counts": observations.groupby("method").size().astype(int).to_dict(),
        "metrics": metrics.to_dict(orient="records"),
        "checks": {
            "trustk_beats_method_constant_rmse": {"pass": trustk_rmse < constant_rmse},
            "trustk_beats_hard_rmse": {"pass": trustk_rmse < hard_rmse},
            "trustk_coverage_at_least_method_constant": {
                "pass": float(indexed.loc["TRUST-K conditional", "coverage_95"])
                >= float(indexed.loc["Method-constant soft data", "coverage_95"])
            },
        },
    }


def _plot_spatial_assimilation(
    fields: pd.DataFrame,
    observations: pd.DataFrame,
    metrics: pd.DataFrame,
    figure_prefix: Path,
    *,
    grid_n: int,
) -> None:
    set_trustk_style()
    fig, axes = plt.subplots(2, 3, figsize=(journal_width(170), 5.35))
    truth = fields["truth_logk"].to_numpy(dtype=float).reshape(grid_n, grid_n)
    x = np.sort(fields["x"].unique())
    y = np.sort(fields["y"].unique())
    error_columns = [
        ("hard_interpretation_error", "Hard error", axes[0, 1]),
        ("method_constant_soft_data_error", "Method-constant error", axes[0, 2]),
        ("trust_k_conditional_error", "TRUST-K error", axes[1, 0]),
    ]

    ax = axes[0, 0]
    mesh = ax.pcolormesh(x, y, truth, shading="auto", cmap=cmc.batlow)
    pumping_obs = observations[observations["method"].eq("pumping")]
    slug_obs = observations[observations["method"].eq("slug")]
    ax.scatter(
        pumping_obs["x"],
        pumping_obs["y"],
        marker="s",
        facecolor="0.1",
        edgecolor="none",
        linewidth=0.0,
        s=30,
    )
    ax.scatter(
        slug_obs["x"],
        slug_obs["y"],
        marker="o",
        facecolor="white",
        edgecolor="0.15",
        linewidth=0.45,
        s=24,
    )
    observation_handles = [
        Line2D([0], [0], marker="s", linestyle="", color="0.1", markerfacecolor="0.1", markeredgecolor="none", markersize=4.5, label="pumping obs."),
        Line2D([0], [0], marker="o", linestyle="", color="0.15", markerfacecolor="white", markeredgecolor="0.15", markersize=4.0, label="slug obs."),
    ]
    ax.set_title("True latent field")
    ax.text(-0.12, 1.08, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5}, clip_on=False)
    fig.colorbar(mesh, ax=ax, fraction=0.040, pad=0.025)

    vmax = max(float(np.max(np.abs(fields[column]))) for column, _, _ in error_columns)
    for label, (column, title, ax) in zip(["(b)", "(c)", "(d)"], error_columns):
        values = fields[column].to_numpy(dtype=float).reshape(grid_n, grid_n)
        mesh = ax.pcolormesh(x, y, values, shading="auto", cmap=cmc.vik, vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.text(-0.12, 1.08, label, transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5}, clip_on=False)
        fig.colorbar(mesh, ax=ax, fraction=0.040, pad=0.025)

    ax = axes[1, 1]
    order = ["Hard interpretation", "Method-constant soft data", "TRUST-K conditional"]
    labels = ["Hard", "Method c", "TRUST-K"]
    colors = [cmc.batlow(0.25), cmc.batlow(0.55), cmc.batlow(0.82)]
    metric_ordered = metrics.set_index("approach").loc[order]
    ax.bar(np.arange(len(order)), metric_ordered["rmse_logk"], color=colors, edgecolor="none", linewidth=0.0)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(r"Grid RMSE in $\ln K$")
    ax.text(-0.12, 1.08, "(e)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5}, clip_on=False)

    ax = axes[1, 2]
    ax.bar(np.arange(len(order)), metric_ordered["coverage_95"], color=colors, edgecolor="none", linewidth=0.0)
    ax.axhline(0.95, color="0.25", lw=0.8, ls="--")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Grid 95% coverage")
    ax.text(-0.12, 1.08, "(f)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5}, clip_on=False)

    for ax in axes.ravel():
        ax.set_aspect("equal" if ax in axes[:, :3].ravel()[:4] else "auto")
        ax.spines[["top", "right"]].set_visible(False)
        if ax not in [axes[1, 1], axes[1, 2]]:
            ax.set_xlabel("x (m)")
            if ax in [axes[0, 0], axes[1, 0]]:
                ax.set_ylabel("y (m)")
            else:
                ax.set_ylabel("")
    fig.legend(handles=observation_handles, loc="upper left", bbox_to_anchor=(0.075, 0.985), frameon=False, ncol=2, handletextpad=0.35, columnspacing=1.0)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.13, top=0.91, wspace=0.58, hspace=0.28)
    export_figure(fig, figure_prefix)
    plt.close(fig)


def _slugify(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


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
    parser.add_argument("--conditional-predictions", default="data/processed/conditional_prior_predictions.csv")
    parser.add_argument("--residuals", default="data/processed/formal_population_1728_engineering_residuals.csv")
    parser.add_argument("--registry", default="data/processed/formal_dimensionless_case_registry_2000.csv")
    parser.add_argument("--qc", default="data/processed/formal_population_1728_engineering_qc.csv")
    parser.add_argument("--metrics", default="data/processed/spatial_assimilation_metrics.csv")
    parser.add_argument("--fields", default="data/processed/spatial_assimilation_fields.csv")
    parser.add_argument("--observations", default="data/processed/spatial_assimilation_observations.csv")
    parser.add_argument("--report", default="outputs/reports/spatial_assimilation.json")
    parser.add_argument("--figure-prefix", default="outputs/figures/fig08_spatial_assimilation")
    parser.add_argument("--grid-n", type=int, default=29)
    parser.add_argument("--n-observations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260518)
    args = parser.parse_args(argv)
    report = run_spatial_assimilation(
        conditional_predictions_path=args.conditional_predictions,
        residuals_path=args.residuals,
        registry_path=args.registry,
        qc_path=args.qc,
        metrics_path=args.metrics,
        fields_path=args.fields,
        observations_path=args.observations,
        report_path=args.report,
        figure_prefix=args.figure_prefix,
        grid_n=args.grid_n,
        n_observations=args.n_observations,
        seed=args.seed,
    )
    all_pass = all(item["pass"] for item in report["checks"].values())
    print(f"n_observations={report['n_observations']}")
    print(f"spatial_assimilation_pass={all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

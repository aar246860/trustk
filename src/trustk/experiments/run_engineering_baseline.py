"""Run engineering-practice interpretation on synthetic TRUST-K responses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trustk.diagnostics.fit_quality import summarize_fit_quality
from trustk.interpretation.engineering import (
    estimate_transmissivity_cooper_jacob,
    estimate_transmissivity_slug_semilog,
)
from trustk.plotting.style import export_figure, journal_width, set_trustk_style
from trustk.priors.transformation_uncertainty import estimate_transformation_uncertainty


def run_engineering_baseline(
    curves_path: str | Path = "data/processed/synthetic_population_curves.csv",
    support_residuals_path: str | Path = "data/processed/synthetic_population_residuals.csv",
    settings_path: str | Path = "data/processed/solver_simulation_settings.csv",
    residuals_path: str | Path = "data/processed/engineering_population_residuals.csv",
    fitted_curves_path: str | Path = "data/processed/engineering_fitted_curves.csv",
    qc_path: str | Path = "data/processed/engineering_fit_quality_qc.csv",
    prior_path: str | Path = "data/processed/engineering_transformation_uncertainty_prior.csv",
    report_path: str | Path = "outputs/reports/engineering_baseline.json",
    qc_figure_prefix: str | Path = "outputs/figures/fig12_engineering_fit_quality_qc",
    prior_figure_prefix: str | Path = "outputs/figures/fig13_engineering_transformation_uncertainty_prior",
    *,
    min_prior_cases: int = 5,
) -> dict:
    """Refit synthetic responses using engineering semi-log methods."""

    curves = pd.read_csv(curves_path)
    support = pd.read_csv(support_residuals_path)
    settings = pd.read_csv(settings_path)

    residuals, fitted = _build_engineering_tables(curves, support, settings)
    qc = summarize_fit_quality(fitted, residuals)
    prior = estimate_transformation_uncertainty(qc, min_cases=min_prior_cases)
    report = _summarize_engineering_baseline(residuals, qc, prior)
    report["purpose"] = "engineering-practice Cooper-Jacob and semi-log slug transformation residual baseline"
    report["input_curves_path"] = str(curves_path)
    report["support_target_source"] = str(support_residuals_path)

    for path, table in [
        (residuals_path, residuals),
        (fitted_curves_path, fitted),
        (qc_path, qc),
        (prior_path, prior),
    ]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    _plot_engineering_qc(fitted, qc, Path(qc_figure_prefix))
    Path(qc_figure_prefix).with_suffix(".csv").write_text(qc.to_csv(index=False), encoding="utf-8")
    Path(qc_figure_prefix).with_suffix(".json").write_text(
        json.dumps(
            {
                "figure": Path(qc_figure_prefix).name,
                "palette": "official cmcrameri cmc.batlow and cmc.vik",
                "purpose": "engineering-practice fit windows and residual QC",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _plot_engineering_prior(qc, prior, Path(prior_figure_prefix))
    Path(prior_figure_prefix).with_suffix(".csv").write_text(prior.to_csv(index=False), encoding="utf-8")
    Path(prior_figure_prefix).with_suffix(".json").write_text(
        json.dumps(
            {
                "figure": Path(prior_figure_prefix).name,
                "palette": "official cmcrameri cmc.batlow and cmc.vik",
                "purpose": "engineering-practice method-level transformation uncertainty",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report


def _build_engineering_tables(
    curves: pd.DataFrame,
    support: pd.DataFrame,
    settings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_support = {
        "case_id",
        "K_star_pumping_m_s",
        "K_star_slug_m_s",
        "sigma_Y2",
    }
    missing = required_support.difference(support.columns)
    if missing:
        raise ValueError(f"support table is missing required columns: {sorted(missing)}")

    settings_index = settings.set_index("case_id")
    support_index = support.set_index("case_id")
    residual_rows = []
    fitted_rows = []
    for case_id, case_curves in curves.groupby("case_id", sort=False):
        if case_id not in settings_index.index or case_id not in support_index.index:
            continue
        setting = settings_index.loc[case_id]
        target = support_index.loc[case_id]
        pumping_curve = case_curves[case_curves["method"].eq("pumping")].sort_values("time_s")
        slug_curve = case_curves[case_curves["method"].eq("slug")].sort_values("time_s")

        pumping_fit = estimate_transmissivity_cooper_jacob(
            times_s=pumping_curve["time_s"].to_numpy(dtype=float),
            drawdown_m=pumping_curve["drawdown_m"].to_numpy(dtype=float),
            radius_m=float(pumping_curve["observation_radius_m"].iloc[0]),
            pumping_rate_m3_s=float(setting["pumping_rate_m3_s"]),
            storativity=float(setting["storativity"]),
            min_points=max(3, min(8, len(pumping_curve))),
        )
        slug_fit = estimate_transmissivity_slug_semilog(
            times_s=slug_curve["time_s"].to_numpy(dtype=float),
            normalized_head=slug_curve["response_value"].to_numpy(dtype=float),
            well_storage_m2=float(setting["well_storage_m2"]),
            well_radius_m=float(setting["well_radius_m"]),
            outer_radius_m=float(setting["r_max_m"]),
            min_points=max(3, min(8, len(slug_curve))),
        )
        k_hat_p = pumping_fit.transmissivity_m2_s / float(setting["aquifer_thickness_m"])
        k_hat_s = slug_fit.transmissivity_m2_s / float(setting["aquifer_thickness_m"])
        residual_rows.append(
            {
                "case_id": case_id,
                "sigma_Y2": float(target["sigma_Y2"]),
                "K_star_pumping_m_s": float(target["K_star_pumping_m_s"]),
                "K_star_slug_m_s": float(target["K_star_slug_m_s"]),
                "K_hat_pumping_m_s": float(k_hat_p),
                "K_hat_slug_m_s": float(k_hat_s),
                "T_hat_pumping_m2_s": float(pumping_fit.transmissivity_m2_s),
                "T_hat_slug_m2_s": float(slug_fit.transmissivity_m2_s),
                "log_residual_pumping": float(np.log(k_hat_p) - np.log(target["K_star_pumping_m_s"])),
                "log_residual_slug": float(np.log(k_hat_s) - np.log(target["K_star_slug_m_s"])),
                "pumping_interpretation_method": "Cooper-Jacob",
                "slug_interpretation_method": "semi-log slug",
                "pumping_fit_rmse_log_response": float(pumping_fit.rmse_log_response),
                "slug_fit_rmse_log_response": float(slug_fit.rmse_log_response),
                "pumping_fit_r_squared": float(pumping_fit.r_squared or np.nan),
                "slug_fit_r_squared": float(slug_fit.r_squared or np.nan),
                "pumping_fit_slope": float(pumping_fit.slope or np.nan),
                "pumping_fit_intercept": float(pumping_fit.intercept or np.nan),
                "slug_fit_slope": float(slug_fit.slope or np.nan),
                "slug_fit_intercept": float(slug_fit.intercept or np.nan),
                "pumping_fit_time_min_s": float(pumping_fit.fit_time_min_s),
                "pumping_fit_time_max_s": float(pumping_fit.fit_time_max_s),
                "slug_fit_time_min_s": float(slug_fit.fit_time_min_s),
                "slug_fit_time_max_s": float(slug_fit.fit_time_max_s),
                "pumping_fit_point_count": int(pumping_fit.fit_point_count),
                "slug_fit_point_count": int(slug_fit.fit_point_count),
            }
        )
        fitted_rows.extend(_fitted_pumping_rows(pumping_curve, pumping_fit, setting, k_hat_p, target))
        fitted_rows.extend(_fitted_slug_rows(slug_curve, slug_fit, setting, k_hat_s, target))
    residual_table = pd.DataFrame(residual_rows)
    fitted_table = pd.DataFrame(fitted_rows)
    floor = np.finfo(float).tiny
    fitted_table["log_error"] = np.log(np.maximum(fitted_table["observed_for_log"].to_numpy(dtype=float), floor)) - np.log(
        np.maximum(fitted_table["fitted_for_log"].to_numpy(dtype=float), floor)
    )
    fitted_table["abs_log_error"] = np.abs(fitted_table["log_error"])
    return residual_table, fitted_table


def _fitted_pumping_rows(curve: pd.DataFrame, fit, setting: pd.Series, k_hat: float, target: pd.Series) -> list[dict]:
    rows = []
    for _, row in curve.iterrows():
        fitted_drawdown = (fit.intercept or 0.0) + (fit.slope or 0.0) * np.log10(float(row["time_s"]))
        fitted_drawdown = max(float(fitted_drawdown), np.finfo(float).tiny)
        used = fit.fit_time_min_s <= float(row["time_s"]) <= fit.fit_time_max_s
        fitted_response = 4.0 * np.pi * float(setting["T0_m2_s"]) * fitted_drawdown / float(setting["pumping_rate_m3_s"])
        rows.append(
            {
                **_common_fitted_row(row, target, "pumping", k_hat),
                "fitted_response_value": fitted_response,
                "observed_for_log": float(row["drawdown_m"]),
                "fitted_for_log": fitted_drawdown,
                "used_for_fit": bool(used),
            }
        )
    return rows


def _fitted_slug_rows(curve: pd.DataFrame, fit, setting: pd.Series, k_hat: float, target: pd.Series) -> list[dict]:
    rows = []
    for _, row in curve.iterrows():
        fitted_response = float(np.exp((fit.intercept or 0.0) + (fit.slope or 0.0) * float(row["time_s"])))
        fitted_response = max(fitted_response, np.finfo(float).tiny)
        used = fit.fit_time_min_s <= float(row["time_s"]) <= fit.fit_time_max_s
        rows.append(
            {
                **_common_fitted_row(row, target, "slug", k_hat),
                "fitted_response_value": fitted_response,
                "observed_for_log": float(row["response_value"]),
                "fitted_for_log": fitted_response,
                "used_for_fit": bool(used),
            }
        )
    return rows


def _common_fitted_row(row: pd.Series, target: pd.Series, method: str, k_hat: float) -> dict:
    k_star_col = "K_star_pumping_m_s" if method == "pumping" else "K_star_slug_m_s"
    log_residual = float(np.log(k_hat) - np.log(target[k_star_col]))
    observed = float(row["drawdown_m"]) if method == "pumping" else float(row["response_value"])
    out = {
        "case_id": row["case_id"],
        "method": method,
        "time_s": float(row["time_s"]),
        "time_D": float(row["time_D"]),
        "response_value": float(row["response_value"]),
        "drawdown_m": float(row["drawdown_m"]) if np.isfinite(row["drawdown_m"]) else np.nan,
        "observation_radius_m": float(row["observation_radius_m"]),
        "sigma_Y2": float(target["sigma_Y2"]),
        "K_hat_m_s": float(k_hat),
        "K_star_m_s": float(target[k_star_col]),
        "log_residual": log_residual,
        "abs_log_residual": abs(log_residual),
    }
    out["observed_for_log"] = observed
    return out


def _summarize_engineering_baseline(residuals: pd.DataFrame, qc: pd.DataFrame, prior: pd.DataFrame) -> dict:
    finite = bool(
        not residuals.empty
        and np.isfinite(
            residuals[
                [
                    "K_hat_pumping_m_s",
                    "K_hat_slug_m_s",
                    "log_residual_pumping",
                    "log_residual_slug",
                ]
            ].to_numpy(dtype=float)
        ).all()
    )
    method_summary = {}
    for method, group in qc.groupby("method"):
        method_summary[method] = {
            "n": int(len(group)),
            "pass": int(group["qc_class"].eq("pass").sum()),
            "warning": int(group["qc_class"].eq("warning").sum()),
            "fail": int(group["qc_class"].eq("fail").sum()),
            "rmse_median": float(group["rmse_log_response"].median()),
            "rmse_q95": float(group["rmse_log_response"].quantile(0.95)),
        }
    return {
        "simulated_case_count": int(len(residuals)),
        "fit_quality_rows": int(len(qc)),
        "checks": {
            "finite_engineering_residuals": {"pass": finite},
            "prior_methods_present": {"pass": set(prior["method"]) == {"pumping", "slug"} if not prior.empty else False},
        },
        "method_summary": method_summary,
        "transformation_uncertainty_prior": prior.set_index("method").to_dict(orient="index")
        if not prior.empty
        else {},
    }


def _plot_engineering_qc(fitted: pd.DataFrame, qc: pd.DataFrame, figure_prefix: Path) -> None:
    set_trustk_style()
    figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(journal_width(170), 3.35))
    colors = {"pumping": cmc.vik(0.25), "slug": cmc.vik(0.75)}
    labels = {"pumping": "Pumping", "slug": "Slug"}

    for method, ax in [("pumping", axes[0]), ("slug", axes[1])]:
        method_qc = qc[qc["method"].eq(method)].sort_values("rmse_log_response")
        if method_qc.empty:
            continue
        selected = method_qc.iloc[[0, len(method_qc) // 2, -1]]["case_id"].tolist()
        for idx, case_id in enumerate(selected):
            data = fitted[fitted["case_id"].eq(case_id) & fitted["method"].eq(method)].sort_values("time_s")
            alpha = [0.95, 0.75, 0.55][idx]
            ax.plot(data["time_D"], data["response_value"], color=colors[method], lw=0.9, alpha=alpha)
            ax.plot(data["time_D"], data["fitted_response_value"], color="0.15", lw=0.8, ls="--", alpha=alpha)
            used = data[data["used_for_fit"].astype(bool)]
            ax.scatter(used["time_D"], used["fitted_response_value"], s=9, color="0.15", alpha=alpha)
        ax.set_xscale("log")
        ax.set_xlabel(r"$t_D$")
        ax.set_ylabel("Drawdown" if method == "pumping" else r"$H_w/H_0$")
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.03, 0.96, "(a)" if method == "pumping" else "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")
        ax.text(0.97, 0.96, labels[method], transform=ax.transAxes, ha="right", va="top", fontsize=8)

    ax = axes[2]
    positions = {"pumping": 0.85, "slug": 1.15}
    for method in ["pumping", "slug"]:
        values = qc[qc["method"].eq(method)]["rmse_log_response"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        parts = ax.violinplot([values], positions=[positions[method]], widths=0.22, showmeans=True, showextrema=False)
        parts["bodies"][0].set_facecolor(colors[method])
        parts["bodies"][0].set_edgecolor("0.25")
        parts["bodies"][0].set_alpha(0.72)
        parts["cmeans"].set_color("0.15")
        jitter = np.linspace(-0.045, 0.045, len(values))
        ax.scatter(np.full(len(values), positions[method]) + jitter, values, s=10, color=colors[method], edgecolor="0.25", linewidth=0.2, alpha=0.72)
    ax.set_xticks([positions["pumping"], positions["slug"]])
    ax.set_xticklabels(["Pumping", "Slug"])
    ax.set_yscale("log")
    ax.set_ylabel(r"Fit RMSE in log response")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.03, 0.96, "(c)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.22, top=0.94, wspace=0.38)
    export_figure(fig, figure_prefix)
    plt.close(fig)


def _plot_engineering_prior(qc: pd.DataFrame, prior: pd.DataFrame, figure_prefix: Path) -> None:
    set_trustk_style()
    figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(journal_width(170), 3.25))
    colors = {"pumping": cmc.vik(0.25), "slug": cmc.vik(0.75)}
    positions = {"pumping": 0.85, "slug": 1.15}
    accepted = qc[qc["qc_class"].eq("pass")]

    ax = axes[0]
    for method in ["pumping", "slug"]:
        values = accepted[accepted["method"].eq(method)]["log_residual"].to_numpy(dtype=float)
        if len(values) == 0:
            continue
        parts = ax.violinplot([values], positions=[positions[method]], widths=0.22, showmeans=True, showextrema=False)
        parts["bodies"][0].set_facecolor(colors[method])
        parts["bodies"][0].set_edgecolor("0.25")
        parts["bodies"][0].set_alpha(0.72)
        parts["cmeans"].set_color("0.15")
        jitter = np.linspace(-0.045, 0.045, len(values))
        ax.scatter(np.full(len(values), positions[method]) + jitter, values, s=10, color=colors[method], edgecolor="0.25", linewidth=0.2, alpha=0.72)
    ax.axhline(0.0, color="0.25", lw=0.8, ls="--")
    ax.set_xticks([positions["pumping"], positions["slug"]])
    ax.set_xticklabels(["Pumping", "Slug"])
    ax.set_ylabel(r"QC-pass log residual $r_m$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.03, 0.96, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")

    ax = axes[1]
    for method in ["pumping", "slug"]:
        row = prior[prior["method"].eq(method)]
        if row.empty:
            continue
        row = row.iloc[0]
        y = float(row["bias_factor_c"])
        yerr = np.array([[y - row["q05_factor"]], [row["q95_factor"] - y]])
        ax.errorbar(positions[method], y, yerr=yerr, fmt="o", color=colors[method], mec="0.25", ms=5, lw=1.0, capsize=3)
        ax.text(positions[method], y * 1.08, f"n={int(row['n_cases'])}", ha="center", va="bottom", fontsize=7)
    ax.axhline(1.0, color="0.25", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xticks([positions["pumping"], positions["slug"]])
    ax.set_xticklabels(["Pumping", "Slug"])
    ax.set_ylabel(r"Bias factor $c_m$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.03, 0.96, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.22, top=0.94, wspace=0.34)
    export_figure(fig, figure_prefix)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", default="data/processed/synthetic_population_curves.csv")
    parser.add_argument("--support-residuals", default="data/processed/synthetic_population_residuals.csv")
    parser.add_argument("--settings", default="data/processed/solver_simulation_settings.csv")
    parser.add_argument("--residuals", default="data/processed/engineering_population_residuals.csv")
    parser.add_argument("--fitted-curves", default="data/processed/engineering_fitted_curves.csv")
    parser.add_argument("--qc", default="data/processed/engineering_fit_quality_qc.csv")
    parser.add_argument("--prior", default="data/processed/engineering_transformation_uncertainty_prior.csv")
    parser.add_argument("--report", default="outputs/reports/engineering_baseline.json")
    parser.add_argument("--qc-figure-prefix", default="outputs/figures/fig12_engineering_fit_quality_qc")
    parser.add_argument("--prior-figure-prefix", default="outputs/figures/fig13_engineering_transformation_uncertainty_prior")
    parser.add_argument("--min-prior-cases", type=int, default=5)
    args = parser.parse_args(argv)
    report = run_engineering_baseline(
        curves_path=args.curves,
        support_residuals_path=args.support_residuals,
        settings_path=args.settings,
        residuals_path=args.residuals,
        fitted_curves_path=args.fitted_curves,
        qc_path=args.qc,
        prior_path=args.prior,
        report_path=args.report,
        qc_figure_prefix=args.qc_figure_prefix,
        prior_figure_prefix=args.prior_figure_prefix,
        min_prior_cases=args.min_prior_cases,
    )
    all_pass = all(item["pass"] for item in report["checks"].values())
    print(f"simulated_case_count={report['simulated_case_count']}")
    print(f"engineering_baseline_pass={all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

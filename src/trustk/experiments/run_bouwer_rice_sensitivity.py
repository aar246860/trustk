from __future__ import annotations

import argparse
import json
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trustk.interpretation.engineering import estimate_hydraulic_conductivity_bouwer_rice
from trustk.plotting.style import export_figure, journal_width, set_trustk_style
from trustk.priors.transformation_uncertainty import estimate_transformation_uncertainty


def run_bouwer_rice_sensitivity(
    curves_path: str | Path,
    support_residuals_path: str | Path,
    settings_path: str | Path,
    engineering_prior_path: str | Path,
    residuals_path: str | Path,
    fitted_curves_path: str | Path,
    qc_path: str | Path,
    prior_path: str | Path,
    comparison_path: str | Path,
    report_path: str | Path,
    figure_prefix: str | Path,
    *,
    min_prior_cases: int = 5,
) -> dict:
    curves = pd.read_csv(curves_path)
    support = pd.read_csv(support_residuals_path)
    settings = pd.read_csv(settings_path)
    engineering_prior = pd.read_csv(engineering_prior_path)
    residuals, fitted = _build_bouwer_rice_tables(curves, support, settings)
    qc = _summarize_bouwer_rice_quality(fitted, residuals)
    prior = estimate_transformation_uncertainty(qc, min_cases=min_prior_cases)
    comparison = _compare_slug_priors(engineering_prior, prior)
    report = _summarize(residuals, qc, prior, comparison)
    report["purpose"] = "Bouwer-Rice-conditioned slug transformation uncertainty sensitivity analysis"
    report["inputs"] = {
        "curves": str(curves_path),
        "support_residuals": str(support_residuals_path),
        "settings": str(settings_path),
        "engineering_prior": str(engineering_prior_path),
    }
    for path, table in [
        (residuals_path, residuals),
        (fitted_curves_path, fitted),
        (qc_path, qc),
        (prior_path, prior),
        (comparison_path, comparison),
    ]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _plot_comparison(residuals, qc, comparison, Path(figure_prefix))
    Path(figure_prefix).with_suffix(".csv").write_text(comparison.to_csv(index=False), encoding="utf-8")
    Path(figure_prefix).with_suffix(".json").write_text(
        json.dumps(
            {
                "figure": Path(figure_prefix).name,
                "palette": "official cmcrameri cmc.batlow and cmc.vik",
                "purpose": report["purpose"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report


def _build_bouwer_rice_tables(
    curves: pd.DataFrame,
    support: pd.DataFrame,
    settings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings_index = settings.set_index("case_id")
    support_index = support.set_index("case_id")
    residual_rows = []
    fitted_rows = []
    for case_id, case_curves in curves[curves["method"].eq("slug")].groupby("case_id", sort=False):
        if case_id not in settings_index.index or case_id not in support_index.index:
            continue
        setting = settings_index.loc[case_id]
        target = support_index.loc[case_id]
        slug_curve = case_curves.sort_values("time_s")
        fit = estimate_hydraulic_conductivity_bouwer_rice(
            times_s=slug_curve["time_s"].to_numpy(dtype=float),
            normalized_head=slug_curve["response_value"].to_numpy(dtype=float),
            casing_radius_m=float(setting["casing_radius_m"]),
            well_radius_m=float(setting["well_radius_m"]),
            screen_length_m=float(setting["aquifer_thickness_m"]),
            aquifer_thickness_m=float(setting["aquifer_thickness_m"]),
            effective_radius_m=float(setting["r_max_m"]),
            min_points=max(3, min(8, len(slug_curve))),
        )
        k_hat = fit.transmissivity_m2_s / float(setting["aquifer_thickness_m"])
        log_residual = float(np.log(k_hat) - np.log(target["K_star_slug_m_s"]))
        residual_rows.append(
            {
                "case_id": case_id,
                "sigma_Y2": float(target["sigma_Y2"]),
                "K_star_slug_m_s": float(target["K_star_slug_m_s"]),
                "K_hat_slug_bouwer_rice_m_s": float(k_hat),
                "T_hat_slug_bouwer_rice_m2_s": float(fit.transmissivity_m2_s),
                "log_residual_slug_bouwer_rice": log_residual,
                "slug_interpretation_method": "Bouwer-Rice",
                "fit_rmse_log_response": float(fit.rmse_log_response),
                "fit_r_squared": float(fit.r_squared if fit.r_squared is not None else np.nan),
                "fit_slope": float(fit.slope if fit.slope is not None else np.nan),
                "fit_intercept": float(fit.intercept if fit.intercept is not None else np.nan),
                "fit_time_min_s": float(fit.fit_time_min_s),
                "fit_time_max_s": float(fit.fit_time_max_s),
                "fit_point_count": int(fit.fit_point_count),
                "casing_radius_m": float(setting["casing_radius_m"]),
                "well_radius_m": float(setting["well_radius_m"]),
                "effective_radius_m": float(setting["r_max_m"]),
                "screen_length_m": float(setting["aquifer_thickness_m"]),
                "well_storage_m2": float(setting["well_storage_m2"]),
            }
        )
        fitted_rows.extend(_fitted_rows(slug_curve, fit, target, k_hat))
    residual_table = pd.DataFrame(residual_rows)
    fitted_table = pd.DataFrame(fitted_rows)
    floor = np.finfo(float).tiny
    fitted_table["log_error"] = np.log(np.maximum(fitted_table["observed_for_log"].to_numpy(dtype=float), floor)) - np.log(
        np.maximum(fitted_table["fitted_for_log"].to_numpy(dtype=float), floor)
    )
    fitted_table["abs_log_error"] = np.abs(fitted_table["log_error"])
    return residual_table, fitted_table


def _fitted_rows(curve: pd.DataFrame, fit, target: pd.Series, k_hat: float) -> list[dict]:
    rows = []
    intercept = float(fit.intercept if fit.intercept is not None else 0.0)
    slope = float(fit.slope if fit.slope is not None else 0.0)
    for _, row in curve.iterrows():
        fitted_response = float(np.exp(intercept + slope * float(row["time_s"])))
        fitted_response = max(fitted_response, np.finfo(float).tiny)
        used = fit.fit_time_min_s <= float(row["time_s"]) <= fit.fit_time_max_s
        rows.append(
            {
                "case_id": row["case_id"],
                "method": "slug_bouwer_rice",
                "time_s": float(row["time_s"]),
                "time_D": float(row["time_D"]),
                "response_value": float(row["response_value"]),
                "drawdown_m": float(row["drawdown_m"]) if np.isfinite(row["drawdown_m"]) else np.nan,
                "observation_radius_m": float(row["observation_radius_m"]),
                "sigma_Y2": float(target["sigma_Y2"]),
                "K_hat_m_s": float(k_hat),
                "K_star_m_s": float(target["K_star_slug_m_s"]),
                "log_residual": float(np.log(k_hat) - np.log(target["K_star_slug_m_s"])),
                "abs_log_residual": abs(float(np.log(k_hat) - np.log(target["K_star_slug_m_s"]))),
                "fitted_response_value": fitted_response,
                "observed_for_log": float(row["response_value"]),
                "fitted_for_log": fitted_response,
                "used_for_fit": bool(used),
            }
        )
    return rows


def _summarize_bouwer_rice_quality(fitted: pd.DataFrame, residuals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    residual_index = residuals.set_index("case_id")
    for case_id, group in fitted.groupby("case_id", sort=False):
        used = group[group["used_for_fit"].astype(bool)]
        errors = used["log_error"].to_numpy(dtype=float)
        finite = errors[np.isfinite(errors)]
        if len(finite) < 3:
            rmse = np.nan
            mean_error = np.nan
            max_abs = np.nan
            early_bias = np.nan
            late_bias = np.nan
        else:
            rmse = float(np.sqrt(np.mean(finite**2)))
            mean_error = float(np.mean(finite))
            max_abs = float(np.max(np.abs(finite)))
            ordered = used.sort_values("time_s")
            n_tail = max(1, int(np.ceil(len(ordered) * 0.25)))
            early_bias = float(ordered["log_error"].head(n_tail).mean())
            late_bias = float(ordered["log_error"].tail(n_tail).mean())
        residual = residual_index.loc[case_id]
        rows.append(
            {
                "case_id": case_id,
                "method": "slug_bouwer_rice",
                "rmse_log_response": rmse,
                "mean_log_error": mean_error,
                "max_abs_log_error": max_abs,
                "early_mean_log_error": early_bias,
                "late_mean_log_error": late_bias,
                "fit_point_count": int(len(used)),
                "qc_class": _slug_qc_class(rmse),
                "sigma_Y2": float(residual["sigma_Y2"]),
                "log_residual": float(residual["log_residual_slug_bouwer_rice"]),
                "abs_log_residual": abs(float(residual["log_residual_slug_bouwer_rice"])),
                "K_hat_m_s": float(residual["K_hat_slug_bouwer_rice_m_s"]),
                "K_star_m_s": float(residual["K_star_slug_m_s"]),
            }
        )
    return pd.DataFrame(rows)


def _slug_qc_class(rmse: float) -> str:
    if not np.isfinite(rmse):
        return "fail"
    if rmse <= 0.25:
        return "pass"
    if rmse <= 0.75:
        return "warning"
    return "fail"


def _compare_slug_priors(engineering_prior: pd.DataFrame, br_prior: pd.DataFrame) -> pd.DataFrame:
    semilog = engineering_prior[engineering_prior["method"].eq("slug")].iloc[0].to_dict()
    br = br_prior[br_prior["method"].eq("slug_bouwer_rice")].iloc[0].to_dict()
    rows = []
    for label, row in [("semi-log slug", semilog), ("Bouwer-Rice slug", br)]:
        rows.append(
            {
                "slug_interpretation": label,
                "n_cases": int(row["n_cases"]),
                "mean_log_residual": float(row["mean_log_residual"]),
                "sd_log_residual": float(row["sd_log_residual"]),
                "bias_factor_c": float(row["bias_factor_c"]),
                "target_correction_factor": float(row["target_correction_factor"]),
                "scatter_factor": float(row["scatter_factor"]),
                "q05_factor": float(row["q05_factor"]),
                "q95_factor": float(row["q95_factor"]),
            }
        )
    out = pd.DataFrame(rows)
    out["delta_mean_log_residual_vs_semilog"] = out["mean_log_residual"] - float(out.loc[0, "mean_log_residual"])
    out["bias_factor_ratio_vs_semilog"] = out["bias_factor_c"] / float(out.loc[0, "bias_factor_c"])
    return out


def _summarize(residuals: pd.DataFrame, qc: pd.DataFrame, prior: pd.DataFrame, comparison: pd.DataFrame) -> dict:
    values = qc[qc["qc_class"].eq("pass")]["log_residual"].to_numpy(dtype=float)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    se = sd / np.sqrt(len(values))
    ci = np.exp([mean - 1.96 * se, mean + 1.96 * se])
    return {
        "simulated_case_count": int(len(residuals)),
        "qc_summary": qc.groupby("qc_class").size().astype(int).to_dict(),
        "bouwer_rice_prior": prior.to_dict(orient="records"),
        "prior_comparison": comparison.to_dict(orient="records"),
        "bias_factor_95_ci": {"lower": float(ci[0]), "upper": float(ci[1])},
        "checks": {
            "all_cases_fit": {"pass": int(len(residuals)) > 0},
            "pass_cases_available": {"pass": int(qc["qc_class"].eq("pass").sum()) >= 5},
            "finite_residuals": {"pass": bool(np.isfinite(residuals["log_residual_slug_bouwer_rice"]).all())},
        },
    }


def _plot_comparison(residuals: pd.DataFrame, qc: pd.DataFrame, comparison: pd.DataFrame, figure_prefix: Path) -> None:
    set_trustk_style()
    fig, axes = plt.subplots(1, 3, figsize=(journal_width(170), 3.35))
    sigma = residuals["sigma_Y2"].to_numpy(dtype=float)
    point_size = 5.0 + 8.0 * (sigma - np.min(sigma)) / max(float(np.max(sigma) - np.min(sigma)), 1.0e-12)

    ax = axes[0]
    ax.scatter(
        residuals["K_star_slug_m_s"],
        residuals["K_hat_slug_bouwer_rice_m_s"],
        c=sigma,
        s=point_size,
        cmap=cmc.batlow,
        edgecolor="none",
        alpha=0.35,
        rasterized=True,
    )
    finite = residuals[["K_star_slug_m_s", "K_hat_slug_bouwer_rice_m_s"]].to_numpy(dtype=float)
    low = 10.0 ** np.floor(np.log10(np.nanmin(finite[finite > 0.0])))
    high = 10.0 ** np.ceil(np.log10(np.nanmax(finite[finite > 0.0])))
    ax.plot([low, high], [low, high], color="0.2", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel(r"$K_S^*$ (m s$^{-1}$)")
    ax.set_ylabel(r"$\hat K_{S,BR}$ (m s$^{-1}$)")
    ax.text(0.03, 0.96, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")

    ax = axes[1]
    values = qc[qc["qc_class"].eq("pass")]["log_residual"].to_numpy(dtype=float)
    parts = ax.violinplot([values], positions=[1.0], widths=0.28, showmeans=True, showextrema=False)
    parts["bodies"][0].set_facecolor(cmc.vik(0.72))
    parts["bodies"][0].set_edgecolor("0.25")
    parts["bodies"][0].set_alpha(0.72)
    parts["cmeans"].set_color("0.15")
    jitter = np.linspace(-0.065, 0.065, len(values))
    ax.scatter(np.full(len(values), 1.0) + jitter, values, s=4.0, color=cmc.vik(0.72), edgecolor="none", alpha=0.28, rasterized=True)
    ax.axhline(0.0, color="0.25", lw=0.8, ls="--")
    ax.set_xticks([1.0])
    ax.set_xticklabels(["BR slug"])
    ax.set_ylabel(r"QC-pass log residual")
    ax.text(0.03, 0.96, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")

    ax = axes[2]
    x = np.arange(len(comparison))
    ax.errorbar(
        x,
        comparison["bias_factor_c"],
        yerr=np.vstack(
            [
                comparison["bias_factor_c"] - comparison["q05_factor"],
                comparison["q95_factor"] - comparison["bias_factor_c"],
            ]
        ),
        fmt="o",
        color=cmc.vik(0.72),
        mec="0.25",
        ms=5,
        lw=1.0,
        capsize=3,
    )
    ax.axhline(1.0, color="0.25", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["Semi-log", "BR"], rotation=20, ha="right")
    ax.set_ylabel(r"Bias factor $c_S$")
    ax.text(0.03, 0.96, "(c)", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")

    for panel in axes:
        panel.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.24, top=0.94, wspace=0.44)
    figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    export_figure(fig, figure_prefix)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curves", default="data/processed/formal_population_1728_curves.csv")
    parser.add_argument("--support-residuals", default="data/processed/formal_population_1728_support_residuals.csv")
    parser.add_argument("--settings", default="data/processed/formal_solver_simulation_settings_2000.csv")
    parser.add_argument("--engineering-prior", default="data/processed/formal_population_1728_engineering_prior.csv")
    parser.add_argument("--residuals", default="data/processed/formal_population_1728_bouwer_rice_residuals.csv")
    parser.add_argument("--fitted-curves", default="data/processed/formal_population_1728_bouwer_rice_fitted_curves.csv")
    parser.add_argument("--qc", default="data/processed/formal_population_1728_bouwer_rice_qc.csv")
    parser.add_argument("--prior", default="data/processed/formal_population_1728_bouwer_rice_prior.csv")
    parser.add_argument("--comparison", default="data/processed/formal_population_1728_bouwer_rice_prior_comparison.csv")
    parser.add_argument("--report", default="outputs/reports/formal_bouwer_rice_sensitivity.json")
    parser.add_argument("--figure-prefix", default="outputs/figures/fig_bouwer_rice_sensitivity")
    args = parser.parse_args(argv)
    report = run_bouwer_rice_sensitivity(
        curves_path=args.curves,
        support_residuals_path=args.support_residuals,
        settings_path=args.settings,
        engineering_prior_path=args.engineering_prior,
        residuals_path=args.residuals,
        fitted_curves_path=args.fitted_curves,
        qc_path=args.qc,
        prior_path=args.prior,
        comparison_path=args.comparison,
        report_path=args.report,
        figure_prefix=args.figure_prefix,
    )
    all_pass = all(item["pass"] for item in report["checks"].values())
    print(f"simulated_case_count={report['simulated_case_count']}")
    print(f"bouwer_rice_sensitivity_pass={all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

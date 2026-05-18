"""Generate TRUST-K field-validation figures from processed Lovelock data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import Rbf
from scipy.optimize import least_squares

from trustk.analytical.theis import theis_drawdown
from trustk.data.extract_pumping import GPM_TO_M3_S
from trustk.plotting.style import export_figure, journal_width, set_trustk_style
from trustk.validation.field_style import build_field_style_validation
from trustk.validation.lovelock import PUMP_START, PUMP_STOP, cyclic_theis_drawdown


def _read_processed(processed_dir: str | Path) -> dict[str, pd.DataFrame]:
    processed = Path(processed_dir)
    return {
        "pump": pd.read_csv(processed / "lovelock_bongo_pumping_well.csv"),
        "summary": pd.read_csv(processed / "lovelock_overlap_summary.csv"),
        "comparison": pd.read_csv(processed / "lovelock_bongo_model_comparison.csv", parse_dates=["datetime"]),
        "hard_slug": pd.read_csv(processed / "lovelock_hard_slug_predictions.csv", parse_dates=["datetime"]),
    }


def _panel_label(ax, label: str) -> None:
    ax.text(
        -0.11,
        1.07,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
        clip_on=False,
    )


def _write_metadata(out_prefix: Path, metadata: dict) -> None:
    out_prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def figure_field_context(processed_dir: str | Path, out_dir: str | Path) -> None:
    data = _read_processed(processed_dir)
    pump = data["pump"].iloc[0]
    summary = data["summary"].copy()
    summary["x_rel_km"] = (summary["x_m"] - pump["x_m"]) / 1000.0
    summary["y_rel_km"] = (summary["y_m"] - pump["y_m"]) / 1000.0
    summary["log10_k_slug"] = np.log10(summary["k_slug_m_s"])
    summary["marker_size"] = 25 + 300 * summary["peak_measured_drawdown_m"] / summary["peak_measured_drawdown_m"].max()

    set_trustk_style()
    layer_colors = {3: cmc.vik(0.18), 4: cmc.vik(0.34), 5: cmc.vik(0.72), 7: cmc.vik(0.9)}
    fig, axes = plt.subplots(1, 2, figsize=(journal_width(170), 3.35), gridspec_kw={"width_ratios": [1.05, 1.15]})
    ax = axes[0]
    ax.scatter(
        summary["x_rel_km"],
        summary["y_rel_km"],
        c=[layer_colors.get(int(layer), "0.4") for layer in summary["model_layer"]],
        s=summary["marker_size"],
        edgecolor="0.15",
        linewidth=0.45,
        zorder=3,
    )
    ax.scatter(0, 0, marker="*", s=140, c="black", edgecolor="white", linewidth=0.4, zorder=4)
    ax.text(0.15, 0.15, "BONGO", ha="left", va="bottom", fontsize=7)
    offsets = {
        "BD": (0.18, -0.38),
        "BS": (0.18, -0.08),
        "MD": (0.15, 0.15),
        "MS": (0.15, -0.35),
        "PINN": (0.15, 0.15),
        "WD": (0.15, -0.35),
        "WS": (0.15, 0.12),
    }
    for row in summary.itertuples(index=False):
        dx, dy = offsets.get(row.well, (0.12, 0.12))
        ax.text(row.x_rel_km + dx, row.y_rel_km + dy, row.well, fontsize=7)
    ax.axhline(0, color="0.82", lw=0.5, zorder=0)
    ax.axvline(0, color="0.82", lw=0.5, zorder=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Easting relative to pumping well (km)")
    ax.set_ylabel("Northing relative to pumping well (km)")
    ax.set_xlim(-2.6, 7.4)
    ax.set_ylim(-16.4, 5.9)
    ax.spines[["top", "right"]].set_visible(False)
    size_handles = [
        plt.scatter([], [], s=s, facecolor="white", edgecolor="0.25", linewidth=0.5)
        for s in [45, 120, 260]
    ]
    ax.legend(
        size_handles,
        ["small", "medium", "large"],
        title="Peak s",
        frameon=False,
        loc="center right",
        bbox_to_anchor=(0.98, 0.48),
        borderpad=0.2,
        handletextpad=0.8,
    )
    _panel_label(ax, "(a)")

    ax = axes[1]
    pair_order = ["BD,BS", "MD,MS", "WD,WS", "PINN"]
    pair_x = {name: i for i, name in enumerate(pair_order)}
    for pair in pair_order:
        rows = summary[summary["co_located_wells"] == pair].sort_values("model_layer")
        if rows.empty:
            continue
        x = pair_x[pair]
        ax.plot([x] * len(rows), rows["log10_k_slug"], color="0.75", lw=0.8, zorder=1)
        for offset, row in zip(np.linspace(-0.08, 0.08, len(rows)), rows.itertuples(index=False)):
            ax.scatter(
                x + offset,
                row.log10_k_slug,
                s=48,
                color=layer_colors.get(int(row.model_layer), "0.4"),
                edgecolor="0.15",
                linewidth=0.4,
                zorder=3,
            )
            ax.text(x + offset + 0.045, row.log10_k_slug, f"{row.well}", va="center", fontsize=7)
    ax.set_xticks(list(pair_x.values()))
    ax.set_xticklabels(pair_order)
    ax.set_ylabel(r"$\log_{10} K_{\mathrm{slug}}$ (m s$^{-1}$)")
    ax.set_xlabel("Co-located or validation well group")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    layer_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color, label=f"Layer {layer}")
        for layer, color in sorted(layer_colors.items())
    ]
    ax.legend(
        handles=layer_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.55, 1.15),
        ncol=4,
        columnspacing=0.8,
    )
    _panel_label(ax, "(b)")

    fig.subplots_adjust(wspace=0.42, top=0.84)
    out_prefix = Path(out_dir) / "fig01_field_context"
    export_figure(fig, out_prefix)
    summary.to_csv(out_prefix.with_suffix(".csv"), index=False)
    _write_metadata(
        out_prefix,
        {
            "figure": "fig01_field_context",
            "question": "Which Lovelock wells link slug-test K to the Bongo pumping/recovery response?",
            "palette": "official cmcrameri cmc.vik",
            "panels": {
                "a": "Map of overlap wells relative to the pumping well; marker color is model layer and marker size is observed peak drawdown.",
                "b": "Co-located well groups showing vertical/layer-dependent slug K differences at the same x-y support.",
            },
        },
    )
    plt.close(fig)


def figure_response_validation(processed_dir: str | Path, out_dir: str | Path) -> None:
    data = _read_processed(processed_dir)
    comparison = data["comparison"]
    hard_slug = data["hard_slug"]
    summary = data["summary"].copy()
    selected = ["MD", "PINN", "WS"]
    colors = {well: cmc.batlow(x) for well, x in zip(selected, [0.18, 0.55, 0.86])}

    response = comparison[comparison["well"].isin(selected)].merge(
        hard_slug[["well", "datetime", "hard_slug_drawdown_m"]],
        on=["well", "datetime"],
        how="left",
    )
    response["time_days"] = response["elapsed_since_pump_start_s"] / 86400.0

    set_trustk_style()
    fig, axes = plt.subplots(1, 2, figsize=(journal_width(170), 3.35), gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    for well in selected:
        group = response[response["well"] == well]
        color = colors[well]
        ax.plot(group["time_days"], group["measured_drawdown_m"], "o", ms=2.4, color=color, alpha=0.82)
        ax.plot(group["time_days"], group["simulated_drawdown_m"], "-", color=color, lw=1.1)
        ax.plot(group["time_days"], group["hard_slug_drawdown_m"], "--", color=color, lw=0.9, alpha=0.9)
    ax.axvline((PUMP_STOP - PUMP_START).total_seconds() / 86400.0, color="0.35", lw=0.8, ls=":")
    ax.text(6.2, ax.get_ylim()[1] * 0.92, "pump off", fontsize=7, ha="left", va="top")
    ax.set_xlabel("Elapsed time since pumping started (days)")
    ax.set_ylabel("Drawdown (m)")
    ax.spines[["top", "right"]].set_visible(False)
    style_handles = [
        Line2D([0], [0], marker="o", linestyle="", color="0.2", label="measured", markersize=3),
        Line2D([0], [0], linestyle="-", color="0.2", label="USGS model", linewidth=1.1),
        Line2D([0], [0], linestyle="--", color="0.2", label="hard slug", linewidth=0.9),
    ]
    well_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=colors[well], label=well, markersize=4)
        for well in selected
    ]
    leg1 = ax.legend(handles=style_handles, frameon=False, loc="upper left", bbox_to_anchor=(0.03, 0.88), handlelength=1.6)
    ax.add_artist(leg1)
    ax.legend(handles=well_handles, frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.88), ncol=1)
    _panel_label(ax, "(a)")

    ax = axes[1]
    x = np.arange(len(summary))
    labels = summary["well"].tolist()
    null_rmse = summary["null_rmse_m"].to_numpy(float)
    rmse_frame = pd.DataFrame(
        {
            "well": labels,
            "USGS model": summary["usgs_model_rmse_m"].to_numpy(float) / null_rmse,
            "independent Theis": summary["theis_rmse_m"].to_numpy(float) / null_rmse,
            "hard slug": summary["hard_slug_rmse_m"].to_numpy(float) / null_rmse,
        }
    )
    offsets = [-0.22, 0.0, 0.22]
    methods = ["USGS model", "independent Theis", "hard slug"]
    method_colors = [cmc.vik(0.18), cmc.vik(0.50), cmc.vik(0.82)]
    for offset, method, color in zip(offsets, methods, method_colors):
        ax.bar(x + offset, rmse_frame[method], width=0.2, color=color, edgecolor="0.2", linewidth=0.3, label=method)
    ax.axhline(1.0, color="0.25", lw=0.8, ls=":", label="null baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("RMSE / null RMSE")
    ax.set_xlabel("Overlap well")
    ax.set_ylim(0, max(1.35, float(rmse_frame[methods].max().max()) * 1.15))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    _panel_label(ax, "(b)")

    fig.subplots_adjust(wspace=0.34)
    out_prefix = Path(out_dir) / "fig02_response_validation"
    export_figure(fig, out_prefix)
    response.to_csv(out_prefix.with_suffix(".csv"), index=False)
    rmse_frame.to_csv(Path(out_dir) / "fig02_response_validation_rmse.csv", index=False)
    _write_metadata(
        out_prefix,
        {
            "figure": "fig02_response_validation",
            "question": "Can slug-test K values be transferred as hard data to predict pumping and recovery drawdown?",
            "palette": "official cmcrameri cmc.batlow and cmc.vik",
            "panels": {
                "a": "Measured response compared with USGS calibrated MODFLOW output and hard-slug Theis baseline.",
                "b": "Predictive error normalized by a zero-drawdown null predictor.",
            },
        },
    )
    plt.close(fig)


def figure_field_leave_one_out(processed_dir: str | Path, out_dir: str | Path) -> None:
    """Generate the final field-style Figure 9 validation."""

    result = build_field_style_validation(processed_dir)
    posterior = result.posterior_wells.copy()
    slug = result.slug_predictions.copy()
    metrics = result.metrics.copy()
    root = Path(__file__).resolve().parents[3]
    pumping_tables = _read_pumping_inventory_tables(root)
    timeseries = pumping_tables["timeseries"]
    periods = pumping_tables["periods"]
    inventory = pumping_tables["inventory"]
    selected = ["BD", "MD", "WS"]
    colors = {well: cmc.batlow(x) for well, x in zip(selected, [0.18, 0.56, 0.86])}
    mwp1_wells = _select_detected_observation_wells(inventory, "MWP1_MooRAH")
    mwp2_wells = _select_detected_observation_wells(inventory, "MWP2_Bongo")

    set_trustk_style()
    fig, axes = plt.subplots(2, 3, figsize=(journal_width(170), 6.65))
    axes = axes.ravel()

    pumping_fit_records = []
    pumping_fit_records.extend(
        _plot_pumping_test_response(axes[0], timeseries, periods, inventory, "MWP1_MooRAH", mwp1_wells)
    )
    pumping_fit_records.extend(
        _plot_pumping_test_response(axes[1], timeseries, periods, inventory, "MWP2_Bongo", mwp2_wells)
    )
    _plot_slug_tail(axes[2], slug, selected, colors)
    _plot_method_disagreement(axes[3], posterior)
    _plot_trustk_fusion(axes[4], posterior)
    _plot_predictive_scores(axes[5], metrics)

    for label, ax in zip(["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"], axes):
        _panel_label(ax, label)

    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.10, top=0.96, wspace=0.64, hspace=0.46)
    out_prefix = Path(out_dir) / "fig09_field_leave_one_out"
    export_figure(fig, out_prefix)
    metrics.to_csv(out_prefix.with_suffix(".csv"), index=False)
    metrics.to_csv(Path(out_dir) / "fig09_field_validation_metrics.csv", index=False)
    timeseries.to_csv(Path(out_dir) / "fig09_field_leave_one_out_pumping_inventory_timeseries.csv", index=False)
    inventory.to_csv(Path(out_dir) / "fig09_field_leave_one_out_pumping_inventory.csv", index=False)
    pd.DataFrame(pumping_fit_records).to_csv(Path(out_dir) / "fig09_field_leave_one_out_pumping_only_fits.csv", index=False)
    slug.to_csv(Path(out_dir) / "fig09_field_leave_one_out_slug_fit.csv", index=False)
    posterior.to_csv(Path(out_dir) / "fig09_field_leave_one_out_posterior_wells.csv", index=False)
    _write_metadata(
        out_prefix,
        {
            "figure": "fig09_field_leave_one_out",
            "question": "How do the two Lovelock multi-well pumping tests and slug-test interpretations fit their own observations, disagree as K estimates, and enter TRUST-K as soft data rather than hard interchangeable K values?",
            "claim_boundary": "This field-style validation does not claim true field K recovery; it uses USGS multi-well pumping/recovery products, observed slug responses, method-specific fits, and predictive checks to show why TRUST-K keeps transformation uncertainty during fusion.",
            "palette": "official cmcrameri cmc.batlow, cmc.vik, and cmc.lipari",
            "panels": {
                "a": "MooRAH multi-well pumping/recovery drawdown observations with pumping-period Theis fits and recovery predictions from the same parameters.",
                "b": "Bongo multi-well pumping/recovery drawdown observations with pumping-period Theis fits and recovery predictions from the same parameters.",
                "c": "Representative selected slug-recovery segments with semi-log tail fits.",
                "d": "Raw disagreement between slug-test and pumping-test apparent K estimates.",
                "e": "TRUST-K soft-data fusion of method-specific apparent K estimates into a posterior K observation.",
                "f": "Predictive score normalized by a zero-response null predictor.",
            },
            "overlap_check": "Panel legends are placed outside or in low-density corners; dense well labels in panels d and e are offset with leader lines to avoid marker and text overlap.",
        },
    )
    plt.close(fig)


def _read_pumping_inventory_tables(root: Path) -> dict[str, pd.DataFrame]:
    tables = root / "outputs" / "tables"
    inventory = pd.read_csv(tables / "usgs_pumping_drawdown_inventory.csv", parse_dates=["start", "end"])
    distances = pd.read_csv(tables / "usgs_pumping_well_distances.csv")
    distance_cols = ["test", "well", "distance_mi_report", "distance_km_report"]
    inventory = inventory.merge(distances[distance_cols], on=["test", "well"], how="left")
    return {
        "timeseries": pd.read_csv(tables / "usgs_pumping_drawdown_timeseries.csv", parse_dates=["datetime"]),
        "periods": pd.read_csv(tables / "usgs_pumping_test_periods.csv", parse_dates=["pump_start", "pump_stop", "recovery_end"]),
        "inventory": inventory,
    }


def _select_detected_observation_wells(inventory: pd.DataFrame, test: str) -> list[str]:
    """Use all observation wells with report-scale drawdown above the detection threshold."""

    plot = inventory[
        inventory["test"].eq(test)
        & inventory["role"].eq("observation well")
        & inventory["above_detection_threshold_0p05ft"].astype(bool)
    ].copy()
    plot = plot.sort_values(["distance_mi_report", "well"])
    return plot["well"].tolist()


def _plot_pumping_theis_fit(
    ax,
    pumping: pd.DataFrame,
    summary: pd.DataFrame,
    pumping_well: pd.Series,
    selected: list[str],
    colors: dict[str, tuple],
) -> None:
    """Plot observed pumping/recovery drawdown with independent Theis fits."""

    markers = {"BD": "o", "MD": "s", "WS": "^"}
    duration_s = (PUMP_STOP - PUMP_START).total_seconds()
    pump_off_day = duration_s / 86400.0
    pumping_rate_m3_s = float(pumping_well["q_m3_s"])
    ax.axvspan(0.0, pump_off_day, color="0.92", zorder=0)
    ax.axvline(pump_off_day, color="0.32", lw=0.75, ls=":", zorder=1)

    selected_data = pumping[pumping["well"].isin(selected)].copy()
    for well in selected:
        group = pumping[pumping["well"].eq(well)].sort_values("time_days")
        meta = summary[summary["well"].eq(well)]
        if group.empty or meta.empty:
            continue
        color = colors[well]
        marker = markers.get(well, "o")
        time_days = group["time_days"].to_numpy(float)
        observed = group["measured_drawdown_m"].to_numpy(float)
        usgs_model = group["simulated_drawdown_m"].to_numpy(float)
        radius = float(meta["distance_to_pump_m"].iloc[0])
        transmissivity = float(meta["theis_transmissivity_m2_s"].iloc[0])
        storativity = float(meta["theis_storativity"].iloc[0])
        fitted = cyclic_theis_drawdown(
            radius,
            group["elapsed_since_pump_start_s"].to_numpy(float),
            transmissivity,
            storativity,
            pumping_rate_m3_s,
            duration_s,
        )
        ax.plot(time_days, observed, linestyle="", marker=marker, ms=2.35, color=color, alpha=0.72)
        ax.plot(time_days, fitted, "-", lw=1.05, color=color, alpha=0.95)
        ax.plot(time_days, usgs_model, ":", lw=0.9, color=color, alpha=0.78)
        distance_km = radius / 1000.0
        y_label = np.nanpercentile(observed, 84)
        ax.text(
            time_days[-1] + 0.08,
            y_label,
            f"{well}\n{distance_km:.1f} km",
            ha="left",
            va="center",
            fontsize=6.4,
            color=color,
            linespacing=0.85,
        )

    observed_max = float(selected_data["measured_drawdown_m"].max())
    fitted_max = float(selected_data["simulated_drawdown_m"].max())
    ax.set_ylim(-0.002, max(0.05, observed_max * 1.30, fitted_max * 1.30))
    ax.set_xlim(selected_data["time_days"].min() - 0.35, selected_data["time_days"].max() + 1.05)
    ax.set_xlabel("Elapsed time since Bongo pumping started (days)")
    ax.set_ylabel("Drawdown (m)")
    ax.spines[["top", "right"]].set_visible(False)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color="0.25", label="observed", markersize=3),
        Line2D([0], [0], linestyle="-", color="0.25", label="Theis fit", linewidth=1.0),
        Line2D([0], [0], linestyle=":", color="0.25", label="USGS model", linewidth=0.9),
    ]
    ax.legend(
        handles=handles,
        frameon=True,
        framealpha=0.82,
        facecolor="white",
        edgecolor="none",
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        handlelength=1.4,
        labelspacing=0.22,
        borderpad=0.25,
    )


def _plot_pumping_test_response(
    ax,
    timeseries: pd.DataFrame,
    periods: pd.DataFrame,
    inventory: pd.DataFrame,
    test: str,
    wells: list[str],
) -> list[dict]:
    row = periods[periods["test"].eq(test)].iloc[0]
    pump_off_day = float(row["pumping_duration_hours"]) / 24.0
    pumping_rate_m3_s = float(row["pumping_rate_gpm"]) * GPM_TO_M3_S
    test_data = timeseries[timeseries["test"].eq(test)].copy()
    test_inventory = inventory[inventory["test"].eq(test)].copy()
    markers = ["o", "s", "^", "D", "v", "P"]
    fit_records: list[dict] = []
    distances = []
    for well in wells:
        info = test_inventory[test_inventory["well"].eq(well)]
        distances.append(_distance_from_inventory(info) if not info.empty else np.nan)
    finite_distances = np.array([value for value in distances if np.isfinite(value)], dtype=float)
    if finite_distances.size:
        norm = plt.Normalize(float(finite_distances.min()), float(finite_distances.max()))
    else:
        norm = plt.Normalize(0.0, 1.0)
    cmap = cmc.vik

    ax.axvspan(0.0, pump_off_day, color="0.92", zorder=0)
    ax.axvline(pump_off_day, color="0.32", lw=0.75, ls=":", zorder=1)
    for idx, well in enumerate(wells):
        group = test_data[test_data["well"].eq(well)].sort_values("elapsed_since_pump_start_days")
        info = test_inventory[test_inventory["well"].eq(well)]
        if group.empty or info.empty:
            continue
        marker = markers[idx % len(markers)]
        report_drawdown = float(info["report_estimated_drawdown_ft"].iloc[0])
        if not np.isfinite(report_drawdown) or report_drawdown <= 0:
            report_drawdown = max(float(group["measured_drawdown_ft"].max()), np.finfo(float).tiny)
        x = group["elapsed_since_pump_start_days"].to_numpy(float)
        measured = group["measured_drawdown_ft"].to_numpy(float) / report_drawdown
        distance_mi = _safe_float(info.get("distance_mi_report", pd.Series([np.nan])).iloc[0])
        if not np.isfinite(distance_mi):
            distance_mi = _distance_from_inventory(info)
        color = cmap(norm(distance_mi)) if np.isfinite(distance_mi) else "0.45"
        pumping_mask = group["phase"].eq("pumping").to_numpy()
        recovery_mask = group["phase"].eq("recovery").to_numpy()
        ax.plot(x[pumping_mask], measured[pumping_mask], linestyle="", marker=marker, ms=1.9, color=color, alpha=0.74)
        if np.any(recovery_mask):
            ax.plot(
                x[recovery_mask],
                measured[recovery_mask],
                linestyle="",
                marker=marker,
                ms=1.75,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=0.45,
                alpha=0.45,
            )
        fit = _fit_pumping_only_theis(
            group,
            radius_m=distance_mi * 1609.344,
            pumping_rate_m3_s=pumping_rate_m3_s,
            pump_off_day=pump_off_day,
        )
        fit_record = {
            "test": test,
            "well": well,
            "distance_mi": distance_mi,
            "n_pumping_fit_points": fit["n"] if fit is not None else 0,
            "transmissivity_m2_s": fit["transmissivity_m2_s"] if fit is not None else np.nan,
            "storativity": fit["storativity"] if fit is not None else np.nan,
            "rmse_m": fit["rmse_m"] if fit is not None else np.nan,
            "recovery_check_rmse_m": fit["recovery_rmse_m"] if fit is not None else np.nan,
        }
        fit_records.append(fit_record)
        if fit is not None:
            x_fit = fit["time_days"]
            y_fit = fit["drawdown_m"] / (report_drawdown * 0.3048)
            ax.plot(x_fit, y_fit, "-", lw=0.85, color=color, alpha=0.90)
            if len(fit["recovery_time_days"]):
                y_recovery = fit["recovery_drawdown_m"] / (report_drawdown * 0.3048)
                ax.plot(fit["recovery_time_days"], y_recovery, "--", lw=0.75, color=color, alpha=0.62)
    ax.axhline(1.0, color="0.35", lw=0.65, ls="--")
    ax.set_xlabel("Elapsed time since pumping start (days)")
    ax.set_ylabel("Drawdown / report drawdown")
    ax.set_ylim(-0.08, 1.45)
    ax.spines[["top", "right"]].set_visible(False)
    label = "MooRAH" if test == "MWP1_MooRAH" else "Bongo"
    ax.text(
        0.04,
        0.94,
        f"{label}, {len(wells)} wells",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 1.0},
    )
    style_handles = [
        Line2D([0], [0], marker="o", linestyle="", color="0.25", markersize=3, label="pumping obs."),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="0.25", color="0.25", markersize=3, label="recovery obs."),
        Line2D([0], [0], linestyle="-", color="0.25", lw=1.0, label="pumping fit"),
        Line2D([0], [0], linestyle="--", color="0.25", lw=0.9, label="recovery check"),
        Line2D([0], [0], linestyle=":", color="0.25", lw=0.8, label="pump stop"),
    ]
    ax.legend(
        frameon=True,
        framealpha=0.78,
        facecolor="white",
        edgecolor="none",
        loc="lower left",
        bbox_to_anchor=(0.0, 1.015),
        handles=style_handles,
        fontsize=5.7,
        ncol=3,
        handlelength=1.35,
        columnspacing=0.75,
        labelspacing=0.15,
        borderpad=0.22,
    )
    if finite_distances.size:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.018)
        cbar.ax.set_title("mi", fontsize=7, pad=3)
        cbar.ax.tick_params(labelsize=6.5)
    return fit_records


def _fit_pumping_only_theis(
    group: pd.DataFrame,
    *,
    radius_m: float,
    pumping_rate_m3_s: float,
    pump_off_day: float,
) -> dict | None:
    if not np.isfinite(radius_m) or radius_m <= 0:
        return None
    fit_data = group[
        group["phase"].eq("pumping")
        & (group["elapsed_since_pump_start_days"] > 0.02)
        & (group["measured_drawdown_m"] >= 0)
    ].copy()
    if len(fit_data) < 5:
        return None

    t = fit_data["elapsed_since_pump_start_days"].to_numpy(float) * 86400.0
    y = fit_data["measured_drawdown_m"].to_numpy(float)
    scale = max(float(np.nanpercentile(y, 90)), 0.006)

    def residual(log_params: np.ndarray) -> np.ndarray:
        transmissivity = float(np.exp(log_params[0]))
        storativity = float(np.exp(log_params[1]))
        pred = theis_drawdown(radius_m, t, transmissivity, storativity, pumping_rate_m3_s)
        return (pred - y) / scale

    try:
        result = least_squares(
            residual,
            x0=np.log([0.1, 1e-3]),
            bounds=(np.log([1e-7, 1e-9]), np.log([100.0, 0.8])),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=500,
        )
        transmissivity = float(np.exp(result.x[0]))
        storativity = float(np.exp(result.x[1]))
        pred = theis_drawdown(radius_m, t, transmissivity, storativity, pumping_rate_m3_s)
    except (FloatingPointError, RuntimeError, ValueError):
        return None

    recovery_time_days = np.array([], dtype=float)
    recovery_pred = np.array([], dtype=float)
    recovery_rmse = np.nan
    recovery_data = group[
        group["phase"].eq("recovery")
        & (group["elapsed_since_pump_start_days"] > pump_off_day)
        & (group["measured_drawdown_m"] >= 0)
    ].copy()
    if not recovery_data.empty:
        recovery_time_days = recovery_data["elapsed_since_pump_start_days"].to_numpy(float)
        recovery_t = recovery_time_days * 86400.0
        pump_off_s = max(float(pump_off_day) * 86400.0, 1.0)
        since_stop = np.maximum(recovery_t - pump_off_s, 1.0)
        recovery_pred = theis_drawdown(radius_m, recovery_t, transmissivity, storativity, pumping_rate_m3_s) - theis_drawdown(
            radius_m,
            since_stop,
            transmissivity,
            storativity,
            pumping_rate_m3_s,
        )
        recovery_obs = recovery_data["measured_drawdown_m"].to_numpy(float)
        recovery_rmse = float(np.sqrt(np.nanmean((recovery_pred - recovery_obs) ** 2)))

    return {
        "time_days": fit_data["elapsed_since_pump_start_days"].to_numpy(float),
        "drawdown_m": pred,
        "recovery_time_days": recovery_time_days,
        "recovery_drawdown_m": recovery_pred,
        "transmissivity_m2_s": transmissivity,
        "storativity": storativity,
        "rmse_m": float(np.sqrt(np.nanmean((pred - y) ** 2))),
        "recovery_rmse_m": recovery_rmse,
        "n": int(len(fit_data)),
    }


def _plot_pumping_distance_inventory(ax, inventory: pd.DataFrame) -> None:
    plot = inventory[inventory["role"].eq("observation well")].copy()
    plot = plot[np.isfinite(plot["report_estimated_drawdown_ft"]) & np.isfinite(plot["distance_mi_report"])]
    test_colors = {"MWP1_MooRAH": cmc.batlow(0.24), "MWP2_Bongo": cmc.batlow(0.72)}
    certainty_markers = {"High": "o", "Moderate": "s", "Low": "^"}
    for (test, certainty), group in plot.groupby(["test", "report_relative_certainty"]):
        ax.scatter(
            group["distance_mi_report"],
            group["report_estimated_drawdown_ft"],
            s=22 + 42 * np.sqrt(group["report_estimated_drawdown_ft"].clip(lower=0.01)),
            marker=certainty_markers.get(str(certainty), "o"),
            color=test_colors.get(test, "0.45"),
            edgecolor="0.18",
            linewidth=0.35,
            alpha=0.82 if certainty != "Low" else 0.55,
        )
    ax.axhline(0.05, color="0.25", lw=0.75, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Distance from pumping well (mi)")
    ax.set_ylabel("Report drawdown (ft)")
    ax.spines[["top", "right"]].set_visible(False)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=test_colors["MWP1_MooRAH"], markeredgecolor="0.18", markersize=4, label="MooRAH"),
        Line2D([0], [0], marker="o", linestyle="", color=test_colors["MWP2_Bongo"], markeredgecolor="0.18", markersize=4, label="Bongo"),
        Line2D([0], [0], linestyle=":", color="0.25", lw=0.9, label="0.05 ft"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=6.4, handlelength=1.4, labelspacing=0.25)
    ax.text(
        0.04,
        0.05,
        "circle high\nsquare moderate\ntriangle low",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.3,
    )


def _plot_co_located_slug_contrasts(ax, posterior: pd.DataFrame) -> None:
    plot = posterior.copy()
    if "co_located_wells" in plot.columns:
        plot["pair"] = plot["co_located_wells"].fillna(plot["well"])
    else:
        plot["xy_pair"] = plot["x_rel_km"].round(6).astype(str) + "_" + plot["y_rel_km"].round(6).astype(str)
        pair_names = plot.groupby("xy_pair")["well"].transform(lambda values: "/".join(sorted(values)))
        plot["pair"] = pair_names
    pairs = [group for _, group in plot.groupby("pair", sort=True) if len(group) > 1]
    colors = [cmc.vik(0.18), cmc.vik(0.55), cmc.vik(0.82)]
    for idx, group in enumerate(pairs):
        group = group.sort_values("slug_log10K")
        x = idx
        color = colors[idx % len(colors)]
        y = group["slug_log10K"].to_numpy(float)
        ax.plot([x, x], [float(np.min(y)), float(np.max(y))], color="0.35", lw=0.8, zorder=1)
        ax.scatter(np.full(len(group), x), y, s=34, color=color, edgecolor="0.18", linewidth=0.35, zorder=2)
        for offset, row in zip(np.linspace(-0.08, 0.08, len(group)), group.itertuples(index=False)):
            ax.text(x + offset, row.slug_log10K + 0.06, row.well, ha="center", va="bottom", fontsize=6.6)
        ratio = 10 ** (float(np.max(y)) - float(np.min(y)))
        ax.text(
            x + 0.16,
            float(np.mean(y)),
            f"{ratio:.1f}x",
            ha="left",
            va="center",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
        )
    ax.set_xticks(np.arange(len(pairs)))
    ax.set_xticklabels([group["pair"].iloc[0].replace(",", "/") for group in pairs], rotation=0)
    ax.set_xlim(-0.35, max(0.65, len(pairs) - 0.55))
    ax.set_ylabel(r"Slug-test $\log_{10} K$")
    ax.set_xlabel("Co-located well group")
    ax.spines[["top", "right"]].set_visible(False)


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _distance_from_inventory(info: pd.DataFrame) -> float:
    for col in ["distance_mi_report", "report_distance_mi", "distance_mi"]:
        if col in info.columns:
            value = _safe_float(info[col].iloc[0])
            if np.isfinite(value):
                return value
    return float("nan")


def _plot_field_layout(ax, posterior: pd.DataFrame) -> None:
    plot = _plot_positions(posterior)
    size = 30 + 260 * plot["peak_measured_drawdown_m"] / plot["peak_measured_drawdown_m"].max()
    for row in plot.itertuples(index=False):
        if abs(row.plot_x_rel_km - row.x_rel_km) > 1e-9 or abs(row.plot_y_rel_km - row.y_rel_km) > 1e-9:
            ax.plot([row.x_rel_km, row.plot_x_rel_km], [row.y_rel_km, row.plot_y_rel_km], color="0.58", lw=0.35, zorder=2)
    scatter = ax.scatter(
        plot["plot_x_rel_km"],
        plot["plot_y_rel_km"],
        c=plot["slug_log10K"],
        s=size,
        cmap=cmc.vik,
        edgecolor="0.15",
        linewidth=0.45,
        zorder=3,
    )
    ax.scatter(0, 0, marker="*", s=120, c="black", edgecolor="white", linewidth=0.45, zorder=4)
    ax.text(0.12, 0.10, "BONGO", fontsize=7, ha="left", va="bottom")
    _label_wells(ax, plot)
    ax.axhline(0, color="0.86", lw=0.45, zorder=0)
    ax.axvline(0, color="0.86", lw=0.45, zorder=0)
    ax.set_xlabel("Relative easting (km)")
    ax.set_ylabel("Relative northing (km)")
    ax.set_aspect("equal", adjustable="box")
    ax.spines[["top", "right"]].set_visible(False)
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.047, pad=0.025)
    cbar.ax.set_title(r"$\log_{10}K_s$", fontsize=7, pad=3)
    cbar.ax.tick_params(labelsize=7)


def _plot_pumping_holdout(ax, pumping: pd.DataFrame, selected: list[str], colors: dict[str, tuple]) -> None:
    markers = {"BD": "o", "MD": "s", "WS": "^"}
    pump_off_day = (PUMP_STOP - PUMP_START).total_seconds() / 86400.0
    ax.axvspan(0.0, pump_off_day, color="0.92", zorder=0)
    for well in selected:
        group = pumping[pumping["well"].eq(well)].sort_values("time_days")
        color = colors[well]
        ax.plot(
            group["time_days"],
            group["measured_drawdown_m"],
            linestyle="",
            marker=markers[well],
            ms=2.2,
            color=color,
            alpha=0.72,
        )
        ax.plot(group["time_days"], group["simulated_drawdown_m"], "-", lw=1.0, color=color)
    ax.axvline(0.0, color="0.35", lw=0.6, ls=":")
    ax.axvline(pump_off_day, color="0.35", lw=0.7, ls=":")
    observed_max = float(pumping[pumping["well"].isin(selected)]["measured_drawdown_m"].max())
    y_top = max(0.05, observed_max * 1.28)
    ax.set_ylim(-0.002, y_top)
    ax.set_xlabel("Elapsed time since MWP2 start (days)")
    ax.set_ylabel("Drawdown (m)")
    ax.spines[["top", "right"]].set_visible(False)
    styles = [
        Line2D([0], [0], marker="o", linestyle="", color="0.25", label="USGS extracted", markersize=3),
        Line2D([0], [0], linestyle="-", color="0.25", label="USGS model", linewidth=1.0),
    ]
    ax.legend(
        handles=styles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.54, 1.02),
        ncol=2,
        handlelength=1.4,
        columnspacing=0.9,
    )
    label_y = {"BD": 0.76, "MD": 0.66, "WS": 0.56}
    for well in selected:
        group = pumping[pumping["well"].eq(well)]
        distance_km = float(group["distance_to_pump_m"].iloc[0]) / 1000.0
        ax.text(
            0.98,
            label_y[well],
            f"{well} ({distance_km:.1f} km)",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=7,
            color=colors[well],
        )


def _plot_method_disagreement(ax, posterior: pd.DataFrame) -> None:
    """Show the raw mismatch between slug- and pumping-derived apparent K."""

    plot = posterior.sort_values("distance_to_pump_m").copy()
    size = 28 + 210 * plot["peak_measured_drawdown_m"] / plot["peak_measured_drawdown_m"].max()
    scatter = ax.scatter(
        plot["slug_log10K"],
        plot["pumping_log10K"],
        c=plot["distance_to_pump_m"] / 1000.0,
        s=size,
        cmap=cmc.lipari,
        edgecolor="none",
        linewidth=0.0,
        alpha=0.86,
        zorder=3,
    )
    low = float(np.nanmin([plot["slug_log10K"].min(), plot["pumping_log10K"].min()])) - 0.18
    high = float(np.nanmax([plot["slug_log10K"].max(), plot["pumping_log10K"].max()])) + 0.18
    ax.plot([low, high], [low, high], color="0.25", lw=0.8, ls="--", zorder=1)
    offsets = {
        "BD": (8, 13),
        "BS": (12, 3),
        "MD": (9, 9),
        "MS": (10, -18),
        "PINN": (8, -12),
        "WD": (8, -8),
        "WS": (9, 12),
    }
    for row in plot.itertuples(index=False):
        dx, dy = offsets.get(row.well, (0.05, 0.05))
        ax.annotate(
            row.well,
            xy=(row.slug_log10K, row.pumping_log10K),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.3,
            ha="left",
            va="center",
            arrowprops={"arrowstyle": "-", "color": "0.45", "lw": 0.25, "alpha": 0.50},
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 0.35},
        )
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel(r"Slug-test $\log_{10}K$ (m s$^{-1}$)")
    ax.set_ylabel(r"Pumping-test $\log_{10}K$ (m s$^{-1}$)")
    ax.set_aspect("equal", adjustable="box")
    ax.spines[["top", "right"]].set_visible(False)
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.047, pad=0.025)
    cbar.ax.set_title("km", fontsize=7, pad=3)
    cbar.ax.tick_params(labelsize=7)


def _plot_trustk_fusion(ax, posterior: pd.DataFrame) -> None:
    """Plot method-specific apparent K values and the TRUST-K soft posterior."""

    plot = posterior.sort_values("apparent_log10K_mean", ascending=False).reset_index(drop=True)
    well_colors = {well: cmc.batlow(x) for well, x in zip(plot["well"], np.linspace(0.12, 0.88, len(plot)))}
    y_label_positions = _spread_label_positions(
        dict(zip(plot["well"], plot["apparent_log10K_mean"], strict=True)),
        min_gap=0.25,
    )
    for row in plot.itertuples(index=False):
        color = well_colors[row.well]
        y_slug = float(row.slug_log10K)
        y_pump = float(row.pumping_log10K)
        y_fused = float(row.apparent_log10K_mean)
        y_err = 1.96 * float(row.apparent_logK_sd) / np.log(10.0)
        ax.plot([0, 2], [y_slug, y_fused], color=color, lw=0.65, alpha=0.40, zorder=1)
        ax.plot([1, 2], [y_pump, y_fused], color=color, lw=0.65, alpha=0.40, zorder=1)
        ax.scatter(0, y_slug, marker="o", s=22, color=color, edgecolor="none", linewidth=0.0, zorder=3)
        ax.scatter(1, y_pump, marker="s", s=22, color=color, edgecolor="none", linewidth=0.0, zorder=3)
        ax.errorbar(
            2,
            y_fused,
            yerr=y_err,
            marker="D",
            markersize=3.4,
            color=color,
            ecolor=color,
            elinewidth=0.7,
            capsize=2.0,
            markeredgecolor="none",
            markeredgewidth=0.0,
            zorder=4,
        )
        y_label = y_label_positions.get(row.well, y_fused)
        if abs(y_label - y_fused) > 0.03:
            ax.plot([2.04, 2.12], [y_fused, y_label], color=color, lw=0.35, alpha=0.55, zorder=2)
        ax.text(2.16, y_label, row.well, color=color, fontsize=6.3, ha="left", va="center")

    y_values = np.concatenate(
        [
            plot["slug_log10K"].to_numpy(float),
            plot["pumping_log10K"].to_numpy(float),
            plot["apparent_log10K_mean"].to_numpy(float),
        ]
    )
    ax.set_xlim(-0.35, 2.88)
    ax.set_ylim(float(np.nanmin(y_values)) - 0.50, float(np.nanmax(y_values)) + 0.50)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Slug\nfit", "Pumping\nfit", "TRUST-K\nsoft fusion"])
    ax.set_ylabel(r"Apparent $\log_{10}K$ (m s$^{-1}$)")
    ax.spines[["top", "right"]].set_visible(False)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color="0.25", label="slug", markersize=3.3),
        Line2D([0], [0], marker="s", linestyle="", color="0.25", label="pumping", markersize=3.3),
        Line2D([0], [0], marker="D", linestyle="", color="0.25", label="posterior", markersize=3.3),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.00, 0.035),
        fontsize=5.8,
        ncol=3,
        handletextpad=0.28,
        columnspacing=0.55,
    )


def _spread_label_positions(targets: dict[str, float], min_gap: float) -> dict[str, float]:
    """Separate right-edge labels while preserving their vertical order."""

    ordered = sorted(targets.items(), key=lambda item: item[1], reverse=True)
    adjusted: dict[str, float] = {}
    previous_y: float | None = None
    for well, y in ordered:
        label_y = float(y)
        if previous_y is not None and previous_y - label_y < min_gap:
            label_y = previous_y - min_gap
        adjusted[well] = label_y
        previous_y = label_y
    return adjusted


def _plot_slug_tail(ax, slug: pd.DataFrame, selected: list[str], colors: dict[str, tuple]) -> None:
    selected_segments = _select_slug_segments(slug, selected)
    markers = {"BD": "o", "MD": "s", "WS": "^"}
    well_handles = []
    for well in selected:
        segment = selected_segments.get(well)
        if segment is None:
            continue
        group = slug[(slug["well"].eq(well)) & (slug["segment"].eq(segment))].sort_values("elapsed_s")
        color = colors[well]
        stride = max(1, len(group) // 90)
        sample = group.iloc[::stride]
        marker = markers.get(well, "o")
        ax.plot(
            sample["plot_elapsed_s"],
            sample["recovery_amplitude_norm"],
            linestyle="",
            marker=marker,
            ms=1.8,
            color=color,
            markeredgecolor="none",
            alpha=0.62,
        )
        ax.plot(
            group["plot_elapsed_s"],
            group["semi_log_tail_prediction"],
            "-",
            lw=0.95,
            color=color,
            alpha=0.88,
        )
        split = float(group["holdout_split_s"].iloc[0])
        if np.isfinite(split):
            ax.axvline(split, color=color, lw=0.55, ls=":", alpha=0.35)
        well_handles.append(
            Line2D([0], [0], marker=marker, linestyle="-", color=color, markerfacecolor=color, markeredgecolor="none", markersize=3.2, lw=0.9, label=f"{well} seg. {segment}")
        )
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel(r"Normalized $|H_w/H_0|$")
    ax.spines[["top", "right"]].set_visible(False)
    split_handle = Line2D([0], [0], linestyle=":", color="0.35", label="fit/holdout split", linewidth=0.8)
    ax.legend(
        handles=well_handles + [split_handle],
        frameon=True,
        framealpha=0.82,
        facecolor="white",
        edgecolor="none",
        loc="lower right",
        bbox_to_anchor=(1.0, 1.015),
        ncol=2,
        handlelength=1.45,
        columnspacing=0.70,
        labelspacing=0.22,
        borderpad=0.25,
        fontsize=5.9,
    )


def _select_slug_segments(slug: pd.DataFrame, selected: list[str]) -> dict[str, int]:
    selected_segments: dict[str, int] = {}
    rows = []
    for well in selected:
        well_data = slug[slug["well"].eq(well)].copy()
        if well_data.empty:
            continue
        for segment, group in well_data.groupby("segment"):
            group = group.sort_values("elapsed_s")
            if len(group) < 10:
                continue
            observed = group["recovery_amplitude_norm"].to_numpy(float)
            predicted = group["semi_log_tail_prediction"].to_numpy(float)
            finite = np.isfinite(observed) & np.isfinite(predicted)
            if finite.sum() < 10:
                continue
            holdout = group["is_tail_holdout"].astype(bool).to_numpy() & finite
            score_mask = holdout if holdout.sum() >= 5 else finite
            rmse = float(np.sqrt(np.nanmean((observed[score_mask] - predicted[score_mask]) ** 2)))
            split = float(group["holdout_split_s"].iloc[0])
            rows.append({"well": well, "segment": int(segment), "rmse": rmse, "n": int(finite.sum()), "split": split})
    if not rows:
        return selected_segments
    report = pd.DataFrame(rows)
    for well, group in report.groupby("well"):
        best = group.sort_values(["rmse", "segment"]).iloc[0]
        selected_segments[well] = int(best["segment"])
    return selected_segments


def _plot_posterior_map(ax, posterior: pd.DataFrame, value_col: str, cmap, colorbar_label: str) -> None:
    plot = _plot_positions(posterior)
    aggregate = (
        posterior.groupby(["x_rel_km", "y_rel_km"], as_index=False)
        .agg({value_col: "mean", "peak_measured_drawdown_m": "max"})
        .copy()
    )
    x = aggregate["x_rel_km"].to_numpy(float)
    y = aggregate["y_rel_km"].to_numpy(float)
    z = aggregate[value_col].to_numpy(float)
    if len(aggregate) >= 4:
        gx, gy = np.meshgrid(
            np.linspace(x.min() - 0.4, x.max() + 0.4, 180),
            np.linspace(y.min() - 0.6, y.max() + 0.6, 180),
        )
        rbf = Rbf(x, y, z, function="linear", smooth=0.05)
        gz = rbf(gx, gy)
        ax.pcolormesh(gx, gy, gz, shading="auto", cmap=cmap, alpha=0.58)
        ax.contour(gx, gy, gz, levels=5, colors="0.25", linewidths=0.25, alpha=0.28)
    scatter = ax.scatter(
        plot["plot_x_rel_km"],
        plot["plot_y_rel_km"],
        c=plot[value_col],
        s=42,
        cmap=cmap,
        edgecolor="0.15",
        linewidth=0.45,
        zorder=3,
    )
    for row in plot.itertuples(index=False):
        if abs(row.plot_x_rel_km - row.x_rel_km) > 1e-9 or abs(row.plot_y_rel_km - row.y_rel_km) > 1e-9:
            ax.plot([row.x_rel_km, row.plot_x_rel_km], [row.y_rel_km, row.plot_y_rel_km], color="0.55", lw=0.3, alpha=0.7, zorder=2)
    _label_wells(ax, plot, fontsize=6.7)
    ax.set_xlabel("Relative easting (km)")
    ax.set_ylabel("Relative northing (km)")
    ax.set_aspect("equal", adjustable="box")
    ax.spines[["top", "right"]].set_visible(False)
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.047, pad=0.025)
    cbar.ax.set_title(colorbar_label, fontsize=7, pad=3)
    cbar.ax.tick_params(labelsize=7)


def _plot_predictive_scores(ax, metrics: pd.DataFrame) -> None:
    order = ["USGS calibrated", "Independent Theis", "Hard slug LOO", "Semi-log slug tail"]
    labels = ["Calibrated\nmodel", "Theis\nfit", "Hard slug\nLOO", "Slug tail\nholdout"]
    colors = [cmc.batlow(0.18), cmc.batlow(0.42), cmc.batlow(0.66), cmc.batlow(0.88)]
    rng = np.random.default_rng(20260517)
    for idx, (method, color) in enumerate(zip(order, colors)):
        values = metrics[metrics["method"].eq(method)]["normalized_rmse"].to_numpy(float)
        if len(values) == 0:
            continue
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(
            np.full(len(values), idx) + jitter,
            values,
            s=18,
            color=color,
            edgecolor="none",
            linewidth=0.0,
            alpha=0.78,
            zorder=3,
        )
        median = float(np.nanmedian(values))
        ax.plot([idx - 0.20, idx + 0.20], [median, median], color="0.1", lw=1.0)
    ax.axhline(1.0, color="0.25", lw=0.8, ls=":", label="null")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(labels, fontsize=6.8)
    ax.set_ylabel("RMSE / null RMSE")
    ax.set_ylim(0, min(4.5, max(1.2, float(metrics["normalized_rmse"].quantile(0.95)) * 1.15)))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")


def _label_wells(ax, posterior: pd.DataFrame, fontsize: float = 7.0) -> None:
    offsets = {
        "BD": (0.22, -0.62),
        "BS": (0.22, 0.50),
        "MD": (0.22, 0.56),
        "MS": (0.22, -0.58),
        "PINN": (0.13, 0.12),
        "WD": (0.22, -0.58),
        "WS": (0.22, 0.56),
    }
    for row in posterior.itertuples(index=False):
        dx, dy = offsets.get(row.well, (0.12, 0.12))
        x = getattr(row, "plot_x_rel_km", row.x_rel_km)
        y = getattr(row, "plot_y_rel_km", row.y_rel_km)
        ax.text(x + dx, y + dy, row.well, fontsize=fontsize, ha="left", va="center")


def _plot_positions(posterior: pd.DataFrame) -> pd.DataFrame:
    """Dodge co-located wells slightly so labels and markers remain readable."""

    offsets = {
        "BD": (-0.22, -0.18),
        "BS": (0.22, 0.18),
        "MD": (-0.22, 0.18),
        "MS": (0.22, -0.18),
        "WD": (-0.22, -0.18),
        "WS": (0.22, 0.18),
    }
    plot = posterior.copy()
    plot["plot_x_rel_km"] = plot["x_rel_km"]
    plot["plot_y_rel_km"] = plot["y_rel_km"]
    for well, (dx, dy) in offsets.items():
        mask = plot["well"].eq(well)
        plot.loc[mask, "plot_x_rel_km"] = plot.loc[mask, "x_rel_km"] + dx
        plot.loc[mask, "plot_y_rel_km"] = plot.loc[mask, "y_rel_km"] + dy
    return plot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out-dir", default="outputs/figures")
    args = parser.parse_args(argv)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    figure_field_context(args.processed_dir, args.out_dir)
    figure_response_validation(args.processed_dir, args.out_dir)
    figure_field_leave_one_out(args.processed_dir, args.out_dir)
    print("wrote=fig01_field_context,fig02_response_validation,fig09_field_leave_one_out")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

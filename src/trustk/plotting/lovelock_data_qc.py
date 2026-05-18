"""Generate diagnostic plots for the Lovelock pumping and slug-test data."""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from trustk.plotting.style import export_figure, journal_width, set_trustk_style
from trustk.validation.field_style import _load_slug_segments_or_fallback
from trustk.validation.lovelock import PUMP_START


PUMPING_WORKBOOK = "OFR2019_1133_DataRelease/Multi-WellPumpingTests/data/Multi-WellPumpingTest2_Data.xlsx"
DETECTION_THRESHOLD_M = 0.05 * 0.3048


def figure_lovelock_data_qc(
    processed_dir: str | Path = "data/processed",
    zip_path: str | Path = "field data/OFR2019_1133_DataRelease.zip",
    out_dir: str | Path = "outputs/figures",
) -> None:
    """Plot raw water-level changes beside USGS extracted drawdown and segmented slug events."""

    selected = ["BD", "MD", "WS"]
    colors = {well: cmc.batlow(x) for well, x in zip(selected, [0.18, 0.55, 0.86])}
    processed = Path(processed_dir)
    zip_file = Path(zip_path)

    raw = _load_raw_pumping_workbook(zip_file, selected)
    comparison = pd.read_csv(processed / "lovelock_bongo_model_comparison.csv", parse_dates=["datetime"])
    comparison = comparison[comparison["well"].isin(selected)].copy()
    comparison["time_days"] = (comparison["datetime"] - PUMP_START).dt.total_seconds() / 86400.0

    slug = _load_slug_segments_or_fallback(zip_file, processed / "slug_curves.csv")
    slug = slug[slug["well"].isin(selected)].copy()

    set_trustk_style()
    fig, axes = plt.subplots(2, 3, figsize=(journal_width(170), 5.6), sharex=False)
    for col, well in enumerate(selected):
        color = colors[well]
        ax = axes[0, col]
        raw_well = raw[raw["well"].eq(well)].copy()
        comp_well = comparison[comparison["well"].eq(well)].copy()
        ax.plot(raw_well["time_days"], raw_well["raw_level_change_m"], color=color, lw=0.75, alpha=0.78)
        ax.plot(comp_well["time_days"], comp_well["measured_drawdown_m"], "o-", color="0.15", ms=2.0, lw=0.8)
        ax.axhline(DETECTION_THRESHOLD_M, color="0.35", lw=0.65, ls=":")
        ax.axhline(0, color="0.82", lw=0.5)
        ax.set_title(well, fontsize=8, pad=3)
        ax.set_xlabel("Elapsed time (days)")
        if col == 0:
            ax.set_ylabel("Response (m)")
        ax.spines[["top", "right"]].set_visible(False)

        ax = axes[1, col]
        slug_well = slug[slug["well"].eq(well)].copy()
        for segment, segment_group in slug_well.groupby("segment"):
            segment_group = segment_group.sort_values("elapsed_s")
            stride = max(1, len(segment_group) // 110)
            ax.plot(
                segment_group.iloc[::stride]["plot_elapsed_s"],
                segment_group.iloc[::stride]["recovery_amplitude_norm"],
                "o",
                ms=1.6,
                color=color,
                alpha=0.34 + 0.13 * min(int(segment), 4),
                label=f"event {int(segment)}",
            )
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.08)
        ax.set_xlabel("Elapsed time (s)")
        if col == 0:
            ax.set_ylabel(r"Normalized $|H_w/H_0|$")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="upper right", handlelength=1.0)

    fig.legend(
        handles=[
            Line2D([0], [0], color=cmc.batlow(0.18), lw=0.9, label="raw water-level change"),
            Line2D([0], [0], marker="o", linestyle="-", color="0.15", ms=2.5, lw=0.8, label="USGS extracted drawdown"),
            Line2D([0], [0], linestyle=":", color="0.35", lw=0.7, label="0.05 ft threshold"),
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.51, 0.995),
        ncol=3,
        handlelength=1.5,
    )
    for label, ax in zip(["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"], axes.ravel()):
        ax.text(0.02, 0.95, label, transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.10, top=0.87, wspace=0.42, hspace=0.47)

    out_prefix = Path(out_dir) / "diagnostic_lovelock_data_qc"
    export_figure(fig, out_prefix)
    _qc_summary(raw, comparison, slug).to_csv(out_prefix.with_suffix(".csv"), index=False)
    plt.close(fig)


def _load_raw_pumping_workbook(zip_file: Path, selected: list[str]) -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(zip_file) as archive:
        workbook_bytes = archive.read(PUMPING_WORKBOOK)
    for well in selected:
        sheet = pd.read_excel(io.BytesIO(workbook_bytes), sheet_name=well, engine="openpyxl")
        sheet.columns = ["datetime", "raw_level_ft"]
        sheet = sheet.dropna().copy()
        sheet["datetime"] = pd.to_datetime(sheet["datetime"])
        baseline_window = sheet[
            (sheet["datetime"] >= PUMP_START - pd.Timedelta(days=1))
            & (sheet["datetime"] < PUMP_START)
        ]
        baseline_ft = float(baseline_window["raw_level_ft"].median())
        sheet["well"] = well
        sheet["time_days"] = (sheet["datetime"] - PUMP_START).dt.total_seconds() / 86400.0
        sheet["raw_level_change_m"] = (sheet["raw_level_ft"] - baseline_ft) * 0.3048
        rows.append(sheet[["well", "datetime", "time_days", "raw_level_change_m"]])
    raw = pd.concat(rows, ignore_index=True)
    return raw[(raw["time_days"] >= -1.0) & (raw["time_days"] <= 8.0)].reset_index(drop=True)


def _qc_summary(raw: pd.DataFrame, comparison: pd.DataFrame, slug: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for well in sorted(raw["well"].unique()):
        raw_well = raw[raw["well"].eq(well)]
        comp_well = comparison[comparison["well"].eq(well)]
        slug_well = slug[slug["well"].eq(well)]
        peak = float(comp_well["measured_drawdown_m"].max())
        rows.append(
            {
                "well": well,
                "raw_level_change_range_m": float(raw_well["raw_level_change_m"].max() - raw_well["raw_level_change_m"].min()),
                "usgs_peak_extracted_drawdown_m": peak,
                "usgs_peak_extracted_drawdown_ft": peak / 0.3048,
                "above_0_05_ft_detection_threshold": bool(peak >= DETECTION_THRESHOLD_M),
                "slug_event_count": int(slug_well["segment"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--zip", dest="zip_path", default="field data/OFR2019_1133_DataRelease.zip")
    parser.add_argument("--out-dir", default="outputs/figures")
    args = parser.parse_args(argv)
    figure_lovelock_data_qc(args.processed_dir, args.zip_path, args.out_dir)
    print("wrote=diagnostic_lovelock_data_qc")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

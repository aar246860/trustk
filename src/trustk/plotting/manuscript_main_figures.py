"""Curated main-text figures for the TRUST-K manuscript."""

from __future__ import annotations

import json
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from trustk.plotting.style import export_figure, journal_width, set_trustk_style


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
        clip_on=False,
    )


def plot_framework(out_prefix: str | Path) -> None:
    """Draw the TRUST-K conceptual workflow as a text-light icon figure."""

    set_trustk_style()
    panel_colors = [cmc.batlow(x) for x in [0.16, 0.30, 0.48, 0.66, 0.82]]
    fig, ax = plt.subplots(figsize=(journal_width(), journal_width(55)))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    x0 = 0.018
    panel_w = 0.174
    gap = 0.022
    y0 = 0.155
    panel_h = 0.690
    ink = "0.16"

    def xy(px: float, u: float, v: float) -> tuple[float, float]:
        return px + u * panel_w, y0 + v * panel_h

    def panel_patch(px: float, color) -> FancyBboxPatch:
        patch = FancyBboxPatch(
            (px, y0),
            panel_w,
            panel_h,
            boxstyle="round,pad=0.005,rounding_size=0.006",
            linewidth=0.75,
            edgecolor=ink,
            facecolor="white",
            alpha=0.98,
            zorder=1,
        )
        ax.add_patch(patch)
        ax.add_patch(
            FancyBboxPatch(
                (px + 0.006, y0 + 0.006),
                panel_w - 0.012,
                panel_h - 0.012,
                boxstyle="round,pad=0.004,rounding_size=0.005",
                linewidth=0.0,
                facecolor=color,
                alpha=0.035,
                zorder=1.1,
            )
        )
        return patch

    def wave(px: float, base: float, amp: float, phase: float = 0.0, n: int = 80) -> tuple[np.ndarray, np.ndarray]:
        u = np.linspace(0.0, 1.0, n)
        x = px + (0.02 + 0.96 * u) * panel_w
        y = y0 + (base + amp * np.sin(2.0 * np.pi * u + phase)) * panel_h
        return x, y

    def add_layer(px: float, upper: tuple[np.ndarray, np.ndarray], lower: tuple[np.ndarray, np.ndarray], color, alpha: float) -> None:
        xu, yu = upper
        xl, yl = lower
        coords = np.column_stack([np.r_[xu, xl[::-1]], np.r_[yu, yl[::-1]]])
        ax.add_patch(Polygon(coords, closed=True, facecolor=color, edgecolor=ink, linewidth=0.30, alpha=alpha, zorder=2))

    def draw_aquifer(px: float, color) -> None:
        rng = np.random.default_rng(20260518)
        panel_patch(px, color)
        top = wave(px, 0.815, 0.006, 0.3)
        b1 = wave(px, 0.665, 0.020, 1.5)
        b2 = wave(px, 0.395, 0.034, -0.5)
        b3 = wave(px, 0.190, 0.021, 0.9)
        bottom = (top[0], np.full_like(top[0], y0 + 0.035 * panel_h))
        add_layer(px, top, b1, cmc.batlow(0.55), 0.36)
        add_layer(px, b1, b2, cmc.lipari(0.35), 0.52)
        add_layer(px, b2, b3, cmc.batlow(0.73), 0.40)
        add_layer(px, b3, bottom, cmc.batlow(0.86), 0.47)
        ax.plot(*top, color=cmc.batlow(0.20), lw=0.75, zorder=5)
        for _ in range(90):
            u, v = rng.uniform(0.04, 0.96), rng.uniform(0.07, 0.78)
            ax.scatter(*xy(px, u, v), s=rng.uniform(1.0, 3.0), color=ink, alpha=0.12, linewidth=0, zorder=3)
        for u, depth in zip([0.23, 0.47, 0.66, 0.83], [0.24, 0.15, 0.35, 0.47], strict=True):
            x, ys = xy(px, u, 0.82)
            _, ye = xy(px, u, depth)
            w = panel_w * 0.028
            ax.add_patch(Rectangle((x - w / 2, ye), w, ys - ye, facecolor="white", edgecolor=ink, linewidth=0.55, zorder=6))
            ax.add_patch(Rectangle((x - w * 0.42, ye), w * 0.84, (ys - ye) * 0.18, facecolor=cmc.lipari(0.38), edgecolor=ink, linewidth=0.25, zorder=7))
            ax.add_patch(Rectangle((x - w * 0.66, ys - 0.020 * panel_h), w * 1.32, 0.032 * panel_h, facecolor="white", edgecolor=ink, linewidth=0.45, zorder=7))

    def draw_test_response(px: float, color) -> None:
        panel_patch(px, color)
        ax.add_patch(Rectangle(xy(px, 0.02, 0.06), panel_w * 0.96, panel_h * 0.60, facecolor=cmc.lipari(0.18), edgecolor="none", alpha=0.28, zorder=2))
        water_x, water_y = wave(px, 0.665, 0.006, 1.0)
        ax.plot(water_x, water_y, color=ink, lw=0.70, zorder=5)
        cx, cy = xy(px, 0.33, 0.50)
        for r in [0.13, 0.20, 0.28, 0.36]:
            th = np.linspace(-2.75, 0.25, 150)
            ex = cx + r * panel_w * np.cos(th)
            ey = cy + 0.68 * r * panel_h * np.sin(th)
            ax.plot(ex, ey, color=cmc.lipari(0.44), lw=0.45, ls=(0, (4, 3)), alpha=0.80, zorder=4)
        for u, depth in [(0.33, 0.34), (0.78, 0.26)]:
            x, ys = xy(px, u, 0.70)
            _, ye = xy(px, u, depth)
            w = panel_w * 0.040
            ax.add_patch(Rectangle((x - w / 2, ye), w, ys - ye, facecolor="white", edgecolor=ink, linewidth=0.55, zorder=6))
            ax.add_patch(Rectangle((x - w * 0.40, ye), w * 0.80, (ys - ye) * 0.26, facecolor=cmc.lipari(0.42), edgecolor="none", alpha=0.75, zorder=7))
        x, y = xy(px, 0.34, 0.70)
        ax.add_patch(Rectangle((x - 0.010 * panel_w, y), 0.020 * panel_w, 0.120 * panel_h, facecolor=cmc.batlow(0.28), edgecolor=ink, lw=0.45, zorder=7))
        ax.plot([x, x, x + 0.090 * panel_w], [y + 0.120 * panel_h, y + 0.180 * panel_h, y + 0.180 * panel_h], color=ink, lw=0.95, zorder=8)
        ax.plot([x + 0.088 * panel_w, x + 0.112 * panel_w], [y + 0.180 * panel_h, y + 0.155 * panel_h], color=ink, lw=0.95, zorder=8)
        for k in range(3):
            ax.plot([x + 0.135 * panel_w, x + 0.145 * panel_w], [y + (0.150 - 0.035 * k) * panel_h, y + (0.135 - 0.035 * k) * panel_h], color=cmc.lipari(0.48), lw=0.75, ls=(0, (3, 3)), zorder=8)
        tx, ty = xy(px, 0.88, 0.77)
        ax.add_patch(Rectangle((tx - 0.018 * panel_w, ty - 0.045 * panel_h), 0.036 * panel_w, 0.090 * panel_h, facecolor="white", edgecolor=ink, linewidth=0.55, zorder=7))
        ax.text(tx, ty, "×", ha="center", va="center", fontsize=5.5, color=ink, zorder=8)

    def draw_curve_fit(px: float, color) -> None:
        rng = np.random.default_rng(20260519)
        panel_patch(px, color)
        x0a, y0a = xy(px, 0.12, 0.18)
        x1a, y1a = xy(px, 0.86, 0.18)
        _, ytop = xy(px, 0.12, 0.80)
        ax.add_patch(FancyArrowPatch((x0a, y0a), (x1a, y1a), arrowstyle="-|>", mutation_scale=7, lw=0.65, color=ink, zorder=5))
        ax.add_patch(FancyArrowPatch((x0a, y0a), (x0a, ytop), arrowstyle="-|>", mutation_scale=7, lw=0.65, color=ink, zorder=5))
        t = np.linspace(0.0, 1.0, 22)
        y_curve = 0.67 * np.exp(-2.55 * t) + 0.18
        pts_y = y_curve + rng.normal(0.0, 0.045, t.size)
        pts_x = 0.18 + 0.64 * t
        ax.scatter(px + pts_x * panel_w, y0 + pts_y * panel_h, s=10, color=cmc.lipari(0.42), edgecolor=ink, linewidth=0.25, zorder=7)
        ts = np.linspace(0.0, 1.0, 120)
        ys = 0.67 * np.exp(-2.55 * ts) + 0.18
        ax.plot(px + (0.18 + 0.64 * ts) * panel_w, y0 + ys * panel_h, color=ink, lw=0.80, zorder=8)

    def draw_transformation_prior(px: float, color) -> None:
        rng = np.random.default_rng(20260520)
        panel_patch(px, color)
        xs = np.linspace(0.14, 0.86, 150)
        base = 0.55
        bell = base + 0.31 * np.exp(-((xs - 0.50) / 0.16) ** 2)
        ax.fill_between(px + xs * panel_w, y0 + base * panel_h, y0 + bell * panel_h, color=cmc.batlow(0.76), alpha=0.68, zorder=3)
        ax.plot(px + xs * panel_w, y0 + bell * panel_h, color=ink, lw=0.75, zorder=5)
        ax.plot([px + 0.14 * panel_w, px + 0.86 * panel_w], [y0 + base * panel_h, y0 + base * panel_h], color=ink, lw=0.55, zorder=5)
        for u in [0.34, 0.50, 0.66]:
            ax.plot([px + u * panel_w, px + u * panel_w], [y0 + base * panel_h, y0 + 0.86 * panel_h], color=ink, lw=0.45, ls=(0, (4, 3)), zorder=6)
        ax.plot([px + 0.23 * panel_w, px + 0.77 * panel_w], [y0 + 0.49 * panel_h, y0 + 0.49 * panel_h], color=cmc.lipari(0.43), lw=0.95, zorder=6)
        ax.plot([px + 0.23 * panel_w, px + 0.23 * panel_w], [y0 + 0.475 * panel_h, y0 + 0.505 * panel_h], color=cmc.lipari(0.43), lw=0.95, zorder=6)
        ax.plot([px + 0.77 * panel_w, px + 0.77 * panel_w], [y0 + 0.475 * panel_h, y0 + 0.505 * panel_h], color=cmc.lipari(0.43), lw=0.95, zorder=6)
        band_y = y0 + 0.275 * panel_h
        ax.add_patch(Rectangle((px + 0.11 * panel_w, band_y - 0.030 * panel_h), 0.78 * panel_w, 0.070 * panel_h, facecolor=cmc.lipari(0.20), edgecolor="none", alpha=0.22, zorder=2))
        ax.plot([px + 0.12 * panel_w, px + 0.88 * panel_w], [band_y, band_y], color=ink, lw=0.45, ls=(0, (4, 3)), zorder=5)
        pts_x = px + rng.uniform(0.13, 0.87, 24) * panel_w
        pts_y = band_y + rng.normal(0.0, 0.070 * panel_h, 24)
        ax.scatter(pts_x, pts_y, s=10, color=cmc.lipari(0.43), edgecolor="white", linewidth=0.20, zorder=7)

    def draw_soft_model(px: float, color) -> None:
        rng = np.random.default_rng(20260521)
        panel_patch(px, color)
        top_x, top_y = wave(px, 0.86, 0.006, 0.3)
        ax.plot(top_x, top_y, color=cmc.batlow(0.18), lw=0.75, zorder=6)
        err_colors = [cmc.lipari(v) for v in np.linspace(0.25, 0.78, 5)]
        for u, yy, ec in zip(np.linspace(0.16, 0.84, 5), [0.70, 0.69, 0.67, 0.68, 0.66], err_colors, strict=True):
            x, y = xy(px, u, yy)
            ax.plot([x, x], [y - 0.090 * panel_h, y + 0.090 * panel_h], color=ec, lw=0.80, zorder=6)
            ax.plot([x - 0.014 * panel_w, x + 0.014 * panel_w], [y - 0.090 * panel_h, y - 0.090 * panel_h], color=ec, lw=0.80, zorder=6)
            ax.plot([x - 0.014 * panel_w, x + 0.014 * panel_w], [y + 0.090 * panel_h, y + 0.090 * panel_h], color=ec, lw=0.80, zorder=6)
            ax.scatter([x], [y], s=16, color=ec, edgecolor=ink, linewidth=0.20, zorder=7)
        ax.add_patch(FancyArrowPatch(xy(px, 0.50, 0.56), xy(px, 0.50, 0.44), arrowstyle="simple", mutation_scale=18, color=cmc.lipari(0.45), alpha=0.90, zorder=5))
        left, right = px + 0.02 * panel_w, px + 0.98 * panel_w
        bottom, top = y0 + 0.04 * panel_h, y0 + 0.37 * panel_h
        xx = np.linspace(0, 1, 80)
        yy = np.linspace(0, 1, 55)
        X, Y = np.meshgrid(xx, yy)
        field = 0.50 * np.sin(5.1 * X + 1.4 * Y) + 0.30 * np.cos(4.2 * Y - 0.5) + 0.70 * X - 0.20 * Y
        ax.imshow(field, extent=(left, right, bottom, top), origin="lower", cmap=cmc.batlow, aspect="auto", alpha=0.92, zorder=2)
        gx = np.linspace(left, right, field.shape[1])
        gy = np.linspace(bottom, top, field.shape[0])
        ax.contour(gx, gy, field, levels=8, colors="0.22", linewidths=0.18, alpha=0.40, zorder=4)
        ax.contour(gx, gy, field, levels=[np.percentile(field, 52)], colors="0.12", linewidths=0.70, linestyles="--", zorder=5)
        ax.plot([left, left, right, right, left], [bottom, top, top, bottom, bottom], color=ink, lw=0.55, zorder=7)
        map_top_x = np.linspace(left, right, 80)
        map_top_y = top - 0.020 * panel_h * np.sin(np.linspace(0, 2 * np.pi, 80) + 0.5)
        ax.plot(map_top_x, map_top_y, color=ink, lw=0.55, zorder=8)

    drawers = [draw_aquifer, draw_test_response, draw_curve_fit, draw_transformation_prior, draw_soft_model]
    for i, draw in enumerate(drawers):
        px = x0 + i * (panel_w + gap)
        draw(px, panel_colors[i])
        if i < len(drawers) - 1:
            next_px = x0 + (i + 1) * (panel_w + gap)
            ymid = y0 + panel_h * 0.50
            ax.add_patch(
                FancyArrowPatch(
                    (px + panel_w + 0.008, ymid),
                    (next_px - 0.008, ymid),
                    arrowstyle="-|>",
                    mutation_scale=13.0,
                    linewidth=0.95,
                    color="0.25",
                    shrinkA=0.0,
                    shrinkB=0.0,
                    zorder=20,
                )
            )

    fig.subplots_adjust(left=0.010, right=0.990, bottom=0.045, top=0.955)
    export_figure(fig, out_prefix)
    plt.close(fig)
    Path(out_prefix).with_suffix(".json").write_text(
        json.dumps(
            {
                "figure": Path(out_prefix).name,
                "palette": "official cmcrameri cmc.batlow",
                "overlap_check": "Text-free five-panel workflow; arrows are separated from panel borders; no legend or in-panel labels are used.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def plot_numerical_benchmark(
    theis_csv: str | Path,
    slug_csv: str | Path,
    mapping_csv: str | Path,
    verification_json: str | Path,
    out_prefix: str | Path,
) -> None:
    """Show solver verification as one-to-one numerical/reference scatter plots."""

    set_trustk_style()
    theis = pd.read_csv(theis_csv)
    theis_window = theis[(theis["radius_m"] >= 1.0) & (theis["radius_m"] <= 350.0)].copy()
    slug = pd.read_csv(slug_csv)
    _ = mapping_csv, verification_json

    slug_window = slug[slug["reference_normalized_head"] > 1.0e-4].copy()
    pumping_l2 = _relative_l2(
        theis_window["fv_drawdown_m"].to_numpy(float),
        theis_window["theis_drawdown_m"].to_numpy(float),
    )
    slug_l2 = _relative_l2(
        slug_window["fv_normalized_head"].to_numpy(float),
        slug_window["reference_normalized_head"].to_numpy(float),
    )
    display_n = min(len(theis_window), len(slug_window))
    theis_display = _evenly_sample_rows(theis_window, display_n)
    slug_display = _evenly_sample_rows(slug_window, display_n)

    fig, axes = plt.subplots(1, 2, figsize=(journal_width(), journal_width(66)))

    ax = axes[0]
    _plot_one_to_one(
        ax,
        reference=theis_display["theis_drawdown_m"].to_numpy(float),
        numerical=theis_display["fv_drawdown_m"].to_numpy(float),
        color=cmc.batlow(0.22),
        xlabel="reference drawdown (m)",
        ylabel="finite-volume drawdown (m)",
        log_scale=False,
    )
    _panel_label_inside(ax, "(a)")

    ax = axes[1]
    _plot_one_to_one(
        ax,
        reference=slug_display["reference_normalized_head"].to_numpy(float),
        numerical=slug_display["fv_normalized_head"].to_numpy(float),
        color=cmc.batlow(0.72),
        xlabel=r"reference recovery, $H_w/H_0$",
        ylabel=r"finite-volume recovery, $H_w/H_0$",
        log_scale=True,
    )
    _panel_label_inside(ax, "(b)")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.19, top=0.93, wspace=0.08)
    export_figure(fig, out_prefix)
    plt.close(fig)
    Path(out_prefix).with_suffix(".json").write_text(
        json.dumps(
            {
                "figure": Path(out_prefix).name,
                "palette": "official cmcrameri cmc.batlow",
                "panels": {
                    "a": "finite-volume pumping drawdown versus reference drawdown",
                    "b": "finite-volume slug recovery versus reference recovery",
                },
                "display_point_count_per_panel": display_n,
                "full_window_point_count": {
                    "pumping": int(len(theis_window)),
                    "slug": int(len(slug_window)),
                },
                "relative_l2_error": {
                    "pumping": pumping_l2,
                    "slug": slug_l2,
                },
                "overlap_check": "No legend used; panel labels placed inside clear upper-left plot corners away from tick labels and data.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _evenly_sample_rows(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    """Return evenly spaced rows for visual balance without changing metrics."""

    if count <= 0:
        return frame.iloc[[]].copy()
    if count >= len(frame):
        return frame.copy()
    indices = np.linspace(0, len(frame) - 1, count, dtype=int)
    return frame.iloc[indices].copy()


def _relative_l2(numerical: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(numerical - reference) / np.linalg.norm(reference))


def _panel_label_inside(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.035,
        0.90,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.0},
        clip_on=False,
        zorder=5,
    )


def _plot_one_to_one(
    ax: plt.Axes,
    reference: np.ndarray,
    numerical: np.ndarray,
    color,
    xlabel: str,
    ylabel: str,
    log_scale: bool,
) -> None:
    values = np.concatenate([reference, numerical])
    low, high = _axis_limits(values, log_scale=log_scale)
    ax.plot([low, high], [low, high], color="0.25", lw=0.8, ls="--", zorder=1)
    ax.scatter(reference, numerical, s=18, color=color, edgecolor="white", linewidth=0.35, alpha=0.88, zorder=2)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.labelpad = 2.0
    ax.yaxis.labelpad = 2.0
    ax.tick_params(axis="both", which="both", pad=1.5)
    ax.grid(True, color="0.88", linewidth=0.35, zorder=0)


def _axis_limits(values: np.ndarray, log_scale: bool) -> tuple[float, float]:
    finite = values[np.isfinite(values) & (values > 0)]
    if log_scale:
        low_log = float(np.floor(np.log10(finite.min()) * 2.0) / 2.0)
        high_log = float(np.ceil(np.log10(finite.max()) * 2.0) / 2.0)
        return 10.0**low_log, 10.0**high_log
    span = float(finite.max() - finite.min())
    pad = 0.08 * span if span > 0 else 0.1 * float(finite.max())
    return float(finite.min() - pad), float(finite.max() + pad)


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    out = root / "outputs" / "figures"
    plot_framework(out / "fig01_trustk_framework")
    plot_numerical_benchmark(
        out / "fig03_benchmark_theis.csv",
        out / "fig04_benchmark_slug.csv",
        out / "fig05_random_field_mapping.csv",
        root / "outputs" / "reports" / "numerical_verification.json",
        out / "fig02_numerical_benchmark",
    )


if __name__ == "__main__":
    main()

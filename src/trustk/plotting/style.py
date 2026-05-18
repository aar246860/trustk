"""Shared plotting style for TRUST-K figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


MM_TO_IN = 1.0 / 25.4


def set_trustk_style() -> None:
    """Apply a compact journal-style Matplotlib theme."""

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.0,
            "lines.markersize": 4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
        }
    )


def journal_width(width_mm: float = 170.0) -> float:
    return width_mm * MM_TO_IN


def export_figure(fig, out_prefix: str | Path) -> None:
    out = Path(out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".svg"))
    fig.savefig(out.with_suffix(".png"))

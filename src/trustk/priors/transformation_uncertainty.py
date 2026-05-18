"""Transformation-uncertainty summaries from QC-screened residuals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_transformation_uncertainty(
    qc: pd.DataFrame,
    *,
    accepted_classes: tuple[str, ...] = ("pass",),
    min_cases: int = 5,
) -> pd.DataFrame:
    """Estimate method-level transformation bias and scatter from QC-screened cases."""

    required = {"method", "qc_class", "log_residual"}
    missing = required.difference(qc.columns)
    if missing:
        raise ValueError(f"qc table is missing required columns: {sorted(missing)}")

    filtered = qc[qc["qc_class"].isin(accepted_classes)].copy()
    rows = []
    for method, group in filtered.groupby("method", sort=False):
        values = group["log_residual"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        q05 = float(np.quantile(values, 0.05))
        q50 = float(np.quantile(values, 0.50))
        q95 = float(np.quantile(values, 0.95))
        rows.append(
            {
                "method": method,
                "accepted_qc_classes": ",".join(accepted_classes),
                "n_cases": int(len(values)),
                "is_sufficient": bool(len(values) >= min_cases),
                "mean_log_residual": mean,
                "sd_log_residual": sd,
                "median_log_residual": q50,
                "q05_log_residual": q05,
                "q95_log_residual": q95,
                "bias_factor_c": float(np.exp(mean)),
                "target_correction_factor": float(np.exp(-mean)),
                "scatter_factor": float(np.exp(sd)),
                "q05_factor": float(np.exp(q05)),
                "q95_factor": float(np.exp(q95)),
            }
        )
    return pd.DataFrame(rows)

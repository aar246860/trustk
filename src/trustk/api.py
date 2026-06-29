"""Public TRUST-K helpers for converting test estimates into soft observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, log
from statistics import NormalDist
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class MethodTransformationPrior:
    """Method-level transformation prior learned from QC-pass synthetic cases."""

    method: str
    mean_log_residual: float
    sd_log_residual: float
    n_cases: int
    source: str = "TRUST-K formal engineering-practice synthetic population"

    @property
    def bias_factor(self) -> float:
        """Multiplicative bias factor for the conventional estimate."""

        return exp(self.mean_log_residual)

    @property
    def target_correction_factor(self) -> float:
        """Factor that maps a conventional estimate to the support target median."""

        return exp(-self.mean_log_residual)


@dataclass(frozen=True)
class SoftConductivityObservation:
    """Corrected soft observation for one test-derived hydraulic conductivity."""

    method: str
    k_estimate_m_s: float
    k_soft_m_s: float
    k_soft_lower_m_s: float
    k_soft_upper_m_s: float
    interval_level: float
    log_residual_mean: float
    log_residual_sd: float
    bias_factor: float
    target_correction_factor: float

    def to_dict(self) -> dict[str, float | str]:
        """Return a flat dictionary suitable for tables or CSV export."""

        return asdict(self)


FORMAL_JOINT_PRIOR_SOURCE = "TRUST-K formal 4096-case joint K-Ss-b synthetic population"

MANUSCRIPT_METHOD_PRIORS: dict[str, MethodTransformationPrior] = {
    "pumping": MethodTransformationPrior(
        method="pumping",
        mean_log_residual=0.26345401414389874,
        sd_log_residual=0.39232099187009656,
        n_cases=4025,
        source=FORMAL_JOINT_PRIOR_SOURCE,
    ),
    "slug_bouwer_rice": MethodTransformationPrior(
        method="slug_bouwer_rice",
        mean_log_residual=0.798700815400023,
        sd_log_residual=2.8010164229597403,
        n_cases=4096,
        source=FORMAL_JOINT_PRIOR_SOURCE,
    ),
}

_METHOD_ALIASES = {
    "pumping": "pumping",
    "pump": "pumping",
    "pumping_test": "pumping",
    "cooper_jacob": "pumping",
    "cooper-jacob": "pumping",
    "theis": "pumping",
    "slug": "slug_bouwer_rice",
    "slug_test": "slug_bouwer_rice",
    "slug_bouwer_rice": "slug_bouwer_rice",
    "micro_water": "slug_bouwer_rice",
    "microwater": "slug_bouwer_rice",
    "bouwer_rice": "slug_bouwer_rice",
    "bouwer-rice": "slug_bouwer_rice",
}


def correct_conductivity(
    method: str,
    k_estimate_m_s: float,
    *,
    interval_level: float = 0.95,
    priors: Mapping[str, MethodTransformationPrior] | None = None,
) -> SoftConductivityObservation:
    """Convert one conventional conductivity estimate into a TRUST-K soft observation.

    The method-level prior follows the manuscript residual definition
    ``r = log(K_hat) - log(K_star)``. Therefore the support-target median is
    ``K_hat * exp(-mean_log_residual)`` and the interval width is controlled by
    the residual scatter.
    """

    if k_estimate_m_s <= 0.0:
        raise ValueError("k_estimate_m_s must be positive")
    if not 0.0 < interval_level < 1.0:
        raise ValueError("interval_level must be between 0 and 1")

    prior_table = priors or MANUSCRIPT_METHOD_PRIORS
    key = normalize_method(method)
    if key not in prior_table:
        raise ValueError(f"no TRUST-K transformation prior is available for method {method!r}")

    prior = prior_table[key]
    z = NormalDist().inv_cdf(0.5 + interval_level / 2.0)
    log_soft = log(float(k_estimate_m_s)) - prior.mean_log_residual
    lower = exp(log_soft - z * prior.sd_log_residual)
    upper = exp(log_soft + z * prior.sd_log_residual)
    return SoftConductivityObservation(
        method=key,
        k_estimate_m_s=float(k_estimate_m_s),
        k_soft_m_s=exp(log_soft),
        k_soft_lower_m_s=lower,
        k_soft_upper_m_s=upper,
        interval_level=float(interval_level),
        log_residual_mean=prior.mean_log_residual,
        log_residual_sd=prior.sd_log_residual,
        bias_factor=prior.bias_factor,
        target_correction_factor=prior.target_correction_factor,
    )


def correct_table(
    table: pd.DataFrame,
    *,
    method_col: str = "method",
    k_col: str = "k_estimate_m_s",
    interval_level: float = 0.95,
    priors: Mapping[str, MethodTransformationPrior] | None = None,
) -> pd.DataFrame:
    """Apply TRUST-K method-level correction to every row in a table."""

    required = {method_col, k_col}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"table is missing required columns: {sorted(missing)}")

    rows = []
    for _, row in table.iterrows():
        corrected = correct_conductivity(
            str(row[method_col]),
            float(row[k_col]),
            interval_level=interval_level,
            priors=priors,
        ).to_dict()
        rows.append(corrected)
    corrected_table = pd.DataFrame(rows)
    return pd.concat([table.reset_index(drop=True), corrected_table.add_prefix("trustk_")], axis=1)


def normalize_method(method: str) -> str:
    """Normalize common aquifer-test method labels to TRUST-K method names."""

    key = method.strip().lower().replace(" ", "_")
    if key not in _METHOD_ALIASES:
        raise ValueError(f"unsupported method {method!r}; use one of {sorted(_METHOD_ALIASES)}")
    return _METHOD_ALIASES[key]

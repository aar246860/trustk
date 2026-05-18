# TRUST-K Figure Governance

Updated: 2026-05-17

This note keeps the figure set aligned with `TRUST_K_Figure_Table_Design_Constitution.md`.
The manuscript should not include every exploratory figure. Main-text figures must each defend one scientific claim in the TRUST-K chain:

```text
response -> engineering interpretation -> support target -> residual -> p(c, sigma | Pi) -> assimilation -> prediction
```

## Main-Text Figure Target

The cleaned manuscript now uses 9 main-text figures. The active figure folder is `manuscript/figures_main/`; older exploratory figures remain available for traceability but are not included in `main.tex`.

| Figure | File in `manuscript/figures_main/` | Constitutional role |
|---|---|---|
| Figure 1 | `fig01_trustk_framework.pdf` | TRUST-K conceptual framework |
| Figure 2 | `fig02_random_field_mapping.pdf` | random-field generation and Cartesian-to-polar mesh mapping check |
| Figure 3 | `fig02_numerical_benchmark.pdf` | two-panel one-to-one solver verification for pumping and slug-test responses |
| Figure 4 | `fig03_support_residuals.pdf` | synthetic support targets and apparent \(K\) residuals |
| Figure 5 | `fig04_fit_quality_qc.pdf` | fit-quality QC before residuals are used as transformation evidence |
| Figure 6 | `fig05_conditional_prior_surfaces.pdf` | conditional \(\ln c_m(\Pi)\) and \(\sigma_m(\Pi)\) surfaces |
| Figure 7 | `fig06_holdout_validation.pdf` | hold-out validation of conditional residual intervals |
| Figure 8 | `fig07_spatial_assimilation.pdf` | latent-\(K\) assimilation against hard and method-constant baselines |
| Figure 9 | `fig08_field_leave_one_out.pdf` | Lovelock field-style predictive validation |

## Current Output Triage

The following outputs are no longer main-text figures:

| Existing output | Main-text decision | Reason |
|---|---|---|
| `fig01_field_context` | excluded | merged conceptually into final field validation |
| `fig02_response_validation` | excluded | superseded by `fig09_field_leave_one_out` |
| `fig03_benchmark_theis` | excluded | merged into `fig02_numerical_benchmark` |
| `fig04_benchmark_slug` | excluded | merged into `fig02_numerical_benchmark` |
| `fig05_random_field_mapping` | superseded | replaced by the main-text `fig02_random_field_mapping` figure |
| `fig06_dimensionless_registry` | excluded | design registry/readiness diagnostic |
| `fig07_solver_settings` | excluded | solver-readiness diagnostic |
| `fig08_synthetic_pilot_responses` | excluded | small pilot health check only |
| `fig11_transformation_uncertainty_prior` | excluded | interim method-level prior superseded by conditional prior surfaces |
| `fig08_assimilation_baseline_comparison` | excluded | support-target baseline comparison only; spatial assimilation is the main result |
| `diagnostic_lovelock_data_qc` | excluded | field-data QC diagnostic, not a main result |

## Immediate Rule For The 2000-Case Expansion

Do not add more one-off response-curve figures. The 2000-case run should feed only these deliverables:

1. Formal synthetic population table with raw curves and support targets.
2. Engineering-practice residual table using Cooper-Jacob and semi-log slug windows.
3. Conditional response surfaces for `ln c_m(Pi)` and `sigma_m(Pi)`.
4. Hold-out calibration figure for residual intervals.
5. Assimilation validation figure against baselines.

All intermediate solver-readiness and pilot figures should remain supplementary unless they are merged into the 9-figure main-text structure above.

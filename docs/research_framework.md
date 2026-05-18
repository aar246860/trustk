# TRUST-K Research Framework

Last updated: 2026-05-17

## Core Claim

TRUST-K does not ask whether slug-test `K` or pumping-test `K` is the true value.

The paper asks a different question:

> How should method-derived hydraulic conductivity estimates be converted into soft observations of a latent model-scale `K(x,y)` field when each test has a different support volume and transformation uncertainty?

## Paper Logic

The manuscript should follow this order.

### 1. Synthetic Truth Layer

Generate known two-dimensional latent conductivity fields:

```text
K(x,y)
```

Because the field is known, this layer can define the true support-scale target:

```text
K*_slug
K*_pump
```

This is the only layer where transformation residuals can be measured rigorously.

### 2. Forward Numerical Model Layer

Run pumping and slug-test simulations on the same latent field.

The current implementation starts with the pumping solver:

```text
2D horizontal confined aquifer
polar finite-volume mesh
implicit time stepping
constant-rate pumping
fixed zero-drawdown outer boundary
```

The solver is benchmarked against the Theis analytical solution before it is allowed to generate synthetic TRUST-K training cases.

### 3. Conventional Interpretation Layer

Interpret the simulated responses using conventional test formulas:

```text
simulated pumping response -> K_hat_pump
simulated slug response    -> K_hat_slug
```

These interpreted values are intentionally treated as method outputs, not true hydraulic conductivity.

### 4. Transformation-Uncertainty Layer

Compute method residuals:

```text
r_m = log(K_hat_m) - log(K*_m)
```

Then estimate:

```text
log c_m(Pi) = E[r_m | Pi]
sigma_m(Pi) = SD[r_m | Pi]
```

This is the main TRUST-K product.

### 5. Soft-Data Assimilation Layer

Convert test-derived `K` into soft observations:

```text
log K_obs_m = H_m[log K(x)] + log c_m(Pi) + e_m
```

This is where TRUST-K differs from hard-data workflows that directly assimilate slug or pumping `K`.

### 6. Field Demonstration Layer

Use the USGS Lovelock data only after the synthetic numerical framework is verified.

The field data can show that:

- slug and pumping observations are not interchangeable;
- co-located wells can have very different slug-test `K`;
- direct hard transfer of slug-test `K` can fail to predict pumping/recovery response.

The field data cannot provide the true latent `K(x,y)`, so it should not be used as the primary proof of transformation uncertainty.

## Why The Field Case Was Inspected First

The Lovelock data were inspected early only to confirm that a real field demonstration exists and that the data contain both slug and pumping/recovery records.

The main paper should not be organized around that field case. It should be organized around the synthetic numerical framework, with Lovelock as the final demonstration.

## Current Numerical Status

Completed:

- 2D polar finite-volume pumping solver.
- Theis benchmark.
- Grid-independence, time-step refinement, and outer-boundary sensitivity report.
- Wellbore-storage slug solver.
- Quasi-steady slug-recovery benchmark.
- Cartesian `ln K(x,y)` random-field generator.
- Cartesian-to-polar mesh mapping verification with nonzero angular heterogeneity.
- Dimensionless case registry with separate common, pumping-specific, and slug-specific controls.
- Conversion from Pi-space cases to dimensional solver settings and random-field grid readiness.
- Paired pumping/slug pilot responses for 12 representative ready cases.
- Baseline synthetic population for all 82 ready cases, including method-specific support targets, conventional interpretations, and log residuals.
- Fit-quality QC for conventional pumping and slug interpretations.
- First QC-screened method-level transformation uncertainty prior.
- Figure governance aligned with the TRUST-K constitution.
- Engineering-practice interpretation baseline:
  - Cooper-Jacob late-time straight-line pumping interpretation.
  - Semi-log slug recovery interpretation.
  - Explicit fit-window tracking and QC screening.
- Formal 2000-case Pi-space design.
- Formal 1728-case directly solver-ready synthetic population under `N_cart <= 1024`.
- Formal engineering-practice residual and QC tables.
- Conditional prior generator over dimensionless controls.
- Hold-out validation of residual mean and interval calibration.
- First baseline comparison among hard interpretation, method-constant correction, and conditional TRUST-K correction at the support-target level.
- Spatial latent-field assimilation experiment comparing hard interpretation, method-constant soft data, and TRUST-K conditional soft data on a known synthetic `K(x,y)` field.
- Lovelock field-style predictive validation:
  - leave-one-well-out pumping/recovery prediction from slug-test `K`;
  - early-to-tail slug-recovery prediction;
  - apparent posterior `K` and uncertainty at observed wells only.

Next:

- Separate strategy for the 272 high-grid small-correlation-length formal cases.
- Decide whether warning pumping cases deserve a separate residual model.
- Convert completed Figure 9 text into the final results/discussion narrative without claiming true field `K` recovery.

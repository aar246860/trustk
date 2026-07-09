# TRUST-K

TRUST-K stands for:

```text
Transformation-uncertainty and Representative-volume Unified Soft-data framework for hydraulic Conductivity
```

This repository is the public source package for the manuscript
"TRUST-K: A Transformation-Uncertainty Framework for Hydraulic Conductivity
Inference". It provides a small Python API for applying the manuscript's
method-level transformation priors to conventional aquifer-test hydraulic
conductivity estimates.

The package returns soft observations of support-scale conductivity. It does
not claim that a corrected field value is the unique true hydraulic
conductivity.

## Installation

Install directly from GitHub:

```bash
python -m pip install "git+https://github.com/aar246860/trustk.git"
```

For local development:

```bash
git clone https://github.com/aar246860/trustk.git
cd trustk
python -m pip install -e ".[dev]"
pytest
```

## Public API Example

```python
import pandas as pd
from trustk import correct_table

tests = pd.DataFrame(
    {
        "method": ["cooper-jacob", "bouwer-rice"],
        "k_estimate_m_s": [1.0e-5, 2.0e-5],
    }
)

soft_observations = correct_table(tests)
print(
    soft_observations[
        [
            "trustk_method",
            "trustk_k_soft_m_s",
            "trustk_k_soft_lower_m_s",
            "trustk_k_soft_upper_m_s",
        ]
    ]
)
```

## Included Priors

The default priors in this source release are the formal joint TRUST-K priors
reported in the revised manuscript:

| Method alias | Internal method | Mean log residual | Standard deviation | Cases |
| --- | --- | ---: | ---: | ---: |
| `cooper-jacob` | `pumping` | 0.26345401414389874 | 0.39232099187009656 | 4025 |
| `bouwer-rice` | `slug_bouwer_rice` | 0.798700815400023 | 2.8010164229597403 | 4096 |

The residual definition is
`r = log(K_hat) - log(K_star)`, so the support-scale median returned by
`correct_conductivity` is `K_hat * exp(-mean_log_residual)`.

## Scope

This public repository contains the lightweight user-facing correction API,
documentation, and tests. The larger manuscript workspace contains raw field
archives, numerical experiments, figure-generation scripts, and intermediate
analysis products that are not included here because of size and data-release
structure.

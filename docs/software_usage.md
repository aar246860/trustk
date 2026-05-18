# TRUST-K Public API Usage

This document records the minimal software interface described in the
manuscript. The API converts conventional aquifer-test hydraulic-conductivity
estimates into method-dependent soft observations.

## Install for Local Use

TRUST-K is currently distributed through the public GitHub repository:

```bash
python -m pip install "git+https://github.com/aar246860/trustk.git"
```

For local source development:

```powershell
git clone https://github.com/aar246860/trustk.git
cd trustk
python -m pip install -e .
```

## Convert Existing Aquifer-Test Estimates

```python
import pandas as pd
from trustk import correct_table

tests = pd.DataFrame(
    {
        "method": ["cooper-jacob", "bouwer-rice"],
        "k_estimate_m_s": [1.0e-5, 2.0e-5],
    }
)

soft = correct_table(tests)
print(soft)
```

The returned columns include:

- `trustk_k_soft_m_s`: TRUST-K median soft observation.
- `trustk_k_soft_lower_m_s`: lower interval bound.
- `trustk_k_soft_upper_m_s`: upper interval bound.
- `trustk_bias_factor`: multiplicative transformation bias factor.
- `trustk_target_correction_factor`: factor applied to the conventional estimate.

## Interpretation

The output is not a unique true hydraulic conductivity. It is a soft
observation that carries the method-level transformation uncertainty learned
from the manuscript's QC-pass synthetic population. Users working outside the
tested design range should augment the synthetic population or retrain the
conditional prior before using the result for design decisions.

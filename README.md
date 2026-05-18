# TRUST-K

TRUST-K stands for:

```text
Transformation-uncertainty and Representative-volume Unified Soft-data framework for hydraulic Conductivity
```

The project develops a reproducible analysis and manuscript package for treating hydraulic conductivity inferred from aquifer tests as method-dependent soft observations of a latent hydraulic-conductivity field.

## Main Field Data

The primary field data are the USGS Lovelock, Nevada data release in:

```text
field data/OFR2019_1133_DataRelease.zip
```

The raw data archive is not stored in this GitHub repository because it is large.
Place the downloaded USGS archive at the path above before running field-data
inventory or extraction commands. The data include slug tests, single-well
pumping, multi-well pumping, pumping recovery, and existing USGS analyses.

## Basic Commands

TRUST-K is currently released as an open-source GitHub source package:

```bash
python -m pip install "git+https://github.com/aar246860/trustk.git"
```

For local development from the source repository:

```powershell
git clone https://github.com/aar246860/trustk.git
cd trustk
python -m pip install -e .
```

Create a field-data inventory:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src)
python -m trustk.data.usgs_inventory --zip "field data/OFR2019_1133_DataRelease.zip" --out outputs/reports/usgs_inventory.json
```

## Public API Example

The manuscript constants can convert conventional aquifer-test conductivity
estimates into TRUST-K soft observations:

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
print(soft_observations[
    [
        "trustk_method",
        "trustk_k_soft_m_s",
        "trustk_k_soft_lower_m_s",
        "trustk_k_soft_upper_m_s",
    ]
])
```

The output is a soft observation with a median and interval, not a claim that
the corrected value is the unique true hydraulic conductivity.

## Local Test

```powershell
pytest -q
```

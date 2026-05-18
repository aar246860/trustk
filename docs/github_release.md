# TRUST-K GitHub Release Checklist

TRUST-K is currently released as an open-source GitHub source package at:

```text
https://github.com/aar246860/trustk
```

## Build and Check

```powershell
python -m pip install --upgrade build twine
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue
python -m build
python -m twine check dist/*
```

The build artifacts are useful for local verification and future archive
releases. They are not uploaded to a package index in the current release plan.

## Install from GitHub

```powershell
python -m pip install "git+https://github.com/aar246860/trustk.git"
```

For local development:

```powershell
git clone https://github.com/aar246860/trustk.git
cd trustk
python -m pip install -e .
```

## Smoke Test

```powershell
python - <<'PY'
import pandas as pd
from trustk import correct_table

tests = pd.DataFrame(
    {
        "method": ["cooper-jacob", "bouwer-rice"],
        "k_estimate_m_s": [1.0e-5, 2.0e-5],
    }
)
print(correct_table(tests)[["trustk_method", "trustk_k_soft_m_s"]])
PY
```

## Notes

The public manuscript should describe TRUST-K as a GitHub source release unless
a future package-index release is completed and verified.

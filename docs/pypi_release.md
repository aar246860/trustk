# TRUST-K PyPI Release Checklist

The package name targeted for the first public release is `trustk`.

## Build and Check

```powershell
python -m pip install --upgrade build twine
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue
python -m build
python -m twine check dist/*
```

## Upload to TestPyPI

Set a TestPyPI API token first:

```powershell
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "pypi-..."
python -m twine upload --repository testpypi dist/*
```

Check installation from TestPyPI:

```powershell
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple trustk
```

## Upload to PyPI

Set the production PyPI API token:

```powershell
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "pypi-..."
python -m twine upload dist/*
```

## Smoke Test After Release

```powershell
python -m pip install trustk
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

Do not upload the release until the repository URL in `pyproject.toml` exists
and the manuscript or supplementary material points to the same release.

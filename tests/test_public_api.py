import math

import pandas as pd
import pytest

from trustk import correct_conductivity, correct_table


def test_correct_conductivity_uses_residual_definition():
    out = correct_conductivity("pumping", 1.0e-5)

    assert out.method == "pumping"
    assert out.k_soft_m_s == pytest.approx(1.0e-5 * math.exp(-0.17))
    assert out.bias_factor == pytest.approx(math.exp(0.17))
    assert out.k_soft_lower_m_s < out.k_soft_m_s < out.k_soft_upper_m_s


def test_correct_table_accepts_common_aliases():
    table = pd.DataFrame(
        {
            "method": ["cooper-jacob", "bouwer-rice"],
            "k_estimate_m_s": [1.0e-5, 2.0e-5],
        }
    )

    out = correct_table(table)

    assert list(out["trustk_method"]) == ["pumping", "slug"]
    assert "trustk_k_soft_lower_m_s" in out.columns
    assert "trustk_k_soft_upper_m_s" in out.columns
    assert out["trustk_k_soft_m_s"].gt(0.0).all()


def test_correct_conductivity_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        correct_conductivity("unknown", 1.0e-5)
    with pytest.raises(ValueError):
        correct_conductivity("slug", -1.0e-5)

from pathlib import Path

import numpy as np
import pytest

from trustk.validation.lovelock import (
    OVERLAP_WELLS,
    cyclic_theis_drawdown,
    extract_bongo_model_comparison,
    extract_bongo_pumping_well,
    extract_lovelock_validation,
)


ZIP_PATH = Path("field data") / "OFR2019_1133_DataRelease.zip"
pytestmark = pytest.mark.skipif(not ZIP_PATH.exists(), reason="USGS raw data release is not present")


def test_bongo_model_comparison_extracts_measured_and_simulated_drawdown():
    data = extract_bongo_model_comparison(ZIP_PATH)

    assert len(data) > 500
    assert OVERLAP_WELLS.issubset(set(data["well"]))
    assert {"pumping", "recovery"}.issubset(set(data["phase"]))
    assert data["measured_drawdown_m"].notna().all()
    assert data["simulated_drawdown_m"].notna().all()


def test_bongo_pumping_well_is_near_bingo_area_and_rate_matches_workbook():
    pumping_well = extract_bongo_pumping_well(ZIP_PATH).iloc[0]

    assert pumping_well["well"] == "BONGO"
    assert 372_000 < pumping_well["x_m"] < 373_000
    assert 4_450_000 < pumping_well["y_m"] < 4_451_000
    assert 0.35 < pumping_well["q_m3_s"] < 0.41


def test_cyclic_theis_drawdown_recovers_after_pump_shutoff():
    t = np.array([3600.0, 24 * 3600.0, 7 * 24 * 3600.0, 20 * 24 * 3600.0])
    s = cyclic_theis_drawdown(
        radius_m=100.0,
        elapsed_s=t,
        transmissivity_m2_s=1e-2,
        storativity=1e-3,
        pumping_rate_m3_s=0.38,
        duration_s=6 * 24 * 3600.0,
    )

    assert s[1] > s[0]
    assert s[3] < s[2]
    assert np.all(s >= 0)


def test_lovelock_validation_builds_overlap_summary_and_theis_fits():
    validation = extract_lovelock_validation(ZIP_PATH)

    assert set(validation.overlap_summary["well"]) == OVERLAP_WELLS
    assert (validation.overlap_summary["distance_to_pump_m"] > 0).all()
    assert (validation.overlap_summary["k_slug_m_s"] > 0).all()
    assert len(validation.theis_fits) >= 5
    assert (validation.theis_fits["theis_transmissivity_m2_s"] > 0).all()
    assert len(validation.hard_slug_summary) == len(OVERLAP_WELLS)
    assert (validation.hard_slug_predictions["hard_slug_drawdown_m"] >= 0).all()
    assert validation.hard_slug_summary["hard_slug_rmse_m"].notna().all()

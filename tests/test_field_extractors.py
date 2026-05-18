from pathlib import Path

import pytest

from trustk.data.extract_pumping import OVERLAP_WELLS, extract_pumping_test2, normalize_well_name
from trustk.data.extract_slug import extract_slug_tests


ZIP_PATH = Path("field data") / "OFR2019_1133_DataRelease.zip"
pytestmark = pytest.mark.skipif(not ZIP_PATH.exists(), reason="USGS raw data release is not present")


def test_slug_extractor_returns_seven_slug_tests():
    extraction = extract_slug_tests(ZIP_PATH)

    assert set(extraction.tests["well"]) == {"BD", "BS", "MD", "MS", "PINN", "WD", "WS"}
    assert len(extraction.curves) > 8_000
    assert set(extraction.curves["well"]) == set(extraction.tests["well"])
    assert (extraction.tests["k_slug_m_s"] > 0).all()


def test_pumping_extractor_finds_overlap_wells_and_recovery():
    extraction = extract_pumping_test2(ZIP_PATH)
    wells = set(extraction.timeseries["well"])

    assert OVERLAP_WELLS.issubset(wells)
    assert {"pumping", "recovery"}.issubset(set(extraction.timeseries["phase"]))
    assert len(extraction.schedule) == 7
    assert extraction.timeseries["is_slug_overlap_well"].sum() > 5000
    assert OVERLAP_WELLS.issubset(set(extraction.coordinates["well"]))


def test_pumping_well_name_normalization_keeps_real_b_and_m_names():
    assert normalize_well_name("BING") == "BING"
    assert normalize_well_name("MOIR") == "MOIR"
    assert normalize_well_name("bBING") == "BING"
    assert normalize_well_name("mMoIr") == "MOIR"
    assert normalize_well_name("bBD__") == "BD"

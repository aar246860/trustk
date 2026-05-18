from pathlib import Path

import pytest

from trustk.data.usgs_inventory import inventory_zip

ZIP_PATH = Path("field data") / "OFR2019_1133_DataRelease.zip"
pytestmark = pytest.mark.skipif(not ZIP_PATH.exists(), reason="USGS raw data release is not present")


def test_usgs_inventory_finds_core_lovelock_files():
    inventory = inventory_zip(ZIP_PATH)

    assert inventory.total_entries > 1000
    assert len(inventory.slug_analysis_workbooks) == 7
    assert len(inventory.slug_data_workbooks) == 2
    assert len(inventory.pumping_data_workbooks) == 3
    assert any("Multi-WellPumpingTest2_Data.xlsx" in p for p in inventory.pumping_data_workbooks)
    assert any("Single-WellPumpingTest_Data.xlsx" in p for p in inventory.pumping_data_workbooks)
    assert len(inventory.well_log_files) >= 10

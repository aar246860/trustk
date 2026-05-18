"""Inventory helpers for the USGS Lovelock data release."""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class UsgsInventory:
    """Compact inventory of files relevant to TRUST-K."""

    total_entries: int
    slug_analysis_workbooks: list[str]
    slug_data_workbooks: list[str]
    pumping_data_workbooks: list[str]
    pumping_analysis_files: list[str]
    metadata_files: list[str]
    well_log_files: list[str]
    top_level_counts: dict[str, int]


def _top_level(path: str) -> str:
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 2 and parts[0] == "OFR2019_1133_DataRelease":
        return parts[1]
    return parts[0] if parts else ""


def inventory_zip(zip_path: str | Path) -> UsgsInventory:
    """Return a categorized inventory for the Lovelock data zip."""

    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()

    files = [name for name in names if not name.endswith("/")]
    lower = {name: name.lower() for name in files}
    top_counts = Counter(_top_level(name) for name in files)

    slug_analysis = [
        name
        for name in files
        if "injectionslug" in lower[name]
        and "analyses" in lower[name]
        and "bouwerrice" in lower[name]
        and lower[name].endswith((".xlsm", ".xlsx"))
    ]
    slug_data = [
        name
        for name in files
        if "injectionslug" in lower[name]
        and "/data/" in lower[name]
        and lower[name].endswith((".xlsx", ".xlsm"))
    ]
    pumping_data = [
        name
        for name in files
        if "pumpingtest" in lower[name]
        and "/data/" in lower[name]
        and lower[name].endswith((".xlsx", ".xlsm"))
    ]
    pumping_analysis = [
        name
        for name in files
        if "pumpingtest" in lower[name]
        and "/analyses/" in lower[name]
        and lower[name].endswith((".aqt", ".xlsm", ".xlsx", ".txt"))
    ]
    metadata = [name for name in files if lower[name].endswith(".xml")]
    well_logs = [
        name
        for name in files
        if "/welllogs/" in lower[name] and lower[name].endswith(".pdf")
    ]

    return UsgsInventory(
        total_entries=len(names),
        slug_analysis_workbooks=sorted(slug_analysis),
        slug_data_workbooks=sorted(slug_data),
        pumping_data_workbooks=sorted(pumping_data),
        pumping_analysis_files=sorted(pumping_analysis),
        metadata_files=sorted(metadata),
        well_log_files=sorted(well_logs),
        top_level_counts=dict(sorted(top_counts.items())),
    )


def write_inventory(inventory: UsgsInventory, out_path: str | Path) -> None:
    """Write an inventory JSON file."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(inventory), indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, help="Path to OFR2019_1133_DataRelease.zip")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args(argv)

    inventory = inventory_zip(args.zip)
    write_inventory(inventory, args.out)
    print(json.dumps(asdict(inventory), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


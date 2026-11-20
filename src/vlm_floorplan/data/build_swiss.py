"""Convert the fetched Swiss Dwellings geometries into `data/processed/swiss_dwellings.jsonl`.

One record per apartment, graph-only (`image` is empty; see `build_msd`).
Record ids are `swiss_dwellings/<site id>/<apartment id>`: apartments of one site (a building
complex) look alike, so the id carries the site, and a train/validation split can be made
by site later without re-reading the CSV.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from vlm_floorplan.converters import swiss
from vlm_floorplan.converters.swiss_rows import group_apartments, iter_site_rows
from vlm_floorplan.data.build import (
    CONVERSION_ERRORS,
    BuildOptions,
    BuildReport,
    sample_items,
)
from vlm_floorplan.data.fetch_swiss import SWISS_ATTRIBUTION
from vlm_floorplan.data.jsonl import write_records
from vlm_floorplan.record import PlanRecord


def _apartment_keys(csv_path: Path) -> list[str]:
    """Every "<site id>/<apartment id>" in the file, in file order (one cheap pass)."""
    keys: dict[str, None] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        site_column, apartment_column = header.index("site_id"), header.index("apartment_id")
        for row in reader:
            if row[apartment_column]:
                keys[f"{row[site_column]}/{row[apartment_column]}"] = None
    return list(keys)


def _records(
    csv_path: Path, wanted: set[str] | None, options: BuildOptions, report: BuildReport
) -> Iterator[PlanRecord]:
    """Convert apartments one site at a time, so memory holds one site, not the whole file."""
    seen_sites: set[str] = set()
    for site_rows in iter_site_rows(csv_path):
        site = site_rows[0]["site_id"]
        if site in seen_sites:
            # The reader assumes a site's rows are consecutive; a second run would
            # produce apartments with missing rows, so it is left out and reported.
            report.skipped.append((f"site {site}", "rows are not consecutive; later run skipped"))
            continue
        seen_sites.add(site)
        for apartment_id, rows in group_apartments(site_rows).items():
            key = f"{site}/{apartment_id}"
            if wanted is not None and key not in wanted:
                continue
            try:
                record = swiss.convert_apartment(rows)
            except CONVERSION_ERRORS as err:
                report.skipped.append((key, f"{type(err).__name__}: {err}"))
                continue
            yield record


def build_swiss(options: BuildOptions) -> BuildReport:
    """Convert the fetched Swiss Dwellings apartments."""
    csv_path = options.data_dir / "swiss_dwellings" / "geometries.csv"
    wanted = (
        None if options.limit is None else set(sample_items(_apartment_keys(csv_path), options))
    )
    output = options.out_dir / "swiss_dwellings.jsonl"
    report = BuildReport(output=output)
    report.written = write_records(output, _records(csv_path, wanted, options, report))
    (options.out_dir / "swiss_dwellings.ATTRIBUTION.txt").write_text(
        SWISS_ATTRIBUTION, encoding="utf-8"
    )
    return report

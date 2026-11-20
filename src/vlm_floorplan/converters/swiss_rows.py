"""Read Swiss Dwellings' `geometries.csv` one building site at a time.

The file has about 4.4 million rows (1.1 GB): every room, wall, door, window
and fixture of every apartment, one polygon (WKT) per row. It cannot be loaded
whole, and rows of one apartment are not always adjacent. The rows of one
*site* (a building complex) are, in every file checked (59 sites, 150 000 rows),
so the reader groups by site: it holds one site at a time and apartments are
assembled inside it.

Columns used: site_id, apartment_id, floor_id, entity_type (area, opening,
separator, feature), entity_subtype, geometry (WKT, metres), elevation.
Rows with an empty `apartment_id` are the building's shared spaces (stairs,
lifts, public corridors); they belong to no apartment and are skipped.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

Row = dict[str, str]


def iter_site_rows(path: Path) -> Iterator[list[Row]]:
    """Yield the rows of each run of consecutive rows that share a `site_id`."""
    with path.open(newline="", encoding="utf-8") as handle:
        site: str | None = None
        rows: list[Row] = []
        for row in csv.DictReader(handle):
            if row["site_id"] != site:
                if rows:
                    yield rows
                site, rows = row["site_id"], []
            rows.append(row)
        if rows:
            yield rows


def group_apartments(rows: Iterable[Row]) -> dict[str, list[Row]]:
    """Rows of a site grouped by apartment id, in first-seen order; shared spaces are dropped."""
    apartments: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        if row["apartment_id"]:
            apartments[row["apartment_id"]].append(row)
    return dict(apartments)

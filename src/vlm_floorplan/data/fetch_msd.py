"""Download MSD (Modified Swiss Dwellings) access graphs into the local data directory.

    data/msd/graph_out/<plan id>.pickle     (plus ATTRIBUTION.txt)

Source: https://data.4tu.nl/datasets/e1d89cb5-6872-48fc-be63-aadd687ee6f9
(doi 10.4121/e1d89cb5-6872-48fc-be63-aadd687ee6f9.v2, CC BY 4.0).

Only `graph_out` is fetched. The converter needs nothing else: the structure and
segmentation arrays (`struct_in`, `full_out`) are images of data the graph already
holds, and `graph_in` is the coarse zoning graph without room types or polygons.
The 4.8 GB train archive is never downloaded whole: a zip lists its members at the
end of the file, so range requests read only the ~10 kB pickles wanted (all 4167
come to about 40 MB).

The 0.8 GB test archive is not fetched at all: its annotations are withheld, and
it holds only `graph_in` and `struct_in`, which carry no room types.
"""

from __future__ import annotations

import random
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from vlm_floorplan.data.fetch import extract_member
from vlm_floorplan.data.http_zip import HttpRangeFile

MSD_TRAIN_ZIP_URL: Final = (
    "https://data.4tu.nl/file/e1d89cb5-6872-48fc-be63-aadd687ee6f9/"
    "279ef4b4-d3bd-41f4-b0c9-5e9af8cce6f6"
)
MSD_ATTRIBUTION: Final = """\
Modified Swiss Dwellings (MSD): Casper van Engelenburg, Seyran Khademi, Fatemeh Mostafavi,
Matthias Standfest, Michael Franzen (2023), 4TU.ResearchData,
https://doi.org/10.4121/e1d89cb5-6872-48fc-be63-aadd687ee6f9.v2
Licensed CC BY 4.0: https://creativecommons.org/licenses/by/4.0/
MSD is derived from Swiss Dwellings (https://zenodo.org/records/7788422, CC BY 4.0).
Changes: converted to room graphs (rooms, links, entrance flags, 0-1000 coordinates).
"""

_PLAN_ID = re.compile(r"\d+")
_GRAPH_MEMBER = re.compile(r"^graph_out/(?P<id>\d+)\.pickle$")


@dataclass(frozen=True)
class MsdSelection:
    """Which MSD plans to fetch."""

    plan_ids: tuple[str, ...] = ()
    """Explicit ids such as "0" or "4172". If given, the options below are ignored."""
    count: int | None = 20
    """A random sample of this many plans; None fetches all 4167."""
    seed: int = 0


def choose_msd_ids(available: list[str], selection: MsdSelection) -> list[str]:
    """Resolve a selection against the ids in the archive; random choices follow the seed."""
    if selection.plan_ids:
        for plan_id in selection.plan_ids:
            # Ids become file names, so accept digits only.
            if not _PLAN_ID.fullmatch(plan_id):
                raise ValueError(f"MSD plan id must be a number, got {plan_id!r}")
            if plan_id not in available:
                raise ValueError(f"MSD has no plan {plan_id!r}")
        return list(selection.plan_ids)
    if selection.count is None or selection.count >= len(available):
        return list(available)
    return sorted(random.Random(selection.seed).sample(available, selection.count), key=int)


def fetch_msd(out_dir: Path, selection: MsdSelection, url: str = MSD_TRAIN_ZIP_URL) -> int:
    """Download the selected graphs into `out_dir/graph_out`; returns how many were chosen.

    Files already on disk are skipped, so an interrupted run can simply be repeated.
    """
    with zipfile.ZipFile(HttpRangeFile(url)) as archive:
        names = archive.namelist()
        matches = (_GRAPH_MEMBER.match(name) for name in names)
        available = sorted((m["id"] for m in matches if m), key=int)
        ids = choose_msd_ids(available, selection)
        # Read in archive order: neighbouring members share one cached 256 kB network block,
        # so this is about 25 times fewer requests than numeric order.
        position = {name: index for index, name in enumerate(names)}
        for plan_id in sorted(ids, key=lambda i: position[f"graph_out/{i}.pickle"]):
            extract_member(
                archive, f"graph_out/{plan_id}.pickle", out_dir / "graph_out" / f"{plan_id}.pickle"
            )
    (out_dir / "ATTRIBUTION.txt").write_text(MSD_ATTRIBUTION, encoding="utf-8")
    return len(ids)

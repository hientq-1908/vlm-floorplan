"""Download Swiss Dwellings' `geometries.csv` into the local data directory.

    data/swiss_dwellings/geometries.csv     (plus ATTRIBUTION.txt)

Source: https://zenodo.org/records/7788422 (v3.0.0, CC BY 4.0, 932 MB zip).
The zip also holds simulation results (1.6 GB, 640 MB compressed) that this project does not
use, so only the `geometries.csv` member is read, by range requests: about 290 MB
compressed, 1.1 GB on disk, 45 000 apartments.

A subset is the first sites of the file, not a random sample: the rows cannot be sampled
without reading everything, and one site (a building complex) is always kept whole, because
apartments are only assembled correctly from all rows of their site.
"""

from __future__ import annotations

import csv
import io
import shutil
import zipfile
from pathlib import Path
from typing import Final

from vlm_floorplan.data.http_zip import HttpRangeFile

SWISS_ZIP_URL: Final = (
    "https://zenodo.org/records/7788422/files/swiss-dwellings-v3.0.0.zip?download=1"
)
SWISS_MEMBER: Final = "swiss-dwellings-v3.0.0/geometries.csv"
SWISS_ATTRIBUTION: Final = """\
Swiss Dwellings: Matthias Standfest, Michael Franzen, Yvonne Schroeder, Luis Gonzalez Medina,
Yarilo Villanueva Hernandez, Jan Hendrik Buck, Yen-Ling Tan, Milena Niedzwiecka,
Rachele Colmegna (2022), https://doi.org/10.5281/zenodo.7788422 (v3.0.0)
Licensed CC BY 4.0: https://creativecommons.org/licenses/by/4.0/
Changes: converted to room graphs (rooms, links, entrance flags, 0-1000 coordinates).
"""


def _copy_first_apartments(source: io.TextIOBase, sink: io.TextIOBase, apartments: int) -> int:
    """Copy rows until `apartments` apartments are in, finishing the current site.

    Returns the number of apartments copied. Shared spaces (empty apartment id) do not count.
    """
    reader = csv.reader(source)
    writer = csv.writer(sink, lineterminator="\n")
    header = next(reader)
    writer.writerow(header)
    site_column, apartment_column = header.index("site_id"), header.index("apartment_id")
    seen: set[str] = set()
    site = None
    for row in reader:
        if row[site_column] != site:
            if len(seen) >= apartments:
                break
            site = row[site_column]
        if row[apartment_column]:
            seen.add(row[apartment_column])
        writer.writerow(row)
    return len(seen)


def fetch_swiss(out_dir: Path, apartments: int | None = 50, url: str = SWISS_ZIP_URL) -> Path:
    """Write `geometries.csv` into `out_dir` (the first sites holding `apartments` apartments).

    `apartments=None` fetches the whole file. An existing file is kept, so delete it
    to fetch a different amount.
    """
    target = out_dir / "geometries.csv"
    if not target.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".csv.part")
        with zipfile.ZipFile(HttpRangeFile(url)) as archive, archive.open(SWISS_MEMBER) as member:
            if apartments is None:
                with partial.open("wb") as sink:
                    shutil.copyfileobj(member, sink)
            else:
                text = io.TextIOWrapper(member, encoding="utf-8", newline="")
                with partial.open("w", encoding="utf-8", newline="") as sink:
                    _copy_first_apartments(text, sink, apartments)
        partial.rename(target)  # a half-written file never looks finished
    (out_dir / "ATTRIBUTION.txt").write_text(SWISS_ATTRIBUTION, encoding="utf-8")
    return target

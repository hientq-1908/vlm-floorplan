"""Download the datasets into a local, gitignored data directory.

Layout after fetching (everything under `data/` is ignored by git):

    data/cubicasa5k/<style>/<plan id>/model.svg, F1_scaled.png
    data/cubicasa5k/train.txt, val.txt, test.txt     (the dataset's own splits)

CubiCasa5K is CC BY-NC 4.0 (non-commercial): read the README before sharing derived data.
"""

from __future__ import annotations

import random
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from vlm_floorplan.data.http_zip import HttpRangeFile

CUBICASA_ZIP_URL: Final = "https://zenodo.org/records/2613548/files/cubicasa5k.zip?download=1"
CUBICASA_STYLES: Final = ("colorful", "high_quality", "high_quality_architectural")
CUBICASA_FILES: Final = ("model.svg", "F1_scaled.png")
"""Only what the converter reads. F2_scaled.png is byte-identical to F1_scaled.png."""
CUBICASA_SPLITS: Final = ("train", "val", "test")

_PLAN_ID = re.compile(r"[a-z_]+/\d+")
_PLAN_MEMBER = re.compile(r"^cubicasa5k/(?P<style>[^/]+)/(?P<number>\d+)/model\.svg$")


@dataclass(frozen=True)
class CubiCasaSelection:
    """Which CubiCasa5K plans to fetch."""

    plan_ids: tuple[str, ...] = ()
    """Explicit ids such as "colorful/8759". If given, the options below are ignored."""
    count: int = 20
    split: str | None = None
    seed: int = 0


def _split_ids(archive: zipfile.ZipFile, split: str) -> list[str]:
    """Plan ids ("colorful/8759") of one of the dataset's own splits."""
    text = archive.read(f"cubicasa5k/{split}.txt").decode()
    return [line.strip("/ \n") for line in text.splitlines() if line.strip("/ \n")]


def _all_ids(archive: zipfile.ZipFile) -> list[str]:
    ids = []
    for name in archive.namelist():
        match = _PLAN_MEMBER.match(name)
        if match:
            ids.append(f"{match['style']}/{match['number']}")
    return sorted(ids)


def choose_cubicasa_ids(archive: zipfile.ZipFile, selection: CubiCasaSelection) -> list[str]:
    """Resolve a selection to plan ids; random choices are reproducible via the seed."""
    if selection.plan_ids:
        for plan_id in selection.plan_ids:
            # Ids become directory names, so reject anything but "<style>/<number>".
            if not _PLAN_ID.fullmatch(plan_id):
                raise ValueError(f"plan id must look like 'colorful/8759', got {plan_id!r}")
        return list(selection.plan_ids)
    if selection.split is not None:
        pool = _split_ids(archive, selection.split)
    else:
        pool = _all_ids(archive)
    return sorted(random.Random(selection.seed).sample(pool, min(selection.count, len(pool))))


def _extract(archive: zipfile.ZipFile, member: str, target: Path) -> bool:
    """Copy one zip member to `target` unless it is already there. True if downloaded."""
    if target.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with archive.open(member) as source, partial.open("wb") as sink:
        shutil.copyfileobj(source, sink)
    partial.rename(target)  # a half-written file never looks finished
    return True


def fetch_cubicasa(out_dir: Path, selection: CubiCasaSelection, url: str = CUBICASA_ZIP_URL) -> int:
    """Download the selected plans (and the split files) into `out_dir`.

    Returns the number of plans fetched. Plans already on disk are skipped,
    so an interrupted run can simply be repeated.
    """
    with zipfile.ZipFile(HttpRangeFile(url)) as archive:
        for split in CUBICASA_SPLITS:
            _extract(archive, f"cubicasa5k/{split}.txt", out_dir / f"{split}.txt")
        ids = choose_cubicasa_ids(archive, selection)
        for plan_id in ids:
            for name in CUBICASA_FILES:
                _extract(archive, f"cubicasa5k/{plan_id}/{name}", out_dir / plan_id / name)
    return len(ids)

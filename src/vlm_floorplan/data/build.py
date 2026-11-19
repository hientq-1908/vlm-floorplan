"""Convert fetched plans into a JSONL file of `PlanRecord`s.

    data/cubicasa5k/...   -> data/processed/cubicasa5k.jsonl

Records point at the original `F1_scaled.png`.
Plans that fail to convert are skipped and reported, never silently dropped.
"""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from pydantic import ValidationError

from vlm_floorplan.converters import cubicasa
from vlm_floorplan.data.jsonl import with_relative_image, write_records
from vlm_floorplan.record import PlanRecord

T = TypeVar("T")

CONVERSION_ERRORS = (ValueError, ValidationError, ET.ParseError, OSError, KeyError)
"""What a malformed plan can raise; anything else is a bug and should surface."""


@dataclass(frozen=True)
class BuildOptions:
    """Which plans to convert and where the output goes."""

    data_dir: Path = Path("data")
    out_dir: Path = Path("data/processed")
    limit: int | None = None
    """Convert at most this many plans (a random, seeded sample); None means all."""
    split: str | None = None
    """Only plans of this split ("train", "val", "test"); None means every plan."""
    seed: int = 0


@dataclass
class BuildReport:
    """What a build did."""

    output: Path
    written: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)
    """(plan id, reason) for every plan left out."""

    def summary(self) -> str:
        lines = [f"wrote {self.written} records to {self.output}"]
        lines += [f"  skipped {plan_id}: {reason}" for plan_id, reason in self.skipped]
        return "\n".join(lines)


def sample_items(items: list[T], options: BuildOptions) -> list[T]:
    """A seeded random subset of `items`, kept in the original order."""
    if options.limit is None or options.limit >= len(items):
        return items
    chosen = sorted(random.Random(options.seed).sample(range(len(items)), options.limit))
    return [items[i] for i in chosen]


def _cubicasa_splits(root: Path) -> dict[str, str]:
    """Map "colorful/8759" to "train", "val" or "test" using the dataset's split files."""
    splits: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = root / f"{split}.txt"
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip("/ "):
                    splits[line.strip("/ ")] = split
    return splits


def build_cubicasa(options: BuildOptions) -> BuildReport:
    """Convert the fetched CubiCasa5K plans."""
    root = options.data_dir / "cubicasa5k"
    splits = _cubicasa_splits(root)
    plan_ids = sorted(
        f"{path.parent.parent.name}/{path.parent.name}"
        for path in root.glob("*/*/model.svg")
        if (path.parent / "F1_scaled.png").exists()
    )
    if options.split is not None:
        plan_ids = [plan_id for plan_id in plan_ids if splits.get(plan_id) == options.split]
    output = options.out_dir / "cubicasa5k.jsonl"
    report = BuildReport(output=output)
    records: list[PlanRecord] = []
    for plan_id in sample_items(plan_ids, options):
        try:
            record = cubicasa.convert_plan(
                root / plan_id, f"cubicasa5k/{plan_id}", splits.get(plan_id)
            )
        except CONVERSION_ERRORS as err:
            report.skipped.append((plan_id, f"{type(err).__name__}: {err}"))
            continue
        records.append(with_relative_image(record, output))
    report.written = write_records(output, records)
    return report

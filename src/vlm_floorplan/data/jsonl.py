"""Read and write `PlanRecord`s as JSON Lines (one record per line).

A record's `image` is stored relative to the directory of the JSONL file that
holds it, so a processed dataset can be moved or copied as a whole.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from pathlib import Path

from vlm_floorplan.record import PlanRecord


def write_records(path: Path, records: Iterable[PlanRecord]) -> int:
    """Write records to `path`; returns how many were written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
            count += 1
    return count


def read_records(path: Path) -> Iterator[PlanRecord]:
    """Yield the records of a JSONL file, validating each one."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield PlanRecord.model_validate_json(line)


def image_path(jsonl_path: Path, record: PlanRecord) -> Path:
    """Absolute path of a record's image."""
    return (jsonl_path.parent / record.image).resolve()


def with_relative_image(record: PlanRecord, jsonl_path: Path) -> PlanRecord:
    """Copy of `record` whose absolute `image` path is made relative to the JSONL file."""
    relative = os.path.relpath(Path(record.image).resolve(), jsonl_path.parent.resolve())
    return record.model_copy(update={"image": Path(relative).as_posix()})

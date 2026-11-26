"""Plan pools: read the converted datasets, split them, and keep what can be a held-out plan.

A candidate is a usable plan (see `frame.unusable_reason`) in the canonical frame whose
exterior wall is a valid polygon containing every indoor room point. Held-out candidates also
pass the leak index: their structure is not one a train plan has.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from shapely.geometry import Polygon

from vlm_floorplan.record import PlanRecord
from vlm_floorplan.text2graph.frame import (
    Canonical,
    FrameError,
    canonicalize,
    indoor_points_inside_wall,
    unusable_reason,
)
from vlm_floorplan.text2graph.leak import LeakIndex, structure_hash
from vlm_floorplan.text2graph.splits import split_of

SOURCES = ("resplan", "swiss_dwellings", "cubicasa5k")


@dataclass(frozen=True)
class Candidate:
    record_id: str
    source: str
    canon: Canonical


@dataclass(frozen=True)
class SourceFiles:
    """Where the converted datasets are. Any may be None to leave a source out."""

    resplan_dir: Path | None = None
    cubicasa_file: Path | None = None
    swiss_file: Path | None = None

    def paths(self) -> list[Path]:
        found: list[Path] = []
        if self.resplan_dir:
            found += [self.resplan_dir / f"{name}.jsonl" for name in ("train", "val", "test")]
        found += [p for p in (self.cubicasa_file, self.swiss_file) if p]
        return [p for p in found if p.exists()]


def read_records(paths: Iterable[Path]) -> Iterator[PlanRecord]:
    for path in paths:
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    yield PlanRecord.model_validate_json(line)


def build_leak_index(records: Iterable[PlanRecord], train_texts: Iterable[str] = ()) -> LeakIndex:
    """Index every train plan (usable or not: the stricter reading of "seen in train")."""
    index = LeakIndex()
    for record in records:
        if split_of(record) == "train":
            index.add_train_plan(record.id, record.graph)
    index.add_train_texts(train_texts)
    return index


def make_candidate(record: PlanRecord) -> Candidate | None:
    if unusable_reason(record) is not None:
        return None
    try:
        canon = canonicalize(record)
    except FrameError:
        return None
    graph = canon.record.graph
    wall = Polygon(graph.exterior_wall or [])
    if not wall.is_valid or not indoor_points_inside_wall(graph):
        return None
    return Candidate(record.id, record.source, canon)


def held_out_candidates(
    records: Iterable[PlanRecord], split: str, index: LeakIndex, rng: random.Random
) -> dict[str, list[Candidate]]:
    """Candidates of one split per source, shuffled, one per structure, none seen in train."""
    seen: set[str] = set()
    pools: dict[str, list[Candidate]] = {source: [] for source in SOURCES}
    for record in records:
        if split_of(record) != split or record.source not in pools:
            continue
        if index.violations(record.id, record.graph):
            continue
        structure = structure_hash(record.graph)
        if structure in seen:
            continue
        candidate = make_candidate(record)
        if candidate is not None:
            seen.add(structure)
            pools[record.source].append(candidate)
    for pool in pools.values():
        pool.sort(key=lambda c: c.record_id)
        rng.shuffle(pool)
    return pools

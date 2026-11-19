"""Build ResPlan records: one PNG per plan, and JSONL files of `PlanRecord`s.

    uv run python -m vlm_floorplan.resplan.fetch
    uv run python -m vlm_floorplan.resplan.build [--out data/resplan/processed]

Output layout (the JSONL files hold paths relative to their own directory, as
`PlanRecord.image` requires):

    processed/
      images/<plan id>.png
      train.jsonl  val.jsonl  test.jsonl      (and augmented.jsonl with --splits augmented)
      ATTRIBUTION.txt

The `augmented` split (rotated, flipped or scaled copies of originals) is off by default:
it is near-duplicate data, and the training pipeline can augment on its own.
Records have the same shape as the CubiCasa5K ones, so one loader reads both.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vlm_floorplan.record import PlanQuality, PlanRecord
from vlm_floorplan.resplan.convert import ConvertedPlan, convert_plan
from vlm_floorplan.resplan.frame import PlanView
from vlm_floorplan.resplan.load import SPLIT_NAMES, load_excluded_ids, load_plans, load_splits
from vlm_floorplan.resplan.render import RenderStyle, RoomLabel, render_plan

SOURCE_NAME = "resplan"
IMAGE_DIR = "images"
DEFAULT_DATA_DIR = Path("data/resplan")
DEFAULT_SPLITS = ("train", "val", "test")
CHUNK_SIZE = 64

ATTRIBUTION = """\
These plans and graphs are derived from ResPlan (https://github.com/m-agour/ResPlan),
which is licensed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
Changes: converted to room graphs, with images rendered from the vector geometry.
Citation: ResPlan, arXiv:2508.14006.
"""


@dataclass(frozen=True)
class BuildOptions:
    out_dir: Path
    labels: bool = False
    """Draw each room's type name on the image."""
    long_side: int = 1024


def _labels(converted: ConvertedPlan) -> list[RoomLabel]:
    return [RoomLabel(room.type.value.title(), room.xy) for room in converted.graph.rooms()]


def build_record(
    plan: dict[str, Any], split: str, options: BuildOptions
) -> tuple[PlanRecord | None, str | None]:
    """Convert and render one plan, saving its PNG. Returns (record, None) or (None, reason)."""
    plan_id = plan["id"]
    image_name = f"{IMAGE_DIR}/{plan_id}.png"
    try:
        view = PlanView.for_plan(plan, long_side=options.long_side)
        converted = convert_plan(plan, view)
    except ValueError as err:
        (options.out_dir / image_name).unlink(missing_ok=True)
        return None, f"plan {plan_id} skipped: {err}"
    labels = _labels(converted) if options.labels else None
    style = RenderStyle(long_side=options.long_side)
    render_plan(plan, view, style, labels).save(options.out_dir / image_name)
    record = PlanRecord(
        id=f"{SOURCE_NAME}/{plan_id}",
        source=SOURCE_NAME,
        split=split,
        image=image_name,
        graph=converted.graph,
        rooms=converted.sources,
        quality=PlanQuality.from_graph(converted.graph, converted.warnings),
    )
    return record, None


def _build_one(job: tuple[dict[str, Any], str, BuildOptions]) -> tuple[str, str | None, str | None]:
    """Worker entry point. Returns (split, record JSON line, skip reason)."""
    plan, split, options = job
    record, reason = build_record(plan, split, options)
    return split, record.model_dump_json() if record else None, reason


def select_plans(
    plans: list[dict[str, Any]],
    split_of: dict[int, str],
    splits: tuple[str, ...],
    excluded: set[int],
) -> list[tuple[dict[str, Any], str]]:
    """Plans in the wanted splits, minus excluded ids, in file order."""
    chosen = []
    for plan in plans:
        split = split_of.get(plan["id"])
        if split in splits and plan["id"] not in excluded:
            chosen.append((plan, split))
    return chosen


def drop_excluded_records(out_dir: Path, splits: tuple[str, ...], excluded: set[int]) -> None:
    """Remove excluded plans from JSONL files of earlier builds whose split is not being rebuilt."""
    excluded_record_ids = {f"{SOURCE_NAME}/{plan_id}" for plan_id in excluded}
    for split in SPLIT_NAMES:
        path = out_dir / f"{split}.jsonl"
        if split in splits or not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        kept = [line for line in lines if json.loads(line)["id"] not in excluded_record_ids]
        if len(kept) < len(lines):
            path.write_text("".join(line + "\n" for line in kept), encoding="utf-8")


def build(
    data_dir: Path,
    options: BuildOptions,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    exclude_ids: Path | None = None,
    limit: int | None = None,
    workers: int = 1,
) -> dict[str, int]:
    """Write images and JSONL files. Returns the number of records written per split."""
    plans = load_plans(data_dir / "ResPlan.pkl")
    split_of = load_splits(data_dir / "split.json")
    excluded = load_excluded_ids(exclude_ids)
    chosen = select_plans(plans, split_of, splits, excluded)
    if limit is not None:
        chosen = chosen[:limit]

    image_dir = options.out_dir / IMAGE_DIR
    image_dir.mkdir(parents=True, exist_ok=True)
    for plan_id in excluded:
        (image_dir / f"{plan_id}.png").unlink(missing_ok=True)
    drop_excluded_records(options.out_dir, splits, excluded)
    (options.out_dir / "ATTRIBUTION.txt").write_text(ATTRIBUTION, encoding="utf-8")
    jobs = [(plan, split, options) for plan, split in chosen]
    counts = dict.fromkeys(splits, 0)
    handles = {
        split: (options.out_dir / f"{split}.jsonl").open("w", encoding="utf-8") for split in splits
    }
    try:
        for split, line, reason in _run(jobs, workers):
            if line is None:
                print(reason)
                continue
            handles[split].write(line + "\n")
            counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    return counts


def _run(jobs: list[Any], workers: int) -> Any:
    if workers <= 1:
        yield from map(_build_one, jobs)
        return
    with multiprocessing.Pool(workers) as pool:
        yield from pool.imap(_build_one, jobs, chunksize=CHUNK_SIZE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ResPlan plans to records and images.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=None, help="default: <data-dir>/processed")
    parser.add_argument("--splits", nargs="+", choices=SPLIT_NAMES, default=list(DEFAULT_SPLITS))
    parser.add_argument("--exclude-ids", type=Path, help="file of plan ids to skip (takedowns)")
    parser.add_argument("--labels", action="store_true", help="draw room type names on the images")
    parser.add_argument("--long-side", type=int, default=1024)
    parser.add_argument("--limit", type=int, help="only the first N plans (for a quick look)")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()

    options = BuildOptions(
        out_dir=args.out or args.data_dir / "processed",
        labels=args.labels,
        long_side=args.long_side,
    )
    counts = build(
        args.data_dir, options, tuple(args.splits), args.exclude_ids, args.limit, args.workers
    )
    print(f"wrote {sum(counts.values())} records to {options.out_dir}: {counts}")


if __name__ == "__main__":
    main()

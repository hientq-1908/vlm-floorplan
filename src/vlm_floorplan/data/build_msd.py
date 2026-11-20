"""Convert fetched MSD graphs into `data/processed/msd.jsonl`.

The records are graph-only: `image` is empty, because MSD has no pictures and no
renderer for it exists yet. They feed editing practice and the retrieval pool.
`split` is left empty too: MSD's own train/test split withholds the test labels, and
plans of one building complex look alike, so a split should be made by the training
pipeline, not guessed here.
"""

from __future__ import annotations

from vlm_floorplan.converters import msd
from vlm_floorplan.data.build import (
    CONVERSION_ERRORS,
    BuildOptions,
    BuildReport,
    sample_items,
)
from vlm_floorplan.data.fetch_msd import MSD_ATTRIBUTION
from vlm_floorplan.data.jsonl import write_records
from vlm_floorplan.record import PlanRecord


def build_msd(options: BuildOptions) -> BuildReport:
    """Convert the fetched MSD plans."""
    root = options.data_dir / "msd" / "graph_out"
    plan_ids = sorted((path.stem for path in root.glob("*.pickle")), key=int)
    output = options.out_dir / "msd.jsonl"
    report = BuildReport(output=output)
    records: list[PlanRecord] = []
    for plan_id in sample_items(plan_ids, options):
        try:
            record = msd.convert_plan((root / f"{plan_id}.pickle").read_bytes(), plan_id)
        except CONVERSION_ERRORS as err:
            report.skipped.append((plan_id, f"{type(err).__name__}: {err}"))
            continue
        records.append(record)
    report.written = write_records(output, records)
    (options.out_dir / "msd.ATTRIBUTION.txt").write_text(MSD_ATTRIBUTION, encoding="utf-8")
    return report

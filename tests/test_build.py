"""Building JSONL from fetched plans, and reading it back."""

from __future__ import annotations

from pathlib import Path

from cubicasa_fixture import write_cubicasa_plan

from vlm_floorplan.data.build import BuildOptions, build_cubicasa
from vlm_floorplan.data.jsonl import image_path, read_records, write_records
from vlm_floorplan.record import PlanRecord


def _fetched_dataset(tmp_path: Path) -> Path:
    """data/cubicasa5k/ laid out as `vlm-floorplan fetch cubicasa` leaves it."""
    root = tmp_path / "data" / "cubicasa5k"
    for plan_id in ("colorful/1", "colorful/2", "high_quality/3"):
        write_cubicasa_plan(root / plan_id)
    (root / "train.txt").write_text("/colorful/1/\n/colorful/2/\n")
    (root / "val.txt").write_text("/high_quality/3/\n")
    return tmp_path / "data"


def test_build_writes_one_record_per_plan_with_relative_images(tmp_path: Path) -> None:
    data = _fetched_dataset(tmp_path)
    report = build_cubicasa(BuildOptions(data_dir=data, out_dir=tmp_path / "out"))

    assert report.written == 3 and not report.skipped
    records = list(read_records(report.output))
    assert [r.id for r in records] == [
        "cubicasa5k/colorful/1", "cubicasa5k/colorful/2", "cubicasa5k/high_quality/3"
    ]  # fmt: skip
    assert [r.split for r in records] == ["train", "train", "val"]
    assert not Path(records[0].image).is_absolute()
    assert image_path(report.output, records[0]).exists()


def test_build_can_filter_by_split_and_limit_reproducibly(tmp_path: Path) -> None:
    data = _fetched_dataset(tmp_path)
    val = build_cubicasa(BuildOptions(data_dir=data, out_dir=tmp_path / "a", split="val"))
    assert val.written == 1
    options = BuildOptions(data_dir=data, out_dir=tmp_path / "b", limit=2, seed=3)
    first = build_cubicasa(options)
    again = build_cubicasa(BuildOptions(data_dir=data, out_dir=tmp_path / "c", limit=2, seed=3))
    ids = lambda report: [r.id for r in read_records(report.output)]  # noqa: E731
    assert first.written == 2 and ids(first) == ids(again)


def test_broken_plans_are_skipped_and_reported(tmp_path: Path) -> None:
    data = _fetched_dataset(tmp_path)
    (data / "cubicasa5k" / "colorful" / "2" / "model.svg").write_text("<svg")
    report = build_cubicasa(BuildOptions(data_dir=data, out_dir=tmp_path / "out"))
    assert report.written == 2
    assert [plan_id for plan_id, _ in report.skipped] == ["colorful/2"]
    assert "ParseError" in report.skipped[0][1]


def test_mostly_unlabeled_plans_can_be_dropped(tmp_path: Path) -> None:
    data = _fetched_dataset(tmp_path)
    svg = data / "cubicasa5k" / "colorful" / "1" / "model.svg"
    svg.write_text(
        svg.read_text()
        .replace("Space Kitchen", "Space Undefined")
        .replace("Space LivingRoom", "Space Undefined")
        .replace("Space Dining", "Space Undefined")
        .replace("Space UserDefined", "Space Undefined")
        .replace("Space Storage", "Space Undefined")
    )
    kept = build_cubicasa(BuildOptions(data_dir=data, out_dir=tmp_path / "a"))
    assert kept.written == 3
    flagged = next(r for r in read_records(kept.output) if r.id.endswith("colorful/1"))
    assert flagged.quality.mostly_unlabeled
    dropped = build_cubicasa(
        BuildOptions(data_dir=data, out_dir=tmp_path / "b", drop_mostly_unlabeled=True)
    )
    assert dropped.written == 2
    assert ("cubicasa5k/colorful/1", "mostly unlabeled") in dropped.skipped


def test_jsonl_round_trip_is_lossless(tmp_path: Path, cubicasa_record: PlanRecord) -> None:
    path = tmp_path / "x.jsonl"
    assert write_records(path, [cubicasa_record]) == 1
    assert list(read_records(path)) == [cubicasa_record]

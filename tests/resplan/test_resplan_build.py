from __future__ import annotations

import json
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from shapely.geometry import box

from vlm_floorplan.record import PlanRecord
from vlm_floorplan.resplan.build import BuildOptions, build

PlanMaker = Callable[[int], dict[str, Any]]


@pytest.fixture
def data_dir(tmp_path: Path, make_plan: PlanMaker) -> Path:
    """Four plans: two train, one val, one augmented."""
    directory = tmp_path / "resplan"
    directory.mkdir()
    plans = [make_plan(1), make_plan(2), make_plan(3), make_plan(100)]
    (directory / "ResPlan.pkl").write_bytes(pickle.dumps(plans))
    splits = {"train": [1, 2], "val": [3], "test": [], "augmented": [100]}
    (directory / "split.json").write_text(json.dumps(splits))
    return directory


def _records(path: Path) -> list[PlanRecord]:
    return [PlanRecord.model_validate_json(line) for line in path.read_text().splitlines()]


def test_build_writes_one_record_and_image_per_plan(data_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    counts = build(data_dir, BuildOptions(out_dir=out))
    assert counts == {"train": 2, "val": 1, "test": 0}
    train = _records(out / "train.jsonl")
    assert [record.id for record in train] == ["resplan/1", "resplan/2"]
    assert {record.source for record in train} == {"resplan"}
    assert {record.split for record in train} == {"train"}


def test_record_image_path_is_relative_to_the_jsonl_file(data_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    build(data_dir, BuildOptions(out_dir=out))
    record = _records(out / "val.jsonl")[0]
    image = Image.open(out / record.image)
    assert (image.width, image.height) == (record.graph.image.w, record.graph.image.h)


def test_the_augmented_split_is_opt_in(data_dir: Path, tmp_path: Path) -> None:
    default_out = tmp_path / "default"
    build(data_dir, BuildOptions(out_dir=default_out))
    assert not (default_out / "augmented.jsonl").exists()

    out = tmp_path / "with_augmented"
    counts = build(data_dir, BuildOptions(out_dir=out), splits=("augmented",))
    assert counts == {"augmented": 1}
    assert _records(out / "augmented.jsonl")[0].id == "resplan/100"


def test_excluded_ids_are_skipped_for_takedowns(data_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    build(data_dir, BuildOptions(out_dir=out))
    assert (out / "images" / "2.png").exists()
    exclude = tmp_path / "exclude.txt"
    exclude.write_text("2\n")
    build(data_dir, BuildOptions(out_dir=out), exclude_ids=exclude)
    assert [record.id for record in _records(out / "train.jsonl")] == ["resplan/1"]
    assert not (out / "images" / "2.png").exists()


def test_a_takedown_rebuild_drops_the_plan_from_splits_it_does_not_rebuild(
    data_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    build(data_dir, BuildOptions(out_dir=out), splits=("train", "val", "test", "augmented"))
    assert [record.id for record in _records(out / "augmented.jsonl")] == ["resplan/100"]
    exclude = tmp_path / "exclude.txt"
    exclude.write_text("100\n")
    build(data_dir, BuildOptions(out_dir=out), exclude_ids=exclude)
    assert _records(out / "augmented.jsonl") == []
    assert not (out / "images" / "100.png").exists()
    assert [record.id for record in _records(out / "train.jsonl")] == ["resplan/1", "resplan/2"]


def test_a_plan_without_rooms_is_skipped_not_fatal(
    data_dir: Path, tmp_path: Path, make_plan: PlanMaker, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    (data_dir / "ResPlan.pkl").write_bytes(pickle.dumps([make_plan(1), make_plan(4)]))
    (data_dir / "split.json").write_text(json.dumps({"train": [1, 4]}))
    build(data_dir, BuildOptions(out_dir=out), splits=("train",))
    assert (out / "images" / "4.png").exists()

    empty = {"id": 4, "inner": box(0, 0, 10, 10), "wall": box(0, 0, 10, 10)}
    (data_dir / "ResPlan.pkl").write_bytes(pickle.dumps([make_plan(1), empty]))
    counts = build(data_dir, BuildOptions(out_dir=out), splits=("train",))
    assert counts == {"train": 1}
    assert "plan 4 skipped" in capsys.readouterr().out
    assert not (out / "images" / "4.png").exists()


def test_limit_caps_the_number_of_plans(data_dir: Path, tmp_path: Path) -> None:
    counts = build(data_dir, BuildOptions(out_dir=tmp_path / "out"), limit=1)
    assert sum(counts.values()) == 1


def test_parallel_build_matches_the_serial_one(data_dir: Path, tmp_path: Path) -> None:
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    build(data_dir, BuildOptions(out_dir=serial), workers=1)
    build(data_dir, BuildOptions(out_dir=parallel), workers=2)
    assert (serial / "train.jsonl").read_text() == (parallel / "train.jsonl").read_text()


def test_labels_option_changes_the_image(data_dir: Path, tmp_path: Path) -> None:
    plain, labeled = tmp_path / "plain", tmp_path / "labeled"
    build(data_dir, BuildOptions(out_dir=plain), limit=1)
    build(data_dir, BuildOptions(out_dir=labeled, labels=True), limit=1)
    assert (plain / "images" / "1.png").read_bytes() != (labeled / "images" / "1.png").read_bytes()


def test_the_output_carries_the_cc_by_attribution(data_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    build(data_dir, BuildOptions(out_dir=out))
    attribution = (out / "ATTRIBUTION.txt").read_text()
    assert "https://github.com/m-agour/ResPlan" in attribution
    assert "https://creativecommons.org/licenses/by/4.0/" in attribution
    assert "Changes:" in attribution

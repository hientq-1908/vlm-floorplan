from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import pytest
from shapely.geometry import MultiPolygon, Point, box

from vlm_floorplan.resplan.load import (
    geometry_parts,
    load_excluded_ids,
    load_plans,
    load_splits,
    normalize_keys,
)


def test_load_plans_reads_a_pickled_list_of_plans(tmp_path: Path, plan: dict[str, Any]) -> None:
    path = tmp_path / "ResPlan.pkl"
    path.write_bytes(pickle.dumps([plan]))
    loaded = load_plans(path)
    assert len(loaded) == 1
    assert loaded[0]["living"].equals(plan["living"])


def test_load_plans_refuses_to_run_code_from_the_pickle(tmp_path: Path) -> None:
    marker = tmp_path / "pwned"

    class Evil:
        def __reduce__(self) -> tuple[Any, tuple[str]]:
            return os.system, (f"touch {marker}",)

    path = tmp_path / "ResPlan.pkl"
    path.write_bytes(pickle.dumps([Evil()]))
    with pytest.raises(pickle.UnpicklingError, match="os|posix"):
        load_plans(path)
    assert not marker.exists()


def test_load_plans_rejects_a_pickle_that_is_not_a_list_of_dicts(tmp_path: Path) -> None:
    path = tmp_path / "ResPlan.pkl"
    path.write_bytes(pickle.dumps({"not": "a list"}))
    with pytest.raises(ValueError, match="list of plan dicts"):
        load_plans(path)


def test_the_balcony_typo_is_fixed() -> None:
    plan = normalize_keys({"id": 1, "balacony": box(0, 0, 1, 1)})
    assert "balcony" in plan
    assert "balacony" not in plan


def test_splits_map_plan_ids_to_split_names(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"train": [1, 2], "val": [3], "test": [], "augmented": [100]}))
    assert load_splits(path) == {1: "train", 2: "train", 3: "val", 100: "augmented"}


def test_unknown_split_names_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"train": [1], "holdout": [2]}))
    with pytest.raises(ValueError, match="holdout"):
        load_splits(path)


def test_excluded_ids_skip_comments_and_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "exclude.txt"
    path.write_text("# takedown 2026-12\n12\n\n  34  # second request\n")
    assert load_excluded_ids(path) == {12, 34}


def test_no_exclusion_file_means_nothing_is_excluded() -> None:
    assert load_excluded_ids(None) == set()


def test_geometry_parts_handles_empty_single_and_multi_geometries() -> None:
    assert geometry_parts(None) == []
    assert geometry_parts(MultiPolygon()) == []
    assert len(geometry_parts(box(0, 0, 1, 1))) == 1
    assert len(geometry_parts(MultiPolygon([box(0, 0, 1, 1), box(2, 0, 3, 1)]))) == 2
    assert geometry_parts(Point(0, 0)) == [Point(0, 0)]

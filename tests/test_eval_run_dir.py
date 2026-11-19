from __future__ import annotations

from pathlib import Path

from vlm_floorplan.evaluation.run_dir import (
    MANIFEST_NAME,
    safe_name,
    verify_manifest,
    write_manifest,
)


def make_dir(root: Path) -> None:
    (root / "sub").mkdir()
    (root / "a.txt").write_text("alpha")
    (root / "sub" / "b.txt").write_text("beta")


def test_manifest_round_trip_verifies(tmp_path: Path) -> None:
    make_dir(tmp_path)
    files = write_manifest(tmp_path)
    assert set(files) == {"a.txt", "sub/b.txt"}
    assert verify_manifest(tmp_path) == []


def test_verify_reports_corruption_missing_and_extra_files(tmp_path: Path) -> None:
    make_dir(tmp_path)
    write_manifest(tmp_path)
    (tmp_path / "a.txt").write_text("changed")
    (tmp_path / "sub" / "b.txt").unlink()
    (tmp_path / "c.txt").write_text("new")
    problems = verify_manifest(tmp_path)
    assert "checksum mismatch: a.txt" in problems
    assert "missing: sub/b.txt" in problems
    assert "unlisted: c.txt" in problems


def test_missing_or_empty_manifest_is_a_problem(tmp_path: Path) -> None:
    assert verify_manifest(tmp_path) == [f"{MANIFEST_NAME} is missing"]
    (tmp_path / MANIFEST_NAME).write_text("{}")
    assert verify_manifest(tmp_path) == [f"{MANIFEST_NAME} lists no files"]


def test_safe_name_is_flat() -> None:
    assert safe_name("cubicasa5k/colorful/8759") == "cubicasa5k__colorful__8759"

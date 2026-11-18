"""Baseline job pieces that run without a GPU: spec, split packing, orchestration, dry run."""

from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest
from helpers_eval import gt_record
from PIL import Image

from vlm_floorplan.compact import to_compact
from vlm_floorplan.evaluation.baseline_job import (
    bundle_results,
    run_baseline,
    spec_from_parameters,
)
from vlm_floorplan.evaluation.baseline_spec import (
    RUNS_DIR_ENV,
    BaselineSpec,
    default_runs_dir,
    pack_split,
    unpack_split,
)
from vlm_floorplan.evaluation.client import ApiError, ChatResponse
from vlm_floorplan.evaluation.harness import load_records
from vlm_floorplan.evaluation.launcher import (
    _without_credentials,
    describe_plan,
    main,
)
from vlm_floorplan.evaluation.run_dir import verify_manifest


@pytest.fixture
def split(tmp_path: Path) -> Path:
    folder = tmp_path / "data"
    folder.mkdir()
    Image.new("RGB", (1600, 1200), "white").save(folder / "plan0.png")
    (folder / "split.jsonl").write_text(gt_record().model_dump_json() + "\n")
    return folder / "split.jsonl"


class GoodClient:
    def complete(self, request: dict[str, Any]) -> ChatResponse:
        return ChatResponse(to_compact(gt_record().graph), "", "stop", 100, 50, 0.1)


class BrokenClient:
    def complete(self, request: dict[str, Any]) -> ChatResponse:
        raise ApiError("server gone")


def test_job_parameters_round_trip_through_strings() -> None:
    spec = BaselineSpec(run_name="r", resolutions=[768, 1280], limit=20)
    rebuilt = spec_from_parameters(spec.job_parameters())
    assert rebuilt.model_dump() == {**spec.model_dump(), "project": "vlm-floorplan"}
    assert spec_from_parameters(BaselineSpec(run_name="r").job_parameters()).limit is None


def test_split_archive_keeps_images_next_to_the_jsonl(split: Path, tmp_path: Path) -> None:
    archive = tmp_path / "split.tar.gz"
    assert pack_split(split, archive) == 1
    with tarfile.open(archive) as tar:
        assert sorted(tar.getnames()) == ["plan0.png", "split.jsonl"]
    unpacked = unpack_split(archive, tmp_path / "unpacked")
    assert load_records(unpacked)[0].id == "synthetic/plan0"
    assert (unpacked.parent / "plan0.png").is_file()


def test_image_path_escaping_the_split_directory_is_refused(tmp_path: Path) -> None:
    folder = tmp_path / "data"
    folder.mkdir()
    Image.new("RGB", (10, 10)).save(tmp_path / "outside.png")
    record = gt_record().model_copy(update={"image": "../outside.png"})
    (folder / "split.jsonl").write_text(record.model_dump_json() + "\n")
    with pytest.raises(ValueError, match="leaves the split directory"):
        pack_split(folder / "split.jsonl", tmp_path / "x.tar.gz")


def test_both_resolutions_run_and_are_bundled_with_a_verifiable_manifest(
    split: Path, tmp_path: Path
) -> None:
    spec = BaselineSpec(run_name="r", resolutions=[768, 1280], concurrency=1)
    out = tmp_path / "out"
    result = run_baseline(spec, split, out, GoodClient(), "http://unused/v1")
    assert result["runs"] == ["long-side-768", "long-side-1280"]
    assert result["errors"] == {}
    assert result["scores"]["long-side-768"]["room_recall"] == 1.0
    # 1600 px is shrunk to 768 and to 1280 respectively.
    assert Image.open(out / "long-side-768/inputs/images/synthetic__plan0.png").width == 768
    assert Image.open(out / "long-side-1280/inputs/images/synthetic__plan0.png").width == 1280

    bundle_results(out, tmp_path / "run.zip")
    assert verify_manifest(out) == []
    assert json.loads((out / "job_summary.json").read_text())["runs"] == result["runs"]


def test_api_failures_are_scored_not_fatal(split: Path, tmp_path: Path) -> None:
    spec = BaselineSpec(run_name="r", resolutions=[768], concurrency=1)
    result = run_baseline(spec, split, tmp_path / "out", BrokenClient(), "http://unused/v1")
    assert result["scores"]["long-side-768"]["validity_rate"] == 0.0


def test_each_run_records_the_request_timeout_the_job_uses(split: Path, tmp_path: Path) -> None:
    spec = BaselineSpec(run_name="r", resolutions=[768], concurrency=1)
    out = tmp_path / "out"
    run_baseline(spec, split, out, GoodClient(), "http://unused/v1")
    assert json.loads((out / "long-side-768" / "config.json").read_text())["timeout_s"] == 1800.0


def test_runs_dir_default_and_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(RUNS_DIR_ENV, raising=False)
    assert default_runs_dir() == Path.home() / "Documents/pets/vlm-floorplan-runs"
    monkeypatch.setenv(RUNS_DIR_ENV, str(tmp_path))
    assert default_runs_dir() == tmp_path


def test_remote_urls_lose_embedded_credentials() -> None:
    assert (
        _without_credentials("https://user:ghp_secret@github.com/o/r.git")
        == "https://github.com/o/r.git"
    )
    assert _without_credentials("git@github.com:o/r.git") == "git@github.com:o/r.git"


def test_describe_plan_lists_the_policy_steps() -> None:
    spec = BaselineSpec(run_name="r")
    state = {"branch": "b", "commit": "a" * 40, "dirty": True, "pushed": False, "remote": "x"}
    text = describe_plan(spec, state, Path("/runs"), plans=5)
    assert "/runs/r" in text
    assert "only if all checks pass: delete the task" in text
    assert "uncommitted changes" in text
    assert "no remote branch" in text


def test_dry_run_prints_the_plan_and_never_imports_clearml(
    split: Path, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    sys.modules.pop("clearml", None)
    code = main(
        [
            "launch",
            "--split",
            str(split),
            "--run-name",
            "dry",
            "--runs-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert out.startswith("DRY RUN")
    assert "queue '3090'" in out
    assert "pack 1 plans" in out
    assert str(tmp_path / "dry") in out
    assert "clearml" not in sys.modules
    assert not (tmp_path / "dry").exists()

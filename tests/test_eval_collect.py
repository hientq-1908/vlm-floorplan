"""The collector must verify before it deletes. Tested with a fake ClearML task."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from vlm_floorplan.evaluation.baseline_job import bundle_results
from vlm_floorplan.evaluation.baseline_spec import RESULT_ARTIFACT
from vlm_floorplan.evaluation.clearml_cleanup import CleanupReport
from vlm_floorplan.evaluation.collect import (
    VERIFIED_MARKER,
    ConsoleLog,
    DownloadVerificationError,
    RemoteArtifact,
    TaskNotFinishedError,
    collect_and_delete,
)


def make_result_zip(root: Path, runs: tuple[str, ...] = ("long-side-768",)) -> Path:
    run_dir = root / "result-src"
    for run in runs:
        (run_dir / run).mkdir(parents=True)
        (run_dir / run / "metrics.json").write_text("{}")
    (run_dir / "job_summary.json").write_text(json.dumps({"runs": list(runs)}))
    return bundle_results(run_dir, root / "run.zip")


class FakeTask:
    task_id = "abc123"

    def __init__(
        self,
        artifacts: dict[str, RemoteArtifact],
        status: str = "completed",
        log_text: str = "job started\njob done\n",
    ):
        self._artifacts = artifacts
        self._status = status
        self._log = log_text
        self.deleted = False

    def status(self) -> str:
        return self._status

    def details(self) -> dict[str, Any]:
        return {"id": self.task_id}

    def scalars(self) -> dict[str, Any]:
        return {"validity_rate": {"long-side-768": {"x": [768], "y": [0.5]}}}

    def console_log(self) -> ConsoleLog:
        return ConsoleLog(text=self._log)

    def download_artifacts(self, dest: Path) -> dict[str, RemoteArtifact]:
        return self._artifacts

    def delete(self) -> list[str]:
        self.deleted = True
        return []

    def remove_empty_project(self) -> CleanupReport:
        return CleanupReport()


def good_task(tmp_path: Path, **kwargs: Any) -> FakeTask:
    archive = make_result_zip(tmp_path)
    artifact = RemoteArtifact(path=archive, expected_size=archive.stat().st_size)
    return FakeTask({RESULT_ARTIFACT: artifact}, **kwargs)


def test_verified_download_is_saved_then_task_deleted(tmp_path: Path) -> None:
    task = good_task(tmp_path)
    dest = tmp_path / "runs" / "my-run"
    collect_and_delete(task, dest)
    assert task.deleted
    assert (dest / "result" / "long-side-768" / "metrics.json").is_file()
    assert (dest / "console.log").read_text().startswith("job started")
    assert json.loads((dest / "scalars.json").read_text())["validity_rate"]
    assert json.loads((dest / "task.json").read_text()) == {"id": "abc123"}
    assert json.loads((dest / VERIFIED_MARKER).read_text())["task_id"] == "abc123"


def test_size_mismatch_blocks_deletion(tmp_path: Path) -> None:
    task = good_task(tmp_path)
    archive = task._artifacts[RESULT_ARTIFACT]
    task._artifacts[RESULT_ARTIFACT] = RemoteArtifact(archive.path, archive.expected_size + 1)
    with pytest.raises(DownloadVerificationError, match="bytes"):
        collect_and_delete(task, tmp_path / "out")
    assert not task.deleted


def test_corrupt_file_inside_the_archive_blocks_deletion(tmp_path: Path) -> None:
    src = tmp_path / "result-src"
    archive = make_result_zip(tmp_path)
    # Rewrite the zip with one file's content changed but the old manifest kept.
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(archive) as zin, zipfile.ZipFile(tampered, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item)
            zout.writestr(item, b'{"x": 1}' if item.filename.endswith("metrics.json") else data)
    assert src.exists()
    task = FakeTask({RESULT_ARTIFACT: RemoteArtifact(tampered, tampered.stat().st_size)})
    with pytest.raises(DownloadVerificationError, match="checksum mismatch"):
        collect_and_delete(task, tmp_path / "out")
    assert not task.deleted


def test_not_a_zip_blocks_deletion(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.zip"
    bogus.write_bytes(b"not a zip")
    task = FakeTask({RESULT_ARTIFACT: RemoteArtifact(bogus, bogus.stat().st_size)})
    with pytest.raises(DownloadVerificationError, match="not a zip"):
        collect_and_delete(task, tmp_path / "out")
    assert not task.deleted


def test_missing_metrics_for_an_expected_resolution_blocks_deletion(tmp_path: Path) -> None:
    run_dir = tmp_path / "result-src"
    run_dir.mkdir()
    (run_dir / "job_summary.json").write_text(json.dumps({"runs": ["long-side-1280"]}))
    archive = bundle_results(run_dir, tmp_path / "run.zip")
    task = FakeTask({RESULT_ARTIFACT: RemoteArtifact(archive, archive.stat().st_size)})
    with pytest.raises(DownloadVerificationError, match=r"long-side-1280/metrics.json"):
        collect_and_delete(task, tmp_path / "out")
    assert not task.deleted


def test_completed_task_without_result_archive_blocks_deletion(tmp_path: Path) -> None:
    task = FakeTask({})
    with pytest.raises(DownloadVerificationError, match="no 'baseline_run' artifact"):
        collect_and_delete(task, tmp_path / "out")
    assert not task.deleted


def test_failed_task_keeps_logs_and_is_deleted_after_download(tmp_path: Path) -> None:
    task = FakeTask({}, status="failed", log_text="Traceback ...\n")
    collect_and_delete(task, tmp_path / "out")
    assert task.deleted
    assert (tmp_path / "out" / "console.log").read_text().startswith("Traceback")


def test_empty_console_log_blocks_deletion(tmp_path: Path) -> None:
    task = good_task(tmp_path, log_text="")
    with pytest.raises(DownloadVerificationError, match="console log"):
        collect_and_delete(task, tmp_path / "out")
    assert not task.deleted


def test_existing_destination_is_never_overwritten(tmp_path: Path) -> None:
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "precious.txt").write_text("keep")
    task = good_task(tmp_path)
    with pytest.raises(FileExistsError):
        collect_and_delete(task, dest)
    assert not task.deleted
    assert (dest / "precious.txt").read_text() == "keep"


@pytest.mark.parametrize("status", ["queued", "in_progress", "created"])
def test_unfinished_task_is_refused_before_download_or_deletion(
    tmp_path: Path, status: str
) -> None:
    task = FakeTask({}, status=status, log_text="still generating\n")
    with pytest.raises(TaskNotFinishedError, match=status):
        collect_and_delete(task, tmp_path / "out")
    assert not task.deleted
    assert not (tmp_path / "out").exists()

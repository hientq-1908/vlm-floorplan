"""Download a finished ClearML task, verify the copy, and only then delete the task.

Captain policy: ClearML is compute only. After a job finishes, every artifact, log and
metric is copied to a local runs directory, and the task is deleted from ClearML.
The order is the whole point: *verify, then delete*. A task is never deleted while any
check fails, because after deletion the remote copy is gone for good.

What "verified" means here:

1. every artifact file exists locally and has the byte size ClearML reports for it,
2. the result archive is a readable zip,
3. the job's own `MANIFEST.json` (SHA-256 of every file it wrote) matches the extracted
   files, so corruption or a partial download cannot pass,
4. a finished task has its result archive and one `metrics.json` per expected resolution,
5. task details, scalars and the console log are saved.

The ClearML calls live behind the `RemoteTask` protocol so this logic is tested with a
fake. The real adapter is in `scripts/clearml_baseline.py`.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from vlm_floorplan.evaluation.baseline_spec import RESULT_ARTIFACT
from vlm_floorplan.evaluation.run_dir import verify_manifest

COMPLETED = "completed"
FINISHED_STATUSES = frozenset({COMPLETED, "failed", "stopped", "closed", "published"})
"""ClearML statuses after which the task no longer runs. Anything else may still be working."""
VERIFIED_MARKER = "DOWNLOAD_VERIFIED.json"
EXPECTED_RUNS_FILE = "job_summary.json"


class DownloadVerificationError(RuntimeError):
    """The local copy is incomplete or corrupt. The remote task was NOT deleted."""


class TaskNotFinishedError(DownloadVerificationError):
    """The task is still queued or running. Nothing was downloaded or deleted."""


@dataclass(frozen=True)
class RemoteArtifact:
    path: Path
    """Local file the artifact was downloaded to."""
    expected_size: int | None
    """Size in bytes that ClearML reports, if it reports one."""


@dataclass
class ConsoleLog:
    text: str
    truncated: bool = False


class RemoteTask(Protocol):
    task_id: str

    def status(self) -> str: ...
    def details(self) -> dict[str, Any]: ...
    def scalars(self) -> dict[str, Any]: ...
    def console_log(self) -> ConsoleLog: ...
    def download_artifacts(self, dest: Path) -> dict[str, RemoteArtifact]: ...
    def delete(self) -> None: ...


@dataclass
class CollectReport:
    dest: Path
    status: str
    checks: list[str] = field(default_factory=list)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")


def _check_artifacts(artifacts: dict[str, RemoteArtifact]) -> list[str]:
    checks = []
    for name, artifact in artifacts.items():
        if not artifact.path.is_file():
            raise DownloadVerificationError(f"artifact {name!r} was not downloaded")
        size = artifact.path.stat().st_size
        if artifact.expected_size is not None and size != artifact.expected_size:
            raise DownloadVerificationError(
                f"artifact {name!r} is {size} bytes, ClearML reports {artifact.expected_size}"
            )
        checks.append(f"artifact {name}: {size} bytes")
    return checks


def _extract_result(archive: Path, dest: Path) -> Path:
    try:
        with zipfile.ZipFile(archive) as zf:
            bad = zf.testzip()
            if bad is not None:
                raise DownloadVerificationError(f"result archive is corrupt at {bad}")
            zf.extractall(dest)
    except zipfile.BadZipFile as err:
        raise DownloadVerificationError(f"result archive is not a zip: {err}") from err
    return dest


def _check_result(result_dir: Path) -> list[str]:
    problems = verify_manifest(result_dir)
    if problems:
        raise DownloadVerificationError("manifest check failed: " + "; ".join(problems[:5]))
    checks = ["manifest: every file matches its SHA-256"]
    summary_path = result_dir / EXPECTED_RUNS_FILE
    if not summary_path.is_file():
        raise DownloadVerificationError(f"{EXPECTED_RUNS_FILE} is missing from the result")
    for run in json.loads(summary_path.read_text()).get("runs", []):
        metrics = result_dir / run / "metrics.json"
        if not metrics.is_file():
            raise DownloadVerificationError(f"{run}/metrics.json is missing from the result")
        checks.append(f"{run}/metrics.json present")
    return checks


def collect_and_delete(task: RemoteTask, dest: Path) -> CollectReport:
    """Download everything to `dest`, verify it, then delete the remote task.

    Raises `DownloadVerificationError` (and leaves the task alone) when anything is off.
    Failed or stopped tasks are downloaded too, since their logs are the point; they
    only need the logs, not the result archive. A task that has not finished is refused
    before anything is touched, since deleting it would kill the job.
    """
    status = task.status()
    if status not in FINISHED_STATUSES:
        raise TaskNotFinishedError(f"task {task.task_id} is {status!r}; collect it once it ends")
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"{dest} already has files; choose another run name")
    dest.mkdir(parents=True, exist_ok=True)

    checks: list[str] = []
    _write_json(dest / "task.json", task.details())
    _write_json(dest / "scalars.json", task.scalars())
    log = task.console_log()
    (dest / "console.log").write_text(log.text)
    if not log.text.strip():
        raise DownloadVerificationError("the console log is empty")
    checks.append(
        "console log saved" + (" (truncated, see job.log in the result)" if log.truncated else "")
    )

    artifacts = task.download_artifacts(dest / "artifacts")
    checks += _check_artifacts(artifacts)
    if RESULT_ARTIFACT in artifacts:
        result_dir = _extract_result(artifacts[RESULT_ARTIFACT].path, dest / "result")
        checks += _check_result(result_dir)
    elif status == COMPLETED:
        raise DownloadVerificationError(f"completed task has no {RESULT_ARTIFACT!r} artifact")
    else:
        checks.append(f"task ended as {status!r} without a result archive; logs kept")

    _write_json(
        dest / VERIFIED_MARKER,
        {
            "task_id": task.task_id,
            "status": status,
            "verified_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "checks": checks,
        },
    )
    task.delete()
    return CollectReport(dest=dest, status=status, checks=checks)

"""The SFT job and launcher pieces that run without a GPU or ClearML.

Like the baseline tests: the spec, the dry run, the bundle written even when training fails, and
the verify-then-delete order against a fake ClearML task.
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Any

import pytest
from sft_fixture import make_sample, write_samples

from vlm_floorplan.evaluation.collect import (
    ConsoleLog,
    DownloadVerificationError,
    RemoteArtifact,
    collect_and_delete,
)
from vlm_floorplan.evaluation.launcher import REPO_ROOT
from vlm_floorplan.training import job as job_module
from vlm_floorplan.training.job import (
    RESULT_ARTIFACT,
    SftSpec,
    agent_config,
    check_sft_result,
    run_job,
    spec_from_parameters,
)
from vlm_floorplan.training.launcher import describe_plan, main, pack_train_file
from vlm_floorplan.training.train import run_dir_for


@pytest.fixture
def train_file(tmp_path: Path) -> Path:
    return write_samples(tmp_path / "train.jsonl", [make_sample("A"), make_sample("B")])


def test_job_parameters_round_trip_through_strings() -> None:
    spec = SftSpec(
        queue="gpu", run_name="r", overrides=["train.learning_rate=5e-5", "lora.r=16"], smoke=True
    )
    rebuilt = spec_from_parameters(spec.job_parameters())
    assert (rebuilt.run_name, rebuilt.overrides, rebuilt.smoke) == ("r", spec.overrides, True)


def test_agent_config_uses_the_agents_paths_and_reports_to_clearml(tmp_path: Path) -> None:
    spec = SftSpec(queue="gpu", run_name="r", overrides=["train.learning_rate=5e-5"])
    config = agent_config(
        spec, REPO_ROOT, tmp_path / "t.jsonl", tmp_path / "run", report_to="clearml"
    )
    assert config.train.learning_rate == 5e-5
    assert config.data.train_file == tmp_path / "t.jsonl"
    assert config.train.output_dir == tmp_path / "run"
    assert config.train.report_to == "clearml"


def test_smoke_job_does_not_report_to_clearml(tmp_path: Path) -> None:
    spec = SftSpec(queue="gpu", run_name="r", smoke=True)
    config = agent_config(spec, REPO_ROOT, tmp_path / "t", tmp_path, report_to="clearml")
    assert config.train.report_to == "none"


def test_a_local_full_job_keeps_the_config_report_to(tmp_path: Path) -> None:
    """Driving the job locally (tests, debugging) must not report to ClearML."""
    config = agent_config(
        SftSpec(queue="gpu", run_name="r"), REPO_ROOT, tmp_path / "t", tmp_path / "run"
    )
    assert config.train.report_to == "none"


def test_pack_train_file_counts_and_gzips(train_file: Path, tmp_path: Path) -> None:
    import gzip

    archive = tmp_path / "train.jsonl.gz"
    assert pack_train_file(train_file, archive) == 2
    assert gzip.decompress(archive.read_bytes()) == train_file.read_bytes()


def test_pack_train_file_refuses_a_malformed_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "A"}\n')
    with pytest.raises(ValueError, match="messages"):
        pack_train_file(bad, tmp_path / "x.gz")


def test_dry_run_prints_the_plan_and_touches_nothing(
    train_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "launch",
            "--train-file",
            str(train_file),
            "--queue",
            "gpu",
            "--dry-run",
            "--smoke",
            "--run-name",
            "r1",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "DRY RUN" in out and "2 training samples" in out
    assert "smoke = true" in out and "smoke test" in out


def test_dry_run_rejects_a_misspelled_override(train_file: Path) -> None:
    with pytest.raises(ValueError, match="unknown config key"):
        main(
            [
                "launch",
                "--train-file",
                str(train_file),
                "--queue",
                "gpu",
                "--dry-run",
                "--set",
                "train.lerning_rate=1",
            ]
        )


def test_queue_is_required(train_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as err:
        main(["launch", "--train-file", str(train_file), "--dry-run"])
    assert err.value.code == 2
    assert "--queue" in capsys.readouterr().err


def test_describe_plan_warns_about_a_dirty_tree(tmp_path: Path) -> None:
    state = {"branch": "b", "commit": "c" * 40, "remote": "r", "dirty": True, "pushed": False}
    text = describe_plan(SftSpec(queue="gpu", run_name="r"), state, tmp_path, 5)
    assert "uncommitted changes" in text


# --- the job: the bundle is written whatever happens -------------------------------------------


def test_failed_training_still_produces_a_verifiable_bundle(
    train_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(job_module, "run_training", explode)
    archive = tmp_path / "run.zip"
    result = run_job(
        SftSpec(queue="gpu", run_name="r"), REPO_ROOT, train_file, tmp_path / "out", archive
    )

    assert result["ok"] is False and "CUDA out of memory" in result["error"]
    extracted = tmp_path / "extracted"
    zipfile.ZipFile(archive).extractall(extracted)
    assert "CUDA out of memory" in (extracted / "job.log").read_text()
    checks = check_sft_result(extracted)
    assert any("did not succeed" in c for c in checks)


def fake_training(files: list[str]) -> Any:
    def run(config: Any, *, smoke: bool) -> Path:
        run_dir = run_dir_for(config.train.output_dir, smoke)
        for name in files:
            (run_dir / name).parent.mkdir(parents=True, exist_ok=True)
            (run_dir / name).write_text("x")
        return run_dir

    return run


FULL_OUTPUTS = [
    "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors",
    "train_summary.json",
    "data_report.json",
]


def bundle(
    tmp_path: Path,
    train_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    files: list[str],
    smoke: bool = False,
) -> Path:
    monkeypatch.setattr(job_module, "run_training", fake_training(files))
    archive = tmp_path / "run.zip"
    result = run_job(
        SftSpec(queue="gpu", run_name="r", smoke=smoke),
        REPO_ROOT,
        train_file,
        tmp_path / "out",
        archive,
    )
    assert result["ok"]
    return archive


def test_successful_bundle_passes_the_collector_check(
    train_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = bundle(tmp_path, train_file, monkeypatch, FULL_OUTPUTS)
    extracted = tmp_path / "x"
    zipfile.ZipFile(archive).extractall(extracted)
    checks = check_sft_result(extracted)
    assert "run/adapter/adapter_model.safetensors present" in checks


def test_missing_adapter_blocks_the_check(
    train_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = bundle(tmp_path, train_file, monkeypatch, ["train_summary.json", "data_report.json"])
    extracted = tmp_path / "x"
    zipfile.ZipFile(archive).extractall(extracted)
    with pytest.raises(DownloadVerificationError, match="adapter_config.json"):
        check_sft_result(extracted)


def test_smoke_bundle_needs_smoke_json(
    train_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = bundle(tmp_path, train_file, monkeypatch, ["smoke.json"], smoke=True)
    extracted = tmp_path / "x"
    zipfile.ZipFile(archive).extractall(extracted)
    assert any("smoke.json" in c for c in check_sft_result(extracted))


def test_smoke_bundle_without_smoke_json_blocks_the_check(
    train_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = bundle(tmp_path, train_file, monkeypatch, ["data_report.json"], smoke=True)
    extracted = tmp_path / "x"
    zipfile.ZipFile(archive).extractall(extracted)
    with pytest.raises(DownloadVerificationError, match="smoke.json"):
        check_sft_result(extracted)


# --- verify, then delete -----------------------------------------------------------------------


class FakeTask:
    task_id = "sft123"

    def __init__(self, artifacts: dict[str, RemoteArtifact], status: str = "completed") -> None:
        self._artifacts = artifacts
        self._status = status
        self.deleted = False

    def status(self) -> str:
        return self._status

    def details(self) -> dict[str, Any]:
        return {"id": self.task_id}

    def scalars(self) -> dict[str, Any]:
        return {}

    def console_log(self) -> ConsoleLog:
        return ConsoleLog(text="training started\n")

    def download_artifacts(self, dest: Path) -> dict[str, RemoteArtifact]:
        return self._artifacts

    def delete(self) -> None:
        self.deleted = True

    def remove_empty_project(self) -> list[str]:
        return []


def collect(task: FakeTask, dest: Path) -> None:
    collect_and_delete(task, dest, result_artifact=RESULT_ARTIFACT, check_result=check_sft_result)


def test_complete_download_is_verified_then_the_task_is_deleted(
    train_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = bundle(tmp_path, train_file, monkeypatch, FULL_OUTPUTS)
    task = FakeTask({RESULT_ARTIFACT: RemoteArtifact(archive, archive.stat().st_size)})
    collect(task, tmp_path / "runs" / "r")
    assert task.deleted
    assert (
        tmp_path / "runs" / "r" / "result" / "run" / "adapter" / "adapter_model.safetensors"
    ).is_file()


def test_incomplete_result_keeps_the_task(
    train_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = bundle(tmp_path, train_file, monkeypatch, ["data_report.json"])
    task = FakeTask({RESULT_ARTIFACT: RemoteArtifact(archive, archive.stat().st_size)})
    with pytest.raises(DownloadVerificationError):
        collect(task, tmp_path / "runs" / "r")
    assert not task.deleted


def test_completed_task_without_the_result_artifact_keeps_the_task(tmp_path: Path) -> None:
    task = FakeTask({})
    with pytest.raises(DownloadVerificationError, match="sft_run"):
        collect(task, tmp_path / "out")
    assert not task.deleted


def test_tampered_adapter_keeps_the_task(
    train_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = bundle(tmp_path, train_file, monkeypatch, FULL_OUTPUTS)
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(archive) as zin, zipfile.ZipFile(tampered, "w") as zout:
        for item in zin.infolist():
            data = b"corrupt" if item.filename.endswith(".safetensors") else zin.read(item)
            zout.writestr(item, data)
    task = FakeTask({RESULT_ARTIFACT: RemoteArtifact(tampered, tampered.stat().st_size)})
    with pytest.raises(DownloadVerificationError, match="checksum mismatch"):
        collect(task, tmp_path / "download")
    assert not task.deleted


def test_failed_task_logs_are_kept_and_the_task_deleted(tmp_path: Path) -> None:
    task = FakeTask({}, status="failed")
    collect(task, tmp_path / "out")
    assert task.deleted
    assert (
        json.loads((tmp_path / "out" / "DOWNLOAD_VERIFIED.json").read_text())["status"] == "failed"
    )


# --- ClearML: metrics only, and only from the agent ---------------------------------------------


def capture_training(seen: dict[str, Any]) -> Any:
    def run(config: Any, *, smoke: bool) -> Path:
        seen["report_to"] = config.train.report_to
        seen["log_model"] = os.environ.get("CLEARML_LOG_MODEL")
        return run_dir_for(config.train.output_dir, smoke)

    return run


def test_a_local_full_job_creates_no_clearml_task(
    train_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from clearml import Task

    seen: dict[str, Any] = {}
    monkeypatch.setattr(job_module, "run_training", capture_training(seen))
    out = tmp_path / "out"
    result = run_job(SftSpec(queue="gpu", run_name="r"), REPO_ROOT, train_file, out, tmp_path / "z")
    assert result["ok"] and seen["report_to"] == "none"
    assert Task.current_task() is None


def test_a_job_reporting_to_clearml_does_not_upload_checkpoints(
    train_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """transformers' ClearMLCallback uploads every checkpoint unless CLEARML_LOG_MODEL is false."""
    monkeypatch.delenv("CLEARML_LOG_MODEL", raising=False)
    seen: dict[str, Any] = {}
    monkeypatch.setattr(job_module, "run_training", capture_training(seen))
    run_job(
        SftSpec(queue="gpu", run_name="r"),
        REPO_ROOT,
        train_file,
        tmp_path / "out",
        tmp_path / "z",
        report_to="clearml",
    )
    assert seen == {"report_to": "clearml", "log_model": "false"}

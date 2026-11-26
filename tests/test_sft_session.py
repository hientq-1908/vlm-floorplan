"""Stopping on a time limit and resuming a run, without training anything."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("transformers")

from vlm_floorplan.training.config import load_config  # noqa: E402
from vlm_floorplan.training.session import (  # noqa: E402
    TimeLimit,
    check_resume,
    latest_checkpoint,
)


class FakeClock:
    now = 1000.0

    def __call__(self) -> float:
        return self.now


def step_end(limit: TimeLimit) -> SimpleNamespace:
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    limit.on_step_end(None, None, control)
    return control


def test_time_limit_lets_training_go_on_before_the_deadline() -> None:
    clock = FakeClock()
    limit = TimeLimit(hours=2, clock=clock)
    limit.on_train_begin(None, None, None)
    clock.now += 2 * 3600 - 1
    control = step_end(limit)
    assert not control.should_save and not control.should_training_stop
    assert not limit.expired


def test_time_limit_saves_a_snapshot_and_stops_at_the_deadline() -> None:
    clock = FakeClock()
    limit = TimeLimit(hours=2, clock=clock)
    limit.on_train_begin(None, None, None)
    clock.now += 2 * 3600
    control = step_end(limit)
    assert control.should_save and control.should_training_stop
    assert limit.expired


def make_run(run_dir: Path, *, snapshots: dict[int, bool], settings: dict) -> None:
    """A run directory with `run_config.json` and snapshots (step to whether it has optimizer)."""
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(json.dumps(settings))
    for step, with_optimizer in snapshots.items():
        (run_dir / f"checkpoint-{step}").mkdir()
        if with_optimizer:
            (run_dir / f"checkpoint-{step}" / "optimizer.pt").write_bytes(b"x")


@pytest.fixture
def settings() -> dict:
    return {
        "smoke": False,
        **load_config(overrides=["train.resumable=true"]).model_dump(mode="json"),
    }


def test_latest_checkpoint_is_the_highest_step_not_the_last_name(tmp_path: Path) -> None:
    make_run(tmp_path / "run", snapshots={9: True, 100: True, 25: True}, settings={})
    (tmp_path / "run" / "checkpoint-final").mkdir()
    assert latest_checkpoint(tmp_path / "run") == tmp_path / "run" / "checkpoint-100"
    assert latest_checkpoint(tmp_path) is None


def test_resume_picks_the_latest_snapshot(tmp_path: Path, settings: dict) -> None:
    make_run(tmp_path / "run", snapshots={25: True, 50: True}, settings=settings)
    snapshot = check_resume(tmp_path / "run", settings, "run_config.json", "adapter")
    assert snapshot.name == "checkpoint-50"


def test_resume_allows_a_different_time_limit_per_session(tmp_path: Path, settings: dict) -> None:
    make_run(tmp_path / "run", snapshots={25: True}, settings=settings)
    later = {**settings, "train": {**settings["train"], "time_limit_hours": 5.0}}
    assert check_resume(tmp_path / "run", later, "run_config.json", "adapter")


def test_resume_refuses_changed_settings(tmp_path: Path, settings: dict) -> None:
    make_run(tmp_path / "run", snapshots={25: True}, settings=settings)
    changed = {**settings, "train": {**settings["train"], "learning_rate": 5e-5}}
    with pytest.raises(ValueError, match="settings differ"):
        check_resume(tmp_path / "run", changed, "run_config.json", "adapter")


def test_resume_refuses_a_snapshot_without_optimizer_state(tmp_path: Path, settings: dict) -> None:
    make_run(tmp_path / "run", snapshots={25: False}, settings=settings)
    with pytest.raises(FileNotFoundError, match="train.resumable"):
        check_resume(tmp_path / "run", settings, "run_config.json", "adapter")


def test_resume_refuses_a_run_that_has_no_snapshot(tmp_path: Path, settings: dict) -> None:
    make_run(tmp_path / "run", snapshots={}, settings=settings)
    with pytest.raises(FileNotFoundError, match="no snapshot"):
        check_resume(tmp_path / "run", settings, "run_config.json", "adapter")


def test_resume_refuses_a_finished_run(tmp_path: Path, settings: dict) -> None:
    make_run(tmp_path / "run", snapshots={25: True}, settings=settings)
    (tmp_path / "run" / "adapter").mkdir()
    with pytest.raises(FileExistsError, match="already finished"):
        check_resume(tmp_path / "run", settings, "run_config.json", "adapter")


def test_resume_refuses_a_directory_that_was_never_a_run(tmp_path: Path, settings: dict) -> None:
    (tmp_path / "run").mkdir()
    with pytest.raises(FileNotFoundError, match="run_config.json"):
        check_resume(tmp_path / "run", settings, "run_config.json", "adapter")

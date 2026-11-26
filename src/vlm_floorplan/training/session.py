"""Stopping and resuming a run across notebook sessions.

Free notebook sessions end after a fixed time (Kaggle: 12 h) and then lose everything that is not
saved. `TimeLimit` ends a run cleanly with a full snapshot a little before that, and
`check_resume` decides whether a run directory can be continued with `--resume`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

CHECKPOINT_PREFIX = "checkpoint-"
UNCHANGED_ON_RESUME_EXCEPT = ("time_limit_hours",)
"""Settings that may differ between the sessions of one run (the limit is per session)."""


class TimeLimit(TrainerCallback):
    """Ask the trainer to save a snapshot and stop once `hours` have passed since training began."""

    def __init__(self, hours: float, clock: Callable[[], float] = time.monotonic):
        self.seconds = hours * 3600
        self._clock = clock
        self._start = 0.0
        self.expired = False

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._start = self._clock()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self._clock() - self._start >= self.seconds:
            self.expired = True
            control.should_save = True
            control.should_training_stop = True


def latest_checkpoint(run_dir: Path) -> Path | None:
    """The snapshot with the highest step in `run_dir`, or None."""
    found = [
        path
        for path in run_dir.glob(f"{CHECKPOINT_PREFIX}*")
        if path.is_dir() and path.name.removeprefix(CHECKPOINT_PREFIX).isdigit()
    ]
    return max(found, key=lambda path: int(path.name.removeprefix(CHECKPOINT_PREFIX)), default=None)


def comparable_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """The resolved settings of a run without the ones allowed to change between sessions."""
    train = {k: v for k, v in settings["train"].items() if k not in UNCHANGED_ON_RESUME_EXCEPT}
    return {**settings, "train": train}


def check_resume(
    run_dir: Path, settings: dict[str, Any], config_file: str, adapter_dir: str
) -> Path:
    """Fail unless `run_dir` holds an unfinished run with the same settings; returns its snapshot.

    `settings` is what this session would write to `config_file`. A snapshot without optimizer
    state cannot continue a run, which is why the run must have been started with
    `train.resumable = true`.
    """
    if (run_dir / adapter_dir).exists():
        raise FileExistsError(f"{run_dir} already finished: it has a final {adapter_dir}/")
    saved = run_dir / config_file
    if not saved.is_file():
        raise FileNotFoundError(f"cannot resume: {saved} does not exist")
    if comparable_settings(json.loads(saved.read_text())) != comparable_settings(settings):
        raise ValueError(
            f"cannot resume {run_dir}: the settings differ from the ones in {config_file}"
        )
    snapshot = latest_checkpoint(run_dir)
    if snapshot is None or not (snapshot / "optimizer.pt").is_file():
        raise FileNotFoundError(
            f"cannot resume: no snapshot with optimizer state in {run_dir} "
            "(start the run with train.resumable = true)"
        )
    return snapshot

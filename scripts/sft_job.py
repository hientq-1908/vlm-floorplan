"""ClearML entry point of the SFT job. Runs on the GPU agent, not locally.

The launcher (`scripts/clearml_sft.py`) creates the task, attaches the repository source and the
training file as artifacts and sets the parameters read here. `scripts/sft_bootstrap.py` unpacks
the source and runs this script with `uv run` in the synced project environment.
This script only does the ClearML glue: read the parameters, unpack the training file, run the job,
upload one result archive. The real work is in `vlm_floorplan.training.job`.

The exit code is non-zero when training failed, so ClearML marks the task failed and the
collector knows to look at the logs.
"""

from __future__ import annotations

import gzip
import logging
import shutil
import sys
import tempfile
from pathlib import Path

from clearml import Task

from vlm_floorplan.training.job import (
    RESULT_ARTIFACT,
    TRAIN_ARTIFACT,
    SftSpec,
    run_job,
    spec_from_parameters,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    task = Task.init(project_name="vlm-floorplan", task_name="sft", reuse_last_task_id=False)
    defaults = SftSpec(run_name="sft", queue="unset").job_parameters()
    spec = spec_from_parameters(task.connect(defaults))

    work = Path(tempfile.mkdtemp(prefix="sft-"))
    packed = Path(task.artifacts[TRAIN_ARTIFACT].get_local_copy(extract_archive=False))
    train_file = work / "train.jsonl"
    with gzip.open(packed, "rb") as src, train_file.open("wb") as dst:
        shutil.copyfileobj(src, dst)

    result = run_job(
        spec, REPO_ROOT, train_file, work / "run", work / "run.zip", report_to="clearml"
    )
    task.upload_artifact(
        RESULT_ARTIFACT, artifact_object=str(work / "run.zip"), wait_on_upload=True
    )
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

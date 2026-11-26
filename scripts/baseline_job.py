"""ClearML entry point of the zero-shot baseline job. Runs on the GPU agent, not locally.

The launcher (`scripts/clearml_baseline.py`) creates the task, attaches the eval split
as an artifact, and sets the parameters read here. This script only does the ClearML
glue: read parameters, unpack the split, run the job, upload one result archive and
report the headline numbers as scalars. The real work is in
`vlm_floorplan.evaluation.baseline_job`.

The exit code is non-zero when any evaluation failed, so ClearML marks the task failed
and the collector knows to look at the logs.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

# The agent clones the repo but does not install it, so make `src/` importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clearml import Task  # noqa: E402

from vlm_floorplan.evaluation.baseline_job import run_job, spec_from_parameters  # noqa: E402
from vlm_floorplan.evaluation.baseline_spec import (  # noqa: E402
    RESULT_ARTIFACT,
    SPLIT_ARTIFACT,
    BaselineSpec,
    unpack_sets,
    unpack_split,
)

HEADLINE_METRICS = ["validity_rate", "room_recall", "room_precision", "type_accuracy"]


def report_scalars(task: Task, result: dict) -> None:
    """One scalar series per metric, with the resolution as the x axis."""
    reporter = task.get_logger()
    for run_name, overall in result["scores"].items():
        side = int(run_name.rsplit("-", 1)[1])
        for metric in HEADLINE_METRICS:
            if overall.get(metric) is not None:
                reporter.report_scalar(metric, run_name, overall[metric], iteration=side)


def report_text_scalars(task: Task, result: dict) -> None:
    """Headline text-to-graph numbers, one series per run (iteration 0)."""
    reporter = task.get_logger()
    for run_name, summary in result["scores"].items():
        for metric in ("tool_call_rate", "validity_rate"):
            if summary.get(metric) is not None:
                reporter.report_scalar(metric, run_name, summary[metric], iteration=0)
        all_met = (summary.get("strict") or {}).get("all_met_rate")
        if all_met is not None:
            reporter.report_scalar("all_met_rate", run_name, all_met, iteration=0)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    task = Task.init(project_name="vlm-floorplan", task_name="baseline", reuse_last_task_id=False)
    defaults = BaselineSpec(run_name="baseline").job_parameters()
    spec = spec_from_parameters(task.connect(defaults))

    work = Path(tempfile.mkdtemp(prefix="baseline-"))
    archive = Path(task.artifacts[SPLIT_ARTIFACT].get_local_copy(extract_archive=False))
    text = spec.task == "text2graph"
    split = (unpack_sets if text else unpack_split)(archive, work / "split")

    result = run_job(spec, split, work / "run", work / "run.zip")
    task.upload_artifact(
        RESULT_ARTIFACT, artifact_object=str(work / "run.zip"), wait_on_upload=True
    )
    (report_text_scalars if text else report_scalars)(task, result)
    return 1 if result["errors"] or not result["runs"] else 0


if __name__ == "__main__":
    sys.exit(main())

"""Launch SFT on ClearML, then download, verify, and delete the task.

    uv run --group ml python scripts/clearml_sft.py launch --train-file TRAIN.jsonl --dry-run
    uv run --group ml python scripts/clearml_sft.py launch --train-file TRAIN.jsonl --smoke --wait
    uv run --group ml python scripts/clearml_sft.py launch --train-file TRAIN.jsonl --wait
    uv run --group ml python scripts/clearml_sft.py collect --task-id TASK_ID --run-name NAME

This follows the zero-shot baseline launcher (`evaluation/launcher.py`) step for step: the
training file travels as a task artifact, the task is created from a pushed git commit, and
`collect` downloads everything, verifies it, and only then deletes the task. The ClearML adapter
(`ClearMlTask`), git checks and waiting loop are shared with that launcher.

Credentials are never read or written here; ClearML finds them in `~/clearml.conf` or the
`CLEARML_*` environment variables. `--dry-run` touches neither ClearML nor the network.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vlm_floorplan.evaluation.baseline_spec import default_runs_dir
from vlm_floorplan.evaluation.collect import (
    CollectReport,
    DownloadVerificationError,
    collect_and_delete,
)
from vlm_floorplan.evaluation.launcher import (
    REPO_ROOT,
    ClearMlTask,
    git_state,
    wait_until_finished,
)
from vlm_floorplan.training.config import load_config
from vlm_floorplan.training.data import read_samples
from vlm_floorplan.training.job import RESULT_ARTIFACT, TRAIN_ARTIFACT, SftSpec, check_sft_result

JOB_SCRIPT = "scripts/sft_job.py"
REQUIREMENTS = "deploy/sft/requirements-sft.txt"


def pack_train_file(train_file: Path, archive: Path) -> int:
    """Validate the training file and gzip it into `archive`; returns the sample count."""
    count = len(read_samples(train_file))  # fails now, not after hours in the queue
    with train_file.open("rb") as src, gzip.open(archive, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return count


def describe_plan(spec: SftSpec, state: dict[str, Any], runs_dir: Path, samples: int) -> str:
    """The dry-run report: what launching would do, in order."""
    mode = "smoke test (a few steps, writes smoke.json)" if spec.smoke else "full training"
    steps = [
        f"gzip and attach the {samples} training samples as artifact {TRAIN_ARTIFACT!r}",
        f"create task {spec.run_name!r} in project {spec.project!r} from "
        f"{state['branch']} @ {state['commit'][:10]} ({state['remote']})",
        f"install the packages in {REQUIREMENTS} on the agent",
        f"enqueue on queue {spec.queue!r}",
        f"on the agent: {mode} with {spec.config}, upload one result archive",
        "wait for the job to finish",
        f"download every artifact, the console log, scalars and task details to "
        f"{runs_dir / spec.run_name}",
        "verify: artifact sizes, zip integrity, SHA-256 manifest, the adapter (or smoke.json)",
        "only if all checks pass: delete the task and its artifacts from ClearML",
    ]
    warnings = []
    if state["dirty"]:
        warnings.append("working tree has uncommitted changes (the agent only sees commits)")
    if not state["pushed"]:
        warnings.append("this commit is on no remote branch, so the agent cannot clone it")
    lines = ["DRY RUN: nothing was sent to ClearML", "", "parameters:"]
    lines += [f"  {key} = {value}" for key, value in spec.job_parameters().items()]
    lines += ["", "steps:"] + [f"  {i}. {step}" for i, step in enumerate(steps, start=1)]
    if warnings:
        lines += ["", "warnings:"] + [f"  - {w}" for w in warnings]
    return "\n".join(lines)


def create_and_enqueue(spec: SftSpec, train_file: Path, state: dict[str, Any]) -> str:
    from clearml import Task

    with tempfile.TemporaryDirectory(prefix="sft-train-") as tmp:
        archive = Path(tmp) / "train.jsonl.gz"
        pack_train_file(train_file, archive)
        task = Task.create(
            project_name=spec.project,
            task_name=spec.run_name,
            task_type="training",
            repo=str(REPO_ROOT),
            branch=state["branch"],
            commit=state["commit"],
            script=JOB_SCRIPT,
            working_directory=".",
            requirements_file=str(REPO_ROOT / REQUIREMENTS),
            docker=spec.docker_image,
            add_task_init_call=False,
        )
        task.set_parameters({f"General/{k}": v for k, v in spec.job_parameters().items()})
        task.upload_artifact(TRAIN_ARTIFACT, artifact_object=str(archive), wait_on_upload=True)
    Task.enqueue(task, queue_name=spec.queue)
    return task.id


def _collect(args: argparse.Namespace, task_id: str, run_name: str) -> CollectReport:
    report = collect_and_delete(
        ClearMlTask(task_id),
        args.runs_dir / run_name,
        result_artifact=RESULT_ARTIFACT,
        check_result=check_sft_result,
    )
    print(f"downloaded and verified: {report.dest}")
    for check in report.checks:
        print(f"  ok: {check}")
    print("ClearML task deleted")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    launch = sub.add_parser("launch", help="create and enqueue the SFT task")
    launch.add_argument("--train-file", type=Path, required=True, help="training JSONL")
    launch.add_argument("--run-name", default=datetime.now(UTC).strftime("sft-%Y%m%d-%H%M"))
    launch.add_argument("--config", default="configs/sft_stage1.toml", help="path in the repo")
    launch.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE"
    )
    launch.add_argument("--smoke", action="store_true", help="a few steps; writes smoke.json")
    launch.add_argument("--queue", default="3090")
    launch.add_argument("--project", default="vlm-floorplan")
    launch.add_argument("--docker-image")
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("--wait", action="store_true", help="wait, then collect and delete")
    launch.add_argument("--poll-seconds", type=float, default=60.0)
    launch.add_argument("--allow-dirty", action="store_true")
    launch.add_argument("--runs-dir", type=Path, default=default_runs_dir())

    collect = sub.add_parser("collect", help="download, verify, then delete a finished task")
    collect.add_argument("--task-id", required=True)
    collect.add_argument("--run-name", required=True, help="local directory name")
    collect.add_argument("--runs-dir", type=Path, default=default_runs_dir())
    return parser


def _launch(args: argparse.Namespace) -> int:
    spec = SftSpec(
        run_name=args.run_name,
        project=args.project,
        queue=args.queue,
        config=args.config,
        overrides=args.overrides,
        smoke=args.smoke,
        docker_image=args.docker_image,
    )
    load_config(REPO_ROOT / spec.config, spec.overrides)  # a typo in a setting fails here
    state = git_state()
    if args.dry_run:
        with tempfile.TemporaryDirectory(prefix="sft-dry-") as tmp:
            samples = pack_train_file(args.train_file, Path(tmp) / "train.jsonl.gz")
        print(describe_plan(spec, state, args.runs_dir, samples))
        return 0
    if state["dirty"] and not args.allow_dirty:
        sys.exit("refusing to launch: uncommitted changes (the agent only sees commits)")
    if not state["pushed"]:
        sys.exit("refusing to launch: this commit is not on any remote branch; push it first")
    task_id = create_and_enqueue(spec, args.train_file, state)
    print(f"enqueued task {task_id} on queue {spec.queue!r}")
    if not args.wait:
        print(f"when it finishes: collect --task-id {task_id} --run-name {spec.run_name}")
        return 0
    try:
        print(f"finished as {wait_until_finished(task_id, args.poll_seconds)!r}")
    except KeyboardInterrupt:
        sys.exit(
            "stopped waiting; the task keeps running. "
            f"Later: collect --task-id {task_id} --run-name {spec.run_name}"
        )
    _collect(args, task_id, spec.run_name)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "launch":
            return _launch(args)
        _collect(args, args.task_id, args.run_name)
        return 0
    except DownloadVerificationError as err:
        print(f"VERIFICATION FAILED, task NOT deleted: {err}", file=sys.stderr)
        return 2

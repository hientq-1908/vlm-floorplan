"""Launch the zero-shot baseline on ClearML, then download, verify, and delete the task.

    uv run --group ml python scripts/clearml_baseline.py launch --split SPLIT.jsonl --dry-run
    uv run --group ml python scripts/clearml_baseline.py launch --split SPLIT.jsonl --wait
    uv run --group ml python scripts/clearml_baseline.py collect --task-id TASK_ID --run-name NAME

Credentials are never read or written here. ClearML finds them in `~/clearml.conf` or
in the `CLEARML_*` environment variables, both outside the repository.

`--dry-run` prints everything that would happen and touches neither ClearML nor the
network. Without it, `launch` creates a task from the current git commit, attaches the
packed split, and enqueues it. With `--wait` it then waits for the job to finish and
collects the result; without it, run `collect` later.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from vlm_floorplan.evaluation.baseline_spec import (
    SPLIT_ARTIFACT,
    BaselineSpec,
    default_runs_dir,
    pack_sets,
    pack_split,
)
from vlm_floorplan.evaluation.collect import (
    FINISHED_STATUSES,
    CollectReport,
    ConsoleLog,
    DownloadVerificationError,
    RemoteArtifact,
    collect_and_delete,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
JOB_SCRIPT = "scripts/baseline_job.py"
REQUIREMENTS = "deploy/vllm/requirements-baseline.txt"
CONSOLE_LOG_EVENTS = 10_000


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _without_credentials(url: str) -> str:
    """Drop `user:token@` from a remote URL before it is printed."""
    parts = urlsplit(url)
    if parts.username or parts.password:
        host = parts.hostname or ""
        netloc = f"{host}:{parts.port}" if parts.port else host
        return urlunsplit(parts._replace(netloc=netloc))
    return url


def _remote_url() -> str:
    try:
        return _without_credentials(_git("remote", "get-url", "origin"))
    except subprocess.CalledProcessError:
        return "(no origin remote)"


def git_state() -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    return {
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": commit,
        "dirty": bool(_git("status", "--porcelain")),
        "pushed": bool(_git("branch", "-r", "--contains", commit)),
        "remote": _remote_url(),
    }


def describe_plan(spec: BaselineSpec, state: dict[str, Any], runs_dir: Path, plans: int) -> str:
    """The dry-run report: what launching would do, in order."""
    local = runs_dir / spec.run_name
    text = spec.task == "text2graph"
    packed = (
        f"pack {plans} set files (prompts, references, reference tables) into one archive"
        if text
        else f"pack {plans} plans and their images from the split into one archive"
    )
    evaluated = (
        f"for each variant {spec.variants} and set {spec.prompt_sets}"
        if text
        else f"once per resolution {spec.resolutions} px"
    )
    steps = [
        packed,
        f"create task {spec.run_name!r} in project {spec.project!r} from "
        f"{state['branch']} @ {state['commit'][:10]} ({state['remote']})",
        f"install the packages in {REQUIREMENTS} (vLLM pinned to 0.33.0) on the agent",
        f"attach the archive as artifact {SPLIT_ARTIFACT!r} and set the job parameters",
        f"enqueue on queue {spec.queue!r}",
        f"on the agent: serve the model with vLLM, evaluate {evaluated}, upload one result archive",
        "wait for the job to finish",
        f"download every artifact, the console log, scalars and task details to {local}",
        "verify: artifact sizes, zip integrity, SHA-256 manifest, one metrics.json per run",
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


class ClearMlTask:
    """`RemoteTask` backed by a real ClearML task (imported lazily)."""

    def __init__(self, task_id: str):
        from clearml import Task

        self.task_id = task_id
        self.task = Task.get_task(task_id=task_id)

    def status(self) -> str:
        return str(self.task.get_status())

    def details(self) -> dict[str, Any]:
        return self.task.export_task()

    def scalars(self) -> dict[str, Any]:
        return self.task.get_all_reported_scalars()

    def console_log(self) -> ConsoleLog:
        lines = list(self.task.get_reported_console_output(number_of_reports=CONSOLE_LOG_EVENTS))
        text = "".join(line if line.endswith("\n") else line + "\n" for line in lines)
        return ConsoleLog(text=text, truncated=len(lines) >= CONSOLE_LOG_EVENTS)

    def download_artifacts(self, dest: Path) -> dict[str, RemoteArtifact]:
        import shutil

        dest.mkdir(parents=True, exist_ok=True)
        downloaded: dict[str, RemoteArtifact] = {}
        for name, artifact in self.task.artifacts.items():
            cached = artifact.get_local_copy(extract_archive=False, raise_on_error=True)
            target = dest / f"{name}{Path(cached).suffix}"
            shutil.copy2(cached, target)
            downloaded[name] = RemoteArtifact(path=target, expected_size=artifact.size)
        return downloaded

    def delete(self) -> None:
        if not self.task.delete(
            delete_artifacts_and_models=True,
            skip_models_used_by_other_tasks=True,
            raise_on_error=True,
        ):
            raise RuntimeError(f"ClearML did not delete task {self.task_id}")


def create_and_enqueue(spec: BaselineSpec, split: Path, state: dict[str, Any]) -> str:
    """`split` is a split JSONL (image task) or a set directory (text2graph task)."""
    from clearml import Task

    with tempfile.TemporaryDirectory(prefix="baseline-split-") as tmp:
        archive = Path(tmp) / "split.tar.gz"
        if spec.task == "text2graph":
            pack_sets(split, archive)
        else:
            pack_split(split, archive, spec.limit)
        task = Task.create(
            project_name=spec.project,
            task_name=spec.run_name,
            task_type="testing",
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
        task.upload_artifact(SPLIT_ARTIFACT, artifact_object=str(archive), wait_on_upload=True)
    Task.enqueue(task, queue_name=spec.queue)
    return task.id


def wait_until_finished(task_id: str, poll_s: float) -> str:
    task = ClearMlTask(task_id)
    while (status := task.status()) not in FINISHED_STATUSES:
        print(f"  task {task_id}: {status}", flush=True)
        time.sleep(poll_s)
    return status


def _collect(args: argparse.Namespace, task_id: str, run_name: str) -> CollectReport:
    dest = args.runs_dir / run_name
    report = collect_and_delete(ClearMlTask(task_id), dest)
    print(f"downloaded and verified: {report.dest}")
    for check in report.checks:
        print(f"  ok: {check}")
    print("ClearML task deleted")
    return report


def _add_collect_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runs-dir", type=Path, default=default_runs_dir())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    launch = sub.add_parser("launch", help="create and enqueue the baseline task")
    launch.add_argument("--split", type=Path, help="image task: eval split JSONL of PlanRecords")
    launch.add_argument("--task", choices=["image", "text2graph"], default="image")
    launch.add_argument("--sets-dir", type=Path, help="text2graph task: dir from `build-sets`")
    launch.add_argument("--variants", nargs="+", default=["as-is", "one-example"])
    launch.add_argument(
        "--prompt-sets", nargs="+", default=["template_test", "paraphrase_test", "behavior_test"]
    )
    launch.add_argument("--run-name", default=datetime.now(UTC).strftime("baseline-%Y%m%d-%H%M"))
    launch.add_argument("--queue", default="3090")
    launch.add_argument("--project", default="vlm-floorplan")
    launch.add_argument("--model", default="Qwen/Qwen3.5-4B")
    launch.add_argument("--resolutions", type=int, nargs="+", default=[768, 1280])
    launch.add_argument("--max-tokens", type=int, help="default 12288 (image) or 8192 (text2graph)")
    launch.add_argument("--limit", type=int)
    launch.add_argument("--concurrency", type=int, help="default 8 (image) or 24 (text2graph)")
    launch.add_argument("--docker-image")
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("--wait", action="store_true", help="wait, then collect and delete")
    launch.add_argument("--poll-seconds", type=float, default=60.0)
    launch.add_argument("--allow-dirty", action="store_true")
    _add_collect_flags(launch)

    collect = sub.add_parser("collect", help="download, verify, then delete a finished task")
    collect.add_argument("--task-id", required=True)
    collect.add_argument("--run-name", required=True, help="local directory name")
    _add_collect_flags(collect)
    return parser


def _launch(args: argparse.Namespace) -> int:
    text = args.task == "text2graph"
    source = args.sets_dir if text else args.split
    if source is None:
        sys.exit("--sets-dir is required for --task text2graph" if text else "--split is required")
    spec = BaselineSpec(
        task=args.task,
        variants=args.variants,
        prompt_sets=args.prompt_sets,
        run_name=args.run_name,
        project=args.project,
        queue=args.queue,
        model=args.model,
        resolutions=args.resolutions,
        max_tokens=args.max_tokens or (8192 if text else 12288),
        limit=args.limit,
        concurrency=args.concurrency or (24 if text else 8),
        docker_image=args.docker_image,
    )
    state = git_state()
    if args.dry_run:
        with tempfile.TemporaryDirectory(prefix="baseline-dry-") as tmp:
            archive = Path(tmp) / "split.tar.gz"
            plans = pack_sets(source, archive) if text else pack_split(source, archive, spec.limit)
        print(describe_plan(spec, state, args.runs_dir, plans))
        return 0
    if state["dirty"] and not args.allow_dirty:
        sys.exit("refusing to launch: uncommitted changes (the agent only sees commits)")
    if not state["pushed"]:
        sys.exit("refusing to launch: this commit is not on any remote branch; push it first")
    task_id = create_and_enqueue(spec, source, state)
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

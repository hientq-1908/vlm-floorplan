"""The job that runs on the GPU machine: serve the model, evaluate it, bundle the results.

Flow (see `scripts/baseline_job.py` for the ClearML glue around it):

1. start `deploy/vllm/serve.sh` and wait until the server answers,
2. for each image resolution, run the evaluation harness against it (the M2 question:
   does 1280 px beat 768 px, because small labels need pixels?),
3. stop the server,
4. write `job_summary.json`, a SHA-256 manifest, and zip everything into one archive.

The run directory this produces is complete on its own: inputs, raw outputs, per-plan
results, metrics, the job log and the server log. Whatever happens in step 2, step 4
still runs, so a failed job leaves its evidence behind.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from vlm_floorplan.evaluation.baseline_spec import BaselineSpec
from vlm_floorplan.evaluation.client import ChatClient, HttpChatClient
from vlm_floorplan.evaluation.config import EvalConfig
from vlm_floorplan.evaluation.harness import collect_outputs
from vlm_floorplan.evaluation.run_dir import write_manifest
from vlm_floorplan.evaluation.scoring import score_run
from vlm_floorplan.evaluation.t2g_harness import T2gConfig, context_tokens
from vlm_floorplan.evaluation.t2g_harness import collect_outputs as collect_t2g
from vlm_floorplan.evaluation.t2g_scoring import score_run as score_t2g

log = logging.getLogger("baseline_job")

SERVER_START_TIMEOUT_S = 1800.0
"""Model download plus load can take a while on a cold machine."""
SERVER_POLL_S = 5.0
REQUEST_TIMEOUT_S = 1800.0
"""Thinking mode can generate for many minutes on one plan."""
PROMPT_CONTEXT_TOKENS = 8192
"""Context beyond the completion budget: a 1280 px plan image (about 1.2K tokens), the
prompt, and the chat template, with room to spare."""
SERVE_SCRIPT = Path(__file__).resolve().parents[3] / "deploy" / "vllm" / "serve.sh"


def run_name_for(side: int) -> str:
    return f"long-side-{side}"


def spec_from_parameters(params: dict[str, str]) -> BaselineSpec:
    """Rebuild the spec from the flat string parameters the launcher attached."""
    return BaselineSpec(
        run_name=params["run_name"],
        model=params["model"],
        served_name=params["served_name"],
        task=params.get("task", "image"),
        resolutions=[int(side) for side in params["resolutions"].split(",")],
        variants=params.get("variants", "as-is,one-example").split(","),
        prompt_sets=params.get("prompt_sets", "template_test,paraphrase_test,behavior_test").split(
            ","
        ),
        max_tokens=int(params["max_tokens"]),
        limit=int(params["limit"]) if params.get("limit") else None,
        concurrency=int(params["concurrency"]),
    )


def run_text_baseline(
    spec: BaselineSpec, sets_dir: Path, out_dir: Path, client: ChatClient, base_url: str
) -> dict[str, Any]:
    """One text-to-graph run per prompt variant and set, all against the same server.

    A run that fails is logged and skipped, so the others still get measured.
    """
    runs: list[str] = []
    scores: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for variant in spec.variants:
        for set_name in spec.prompt_sets:
            prompts = sets_dir / f"{set_name}.jsonl"
            if not prompts.is_file() or not prompts.read_text().strip():
                log.info("skipping %s: no prompts", set_name)
                continue
            name = f"{variant}-{set_name.replace('_', '-')}"
            config = T2gConfig(
                run_name=name,
                prompts=str(prompts),
                variant=variant,
                base_url=base_url,
                model=spec.served_name,
                max_tokens=spec.max_tokens,
                limit=spec.limit,
                concurrency=spec.concurrency,
                timeout_s=REQUEST_TIMEOUT_S,
            )
            try:
                log.info("evaluating %s", name)
                rows = collect_t2g(config, client, out_dir / name)
                scores[name] = score_t2g(out_dir / name, rows, sets_dir)
            except Exception as err:
                log.exception("run %s failed", name)
                errors[name] = f"{type(err).__name__}: {err}"
                continue
            runs.append(name)
    result = {"spec": spec.model_dump(), "runs": runs, "scores": scores, "errors": errors}
    (out_dir / "job_summary.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


def run_baseline(
    spec: BaselineSpec, split: Path, out_dir: Path, client: ChatClient, base_url: str
) -> dict[str, Any]:
    """One harness run per resolution, all against the same server.

    A resolution that fails is logged and skipped, so the other still gets measured.
    """
    runs: list[str] = []
    scores: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for side in spec.resolutions:
        name = run_name_for(side)
        config = EvalConfig(
            run_name=name,
            split=str(split),
            base_url=base_url,
            model=spec.served_name,
            max_long_side=side,
            max_tokens=spec.max_tokens,
            limit=spec.limit,
            concurrency=spec.concurrency,
            timeout_s=REQUEST_TIMEOUT_S,
        )
        run_dir = out_dir / name
        try:
            log.info("evaluating at %d px", side)
            records = collect_outputs(config, client, run_dir)
            summary = score_run(run_dir, records)
        except Exception as err:  # keep the other resolution alive, record why
            log.exception("run %s failed", name)
            errors[name] = f"{type(err).__name__}: {err}"
            continue
        runs.append(name)
        scores[name] = summary["overall"]
    result = {"spec": spec.model_dump(), "runs": runs, "scores": scores, "errors": errors}
    (out_dir / "job_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def wait_for_server(base_url: str, process: subprocess.Popen[bytes], timeout_s: float) -> None:
    """Poll `GET /models` until it answers; fail early if the server process died."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {process.returncode} before it was ready")
        try:
            with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=5) as reply:
                if reply.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(SERVER_POLL_S)
    raise TimeoutError(f"vLLM did not become ready within {timeout_s:.0f} s")


def serve_env(spec: BaselineSpec, port: int) -> dict[str, str]:
    """Settings for `serve.sh`.

    The served context is sized from the request budget: vLLM rejects (HTTP 400) any
    request whose prompt plus `max_tokens` exceeds `MAX_MODEL_LEN`, so raising
    `--max-tokens` without it would turn every plan into an API error. An image run sends one
    request per plan; a text-to-graph run is an agent loop whose later requests resend the
    earlier turns (`t2g_harness.context_tokens`).
    """
    if spec.task == "text2graph":
        context = context_tokens(spec.max_tokens)
    else:
        context = spec.max_tokens + PROMPT_CONTEXT_TOKENS
    return {
        "MODEL": spec.model,
        "SERVED_NAME": spec.served_name,
        "SERVE_PORT": str(port),
        "MAX_MODEL_LEN": str(context),
    }


@contextmanager
def vllm_server(spec: BaselineSpec, log_path: Path, port: int = 8000) -> Iterator[str]:
    """Run `serve.sh` for the duration of the block; yields the OpenAI base URL."""
    env = {**os.environ, **serve_env(spec, port)}
    with log_path.open("wb") as server_log:
        process = subprocess.Popen(
            ["bash", str(SERVE_SCRIPT)],
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # so we can stop the whole process group
        )
        base_url = f"http://127.0.0.1:{port}/v1"
        try:
            wait_for_server(base_url, process, SERVER_START_TIMEOUT_S)
            log.info("vLLM is ready")
            yield base_url
        finally:
            _stop(process)


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def bundle_results(out_dir: Path, archive: Path) -> Path:
    """Write the manifest, then zip the directory (the manifest is inside the zip)."""
    write_manifest(out_dir)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(out_dir))
    return archive


def run_job(spec: BaselineSpec, split: Path, out_dir: Path, archive: Path) -> dict[str, Any]:
    """The whole job without ClearML: serve, evaluate, bundle."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # The job's own log goes into the result too, so it survives even if ClearML's
    # console copy is truncated.
    file_handler = logging.FileHandler(out_dir / "job.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(file_handler)
    result: dict[str, Any] = {"runs": [], "scores": {}, "errors": {}}
    try:
        with vllm_server(spec, out_dir / "vllm.log") as base_url:
            client = HttpChatClient(base_url, timeout_s=REQUEST_TIMEOUT_S)
            runner = run_text_baseline if spec.task == "text2graph" else run_baseline
            result = runner(spec, split, out_dir, client, base_url)
    except Exception as err:  # the bundle below must still be written
        log.exception("job failed before or during evaluation")
        result["errors"]["job"] = f"{type(err).__name__}: {err}"
        (out_dir / "job_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()
    bundle_results(out_dir, archive)
    return result

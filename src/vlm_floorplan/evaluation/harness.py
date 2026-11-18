"""Run a model over a JSONL split and record everything.

Phase 1 (this module): for each plan, build the request, call the endpoint, and append
the raw reply to `outputs.jsonl`. Nothing is interpreted here, so a run can always be
re-scored later (see `scoring.py`) without calling the model again.

Run directory:

    <run>/config.json       settings (no secrets)
    <run>/prompt.txt        the exact prompt text
    <run>/inputs.jsonl      per plan: image file, sizes, the request sent (image elided)
    <run>/inputs/images/    the exact pixels sent, one PNG per plan
    <run>/outputs.jsonl     per plan: content, reasoning, finish_reason, tokens, latency
    <run>/per_plan.jsonl    per plan: parse result and metrics   (scoring.py)
    <run>/metrics.json      aggregate scores and breakdowns      (scoring.py)

Failed API calls are recorded as outputs with an `api_error`, and images that cannot be
loaded as outputs with an `input_error`; both count as invalid.
A run can be resumed: plans with a reply in `outputs.jsonl` are skipped, while failed
plans are tried again. Scoring reads the last row per plan, so a retry replaces a failure.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

from vlm_floorplan.evaluation.client import ApiError, ChatClient
from vlm_floorplan.evaluation.config import EvalConfig
from vlm_floorplan.evaluation.images import prepare_image, to_data_url, to_png_bytes
from vlm_floorplan.evaluation.prompt import build_prompt
from vlm_floorplan.evaluation.run_dir import append_jsonl, read_jsonl, safe_name
from vlm_floorplan.record import PlanRecord

IMAGE_PLACEHOLDER = "<image: see inputs/images/{name}.png>"
FAILURE_KEYS = ("api_error", "input_error")
"""Output-row keys that mark a plan with no model reply; such plans are retried on resume."""


def load_records(split: Path, limit: int | None = None) -> list[PlanRecord]:
    records: list[PlanRecord] = []
    with split.open() as handle:
        for line in handle:
            if line.strip():
                records.append(PlanRecord.model_validate_json(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def build_request(config: EvalConfig, prompt: str, image_url: str) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
    }
    if config.seed is not None:
        request["seed"] = config.seed
    # top_k and chat_template_kwargs are vLLM extensions; vLLM accepts them at top level.
    if config.top_k is not None:
        request["top_k"] = config.top_k
    if config.enable_thinking is not None:
        request["chat_template_kwargs"] = {"enable_thinking": config.enable_thinking}
    return request


# Settings that decide what the model is asked and how, versus where and how fast.
# Only the first group must match to resume a run: a restarted server may have a new URL.
NON_IDENTITY_FIELDS = {"base_url", "api_key_env", "timeout_s", "concurrency", "limit"}


def prepare_run_dir(config: EvalConfig, run_dir: Path) -> None:
    """Create the run directory, or check that an existing one has the same settings."""
    config_path = run_dir / "config.json"
    new = json.loads(config.model_dump_json())
    if config_path.exists():
        old = json.loads(config_path.read_text())
        if any(old.get(k) != v for k, v in new.items() if k not in NON_IDENTITY_FIELDS):
            raise ValueError(f"{run_dir} holds a run with different settings; use a new --run-name")
    else:
        (run_dir / "inputs" / "images").mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(new, indent=2) + "\n")
    (run_dir / "prompt.txt").write_text(build_prompt() + "\n")


def _prepare_input(
    record: PlanRecord, split_dir: Path, config: EvalConfig, run_dir: Path, prompt: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resize the image, save what the model will see, return (input row, request)."""
    name = safe_name(record.id)
    resized = prepare_image(split_dir / record.image, config.max_long_side)
    with Image.open(split_dir / record.image) as original:  # reads only the header
        original_size = original.size
    png = to_png_bytes(resized)
    (run_dir / "inputs" / "images" / f"{name}.png").write_bytes(png)
    request = build_request(config, prompt, to_data_url(png))
    shown = json.loads(json.dumps(request))
    shown["messages"][0]["content"][0]["image_url"]["url"] = IMAGE_PLACEHOLDER.format(name=name)
    row = {
        "plan_id": record.id,
        "image": record.image,
        "original_size": list(original_size),
        "sent_size": list(resized.size),
        "request": shown,
    }
    return row, request


def _call(client: ChatClient, plan_id: str, request: dict[str, Any]) -> dict[str, Any]:
    try:
        reply = client.complete(request)
    except ApiError as err:
        return {"plan_id": plan_id, "api_error": str(err)}
    return {
        "plan_id": plan_id,
        "content": reply.content,
        "reasoning": reply.reasoning,
        "finish_reason": reply.finish_reason,
        "prompt_tokens": reply.prompt_tokens,
        "completion_tokens": reply.completion_tokens,
        "latency_s": round(reply.latency_s, 3),
    }


def collect_outputs(config: EvalConfig, client: ChatClient, run_dir: Path) -> list[PlanRecord]:
    """Phase 1: call the endpoint for every plan not yet in `outputs.jsonl`."""
    split = Path(config.split)
    records = load_records(split, config.limit)
    prepare_run_dir(config, run_dir)
    prompt = build_prompt()
    done = {
        row["plan_id"]
        for row in read_jsonl(run_dir / "outputs.jsonl")
        if not any(key in row for key in FAILURE_KEYS)
    }
    todo = [record for record in records if record.id not in done]
    has_input_row = {row["plan_id"] for row in read_jsonl(run_dir / "inputs.jsonl")}

    def work(record: PlanRecord) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        try:
            input_row, request = _prepare_input(record, split.parent, config, run_dir, prompt)
        except (OSError, ValueError, Image.DecompressionBombError) as err:
            return None, {"plan_id": record.id, "input_error": f"{type(err).__name__}: {err}"}
        return input_row, _call(client, record.id, request)

    # Threads, not processes: the work is waiting on the server, and vLLM batches
    # concurrent requests, so concurrency is what keeps the GPU busy.
    with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        for input_row, output_row in pool.map(work, todo):
            if input_row is not None and input_row["plan_id"] not in has_input_row:
                append_jsonl(run_dir / "inputs.jsonl", input_row)
            append_jsonl(run_dir / "outputs.jsonl", output_row)
    return records

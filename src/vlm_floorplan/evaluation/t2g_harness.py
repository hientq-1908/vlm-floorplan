"""Run a model over a text-to-graph prompt set and record everything it says.

Phase 1 sends, for each prompt, the training system prompt, the twelve tools and the user turn
to an OpenAI-compatible endpoint and appends the raw reply, structured tool calls included, to
`outputs.jsonl`. Nothing is interpreted here, so a run can be re-scored without calling the
model again (`t2g_scoring.py`). Base and fine-tuned models run the same way: serve the base
model with `deploy/vllm/serve.sh`, and a LoRA adapter with `LORA_MODULES=name=/path` (vLLM
`--enable-lora`), then pass the served name as `--model`.

Run directory: `config.json`, `system_prompt.txt`, `inputs.jsonl` (the request sent),
`outputs.jsonl`, then `per_prompt.jsonl` and `metrics.json` from scoring and `MANIFEST.json`.
A run resumes: prompts with a reply are skipped, failed ones retried.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vlm_floorplan.agent.tools import agent_tool_schemas
from vlm_floorplan.evaluation.client import ApiError, ChatClient
from vlm_floorplan.evaluation.harness import FAILURE_KEYS, NON_IDENTITY_FIELDS
from vlm_floorplan.evaluation.run_dir import append_jsonl, read_jsonl
from vlm_floorplan.evaluation.t2g_prompt import (
    PROMPT_VERSION,
    VARIANTS,
    build_messages,
    system_prompt,
)
from vlm_floorplan.text2graph.rows import PromptRow, read_rows


class T2gConfig(BaseModel):
    """Settings of one run. Saved as `config.json`; the API key itself is never stored."""

    model_config = ConfigDict(extra="forbid")

    run_name: str
    prompts: str
    """JSONL of `PromptRow`; `references.jsonl` and `reference_tables.json` sit next to it."""
    variant: str = "as-is"
    base_url: str = "http://localhost:8000/v1"
    model: str
    """The served model name: the base model's, or a LoRA module's."""
    api_key_env: str | None = None
    max_tokens: int = Field(default=8192, gt=0)
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int | None = 20
    seed: int | None = 0
    enable_thinking: bool | None = True
    concurrency: int = Field(default=8, gt=0)
    limit: int | None = None
    timeout_s: float = 1800.0
    prompt_version: str = PROMPT_VERSION


def build_request(config: T2gConfig, row: PromptRow) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": config.model,
        "messages": build_messages(row, config.variant),
        "tools": agent_tool_schemas(),
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
    }
    if config.seed is not None:
        request["seed"] = config.seed
    if config.top_k is not None:
        request["top_k"] = config.top_k
    if config.enable_thinking is not None:
        request["chat_template_kwargs"] = {"enable_thinking": config.enable_thinking}
    return request


def prepare_run_dir(config: T2gConfig, run_dir: Path) -> None:
    """Create the run directory, or check that an existing one has the same settings."""
    if config.variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    new = json.loads(config.model_dump_json())
    if config_path.exists():
        old = json.loads(config_path.read_text())
        if any(old.get(k) != v for k, v in new.items() if k not in NON_IDENTITY_FIELDS):
            raise ValueError(f"{run_dir} holds a run with different settings; use a new run name")
    config_path.write_text(json.dumps(new, indent=2) + "\n")
    (run_dir / "system_prompt.txt").write_text(system_prompt(config.variant) + "\n")


def _call(client: ChatClient, prompt_id: str, request: dict[str, Any]) -> dict[str, Any]:
    try:
        reply = client.complete(request)
    except ApiError as err:
        return {"prompt_id": prompt_id, "api_error": str(err)}
    return {
        "prompt_id": prompt_id,
        "content": reply.content,
        "reasoning": reply.reasoning,
        "tool_calls": reply.tool_calls,
        "finish_reason": reply.finish_reason,
        "prompt_tokens": reply.prompt_tokens,
        "completion_tokens": reply.completion_tokens,
        "latency_s": round(reply.latency_s, 3),
    }


def collect_outputs(config: T2gConfig, client: ChatClient, run_dir: Path) -> list[PromptRow]:
    """Phase 1: call the endpoint for every prompt not yet answered."""
    rows = read_rows(Path(config.prompts))
    if config.limit is not None:
        rows = rows[: config.limit]
    prepare_run_dir(config, run_dir)
    done = {
        row["prompt_id"]
        for row in read_jsonl(run_dir / "outputs.jsonl")
        if not any(key in row for key in FAILURE_KEYS)
    }
    todo = [row for row in rows if row.id not in done]
    sent = {row["prompt_id"] for row in read_jsonl(run_dir / "inputs.jsonl")}

    def work(row: PromptRow) -> tuple[dict[str, Any], dict[str, Any]]:
        request = build_request(config, row)
        return request, _call(client, row.id, request)

    with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        for row, (request, output) in zip(todo, pool.map(work, todo), strict=True):
            if row.id not in sent:
                # The system prompt and the tools are the same for every row (system_prompt.txt).
                shown = {k: v for k, v in request.items() if k not in ("tools", "messages")}
                shown["user"] = request["messages"][-1]["content"]
                append_jsonl(run_dir / "inputs.jsonl", {"prompt_id": row.id, "request": shown})
            append_jsonl(run_dir / "outputs.jsonl", output)
    return rows


__all__ = ["NON_IDENTITY_FIELDS", "T2gConfig", "build_request", "collect_outputs"]

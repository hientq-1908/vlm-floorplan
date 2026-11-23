"""Run a model over a text-to-graph prompt set and record everything it says.

Phase 1 runs, for each prompt, the agent loop the model is trained on: the training system
prompt, the twelve tools and the user turn go to an OpenAI-compatible endpoint; every structured
tool call in the reply is run in the real environment (`agent.tools.call_tool`) on the running
plan, the assistant turn and one tool message per call are appended, and the conversation is
sent again, until the model replies without a tool call or has made `max_tool_calls` calls.
The whole transcript goes to `outputs.jsonl`; nothing is scored here, so a run can be re-scored
without calling the model again (`t2g_scoring.py`). Base and fine-tuned models run the same way:
serve the base model with `deploy/vllm/serve.sh`, and a LoRA adapter with
`LORA_MODULES=name=/path` (vLLM `--enable-lora`), then pass the served name as `--model`.

Run directory: `config.json`, `system_prompt.txt`, `inputs.jsonl` (the first request sent),
`outputs.jsonl`, then `per_prompt.jsonl` and `metrics.json` from scoring and `MANIFEST.json`.
A run resumes: prompts with a complete transcript are skipped, failed ones retried from the start.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vlm_floorplan.agent.tools import agent_tool_schemas, call_tool
from vlm_floorplan.evaluation.client import ApiError, ChatClient, ChatResponse
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
    max_tool_calls: int = Field(default=8, gt=0)
    """Tool calls allowed per prompt; a transcript that reaches it is flagged `hit_call_cap`."""
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


def call_arguments(call: dict[str, Any]) -> dict[str, Any] | None:
    """The arguments of a structured tool call as a dict, or None when not a JSON object."""
    raw = call.get("arguments")
    if isinstance(raw, dict):
        return raw
    try:
        data = json.loads(raw or "")
    except (TypeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _turn(reply: ChatResponse) -> dict[str, Any]:
    return {
        "content": reply.content,
        "reasoning": reply.reasoning,
        "tool_calls": reply.tool_calls,
        "finish_reason": reply.finish_reason,
        "prompt_tokens": reply.prompt_tokens,
        "completion_tokens": reply.completion_tokens,
        "latency_s": round(reply.latency_s, 3),
        "tool_responses": [],
    }


def _assistant_message(turn: dict[str, Any], ids: list[str]) -> dict[str, Any]:
    calls = [
        {
            "id": call_id,
            "type": "function",
            "function": {"name": call.get("name"), "arguments": call.get("arguments")},
        }
        for call_id, call in zip(ids, turn["tool_calls"], strict=True)
    ]
    return {
        "role": "assistant",
        "content": turn["content"],
        "reasoning_content": turn["reasoning"],
        "tool_calls": calls,
    }


def run_prompt(
    config: T2gConfig, client: ChatClient, row: PromptRow, request: dict[str, Any]
) -> dict[str, Any]:
    """The agent loop for one prompt; the transcript, or an `api_error` record."""
    messages = list(request["messages"])
    graph = row.start_graph()
    turns: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    while len(calls) < config.max_tool_calls:
        try:
            reply = client.complete(request | {"messages": list(messages)})
        except ApiError as err:
            return {"prompt_id": row.id, "api_error": str(err)}
        turn = _turn(reply)
        turns.append(turn)
        if not reply.tool_calls:
            break
        ids = [
            call.get("id") or f"call_{len(calls) + i}" for i, call in enumerate(reply.tool_calls)
        ]
        messages.append(_assistant_message(turn, ids))
        for call_id, call in zip(ids, reply.tool_calls, strict=True):
            if len(calls) == config.max_tool_calls:
                break
            name = call.get("name") or ""
            result = call_tool(graph, name, call_arguments(call) or {})
            graph = result.graph
            calls.append(
                {"name": name, "arguments": call.get("arguments"), "applied": result.applied}
            )
            turn["tool_responses"].append(result.text)
            messages.append(
                {"role": "tool", "tool_call_id": call_id, "name": name, "content": result.text}
            )
    first, last = turns[0], turns[-1]
    return {
        "prompt_id": row.id,
        **{k: v for k, v in first.items() if k != "tool_responses"},
        "turns": turns,
        "calls": calls,
        "hit_call_cap": bool(last["tool_calls"]),
        "final_content": last["content"],
        "final_graph": graph.model_dump(mode="json"),
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
        return request, run_prompt(config, client, row, request)

    with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        for row, (request, output) in zip(todo, pool.map(work, todo), strict=True):
            if row.id not in sent:
                # The system prompt and the tools are the same for every row (system_prompt.txt).
                shown = {k: v for k, v in request.items() if k not in ("tools", "messages")}
                shown["user"] = request["messages"][-1]["content"]
                append_jsonl(run_dir / "inputs.jsonl", {"prompt_id": row.id, "request": shown})
            append_jsonl(run_dir / "outputs.jsonl", output)
    return rows


__all__ = [
    "NON_IDENTITY_FIELDS",
    "T2gConfig",
    "build_request",
    "call_arguments",
    "collect_outputs",
    "run_prompt",
]

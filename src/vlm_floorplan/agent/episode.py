"""Episodes as TRL conversational records.

`build_episode` replays an environment's scripted solution through `step()` and writes
the conversation the model should learn:

    system   a short instruction; the template adds the tool list in front of it
    user     the damaged plan (`summarize_graph`) and the request
    assistant  reasoning_content + one tool call
    tool     the real tool response from the environment
    ...      one assistant/tool pair per call
    assistant  reasoning_content + a short final reply

The record also carries `tools` (the eleven tool schemas), which TRL passes to the chat template.
Arguments stay a dict: the Qwen3.5 template renders a dict as `<parameter=...>` blocks.
`reward` and `task` are metadata for filtering and are not part of the conversation.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from vlm_floorplan.agent.env import FloorplanEditEnv
from vlm_floorplan.agent.tasks import TASK_KINDS, Task, candidate_tasks
from vlm_floorplan.agent.tools import agent_tool_schemas
from vlm_floorplan.record import PlanRecord

SYSTEM_PROMPT = (
    "You edit a floorplan room graph with tools. Rooms are points on a 0-1000 grid"
    " (x to the right, y down); links say how two rooms connect (door, opening, wall)."
    " Call one tool at a time and read its response before the next call."
    " A door or opening needs a wall link first. When the request is done, reply in one sentence."
)


def assistant_turn(reasoning: str, content: str = "", call: Any = None) -> dict[str, Any]:
    """An assistant message. A tool-calling turn has empty content; the reasoning is separate."""
    message: dict[str, Any] = {
        "role": "assistant",
        "reasoning_content": reasoning,
        "content": content,
    }
    if call is not None:
        function = {"name": call.name, "arguments": dict(call.arguments)}
        message["tool_calls"] = [{"type": "function", "function": function}]
    return message


def build_episode(record: PlanRecord, task: Task) -> dict[str, Any]:
    """Run the scripted solution and return one TRL record (messages, tools, metadata)."""
    env = FloorplanEditEnv()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": env.reset(record, task)},
    ]
    for call in env.solution():
        messages.append(assistant_turn(call.reasoning, call=call))
        text = env.step(call.name, call.arguments)
        messages.append({"role": "tool", "name": call.name, "content": text})
    reply = task.reply(record.graph)
    messages.append(assistant_turn(reply.reasoning, reply.text))
    return {
        "id": f"{record.id}#{task.kind}",
        "task": task.kind,
        "reward": env.reward(),
        "messages": messages,
        "tools": agent_tool_schemas(),
    }


def sample_episodes(
    records: Sequence[PlanRecord], per_kind: int = 1, seed: int = 0
) -> list[dict[str, Any]]:
    """`per_kind` episodes of every task kind, each on a random plan with a random task.

    A plan that has no task of a kind (no doors to close) is skipped for that kind.
    """
    rng = random.Random(seed)
    episodes = []
    for kind in TASK_KINDS:
        pool = [(record, task) for record in records for task in candidate_tasks(record, kind)]
        for record, task in rng.sample(pool, min(per_kind, len(pool))):
            episodes.append(build_episode(record, task))
    return episodes

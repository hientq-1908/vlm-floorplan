"""The tool set the editing agent is served with: eleven edit tools plus `new_plan`.

`agent_tool_schemas()` is the one list handed to the chat template (and to vLLM as `tools`).
`call_tool(graph, name, arguments)` is the one function that runs a call.
Edit tools go through `apply_tool_call`, the same path the app uses, so what the model
practices is what the app does.

`new_plan` is the only tool that is not an edit op.
It draws a whole plan from scratch: the argument is the flat plan as compact JSON
(see `compact.py`), with a size per room.
It replaces the current plan, keeps the image size and the exterior wall (the model reads
the wall and never writes it), and is refused with an `invalid_plan` error when the text
does not parse or the plan is empty or broken.
The edit tools stay available for every later change.

Every response, applied or refused, starts with the status (`OK:` or `REJECTED (code):`)
and ends with the full current plan, see `edits/summary.py`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from vlm_floorplan.compact import from_compact
from vlm_floorplan.edits import (
    EditApplied,
    EditError,
    EditErrorCode,
    EditRejected,
    apply_tool_call,
    format_edit_outcome,
    tool_schemas,
    validate_graph,
)
from vlm_floorplan.edits.summary import summarize_graph
from vlm_floorplan.schema import FloorplanGraph

NEW_PLAN = "new_plan"

_NEW_PLAN_EXAMPLE = (
    '{"rooms":[{"id":"r0","type":"entry","xy":[474,735],"size":52,"entrance":true},'
    '{"id":"r1","type":"living","label":"OH","xy":[583,295],"size":160}],'
    '"links":[["r0","r1","door"]]}'
)

NEW_PLAN_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": NEW_PLAN,
        "description": (
            "Draw a whole new plan from scratch. It replaces the current plan, so use it only"
            " when there is nothing to keep; to change the current plan use the other tools."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "string",
                    "description": (
                        "The plan as compact JSON on one line: rooms (id r0, r1, ..., type, xy on"
                        " the 0-1000 grid, size = the radius of a circle with the room's area on"
                        " the same grid, optional label, optional entrance: true for a room an"
                        " outside door opens into) and links [a, b, kind] with kind door, opening"
                        " or wall. For example: " + _NEW_PLAN_EXAMPLE
                    ),
                }
            },
            "required": ["plan"],
        },
    },
}


def agent_tool_schemas() -> list[dict[str, Any]]:
    """The twelve tools in the OpenAI function-calling format: eleven edits, then `new_plan`."""
    return [*tool_schemas(), NEW_PLAN_SCHEMA]


@dataclass(frozen=True)
class ToolResult:
    """What one tool call did."""

    graph: FloorplanGraph
    """The plan after the call; the same plan as before when the call was refused."""
    text: str
    """The tool response the model reads. A refusal starts with `REJECTED (<code>):`."""
    applied: bool


def _rejected(graph: FloorplanGraph, code: EditErrorCode, message: str) -> ToolResult:
    text = format_edit_outcome(EditRejected(EditError(code, message)), graph)
    return ToolResult(graph, text, applied=False)


def _reason(err: ValueError) -> str:
    """Why the plan text did not parse, in plain words (no pydantic boilerplate)."""
    if isinstance(err, json.JSONDecodeError):
        return f"it is not JSON ({err.msg} at character {err.pos})"
    if isinstance(err, ValidationError):
        found = []
        for item in err.errors(include_url=False):
            where = ".".join(str(part) for part in item["loc"])
            message = item["msg"].removeprefix("Value error, ")
            found.append(f"{where}: {message}" if where else message)
        return "; ".join(found)
    return " ".join(str(err).split())


def _new_plan(graph: FloorplanGraph, arguments: Mapping[str, Any]) -> ToolResult:
    unexpected = sorted(set(arguments) - {"plan"})
    if unexpected or not isinstance(arguments.get("plan"), str):
        message = "new_plan takes exactly one argument, 'plan', a string of compact JSON."
        return _rejected(graph, EditErrorCode.INVALID_ARGUMENTS, message)
    try:
        plan = from_compact(arguments["plan"], graph.image, graph.exterior_wall)
    except ValueError as err:  # bad JSON, a wrong shape, or pydantic's ValidationError
        return _rejected(
            graph, EditErrorCode.INVALID_PLAN, f"The plan is not valid: {_reason(err)}."
        )
    report = validate_graph(plan)
    if report.errors:
        reasons = "; ".join(issue.message for issue in report.errors)
        return _rejected(graph, EditErrorCode.INVALID_PLAN, f"The plan is not valid: {reasons}.")
    if not plan.rooms():
        return _rejected(graph, EditErrorCode.INVALID_PLAN, "The plan has no rooms.")
    lines = ["OK: replaced the plan with a new one."]
    lines.extend(f"Warning: {issue.message}." for issue in report.warnings)
    lines.append(summarize_graph(plan))
    return ToolResult(plan, "\n".join(lines), applied=True)


def call_tool(graph: FloorplanGraph, name: str, arguments: Mapping[str, Any]) -> ToolResult:
    """Run one tool call. Never raises for a bad call: the refusal is the response text."""
    if name == NEW_PLAN:
        return _new_plan(graph, arguments)
    outcome = apply_tool_call(graph, name, arguments)
    text = format_edit_outcome(outcome, graph)
    if isinstance(outcome, EditApplied):
        return ToolResult(outcome.graph, text, applied=True)
    return ToolResult(graph, text, applied=False)

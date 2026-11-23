"""Score one reply of a text-to-graph run: what the model did, and whether it was right.

Strict reading: the reply holds a structured `new_plan` tool call and the environment accepts
it (`call_tool`), exactly what the app would do. Lenient reading: no accepted call, but a plan
can still be found (XML tool call in the reasoning, bare JSON in the answer) by the same parser
the perception baseline uses. Both are kept, so "how often does the model follow the format"
stays visible next to "how good is the plan it meant".
"""

from __future__ import annotations

import json
import re
from typing import Any

from vlm_floorplan.agent.env import plan_score
from vlm_floorplan.agent.tools import NEW_PLAN, call_tool
from vlm_floorplan.edits.validator import validate_graph
from vlm_floorplan.evaluation.parser import parse_model_output
from vlm_floorplan.evaluation.plausibility import ReferenceTables, plan_plausibility
from vlm_floorplan.record import PlanRecord
from vlm_floorplan.requirements import check_requirements
from vlm_floorplan.schema import FloorplanGraph
from vlm_floorplan.text2graph.rows import PromptRow

_REASONS = (
    ("not JSON", "invalid_json"),
    ('no "size"', "missing_size"),
    ("does not exist", "dangling_link"),
    ("no rooms", "no_rooms"),
    ("linked more than once", "duplicate_link"),
    ("itself", "self_link"),
    ("takes exactly one argument", "bad_arguments"),
)


def reject_reason(response: str) -> str:
    """A short category for a refused `new_plan`, read from the tool response."""
    first = response.splitlines()[0] if response else ""
    for needle, reason in _REASONS:
        if needle in first:
            return reason
    return "other_invalid_plan"


def _arguments(call: dict[str, Any]) -> dict[str, Any] | None:
    raw = call.get("arguments")
    if isinstance(raw, dict):
        return raw
    try:
        data = json.loads(raw or "")
    except (TypeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _lenient_graph(output: dict[str, Any], start: FloorplanGraph) -> FloorplanGraph | None:
    result = parse_model_output(output.get("content"), start.image, output.get("reasoning"))
    if result.graph is None:
        return None
    graph = result.graph.model_copy(update={"exterior_wall": start.exterior_wall})
    if not graph.rooms() or validate_graph(graph).errors or len(graph.floors) != 1:
        return None
    return graph


def draw_outcome(output: dict[str, Any], start: FloorplanGraph) -> dict[str, Any]:
    """What the reply drew: the strict graph, else the lenient one, and why strict failed."""
    calls = output.get("tool_calls") or []
    plan_calls = [call for call in calls if call.get("name") == NEW_PLAN]
    out: dict[str, Any] = {
        "tools": [call.get("name") for call in calls],
        "called_new_plan": bool(plan_calls),
        "strict_graph": None,
        "lenient_graph": None,
        "reject_reason": None,
    }
    if plan_calls:
        arguments = _arguments(plan_calls[0])
        if arguments is None:
            out["reject_reason"] = "bad_arguments"
        else:
            result = call_tool(start, NEW_PLAN, arguments)
            if result.applied:
                out["strict_graph"] = result.graph
            else:
                out["reject_reason"] = reject_reason(result.text)
    if out["strict_graph"] is None:
        out["lenient_graph"] = _lenient_graph(output, start)
    return out


def _met(graph: FloorplanGraph | None, row: PromptRow) -> list[bool] | None:
    if graph is None:
        return None
    return [r.met for r in check_requirements(graph, row.parsed_requirements())]


def score_draw(
    row: PromptRow,
    output: dict[str, Any],
    reference: PlanRecord | None,
    tables: ReferenceTables,
) -> dict[str, Any]:
    """Score a draw or restart reply against the prompt's requirements."""
    outcome = draw_outcome(output, row.start_graph())
    strict, lenient = outcome["strict_graph"], outcome["lenient_graph"]
    used = strict or lenient
    requirements = row.parsed_requirements()
    return {
        "called_new_plan": outcome["called_new_plan"],
        "tools": outcome["tools"],
        "strict_valid": strict is not None,
        "lenient_found": used is not None,
        "reject_reason": outcome["reject_reason"],
        "requirement_kinds": [req.kind for req in requirements],
        "met_strict": _met(strict, row),
        "met_any": _met(used, row),
        "plan_score": plan_score(used, reference) if reference and row.level == 4 else None,
        "plausibility": (plan_plausibility(used, tables).model_dump() if used else None),
        "plausibility_strict": strict is not None,
        "rooms": len(used.rooms()) if used else None,
    }


_QUESTION_END = re.compile(r"\?\s*$")


def _mentions(content: str, groups: list[list[str]], need: str) -> bool:
    lowered = content.lower()
    hits = [any(word.lower() in lowered for word in group) for group in groups]
    return all(hits) if need == "all" else any(hits)


def _edit_effect(start: FloorplanGraph, calls: list[dict[str, Any]], expect: dict) -> bool:
    graph = start
    for call in calls:
        arguments = _arguments(call)
        if arguments is not None:
            graph = call_tool(graph, call.get("name") or "", arguments).graph
    rooms = {room.id: room for room in graph.rooms()}
    links = graph.floors[0].links
    if expect["edit"] == "link":
        return any(
            {link.a, link.b} == {expect["a"], expect["b"]} and link.kind.value == expect["link"]
            for link in links
        )
    before = {room.id for room in start.rooms()}
    new = [room for room in rooms.values() if room.id not in before and room.type == expect["type"]]
    return any({link.a, link.b} == {room.id, expect["near"]} for room in new for link in links)


def score_behavior(row: PromptRow, output: dict[str, Any]) -> dict[str, Any]:
    """Score an edit, contradiction or unsupported-ask reply."""
    calls = output.get("tool_calls") or []
    names = [call.get("name") for call in calls]
    content = (output.get("content") or "").strip()
    result: dict[str, Any] = {
        "tools": names,
        "called_new_plan": NEW_PLAN in names,
        "no_tool_call": not calls,
    }
    if row.behavior == "edit":
        edited = bool(calls) and NEW_PLAN not in names
        effect = edited and _edit_effect(row.start_graph(), calls, row.expect)
        result |= {"edit_tools_only": edited, "edit_effect_ok": effect, "correct": effect}
    else:
        asked = not calls and bool(_QUESTION_END.search(content) or "?" in content)
        named = asked and _mentions(content, row.expect["groups"], row.expect["need"])
        result |= {"asked_question": asked, "names_issue": named, "correct": named}
    return result


def score_row(
    row: PromptRow,
    output: dict[str, Any],
    reference: PlanRecord | None,
    tables: ReferenceTables,
) -> dict[str, Any]:
    """One per-prompt record: identity fields, the reply's facts and the scores."""
    base = {
        "prompt_id": row.id,
        "source": row.source,
        "level": row.level,
        "kind": row.kind,
        "behavior": row.behavior,
        "finish_reason": output.get("finish_reason"),
        "api_error": output.get("api_error") or output.get("input_error"),
    }
    if base["api_error"]:
        scored: dict[str, Any] = {"called_new_plan": False, "tools": [], "failed_call": True}
        if row.behavior in ("draw", "restart"):
            scored |= score_draw(row, {}, reference, tables)
        else:
            scored |= {"correct": False, "no_tool_call": False}
        return base | scored | {"api_error": base["api_error"]}
    if row.behavior in ("draw", "restart"):
        return base | score_draw(row, output, reference, tables)
    return base | score_behavior(row, output)

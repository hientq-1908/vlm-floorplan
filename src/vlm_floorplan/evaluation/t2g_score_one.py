"""Score one transcript of a text-to-graph run: what the model did, and whether it was right.

A transcript is the agent loop of `t2g_harness.py`: every tool call was run in the environment
and its response fed back. Strict reading of a draw: an accepted structured `new_plan` call in
the transcript, and the plan the environment holds at the end, exactly what the app would show.
`tool_call_rate` and `validity_rate` read the first `new_plan` call. Lenient reading: no
accepted call, but a plan can still be found in the first reply (XML tool call in the
reasoning, bare JSON in the answer) by the same parser the perception baseline uses. Both are
kept, so "how often does the model follow the format" stays visible next to "how good is the
plan it meant". A transcript that reached the call cap is not a clean finish: it has no strict
plan and no behavior of it is correct.
"""

from __future__ import annotations

import re
from typing import Any

from vlm_floorplan.agent.env import plan_score
from vlm_floorplan.agent.tools import NEW_PLAN
from vlm_floorplan.edits.validator import validate_graph
from vlm_floorplan.evaluation.parser import parse_model_output
from vlm_floorplan.evaluation.plausibility import ReferenceTables, plan_plausibility
from vlm_floorplan.record import PlanRecord
from vlm_floorplan.requirements import check_requirements
from vlm_floorplan.schema import FloorplanGraph, LinkKind
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


def _lenient_graph(output: dict[str, Any], start: FloorplanGraph) -> FloorplanGraph | None:
    result = parse_model_output(output.get("content"), start.image, output.get("reasoning"))
    if result.graph is None:
        return None
    graph = result.graph.model_copy(update={"exterior_wall": start.exterior_wall})
    if not graph.rooms() or validate_graph(graph).errors or len(graph.floors) != 1:
        return None
    return graph


def _executed(output: dict[str, Any]) -> list[tuple[dict[str, Any], str]]:
    """Every call the environment ran, in order, with the response the model read."""
    responses = [text for turn in output.get("turns") or [] for text in turn["tool_responses"]]
    return list(zip(output.get("calls") or [], responses, strict=True))


def _final_graph(output: dict[str, Any]) -> FloorplanGraph:
    return FloorplanGraph.model_validate(output["final_graph"])


def draw_outcome(output: dict[str, Any], start: FloorplanGraph) -> dict[str, Any]:
    """What the transcript drew: the strict graph, else the lenient one, and why strict failed."""
    executed = _executed(output)
    plan_calls = [(call, text) for call, text in executed if call["name"] == NEW_PLAN]
    capped = bool(output.get("hit_call_cap"))
    out: dict[str, Any] = {
        "tools": [call["name"] for call, _ in executed],
        "called_new_plan": bool(plan_calls),
        "first_plan_valid": bool(plan_calls) and plan_calls[0][0]["applied"],
        "hit_call_cap": capped,
        "strict_graph": None,
        "lenient_graph": None,
        "reject_reason": None,
    }
    if plan_calls and not out["first_plan_valid"]:
        out["reject_reason"] = reject_reason(plan_calls[0][1])
    if any(call["applied"] for call, _ in plan_calls) and not capped:
        out["strict_graph"] = _final_graph(output)
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
    """Score a draw or restart transcript against the prompt's requirements.

    A restart is correct when an accepted `new_plan` left a plan that meets every requirement.
    """
    outcome = draw_outcome(output, row.start_graph())
    strict, lenient = outcome["strict_graph"], outcome["lenient_graph"]
    used = strict or lenient
    requirements = row.parsed_requirements()
    met_strict = _met(strict, row)
    scored = {
        "called_new_plan": outcome["called_new_plan"],
        "tools": outcome["tools"],
        "hit_call_cap": outcome["hit_call_cap"],
        "strict_valid": outcome["first_plan_valid"],
        "strict_plan": strict is not None,
        "lenient_found": used is not None,
        "reject_reason": outcome["reject_reason"],
        "requirement_kinds": [req.kind for req in requirements],
        "met_strict": met_strict,
        "met_any": _met(used, row),
        "plan_score": plan_score(used, reference) if reference and row.level == 4 else None,
        "plausibility": (plan_plausibility(used, tables).model_dump() if used else None),
        "plausibility_strict": strict is not None,
        "rooms": len(used.rooms()) if used else None,
    }
    if row.behavior == "restart":
        scored |= {
            "correct": bool(met_strict) and all(met_strict),
            "no_tool_call": not output.get("tool_calls"),
        }
    return scored


def _has_word(text: str, word: str) -> bool:
    """Whole-word match, so "one" is not found in "someone" or "floor" in "floor plan" by luck."""
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(word)}(?![A-Za-z0-9])", text, re.I) is not None


def _mentions(content: str, groups: list[list[str]], need: str) -> bool:
    hits = [any(_has_word(content, word) for word in group) for group in groups]
    return all(hits) if need == "all" else any(hits)


def _pair(a: str, b: str) -> frozenset[str]:
    return frozenset((a, b))


def _kept(start: FloorplanGraph, final: FloorplanGraph, changed: frozenset[str] | None) -> bool:
    """Every start room keeps its id, type, point, size and entrance flag, and every start link
    keeps its kind, except the `changed` link."""
    rooms = {room.id: room for room in final.rooms()}
    for room in start.rooms():
        got = rooms.get(room.id)
        if got is None or (got.type, got.xy, got.size, got.entrance) != (
            room.type,
            room.xy,
            room.size,
            room.entrance,
        ):
            return False
    kinds = {_pair(link.a, link.b): link.kind for link in final.floors[0].links}
    return all(
        kinds.get(_pair(link.a, link.b)) == link.kind
        for link in start.floors[0].links
        if _pair(link.a, link.b) != changed
    )


def _edit_effect(start: FloorplanGraph, final: FloorplanGraph, expect: dict) -> bool:
    """The final plan is the start plan plus the asked change: the link of the kind asked for,
    or a new room of the type with a door or opening to the anchor."""
    links = final.floors[0].links
    if expect["edit"] == "link":
        changed = _pair(expect["a"], expect["b"])
        made = any(
            _pair(link.a, link.b) == changed and link.kind.value == expect["link"] for link in links
        )
        return made and _kept(start, final, changed)
    before = {room.id for room in start.rooms()}
    new = [room for room in final.rooms() if room.id not in before and room.type == expect["type"]]
    passage = any(
        _pair(link.a, link.b) == _pair(room.id, expect["near"]) and link.kind != LinkKind.WALL
        for room in new
        for link in links
    )
    return passage and _kept(start, final, None)


def score_behavior(row: PromptRow, output: dict[str, Any]) -> dict[str, Any]:
    """Score an edit transcript, or the first reply to a contradiction or unsupported ask.

    A correct model asks its question in the first reply and stops there, so only that reply
    is read for the question kinds.
    """
    names = [call["name"] for call in output.get("calls") or []]
    capped = bool(output.get("hit_call_cap"))
    content = (output.get("content") or "").strip()
    result: dict[str, Any] = {
        "tools": names,
        "called_new_plan": NEW_PLAN in names,
        "no_tool_call": not output.get("tool_calls"),
        "hit_call_cap": capped,
    }
    if row.behavior == "edit":
        edited = bool(names) and NEW_PLAN not in names
        effect = edited and not capped
        effect = effect and _edit_effect(row.start_graph(), _final_graph(output), row.expect)
        result |= {"edit_tools_only": edited, "edit_effect_ok": effect, "correct": effect}
    else:
        asked = result["no_tool_call"] and "?" in content
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
        "finish_reason": (output.get("turns") or [output])[-1].get("finish_reason"),
        "api_error": output.get("api_error") or output.get("input_error"),
    }
    if base["api_error"]:
        failed = {"failed_call": True, "correct": False, "no_tool_call": False}
        if row.behavior in ("draw", "restart"):
            scored = score_draw(row, {}, reference, tables)
        else:
            scored = {"called_new_plan": False, "tools": [], "hit_call_cap": False}
        return base | scored | failed
    if row.behavior in ("draw", "restart"):
        return base | score_draw(row, output, reference, tables)
    return base | score_behavior(row, output)

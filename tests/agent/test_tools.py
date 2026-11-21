from __future__ import annotations

import json

import pytest

from vlm_floorplan.agent import NEW_PLAN, agent_tool_schemas, call_tool
from vlm_floorplan.agent.sample_plan import sample_record
from vlm_floorplan.compact import to_compact
from vlm_floorplan.edits import OP_CLASSES
from vlm_floorplan.schema import Floor, FloorplanGraph

SCALAR_TYPES = {"string", "integer", "boolean"}


def empty_plan() -> FloorplanGraph:
    graph = sample_record().graph
    return FloorplanGraph(floors=[Floor(floor=1, rooms=[], links=[])], image=graph.image)


def test_the_tool_list_has_the_ten_edit_tools_then_new_plan() -> None:
    names = [schema["function"]["name"] for schema in agent_tool_schemas()]
    assert names == [*OP_CLASSES, NEW_PLAN]
    assert len(names) == 11


def test_every_tool_argument_is_a_flat_scalar() -> None:
    for schema in agent_tool_schemas():
        assert json.loads(json.dumps(schema)) == schema
        function = schema["function"]
        assert function["description"]
        for name, prop in function["parameters"]["properties"].items():
            assert prop["type"] in SCALAR_TYPES, f"{function['name']}.{name}"


def test_new_plan_takes_the_whole_plan_as_one_required_string() -> None:
    schema = next(s for s in agent_tool_schemas() if s["function"]["name"] == NEW_PLAN)
    parameters = schema["function"]["parameters"]
    assert parameters["required"] == ["plan"]
    assert set(parameters["properties"]) == {"plan"}


def test_new_plan_replaces_the_plan_and_keeps_the_image_size() -> None:
    target = sample_record().graph
    result = call_tool(empty_plan(), NEW_PLAN, {"plan": to_compact(target)})
    assert result.applied
    assert result.graph == target
    assert result.text.startswith("OK: replaced the plan")
    assert "The plan now has 6 rooms and 9 links" in result.text


@pytest.mark.parametrize(
    ("plan", "fragment"),
    [
        ("this is not json", "not valid"),
        ('{"floors":[{"floor":1,"rooms":[],"links":[]}]}', "no rooms"),
        (
            '{"floors":[{"floor":1,"rooms":[{"id":"r0","type":"kitchen","xy":[1,1]}],'
            '"links":[["r0","r9","door"]]}]}',
            "r9",
        ),
        (
            '{"floors":[{"floor":1,"rooms":[{"id":"r0","type":"kitchen","xy":[5000,1]}],'
            '"links":[]}]}',
            "not valid",
        ),
    ],
)
def test_new_plan_rejects_an_invalid_plan_and_leaves_the_plan_alone(
    plan: str, fragment: str
) -> None:
    before = sample_record().graph
    result = call_tool(before, NEW_PLAN, {"plan": plan})
    assert not result.applied
    assert result.graph == before
    assert result.text.startswith("REJECTED (invalid_plan):")
    assert fragment in result.text
    assert result.text.endswith("The plan was not changed.")


@pytest.mark.parametrize("arguments", [{}, {"plan": 5}, {"plan": "{}", "extra": "x"}])
def test_new_plan_rejects_bad_arguments(arguments: dict) -> None:
    result = call_tool(empty_plan(), NEW_PLAN, arguments)
    assert not result.applied
    assert result.text.startswith("REJECTED (invalid_arguments):")


def test_edit_tools_go_through_the_shared_edit_path() -> None:
    graph = sample_record().graph
    ok = call_tool(graph, "set_link_kind", {"room_a": "r0", "room_b": "r1", "kind": "door"})
    assert ok.applied
    assert ok.text.startswith("OK: changed the link between r0 and r1 from wall to door.")
    bad = call_tool(graph, "add_link", {"room_a": "r0", "room_b": "r2", "kind": "door"})
    assert not bad.applied
    assert bad.graph == graph
    assert bad.text.startswith("REJECTED (not_adjacent):")

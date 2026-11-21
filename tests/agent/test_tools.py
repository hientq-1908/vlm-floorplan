from __future__ import annotations

import json

import pytest

from vlm_floorplan.agent import NEW_PLAN, agent_tool_schemas, call_tool
from vlm_floorplan.agent.sample_plan import sample_record
from vlm_floorplan.compact import to_compact
from vlm_floorplan.edits import OP_CLASSES, summarize_graph
from vlm_floorplan.schema import Floor, FloorplanGraph

SCALAR_TYPES = {"string", "integer", "boolean"}


def empty_plan() -> FloorplanGraph:
    graph = sample_record().graph
    return FloorplanGraph(
        floors=[Floor(floor=1, rooms=[], links=[])],
        image=graph.image,
        exterior_wall=graph.exterior_wall,
    )


def test_the_tool_list_has_the_edit_tools_then_new_plan() -> None:
    names = [schema["function"]["name"] for schema in agent_tool_schemas()]
    assert names == [*OP_CLASSES, NEW_PLAN]
    assert len(names) == 12
    assert "set_room_size" in names


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


def test_new_plan_describes_a_flat_plan_with_a_size_per_room() -> None:
    schema = next(s for s in agent_tool_schemas() if s["function"]["name"] == NEW_PLAN)
    description = schema["function"]["parameters"]["properties"]["plan"]["description"]
    assert '{"rooms":[' in description
    assert "floor" not in description
    assert '"size":' in description


def test_new_plan_replaces_the_plan_and_keeps_the_image_size_and_the_wall() -> None:
    target = sample_record().graph
    result = call_tool(empty_plan(), NEW_PLAN, {"plan": to_compact(target)})
    assert result.applied
    assert result.graph == target
    assert result.graph.exterior_wall == target.exterior_wall
    assert result.text == "OK: replaced the plan with a new one.\n" + summarize_graph(target)


def test_new_plan_ignores_a_wall_the_model_tries_to_write() -> None:
    target = sample_record().graph
    plan = json.loads(to_compact(target)) | {"exterior_wall": [[1, 1], [2, 2], [3, 3]]}
    result = call_tool(empty_plan(), NEW_PLAN, {"plan": json.dumps(plan)})
    assert result.applied
    assert result.graph.exterior_wall == target.exterior_wall


@pytest.mark.parametrize(
    ("plan", "fragment"),
    [
        ("this is not json", "not valid"),
        ('{"rooms":[],"links":[]}', "no rooms"),
        (
            '{"rooms":[{"id":"r0","type":"kitchen","xy":[1,1],"size":9}],"links":[["r0","r9","door"]]}',
            "r9",
        ),
        ('{"rooms":[{"id":"r0","type":"kitchen","xy":[5000,1],"size":9}],"links":[]}', "not valid"),
        ('{"floors":[{"floor":1,"rooms":[],"links":[]}]}', 'no "floors"'),
        ('{"rooms":[{"id":"r0","type":"kitchen","xy":[1,1],"size":0}],"links":[]}', "size"),
        ('{"rooms":[{"id":"r3","type":"kitchen","xy":[1,1]}],"links":[]}', "room 'r3' has no"),
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
    assert "The plan was not changed.\n" + summarize_graph(before) in result.text
    assert result.text.endswith(summarize_graph(before))


@pytest.mark.parametrize("arguments", [{}, {"plan": 5}, {"plan": "{}", "extra": "x"}])
def test_new_plan_rejects_bad_arguments(arguments: dict) -> None:
    result = call_tool(empty_plan(), NEW_PLAN, arguments)
    assert not result.applied
    assert result.text.startswith("REJECTED (invalid_arguments):")


def test_edit_tools_go_through_the_shared_edit_path() -> None:
    graph = sample_record().graph
    ok = call_tool(graph, "set_link_kind", {"room_a": "r0", "room_b": "r1", "kind": "door"})
    assert ok.applied
    assert ok.text.startswith("OK: changed the link between r0 and r1 from wall to door.\nplan: 6")
    assert ok.text.endswith(
        "r0-r1 door, r1-r2 wall, r0-r3 door, r1-r3 door, r2-r3 opening,"
        " r2-r4 door, r3-r4 door, r3-r5 wall, r4-r5 door"
    )
    bad = call_tool(graph, "add_link", {"room_a": "r0", "room_b": "r2", "kind": "door"})
    assert not bad.applied
    assert bad.graph == graph
    assert bad.text.startswith("REJECTED (not_adjacent):")
    assert bad.text.endswith(summarize_graph(graph))

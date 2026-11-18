from __future__ import annotations

import json

import pytest

from vlm_floorplan.edits import (
    OP_CLASSES,
    AddLink,
    EditApplied,
    EditErrorCode,
    EditRejected,
    MoveRoom,
    apply_tool_call,
    parse_tool_call,
    tool_schemas,
)

from .helpers import sample_graph

SCALAR_TYPES = {"string", "integer", "boolean"}


def test_there_is_one_tool_per_op_named_after_it() -> None:
    names = [schema["function"]["name"] for schema in tool_schemas()]
    assert names == list(OP_CLASSES)
    assert sorted(names) == sorted(
        [
            "add_room", "move_room", "delete_room", "set_room_type", "set_room_label",
            "add_link", "remove_link", "set_link_kind", "set_entrance",
        ]
    )  # fmt: skip


def test_schemas_are_json_and_in_function_calling_format() -> None:
    for schema in tool_schemas():
        assert json.loads(json.dumps(schema)) == schema
        assert schema["type"] == "function"
        function = schema["function"]
        assert function["description"]
        assert function["parameters"]["type"] == "object"


def test_every_argument_is_a_flat_scalar_with_a_description() -> None:
    for schema in tool_schemas():
        name = schema["function"]["name"]
        for key, prop in schema["function"]["parameters"]["properties"].items():
            where = f"{name}.{key}"
            assert prop["type"] in SCALAR_TYPES, where
            assert prop["description"], where
            assert not {"$ref", "anyOf", "allOf", "oneOf", "items", "properties", "default"} & set(
                prop
            ), where


def test_the_discriminator_and_hidden_fields_are_not_offered_to_the_model() -> None:
    properties = {
        s["function"]["name"]: s["function"]["parameters"]["properties"] for s in tool_schemas()
    }
    assert all("op" not in props for props in properties.values())
    assert "room_id" not in properties["add_room"]
    assert "room_id" in properties["move_room"]


def test_required_lists_only_arguments_without_defaults() -> None:
    required = {
        s["function"]["name"]: set(s["function"]["parameters"]["required"]) for s in tool_schemas()
    }
    assert required["add_room"] == {"type", "x", "y"}
    assert required["add_link"] == {"room_a", "room_b", "kind"}
    assert required["set_entrance"] == {"room_id", "entrance"}
    assert required["set_room_label"] == {"room_id", "label"}


def test_enums_list_the_schema_values() -> None:
    add_link = next(s for s in tool_schemas() if s["function"]["name"] == "add_link")
    assert add_link["function"]["parameters"]["properties"]["kind"]["enum"] == [
        "door",
        "opening",
        "wall",
    ]


def test_parse_tool_call_builds_the_op() -> None:
    op = parse_tool_call("add_link", {"room_a": "r1", "room_b": "r3", "kind": "door"})
    assert op == AddLink(room_a="r1", room_b="r3", kind="door")


def test_parse_tool_call_accepts_strings_for_numbers_like_the_xml_parser_may_send() -> None:
    op = parse_tool_call("move_room", {"room_id": "r1", "x": "250", "y": "300"})
    assert op == MoveRoom(room_id="r1", x=250, y=300)


def test_unknown_tool_lists_the_valid_ones() -> None:
    result = parse_tool_call("teleport", {})
    assert isinstance(result, EditRejected)
    assert result.error.code == EditErrorCode.UNKNOWN_TOOL
    assert "add_room" in result.error.message


@pytest.mark.parametrize(
    ("name", "arguments", "fragment"),
    [
        ("move_room", {"room_id": "r1", "x": 5}, "missing required argument 'y'"),
        ("move_room", {"room_id": "r1", "x": 5000, "y": 5}, "argument 'x'"),
        ("move_room", {"room_id": "kitchen", "x": 5, "y": 5}, "argument 'room_id'"),
        ("move_room", {"room_id": "r1", "x": 5, "y": 5, "z": 1}, "unknown argument 'z'"),
        ("add_link", {"room_a": "r1", "room_b": "r2", "kind": "window"}, "argument 'kind'"),
        ("set_entrance", {"room_id": "r1", "entrance": "maybe"}, "argument 'entrance'"),
        ("add_room", {"type": "kitchen", "x": 5, "y": 5, "x1": 1}, "all four of x1, y1, x2, y2"),
        (
            "add_room",
            {"type": "kitchen", "x": 50, "y": 5, "x1": 0, "y1": 0, "x2": 10, "y2": 10},
            "inside its box",
        ),
    ],
)
def test_bad_arguments_become_a_readable_error(name: str, arguments: dict, fragment: str) -> None:
    result = parse_tool_call(name, arguments)
    assert isinstance(result, EditRejected)
    assert result.op is None
    assert result.error.code == EditErrorCode.INVALID_ARGUMENTS
    assert fragment in result.error.message
    assert result.error.message.startswith(f"Bad arguments for {name}:")
    assert "Value error" not in result.error.message


def test_an_op_field_in_the_arguments_cannot_change_the_tool() -> None:
    result = parse_tool_call("delete_room", {"room_id": "r1", "op": "move_room"})
    assert result.room_id == "r1"
    assert result.op == "delete_room"


def test_apply_tool_call_applies_and_rejects() -> None:
    graph = sample_graph()
    assert isinstance(apply_tool_call(graph, "delete_room", {"room_id": "r4"}), EditApplied)
    bad = apply_tool_call(graph, "add_link", {"room_a": "r0", "room_b": "r4", "kind": "door"})
    assert isinstance(bad, EditRejected)
    assert bad.error.code == EditErrorCode.NOT_ADJACENT
    assert apply_tool_call(graph, "delete_room", {}).error.code == EditErrorCode.INVALID_ARGUMENTS

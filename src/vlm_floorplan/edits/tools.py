"""Edit ops as model tools: JSON schemas out, tool calls in.

`tool_schemas()` lists one tool per op in the OpenAI function-calling format that
Qwen3.5's chat template and vLLM's `qwen3_coder` tool parser both take.
`parse_tool_call` / `apply_tool_call` go the other way: a tool name plus an arguments
dict becomes an op, and bad arguments become a typed, readable error instead of an exception.

The schemas are generated from the op models, so the ops stay the single source of truth.
They are flattened on purpose: every property is a plain string, integer or boolean
(enums stay as `enum` lists), with no `$ref`, `anyOf`, objects or arrays.
Qwen3.5 renders arguments as XML text, where nested structures are fragile and
`"75"` and `75` cannot be told apart.
Optional arguments are simply left out of `required`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from vlm_floorplan.edits.apply import apply_op
from vlm_floorplan.edits.ops import EDIT_OP_ADAPTER, OP_CLASSES, EditOp
from vlm_floorplan.edits.outcome import EditError, EditErrorCode, EditOutcome, EditRejected
from vlm_floorplan.schema import FloorplanGraph

# Fields that exist on the op for replays and tests but that the model should not set.
_HIDDEN_FIELDS = {"op"}
_HIDDEN_PER_TOOL = {"add_room": {"room_id"}}


def _flatten_property(prop: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    """Turn one pydantic property schema into a flat scalar schema."""
    if "$ref" in prop:
        prop = {
            **defs[prop["$ref"].rsplit("/", 1)[-1]],
            **{k: v for k, v in prop.items() if k != "$ref"},
        }
    if "anyOf" in prop:  # Optional[X] is anyOf [X, null]; the tool just omits the argument
        branches = [b for b in prop["anyOf"] if b.get("type") != "null"]
        rest = {k: v for k, v in prop.items() if k != "anyOf"}
        return _flatten_property({**branches[0], **rest}, defs)
    return {k: v for k, v in prop.items() if k not in {"title", "default"}}


def _tool_schema(name: str, cls: type) -> dict[str, Any]:
    schema = cls.model_json_schema()  # type: ignore[attr-defined]
    defs = schema.get("$defs", {})
    hidden = _HIDDEN_FIELDS | _HIDDEN_PER_TOOL.get(name, set())
    properties = {
        key: _flatten_property(prop, defs)
        for key, prop in schema["properties"].items()
        if key not in hidden
    }
    required = [key for key in schema.get("required", []) if key not in hidden]
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": " ".join(schema["description"].split()),
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def tool_schemas() -> list[dict[str, Any]]:
    """The tool list to hand to the chat template, one entry per edit op."""
    return [_tool_schema(name, cls) for name, cls in OP_CLASSES.items()]


def format_validation_error(err: ValidationError, tool: str | None = None) -> str:
    """Pydantic's error list as plain sentences a model can act on."""
    parts = []
    for item in err.errors(include_url=False):
        # pydantic prefixes the location with the union tag (the tool name) and the
        # discriminator field is internal, so neither is named to the model.
        path = ".".join(str(p) for p in item["loc"] if p not in ("op", tool))
        kind = item["type"]
        if kind == "missing":
            parts.append(f"missing required argument '{path}'")
        elif kind == "extra_forbidden":
            parts.append(f"unknown argument '{path}'")
        else:
            message = item["msg"].removeprefix("Value error, ")
            parts.append(f"argument '{path}': {message}" if path else message)
    return "; ".join(parts)


def parse_tool_call(name: str, arguments: Mapping[str, Any]) -> EditOp | EditRejected:
    """Build an op from a model's tool call, or explain what is wrong with the call."""
    if name not in OP_CLASSES:
        valid = ", ".join(OP_CLASSES)
        error = EditError(
            EditErrorCode.UNKNOWN_TOOL, f"There is no tool '{name}'. Tools are: {valid}."
        )
        return EditRejected(error)
    try:
        return EDIT_OP_ADAPTER.validate_python({**arguments, "op": name})
    except ValidationError as err:
        message = f"Bad arguments for {name}: {format_validation_error(err, name)}."
        return EditRejected(EditError(EditErrorCode.INVALID_ARGUMENTS, message))


def apply_tool_call(graph: FloorplanGraph, name: str, arguments: Mapping[str, Any]) -> EditOutcome:
    """Parse a tool call and apply it: the one-line entry point for the agent loop."""
    op = parse_tool_call(name, arguments)
    if isinstance(op, EditRejected):
        return op
    return apply_op(graph, op)

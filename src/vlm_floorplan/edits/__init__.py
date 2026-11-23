"""Graph edit operations and the graph validator (no torch, no web dependencies).

See docs/edit-operations.md for the design.
"""

from vlm_floorplan.edits.apply import apply_op
from vlm_floorplan.edits.history import EditHistory, replay
from vlm_floorplan.edits.ops import (
    EDIT_OP_ADAPTER,
    OP_CLASSES,
    AddExteriorDoor,
    AddLink,
    AddRoom,
    DeleteRoom,
    EditOp,
    MoveRoom,
    RemoveExteriorDoor,
    RemoveLink,
    SetLinkKind,
    SetRoomLabel,
    SetRoomSize,
    SetRoomType,
    dump_ops,
    load_ops,
)
from vlm_floorplan.edits.outcome import (
    EditApplied,
    EditError,
    EditErrorCode,
    EditOutcome,
    EditRejected,
)
from vlm_floorplan.edits.summary import format_edit_outcome, summarize_graph
from vlm_floorplan.edits.tools import apply_tool_call, parse_tool_call, tool_schemas
from vlm_floorplan.edits.validator import (
    Issue,
    IssueCode,
    Severity,
    ValidationReport,
    validate_graph,
)

__all__ = [
    "EDIT_OP_ADAPTER",
    "OP_CLASSES",
    "AddExteriorDoor",
    "AddLink",
    "AddRoom",
    "DeleteRoom",
    "EditApplied",
    "EditError",
    "EditErrorCode",
    "EditHistory",
    "EditOp",
    "EditOutcome",
    "EditRejected",
    "Issue",
    "IssueCode",
    "MoveRoom",
    "RemoveExteriorDoor",
    "RemoveLink",
    "SetLinkKind",
    "SetRoomLabel",
    "SetRoomSize",
    "SetRoomType",
    "Severity",
    "ValidationReport",
    "apply_op",
    "apply_tool_call",
    "dump_ops",
    "format_edit_outcome",
    "load_ops",
    "parse_tool_call",
    "replay",
    "summarize_graph",
    "tool_schemas",
    "validate_graph",
]

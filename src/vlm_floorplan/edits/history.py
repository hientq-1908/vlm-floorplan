"""Edit history: replay, undo and redo over a list of ops.

A history is a starting graph plus the ops applied to it.
That is all that has to be saved: the current plan is `replay(base, ops)`.
Because ops are recorded as applied (ids and floors filled in, see `EditApplied.op`),
replaying the same ops on the same base always gives the same graph,
which is what synthetic training episodes and RL rollouts rely on.

For fast undo the history also keeps one graph snapshot per step.
Graphs are immutable, so snapshots share their unchanged rooms and cost little.
Undo and redo just move a cursor along the snapshots.
A new edit after an undo discards the undone ops, like any editor.
"""

from __future__ import annotations

from collections.abc import Iterable

from vlm_floorplan.edits.apply import apply_op
from vlm_floorplan.edits.ops import EditOp
from vlm_floorplan.edits.outcome import EditApplied, EditOutcome, EditRejected
from vlm_floorplan.schema import FloorplanGraph


def replay(base: FloorplanGraph, ops: Iterable[EditOp]) -> FloorplanGraph | EditRejected:
    """Apply ops in order. Returns the final graph, or the first refused op."""
    graph = base
    for op in ops:
        outcome = apply_op(graph, op)
        if isinstance(outcome, EditRejected):
            return outcome
        graph = outcome.graph
    return graph


class EditHistory:
    """The edit session of one plan: apply, undo, redo."""

    def __init__(self, base: FloorplanGraph) -> None:
        self._ops: list[EditOp] = []
        self._snapshots: list[FloorplanGraph] = [base]
        self._cursor = 0

    @property
    def base(self) -> FloorplanGraph:
        """The graph before any edit."""
        return self._snapshots[0]

    @property
    def graph(self) -> FloorplanGraph:
        """The current graph."""
        return self._snapshots[self._cursor]

    @property
    def ops(self) -> tuple[EditOp, ...]:
        """The ops currently in effect, oldest first. Undone ops are not included."""
        return tuple(self._ops[: self._cursor])

    @property
    def can_undo(self) -> bool:
        return self._cursor > 0

    @property
    def can_redo(self) -> bool:
        return self._cursor < len(self._ops)

    def apply(self, op: EditOp) -> EditOutcome:
        """Apply an op to the current graph. Only a successful edit is recorded."""
        outcome = apply_op(self.graph, op)
        if isinstance(outcome, EditApplied):
            del self._ops[self._cursor :]
            del self._snapshots[self._cursor + 1 :]
            self._ops.append(outcome.op)
            self._snapshots.append(outcome.graph)
            self._cursor += 1
        return outcome

    def undo(self) -> EditOp | None:
        """Step back one edit. Returns the undone op, or None if there is nothing to undo."""
        if not self.can_undo:
            return None
        self._cursor -= 1
        return self._ops[self._cursor]

    def redo(self) -> EditOp | None:
        """Step forward again after an undo. Returns the redone op, or None."""
        if not self.can_redo:
            return None
        op = self._ops[self._cursor]
        self._cursor += 1
        return op

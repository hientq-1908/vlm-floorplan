# Edit operations and the graph validator

This document explains `vlm_floorplan.edits`: how a plan changes, and why it is built this way.
The package has no torch or web dependencies, so the app backend, the synthetic data generator and the RL environment can all import it.

## One code path for every change

A plan changes in exactly one way: `apply_op(graph, op)`.
A chat request, a drag in the UI, a synthetic training episode and an RL rollout all build the same typed `EditOp` and call the same function.
This matters for learning: the model is trained on the behavior the app really runs, so there is no gap between "what the model practiced" and "what the app does".

The result is either `EditApplied` (a new graph) or `EditRejected` (a typed error).
Bad requests never raise.
A model asking for something impossible is an expected event, and the reason has to go back to it as text.

Graphs are immutable by convention.
`apply_op` builds a new `FloorplanGraph` through the schema constructors and never touches the one it was given.
That makes undo cheap (keep the old graph) and makes replays trivially safe.

## The ops

| Op | Arguments | Notes |
|----|-----------|-------|
| `add_room` | `type`, `x`, `y`, optional `label`, `floor`, `entrance`, box `x1 y1 x2 y2` | The id is assigned (next free number). The new room has no links. |
| `move_room` | `room_id`, `x`, `y` | The room's box, if it has one, moves with it so the point stays inside. |
| `delete_room` | `room_id` | Drops every link that touches the room. |
| `set_room_type` | `room_id`, `type` | |
| `set_room_label` | `room_id`, `label` | An empty label removes the name. |
| `add_link` | `room_a`, `room_b`, `kind` | See "Doors only go into shared walls". |
| `remove_link` | `room_a`, `room_b` | Removes adjacency entirely. To close a door, use `set_link_kind` with `wall`. |
| `set_link_kind` | `room_a`, `room_b`, `kind` | The pair must already be linked. |
| `set_entrance` | `room_id`, `entrance` | |

Walls and geometry are never edited.
`move_room` moves a graph node, not a wall.

## Flat scalar arguments

Every argument is a string, an integer or a boolean.
There are no nested objects and no lists.
Qwen3.5 renders tool arguments as XML text, so nested structures are fragile, and `"75"` and `75` look the same to it.
Flat arguments also keep the tool schemas simple and the model's tool calls short.

`tool_schemas()` generates the JSON schemas from the op models, so the models stay the single source of truth.
It flattens what pydantic emits: `$ref` is inlined, `Optional` becomes "not required", and the internal `op` discriminator is hidden.
A test asserts that no schema contains `$ref`, `anyOf`, objects or arrays.

`add_room` hides its `room_id` argument.
The model should not invent ids, so the next free one is assigned.
The id that was assigned is written back into the op (`EditApplied.op`), so replaying a history gives the same ids every time.

## Doors only go into shared walls

The model never sees walls after perception.
The graph remembers adjacency as a `wall` link, so that is the evidence the validator uses.

- `add_link` with kind `door` or `opening` is allowed only where the pair is already linked by a `wall` (the door goes into that wall).
- Between rooms with no link at all it is refused with `not_adjacent`, and the message says what to do if they do touch.
- `add_link` with kind `wall` declares adjacency between two unlinked rooms.
  This is needed to connect a newly added room.
- Any other `add_link` on an already linked pair is a `duplicate_link`, and the message points to `set_link_kind`.

An honest limit: a model can still say "wall, then door" to two far-apart rooms.
Checking real adjacency would need room polygons, which the graph does not keep, and drag edits move nodes around freely.
So the check is a guard against the common mistake (a door out of nowhere), not a geometric proof.

## The validator

`validate_graph(graph)` returns every finding as an `Issue` with a severity, a code, a sentence naming the rooms, and the room ids.

- Errors: duplicate floors or room ids, malformed ids, coordinates outside the 0-1000 grid, bad boxes, self links, links to rooms that do not exist or are on another floor, duplicate links.
- Warnings: no room is marked as the entrance, and a room with no links on a floor that has other rooms.

The schema's pydantic validators already refuse most of these, so a graph built normally is valid.
The validator still earns its place:

1. It runs on every edit result, so a handler bug becomes `invalid_result` instead of a silently broken plan.
2. It accepts graphs that skipped pydantic (`model_construct`, patched objects).
3. It reports all problems at once, in sentences, where pydantic raises on the first.

Warnings never refuse an edit.
A freshly added room is isolated by definition, and the model needs to be told so it can link it.
Edit results report only the warnings the edit introduced (`new_warnings`), not ones that were already true.

## Edits that change nothing are refused

Moving a room to where it already is, or setting a type it already has, returns `no_change`.
A history full of no-ops would teach a model that they are fine, and the message tells the model its request was already true.

## Text for the model

`format_edit_outcome` produces the tool response:

```text
OK: put a door into the wall between r1 and r3.
The plan now has 7 rooms and 5 links (3 door, 1 opening, 1 wall).

REJECTED (not_adjacent): Cannot add a door between r0 and r4: they share no wall or opening, ...
The plan was not changed.
```

`summarize_graph` writes the whole plan as lines (one per room, one `links:` line per floor).
It is shorter than JSON and easy to scan.
The compact JSON in `vlm_floorplan.compact` stays the format the model reads and writes when it perceives a plan.

## History: replay and undo

An `EditHistory` is a base graph plus the ops applied to it.
`replay(base, ops)` rebuilds the current plan, and `dump_ops` / `load_ops` save and load the op list as JSON.
The history also keeps one graph snapshot per step, so undo and redo only move a cursor.
A new edit after an undo discards the undone ops.
Only successful edits are recorded.

## Out of scope

The HTTP backend, the frontend, model calls and the RL environment are separate work.
They are expected to call `apply_tool_call(graph, name, arguments)` and return `format_edit_outcome(...)` as the tool response.

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
| `add_room` | `type`, `x`, `y`, `size`, optional `label` | The id is assigned (next free number). The new room has no links and no exterior door. |
| `move_room` | `room_id`, `x`, `y` | Changes only the point. |
| `set_room_size` | `room_id`, `size` | Changes only the size. Refused with `no_change` when the room already has that size. |
| `delete_room` | `room_id` | Drops every link that touches the room. |
| `set_room_type` | `room_id`, `type` | |
| `set_room_label` | `room_id`, `label` | An empty label removes the name. |
| `add_link` | `room_a`, `room_b`, `kind` | Only `wall` between unlinked rooms. See "Doors only go into shared walls". |
| `remove_link` | `room_a`, `room_b` | Removes adjacency entirely. To close a door, use `set_link_kind` with `wall`. |
| `set_link_kind` | `room_a`, `room_b`, `kind` | The pair must already be linked. The only way to change a link's kind. |
| `add_exterior_door` | `room_id` | Marks that a door to the outside opens into the room (sets its `entrance` flag). Refused with `no_change` when the room already has one. |
| `remove_exterior_door` | `room_id` | Clears the flag. Refused with `no_change` when the room has none. |

A room's exterior door is one flag on the room (`entrance`), set and cleared by two tools that mirror each other.
A new room never starts with one: `add_room` has no entrance argument, so the outside door is a separate, visible call.
Walls and geometry are never edited.
`move_room` moves a graph node, not a wall.

## Rooms are points

A room has no box, so `add_room` and `move_room` place a room by its point alone.
See "Rooms are points only" in `docs/architecture.md`.

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

- `add_link` only creates a link between two rooms that are not linked yet.
- `add_link` with kind `wall` declares adjacency between two unlinked rooms.
  This is needed to connect a newly added room.
- `add_link` with kind `door` or `opening` between unlinked rooms is refused with `not_adjacent`.
  The message says that the rooms are not linked, and that if they do touch, a `wall` link comes first and `set_link_kind` then turns it into a door or opening.
- Any `add_link` on an already linked pair is a `duplicate_link`, and the message points to `set_link_kind`.
- `set_link_kind` is the only way to change a link's kind.
  One tool per change means synthetic data and RL never produce two spellings of the same edit.

An honest limit: a model can still say "wall, then door" to two far-apart rooms.
Checking real adjacency would need room polygons, which the graph does not keep, and drag edits move nodes around freely.
So the check is a guard against the common mistake (a door out of nowhere), not a geometric proof.

## The validator

`validate_graph(graph)` returns every finding as an `Issue` with a severity, a code, a sentence naming the rooms, and the room ids.

- Errors: duplicate floors or room ids, malformed ids, coordinates outside the 0-1000 grid, self links, links to rooms that do not exist or are on another floor, duplicate links, a room size outside 1-500.
- Warnings: no room is marked as having an exterior door, and a room with no links when the plan has other rooms.

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
OK: changed the link between r1 and r3 from wall to door.
plan: 5 rooms, 4 links
exterior wall: (50, 400), (50, 900), (900, 900), (900, 400)
r0 entry at (100, 500) size 40 entrance
r1 living 'OH' at (400, 500) size 50
...
links: r0-r1 door, r1-r2 opening, r1-r3 door, r3-r4 wall

REJECTED (not_adjacent): Cannot add a door between r0 and r4: they are not linked. ...
The plan was not changed.
plan: 5 rooms, 4 links
...
```

Every response is a status line, then any warnings, then `summarize_graph` of the plan as it is now (the unchanged plan for a refusal).
`summarize_graph` writes the plan as lines: a `plan:` header (`plan: empty` for no rooms), the `exterior wall:` line, one line per room, one `links:` line.
It is single-floor and unindented: the text names no floor, and it raises `ValueError` for a graph with several floors.
It is shorter than JSON and easy to scan.
The model writes a whole plan only as the flat compact JSON of `vlm_floorplan.compact` (`new_plan`), and smaller changes through the edit tools.

## History: replay and undo

An `EditHistory` is a base graph plus the ops applied to it.
`replay(base, ops)` rebuilds the current plan, and `dump_ops` / `load_ops` save and load the op list as JSON.
The history also keeps one graph snapshot per step, so undo and redo only move a cursor.
A new edit after an undo discards the undone ops.
Only successful edits are recorded.

## Out of scope

The HTTP backend, the frontend, model calls and the RL environment are separate work.
They are expected to call `apply_tool_call(graph, name, arguments)` and return `format_edit_outcome(...)` as the tool response.

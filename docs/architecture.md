# Architecture

This page explains how vlm-floorplan is put together and why.
It is written for someone learning the project, so most sections say not only what was chosen but also what the alternative was.

## What the product does

A user captures a floorplan image.
A vision-language model (Qwen3.5-4B) reads it and produces a room graph.
Rooms are nodes, and connections between rooms are edges.
The user then edits the graph, either by chatting ("open the kitchen to the living room") or by dragging nodes and links in a UI.
The output of the whole system is the graph.

Two rules shape everything else:

- Walls and geometry are never edited.
  The model reads them once, during perception, and after that only the graph exists.
- A small model has to do two different jobs: perception (image to graph) and editing (graph plus request to tool calls).
  They are trained and evaluated separately.

```text
image --[perception model]--> FloorplanGraph --[edit tools + validator]--> FloorplanGraph
                                    ^                      ^
                              dataset converters      chat or drag
                            (training targets)
```

## The graph schema

The schema lives in `src/vlm_floorplan/schema.py`.
It is the contract between every part of the project: converters, the viewer, the model's output parser, and the edit tools (`docs/edit-operations.md`).

```text
FloorplanGraph { floors: [Floor], image: {w, h}, exterior_wall: [[x, y], ...] | None }
Floor          { floor: int, rooms: [Room], links: [Link] }
Room           { id: "r<n>", type, label?, xy: [x, y], size?, entrance? }
Link           { a: room id, b: room id, kind: door | opening | wall }
```

Each choice has a reason.

**Coordinates are integers from 0 to 1000 on the image the model saw.**
They do not depend on the image's pixel size, so one model can read images of any resolution.
The grid matches the convention Qwen3-VL documents for grounding ("relative coordinates ranging from 0 to 1000").
The Qwen3.5-4B model card does not say this, so milestone M2 checks that the base model uses the same grid.
Integers keep the output short: every digit is a token.

**`xy` is a point inside the room, not its centroid.**
The centroid of an L-shaped room can lie outside it.
The converters use shapely's `representative_point()`, which always lies inside the polygon.
This also gives a simple evaluation rule: a predicted room matches the ground-truth room whose polygon contains the predicted point, with no matching heuristic.
That is why `PlanRecord` keeps the room polygons even though the model only outputs points.

**A room has a size, the radius of a circle with its area; the plan has an exterior wall.**
`size` (1 to 500, on the same grid as `xy`) tells the model how big a room is without a box and without a metre scale, so it works for every source.
It is optional in the graph, because the converters do not write it yet, but the model-facing forms carry it: the plan text, `new_plan` and `add_room`, and `set_room_size` changes it.
`exterior_wall` is the building outline as a polygon of grid points, shown to the model on the second line of the plan text and never written by it.
Indoor rooms go inside it and outdoor rooms outside it, against it.
Edits and `new_plan` keep it.

**The model sees one floor.**
The plan text and the compact JSON have no floor header, field or wrapper, and the tools take no floor argument.
The graph underneath keeps its list of floors (converters and the validator still work on it), always with exactly one for what the model reads and writes.

**Rooms are points only, with no bbox.**
The graph is what the user edits, by chat or by dragging a room, and a drag moves one point.
A box beside the point would be a second position that every edit must keep in sync, and syncing it at the grid edge silently changes it.
It also cost about 12 tokens per room in every training target, and the grounding metric needs only the point (it matches a point to the ground-truth polygon, `docs/evaluation.md`).
The captain decided on 2026-11-19 to remove it entirely: the schema, the compact format, the converters, the viewer and the edit ops no longer have a box.
Room polygons stay in `PlanRecord` for evaluation, but they are not part of the graph.
A record that still carries a `bbox` field is rejected, not silently stripped: `Room` forbids unknown fields, so loading it raises a validation error.
Such a record was built before this change and its training targets would not match the current format, so it must be rebuilt from the source dataset.

**Links have three kinds: `door`, `opening` and `wall`.**
`door` and `opening` are both needed because open-plan areas are common: a kitchen, dining area and living room are often separate polygons with no wall between them.
A door-only schema would lose those connections.
`wall` means "adjacent, but a wall separates them".
The model never sees walls again after perception, so the edit validator needs `wall` links to reject a request like "add a door between two rooms that do not touch".
It costs about 11% of the graph's tokens, which is cheap compared with the image.
There is at most one link per pair of rooms.

**Exterior access is an `entrance` flag on a room, not a synthetic "outside" node.**
The flag means "an exterior door opens into this room": every outside door counts, front, back and patio doors alike, so a plan can have several entrance rooms.
Sources differ in what they mark: CubiCasa5K flags every exterior door, while ResPlan, MSD and Swiss Dwellings mark only each apartment's front door.
A ResPlan or Swiss Dwellings plan is one apartment, so it has one entrance room, while an MSD record holds several apartments and so has several.
Outdoor spaces such as balconies and terraces are ordinary rooms of type `outdoor`.
An outside node would be a node with no position and no polygon, which breaks the rule that every node is a point on the image.

**There is one graph per floor, and no link ever crosses floors.**
In CubiCasa5K, 1,021 of 5,000 plans show several floors side by side in one image, and `F1_scaled.png` and `F2_scaled.png` are the same file.
The data has no stairs links between floors, so the schema does not invent them.
Room ids (`r0`, `r1`, ...) are unique across the whole plan so that "r7" is unambiguous in a chat edit.

**Validators enforce the contract.**
A graph is rejected for duplicate ids, links to rooms that do not exist or sit on another floor, self links, duplicate links, and coordinates outside 0 to 1000.
Because the model output is parsed through the same validators, text that parses is a structurally valid graph.
This idea comes from paperbanana's `DiagramIR`, which validates references in the same way.

## Room types

The coarse type set is `living, kitchen, dining, bedroom, bathroom, entry, storage, utility, outdoor, garage, stairs, other, unknown`.
It is deliberately small, for two reasons.
A 4B model has to predict it from an image, and the retrieval tool in M6 has to compare plans from different datasets.
`room_types.py` maps CubiCasa5K (60+ raw classes), ResPlan (7 plus a front door), MSD (9) and Swiss Dwellings (77 `area` subtypes) into this set.

`unknown` is different from `other`.
`other` is a labeled room that fits no category (an office, an attic).
`unknown` means the source did not say, and such rooms are excluded from the type loss and from type metrics.
The raw source label is kept next to the coarse type, in `PlanRecord.rooms[...].source_type`, so mapping loses nothing.

## Compact JSON

The model reads and writes a compact text form (`compact.py`):

```json
{"rooms":[{"id":"r0","type":"outdoor","xy":[564,839],"size":90},{"id":"r2","type":"entry","xy":[474,735],"size":52}],"links":[["r0","r2","door"]]}
```

It has no whitespace, no floor (the model sees one floor; the graph underneath still holds a list), links are `[a, b, kind]` triples, optional fields appear only when set, and the image size is left out.
`size` is the radius of a circle with the room's area, on the same 0-1000 grid as `xy`.
The exterior wall is input only: the model reads it in the plan text and never writes it, so it is not in this form.
In the roadmap check, the median graph was 365 tokens and the largest was 877.
Graph text is cheap next to the image: a 1024 px image costs about 768 tokens.
`from_compact(to_compact(graph), image, graph.exterior_wall) == graph` holds for every valid single-floor graph, and the tests check it.
The zero-shot perception baseline keeps its own `floors` form (`evaluation/floors_json.py`), because its prompt is versioned and may describe several floors.

## Graph versus record

`FloorplanGraph` holds only what the model reads or writes.
`PlanRecord` (`record.py`) wraps it with what training and evaluation need but the model never produces: the image path, the source label and polygon of every room, and quality flags.
Keeping these apart means the model's contract stays small and stable, while the dataset can carry richer ground truth.
Records are stored as JSON Lines, with the image path relative to the JSONL file so a processed dataset can be moved as a whole.

## Converting CubiCasa5K

CubiCasa5K draws rooms, walls and doors as polygons in `model.svg`, and it stores no graph.
The converter (`converters/cubicasa.py`, with `cubicasa_svg.py` doing the parsing) recovers the graph from geometry.
All distances are in pixels of `F1_scaled.png`, whose coordinates the SVG uses.
They do not match `F1_original.png`.

| Step | Rule |
|---|---|
| Rooms | Each `Space` group is a room. Its polygon is repaired with `make_valid`, and the largest part is kept. |
| Door link | Only doors whose parent is a wall count. Grow the door polygon by 6 px, and link the two rooms it overlaps most. |
| Entrance | A door on an external wall is an exterior door. The room inside is an entrance when the door touches only one room, when its other neighbour is an open-air space (outdoor, porch, terrace, car port; not a balcony, loggia, glazed space or sun room), when both touched rooms lie on the inner face of the wall, or when the door is drawn up to 20 px off the room. On a plan with no wall tagged External, a door touching one room counts. |
| Opening link | Two rooms less than 3 px apart that face each other along at least 20 px of boundary. |
| Wall link | Two rooms within 30 px that face each other along at least 20 px of boundary, where walls cover at least 20% of the gap. Facing means straight across, so rooms that only meet diagonally at a wall cross are not linked. |
| Position | `representative_point()` of the polygon, scaled to the 0 to 1000 grid by the image width and height. |

A pair gets the strongest link found: door beats opening, and opening beats wall (`converters/assemble.py`).
Rooms are numbered in reading order, top to bottom and then left to right, so the order the model must produce is predictable.

The roadmap check found the pitfalls below, and each has a mitigation in the code.

| Pitfall | Evidence | Mitigation |
|---|---|---|
| Room-type label noise | About 20% of rooms are `Undefined` or `UserDefined`, and 6% of plans are mostly unlabeled. | Map Finnish names (`MH` is a bedroom, `OH` a living room, `KH` a bathroom, and so on) from the printed text. Anything still unmapped is `unknown`. Plans with more than 50% unknown rooms are flagged `mostly_unlabeled`, and training-data assembly filters on that flag. |
| Multi-floor plans share one image | 63 of 300 sampled plans. | One graph per floor, and no cross-floor links. |
| Appliance doors | 86% of plans have a fridge door, which is not a passage. | Only wall doors count. |
| Doors touching one room | 0.6% of wall doors. | The `entrance` flag when the wall is external, and a warning otherwise. |
| Invalid polygons | 1.5% of rooms. | The largest valid part. |
| Mirrored or rotated images, watermarks, small Finnish labels | 3 of 14 samples. | No flip augmentation that assumes upright text, and type accuracy evaluated separately on readable plans. |

### Entrance counts

The first rule missed most front doors: they open onto an outdoor space that the source draws as a room, so the door touches two rooms.
Full run over 4,992 plans, plans with 0 / 1 / 2 / 3+ entrance rooms:

| Split | Before | After |
|---|---|---|
| train | 1268 / 2391 / 345 / 190 | 24 / 1138 / 1522 / 1510 |
| val | 126 / 230 / 30 / 14 | 5 / 124 / 129 / 142 |
| test | 123 / 214 / 43 / 18 | 3 / 117 / 133 / 145 |
| all | 1517 / 2835 / 418 / 222 | 32 / 1379 / 1784 / 1797 |

Of the 1,517 plans with none, 1,490 had an external door onto an outdoor space.
The rest were doors with both rooms on one side, doors drawn past the room outline, and plans with no External wall tag.
Many plans now have several entrance rooms, because every exterior door counts (back and patio doors too), not only the front door.
32 plans stay at none, because the source gives no usable door: a porch-to-yard door only (12), balcony doors only (7), a door onto an annex room (5), a door on a wall not tagged External (4), doors onto a glazed porch or terrace only (2), or no door (2).
Same-side doors no longer create a door link between the two rooms (40 plans changed links).

The tests do not use real CubiCasa5K data, because it is CC BY-NC.
`tests/cubicasa_fixture.py` writes a small synthetic plan in the same SVG dialect, with one case for each rule above.

## Fetching and viewing

`data/http_zip.py` shows a useful trick.
A zip file keeps its table of contents at the end, and each member at a known byte offset.
Python's `zipfile` works on any seekable file, so a small file object that answers `seek` and `read` with HTTP `Range` requests lets it open the 5.5 GB archive and extract three small files per plan.

The overlay viewer (`viewer/overlay.py`) is the main way to check a converter.
A dot in the wrong room, or a door line that crosses three rooms, is something no unit test finds as fast as a glance.
The contact sheet shows many plans on one page with their counts and quality flags.

## Training format

Training uses TRL's SFT with `assistant_only_loss=True`.
Only the tokens the model has to produce get a loss: its reasoning, the graph JSON, tool calls and replies.
The image, the user's text and tool responses are masked out.
`tests/test_training_format.py` checks this on CPU with TRL's own vision collator and the Qwen3.5-4B processor, for a perception sample (image to graph) and an editing sample (a tool call, a tool message that returns a picture, and a reply).
A control test turns the masking off and shows that the same assertions then fail, so the checks are not vacuous.

**Thinking mode is on** (decision 4).
With the Qwen3.5 template, the generation prompt ends with `<think>\n`.
Training samples therefore carry their reasoning in `reasoning_content`, and the trained span starts right after `<think>\n`: the reasoning, then `</think>`, then the answer.
Training and serving must use the same template.
The roadmap report preferred non-thinking mode, to avoid an empty-think training artifact and TRL issue 8050 (tool calls that land in `reasoning_content` when the model skips the prefilled `<think>`).
Thinking mode was chosen anyway, so the editing episodes need a short reasoning text, and M2 and M5 should watch for that issue.

## Decisions recorded for M0

1. **Data.** Use all accessible datasets (CubiCasa5K, ResPlan, MSD, Swiss Dwellings).
   The whole project is non-commercial, so the CC BY-NC data of CubiCasa5K is fine.
   Real images exist only in CubiCasa5K.
   The others are vectors, so their images are renders, which are augmentation and not a substitute for real plans.
2. **Schema.** Keep wall links, an entrance flag, outdoor spaces as rooms, per-floor graphs, and a point per room.
   Amended 2026-11-19: the earlier plan of an optional bbox per room was dropped, so rooms are points only.
3. **Scope.** Retrieval is a later tool (M6) over a ResPlan pool.
   "Draw a new one" means composing a graph with the same edit tools, from an empty graph.
   Geometric generation is out of scope: it is a different research problem, and it contradicts the rule that geometry is not edited.
4. **Thinking mode** is on.
5. **ClearML is compute only.**
   Every job's outputs are downloaded locally after it finishes and then deleted from ClearML.
   Nothing from this project stays in the shared room-sketcher workspace, and it stays apart from company product work.

## Milestones

| # | Milestone | Notes |
|---|---|---|
| M0 | Decisions | The list above. |
| M1 | Schema, converters, overlay viewer, CPU format tests | This repository. The CubiCasa5K, ResPlan, MSD and Swiss Dwellings converters are built (ResPlan: `docs/resplan.md`, MSD and Swiss Dwellings: `docs/msd-swiss.md`). |
| M2 | Zero-shot baseline, served through vLLM | One cheap ClearML job. Measures the base model at two image resolutions and checks that it uses the 0 to 1000 grid. |
| M3 | Perception SFT | LoRA on the language model with the vision tower frozen, evaluated through vLLM with the adapter. |
| M4 | App skeleton | Core library (edit ops, validator, renderer, metrics), FastAPI agent loop and a React Flow UI. It depends only on the schema, so it can run in parallel with M2 and M3. The edit ops and validator are built (`docs/edit-operations.md`). |
| M5 | Editing agent | Programmatic edit episodes, then SFT, then multi-turn GRPO. Edit tools take flat scalar arguments such as room ids and enums. |
| M6 | Retrieval as a `search_plans` tool | Hard filters on room counts and typed adjacencies, then Weisfeiler-Lehman ranking over a ResPlan pool. |

Follow-ups to M1 that are not built yet: the zero-shot baseline run (its code is built, see `docs/evaluation.md`), any GPU or ClearML job, the backend and frontend, retrieval, and training scripts.

## Sources

The numbers in this page come from the roadmap check, which pulled 14 CubiCasa5K plans by range request, sampled 300 `model.svg` files for statistics, and verified the TRL and vLLM behavior on the pinned stack (transformers 5.20.2, TRL 1.18.1, torch 2.14.0, PEFT 0.21.3).

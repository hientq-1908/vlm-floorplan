# ResPlan: converter and renderer

ResPlan has 17,000 residential plans as vector geometry (shapely), with no images.
This page explains how the project turns them into room graphs and pictures, and why each choice was made.
The code is in `src/vlm_floorplan/resplan/`.

## What ResPlan is for here

- **Graph-only data.** Editing practice (add, remove, relink rooms) needs graphs, not pictures, and the later retrieval tool needs a pool of graphs to search.
- **Perception augmentation.** The renders are synthetic: no door swings, text, furniture or paper texture.
  Next to real plans (CubiCasa5K) they add variety.
  On their own they would teach the model a look that real captures do not have.
- **Limits to remember.** All plans are South Asian apartments, and about 6.9 % are near-duplicates of each other.
  The `augmented` split (683 rotated, flipped or scaled copies of 667 originals) is left out by default for the same reason.

## Pipeline

```text
fetch.py   GitHub zip (pinned commit, SHA-256 checked) -> data/resplan/ResPlan.pkl
load.py    restricted unpickler -> list of plan dicts of shapely geometry
frame.py   PlanView: plan units -> image pixels -> 0-1000 grid (one object, shared)
links.py   which rooms connect, and how
convert.py plan -> FloorplanGraph + per-room ground truth
render.py  plan -> PNG, drawn through the same PlanView
build.py   JSONL PlanRecords + images, one file per split
```

Run it with `uv run python -m vlm_floorplan.resplan.fetch` and then `uv run python -m vlm_floorplan.resplan.build`.
The output goes to `data/resplan/processed/`: `images/<id>.png`, `train.jsonl`, `val.jsonl`, `test.jsonl` and `ATTRIBUTION.txt`.
Records are `PlanRecord`s, the same shape as the CubiCasa5K ones, and `image` is relative to the JSONL file.
Nothing under `data/` is committed.

## Design choices

### The picture and the graph share one view

The graph's points are only useful if they land on the rooms in the image.
So the converter and the renderer both ask `PlanView` how plan coordinates map to the image, and nothing else.
`PlanView` fits the building (walls, rooms, doors and windows, but not the plot or neighbours) plus a 4 % margin into an image whose longer side is 1024 px.
A graph point `(x, y)` is at pixel `(x / 1000 * width, y / 1000 * height)`.
The tests check this by sampling the rendered pixel under every room's point.

ResPlan's y axis points up and an image's points down, so the map flips y.
The result looks like the plots in ResPlan's own README.

### Room types

Types come from `RESPLAN_TYPES` in `room_types.py`: living, kitchen, bedroom, bathroom, storage and stairs map to themselves, and a balcony becomes `outdoor`.
ResPlan merges corridors into `living` and does not say which part is which, so a `living` room can be a hallway too.
We do not try to guess.
Every `living` part of a plan is merged into one room, which is also what ResPlan's own graph does.
Every other polygon is its own room, so a plan with three bedrooms has three bedroom nodes.

The yard layers (`garden`, `parking`, `pool`) are not rooms.
ResPlan's own graph leaves them out, they have no doors, and they would add unlinked nodes.
They appear in only 543, 337 and 1 plans.

ResPlan rooms have no printed names, so `label` is empty.
The room's ResPlan category is kept in the record (`rooms[id].source_type`), next to its outline in grid coordinates.

### The front door is a flag, not a node

ResPlan's graph has a `front_door` node.
Our schema has an `entrance` flag on the room the door opens into.
The converter picks the one room the front door overlaps most (after a small tolerance), even when the door sits in a corner next to a second room.
This also removes a bug class: the roadmap report found a front-door edge that came out as `via_window`, because a window next to the door linked the door node to a room.
With no door node, that cannot happen.

### Links

`links.py` reimplements ResPlan's `plan_to_graph` and `add_adjacency_edges` (MIT, see `THIRD_PARTY_NOTICES.md`) with shapely only, because the upstream module imports OpenCV, matplotlib and networkx.

| Upstream edge | Our link | Notes |
|---|---|---|
| `via_door` | `door` | A door joins exactly the two rooms with the most contact, never three. |
| `via_window`, two interior rooms | `door` | Upstream relabels these too: ResPlan sometimes draws a door as a window. |
| `via_window`, a balcony involved | none | A window is not a passage. The pair falls through to `opening` or `wall`. |
| `via_opening` | `opening` | A wall-free stretch of the shared boundary at least two wall depths wide. |
| `adjacency` (within a wall gap) | `wall` | Adjacent, separated by a wall. |
| `fallback` | dropped | Upstream adds these only to keep the graph in one piece. |

A pair keeps its strongest link: door, then opening, then wall.

Two deliberate differences from upstream:

- Windows count as wall when looking for an opening.
  ResPlan cuts door and window areas out of its wall layer, so a window looks like a gap in the wall.
  Upstream counts that gap as an open passage, and we do not.
- `fallback` edges are dropped.
  In our schema a `wall` link promises "these rooms touch", and a nearest-neighbour fallback does not.
  A room that ends up with no link is listed in the record's `quality.warnings` instead.

Against upstream's `plan_to_graph` on the first 400 plans, 2,829 of 2,848 `via_door` edges came out as `door`.
The rest are the "exactly two rooms" rule.

### Results on the full dataset

A full build of all 17,000 plans (including the `augmented` split) gave:

- 8.2 rooms per plan on average (at most 22) and 11.8 links (at most 36).
- Links: 117,883 door, 75,694 wall and 6,682 opening.
- Exactly one entrance room in 16,989 plans. 11 plans have none: 5 have no front door, and 6 have a front door that touches no room.
- 112 plans have a room with no link, listed in `quality.warnings`.
- Every record loads as a `PlanRecord`, every graph survives a compact-JSON round trip, and every image has the size the graph states.
- The pixel under a room's point has the room's fill color for 99.7 % of rooms.
  The rest are thin rooms where a door or window covers the point.

### Room ids are in reading order

`r0` is the top-most room (ties go left to right), by the room's point.
Ids then do not depend on how ResPlan groups rooms by type, and a model can learn "r0 is the top-left room" more easily than an arbitrary order.

### One floor

ResPlan plans are single-floor, so every graph has one floor, numbered 1.

## Known limitations

- A window between two interior rooms is promoted to a door, as upstream does.
  When a room polygon overlaps its neighbour at a window, that gives a door where the plan really has an open boundary.
  It is rare, and we kept upstream's rule so the door links stay comparable to the published graphs.
- The renders are clean and synthetic.
  `--labels` draws English type names, but real plans print names in their local language (Finnish for CubiCasa5K), so labels are a style difference too.
- Plan coordinates are in ResPlan's plan units (at most 256 across), and the thresholds in `links.py` are in those units.

## Loading the pickle safely

`ResPlan.pkl` is a pickle, and unpickling can run code.
`load.py` only allows the three globals that the real file uses (found by listing every `find_class` call on it): `shapely.io.from_wkb` and two numpy scalar helpers.
Anything else raises, and a test checks that a malicious pickle never runs.

## Reproducible downloads

`fetch.py` pins one upstream commit and checks the zip against its SHA-256.
A force-pushed or tampered file is rejected, and every run gets the same plans.
To take a newer release, update `COMMIT` and `ZIP_SHA256` together.

## Licence and takedowns

The data is CC BY 4.0, so every graph or image derived from it needs attribution.
The builder writes `ATTRIBUTION.txt` next to its output, and `THIRD_PARTY_NOTICES.md` has the full notice.

ResPlan's takedown policy (`TAKEDOWN.md`, downloaded next to the data) says that a plan is removed in the next release and its id is listed in `REMOVED.md`.
Because we pin a release, we do not get that removal automatically.
The fetch script therefore checks the upstream `main` branch for `REMOVED.md` and saves it next to the data when it exists.
To honor it, write the removed ids to a file (one per line, `#` starts a comment) and run the builder with `--exclude-ids <file>`.
Plans listed there are skipped, and their images are not written.
To drop them from anything built earlier, rebuild.

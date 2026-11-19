# MSD and Swiss Dwellings

Both datasets are CC BY 4.0 and graph-only: they hold vector geometry in metres, and no images.
They feed editing practice (the agent edits graphs, so it needs many graphs) and the later retrieval pool.
Perception training still needs real images, which only CubiCasa5K supplies.

MSD (Modified Swiss Dwellings, 2023) was derived from Swiss Dwellings (2022).
They overlap in content, but they are different shapes of data, so each has its own converter.

| | MSD | Swiss Dwellings |
|---|---|---|
| Unit of one record | one storey of a building complex (16 to 75 rooms) | one apartment (duplexes have several floors) |
| Source | one networkx graph pickle per plan | rows of `geometries.csv` (WKT polygons) |
| Links | given as graph edges | derived from door and wall polygons |
| Room types | 9 classes | 77 `area` subtypes, folded to the same coarse set |
| Download | 4.8 GB zip, 40 MB needed | 932 MB zip, 290 MB needed |

## What is fetched, and what is not

Both archives are zips on servers that support HTTP range requests, so the fetch scripts read single members and never download a whole archive.

- MSD: only `graph_out/<id>.pickle`, about 10 kB each.
  `struct_in` and `full_out` are images of data the graph already holds.
  `graph_in` is the coarse zoning graph, with no room types or polygons.
  The test archive is skipped: its annotations are withheld, and it holds only `graph_in` and `struct_in`.
- Swiss Dwellings: only the `geometries.csv` member.
  `simulations.csv` (viewshed, noise, light) is 640 MB compressed and unused here.
  A subset is the first sites of the file, not a random sample, and a site is always kept whole (see below).

## MSD: what the real files look like

The dataset README says nodes have `roomtype` and a shapely `geometry`.
The files differ, so the loader follows the files:

- a node has `room_type` (an index into MSD's `ROOM_NAMES`), `geometry` (a plain list of vertices, in metres) and `centroid` (a torch tensor);
- an edge has `connectivity`, one of `door`, `passage` or `entrance`.

The pickles are read by `msd_pickle.py` with an unpickler that builds only dicts and lists.
Every other class the file names (`networkx.Graph`, the torch tensors) becomes an inert stand-in.
Unpickling a downloaded file with `pickle.load` would let the file run code, and this way neither networkx nor torch has to be installed.
A test feeds it a pickle that tries to run a shell command and checks that nothing runs.

## Link rules

| Source | Graph link |
|---|---|
| MSD `door` edge, Swiss `DOOR` polygon overlapping two rooms | `door` |
| MSD `passage` edge, Swiss rooms that touch (gap of at most 4 cm) | `opening` |
| MSD `entrance` edge | `door`, plus the `entrance` flag |
| Swiss `ENTRANCE_DOOR` polygon | the `entrance` flag on the room it opens into |
| Rooms a wall apart (gap of at most 0.4 m, with at least 0.5 m of one room's outline facing the other straight across that gap) | `wall` |

The thresholds come from measuring real plans, not from guessing.
In 40 MSD plans, door-linked rooms are a median 0.15 m apart, passage-linked rooms 0.01 m, and rooms that share a wall without any edge are up to about 0.4 m apart.
Door polygons in Swiss Dwellings are 0.17 m thick (median), and in thick walls they stop up to 0.5 m short of a room.
The converter therefore grows a door polygon in steps (5, 15, 30, 50 cm) until it reaches enough rooms.
Growing in steps keeps the tightest match, so a door in a thin wall is never linked to a room one wall further away.

Swiss Dwellings also has a wall layer, so a `wall` link there requires that at least 20 % of the strip between the two rooms is covered by wall polygons.
MSD has no wall layer, so MSD uses the distance rules alone.
Both share `adjacency.py`.

### The entrance flag

The schema says `entrance` means "an exterior door opens into this room".
Neither dataset has an outside, so the flag marks where one enters an apartment.

- Swiss Dwellings: the apartment's `ENTRANCE_DOOR` touches one of its rooms, the hall.
  The other side is the shared stairwell, which belongs to no apartment (its rows have an empty `apartment_id`) and is skipped.
- MSD: a plan holds several apartments, and an `entrance` edge joins an apartment's hall to shared circulation.
  The converter removes the entrance edges and counts the dwelling rooms (bedrooms, kitchens, and so on) in the component on each side.
  The side with more dwelling rooms is the apartment, and it gets the flag.
  If neither side has more, both ends are flagged, because the door does open into both and guessing would hide one of them.

## Coordinates and the virtual image

The schema wants coordinates on a 0-1000 grid over "the image the model saw", but these datasets have no image.
`PlanView` in `resplan/frame.py`, the same object the ResPlan converter and renderer use, invents one and records it in `graph.image`:

- the plan's bounding box (rooms and walls) plus a 4 % margin fits into a 1024 px long side,
- y is flipped, because plan y points up and image y points down.

Because all three vector datasets go through this one `PlanView`, a renderer draws the plan where the graph says it is, and rendered plans from all three look alike.
`PlanRecord.image` is empty for graph-only records.
The overlay viewer needs an image, so `view` and `sheet` do not work on them yet.

MSD plans are stored as one floor numbered 1, because MSD does not say which storey a plan is.
Swiss Dwellings floors are numbered 1, 2, ... by elevation, because the file gives elevations but no storey numbers.
The floors of a duplex share one coordinate frame, so a renderer should draw each floor on its own image.

## Room types

`room_types.py` holds `MSD_ROOM_NAMES` (index to name), `MSD_TYPES` and `SWISS_TYPES`.
The dataset's own label is always kept in `PlanRecord.rooms[...].source_type`.

MSD folded finer Swiss labels into its nine (`ROOM` became Bedroom, `KITCHEN_DINING` became Kitchen, lifts and voids became Stairs).
`SWISS_TYPES` follows MSD where MSD has an opinion, so one room gets one type in both datasets.
Swiss `SHAFT`, `TRANSPORT_SHAFT`, `LIGHTWELL` and `OUTDOOR_VOID` are not rooms and are dropped, as MSD does.
A test fails if a subtype of the full file is in neither table.
Most of the rare subtypes (offices, workshops, medical rooms) come from mixed-use buildings and map to `other`.

## Splits and ids

MSD records have no split: its own split withholds the test labels, and plans of one complex look alike, so the training pipeline should split.
Swiss Dwellings records are named `swiss_dwellings/<site id>/<apartment id>`.
Apartments of one site (a building complex) are near-duplicates, so the site is in the id, and a split by site needs no second pass over the CSV.

## Results on the full datasets

Both builds ran on the complete data without a single skipped plan (checked 2026-11-19).

| | MSD | Swiss Dwellings |
|---|---|---|
| Records | 4167 | 45 242 |
| Rooms per record (min, median, max) | 15, 25, 331 | 1, 8, 567 |
| Links | 126 045 wall, 116 374 door, 17 215 opening | 212 870 wall, 304 679 door, 43 993 opening |
| Rooms with no link at all | 114 of 132 232 | 1131 of 371 434 |
| Records with no entrance flag | 339 (no entrance edge) | 1742 floors of 47 000 |
| Build time | 3 minutes | 11 minutes |

Swiss Dwellings had 3629 doors that reach only one room and 278 that reach none, out of about 350 000; they are skipped and recorded in `quality.warnings`.
Sites were consecutive in the whole file, not only in the first 150 000 rows.
The largest records (hundreds of rooms) will not fit a model's context as compact JSON, so training should filter by room count.

## Known limits

- `geometries.csv` rows are consecutive per site, not per apartment (checked on 59 sites).
  The reader holds one site at a time and reports any site that reappears later, instead of silently building half-apartments.
- Links are derived from geometry, so they are only as good as the thresholds.
  A door that reaches fewer than two rooms is skipped, and the skip is recorded in the record's `quality.warnings`.
- Derived `wall` and `opening` links measure how far one room's outline faces the other straight across the gap, the same measure the CubiCasa5K converter uses (`converters/facing.py`).
  Rooms that meet only at a corner, or overlap at an offset corner by less than 0.5 m, face each other along too little and stay unlinked, even in MSD, which has no wall layer to check.
  The measure looks only at outlines, so in MSD two rooms 0.4 m apart are linked as `wall` even if the gap is really a shaft or a gap in the drawing.
- Swiss `ROOM` becoming `bedroom` is a coarse guess: it is an undesignated private room, mostly a bedroom.

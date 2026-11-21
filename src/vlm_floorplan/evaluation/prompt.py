"""The fixed zero-shot prompt.

One prompt for every model and every resolution, so a score difference comes from the
model or the image size and never from the wording. Change the text only together with
`PROMPT_VERSION`, which is stored in every run's config.

The prompt is built from the schema enums, so the list of room types it offers cannot
drift from the one the metrics score against.
"""

from __future__ import annotations

from vlm_floorplan.schema import GRID_MAX, LinkKind, RoomType

PROMPT_VERSION = "zero-shot-v1"

_EXAMPLE = (
    '{"floors":[{"floor":1,"rooms":[{"id":"r0","type":"entry","xy":[474,735],"entrance":true},'
    '{"id":"r1","type":"living","label":"OH","xy":[583,295]}],"links":[["r0","r1","door"]]}]}'
)


def build_prompt() -> str:
    room_types = ", ".join(t.value for t in RoomType if t != RoomType.UNKNOWN)
    link_kinds = ", ".join(k.value for k in LinkKind)
    return f"""You are given a floorplan image. Describe it as a room graph.

Output one JSON object and nothing after it, with this structure:
- "floors": a list with one entry per floor drawn in the image.
- each floor has "floor" (1 = first floor), "rooms" and "links".
- each room has "id" ("r0", "r1", ... unique across the whole image), "type", "xy",
  and optionally "label" (the text printed in the room) and "entrance": true.
- "xy" is [x, y], a point inside the room, on a grid from 0 to {GRID_MAX} over the whole
  image (x to the right, y down; [0,0] is the top-left corner).
- "type" is one of: {room_types}.
- "entrance": true marks a room that an exterior door opens into.
- every balcony, terrace or porch is a room of type "terrace".
- "links" lists connections between two rooms on the same floor as ["a","b","kind"],
  where kind is one of: {link_kinds}.
  "door": a door connects them. "opening": no wall between them (open plan).
  "wall": they touch but a wall separates them.
- give each pair of rooms at most one link.

Example of the format (not of this plan):
{_EXAMPLE}"""

"""Read an MSD access graph from its pickle file, without trusting the pickle.

MSD ships each plan's graph as a pickled `networkx.Graph`. Unpickling runs code
named inside the file, so a pickle from the internet must never go through
`pickle.load`. This module reads it with an unpickler that only ever builds
plain dicts and lists:

- `dict` and `OrderedDict` are real,
- every other class or function the file mentions (`networkx.Graph`, and the
  `torch` tensors that hold each room's centroid) becomes an inert stand-in
  that records its arguments and does nothing.

That also means neither networkx nor torch has to be installed. The centroid
tensors are not needed: the converter computes its own interior point from the
room polygon.

What the real files look like (checked against the 4TU v2 train archive; the
dataset README describes `roomtype` and shapely geometry, but the files differ):

- node attributes: `room_type` (an index, see `MSD_ROOM_NAMES`), `geometry`
  (a plain list of (x, y) vertices in metres, the polygon's closed exterior) and `centroid`;
- edge attribute: `connectivity`, one of "door", "passage" or "entrance".
"""

from __future__ import annotations

import io
import pickle
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

Point = tuple[float, float]

_REAL_CLASSES: dict[tuple[str, str], type] = {
    ("builtins", "dict"): dict,
    ("collections", "OrderedDict"): OrderedDict,
}


class _Inert:
    """Stand-in for any class the pickle mentions: remembers its state, runs nothing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args

    def __setstate__(self, state: Any) -> None:
        self.state = state


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> type:
        return _REAL_CLASSES.get((module, name), _Inert)


@dataclass(frozen=True)
class MsdNode:
    room_type: int
    polygon: list[Point]


@dataclass(frozen=True)
class MsdEdge:
    a: int
    b: int
    connectivity: str


@dataclass(frozen=True)
class MsdGraph:
    nodes: dict[int, MsdNode]
    edges: list[MsdEdge]


def parse_msd_graph(data: bytes) -> MsdGraph:
    """Parse the bytes of one `graph_out/<id>.pickle`. Raises `ValueError` on any other shape."""
    try:
        graph = _SafeUnpickler(io.BytesIO(data)).load()
        raw_nodes, raw_adj = graph.state["_node"], graph.state["_adj"]
        nodes = {
            int(key): MsdNode(
                room_type=int(attrs["room_type"]),
                polygon=[(float(x), float(y)) for x, y in attrs["geometry"]],
            )
            for key, attrs in raw_nodes.items()
        }
        edges = [
            MsdEdge(a=int(a), b=int(b), connectivity=str(attrs["connectivity"]))
            for a, neighbours in raw_adj.items()
            for b, attrs in neighbours.items()
            if int(a) < int(b)  # networkx stores every undirected edge twice
        ]
    except (pickle.UnpicklingError, AttributeError, KeyError, EOFError, TypeError) as err:
        # ValueError (a bad number) already is the error type callers expect.
        raise ValueError(f"not an MSD networkx graph pickle: {err!r}") from err
    return MsdGraph(nodes=nodes, edges=edges)

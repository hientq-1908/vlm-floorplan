"""Build MSD-style graph pickles from small synthetic plans (MSD itself is not committed).

The real files are pickled `networkx.Graph`s whose centroids are `torch` tensors.
Neither library is installed here, so the helper plants stand-in modules under
those names just long enough to pickle with the same class paths, then removes
them (restoring any real module that was already imported, such as torch when the ml group
is installed). The result has the byte-level shape of a real file: a `networkx.classes.graph.Graph`
object whose `__dict__` holds `_node` and `_adj`.
"""

from __future__ import annotations

import pickle
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

Rect = tuple[float, float, float, float]  # x1, y1, x2, y2 in metres

BEDROOM, LIVINGROOM, KITCHEN, DINING, CORRIDOR, STAIRS, STOREROOM, BATHROOM, BALCONY = range(9)


def ring(rect: Rect) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = rect
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]


class _Graph:
    __module__ = "networkx.classes.graph"
    __qualname__ = "Graph"


class _Tensor:
    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return _rebuild_tensor_v2, (b"storage-bytes", 0, (2,), (1,), False)


def _rebuild_tensor_v2(*args: Any) -> None:  # never called: pickling only records its path
    raise AssertionError("not meant to run")


_rebuild_tensor_v2.__module__ = "torch._utils"
_rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"


@contextmanager
def _standins() -> Iterator[None]:
    # pickle imports a class's module by its dotted path, so the parents must exist too.
    planted = {
        "networkx": {},
        "networkx.classes": {},
        "networkx.classes.graph": {"Graph": _Graph},
        "torch": {},
        "torch._utils": {"_rebuild_tensor_v2": _rebuild_tensor_v2},
    }
    originals = {name: sys.modules.get(name) for name in planted}
    for name, attrs in planted.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        sys.modules[name] = module
    try:
        yield
    finally:
        for name, original in originals.items():
            if original is None:
                del sys.modules[name]
            else:  # deleting a real torch here would make the next `import torch` run it twice
                sys.modules[name] = original


def msd_pickle(rooms: dict[int, tuple[int, Rect]], edges: list[tuple[int, int, str]]) -> bytes:
    """Pickle a graph: `rooms` maps node key to (room_type index, rectangle)."""
    graph = _Graph()
    adjacency: dict[int, dict[int, dict[str, str]]] = {key: {} for key in rooms}
    for a, b, connectivity in edges:
        adjacency[a][b] = {"connectivity": connectivity}
        adjacency[b][a] = {"connectivity": connectivity}
    graph.__dict__.update(
        graph={},
        _node={
            key: {"geometry": ring(rect), "room_type": room_type, "centroid": _Tensor()}
            for key, (room_type, rect) in rooms.items()
        },
        _adj=adjacency,
    )
    with _standins():
        return pickle.dumps(graph, protocol=2)


# One apartment (hall, bedroom, kitchen, living room) next to a shared stairwell.
# Rooms of the apartment are 0.2 m apart (a wall), except where noted.
APARTMENT_ROOMS: dict[int, tuple[int, Rect]] = {
    0: (CORRIDOR, (0, 0, 3, 2)),  # the apartment's hall
    1: (BEDROOM, (3.2, 0, 6, 2)),
    2: (KITCHEN, (0, 2.2, 3, 5)),
    3: (LIVINGROOM, (3.2, 2.2, 6, 5)),
    4: (STAIRS, (-3.2, 0, -0.2, 2)),  # shared stairwell
    5: (BALCONY, (6.2, 2.2, 8, 5)),
}
APARTMENT_EDGES: list[tuple[int, int, str]] = [
    (0, 1, "door"),  # hall - bedroom
    (2, 3, "passage"),  # open-plan kitchen and living room
    (0, 4, "entrance"),  # front door from the stairwell
    (3, 5, "door"),  # living room - balcony
]


def apartment_pickle() -> bytes:
    return msd_pickle(APARTMENT_ROOMS, APARTMENT_EDGES)

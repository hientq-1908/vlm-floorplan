"""Load ResPlan's pickle, its splits and the takedown exclusion list.

`ResPlan.pkl` is a pickle, and unpickling can run arbitrary code.
The file is a plain list of dicts of shapely geometry, so the loader only lets
the few globals that such a file needs through (found by listing every
`find_class` call on the real file: `shapely.io.from_wkb` and two numpy scalars).
Anything else raises, so a tampered file fails instead of executing.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

PlanDict = dict[str, Any]

_ALLOWED_GLOBALS = frozenset(
    {
        ("shapely.io", "from_wkb"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy", "dtype"),
    }
)

SPLIT_NAMES = ("train", "val", "test", "augmented")
"""`augmented` holds 683 rotated, flipped or scaled copies of 667 originals."""


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(
                f"refusing to unpickle {module}.{name}: not part of a ResPlan file"
            )
        return super().find_class(module, name)


def load_plans(pickle_path: Path) -> list[PlanDict]:
    """Read every plan from `ResPlan.pkl`, with key typos fixed."""
    with pickle_path.open("rb") as handle:
        plans = _SafeUnpickler(handle).load()
    if not isinstance(plans, list) or not all(isinstance(plan, dict) for plan in plans):
        raise ValueError(f"{pickle_path} is not a list of plan dicts")
    return [normalize_keys(plan) for plan in plans]


def normalize_keys(plan: PlanDict) -> PlanDict:
    """Fix the one known key typo (`balacony`), as ResPlan's own utilities do."""
    if "balacony" in plan and "balcony" not in plan:
        plan["balcony"] = plan.pop("balacony")
    return plan


def load_splits(split_path: Path) -> dict[str, str]:
    """Map plan id -> split name, from `split.json`."""
    raw = json.loads(split_path.read_text(encoding="utf-8"))
    unknown = set(raw) - set(SPLIT_NAMES)
    if unknown:
        raise ValueError(f"{split_path}: unexpected split names {sorted(unknown)}")
    return {int(plan_id): name for name, ids in raw.items() for plan_id in ids}


def load_excluded_ids(path: Path | None) -> set[int]:
    """Plan ids to skip, one per line (`#` starts a comment).

    This is how a takedown is honored: ResPlan removes a plan in its next release and
    lists the id, so a copy downloaded earlier can be cleaned by listing the same id here.
    """
    if path is None:
        return set()
    ids: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        text = re.sub(r"#.*", "", line).strip()
        if text:
            ids.add(int(text))
    return ids


def geometry_parts(geom: BaseGeometry | None) -> list[BaseGeometry]:
    """The non-empty pieces of a layer: one polygon, or each part of a multi-geometry."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    parts = getattr(geom, "geoms", None)
    if parts is None:
        return [geom]
    found: list[BaseGeometry] = []
    for part in parts:
        found.extend(geometry_parts(part))
    return found


def polygons(geom: BaseGeometry | None) -> list[Polygon]:
    """Only the polygon pieces of a layer (lines and points are dropped)."""
    return [part for part in geometry_parts(geom) if isinstance(part, Polygon)]

"""Small geometry helpers shared by the converters.

Converters work in two coordinate spaces:

1. Source space (CubiCasa5K SVG pixels, ResPlan plan units), where the
   connectivity heuristics run, because their thresholds are in those units.
2. Grid space: 0-1000 over the image, where the graph's points and boxes live.

`GridFrame` is the affine map from the first to the second.
"""

from __future__ import annotations

from dataclasses import dataclass

from shapely import affinity, make_valid
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from vlm_floorplan.schema import GRID_MAX

Ring = list[tuple[float, float]]


def polygon_parts(geom: BaseGeometry) -> list[Polygon]:
    """All non-empty polygons inside any geometry (Polygon, Multi*, collections)."""
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    parts: list[Polygon] = []
    for sub in getattr(geom, "geoms", []):
        parts.extend(polygon_parts(sub))
    return parts


def largest_polygon(geom: BaseGeometry) -> Polygon | None:
    """Repair a geometry and keep its largest polygon part.

    Source polygons are sometimes self-intersecting. `make_valid` repairs them,
    but may return a GeometryCollection; the room is then its biggest piece.
    """
    parts = polygon_parts(make_valid(geom))
    if not parts:
        return None
    return max(parts, key=lambda part: part.area)


def _clamp(value: float) -> int:
    return min(GRID_MAX, max(0, round(value)))


@dataclass(frozen=True)
class GridFrame:
    """Affine map from source coordinates to the 0-1000 grid.

    grid_x = scale_x * x + offset_x, grid_y = scale_y * y + offset_y.
    A negative `scale_y` flips the y axis (used for ResPlan, whose y points up).
    """

    scale_x: float
    scale_y: float
    offset_x: float = 0.0
    offset_y: float = 0.0

    @classmethod
    def for_image(cls, width: float, height: float) -> GridFrame:
        """Frame for geometry already in image pixels (x right, y down)."""
        return cls(scale_x=GRID_MAX / width, scale_y=GRID_MAX / height)

    def apply(self, geom: BaseGeometry) -> BaseGeometry:
        """Move a geometry into grid space (float coordinates)."""
        matrix = [self.scale_x, 0.0, 0.0, self.scale_y, self.offset_x, self.offset_y]
        return affinity.affine_transform(geom, matrix)

    def point(self, geom: BaseGeometry) -> tuple[int, int]:
        """A grid point guaranteed to lie inside the geometry.

        `representative_point` is always inside the shape; a centroid is not
        (think of an L-shaped room).
        """
        p = self.apply(geom).representative_point()
        return _clamp(p.x), _clamp(p.y)

    def bbox(self, geom: BaseGeometry) -> tuple[int, int, int, int]:
        """Grid bounding box (x1, y1, x2, y2) of a geometry."""
        x1, y1, x2, y2 = self.apply(geom).bounds
        return _clamp(x1), _clamp(y1), _clamp(x2), _clamp(y2)

    def rings(self, geom: BaseGeometry, decimals: int = 1) -> list[Ring]:
        """Exterior rings of each polygon part, in grid space, rounded.

        These are kept in the dataset (not the model target) so evaluation can
        match a predicted point to the room polygon that contains it.
        """
        moved = self.apply(geom)
        return [
            [(round(x, decimals), round(y, decimals)) for x, y in part.exterior.coords]
            for part in polygon_parts(moved)
        ]

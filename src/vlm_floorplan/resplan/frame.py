"""One shared view of a plan: where it sits in the image and on the 0-1000 grid.

The converter and the renderer must agree exactly on how plan coordinates map to
the image, or the graph's points would not land on the rooms they describe.
Both therefore ask this module, and nothing else, for the mapping.

ResPlan plan coordinates are in plan units (at most 256 across), with y pointing up,
like a mathematical plot. Images have y pointing down, so the map flips y.
That also makes the rendered plan look like the plots in ResPlan's own README.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shapely.geometry.base import BaseGeometry

from vlm_floorplan.geometry import GridFrame
from vlm_floorplan.resplan.load import geometry_parts
from vlm_floorplan.schema import GRID_MAX, ImageSize

# Layers that make up the building itself. The plot ("land"), the "neighbor"
# footprints and loose yard features ("garden", "parking", "pool") are left out
# on purpose: they would shrink the plan to a corner of a mostly empty image.
BUILDING_LAYERS = (
    "inner",
    "living",
    "kitchen",
    "bedroom",
    "bathroom",
    "balcony",
    "storage",
    "stair",
    "wall",
    "door",
    "window",
    "front_door",
)

DEFAULT_LONG_SIDE = 1024
"""Pixels along the longer image side. 1024 px costs 768 Qwen3.5 image tokens (see the roadmap)."""

DEFAULT_MARGIN = 0.04
"""Blank border around the building, as a fraction of its longer side."""


@dataclass(frozen=True)
class PlanView:
    """Plan coordinates -> image pixels -> 0-1000 grid, for one plan."""

    min_x: float
    max_y: float
    """Top edge of the view in plan units (y points up in the plan, down in the image)."""
    width: float
    height: float
    """Size of the view in plan units."""
    image: ImageSize

    @classmethod
    def for_plan(
        cls,
        plan: dict[str, Any],
        long_side: int = DEFAULT_LONG_SIDE,
        margin: float = DEFAULT_MARGIN,
    ) -> PlanView:
        """Fit the building's bounding box plus a margin into an image with `long_side` pixels."""
        bounds = [
            geom.bounds
            for layer in BUILDING_LAYERS
            for geom in geometry_parts(plan.get(layer))
            if not geom.is_empty
        ]
        if not bounds:
            raise ValueError("plan has no building geometry to view")
        x1 = min(b[0] for b in bounds)
        y1 = min(b[1] for b in bounds)
        x2 = max(b[2] for b in bounds)
        y2 = max(b[3] for b in bounds)
        pad = margin * max(x2 - x1, y2 - y1)
        width, height = (x2 - x1) + 2 * pad, (y2 - y1) + 2 * pad
        scale = long_side / max(width, height)
        image = ImageSize(w=max(1, round(width * scale)), h=max(1, round(height * scale)))
        return cls(min_x=x1 - pad, max_y=y2 + pad, width=width, height=height, image=image)

    @property
    def grid(self) -> GridFrame:
        """Affine map from plan units to the 0-1000 grid (y flipped)."""
        scale_x = GRID_MAX / self.width
        scale_y = -GRID_MAX / self.height
        return GridFrame(
            scale_x=scale_x,
            scale_y=scale_y,
            offset_x=-self.min_x * scale_x,
            offset_y=-self.max_y * scale_y,
        )

    def pixel_frame(self) -> GridFrame:
        """Affine map from plan units to image pixels (y flipped), for drawing."""
        scale_x = self.image.w / self.width
        scale_y = -self.image.h / self.height
        return GridFrame(
            scale_x=scale_x,
            scale_y=scale_y,
            offset_x=-self.min_x * scale_x,
            offset_y=-self.max_y * scale_y,
        )

    def to_pixels(self, geom: BaseGeometry) -> BaseGeometry:
        return self.pixel_frame().apply(geom)

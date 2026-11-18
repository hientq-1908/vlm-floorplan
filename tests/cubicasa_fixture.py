"""A synthetic CubiCasa5K-style plan for tests (no real dataset files are committed).

CubiCasa5K is CC BY-NC, so its images and derived graphs must not live in this
repository. Instead this module writes a small plan in the same SVG dialect:

    x:      20 ... 290 | 310 ... 440 | 440 ... 580
    y 20-200   kitchen | living      | dining           (image is 600 x 400 px)
    y 220-380  bedroom | unlabeled   | storage

Walls are 20 px thick. What the converter should find:

- kitchen -door- living   (a door in the wall between them)
- living -opening- dining (they touch, no wall)
- the living room has an exterior door: `entrance`
- a fridge door inside the kitchen (parent is furniture, not a wall): ignored
- storage is drawn as a self-intersecting "bowtie" polygon: repaired
- bedroom is "Space UserDefined" with the printed name "MH": Finnish for bedroom
- the bottom-right-middle room is "Space Undefined" with no name: unknown
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

WIDTH, HEIGHT = 600, 400


def _rect(x1: float, y1: float, x2: float, y2: float) -> str:
    return f"{x1},{y1} {x2},{y1} {x2},{y2} {x1},{y2}"


def _space(space_class: str, points: str, label: str | None = None) -> str:
    text = ""
    if label is not None:
        text = f'<g class="TextLabel NameLabel"><text>{label}</text></g>'
    return f'<g class="{space_class}"><polygon points="{points}"/>{text}</g>'


def _wall(points: str, external: bool = False, doors: str = "") -> str:
    kind = "Wall External" if external else "Wall Internal"
    return f'<g class="{kind}"><polygon points="{points}"/>{doors}</g>'


def _door(points: str) -> str:
    return f'<g class="Door Swing Beside"><polygon points="{points}"/></g>'


def _floor_one() -> str:
    bowtie = "450,220 580,380 580,220 450,380"
    parts = [
        _space("Space Kitchen", _rect(20, 20, 290, 200), "KEITTIÖ"),
        _space("Space LivingRoom", _rect(310, 20, 440, 200)),
        _space("Space Dining", _rect(440, 20, 580, 200)),
        _space("Space UserDefined", _rect(20, 220, 290, 380), "MH"),
        _space("Space Undefined", _rect(310, 220, 450, 380), "Undefined"),
        _space("Space Storage", bowtie),
        # exterior walls; the top one has the entrance door into the living room
        _wall(_rect(0, 0, 600, 20), external=True, doors=_door(_rect(330, 0, 400, 20))),
        _wall(_rect(0, 380, 600, 400), external=True),
        _wall(_rect(0, 0, 20, 400), external=True),
        _wall(_rect(580, 0, 600, 400), external=True),
        # interior walls; the vertical one has the kitchen-living door
        _wall(_rect(290, 20, 310, 380), doors=_door(_rect(290, 80, 310, 120))),
        _wall(_rect(20, 200, 580, 220)),
        # a fridge: its door is not a passage between rooms
        '<g class="FixedFurniture ElectricalAppliance Refrigerator">'
        f'<polygon points="{_rect(30, 30, 60, 60)}"/>'
        f"{_door(_rect(30, 55, 60, 65))}</g>",
    ]
    return '<g class="Floorplan Floor-1">' + "".join(parts) + "</g>"


def _floor_two() -> str:
    parts = [
        _space("Space LivingRoom", _rect(20, 20, 290, 380)),
        _space("Space Bedroom", _rect(310, 20, 580, 380)),
        _wall(_rect(290, 20, 310, 380), doors=_door(_rect(290, 150, 310, 250))),
    ]
    return '<g class="Floorplan Floor-2">' + "".join(parts) + "</g>"


def write_wall_cross_plan(plan_dir: Path) -> Path:
    """Four 200 px rooms around a 15 px wall cross, as in many CubiCasa5K flats.

        x:     0 ... 200 | 215 ... 415
        y 0-200  kitchen | living
        y 215-415 bedroom | dining

    Each room faces two neighbours through a wall; the diagonal pairs (kitchen and
    dining, living and bedroom) only meet at the junction and must stay unlinked.
    """
    plan_dir.mkdir(parents=True, exist_ok=True)
    parts = [
        _space("Space Kitchen", _rect(0, 0, 200, 200)),
        _space("Space LivingRoom", _rect(215, 0, 415, 200)),
        _space("Space Bedroom", _rect(0, 215, 200, 415)),
        _space("Space Dining", _rect(215, 215, 415, 415)),
        _wall(_rect(200, 0, 215, 415)),
        _wall(_rect(0, 200, 415, 215)),
    ]
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="415" height="415"><g class="Model">'
        '<g class="Floorplan Floor-1">' + "".join(parts) + "</g></g></svg>"
    )
    (plan_dir / "model.svg").write_text(svg, encoding="utf-8")
    Image.new("RGB", (415, 415), (255, 255, 255)).save(plan_dir / "F1_scaled.png")
    return plan_dir


def write_cubicasa_plan(plan_dir: Path, two_floors: bool = False) -> Path:
    """Write model.svg and F1_scaled.png into `plan_dir`; returns the directory."""
    plan_dir.mkdir(parents=True, exist_ok=True)
    floors = _floor_one() + (_floor_two() if two_floors else "")
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}">'
        f'<g class="Model">{floors}</g></svg>'
    )
    (plan_dir / "model.svg").write_text(svg, encoding="utf-8")
    Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255)).save(plan_dir / "F1_scaled.png")
    return plan_dir

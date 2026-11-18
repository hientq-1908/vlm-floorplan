"""An HTML contact sheet: many overlays on one page, for eyeballing a converter.

`write_contact_sheet` renders every record's overlay to a PNG (downsized) and
writes an `index.html` next to them with the plan id, counts and quality flags.
Open the page in a browser and scan for dots in the wrong room, missing doors,
or plans flagged mostly unlabeled.
"""

from __future__ import annotations

import html
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from PIL import Image

from vlm_floorplan.data.jsonl import image_path
from vlm_floorplan.record import PlanRecord
from vlm_floorplan.schema import LinkKind, RoomType
from vlm_floorplan.viewer.overlay import LINK_STYLES, TYPE_COLORS, draw_overlay

THUMBNAIL_LONG_SIDE: Final = 900

_STYLE = """
body { font-family: system-ui, sans-serif; margin: 1rem; background: #fafafa; color: #222; }
.legend span { display: inline-block; margin: 0 .8rem .3rem 0; }
.dot { display: inline-block; width: .8em; height: .8em; border-radius: 50%; margin-right: .3em; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 1rem; }
.card { background: white; border: 1px solid #ddd; border-radius: 6px; padding: .6rem; }
.card img { width: 100%; height: auto; }
.meta { font-size: .85rem; color: #555; }
.flag { color: #b00; font-weight: 600; }
footer { margin-top: 2rem; font-size: .8rem; color: #777; }
"""


def overlay_file_name(plan_id: str) -> str:
    """A file name for a plan id such as "cubicasa5k/colorful/8759"."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", plan_id) + ".png"


def _legend() -> str:
    types = "".join(
        f'<span><i class="dot" style="background: rgb{color}"></i>{kind.value}</span>'
        for kind, color in TYPE_COLORS.items()
        if kind != RoomType.UNKNOWN
    )
    unknown = '<span><i class="dot" style="border: 2px solid rgb(160,160,160)"></i>unknown</span>'
    links = "".join(
        f'<span><b style="color: rgb{LINK_STYLES[kind].color}">&#9473;</b> {kind.value}</span>'
        for kind in LinkKind
    )
    ring = '<span><i class="dot" style="border: 2px solid rgb(20,160,60)"></i>entrance</span>'
    return f'<div class="legend">{types}{unknown}{ring}<br>{links}</div>'


def _card(record: PlanRecord, picture: str) -> str:
    rooms = record.graph.rooms()
    kinds = Counter(link.kind.value for floor in record.graph.floors for link in floor.links)
    link_text = ", ".join(f"{count} {kind}" for kind, count in sorted(kinds.items())) or "no links"
    flags = ""
    if record.quality.mostly_unlabeled:
        flags = '<span class="flag">mostly unlabeled</span> '
    warnings = "; ".join(record.quality.warnings)
    return (
        '<div class="card">'
        f'<img src="{html.escape(picture)}" alt="{html.escape(record.id)}" loading="lazy">'
        f"<div><b>{html.escape(record.id)}</b> {html.escape(record.split or '')}</div>"
        f'<div class="meta">{flags}{len(record.graph.floors)} floor(s), {len(rooms)} rooms, '
        f"{link_text}; {record.quality.unlabeled_fraction:.0%} unlabeled</div>"
        f'<div class="meta">{html.escape(warnings)}</div>'
        "</div>"
    )


def render_overlay_png(jsonl_path: Path, record: PlanRecord, target: Path) -> None:
    """Draw one record's overlay and save it, downsized, to `target`."""
    with Image.open(image_path(jsonl_path, record)) as image:
        overlay = draw_overlay(image, record.graph)
    overlay.thumbnail((THUMBNAIL_LONG_SIDE, THUMBNAIL_LONG_SIDE))
    target.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(target)


def write_contact_sheet(jsonl_path: Path, records: Sequence[PlanRecord], out_dir: Path) -> Path:
    """Write `out_dir/index.html` and the overlay PNGs it shows; returns the HTML path."""
    cards = []
    for record in records:
        name = overlay_file_name(record.id)
        render_overlay_png(jsonl_path, record, out_dir / "overlays" / name)
        cards.append(_card(record, f"overlays/{name}"))
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>Floorplan graphs</title><style>{_STYLE}</style></head><body>"
        f"<h1>Floorplan graphs ({len(records)})</h1>{_legend()}"
        f'<div class="cards">{"".join(cards)}</div>'
        "<footer>Overlays of CubiCasa5K plans derive from CC BY-NC 4.0 images: "
        "for personal, non-commercial use only; do not publish them.</footer>"
        "</body></html>"
    )
    index = out_dir / "index.html"
    out_dir.mkdir(parents=True, exist_ok=True)
    index.write_text(page, encoding="utf-8")
    return index

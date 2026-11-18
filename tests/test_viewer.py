"""Overlay drawing, the contact sheet and the command line."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageChops

from vlm_floorplan import cli
from vlm_floorplan.data.jsonl import write_records
from vlm_floorplan.record import PlanRecord
from vlm_floorplan.schema import RoomType
from vlm_floorplan.viewer.contact_sheet import overlay_file_name, write_contact_sheet
from vlm_floorplan.viewer.overlay import LINK_STYLES, TYPE_COLORS, draw_overlay


def test_every_room_type_and_link_kind_has_a_style() -> None:
    assert set(TYPE_COLORS) == set(RoomType)
    assert len(LINK_STYLES) == 3


def test_overlay_draws_on_a_copy_and_keeps_the_image_size(cubicasa_record: PlanRecord) -> None:
    blank = Image.new("RGB", (600, 400), (255, 255, 255))
    result = draw_overlay(blank, cubicasa_record.graph)
    assert result.size == blank.size
    assert blank.getbbox() is not None and ImageChops.invert(blank).getbbox() is None, (
        "input untouched"
    )
    assert ImageChops.difference(result, blank).getbbox() is not None


def test_each_room_dot_is_drawn_in_its_type_color(cubicasa_record: PlanRecord) -> None:
    blank = Image.new("RGB", (600, 400), (255, 255, 255))
    result = draw_overlay(blank, cubicasa_record.graph)
    for room in cubicasa_record.graph.rooms():
        if room.type == RoomType.UNKNOWN:
            continue  # drawn hollow
        pixel = result.getpixel((round(room.xy[0] * 0.6), round(room.xy[1] * 0.4)))
        assert pixel == TYPE_COLORS[room.type], room.id


def test_overlay_works_on_other_image_sizes_and_modes(cubicasa_record: PlanRecord) -> None:
    grey = Image.new("L", (1200, 800), 200)
    assert draw_overlay(grey, cubicasa_record.graph).size == (1200, 800)


def test_contact_sheet_lists_every_plan_and_its_flags(
    tmp_path: Path, cubicasa_record: PlanRecord
) -> None:
    jsonl = tmp_path / "plans.jsonl"
    record = cubicasa_record.model_copy(
        update={"image": str(Path("plans/colorful/1/F1_scaled.png"))}
    )
    # the fixture's image lives at tmp_path / "plans" / ..., next to the JSONL
    write_records(jsonl, [record])
    index = write_contact_sheet(jsonl, [record], tmp_path / "sheet")

    page = index.read_text()
    assert "cubicasa5k/colorful/1" in page
    assert "6 rooms" in page and "1 door" in page and "2 opening" in page
    overlay = tmp_path / "sheet" / "overlays" / overlay_file_name(record.id)
    assert overlay.exists()
    assert "CC BY-NC" in page


def test_overlay_file_names_are_filesystem_safe() -> None:
    assert overlay_file_name("cubicasa5k/colorful/8759") == "cubicasa5k_colorful_8759.png"


def test_cli_build_view_and_sheet_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from cubicasa_fixture import write_cubicasa_plan

    data = tmp_path / "data"
    write_cubicasa_plan(data / "cubicasa5k" / "colorful" / "1")
    out = tmp_path / "processed"
    assert cli.main(["build", "cubicasa", "--data-dir", str(data), "--out-dir", str(out)]) == 0
    jsonl = out / "cubicasa5k.jsonl"

    png = tmp_path / "plan.png"
    assert cli.main(["view", str(jsonl), "--id", "cubicasa5k/colorful/1", "--out", str(png)]) == 0
    assert Image.open(png).size == (600, 400)

    assert cli.main(["sheet", str(jsonl), "--out-dir", str(tmp_path / "sheet")]) == 0
    assert (tmp_path / "sheet" / "index.html").exists()
    assert "wrote 1 records" in capsys.readouterr().out


def test_cli_view_reports_unknown_ids(tmp_path: Path, cubicasa_record: PlanRecord) -> None:
    jsonl = tmp_path / "plans" / "p.jsonl"
    write_records(jsonl, [cubicasa_record.model_copy(update={"image": "x.png"})])
    with pytest.raises(SystemExit, match="no record with id"):
        cli.main(["view", str(jsonl), "--id", "nope"])

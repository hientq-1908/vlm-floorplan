"""Shared fixtures.

Markers (declared in pyproject.toml):
    network  needs internet (dataset servers): skipped by default, run `pytest -m network`
    hf       needs the ml group and the HF processor: skips itself when unavailable;
             deselect explicitly with `pytest -m "not hf"`
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))  # lets tests import cubicasa_fixture

from cubicasa_fixture import write_cubicasa_plan  # noqa: E402

from vlm_floorplan.converters.cubicasa import convert_plan  # noqa: E402
from vlm_floorplan.record import PlanRecord  # noqa: E402


@pytest.fixture
def cubicasa_plan_dir(tmp_path: Path) -> Path:
    return write_cubicasa_plan(tmp_path / "plans" / "colorful" / "1")


@pytest.fixture
def cubicasa_record(cubicasa_plan_dir: Path) -> PlanRecord:
    return convert_plan(cubicasa_plan_dir, "cubicasa5k/colorful/1", split="train")

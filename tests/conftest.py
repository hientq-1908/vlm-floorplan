"""Shared fixtures.

Markers (declared in pyproject.toml):
    network  needs internet (dataset servers): skipped by default, run `pytest -m network`
    hf       needs the ml group and the HF processor: skips itself when unavailable;
             deselect explicitly with `pytest -m "not hf"`
    gpu      needs a 24 GB GPU and the full model weights: skipped by default, run `pytest -m gpu`
    slow     a few seconds or more (tiny-model training on the CPU)
"""

from __future__ import annotations

import os
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


@pytest.fixture(scope="session")
def qwen_tokenizer():  # noqa: ANN201 - transformers is optional (the ml group)
    """The Qwen3.5-4B tokenizer with its shipped template; skips without the ml group or Hub."""
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("trl")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    except OSError as err:  # offline and not cached
        pytest.skip(f"cannot load the Qwen3.5-4B tokenizer: {err}")


@pytest.fixture(autouse=True)
def no_real_clearml(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep every test away from the operator's ClearML workspace.

    Drops the `CLEARML_*` credentials and points ClearML at an empty config in offline mode, so
    nothing a test drives can read `~/clearml.conf` and create a task on the real server.
    """
    for name in list(os.environ):
        if name.startswith(("CLEARML_", "TRAINS_")):
            monkeypatch.delenv(name)
    config = tmp_path_factory.getbasetemp() / "clearml.conf"
    config.touch()
    monkeypatch.setenv("CLEARML_CONFIG_FILE", str(config))
    monkeypatch.setenv("CLEARML_OFFLINE_MODE", "1")

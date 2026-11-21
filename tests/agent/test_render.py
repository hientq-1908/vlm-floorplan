"""The rendered text view, through the real Qwen3.5 chat template.

Needs the `ml` group and the tokenizer files (a few MB, cached after the first download):
    uv run --group ml pytest -m hf
"""

from __future__ import annotations

from typing import Any

import pytest

from vlm_floorplan.agent import DeleteRoomTask, build_episode
from vlm_floorplan.agent.render import load_tokenizer, render_episode
from vlm_floorplan.agent.sample_plan import sample_record

pytestmark = pytest.mark.hf

pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tokenizer() -> Any:
    try:
        return load_tokenizer()
    except OSError as err:  # offline and not cached
        pytest.skip(f"cannot load the tokenizer: {err}")


def test_the_template_renders_tools_reasoning_calls_and_responses(tokenizer: Any) -> None:
    text = render_episode(build_episode(sample_record(), DeleteRoomTask("r1")), tokenizer)
    assert text.startswith("<|im_start|>system\n# Tools")
    assert '"name": "new_plan"' in text
    assert "<|im_start|>assistant\n<think>\nThe user wants a bathroom" in text
    assert "<tool_call>\n<function=add_room>\n<parameter=type>\nbathroom\n</parameter>" in text
    assert "<tool_response>\nOK: added bathroom room r6" in text
    assert text.endswith("<|im_end|>\n")

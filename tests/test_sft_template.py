"""The training template must render exactly what the model's shipped template renders.

SFT trains on text produced by TRL's training variant of the Qwen3.5 chat template (it adds the
`{% generation %}` markers that make the assistant-only loss possible). Serving uses the template
shipped with the model. If the two ever differ for a sample, the model is trained on a prompt
format it will never see.

Needs the ml group and the Qwen3.5-4B tokenizer files (`uv run --group ml pytest -m hf`).
"""

from __future__ import annotations

from typing import Any

import pytest
from sft_fixture import make_sample, make_two_call_sample

from vlm_floorplan.training.data import training_template

pytestmark = pytest.mark.hf


def render(tokenizer: Any, sample: dict[str, Any], template: str | None) -> str:
    kwargs = {"chat_template": template} if template else {}
    return tokenizer.apply_chat_template(
        sample["messages"], tools=sample["tools"], tokenize=False, **kwargs
    )


@pytest.mark.parametrize(
    "sample",
    [
        make_sample("one-call"),
        make_sample("question", tool_call=False),
        make_two_call_sample("two-calls"),
    ],
    ids=lambda s: s["id"],
)
def test_one_turn_sample_renders_identically_in_both_templates(
    qwen_tokenizer: Any, sample: dict[str, Any]
) -> None:
    shipped = render(qwen_tokenizer, sample, None)
    training = render(qwen_tokenizer, sample, training_template(qwen_tokenizer))
    assert training == shipped
    assert "<think>\n" in shipped and "<tool_call>" in render(qwen_tokenizer, make_sample(), None)


def test_templates_differ_once_a_second_user_turn_exists(qwen_tokenizer: Any) -> None:
    """Control: the shipped template drops the reasoning of turns before the last user message.

    The training template keeps it, which is why stage 1 uses one-turn conversations only and
    this test is limited to them. If this control ever fails, the comparison above stopped
    proving anything.
    """
    sample = make_sample()
    followup = {"role": "user", "content": "Thanks. Now add one more bedroom."}
    reply = {"role": "assistant", "reasoning_content": "Adding it.", "content": "Done."}
    sample["messages"] = [*sample["messages"], followup, reply]
    shipped = render(qwen_tokenizer, sample, None)
    training = render(qwen_tokenizer, sample, training_template(qwen_tokenizer))
    assert training != shipped


def test_training_template_is_the_one_trl_ships_for_qwen3_5_think(qwen_tokenizer: Any) -> None:
    from trl.chat_template_utils import qwen3_5_think_training_chat_template

    assert training_template(qwen_tokenizer) == qwen3_5_think_training_chat_template


def test_unknown_chat_template_is_refused() -> None:
    pytest.importorskip("trl")

    class Other:
        chat_template = "{{ messages }}"

    with pytest.raises(ValueError, match="training"):
        training_template(Other())

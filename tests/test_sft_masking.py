"""Only what the assistant writes is trained: reasoning, tool calls, replies, end of turn.

The system prompt, tool schemas, user request and tool responses are in the input but must have
label -100, or the model would be taught to imitate them.

Needs the ml group and the Qwen3.5-4B tokenizer files (`uv run --group ml pytest -m hf`).
"""

from __future__ import annotations

from typing import Any

import pytest
from sft_fixture import make_sample, make_two_call_sample

from vlm_floorplan.training.data import (
    IGNORE_INDEX,
    Sample,
    render_sample,
    training_template,
)

pytestmark = pytest.mark.hf


def as_sample(raw: dict[str, Any]) -> Sample:
    return Sample(raw["id"], raw["messages"], raw["tools"])


def trained_segments(tokenizer: Any, labels: list[int]) -> list[str]:
    """The text of each contiguous run of trained tokens."""
    segments: list[list[int]] = []
    previous_trained = False
    for label in labels:
        trained = label != IGNORE_INDEX
        if trained:
            if not previous_trained:
                segments.append([])
            segments[-1].append(label)
        previous_trained = trained
    return [tokenizer.decode(ids) for ids in segments]


def render(tokenizer: Any, raw: dict[str, Any]):  # noqa: ANN201
    return render_sample(tokenizer, as_sample(raw), training_template(tokenizer))


def test_trains_on_the_assistant_turns_and_nothing_else(qwen_tokenizer: Any) -> None:
    raw = make_sample(rooms=3)
    segments = trained_segments(qwen_tokenizer, render(qwen_tokenizer, raw).labels)

    assert len(segments) == 2  # the tool-call turn and the final reply
    call_turn, reply_turn = segments
    assert call_turn.startswith("Copy the 3 rooms into one new_plan call.\n</think>\n\n")
    assert "<tool_call>" in call_turn and "<function=new_plan>" in call_turn
    assert call_turn.endswith("<|im_end|>\n")
    assert reply_turn == "The tool accepted the plan.\n</think>\n\nDrew 3 bedrooms.<|im_end|>\n"


def test_prompt_parts_are_masked(qwen_tokenizer: Any) -> None:
    raw = make_sample()
    trained = "".join(trained_segments(qwen_tokenizer, render(qwen_tokenizer, raw).labels))
    assert raw["messages"][0]["content"] not in trained  # system prompt
    assert "Replace the plan with a new one." not in trained  # tool schemas
    assert "Request: Draw" not in trained  # user request
    assert "OK: replaced the plan." not in trained  # tool response
    assert "<tool_response>" not in trained


def test_every_call_turn_is_trained_but_not_its_response(qwen_tokenizer: Any) -> None:
    raw = make_two_call_sample()
    segments = trained_segments(qwen_tokenizer, render(qwen_tokenizer, raw).labels)
    assert len(segments) == 3
    assert any("<function=add_room>" in segment for segment in segments)
    assert "OK: added r2." not in "".join(segments)


def test_a_reply_without_tool_call_is_trained(qwen_tokenizer: Any) -> None:
    raw = make_sample(tool_call=False)
    (segment,) = trained_segments(qwen_tokenizer, render(qwen_tokenizer, raw).labels)
    assert segment.endswith("Do you want square metres or relative sizes?<|im_end|>\n")


def test_labels_equal_input_ids_where_trained(qwen_tokenizer: Any) -> None:
    item = render(qwen_tokenizer, make_sample())
    assert len(item.labels) == len(item.input_ids)
    assert all(
        label in (IGNORE_INDEX, token)
        for label, token in zip(item.labels, item.input_ids, strict=True)
    )
    assert item.labels[0] == IGNORE_INDEX  # the sequence starts with the masked system prompt


def test_same_tokens_and_mask_as_trls_own_vlm_path(qwen_tokenizer: Any) -> None:
    """TRL tokenizes through the processor for a vision-language model; the result must agree."""
    transformers = pytest.importorskip("transformers")
    from trl.data_utils import prepare_multimodal_messages

    try:
        processor = transformers.AutoProcessor.from_pretrained("Qwen/Qwen3.5-4B")
    except OSError as err:
        pytest.skip(f"cannot load the processor: {err}")
    raw = make_two_call_sample()
    out = processor.apply_chat_template(
        prepare_multimodal_messages(raw["messages"]),
        tools=raw["tools"],
        chat_template=training_template(qwen_tokenizer),
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    ours = render(qwen_tokenizer, raw)
    assert list(out["input_ids"][0]) == ours.input_ids
    expected = [
        t if m else IGNORE_INDEX
        for t, m in zip(out["input_ids"][0], out["assistant_masks"][0], strict=True)
    ]
    assert expected == ours.labels

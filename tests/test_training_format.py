"""CPU check of the training format: what the loss is computed on.

SFT with TRL builds each batch with `DataCollatorForVisionLanguageModeling`.
With `assistant_only_loss=True` it should train only on what the model has to
produce: its reasoning, the graph JSON, tool calls and replies.
Everything else (images, the user's text, tool responses) must be masked out
(label -100), or the model would be taught to imitate them.

The tests use the THINKING-ON chat template (project decision 4): the
generation prompt ends with `<think>\\n`, and every assistant turn carries its
reasoning in `reasoning_content`.

They need the `ml` dependency group and the Qwen3.5-4B processor files
(a few MB, downloaded once from the Hugging Face Hub), hence the `hf` marker:
    uv run --group ml pytest -m hf
Without the ml group they skip themselves; deselect explicitly with `pytest -m "not hf"`.
"""

from __future__ import annotations

from typing import Any

import pytest
from PIL import Image

from vlm_floorplan.compact import to_compact
from vlm_floorplan.schema import Floor, FloorplanGraph, ImageSize, Link, Room

pytestmark = pytest.mark.hf

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
trl_utils = pytest.importorskip("trl.chat_template_utils")
trl_sft = pytest.importorskip("trl.trainer.sft_trainer")

MODEL_ID = "Qwen/Qwen3.5-4B"
IGNORE_INDEX = -100
FLOORPLAN_REASONING = "The plan shows a kitchen next to a living room, joined by a doorway."
EDIT_REASONING = "The user wants the two rooms connected, so I add a door link between them."


@pytest.fixture(scope="module")
def processor() -> Any:
    try:
        return transformers.AutoProcessor.from_pretrained(MODEL_ID)
    except OSError as err:  # offline and not cached
        pytest.skip(f"cannot load the {MODEL_ID} processor: {err}")


@pytest.fixture(scope="module")
def collator(processor: Any) -> Any:
    template = trl_utils.get_training_chat_template(processor.tokenizer)
    assert template is not None, "TRL ships no training template for this model"
    return trl_sft.DataCollatorForVisionLanguageModeling(
        processor, assistant_only_loss=True, chat_template=template
    )


@pytest.fixture(scope="module")
def graph() -> FloorplanGraph:
    rooms = [
        Room(id="r0", type="kitchen", xy=(250, 300), label="K"),
        Room(id="r1", type="living", xy=(700, 300), label="OH", entrance=True),
        Room(id="r2", type="bedroom", xy=(500, 800)),
    ]
    links = [Link(a="r0", b="r1", kind="opening"), Link(a="r1", b="r2", kind="wall")]
    return FloorplanGraph(
        floors=[Floor(floor=1, rooms=rooms, links=links)], image=ImageSize(w=320, h=256)
    )


@pytest.fixture(scope="module")
def image() -> Image.Image:
    return Image.new("RGB", (320, 256), (240, 240, 240))


def perception_sample(image: Image.Image, graph: FloorplanGraph) -> dict[str, Any]:
    """Image in, graph JSON out."""
    return {
        "images": [image],
        "messages": [
            {"role": "user", "content": "Read this floorplan and output its room graph."},
            {
                "role": "assistant",
                "reasoning_content": FLOORPLAN_REASONING,
                "content": to_compact(graph),
            },
        ],
    }


def editing_sample(image: Image.Image, graph: FloorplanGraph) -> dict[str, Any]:
    """Current graph + request in; tool call, tool result (with a picture), reply out."""
    arguments = {"a": "r0", "b": "r1", "kind": "door"}
    return {
        "images": [image],
        "messages": [
            {
                "role": "user",
                "content": f"Graph: {to_compact(graph)}\nOpen the kitchen with a door.",
            },
            {
                "role": "assistant",
                "reasoning_content": EDIT_REASONING,
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": "set_link", "arguments": arguments}}
                ],
            },
            {
                "role": "tool",
                "name": "set_link",
                "content": [
                    {"type": "text", "text": "TOOLRESULT ok: r0-r1 is now a door"},
                    {"type": "image", "image": image},  # a tool may return a rendered graph
                ],
            },
            {
                "role": "assistant",
                "reasoning_content": "The tool succeeded.",
                "content": "Done: the kitchen now connects to the living room through a door.",
            },
        ],
    }


def _trained_text(batch: dict[str, Any], row: int, processor: Any) -> str:
    labels = batch["labels"][row]
    return processor.tokenizer.decode(labels[labels != IGNORE_INDEX])


def _image_pad_id(processor: Any) -> int:
    return processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")


def test_thinking_on_generation_prompt_opens_a_think_block(processor: Any) -> None:
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert prompt.endswith("<|im_start|>assistant\n<think>\n")


def test_perception_sample_trains_only_on_reasoning_and_graph(
    collator: Any, processor: Any, image: Image.Image, graph: FloorplanGraph
) -> None:
    batch = collator([perception_sample(image, graph)])

    trained = _trained_text(batch, 0, processor)
    # The prompt already ends with "<think>\n", so the trained span starts right after it.
    assert trained.startswith(FLOORPLAN_REASONING + "\n</think>\n\n")
    assert trained.endswith(to_compact(graph) + "<|im_end|>\n")
    assert "Read this floorplan" not in trained  # the user's prompt is masked
    image_pad = _image_pad_id(processor)
    assert (batch["input_ids"] == image_pad).sum() > 0, "the sample should contain image tokens"
    assert not (batch["labels"] == image_pad).any(), "image tokens must be masked out of the loss"


def test_editing_sample_masks_images_and_tool_response(
    collator: Any, processor: Any, image: Image.Image, graph: FloorplanGraph
) -> None:
    batch = collator([editing_sample(image, graph)])

    trained = _trained_text(batch, 0, processor)
    assert "<tool_call>" in trained and "<function=set_link>" in trained
    assert "Done: the kitchen now connects" in trained
    assert EDIT_REASONING in trained
    assert "TOOLRESULT" not in trained, "the tool's output is not something the model writes"
    assert "<tool_response>" not in trained
    assert "Open the kitchen" not in trained
    image_pad = _image_pad_id(processor)
    assert (batch["input_ids"] == image_pad).sum() > 0
    assert not (batch["labels"] == image_pad).any()


def test_editing_sample_contains_both_images(
    collator: Any, processor: Any, image: Image.Image, graph: FloorplanGraph
) -> None:
    """The user's image and the tool-returned image both reach the model."""
    batch = collator([editing_sample(image, graph)])
    assert batch["image_grid_thw"].shape[0] == 2
    expected = int(batch["image_grid_thw"].prod(dim=1).sum()) // 4  # 2x2 patches merge per token
    assert int((batch["input_ids"] == _image_pad_id(processor)).sum()) == expected


def test_without_assistant_only_loss_the_tool_response_would_be_trained(
    processor: Any, image: Image.Image, graph: FloorplanGraph
) -> None:
    """Control: proves the masking assertions above can fail."""
    unmasked = trl_sft.DataCollatorForVisionLanguageModeling(processor, assistant_only_loss=False)
    batch = unmasked([editing_sample(image, graph)])
    assert "TOOLRESULT" in _trained_text(batch, 0, processor)
    assert (batch["labels"] == _image_pad_id(processor)).any()

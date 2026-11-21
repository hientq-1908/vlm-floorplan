"""From JSONL samples to token ids and a loss mask.

One line of the training file is one conversation:

    {"id": "T0001", "messages": [...], "tools": [...]}

`messages` are chat messages (system, user, assistant, tool). An assistant message carries its
thinking in `reasoning_content` and its tool calls in `tool_calls`. `tools` is the tool schema
block the chat template writes into the prompt. Nothing here knows tool names or room types, so
the data can change without touching this code.

Why the samples are tokenized here and not inside TRL: the tool calls of different samples have
different argument names, and a Hugging Face `Dataset` would merge them into one record type
and fill the gaps with nulls. Rendering from the raw dicts avoids that, and the length filter
needs the exact token count anyway.

What the model is trained on (the assistant-only loss):

1. The chat template is TRL's Qwen3.5 *think training* template. It renders exactly the text of
   the template shipped with the model (`tests/test_sft_template.py` checks this), and marks
   every assistant turn with `{% generation %}` so the tokenizer can say which tokens the
   assistant wrote.
2. `labels` equal `input_ids` on assistant tokens and -100 (ignored by the loss) everywhere else.
   Trained: the reasoning, the tool call, the reply and the end-of-turn token. Masked: the
   system prompt, the tool schemas, the user request and the tool responses.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedTokenizerBase

log = logging.getLogger(__name__)

IGNORE_INDEX = -100
"""Label value the loss skips."""
ROLES = frozenset({"system", "user", "assistant", "tool"})


@dataclass(frozen=True)
class Sample:
    id: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None


@dataclass(frozen=True)
class Rendered:
    id: str
    input_ids: list[int]
    labels: list[int]

    @property
    def length(self) -> int:
        return len(self.input_ids)


@dataclass
class FilterReport:
    """What the length filter did. Written to the run directory as `data_report.json`."""

    max_length: int
    total: int = 0
    kept: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    """Sample id to its token length, for every dropped sample."""
    kept_tokens: int = 0
    trained_tokens: int = 0
    longest_kept: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "max_length": self.max_length,
            "total": self.total,
            "kept": self.kept,
            "dropped_count": len(self.dropped),
            "dropped": self.dropped,
            "kept_tokens": self.kept_tokens,
            "trained_tokens": self.trained_tokens,
            "longest_kept": self.longest_kept,
        }


def _check_sample(record: Any, where: str) -> Sample:
    if not isinstance(record, dict):
        raise ValueError(f"{where}: expected a JSON object, got {type(record).__name__}")
    sample_id = record.get("id")
    messages = record.get("messages")
    tools = record.get("tools")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"{where}: 'id' must be a non-empty string")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{where} ({sample_id}): 'messages' must be a non-empty list")
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in ROLES:
            raise ValueError(f"{where} ({sample_id}): messages[{index}] has no valid 'role'")
    if messages[-1]["role"] != "assistant":
        raise ValueError(f"{where} ({sample_id}): the last message must be from the assistant")
    if tools is not None and not isinstance(tools, list):
        raise ValueError(f"{where} ({sample_id}): 'tools' must be a list when present")
    return Sample(id=sample_id, messages=messages, tools=tools or None)


def read_samples(path: Path) -> list[Sample]:
    """Read and validate a training JSONL file. Duplicate ids are an error."""
    samples: list[Sample] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            where = f"{path}:{number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"{where}: not valid JSON: {err}") from err
            sample = _check_sample(record, where)
            if sample.id in seen:
                raise ValueError(f"{where}: duplicate sample id {sample.id!r}")
            seen.add(sample.id)
            samples.append(sample)
    if not samples:
        raise ValueError(f"{path} holds no samples")
    return samples


def training_template(tokenizer: PreTrainedTokenizerBase) -> str:
    """TRL's training variant of the tokenizer's own Qwen3.5 template (thinking on)."""
    from trl.chat_template_utils import (
        get_training_chat_template,
        qwen3_5_think_training_chat_template,
    )

    template = get_training_chat_template(tokenizer, require_prefix_preservation=False)
    if template != qwen3_5_think_training_chat_template:
        raise ValueError(
            "the tokenizer's chat template is not the Qwen3.5 think template that TRL knows, so "
            "there is no training template with generation markers for it"
        )
    return template


def render_sample(tokenizer: PreTrainedTokenizerBase, sample: Sample, template: str) -> Rendered:
    """Token ids and labels of one sample under the training template."""
    out = tokenizer.apply_chat_template(
        sample.messages,
        tools=sample.tools,
        chat_template=template,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    input_ids: list[int] = list(out["input_ids"])
    mask: list[int] = list(out["assistant_masks"])
    if not any(mask):
        raise ValueError(f"sample {sample.id}: no assistant tokens, nothing to train on")
    labels = [
        token if is_assistant else IGNORE_INDEX
        for token, is_assistant in zip(input_ids, mask, strict=True)
    ]
    return Rendered(id=sample.id, input_ids=input_ids, labels=labels)


def filter_by_length(
    rendered: Iterable[Rendered], max_length: int
) -> tuple[list[Rendered], FilterReport]:
    """Keep samples of at most `max_length` tokens. Longer ones are dropped, never truncated.

    A truncated sample would lose its end: the tool call or the reply, the very thing the model
    must learn to write. The dropped ids are logged so the loss of data stays visible.
    """
    report = FilterReport(max_length=max_length)
    kept: list[Rendered] = []
    for item in rendered:
        report.total += 1
        if item.length > max_length:
            report.dropped[item.id] = item.length
            continue
        kept.append(item)
        report.kept_tokens += item.length
        report.trained_tokens += sum(label != IGNORE_INDEX for label in item.labels)
        report.longest_kept = max(report.longest_kept, item.length)
    report.kept = len(kept)
    if report.dropped:
        log.warning(
            "dropped %d of %d samples longer than %d tokens: %s",
            len(report.dropped),
            report.total,
            max_length,
            ", ".join(f"{sid} ({n})" for sid, n in sorted(report.dropped.items())),
        )
    else:
        log.info("kept all %d samples (none longer than %d tokens)", report.total, max_length)
    return kept, report


def load_training_data(
    path: Path,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[Rendered], FilterReport]:
    """Read, render and length-filter a training file."""
    template = training_template(tokenizer)
    samples = read_samples(path)
    rendered = []
    for index, sample in enumerate(samples, start=1):
        rendered.append(render_sample(tokenizer, sample, template))
        if on_progress:
            on_progress(index, len(samples))
    kept, report = filter_by_length(rendered, max_length)
    if not kept:
        raise ValueError(f"every sample in {path} is longer than {max_length} tokens")
    return kept, report


def to_dataset(items: list[Rendered]) -> Dataset:
    """The columns TRL's `SFTTrainer` accepts for a pre-tokenized dataset."""
    from datasets import Dataset

    return Dataset.from_dict(
        {"input_ids": [i.input_ids for i in items], "labels": [i.labels for i in items]}
    )

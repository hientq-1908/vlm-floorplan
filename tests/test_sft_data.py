"""Reading the training file and the length filter (no tokenizer needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sft_fixture import make_sample, write_samples

from vlm_floorplan.training.data import (
    IGNORE_INDEX,
    Rendered,
    filter_by_length,
    read_samples,
)


def rendered(sample_id: str, length: int, trained: int = 3) -> Rendered:
    labels = [IGNORE_INDEX] * (length - trained) + list(range(trained))
    return Rendered(sample_id, list(range(length)), labels)


def test_read_samples_returns_every_line(tmp_path: Path) -> None:
    path = write_samples(
        tmp_path / "t.jsonl", [make_sample("A"), make_sample("B", tool_call=False)]
    )
    samples = read_samples(path)
    assert [s.id for s in samples] == ["A", "B"]
    assert samples[0].tools and samples[0].messages[-1]["role"] == "assistant"


def test_read_samples_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    path.write_text("\n" + json.dumps(make_sample("A")) + "\n\n")
    assert len(read_samples(path)) == 1


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda s: s.pop("id"), "'id'"),
        (lambda s: s.update(messages=[]), "'messages'"),
        (lambda s: s["messages"].pop(), "last message"),
        (lambda s: s["messages"][1].update(role="robot"), "'role'"),
        (lambda s: s.update(tools="nope"), "'tools'"),
        (
            lambda s: s["messages"].extend(
                [
                    {"role": "user", "content": "Now add one more bedroom."},
                    {"role": "assistant", "reasoning_content": "Adding.", "content": "Done."},
                ]
            ),
            "exactly one user message, got 2",
        ),
        (
            lambda s: s.update(messages=[m for m in s["messages"] if m["role"] != "user"]),
            "exactly one user message, got 0",
        ),
    ],
)
def test_malformed_sample_is_rejected_with_its_position(tmp_path: Path, mutate, message) -> None:
    sample = make_sample("A")
    mutate(sample)
    path = write_samples(tmp_path / "t.jsonl", [sample])
    with pytest.raises(ValueError, match=message) as err:
        read_samples(path)
    assert "t.jsonl:1" in str(err.value)


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    path = write_samples(tmp_path / "t.jsonl", [make_sample("A"), make_sample("A")])
    with pytest.raises(ValueError, match="duplicate"):
        read_samples(path)


def test_invalid_json_and_empty_files_are_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n")
    with pytest.raises(ValueError, match="not valid JSON"):
        read_samples(bad)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    with pytest.raises(ValueError, match="no samples"):
        read_samples(empty)


def test_filter_drops_longer_samples_and_keeps_the_limit_itself() -> None:
    items = [rendered("short", 10), rendered("exact", 20), rendered("long", 21)]
    kept, report = filter_by_length(items, max_length=20)
    assert [i.id for i in kept] == ["short", "exact"]
    assert report.dropped == {"long": 21}
    assert (report.total, report.kept) == (3, 2)


def test_filter_never_truncates() -> None:
    items = [rendered("a", 50), rendered("b", 100)]
    kept, _ = filter_by_length(items, max_length=60)
    assert [len(i.input_ids) for i in kept] == [50]
    assert all(len(i.input_ids) == len(i.labels) for i in kept)


def test_filter_logs_the_dropped_count(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        filter_by_length([rendered("a", 5), rendered("b", 50), rendered("c", 60)], max_length=10)
    assert "dropped 2 of 3 samples longer than 10 tokens" in caplog.text


def test_filter_report_counts_tokens() -> None:
    _, report = filter_by_length([rendered("a", 10, trained=4), rendered("b", 20, trained=5)], 20)
    assert (report.kept_tokens, report.trained_tokens, report.longest_kept) == (30, 9, 20)
    assert report.to_json()["dropped_count"] == 0

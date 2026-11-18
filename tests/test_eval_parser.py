from __future__ import annotations

import json

import pytest
from helpers_eval import IMAGE, perfect_prediction

from vlm_floorplan.compact import to_compact
from vlm_floorplan.evaluation.parser import ParseError, Source, parse_model_output, strip_think

GRAPH = perfect_prediction()
COMPACT = to_compact(GRAPH)


def test_canonical_compact_json_parses_strictly() -> None:
    result = parse_model_output(COMPACT, IMAGE)
    assert result.graph == GRAPH
    assert (result.source, result.lenient) == (Source.CONTENT, False)


def test_think_block_markdown_fence_and_prose_are_tolerated() -> None:
    text = (
        f"<think>The plan has {{four}} rooms.</think>\nHere you go:\n```json\n{COMPACT}\n```\nDone."
    )
    result = parse_model_output(text, IMAGE)
    assert result.graph == GRAPH
    assert not result.lenient


def test_missing_opening_think_tag_is_handled() -> None:
    # Qwen3.5's template prefills "<think>", so the output starts mid-thought.
    text = f"Let me look at the image {{carefully}}...\n</think>\n\n{COMPACT}"
    assert parse_model_output(text, IMAGE).graph == GRAPH


def test_strip_think_keeps_only_text_after_the_last_closing_tag() -> None:
    assert strip_think("a</think>b</think>c") == "c"
    assert strip_think("<think>x</think>y") == "y"
    assert strip_think("plain") == "plain"


def test_answer_that_landed_in_reasoning_is_found_there() -> None:
    result = parse_model_output("", IMAGE, reasoning=f"thinking...\n{COMPACT}")
    assert result.graph == GRAPH
    assert result.source == Source.REASONING


def test_tool_call_text_in_reasoning_is_parsed() -> None:
    reasoning = (
        "I will answer now.\n<tool_call>\n<function=emit_graph>\n"
        f"<parameter=graph>\n{COMPACT}\n</parameter>\n</function>\n</tool_call>"
    )
    result = parse_model_output(None, IMAGE, reasoning=reasoning)
    assert result.graph == GRAPH
    assert result.source in (Source.REASONING, Source.TOOL_CALL)


def test_answer_text_wins_over_reasoning() -> None:
    other = json.dumps({"floors": [{"floor": 1, "rooms": [], "links": []}]})
    result = parse_model_output(COMPACT, IMAGE, reasoning=other)
    assert result.source == Source.CONTENT


def test_last_graph_wins_when_the_answer_repeats_itself() -> None:
    draft = json.dumps({"floors": [{"floor": 1, "rooms": [], "links": []}]})
    result = parse_model_output(f"draft: {draft}\nfinal: {COMPACT}", IMAGE)
    assert result.graph == GRAPH


def test_single_floor_without_wrapper_is_accepted_as_lenient() -> None:
    floor = json.loads(COMPACT)["floors"][0]
    del floor["floor"]
    result = parse_model_output(json.dumps(floor), IMAGE)
    assert result.ok
    assert result.lenient
    assert result.graph is not None
    assert result.graph.floors[0].floor == 1


def test_links_as_objects_are_accepted_as_lenient() -> None:
    data = json.loads(COMPACT)
    data["floors"][0]["links"] = [
        {"a": a, "b": b, "kind": kind} for a, b, kind in data["floors"][0]["links"]
    ]
    result = parse_model_output(json.dumps(data), IMAGE)
    assert result.graph == GRAPH
    assert result.lenient


@pytest.mark.parametrize(
    ("content", "reasoning", "expected"),
    [
        ("", "", ParseError.EMPTY),
        (None, None, ParseError.EMPTY),
        ("I cannot see an image.", "", ParseError.NO_JSON),
        (COMPACT[:80], "", ParseError.TRUNCATED),
        ('{"floors": [oops]}', "", ParseError.INVALID_JSON),
        ('{"hello": "world"}', "", ParseError.WRONG_SHAPE),
        (
            '{"floors":[{"floor":1,"rooms":[{"id":"r0","type":"sofa","xy":[1,1]}]}]}',
            "",
            ParseError.SCHEMA_INVALID,
        ),
        (
            '{"floors":[{"floor":1,"rooms":[{"id":"r0","type":"living","xy":[1,1]}],'
            '"links":[["r0","r5","door"]]}]}',
            "",
            ParseError.SCHEMA_INVALID,
        ),
        (
            '{"floors":[{"floor":1,"rooms":[{"id":"r0","type":"living","xy":[1,2000]}]}]}',
            "",
            ParseError.SCHEMA_INVALID,
        ),
    ],
)
def test_error_categories(content: str | None, reasoning: str | None, expected: ParseError) -> None:
    result = parse_model_output(content, IMAGE, reasoning)
    assert result.graph is None
    assert result.error == expected
    assert result.detail or expected == ParseError.EMPTY


def test_truncated_reasoning_with_no_answer_is_truncated_not_no_json() -> None:
    result = parse_model_output("", IMAGE, reasoning="thinking " + COMPACT[:100])
    assert result.error == ParseError.TRUNCATED


def test_parser_never_raises_on_garbage() -> None:
    for text in ["}{", "{" * 50, '{"a":', "\x00\x01", "```json\n```", "<think>", "[[[[", "{}"]:
        assert not parse_model_output(text, IMAGE).ok

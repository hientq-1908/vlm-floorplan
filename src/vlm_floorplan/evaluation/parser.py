"""Turn model text into a `FloorplanGraph`, and say precisely why it failed when it fails.

A zero-shot 4B model with thinking on produces messy text:

- a reasoning block, possibly with the opening `<think>` missing (Qwen3.5's chat
  template pre-fills it, so the output starts mid-thought and only has `</think>`),
- prose and markdown fences around the JSON,
- JSON in a looser shape than asked for (links as objects, one floor without the
  `floors` wrapper, floors without numbers),
- when the server's reasoning parser never sees a `</think>`, the *entire* answer in
  the reasoning field and an empty content field,
- a tool call as XML text inside the reasoning (TRL issue #8050).

`parse_model_output` tries the answer text first, then the reasoning text, and returns
either a validated graph or one error category. It never raises on bad model output.

Leniency is recorded, not hidden: `ParseResult.lenient` is True when the output needed
a tolerance rule beyond the canonical compact form. The evaluation reports both the
lenient and the strict validity rate, so "how often does the model follow the format"
stays visible.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import ValidationError

from vlm_floorplan.compact import from_compact
from vlm_floorplan.schema import FloorplanGraph, ImageSize

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
TOOL_PARAMETER = re.compile(r"<parameter=[^>]*>\s*(.*?)\s*</parameter>", re.DOTALL)


class ParseError(StrEnum):
    """Why an output is not a valid graph. One category per failed output."""

    EMPTY = "empty"
    """No text at all (the model spent its whole budget thinking, or the call failed)."""
    NO_JSON = "no_json"
    """Text, but no JSON object in it."""
    TRUNCATED = "truncated"
    """A JSON object that starts but never closes (hit max_tokens)."""
    INVALID_JSON = "invalid_json"
    """Braces balance but the text is not JSON."""
    WRONG_SHAPE = "wrong_shape"
    """Valid JSON, but not a floorplan (no "floors" or "rooms" list)."""
    SCHEMA_INVALID = "schema_invalid"
    """A floorplan-shaped object that breaks the schema (bad type, dangling link, ...)."""


class Source(StrEnum):
    CONTENT = "content"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"


@dataclass(frozen=True)
class ParseResult:
    graph: FloorplanGraph | None
    error: ParseError | None = None
    detail: str = ""
    source: Source | None = None
    """Where the graph was found. `reasoning` and `tool_call` mean the answer text had none."""
    lenient: bool = False
    """True if a tolerance rule (not the canonical compact form) was needed."""

    @property
    def ok(self) -> bool:
        return self.graph is not None


def strip_think(text: str) -> str:
    """Remove reasoning from answer text.

    Handles a full `<think>...</think>` block and the Qwen3.5 case where only the
    closing tag is present because the opening one was part of the prompt.
    """
    text = THINK_BLOCK.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.replace("<think>", "").strip()


def _json_candidates(text: str) -> tuple[list[Any], list[str]]:
    """Every top-level JSON object found in `text`, plus problems met on the way.

    Scans for each `{` and tries `raw_decode`, skipping past what it consumed, so a
    leading sentence or several fenced blocks are fine.
    """
    decoder = json.JSONDecoder()
    found: list[Any] = []
    problems: list[str] = []
    pos = 0
    while (start := text.find("{", pos)) != -1:
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError as err:
            problem = _classify_json_failure(text[start:], err)
            problems.append(problem)
            if problem.startswith(ParseError.TRUNCATED):
                break  # the rest of the text is inside this unfinished object
            pos = start + 1
            continue
        found.append(obj)
        pos = end
    return found, problems


def _classify_json_failure(fragment: str, err: json.JSONDecodeError) -> str:
    """ "truncated" when the object runs to the end of the text still open, else "invalid"."""
    depth, in_string, escaped = 0, False, False
    for char in fragment:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
    if depth > 0 or in_string:
        return f"{ParseError.TRUNCATED}: {err.msg}"
    return f"{ParseError.INVALID_JSON}: {err.msg}"


def _to_graph(obj: Any, image: ImageSize) -> tuple[FloorplanGraph, bool]:
    """Validate one decoded object; returns (graph, needed_leniency). Raises ValueError."""
    if not isinstance(obj, dict):
        raise _WrongShapeError("top-level JSON is not an object")
    lenient = False
    if "floors" not in obj:
        if "rooms" not in obj:
            raise _WrongShapeError('object has neither "floors" nor "rooms"')
        # A single floor without the wrapper, as in the roadmap report's example.
        obj = {"floors": [{"floor": 1, **obj}]}
        lenient = True
    floors = obj["floors"]
    if isinstance(floors, list) and any(isinstance(f, dict) and "floor" not in f for f in floors):
        # Floors without a number: number them in order, as a person would.
        obj = {
            **obj,
            "floors": [
                {"floor": i, **f} if isinstance(f, dict) and "floor" not in f else f
                for i, f in enumerate(floors, start=1)
            ],
        }
        lenient = True
    try:
        return from_compact(json.dumps(obj), image), lenient
    except ValueError as compact_err:
        # Links as {"a":..,"b":..,"kind":..} objects instead of triples: the verbose form.
        try:
            return FloorplanGraph.model_validate({**obj, "image": image.model_dump()}), True
        except ValidationError:
            raise compact_err from None


class _WrongShapeError(ValueError):
    pass


def _try_text(text: str, image: ImageSize, source: Source) -> ParseResult | list[str]:
    """A ParseResult on success, else the failure reasons found in this text."""
    # Markdown fences need no special handling: the scan finds the `{` inside them.
    candidates, problems = _json_candidates(text)
    errors: list[str] = []
    # The last candidate is usually the final answer; earlier ones are drafts.
    for obj in reversed(candidates):
        try:
            graph, lenient = _to_graph(obj, image)
        except _WrongShapeError as err:
            errors.append(f"{ParseError.WRONG_SHAPE}: {err}")
        except ValueError as err:
            errors.append(f"{ParseError.SCHEMA_INVALID}: {_short(err)}")
        else:
            return ParseResult(graph=graph, source=source, lenient=lenient)
    return errors + problems


def _short(err: Exception, limit: int = 300) -> str:
    text = " ".join(str(err).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _tool_call_payloads(text: str) -> list[str]:
    """JSON-looking bodies of Qwen XML tool-call parameters (`<parameter=graph>{...}`)."""
    return [body for body in TOOL_PARAMETER.findall(text) if "{" in body]


def parse_model_output(
    content: str | None,
    image: ImageSize,
    reasoning: str | None = None,
) -> ParseResult:
    """Parse the answer (and, as a fallback, the reasoning) into a graph.

    Order of attempts, first success wins:
    1. the answer text with reasoning stripped,
    2. tool-call parameters inside the answer text,
    3. the reasoning text (answer landed there), then tool-call parameters in it.
    """
    answer = strip_think(content or "")
    thoughts = reasoning or ""
    if not answer and not thoughts.strip():
        return ParseResult(graph=None, error=ParseError.EMPTY, detail="no content and no reasoning")

    failures: list[str] = []
    attempts: list[tuple[str, Source]] = [(answer, Source.CONTENT)]
    attempts += [(p, Source.TOOL_CALL) for p in _tool_call_payloads(answer)]
    attempts += [(strip_think(thoughts), Source.REASONING)]
    attempts += [(p, Source.TOOL_CALL) for p in _tool_call_payloads(thoughts)]
    for text, source in attempts:
        if not text:
            continue
        outcome = _try_text(text, image, source)
        if isinstance(outcome, ParseResult):
            return outcome
        failures += outcome
    return _failure(failures, answer)


def _failure(failures: list[str], answer: str) -> ParseResult:
    """Pick the most informative failure; answer-text failures come first in `failures`."""
    if not failures:
        return ParseResult(graph=None, error=ParseError.NO_JSON, detail=_short(answer or "no JSON"))
    # Prefer structural reasons over "truncated": a complete-but-wrong object is a
    # different bug from running out of tokens.
    order = [
        ParseError.SCHEMA_INVALID,
        ParseError.TRUNCATED,
        ParseError.INVALID_JSON,
        ParseError.WRONG_SHAPE,
    ]
    for kind in order:
        for failure in failures:
            if failure.startswith(f"{kind}:"):
                return ParseResult(graph=None, error=kind, detail=failure.split(": ", 1)[1])
    return ParseResult(graph=None, error=ParseError.NO_JSON, detail=failures[0])

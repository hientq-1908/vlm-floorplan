"""Paraphrase a template prompt with a language model, and only keep one that means the same.

Template text is exact but repetitive; a paraphrase is natural and varied. The two checks that
decide whether a paraphrase may stand in for its template are the same ones the training-data
generator uses:

1. the lexical slot check (`words.slot_check`): every room type and number of 2 or more in the
   requirements appears, and no other room type or number does;
2. a blind round trip: a second call sees only the paraphrase and extracts the requirements as
   JSON, and they must equal the intended set exactly.

A paraphrase that fails either is retried a few times and then dropped; the set builder
replaces the prompt, so the paraphrased test set holds only prompts whose requirements are
known to be exact. The endpoint is any OpenAI-compatible one; for DeepSeek the key and URL come
from `DEEPSEEK_API_KEY` and `DEEPSEEK_BASE_URL` and are never printed, logged or stored.
"""

from __future__ import annotations

import json
import os
import random
import re
import urllib.request
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import ValidationError

from vlm_floorplan.evaluation.client import ApiError, ChatClient, HttpChatClient
from vlm_floorplan.requirements import (
    Adjacent,
    Requirement,
    requirements_from_json,
    requirements_to_json,
)
from vlm_floorplan.schema import RoomType
from vlm_floorplan.text2graph.templates import Describer
from vlm_floorplan.text2graph.words import NOUNS, Style, slot_check

API_KEY_ENV = "DEEPSEEK_API_KEY"
BASE_URL_ENV = "DEEPSEEK_BASE_URL"
MODEL_ENV = "DEEPSEEK_MODEL"
PARAPHRASE_TEMPERATURE = 0.9
PERSONA_SHARE = 0.1
PERSONAS = (
    "a couple who both work from home",
    "a family with a small child",
    "someone who has just retired",
    "a student sharing with a friend",
    "a person who loves hosting dinners",
)
_PARAPHRASE_PROMPT = """You rewrite a floor-plan request so it sounds like a real person wrote it.

Rules:
- State every requirement listed below, with exactly the same numbers and the same meaning.
- Do not add any other requirement, room type or number (no square metres, floors or directions).
- Keep any quoted name exactly as written, in quotes.
- Write one to three sentences. Reply with the rewritten request only.
{persona}
Requirements:
{requirements}

Original request:
{original}"""
_PERSONA_LINE = (
    "- Write it as {persona}: add one short sentence of personal context that mentions "
    "no room and no number.\n"
)
_EXTRACT_PROMPT = """Extract the requirements of a floor-plan request as a JSON list.
Reply with only the JSON list. Each item is one of these objects (T is a room type:
{types}):

{{"kind":"room_count","type":T,"op":"eq"|"ge"|"le","n":N}}
  "3 bedrooms" is eq 3, "at least 2 bathrooms" is ge 2, "at most 1" is le 1, "no garage" is eq 0.
{{"kind":"adjacent","a":T,"b":T,"link":"door"|"opening"|"connected"|"touching"}}
  "open to" or "open kitchen" is opening, "a door between" is door, "connected to", "off", "next
  to with access" is connected, "shares a wall with" is touching.
{{"kind":"ensuites","op":"eq"|"ge"|"le","n":N}}   N ensuite (en-suite, attached) bathrooms.
{{"kind":"entrances","op":"eq"|"ge"|"le","n":N}}  N front doors or entrances.
{{"kind":"entrance_into","type":T}}               the front door opens into a room of type T.
{{"kind":"in_region","type":T,"region":R}}
  R is one of top, bottom, left, right, top_left, top_right, bottom_left, bottom_right.
{{"kind":"relative_position","a":T,"b":T,"relation":"left_of"|"above"}}
  a is the room that is left of (or above) b: "the kitchen right of the hall" is a=entry, b=kitchen,
  left_of.
{{"kind":"reachable_avoiding","type":T,"avoid":T}}
  a room of type T you can reach without walking through a room of type `avoid`.
{{"kind":"has_label","type":T,"text":"<the quoted name>"}}

Request:
{text}"""
_JSON_LIST = re.compile(r"\[.*\]", re.DOTALL)


@dataclass(frozen=True)
class ParaphraseResult:
    text: str | None
    """The accepted paraphrase, or None when every attempt failed a check."""
    persona: bool = False
    attempts: int = 0
    reasons: list[str] = field(default_factory=list)
    """Why each failed attempt failed, in order."""


def plain_requirements(requirements: Sequence[Requirement]) -> list[str]:
    """The requirements as short fixed phrases, the form the paraphrase prompt lists."""
    style = Style(nouns={t: options[0] for t, options in NOUNS.items()}, use_words=False)
    describer = Describer(style, Counter(), random.Random(0))
    return [describer.describe(req) for req in requirements]


def paraphrase_prompt(
    original: str, requirements: Sequence[Requirement], persona: str | None
) -> str:
    phrases = "\n".join(f"- {phrase}" for phrase in plain_requirements(requirements))
    line = _PERSONA_LINE.format(persona=persona) if persona else ""
    return _PARAPHRASE_PROMPT.format(persona=line, requirements=phrases, original=original)


def extraction_prompt(text: str) -> str:
    types = ", ".join(t.value for t in RoomType if t != RoomType.UNKNOWN)
    return _EXTRACT_PROMPT.format(types=types, text=text)


def _normalize(requirements: Sequence[Requirement]) -> set[str]:
    """A set of canonical strings, so order and the a/b order of symmetric pairs do not matter."""
    items = []
    for req in requirements:
        if isinstance(req, Adjacent) and req.a.value > req.b.value:
            req = req.model_copy(update={"a": req.b, "b": req.a})
        items.append(json.dumps(req.model_dump(mode="json"), sort_keys=True))
    return set(items)


def parse_extraction(text: str) -> list[Requirement] | None:
    """The requirement list in a model reply, or None when it is not one."""
    match = _JSON_LIST.search(text)
    if match is None:
        return None
    try:
        return requirements_from_json(json.loads(match.group(0)))
    except (json.JSONDecodeError, ValidationError, TypeError):
        return None


class Teacher:
    """A chat endpoint used to paraphrase and to extract requirements blind."""

    def __init__(self, client: ChatClient, model: str) -> None:
        self.client, self.model = client, model

    def _ask(self, prompt: str, temperature: float, thinking: bool) -> str:
        request = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 4096,
            "chat_template_kwargs": {"thinking": thinking},
        }
        return self.client.complete(request).content.strip()

    def paraphrase(
        self, original: str, requirements: Sequence[Requirement], persona: str | None
    ) -> str:
        reply = self._ask(
            paraphrase_prompt(original, requirements, persona), PARAPHRASE_TEMPERATURE, False
        )
        return reply.strip().strip('"').strip()

    def extract(self, text: str) -> list[Requirement] | None:
        return parse_extraction(self._ask(extraction_prompt(text), 0.0, True))

    def checked_paraphrase(
        self,
        original: str,
        requirements: Sequence[Requirement],
        rng: random.Random,
        attempts: int = 3,
    ) -> ParaphraseResult:
        """A paraphrase that passes both checks, or a result with `text` None."""
        persona = rng.choice(PERSONAS) if rng.random() < PERSONA_SHARE else None
        reasons: list[str] = []
        for attempt in range(1, attempts + 1):
            try:
                candidate = self.paraphrase(original, requirements, persona)
                problems = slot_check(candidate, requirements) if candidate else ["empty reply"]
                if not problems:
                    found = self.extract(candidate)
                    if found is None:
                        problems = ["the extraction was not a requirement list"]
                    elif _normalize(found) != _normalize(requirements):
                        problems = ["the blind extraction differs from the requirements"]
            except ApiError as err:
                problems = [f"api error: {str(err)[:80]}"]
                candidate = ""
            if not problems:
                return ParaphraseResult(candidate, persona is not None, attempt, reasons)
            reasons.append("; ".join(problems))
        return ParaphraseResult(None, persona is not None, attempts, reasons)


def requirements_match(found: Sequence[Requirement] | None, wanted: Sequence[Requirement]) -> bool:
    return found is not None and _normalize(found) == _normalize(wanted)


def teacher_from_env() -> Teacher:
    """A `Teacher` for the DeepSeek endpoint named in the environment.

    The model is `DEEPSEEK_MODEL` if set, else the first listed model whose id mentions
    "deepseek" (preferring a "flash" one). Raises `RuntimeError` when the key or URL is missing.
    """
    base_url, key = os.environ.get(BASE_URL_ENV), os.environ.get(API_KEY_ENV)
    if not base_url or not key:
        raise RuntimeError(f"set {BASE_URL_ENV} and {API_KEY_ENV} to paraphrase prompts")
    client = HttpChatClient(base_url, api_key_env=API_KEY_ENV, timeout_s=300.0)
    model = os.environ.get(MODEL_ENV) or _pick_model(base_url, key)
    return Teacher(client, model)


def _pick_model(base_url: str, key: str) -> str:
    url = base_url.rstrip("/") + "/models"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=60) as reply:
            ids = [item["id"] for item in json.loads(reply.read()).get("data", [])]
    except (OSError, ValueError, KeyError) as err:
        raise RuntimeError(f"could not list the teacher's models: {type(err).__name__}") from err
    candidates = [i for i in ids if "deepseek" in i.lower()] or ids
    if not candidates:
        raise RuntimeError("the teacher endpoint lists no models")
    return next((i for i in candidates if "flash" in i.lower()), candidates[0])


def requirements_json(requirements: Sequence[Requirement]) -> list[dict]:
    return requirements_to_json(list(requirements))

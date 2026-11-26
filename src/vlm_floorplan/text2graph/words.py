"""The words a prompt uses for rooms and numbers, and the lexical slot check.

One request uses one noun per room type and one number style, so "lounge" and "living room"
never name two different rooms in the same text (`Style`). The slot check is the first of the
two checks a paraphrase must pass: every room type and every number of 2 or more that the
requirements state appears in the text, and no other room type or number does.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable
from dataclasses import dataclass

from vlm_floorplan.requirements import (
    Adjacent,
    Ensuites,
    EntranceInto,
    Entrances,
    HasLabel,
    InRegion,
    ReachableAvoiding,
    RelativePosition,
    Requirement,
    RoomCount,
)
from vlm_floorplan.schema import RoomType

NOUNS: dict[RoomType, list[tuple[str, str]]] = {
    RoomType.LIVING: [("living room", "living rooms"), ("lounge", "lounges")],
    RoomType.KITCHEN: [("kitchen", "kitchens")],
    RoomType.DINING: [("dining room", "dining rooms")],
    RoomType.BEDROOM: [("bedroom", "bedrooms")],
    RoomType.BATHROOM: [("bathroom", "bathrooms"), ("shower room", "shower rooms")],
    RoomType.ENTRY: [("hall", "halls"), ("entry hall", "entry halls")],
    RoomType.STORAGE: [("storage room", "storage rooms"), ("storeroom", "storerooms")],
    RoomType.UTILITY: [("utility room", "utility rooms"), ("laundry room", "laundry rooms")],
    RoomType.TERRACE: [("balcony", "balconies"), ("terrace", "terraces")],
    RoomType.GARAGE: [("garage", "garages")],
    RoomType.STAIRS: [("staircase", "staircases")],
    RoomType.OTHER: [("room", "rooms")],
}
NUMBER_WORDS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
]  # fmt: skip

# Words that name a room type in any text, a paraphrase included.
TYPE_STEMS: dict[RoomType, re.Pattern[str]] = {
    RoomType.LIVING: re.compile(r"\b(living|lounge|sitting room|family room)", re.I),
    RoomType.KITCHEN: re.compile(r"\bkitchen", re.I),
    RoomType.DINING: re.compile(r"\bdining", re.I),
    RoomType.BEDROOM: re.compile(r"\bbed(room)?s?\b|\bmaster\b", re.I),
    RoomType.BATHROOM: re.compile(r"\bbath|\bshower|\bwc\b|\btoilet|\bwashroom|\blavator", re.I),
    RoomType.ENTRY: re.compile(r"\bhall|\bentry|\bentryway|\bfoyer|\bvestibule|\blobby", re.I),
    RoomType.STORAGE: re.compile(r"\bstorage|\bstoreroom|\bstore room|\bcloset|\bpantry", re.I),
    RoomType.UTILITY: re.compile(r"\butility|\blaundry", re.I),
    RoomType.TERRACE: re.compile(r"\bbalcon|\bterrace|\bpatio|\bveranda|\bporch", re.I),
    RoomType.GARAGE: re.compile(r"\bgarage|\bcarport", re.I),
    RoomType.STAIRS: re.compile(r"\bstair", re.I),
}
_QUOTED = re.compile(r"""(?<![A-Za-z])'[^']+'(?![A-Za-z])|"[^"]+"|“[^”]+”|‘[^’]+’""")
_NUMBER_TOKEN = re.compile(r"\b(\d+)\b|\b(" + "|".join(NUMBER_WORDS[2:]) + r")\b", re.I)
_ORDINAL_OR_SIZE = re.compile(r"\b\d+\s*(?:m2|m²|sq|square|sqm)\b|\b\d+(?:st|nd|rd|th)\b", re.I)


@dataclass(frozen=True)
class Style:
    """The wording choices of one prompt."""

    nouns: dict[RoomType, tuple[str, str]]
    use_words: bool

    @classmethod
    def draw(cls, rng: random.Random) -> Style:
        nouns = {t: rng.choice(options) for t, options in NOUNS.items()}
        return cls(nouns=nouns, use_words=rng.random() < 0.5)

    def number(self, n: int) -> str:
        if n == 1 or (self.use_words and n < len(NUMBER_WORDS)):
            return NUMBER_WORDS[n]
        return str(n)

    def noun(self, room_type: RoomType, plural: bool = False) -> str:
        singular, many = self.nouns[room_type]
        return many if plural else singular

    def counted(self, room_type: RoomType, n: int) -> str:
        """ "3 bedrooms" or "one bedroom"."""
        return f"{self.number(n)} {self.noun(room_type, plural=n != 1)}"


def requirement_types(requirement: Requirement) -> set[RoomType]:
    """The room types a requirement names."""
    if isinstance(requirement, Adjacent | RelativePosition):
        return {requirement.a, requirement.b}
    if isinstance(requirement, ReachableAvoiding):
        return {requirement.type, requirement.avoid}
    if isinstance(requirement, RoomCount | EntranceInto | InRegion | HasLabel):
        return {requirement.type}
    if isinstance(requirement, Ensuites):
        return {RoomType.BATHROOM}
    return set()


def requirement_numbers(requirement: Requirement) -> set[int]:
    """The numbers of 2 or more a requirement states."""
    if isinstance(requirement, RoomCount | Ensuites | Entrances):
        return {requirement.n} if requirement.n >= 2 else set()
    return set()


def mentioned_types(text: str) -> set[RoomType]:
    scrubbed = _QUOTED.sub(" ", text)
    return {room_type for room_type, stem in TYPE_STEMS.items() if stem.search(scrubbed)}


def mentioned_numbers(text: str) -> set[int]:
    scrubbed = _ORDINAL_OR_SIZE.sub(" ", _QUOTED.sub(" ", text))
    found = set()
    for digits, word in _NUMBER_TOKEN.findall(scrubbed):
        value = int(digits) if digits else NUMBER_WORDS.index(word.lower())
        if value >= 2:
            found.add(value)
    return found


def slot_check(text: str, requirements: Iterable[Requirement]) -> list[str]:
    """Problems with a text against the slots of its requirements; empty means it passes.

    Every room type and every number of 2 or more in the requirements must appear, and no
    other room type or number may. Quoted labels are ignored.
    """
    types: set[RoomType] = set()
    numbers: set[int] = set()
    for requirement in requirements:
        types |= requirement_types(requirement)
        numbers |= requirement_numbers(requirement)
    seen_types, seen_numbers = mentioned_types(text), mentioned_numbers(text)
    problems = [f"missing room type {t.value}" for t in sorted(types - seen_types, key=str)]
    problems += [f"extra room type {t.value}" for t in sorted(seen_types - types, key=str)]
    problems += [f"missing number {n}" for n in sorted(numbers - seen_numbers)]
    problems += [f"extra number {n}" for n in sorted(seen_numbers - numbers)]
    return problems

"""What a user can ask a plan to have: the typed requirements.

Each requirement is a small frozen model with a `kind` tag, so a requirement set serializes to
JSON and back (the held-out sets store them, and the blind round-trip check reads them from a
language model). Whether a plan meets one is decided in `check.py`, from the plan alone.

| Requirement          | Example text                                          |
|----------------------|-------------------------------------------------------|
| `RoomCount`          | "3 bedrooms", "at least 2 bathrooms", "no garage"     |
| `Adjacent`           | "an open kitchen", "a balcony off the living room"    |
| `Ensuites`           | "2 ensuite bathrooms"                                 |
| `Entrances`          | "2 front doors"                                       |
| `EntranceInto`       | "the front door opens into the hall"                  |
| `InRegion`           | "bedrooms along the top", "the kitchen top right"     |
| `RelativePosition`   | "the kitchen to the left of the living room"          |
| `ReachableAvoiding`  | "a bathroom you reach without crossing a bedroom"     |
| `HasLabel`           | "a bedroom labeled 'Master bedroom'"                  |
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from vlm_floorplan.schema import RoomType

Comparison = Literal["eq", "ge", "le"]
"""How a count is compared: exactly, at least, at most."""


class Region(StrEnum):
    """A part of the plan: thirds of the bounding box of the room points."""

    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


class _Requirement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RoomCount(_Requirement):
    """The plan has this many rooms of a type (0 with `eq` means "no such room")."""

    kind: Literal["room_count"] = "room_count"
    type: RoomType
    op: Comparison
    n: int = Field(ge=0)


class Adjacent(_Requirement):
    """Some room of type `a` and some room of type `b` are linked in this way.

    `door` and `opening` need exactly that link kind, `connected` is either of the two, and
    `touching` is any link at all, a shared wall included.
    """

    kind: Literal["adjacent"] = "adjacent"
    a: RoomType
    b: RoomType
    link: Literal["door", "opening", "connected", "touching"]


class Ensuites(_Requirement):
    """This many bathrooms belong to a bedroom (reached only through bedrooms or storage)."""

    kind: Literal["ensuites"] = "ensuites"
    op: Comparison
    n: int = Field(ge=0)


class Entrances(_Requirement):
    """This many rooms have an exterior door."""

    kind: Literal["entrances"] = "entrances"
    op: Comparison
    n: int = Field(ge=0)


class EntranceInto(_Requirement):
    """Some room with an exterior door is of this type."""

    kind: Literal["entrance_into"] = "entrance_into"
    type: RoomType


class InRegion(_Requirement):
    """There is a room of this type and every room of this type lies in the region."""

    kind: Literal["in_region"] = "in_region"
    type: RoomType
    region: Region


class RelativePosition(_Requirement):
    """Every room of type `a` is left of (or above) every room of type `b`."""

    kind: Literal["relative_position"] = "relative_position"
    a: RoomType
    b: RoomType
    relation: Literal["left_of", "above"]


class ReachableAvoiding(_Requirement):
    """A room of this type is reachable from an entrance without entering a room of type `avoid`.

    Movement is over door and opening links.
    """

    kind: Literal["reachable_avoiding"] = "reachable_avoiding"
    type: RoomType
    avoid: RoomType


class HasLabel(_Requirement):
    """A room of this type carries exactly this label (the user's own name for it)."""

    kind: Literal["has_label"] = "has_label"
    type: RoomType
    text: str = Field(min_length=1, max_length=40)


Requirement = Annotated[
    RoomCount
    | Adjacent
    | Ensuites
    | Entrances
    | EntranceInto
    | InRegion
    | RelativePosition
    | ReachableAvoiding
    | HasLabel,
    Field(discriminator="kind"),
]

REQUIREMENTS_ADAPTER: TypeAdapter[list[Requirement]] = TypeAdapter(list[Requirement])


def requirements_to_json(requirements: list[Requirement]) -> list[dict]:
    """The JSON-ready form stored in the held-out sets."""
    return REQUIREMENTS_ADAPTER.dump_python(requirements, mode="json")


def requirements_from_json(data: list[dict]) -> list[Requirement]:
    return REQUIREMENTS_ADAPTER.validate_python(data)

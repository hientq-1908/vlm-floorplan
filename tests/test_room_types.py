from __future__ import annotations

import pytest

from vlm_floorplan.room_types import (
    CUBICASA_CLASS_TYPES,
    MSD_TYPES,
    RESPLAN_TYPES,
    cubicasa_room_type,
    finnish_label_type,
)
from vlm_floorplan.schema import RoomType


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("MH", RoomType.BEDROOM),
        ("MH2", RoomType.BEDROOM),  # digits split words: "MH2" is "MH"
        ("MH/TYÖH", RoomType.BEDROOM),  # first known word wins
        ("OH", RoomType.LIVING),
        ("K+RT", RoomType.KITCHEN),
        ("KHH", RoomType.UTILITY),
        ("PARVEKE", RoomType.OUTDOOR),
        ("ALATERASSI", RoomType.OUTDOOR),  # compound word: matched by stem
        ("KOMERO", RoomType.STORAGE),  # "K" must not match inside a longer word
        ("", RoomType.UNKNOWN),
        (None, RoomType.UNKNOWN),
        ("???", RoomType.UNKNOWN),
    ],
)
def test_finnish_label_mapping(label: str | None, expected: RoomType) -> None:
    assert finnish_label_type(label) == expected


def test_cubicasa_class_decides_when_it_is_specific() -> None:
    assert cubicasa_room_type("Space Bath Shower", "whatever") == RoomType.BATHROOM
    assert cubicasa_room_type("Space Kitchen", None) == RoomType.KITCHEN


def test_cubicasa_undefined_classes_fall_back_to_the_printed_name() -> None:
    assert cubicasa_room_type("Space Undefined", "MH") == RoomType.BEDROOM
    assert cubicasa_room_type("Space UserDefined", "APUT") == RoomType.UNKNOWN
    assert cubicasa_room_type("Space Undefined", None) == RoomType.UNKNOWN


def test_unseen_cubicasa_class_is_other_not_unknown() -> None:
    assert cubicasa_room_type("Space Observatory", None) == RoomType.OTHER


def test_non_space_class_is_an_error() -> None:
    with pytest.raises(ValueError):
        cubicasa_room_type("Wall External", None)


def test_every_dataset_maps_into_the_coarse_set_without_unknown() -> None:
    for table in (CUBICASA_CLASS_TYPES, RESPLAN_TYPES, MSD_TYPES):
        assert RoomType.UNKNOWN not in table.values()
        assert all(isinstance(value, RoomType) for value in table.values())

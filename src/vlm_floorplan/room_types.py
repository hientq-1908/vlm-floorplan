"""Map each dataset's room labels onto the coarse `RoomType` set.

Every dataset names rooms differently:

- CubiCasa5K uses SVG classes such as "Space Bath Shower" (60+ values), plus
  free-text Finnish names for rooms classed "Undefined" or "UserDefined".
- ResPlan uses 7 room categories plus a front door.
- MSD uses 9 room types.

The tables below are deliberately plain dictionaries so they are easy to read,
audit and extend. The dataset's original label is always kept next to the
coarse type, so nothing is lost by mapping.
"""

from __future__ import annotations

import re

from vlm_floorplan.schema import RoomType

T = RoomType

# CubiCasa5K: the second token of the "Space ..." class, e.g. "Space Bath Shower" -> "Bath".
# Based on the class list in CubiCasa5K's floortrans/loaders/house.py, but coarser
# (house.py folds many of these into a generic "Room").
CUBICASA_CLASS_TYPES: dict[str, RoomType] = {
    "LivingRoom": T.LIVING,
    "Lounge": T.LIVING,
    "Den": T.LIVING,
    "RecreationRoom": T.LIVING,
    "Kitchen": T.KITCHEN,
    "Dining": T.DINING,
    "EatingArea": T.DINING,
    "Bedroom": T.BEDROOM,
    "Bath": T.BATHROOM,
    "Sauna": T.BATHROOM,
    "HotTub": T.BATHROOM,
    "Entry": T.ENTRY,
    "DraughtLobby": T.ENTRY,
    "Hall": T.ENTRY,
    "HallWay": T.ENTRY,
    "Storage": T.STORAGE,
    "Closet": T.STORAGE,
    "DressingRoom": T.STORAGE,
    "Pantry": T.STORAGE,
    "Garbage": T.STORAGE,
    "Utility": T.UTILITY,
    "TechnicalRoom": T.UTILITY,
    "ServiceRoom": T.UTILITY,
    "Outdoor": T.OUTDOOR,
    "SunRoom": T.OUTDOOR,
    "Garage": T.GARAGE,
    "CarPort": T.GARAGE,
    "StairWell": T.STAIRS,
    "Room": T.OTHER,
    "Office": T.OTHER,
    "Library": T.OTHER,
    "Alcove": T.OTHER,
    "Attic": T.OTHER,
    "Basement": T.OTHER,
    "Elevated": T.OTHER,
    "Elevator": T.OTHER,
    "OpenToBelow": T.OTHER,
}

# CubiCasa5K classes that carry no type: the room's printed name decides instead.
CUBICASA_UNLABELED_CLASSES = frozenset({"Undefined", "UserDefined"})

# Finnish room names and abbreviations as printed on CubiCasa5K plans.
# A label is split into words ("MH/TYÖH" -> "MH", "TYÖH"); the first word found here wins.
FINNISH_WORD_TYPES: dict[str, RoomType] = {
    # living
    "OH": T.LIVING, "OLOHUONE": T.LIVING, "TUPA": T.LIVING, "ARK": T.LIVING,
    "OLESKELU": T.LIVING, "OLESKELUTILA": T.LIVING,
    # kitchen
    "K": T.KITCHEN, "KK": T.KITCHEN, "KT": T.KITCHEN, "AVOK": T.KITCHEN,
    "KEITTIÖ": T.KITCHEN, "KEITTOKOMERO": T.KITCHEN, "KEITTIOKOMERO": T.KITCHEN,
    # dining
    "RT": T.DINING, "RUOK": T.DINING, "RUOKAILU": T.DINING, "RUOKAILUTILA": T.DINING,
    "RUOKALA": T.DINING,
    # bedroom
    "MH": T.BEDROOM, "MAKUUHUONE": T.BEDROOM, "MAKUUH": T.BEDROOM,
    # bathroom (all wet rooms)
    "KH": T.BATHROOM, "PH": T.BATHROOM, "WC": T.BATHROOM, "KPH": T.BATHROOM,
    "PSH": T.BATHROOM, "SH": T.BATHROOM, "PESUH": T.BATHROOM, "PESU": T.BATHROOM,
    "PESUHUONE": T.BATHROOM, "KYLPYHUONE": T.BATHROOM, "S": T.BATHROOM, "SAUNA": T.BATHROOM,
    # entry and circulation
    "ET": T.ENTRY, "TK": T.ENTRY, "ETEINEN": T.ENTRY, "TUULIKAAPPI": T.ENTRY,
    "KURAET": T.ENTRY, "KURA": T.ENTRY, "KÄYTÄVÄ": T.ENTRY, "KÄYT": T.ENTRY, "HALL": T.ENTRY,
    "AULA": T.ENTRY,
    # storage
    "VAR": T.STORAGE, "VARASTO": T.STORAGE, "KOMERO": T.STORAGE, "KOM": T.STORAGE,
    "VH": T.STORAGE, "VAATEHUONE": T.STORAGE, "PUKUH": T.STORAGE, "PKH": T.STORAGE,
    "ULKOVAR": T.STORAGE, "TEKNVAR": T.STORAGE,
    # utility
    "KHH": T.UTILITY, "KODINHOITOHUONE": T.UTILITY, "TEKN": T.UTILITY, "LÄMMÖNJAKOH": T.UTILITY,
    "LJK": T.UTILITY, "SIIVOUS": T.UTILITY,
    # outdoor
    "PARVEKE": T.OUTDOOR, "PARV": T.OUTDOOR, "TERASSI": T.OUTDOOR, "KUISTI": T.OUTDOOR,
    "AVOKUISTI": T.OUTDOOR, "ULKOTILA": T.OUTDOOR, "PATIO": T.OUTDOOR, "PIHA": T.OUTDOOR,
    "KATOS": T.OUTDOOR, "LASITETTU": T.OUTDOOR,
    # garage
    "AT": T.GARAGE, "AUTOTALLI": T.GARAGE, "AUTOKATOS": T.GARAGE,
    # stairs
    "PORRAS": T.STAIRS, "PORRASHUONE": T.STAIRS, "PRH": T.STAIRS,
    # other labeled rooms
    "TH": T.OTHER, "TYÖH": T.OTHER, "TYÖHUONE": T.OTHER, "KIRJASTO": T.OTHER,
    "ULLAKKO": T.OTHER, "KELLARI": T.OTHER, "TOIMISTO": T.OTHER,
}  # fmt: skip

# Compound words ("ALATERASSI" = lower terrace) contain a known word. Only long,
# unambiguous stems are matched inside words, so "K" never matches inside "KOMERO".
FINNISH_STEM_TYPES: dict[str, RoomType] = {
    "TERASSI": T.OUTDOOR,
    "PARVEKE": T.OUTDOOR,
    "KUISTI": T.OUTDOOR,
    "KEITTIÖ": T.KITCHEN,
    "SAUNA": T.BATHROOM,
    "VARASTO": T.STORAGE,
    "ULLAKKO": T.OTHER,
}

_WORD_SPLIT = re.compile(r"[^A-ZÅÄÖ]+")


def finnish_label_type(label: str | None) -> RoomType:
    """Guess a room type from a printed Finnish room name, or `UNKNOWN`."""
    if not label:
        return T.UNKNOWN
    words = [word for word in _WORD_SPLIT.split(label.upper()) if word]
    for word in words:
        if word in FINNISH_WORD_TYPES:
            return FINNISH_WORD_TYPES[word]
    for word in words:
        for stem, room_type in FINNISH_STEM_TYPES.items():
            if stem in word:
                return room_type
    return T.UNKNOWN


def cubicasa_room_type(space_class: str, label: str | None) -> RoomType:
    """Map a CubiCasa5K "Space ..." class (and printed name) to a room type.

    `space_class` is the full class attribute, e.g. "Space Bath Shower".
    """
    tokens = space_class.split()
    if len(tokens) < 2 or tokens[0] != "Space":
        raise ValueError(f"not a CubiCasa5K space class: {space_class!r}")
    main = tokens[1]
    if main in CUBICASA_UNLABELED_CLASSES:
        return finnish_label_type(label)
    # A class we have never seen is still a labeled room, so it is "other", not "unknown".
    return CUBICASA_CLASS_TYPES.get(main, T.OTHER)


# ResPlan geometry keys that are rooms. "front_door" is not a room: it becomes
# the `entrance` flag on the room it opens into.
RESPLAN_TYPES: dict[str, RoomType] = {
    "living": T.LIVING,  # ResPlan merges corridors into living
    "kitchen": T.KITCHEN,
    "bedroom": T.BEDROOM,
    "bathroom": T.BATHROOM,
    "balcony": T.OUTDOOR,
    "storage": T.STORAGE,
    "stair": T.STAIRS,
}

# MSD (Modified Swiss Dwellings) room types. The ResPlan and MSD converters are
# being built in parallel tasks on this schema and consume these tables.
MSD_TYPES: dict[str, RoomType] = {
    "Bedroom": T.BEDROOM,
    "Livingroom": T.LIVING,
    "Kitchen": T.KITCHEN,
    "Dining": T.DINING,
    "Corridor": T.ENTRY,
    "Stairs": T.STAIRS,
    "Storeroom": T.STORAGE,
    "Bathroom": T.BATHROOM,
    "Balcony": T.OUTDOOR,
}

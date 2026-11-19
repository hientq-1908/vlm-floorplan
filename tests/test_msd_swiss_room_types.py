"""The MSD and Swiss Dwellings label tables."""

from __future__ import annotations

from vlm_floorplan.room_types import (
    MSD_ROOM_NAMES,
    MSD_TYPES,
    SWISS_NON_ROOM_AREAS,
    SWISS_TYPES,
)
from vlm_floorplan.schema import RoomType

# Every `area` subtype in Swiss Dwellings' full geometries.csv (v3.0.0): 77 values.
SWISS_AREA_SUBTYPES_SEEN = {
    "BALCONY", "BASEMENT", "BASEMENT_COMPARTMENT", "BATHROOM", "BEDROOM", "BIKE_STORAGE",
    "CORRIDOR", "CORRIDORS_AND_HALLS", "COUNTER_ROOM", "DINING", "ELEVATOR", "HEATING",
    "KITCHEN", "KITCHEN_DINING", "LIGHTWELL", "LIVING_DINING", "LIVING_ROOM", "LOBBY",
    "LOGGIA", "NOT_DEFINED", "OFFICE", "OUTDOOR_VOID", "PRAM", "PRAM_AND_BIKE_STORAGE_ROOM",
    "ROOM", "SHAFT", "STAIRCASE", "STOREROOM", "TECHNICAL_AREA", "TERRACE", "VOID",
    "WASH_AND_DRY_ROOM", "WINTERGARTEN",
    "GARDEN", "PATIO", "GARAGE", "CARPARK", "SANITARY_ROOMS", "COMMON_KITCHEN", "FOYER",
    "CLOAKROOM", "WAREHOUSE", "ARCHIVE", "COLD_STORAGE", "LOGISTICS", "OIL_TANK",
    "HOUSE_TECHNICS_FACILITIES", "ELEVATOR_FACILITIES", "ELECTRICAL_SUPPLY", "WATER_SUPPLY",
    "GAS", "AIR", "OFFICE_TECH_ROOM", "OPERATIONS_FACILITIES", "OFFICE_SPACE",
    "OPEN_PLAN_OFFICE", "MEETING_ROOM", "COMMUNITY_ROOM", "SALESROOM", "SHOWROOM",
    "SPORTS_ROOMS", "WAITING_ROOM", "RECEPTION_ROOM", "CANTEEN", "BREAK_ROOM", "STUDIO",
    "WORKSHOP", "FACTORY_ROOM", "TEACHING_ROOM", "MEDICAL_ROOM", "DEDICATED_MEDICAL_ROOM",
    "RADATION_THERAPY", "PHYSIO_AND_REHABILITATION", "ARCADE", "SHELTER",
    "VEHICLE_TRAFFIC_AREA", "TRANSPORT_SHAFT",
}  # fmt: skip


def test_msd_indices_follow_msds_room_names_order() -> None:
    # Index = position in ROOM_NAMES of cvaad-workshop/iccv23-challenge/constants.py.
    assert MSD_ROOM_NAMES[:4] == ("Bedroom", "Livingroom", "Kitchen", "Dining")
    assert MSD_ROOM_NAMES[7] == "Bathroom" and len(MSD_ROOM_NAMES) == 9
    assert set(MSD_ROOM_NAMES) == set(MSD_TYPES)


def test_every_swiss_area_subtype_is_either_a_room_or_deliberately_not() -> None:
    assert set(SWISS_TYPES) | SWISS_NON_ROOM_AREAS == SWISS_AREA_SUBTYPES_SEEN
    assert not set(SWISS_TYPES) & SWISS_NON_ROOM_AREAS


def test_only_the_explicitly_undefined_swiss_area_is_unknown() -> None:
    unknown = {name for name, room_type in SWISS_TYPES.items() if room_type == RoomType.UNKNOWN}

    assert unknown == {"NOT_DEFINED"}


def test_swiss_and_msd_agree_on_the_rooms_msd_names_directly() -> None:
    pairs = {
        "BEDROOM": "Bedroom",
        "LIVING_ROOM": "Livingroom",
        "KITCHEN": "Kitchen",
        "DINING": "Dining",
        "CORRIDOR": "Corridor",
        "STAIRCASE": "Stairs",
        "STOREROOM": "Storeroom",
        "BATHROOM": "Bathroom",
        "BALCONY": "Balcony",
        # MSD's own folding of finer Swiss labels:
        "ROOM": "Bedroom",
        "KITCHEN_DINING": "Kitchen",
        "LIVING_DINING": "Livingroom",
        "ELEVATOR": "Stairs",
        "CORRIDORS_AND_HALLS": "Corridor",
    }
    for swiss_name, msd_name in pairs.items():
        assert SWISS_TYPES[swiss_name] == MSD_TYPES[msd_name], swiss_name

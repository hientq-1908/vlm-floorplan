"""Requirements: what a user asks a plan to have, and the one function that checks it."""

from vlm_floorplan.requirements.check import (
    RequirementResult,
    all_met,
    check_requirements,
    holds,
)
from vlm_floorplan.requirements.derive import apply_labels, derive_requirements
from vlm_floorplan.requirements.facts import Facts, compute_facts
from vlm_floorplan.requirements.types import (
    Adjacent,
    Ensuites,
    EntranceInto,
    Entrances,
    HasLabel,
    InRegion,
    ReachableAvoiding,
    Region,
    RelativePosition,
    Requirement,
    RoomCount,
    requirements_from_json,
    requirements_to_json,
)

__all__ = [
    "Adjacent",
    "EntranceInto",
    "Entrances",
    "Ensuites",
    "Facts",
    "HasLabel",
    "InRegion",
    "ReachableAvoiding",
    "Region",
    "RelativePosition",
    "Requirement",
    "RequirementResult",
    "RoomCount",
    "all_met",
    "apply_labels",
    "check_requirements",
    "compute_facts",
    "derive_requirements",
    "holds",
    "requirements_from_json",
    "requirements_to_json",
]
